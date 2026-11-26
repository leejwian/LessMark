"""Persistent interruption outbox. Experiment retries and mail retries are separate."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import smtplib
import ssl
import stat
import sys
import time
from email.message import EmailMessage

from .storage import Blocked, identity, read_json, write_json

DELIVERY_ATTEMPTS = 3
LOCK_WAIT_SECONDS = 5
# If SMTP accepted but the receipt could not be committed, remember that fact
# for this process. Cross-process recovery still requires a writable outbox.
_accepted_without_receipt = {}
_sensitive_literals = set()


def resolve_mode(*, enabled=False, smoke=False, mode=None):
    """Only servers with explicit SMTP configuration attempt network delivery."""
    policy = os.environ.get("LERRA_NOTIFICATION_MODE", "local")
    if policy not in ("local", "smtp"):
        raise ValueError("LERRA_NOTIFICATION_MODE must be local or smtp")
    if policy == "local":
        return "local"
    if mode not in (None, "smtp", "mock", "local"):
        raise ValueError("invalid notification mode")
    if mode == "local":
        return "local"
    if smoke:
        return "mock"
    return mode or ("smtp" if enabled else "mock")


@contextlib.contextmanager
def execution_policy(mode):
    """Propagate a local restriction to workers without leaking caller state."""
    keys = ("LERRA_NOTIFICATION_MODE", "LERRA_GENERATOR_NOTIFICATIONS")
    previous = {key: os.environ.get(key) for key in keys}
    if mode == "local":
        for key in keys:
            os.environ[key] = "local"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def mail_account():
    return os.environ.get("LERRA_MAIL_ACCOUNT", "").strip()


def credential_path():
    return Path(os.environ.get("LERRA_MAIL_PASSWORD_FILE", str(Path.home()/".config/lerra_ft_comparison/gmail_app_password")))


def configuration_status():
    """Inspect credential-file metadata only; no secret read or SMTP connection."""
    path = credential_path()
    try:
        info = path.stat()
        regular = stat.S_ISREG(info.st_mode)
        private = not bool(info.st_mode & 0o077)
        owned = info.st_uid == os.getuid()
    except OSError:
        regular = private = owned = False
    return {"recipient": mail_account(), "credential_path": str(path),
            "credential_file_exists": regular, "private_permissions": private,
            "owned_by_current_user": owned, "metadata_ready": regular and private and owned,
            "credential_contents_read": False, "smtp_contacted": False}


def _password():
    path = credential_path()
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise Blocked("notification_configuration", "Gmail app-password file must exist with private permissions; configure LERRA_MAIL_PASSWORD_FILE")
    password = path.read_text().strip().replace(" ", "")
    if not password:
        raise Blocked("notification_configuration", "Gmail app-password file is empty")
    _sensitive_literals.add(password)
    return password


def preflight(*, mode=None):
    """Authenticate before costly work; this sends no email."""
    if resolve_mode(enabled=True, mode=mode) == "local":
        return {"mode": "local", "smtp_contacted": False,
                "smtp_authenticated": False, "message_sent": False}
    try:
        if not mail_account():
            raise Blocked("notification_configuration", "Set LERRA_MAIL_ACCOUNT to the Gmail notification account")
        password = _password()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as smtp:
            smtp.login(mail_account(), password)
            code, _ = smtp.noop()
            if code != 250:
                raise RuntimeError("SMTP NOOP failed")
    except Blocked:
        raise
    except Exception as exc:
        raise Blocked("notification_configuration", "Gmail SMTP preflight failed ("+type(exc).__name__+"); no experiments started") from None
    return {"recipient": mail_account(), "smtp_authenticated": True, "message_sent": False}


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if re.search(r"(?i)(password|token|secret|authorization|api_key|fingerprint_key)", str(k)) else sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    for secret in tuple(_sensitive_literals):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    for key, secret in os.environ.items():
        if len(secret) > 4 and re.search(r"(?i)(password|token|secret|credential|api_key)", key):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"KGAT_[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", value)
    value = re.sub(r"(?i)(Bearer|Basic)\s+\S+", r"\1 [REDACTED]", value)
    value = re.sub(r"(?i)(password|token|secret|api_key|fingerprint|key)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return value


def log_diagnostic(value):
    # stderr is normally a log on the same /data filesystem as the outbox.
    # Disk-full or a closed log must not disable the network fallback as well.
    try:
        print(json.dumps(sanitize(value), ensure_ascii=False), file=sys.stderr)
    except (OSError, ValueError):
        pass


def _storage_error(record, error):
    record["notification_storage_error_type"] = type(error).__name__
    log_diagnostic({"notification_storage_error_type": type(error).__name__,
                    "job_id": record["event"].get("job_id")})


def _persist(path, record):
    """Outbox failure must never prevent attempting the actual error email."""
    record["receipt_persisted"] = False
    if path is None:
        return
    try:
        record["receipt_persisted"] = True
        write_json(path, record)
    except (OSError, ValueError) as exc:
        record["receipt_persisted"] = False
        _storage_error(record, exc)


def _deliver(path, record, *, message_id=None, deadline=None):
    # Also protect explicit retries of historic SMTP receipts under a current
    # local-only invocation; leave those receipts unchanged for provenance.
    if resolve_mode(enabled=True) == "local":
        return record
    # A sent receipt is never sent again. SMTP acceptance is not inbox delivery.
    remembered = _accepted_without_receipt.get(str(path.resolve())) if path is not None else None
    if remembered is not None:
        _persist(path, remembered)
        return remembered
    if record["sent"] or record["mode"] != "smtp":
        return record
    for attempt in range(DELIVERY_ATTEMPTS):
        if deadline is not None and time.monotonic() >= deadline:
            record["pending"] = True
            _persist(path, record)
            break
        record["delivery_attempts"] = record.get("delivery_attempts", 0)+1
        record["last_attempt_unix"] = time.time()
        _persist(path, record)
        accepted = False
        try:
            if not mail_account() or record["recipient"] != mail_account():
                raise ValueError("Pending receipt requires the original LERRA_MAIL_ACCOUNT")
            msg = EmailMessage()
            msg["From"] = msg["To"] = mail_account()
            msg["Message-ID"] = "<lerra-"+(message_id or path.stem)+"@gmail.com>"
            subject = f"[LERRA interrupted] {record['event'].get('stage', 'startup')} / {record['event'].get('job_id', 'pipeline')} ({record['event'].get('exit_code')})"
            msg["Subject"] = " ".join(subject.splitlines())[:240]
            msg.set_content(json.dumps(record["event"], ensure_ascii=False, indent=2))
            timeout = 30 if deadline is None else max(.1, min(30, deadline-time.monotonic()))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=timeout) as smtp:
                smtp.login(mail_account(), _password())
                refused = smtp.send_message(msg)
                if refused:
                    raise RuntimeError("SMTP recipient refused")
                accepted = True
        except Exception as exc:
            if accepted:
                # DATA was acknowledged. A later QUIT/disconnect error must
                # not cause a known accepted interruption message to be sent again.
                record["smtp_close_error_type"] = type(exc).__name__
            else:
                # Never serialize SMTP responses, password contents, or exception repr.
                record["delivery_error_type"] = type(exc).__name__
                record["pending"] = True
                if path is not None:
                    try:
                        write_json(path.with_suffix(".delivery_error.json"), {"error_type": type(exc).__name__, "original_event": path.stem, "delivery_attempts": record["delivery_attempts"]})
                    except (OSError, ValueError) as storage_exc:
                        _storage_error(record, storage_exc)
                _persist(path, record)
                if attempt+1 < DELIVERY_ATTEMPTS:
                    delay = 2 if deadline is None else min(2, max(0, deadline-time.monotonic()))
                    time.sleep(delay)
        if accepted:
            record["sent"], record["pending"] = True, False
            record["smtp_accepted_unix"] = time.time()
            _persist(path, record)
            if path is not None and not record["receipt_persisted"]:
                _accepted_without_receipt[str(path.resolve())] = record
            return record
    return record


def _envelope(event, enabled, smoke, mode=None):
    context = os.environ.get("LERRA_PIPELINE_ALERT_CONTEXT")
    event = sanitize(dict(event))
    if context and not smoke:
        event.setdefault("flow_context", context)
    mode = resolve_mode(enabled=enabled, smoke=smoke, mode=mode)
    key = identity({"mode": mode, **{k: event.get(k) for k in ("job_id", "attempt", "stage", "exit_code", "flow_context", "event_id")}})
    return key, {"event": event, "recipient": mail_account() if mode == "smtp" else None,
                 "mode": mode, "sent": False, "pending": mode == "smtp", "delivery_attempts": 0,
                 "receipt_persisted": False}


def emergency_notify(event, *, enabled=False, smoke=False, mode=None, error=None):
    """Last resort with no filesystem dependency; original failure stays intact."""
    key, record = _envelope(event, enabled, smoke, mode)
    if error is not None:
        _storage_error(record, error)
    record["receipt_persisted"] = False
    return _deliver(None, record, message_id=key)


class OutboxBusy(RuntimeError):
    """Another process is delivering this exact event; do not send it twice."""


@contextlib.contextmanager
def _event_lock(path, *, deadline=None):
    # Independent events do not share a network-duration global lock. A stuck
    # sender can delay only its own event, and even that wait is bounded.
    limit = time.monotonic()+LOCK_WAIT_SECONDS
    if deadline is not None:
        limit = min(limit, deadline)
    with open(path.with_suffix(".lock"), "a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = limit-time.monotonic()
                if remaining <= 0:
                    raise OutboxBusy("notification event is already being delivered") from None
                time.sleep(min(.05, remaining))
        yield


def _receipt(path):
    previous = read_json(path)
    if not (isinstance(previous, dict) and previous.get("mode") in ("smtp", "mock", "local")
            and isinstance(previous.get("sent"), bool) and isinstance(previous.get("pending"), bool)
            and isinstance(previous.get("event"), dict)):
        raise ValueError("invalid notification receipt")
    if previous["sent"] and previous["pending"]:
        raise ValueError("contradictory notification receipt")
    if previous["mode"] == "local" and (previous["sent"] or previous["pending"]):
        raise ValueError("local notification cannot be sent or pending SMTP")
    return previous


def notify(directory, event, *, enabled=False, smoke=False, mode=None):
    context = os.environ.get("LERRA_PIPELINE_ALERT_CONTEXT")
    destination = os.environ.get("LERRA_PIPELINE_ALERT_DIRECTORY") if context and not smoke else None
    directory = Path(destination or directory)/"notifications"
    key, record = _envelope(event, enabled, smoke, mode)
    path = directory/(key+".json")
    remembered = _accepted_without_receipt.get(str(path.resolve()))
    if remembered is not None:
        _persist(path, remembered)
        return remembered
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _event_lock(path):
            if path.exists():
                previous = _receipt(path)
                if previous["mode"] != record["mode"]:
                    raise ValueError("notification mode mismatch")
                return previous
            _persist(path, record)
            return _deliver(path, record)
    except OutboxBusy:
        # The owner may be in SMTP DATA. Preserve its receipt without racing a
        # second send or overwriting known SMTP acceptance.
        try:
            if path.exists():
                return {**_receipt(path), "delivery_deferred": "event_lock_busy"}
        except (OSError, ValueError):
            pass
        record["delivery_deferred"] = "event_lock_busy"
        _persist(path, record)
        return record
    except (OSError, ValueError) as exc:
        _storage_error(record, exc)
        # This also handles mkdir, lock and malformed-receipt failures. A
        # known accepted record is never resent, even when closing a lock fails.
        return _deliver(path, record)


def flush_pending(directory, *, max_records=None, deadline_seconds=None, mode=None):
    """Bounded retry of durable unsent messages, with stable Message-ID."""
    if resolve_mode(enabled=True, mode=mode) == "local":
        return []
    directory = Path(directory)/"notifications"
    if max_records is not None and (type(max_records) is not int or max_records < 1):
        raise ValueError("max_records must be a positive integer")
    if deadline_seconds is not None and deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if not directory.exists():
        return []
    deadline = None if deadline_seconds is None else time.monotonic()+deadline_seconds
    results = []
    for path in sorted(directory.glob("*.json")):
        if ((max_records is not None and len(results) >= max_records)
                or (deadline is not None and time.monotonic() >= deadline)):
            break
        if path.name.endswith(".delivery_error.json"):
            continue
        try:
            with _event_lock(path, deadline=deadline):
                record = _receipt(path)
                if record.get("pending"):
                    results.append(_deliver(path, record, deadline=deadline))
        except OutboxBusy:
            log_diagnostic({"notification_retry_deferred": "event_lock_busy", "notification_event": path.stem})
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            # One damaged receipt must not starve unrelated pending mail.
            log_diagnostic({"notification_retry_error_type": type(exc).__name__,
                            "notification_event": path.stem})
    return results


def outbox_status(directory):
    """No SMTP: include unvisited pending and invalid receipts in retry status."""
    directory = Path(directory)/"notifications"
    result = {"sent": 0, "pending": 0, "mock": 0, "local": 0, "invalid_receipts": []}
    if not directory.exists():
        return result
    if not directory.is_dir():
        result["invalid_receipts"].append({"path": str(directory), "error_type": "NotADirectoryError"})
        return result
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".delivery_error.json"):
            continue
        try:
            record = _receipt(path)
            result["sent"] += int(record["sent"])
            result["pending"] += int(record["pending"])
            result["mock"] += int(record["mode"] == "mock")
            result["local"] += int(record["mode"] == "local")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            result["invalid_receipts"].append({"path": str(path), "error_type": type(exc).__name__})
    return result


def context_alerts(directory, context, *, event_id=None, smtp_only=False):
    records = []
    for path in (Path(directory)/"notifications").glob("*.json"):
        if path.name.endswith(".delivery_error.json"):
            continue
        try:
            record = read_json(path)
            if smtp_only and record.get("mode") != "smtp":
                continue
            event = record.get("event", {})
            if event.get("flow_context") == context and (event_id is None or event.get("event_id") == event_id):
                records.append({"path": str(path), "sent": record["sent"], "pending": record.get("pending", False)})
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            # Unreadable receipts cannot establish that an error was reported.
            continue
    return records


def main(argv=None):
    """Explicit operator actions; metadata is the default and never reads secrets."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--mail-check", action="store_true", help="authenticate SMTP without sending a message")
    action.add_argument("--retry-pending", type=Path, metavar="RUN_DIRECTORY",
                        help="retry existing pending failure mail without running experiments")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--deadline-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if args.mail_check:
        result = preflight()
    elif args.retry_pending is not None:
        attempts = flush_pending(args.retry_pending, max_records=args.max_records,
                                 deadline_seconds=args.deadline_seconds)
        result = {"attempted": attempts, "outbox": outbox_status(args.retry_pending)}
    else:
        result = configuration_status()
    print(json.dumps(sanitize(result), ensure_ascii=False, indent=2))
    status = result.get("outbox", {})
    return 20 if status.get("pending") or status.get("invalid_receipts") else 0


if __name__ == "__main__":
    raise SystemExit(main())

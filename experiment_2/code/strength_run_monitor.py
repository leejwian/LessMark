"""Supervise a Linux training worker and report failures through the LERRA error outbox."""
from contextlib import contextmanager
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from runtime.comparison.protocol_v2 import notifications, process_monitor


_phase_sender = None
RETRY_MAX_RECORDS = 8
RETRY_DEADLINE_SECONDS = 30


def set_phase(stage, *, condition=None, method=None, candidate=None):
    """Publish a coarse checkpoint from the worker, including before long native operations."""
    if _phase_sender is not None:
        _phase_sender.send({'stage': stage, 'condition': condition, 'method': method, 'candidate': candidate})


def _worker(args, target, sender, parent_pid):
    global _phase_sender
    try:
        # A vanished supervisor must not leave an expensive, unmonitored training run.
        process_monitor.bind_parent(parent_pid)
        os.setsid()
        _phase_sender = sender
        target(args)
    except KeyboardInterrupt:
        sender.send({'cancelled': True})
        raise SystemExit(130) from None
    except Exception as exc:
        sender.send({'error_type': type(exc).__name__})
        raise
    finally:
        _phase_sender = None
        sender.close()


@contextmanager
def _stop_signals():
    requested = []
    def remember(signum, frame):
        if not requested:
            requested.append(signum)
    old = {sig: signal.signal(sig, remember) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield requested
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _receive(worker, reader, requested, event):
    # Wait for progress, process death, or a signal sent to the supervisor alone.
    watched = [reader, worker.sentinel]
    while not requested:
        available = wait(watched, timeout=0.5)
        if reader in available:
            try:
                event.update(reader.recv())
                continue
            except EOFError:
                watched.remove(reader)
        if worker.sentinel in available:
            worker.join()
            return worker.exitcode
    return -requested[0]


def _stop_worker(worker, leader):
    if leader:
        # The child creates its own session, possibly after the initial /proc snapshot.
        # The original start ticks also protect against PID reuse after join().
        owned = process_monitor.session_members({**leader, 'session': leader['pid']})
        if owned:
            process_monitor.stop_owned(owned)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
    worker.join()


def _schedule_delivery(outbox):
    """Schedule only the mail outbox. No experiment arguments or credential contents travel."""
    project = Path(__file__).resolve().parents[2]
    unit = 'lerra-strength-mail-' + uuid.uuid4().hex
    command = ['systemd-run', '--user', '--quiet', '--collect', '--unit=' + unit,
               '--on-active=5min', '--timer-property=AccuracySec=1s', '--timer-property=RemainAfterElapse=no',
               '--service-type=exec', '--property=RestartForceExitStatus=20', '--property=RestartSec=5min',
               '--property=StartLimitIntervalSec=0', '--working-directory=' + str(project),
               '--setenv=PYTHONPATH=' + str(project), '--setenv=LERRA_NOTIFICATION_MODE=smtp']
    # The secret remains in its existing owner-only file. Pin its path for the independent service.
    password_file = notifications.credential_path().resolve()
    command += ['--setenv=LERRA_MAIL_ACCOUNT=' + notifications.mail_account(),
                '--setenv=LERRA_MAIL_PASSWORD_FILE=' + str(password_file),
                sys.executable, '-B', '-m', 'experiment_2.code.strength_run_monitor',
                '--retry-pending', str(outbox)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=15, check=False)
        if result.returncode:
            notifications.log_diagnostic({'mail_retry_setup_failed': result.returncode, 'outbox': str(outbox)})
        else:
            notifications.log_diagnostic({'mail_retry_timer': unit + '.timer', 'outbox': str(outbox)})
    except (OSError, subprocess.TimeoutExpired) as exc:
        notifications.log_diagnostic({'mail_retry_setup_failed': type(exc).__name__, 'outbox': str(outbox)})


def _retry_batch(outbox):
    """20 triggers a systemd restart. Stop on a drained outbox or permanent invalid receipts."""
    status = notifications.main(['--retry-pending', str(outbox), '--max-records', str(RETRY_MAX_RECORDS),
                                 '--deadline-seconds', str(RETRY_DEADLINE_SECONDS)])
    if status == 20:
        remaining = notifications.outbox_status(outbox)
        if not remaining['pending']:
            return os.EX_DATAERR if remaining['invalid_receipts'] else 0
    return status


def _report(outbox, event):
    # An upstream experiment-1 pipeline may have exported a different outbox.
    previous = os.environ.pop('LERRA_PIPELINE_ALERT_CONTEXT', None)
    try:
        receipt = notifications.notify(outbox, event, enabled=True, mode='smtp')
        notifications.log_diagnostic({'failure_mail_sent': receipt['sent'], 'failure_mail_pending': receipt['pending'],
                                      'outbox': str(outbox / 'notifications'), 'exit_code': event['exit_code']})
        if receipt['mode'] == 'smtp' and receipt['pending'] and receipt.get('receipt_persisted'):
            _schedule_delivery(outbox)
    except Exception as exc:
        notifications.log_diagnostic({'failure_mail_error': type(exc).__name__, 'exit_code': event['exit_code']})
    finally:
        if previous is not None:
            os.environ['LERRA_PIPELINE_ALERT_CONTEXT'] = previous


def supervise(args, target):
    """Run target(args) in a fresh process. Return a shell exit code; no success or Ctrl-C mail."""
    output = Path(args.output).resolve()
    outbox = output.with_name(output.name + '.alerts') if output.name else output / '.alerts'
    attempt = uuid.uuid4().hex
    event = {'job_id': 'experiment_2_strength_selection', 'attempt': attempt, 'stage': 'smtp_preflight',
             'output_directory': str(output)}
    exitcode = 1
    with _stop_signals() as requested:
        try:
            if notifications.resolve_mode(enabled=True) == 'smtp':
                if not notifications.preflight(mode='smtp').get('smtp_authenticated'):
                    raise RuntimeError('The configured notification mode does not authenticate SMTP')
                notifications.flush_pending(outbox, mode='smtp', max_records=RETRY_MAX_RECORDS,
                                            deadline_seconds=RETRY_DEADLINE_SECONDS)
                if any(r['pending'] is True and r['sent'] is False
                       for r in notifications.context_alerts(outbox, None, smtp_only=True)):
                    _schedule_delivery(outbox)
            if not requested:
                event['stage'] = 'worker_start'
                ctx = multiprocessing.get_context('spawn')
                with _training_process(ctx, args, target) as pair:
                    worker, reader = pair
                    exitcode = _receive(worker, reader, requested, event)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            event['error_type'] = type(exc).__name__
            notifications.log_diagnostic({'stage': event['stage'], 'error_type': type(exc).__name__, 'error': str(exc)})
            exitcode = 1
        if requested:
            exitcode = -requested[0]
        if exitcode < 0:
            event['signal'] = signal.Signals(-exitcode).name
            exitcode = 128 - exitcode
        if exitcode and not (exitcode == 130 and (requested or event.get('cancelled') or event.get('signal') == 'SIGINT')):
            _report(outbox, {**event, 'exit_code': exitcode})
    return exitcode


@contextmanager
def _training_process(ctx, args, target):
    reader, sender = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_worker, args=(args, target, sender, os.getpid()), name='strength-selection')
    leader = None
    try:
        worker.start()
        leader = process_monitor.process(worker.pid)
        sender.close()
        yield worker, reader
    finally:
        sender.close()
        reader.close()
        if worker.pid is not None:
            _stop_worker(worker, leader)
        worker.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retry one bounded batch of durable experiment 2 error email.')
    parser.add_argument('--retry-pending', type=Path, required=True)
    raise SystemExit(_retry_batch(parser.parse_args().retry_pending))

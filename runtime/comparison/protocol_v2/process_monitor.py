"""Linux process identity and bounded tree cleanup, without command/env inspection."""
from pathlib import Path
import os
import signal
import time

try:
    BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
except OSError:
    BOOT_ID = None


def bind_parent(expected_pid):
    """Terminate on parent death, including SIGKILL; close the startup race."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # Linux PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "could not attach process lifetime to supervisor")
    if os.getppid() != int(expected_pid):
        raise Interrupted(signal.SIGTERM)


class Interrupted(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__("execution interrupted by "+signal.Signals(signum).name)


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "parent": int(fields[1]), "session": int(fields[3]),
                "start": fields[19], "state": fields[0], "boot_id": BOOT_ID}
    except (OSError, ValueError, IndexError):
        return None


def alive(pid, start):
    item = process(pid)
    return bool(item and item["start"] == start and item["state"] != "Z")


def descendants(pid, start):
    """Snapshot owned identities while the root is live, for orphan cleanup later.

    No command lines or environments are read. Start ticks prevent PID reuse
    from targeting unrelated processes after an interrupted controller exits.
    """
    root = process(pid)
    if not root or root["start"] != start or root["state"] == "Z":
        return []
    snapshot = [p for path in Path("/proc").iterdir()
                if path.name.isdigit() and (p := process(int(path.name)))]
    selected = {int(pid)}
    while True:
        expanded = selected | {p["pid"] for p in snapshot if p["parent"] in selected}
        if expanded == selected:
            break
        selected = expanded
    return [p for p in snapshot if p["pid"] in selected and p["state"] != "Z"]


def session_members(leader):
    """Recover children reparented between ancestry snapshots in our own session.

    The supervisor creates a new session for each child. A recycled session
    leader identity is rejected; a live member keeps its session ID reserved.
    """
    if (leader.get("session") != leader["pid"]
            or (leader.get("boot_id") is not None and leader["boot_id"] != BOOT_ID)):
        return []
    current = process(leader["pid"])
    if current and current["start"] != leader["start"]:
        return []
    return [p for path in Path("/proc").iterdir() if path.name.isdigit()
            and (p := process(int(path.name))) and p["session"] == leader["pid"] and p["state"] != "Z"]


def stop_owned(identities, grace=5):
    """Stop a saved tree even if its former parent is already dead/reparented."""
    targets = {(int(p["pid"]), str(p["start"])): p for p in identities
               if int(p["pid"]) > 1 and int(p["pid"]) != os.getpid()
               and (p.get("boot_id") is None or p["boot_id"] == BOOT_ID)}
    errors = []
    def send(signum):
        for (pid, start), _ in targets.items():
            if alive(pid, start):
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append({"pid": pid, "signal": int(signum), "error_type": type(exc).__name__})
    send(signal.SIGTERM)
    deadline = time.monotonic()+max(0, grace)
    while time.monotonic() < deadline and any(alive(pid, start) for pid, start in targets):
        time.sleep(.05)
    send(signal.SIGKILL)
    # Give an acknowledged SIGKILL a short chance to reach terminal state.
    deadline = time.monotonic()+.25
    while time.monotonic() < deadline and any(alive(pid, start) for pid, start in targets):
        time.sleep(.01)
    return {"tracked": len(targets), "errors": errors, "remaining_alive":
            [pid for pid, start in targets if alive(pid, start)]}


def stop_tree(pid, start, grace=5):
    if not alive(pid, start):
        return
    snapshot = [p for path in Path("/proc").iterdir() if path.name.isdigit() and (p := process(int(path.name)))]
    selected = {pid}
    while True:
        expanded = selected | {p["pid"] for p in snapshot if p["parent"] in selected}
        if expanded == selected:
            break
        selected = expanded
    targets = [p for p in snapshot if p["pid"] in selected]
    for p in targets:
        if alive(p["pid"], p["start"]):
            try:
                os.kill(p["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+grace
    while time.monotonic() < deadline and any(alive(p["pid"], p["start"]) for p in targets):
        time.sleep(.05)
    for p in targets:
        if alive(p["pid"], p["start"]):
            try:
                os.kill(p["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass

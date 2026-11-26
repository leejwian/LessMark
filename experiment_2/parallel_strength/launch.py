"""Start one strength-selection worker on each selected full NVIDIA GPU."""
import argparse
import csv
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from runtime.comparison.protocol_v2 import process_monitor

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get('LERRA_OUTPUT', PROJECT / 'runs/experiment_2_strength_selection')).expanduser().resolve()
OPS = OUTPUT.with_name(OUTPUT.name + '.parallel')
ACTIVE = {'activating', 'active', 'reloading'}


@dataclass(frozen=True)
class GPU:
    uuid: str
    free_mib: int | None
    pids: tuple[int | None, ...]


def unit_name(slot):
    campaign = hashlib.sha256(os.fsencode(OUTPUT)).hexdigest()[:12]
    return f'lerra-exp2-{campaign}-gpu{slot}.service'


def _command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} exited {result.returncode}: {result.stderr.strip()}')
    return result.stdout


def inventory():
    def query(kind, fields):
        output = _command(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'])
        return csv.reader(output.splitlines(), skipinitialspace=True)

    devices, processes = {}, {}
    for index, uid, free in query('gpu', 'index,uuid,memory.free'):
        slot = int(index)
        if slot in devices or uid in processes or not re.fullmatch(r'GPU-[A-Fa-f0-9-]+', uid):
            raise ValueError('nvidia-smi returned an invalid or duplicate GPU identity')
        processes[uid] = []
        devices[slot] = uid, int(free) if free.isdecimal() else None
    for uid, pid in query('compute-apps', 'gpu_uuid,pid'):
        if uid not in processes:
            raise ValueError(f'Cannot attribute GPU compute process to a physical device: {uid}')
        processes[uid].append(int(pid) if pid.isdecimal() else None)
    return {slot: GPU(uid, free, tuple(processes[uid])) for slot, (uid, free) in devices.items()}


def visible_uuids():
    # CUDA ordinals can differ from nvidia-smi indices. Query the driver with
    # the inherited visibility mask intact, without creating a training context.
    driver = ctypes.CDLL('libcuda.so.1')
    def check(code):
        if code:
            raise RuntimeError(f'CUDA GPU visibility query failed with driver error {code}')
    check(driver.cuInit(0))
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    result = set()
    for index in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), index))
        raw = (ctypes.c_ubyte * 16)()
        check(get_uuid(ctypes.byref(raw), device))
        result.add('GPU-' + str(uuid.UUID(bytes=bytes(raw))))
    return result


def selected_slots(devices, requested=None):
    by_uuid = {gpu.uuid: slot for slot, gpu in devices.items()}
    allowed = set(devices)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible = visible_uuids()
        if visible - by_uuid.keys():
            raise ValueError('Expected full NVIDIA GPUs matching the nvidia-smi inventory')
        allowed = {by_uuid[uid] for uid in visible}
    if requested is None:
        requested = os.environ.get('LERRA_GPU_IDS')
    if requested is None or not requested.strip():
        slots = sorted(allowed)
    else:
        slots = []
        for value in requested.split(','):
            value = value.strip()
            slot = int(value) if value.isdecimal() else by_uuid.get(value)
            if slot not in allowed or slot in slots:
                raise ValueError(f'Unknown, unallocated, or repeated GPU {value!r}')
            slots.append(slot)
    if not slots:
        raise ValueError('No physical GPUs visible in this allocation')
    return slots


def unit_state(name):
    out = _command(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '-p', 'ControlGroup',
                    '-p', 'LoadState'])
    state = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
    if not state.get('ActiveState') or not state.get('LoadState'):
        raise ValueError(f'Incomplete systemd status for {name}')
    return state


def owns(pid, control_group):
    if not pid or not control_group or control_group == '/':
        return False
    try:
        lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    except OSError:
        return False
    return any((group := line.split(':', 2)[-1]) == control_group
               or group.startswith(control_group + '/') for line in lines)


def live_task(slot, gpu_uuid):
    path = OPS / f'slot-{slot}.json'
    try:
        status = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if (status.get('gpu_uuid') == gpu_uuid and status.get('boot_id') == process_monitor.BOOT_ID
            and process_monitor.alive(status['pid'], status.get('start_ticks'))):
        return status
    return {}


def admit(slot, gpu, managed, task, minimum):
    if managed.get('ActiveState') in ACTIVE:
        unit = unit_name(slot)
        if any(not owns(pid, managed.get('ControlGroup')) for pid in gpu.pids):
            return 'blocked', f'{unit} shares the GPU with unknown compute processes: {gpu.pids}'
        return 'existing', unit
    if task and all(pid == task['pid'] for pid in gpu.pids):
        return 'existing', f"campaign worker PID {task['pid']}"
    if managed.get('ActiveState') not in (None, 'inactive'):
        return 'blocked', f"service is {managed['ActiveState']}"
    if gpu.pids:
        return 'blocked', f'GPU already has compute processes: {gpu.pids}'
    if gpu.free_mib is None or gpu.free_mib < minimum:
        return 'blocked', f'free memory {gpu.free_mib} MiB; {minimum} MiB required'
    return 'start', f'{gpu.free_mib} MiB free'


def worker_command(slot):
    return [sys.executable, '-B', '-u', '-m', 'experiment_2.parallel_strength.worker',
            '--output', str(OUTPUT), '--slot', str(slot)]


def submit(slot, gpu):
    # Explicitly carry the training environment across the user service boundary.
    names = ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'TORCH_EXTENSIONS_DIR', 'TMPDIR',
             'LERRA_OUTPUT', 'LERRA_DATA_ROOT', 'LERRA_NOTIFICATION_MODE', 'LERRA_MAIL_PASSWORD_FILE', 'LERRA_MAIL_ACCOUNT')
    environment = [f'--setenv={name}={os.environ[name]}' for name in names if name in os.environ]
    _command(['systemd-run', '--user', '--quiet', '--collect', '--no-ask-password',
              '--unit=' + unit_name(slot), '--property=Type=exec', '--property=OOMPolicy=continue',
              '--working-directory=' + str(PROJECT), '--setenv=CUDA_VISIBLE_DEVICES=' + gpu.uuid,
              *environment, *worker_command(slot)])


def launch(check=False, foreground=False, requested=None, children=None):
    devices = inventory()
    slots = selected_slots(devices, requested)
    minimum = int(os.environ.get('LERRA_MIN_FREE_MIB', '20480'))
    if minimum < 0:
        raise ValueError('LERRA_MIN_FREE_MIB must be nonnegative')
    blocked = False
    for slot in slots:
        gpu = devices[slot]
        service = {} if foreground else unit_state(unit_name(slot))
        action, reason = admit(slot, gpu, service, live_task(slot, gpu.uuid), minimum)
        if action == 'start' and not check:
            if foreground:
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu.uuid}
                children.append(subprocess.Popen(worker_command(slot), cwd=PROJECT, env=env))
                action = 'started'
            else:
                submit(slot, gpu)
                action, reason = 'submitted', unit_name(slot)
        blocked |= action == 'blocked'
        print(f'GPU {slot}: {action} — {reason}', flush=True)
    return int(blocked)


def run_workers(args):
    children = []
    try:
        OPS.mkdir(parents=True, exist_ok=True)
        with (OPS / 'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = launch(foreground=args.foreground, requested=args.gpus, children=children)
        # A failed worker can leave a pending claim. End the foreground group
        # so that the user can explicitly resume it from its saved checkpoints.
        while children:
            for child in children.copy():
                code = child.poll()
                if code is not None:
                    children.remove(child)
                    if code:
                        return code if code > 0 else 128 - code
            if children:
                time.sleep(0.5)
        return status
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children:
            child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='inspect GPUs without starting workers')
    parser.add_argument('--foreground', action='store_true', help='run on this terminal without a systemd user manager')
    parser.add_argument('--gpus', help='comma-separated nvidia-smi indices or full GPU UUIDs (default LERRA_GPU_IDS or allocated GPUs)')
    args = parser.parse_args(argv)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    try:
        if args.check:
            return launch(check=True, foreground=args.foreground, requested=args.gpus)
        if args.foreground:
            for signum in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, stop)
        return run_workers(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'{exc}\n' + ('' if args.foreground else 'For a host without user systemd use parallel --foreground.\n'))


if __name__ == '__main__':
    raise SystemExit(main())

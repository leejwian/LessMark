"""Show live GPU occupation and the latest update reported by each campaign worker."""
import argparse
import csv
import json
import sys
import subprocess

from experiment_2.code import run_strength_selection as study
from experiment_2.parallel_strength.launch import OUTPUT, live_task, unit_name, unit_state


def output(*command):
    return subprocess.check_output(command, text=True, timeout=10).strip()


def rows(query):
    data = output('nvidia-smi', query, '--format=csv,noheader,nounits')
    return list(csv.reader(data.splitlines(), skipinitialspace=True))


def inspect_slots():
    gpu_rows = rows('--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total')
    processes = rows('--query-compute-apps=gpu_uuid,pid,used_memory')
    snapshot = []
    for index, uuid, utilization, used, total in gpu_rows:
        unit = unit_name(int(index))
        task = live_task(int(index), uuid)
        try:
            state = unit_state(unit)['ActiveState']
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            state = 'unavailable'
        if task and state not in ('active', 'activating', 'reloading'):
            unit, state = None, 'foreground'
        pids = [pid for gpu, pid, _ in processes if gpu == uuid]
        snapshot.append({'gpu': int(index), 'utilization_pct': int(utilization), 'used_mib': int(used),
                         'total_mib': int(total), 'gpu_compute_pids': pids, 'unit': unit,
                         'service_state': state, 'latest_task': task})
    return snapshot


def inspect_controls(root, slots, verify=False):
    plan = study.load_saved_plan(root)
    training = set()
    for slot in slots:
        task = slot['latest_task']
        name = task.get('condition') or ''
        if task.get('state') == 'running' and not task.get('method'):
            training.add(name)
        elif name.endswith('/clean'):
            training.add(name.split('/')[0])
    controls = []
    for condition in plan['conditions']:
        clean = root / condition / 'clean'
        expected_path = clean / 'terminal.pt'
        refs = [study.read_json(path)['C_T'] for method in plan['methods']
                for path in (root / condition / method).glob('candidate_*/result.json')]
        receipt = clean / 'ready.json'
        sha256 = None
        if receipt.exists():
            entry = study.read_json(receipt)
            link = entry['checkpoint']
            if (link['path'] != expected_path.relative_to(root).as_posix()
                    or entry['step'] != plan['terminal_updates'][condition]
                    or any(ref != link for ref in refs)):
                raise ValueError(f'{condition}: inconsistent clean endpoint or candidate references')
            study.check_bound(root, link, full=verify)
            state, sha256 = 'ready', link['sha256']
        elif refs:
            raise ValueError(f'{condition}: completed candidate has no published clean endpoint')
        elif expected_path.exists():
            state = 'validation_pending'
        elif condition in training:
            state = 'training'
        elif (clean / 'progress.pt').exists():
            state = 'checkpoint_available'
        else:
            state = 'pending'
        controls.append({'condition': condition, 'state': state, 'terminal_updates': plan['terminal_updates'][condition],
                         'candidate_references': len(refs), 'sha256': sha256})
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--clean', action='store_true', help='show the one clean endpoint per condition and check candidate references')
    parser.add_argument('--verify', action='store_true', help='also rehash each ready clean checkpoint with --clean')
    args = parser.parse_args()
    if args.verify and not args.clean:
        parser.error('--verify requires --clean')
    try:
        slots = inspect_slots()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if not args.clean:
            parser.exit(1, f'GPU status unavailable: {exc}\n')
        print(f'GPU status unavailable: {exc}', file=sys.stderr)
        slots = []
    if args.clean:
        controls = inspect_controls(OUTPUT, slots, verify=args.verify)
        if args.json:
            print(json.dumps(controls, indent=2))
        else:
            print('model/dataset                      clean status          target updates  completed candidates  C_T SHA-256')
            for c in controls:
                print(f"{c['condition']:<34} {c['state']:<22} {c['terminal_updates']:>8}  "
                      f"{c['candidate_references']:>20}  {c['sha256'] or '-'}")
        return
    if args.json:
        print(json.dumps(slots, indent=2))
        return
    print('GPU  VRAM (MiB)     GPU%  compute PID      service      latest task')
    for s in slots:
        t = s['latest_task']
        task = ' '.join(str(t[key]) for key in ('condition', 'method', 'phase') if t.get(key))
        if t.get('completed_step') is not None:
            task += f" {t['completed_step']}/{t.get('terminal_updates', '?')}"
        memory = f"{s['used_mib']}/{s['total_mib']}"
        print(f"{s['gpu']:>3}  {memory:<13}  {s['utilization_pct']:>3}%  "
              f"{','.join(s['gpu_compute_pids']) or '-':<15}  {s['service_state']:<11}  {task or '-'}")


if __name__ == '__main__':
    main()

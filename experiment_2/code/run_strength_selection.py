"""Run or report the fixed-endpoint experiment 2 strength comparison."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parent
sys.path.insert(0, str(ROOT))
from runtime.comparison.protocol_v2.storage import digest, identity
from experiment_2.code.strength_run_monitor import set_phase
from experiment_2.code.input_paths import data_root, read_registered_json

CONFIG = EXPERIMENT / 'config/NEXT_EXPERIMENT_DESIGN.json'


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


@contextlib.contextmanager
def atomic_output(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if mode == 'wb' else {'encoding': 'utf-8', 'newline': ''}
    with tempfile.NamedTemporaryFile(mode, dir=path.parent, prefix='.writing-', delete=False, **options) as out:
        temp = Path(out.name)
        try:
            yield out
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temp, path)
            parent_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            temp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path, 'w') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def build_plan():
    design = read_registered_json(CONFIG)
    strength = design['strength']
    recipe_binding = strength['campaign_training']
    recipe_path = CONFIG.parent / recipe_binding['path_relative_to_this_file']
    if digest(recipe_path) != recipe_binding['sha256']:
        raise ValueError('Training settings differ from the registered design')
    recipes = read_json(recipe_path)
    binding = strength['candidate_table']
    table = CONFIG.parent / binding['path_relative_to_this_file']
    if digest(table) != binding['sha256']:
        raise ValueError('Candidate table SHA-256 differs from the registered design')
    with table.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    conditions = list(design['embedding']['fixed_source_checkpoints'])
    methods = list(design['method_display_names'])
    groups = {condition: {} for condition in conditions}
    for row in rows:
        condition, method = row['model_dataset'], row['method']
        strengths = [row[k] for k in strength['candidate_columns_ascending']]
        center = Decimal(str(strength['lessmark_alpha_by_model'][condition] if method == 'lerra'
                             else strength['paper_strength_by_method'][method]))
        if (method in groups[condition] or list(map(Decimal, strengths)) !=
                [center * Decimal(str(factor)) for factor in strength['multipliers']]):
            raise ValueError(f'Invalid or duplicate grid for {condition}/{method}')
        groups[condition][method] = strengths
    if len(rows) != 42 or any(set(group) != set(methods) for group in groups.values()):
        raise ValueError('Expected 7 conditions with six distinct methods each')
    schedule = design['checkpoint_schedule']
    endpoints = {c: schedule['classification_terminal_updates'] if c.endswith('_cifar100')
                 else schedule['generation_updates'][-1] for c in conditions}
    sources = list((EXPERIMENT / 'code').glob('*.py')) + list((ROOT / 'runtime/comparison/protocol_v2').glob('*.py'))
    sources += [ROOT / 'experiments/sit_internal_readout_20261110/internal_carrier.py',
                EXPERIMENT / 'parallel_strength/worker.py', EXPERIMENT / 'parallel_strength/source_pixels.py']
    sources = sorted(p for p in sources if not p.name.startswith('test_'))
    recipient = design['payload']['single_watermark']['recipient_id']
    plan = {'schema': 1, 'design_sha256': digest(CONFIG), 'candidate_table_sha256': digest(table),
            'training_recipe_sha256': recipe_binding['sha256'],
            'implementation_sha256': identity({str(p.relative_to(ROOT)): digest(p) for p in sources}),
            'recipes': recipes, 'conditions': conditions, 'methods': methods, 'strengths': groups,
            'direct_methods': design['embedding']['direct_methods'], 'terminal_updates': endpoints,
            'recipient_id': recipient, 'message_bits': 256,
            'payload': design['payload']['physical_payloads_by_recipient'][str(recipient)],
            'validation_size': {c: (design['data']['source_CIFAR100']['counts']['validation'] if c.endswith('_cifar100')
                                    else design['generation_evaluation']['pairs_per_split']) for c in conditions}}
    plan['study_id'] = identity(plan)
    return design, plan


def load_saved_plan(root):
    plan = read_json(root / 'plan.json')
    if identity({k: v for k, v in plan.items() if k != 'study_id'}) != plan['study_id']:
        raise ValueError(f'Damaged plan at {root}')
    return plan


@lru_cache(maxsize=1024)
def _file_hash(path, size, mtime_ns, ctime_ns):
    return digest(path)


def file_hash(path):
    s = path.stat()
    return _file_hash(path, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def binding(root, path):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': file_hash(path)}


def check_bound(root, link, *, full=True):
    path = root / link['path']
    if path.stat().st_size != link['bytes'] or (full and file_hash(path) != link['sha256']):
        raise ValueError(f'Artifact differs from its receipt: {path}')
    return path


def save_tensor(path, value):
    import torch
    with atomic_output(path, 'wb') as out:
        torch.save(value, out)


def load_tensor(path):
    import torch
    # Campaign-owned artifacts carry optimizer and decoder state as well as tensors.
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def rng_snapshot(device):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def seed_arm(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)


def scored_readout(recovery, target):
    raw = recovery.bits
    if hasattr(raw, 'tolist'):
        raw = raw.tolist()
    valid_bits = isinstance(raw, (list, tuple)) and len(raw) == len(target) and all(
        isinstance(bit, (bool, int, float)) and bit in (0, 1) for bit in raw)
    valid = recovery.status == 'ok' and valid_bits
    count = sum(int(bit) != truth for bit, truth in zip(raw, target)) if valid_bits else None
    details = getattr(recovery, 'details', None) or {}
    reason = None if valid else str(details.get('reason') or (recovery.status if recovery.status != 'ok' else 'invalid_bits'))
    return {'decoder_status': recovery.status, 'valid': valid, 'reason': reason,
            'returned_bit_count': len(raw) if isinstance(raw, (list, tuple)) else None,
            'bits': ''.join(str(int(b)) for b in raw) if valid else None,
            'bit_errors': count if valid else None, 'ber': count / len(target) if valid else None,
            'decoder_details': details, 'diagnostic_bit_errors': count if not valid else None}


def selection(records):
    if len(records) != 5:
        raise ValueError('All five measured endpoints are needed to rank a combination')
    valid = [r for r in records if r['readout']['valid']]
    return min(valid, key=lambda r: (r['readout']['bit_errors'], -Decimal(r['strength']))) if valid else None


def validate_quality(value, classifier, expected):
    status = value.get('quality_status', 'ok')
    if status not in ('ok', 'undefined'):
        raise ValueError('Unknown terminal quality status')
    if classifier:
        if value['examples'] != expected:
            raise ValueError('Incomplete classification validation')
        invalid = []
        for arm in ('marked', 'clean'):
            count = value[f'{arm}_correct']
            percent = value[f'{arm}_top1_pct']
            bad = value.get(f'{arm}_invalid_examples', 0)
            if type(bad) is not int or not 0 <= bad <= expected:
                raise ValueError('Invalid nonfinite prediction count')
            invalid.append(bad)
            if bad:
                if count is not None or percent is not None:
                    raise ValueError('Undefined predictions cannot have aggregate top-1 accuracy')
            elif type(count) is not int or not 0 <= count <= expected:
                raise ValueError('Invalid classification validation counts')
            elif not math.isclose(percent, 100 * count / expected, abs_tol=1e-9):
                raise ValueError(f'Inconsistent top-1 accuracy for {arm}')
        if (status == 'undefined') != any(invalid):
            raise ValueError('Classification quality status disagrees with prediction counts')
        if any(invalid):
            if value['accuracy_drop_pp'] is not None:
                raise ValueError('Undefined accuracy difference')
        elif not math.isclose(value['accuracy_drop_pp'], value['clean_top1_pct'] - value['marked_top1_pct'], abs_tol=1e-9):
            raise ValueError('Inconsistent accuracy difference')
    else:
        bad = value.get('invalid_pairs', 0)
        if value['pairs'] != expected or type(bad) is not int or not 0 <= bad <= expected:
            raise ValueError('Incomplete RGB validation')
        if (status == 'undefined') != (bad > 0):
            raise ValueError('RGB quality status disagrees with pair counts')
        metrics = (value['mean_mse'], value['mean_ssim'])
        if not (all(m is None for m in metrics) if bad else all(math.isfinite(m) for m in metrics)):
            raise ValueError('Invalid aggregate RGB metrics')
    return value


class ConditionRun:
    def __init__(self, root, design, plan, condition, device, interval, backend, prepare_assets, bind_watermark):
        self.root, self.design, self.plan = root, design, plan
        self.condition, self.device, self.interval = condition, device, interval
        self.backend = backend
        self.prepare_assets, self.bind_watermark = prepare_assets, bind_watermark
        self.base = root / condition
        self.clean_dir = self.base / 'clean'
        self.cache = self.clean_dir / 'validation'
        self.total = plan['terminal_updates'][condition]
        self.target = [int(b) for b in design['payload']['physical_payloads_by_recipient'][str(plan['recipient_id'])]]
        self.classifier = condition.endswith('_cifar100')
        self.quality_count = (design['data']['source_CIFAR100']['counts']['validation'] if self.classifier
                              else design['generation_evaluation']['pairs_per_split'])
        self.clean = None

    def reset(self):
        seed_arm(self.plan['recipes']['shared_arm_seed'], self.device)
        self.backend.reset_training()

    def load_weights(self, receipt):
        record = load_tensor(check_bound(self.root, receipt))
        if record['step'] != self.total or record['study_id'] != self.plan['study_id']:
            raise ValueError('Incorrect terminal checkpoint')
        self.backend.model.load_state_dict(record['model'], strict=True)

    def endpoint(self, directory, *, method=None, adapter=None, strength=None):
        final = directory / 'terminal.pt'
        progress = directory / 'progress.pt'
        arm = directory.relative_to(self.root).as_posix()
        stamp = {'study_id': self.plan['study_id'], 'arm': arm}
        direct = method in self.plan['direct_methods']
        cached = final.exists()
        direct_readout = None
        if not cached:
            if direct:
                self.backend.model.eval()
                recovery = adapter.insert(float(strength))
                if recovery is not None:
                    direct_readout = scored_readout(recovery, self.target)
                save_tensor(final, {**stamp, 'step': self.total, 'model': self.backend.model.state_dict(),
                                    'direct_readout': direct_readout})
            else:
                start = 0
                if progress.exists():
                    previous = load_tensor(progress)
                    if any(previous[k] != v for k, v in stamp.items()):
                        raise ValueError(f'Incompatible progress checkpoint: {progress}')
                    start = previous['step']
                    if not 0 < start < self.total:
                        raise ValueError(f'Invalid progress step: {start}')
                    self.backend.model.load_state_dict(previous['model'], strict=True)
                    self.backend.load_training_state_dict(previous['training'])
                    restore_rng(previous['rng'], self.device)
                    del previous
                loss = (lambda: float(strength) * adapter.loss()) if adapter is not None else None
                for step in range(start, self.total):
                    measurement = self.backend.train_step(step, loss)
                    completed = step + 1
                    if completed == self.total or completed % self.interval == 0:
                        destination = final if completed == self.total else progress
                        save_tensor(destination, {**stamp, 'step': completed, 'model': self.backend.model.state_dict(),
                                                  'training': self.backend.training_state_dict(), 'rng': rng_snapshot(self.device),
                                                  'last_training_measurement': measurement})
                        print(f'{arm}: {completed}/{self.total}', flush=True)
        else:
            record = load_tensor(final)
            if any(record[k] != v for k, v in stamp.items()) or record['step'] != self.total:
                raise ValueError(f'Incompatible endpoint: {final}')
            self.backend.model.load_state_dict(record['model'], strict=True)
            direct_readout = record.get('direct_readout')
        if direct_readout is not None and not (directory / 'readout.json').exists():
            write_json(directory / 'readout.json', direct_readout)
        self.backend.model.eval()
        progress.unlink(missing_ok=True)
        return binding(self.root, final)

    def ensure_clean(self):
        receipt_path = self.clean_dir / 'ready.json'
        if receipt_path.exists():
            record = read_json(receipt_path)
            self.clean = record['checkpoint']
            check_bound(self.root, self.clean)
            return
        self.reset()
        self.clean = self.endpoint(self.clean_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        set_phase('clean_validation', condition=self.condition)
        self.backend.prepare_quality(self.cache)
        write_json(receipt_path, {'checkpoint': self.clean, 'step': self.total})

    def method_assets(self, method):
        directory = self.base / method / 'assets'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'decoder.pt'
        receipt_path = directory / 'ready.json'
        if receipt_path.exists():
            link = read_json(receipt_path)['assets']
            return load_tensor(check_bound(self.root, link)), link
        if not path.exists():
            if method in self.plan['direct_methods']:
                self.load_weights(self.clean)
            else:
                self.reset()
            self.backend.model.eval()
            assets = self.prepare_assets(self.design, self.condition, method, self.backend, cache_dir=directory)
            save_tensor(path, assets)
        else:
            assets = load_tensor(path)
        link = binding(self.root, path)
        write_json(receipt_path, {'assets': link})
        return assets, link

    def run_method(self, method):
        strengths = self.plan['strengths'][self.condition][method]
        area = self.base / method
        ready = [area / f'candidate_{i}' / 'result.json' for i in range(1, 6)]
        if all(p.exists() for p in ready):
            return
        set_phase('decoder_assets', condition=self.condition, method=method)
        assets, asset_link = self.method_assets(method)
        direct = method in self.plan['direct_methods']
        for i, (strength, result) in enumerate(zip(strengths, ready), 1):
            if result.exists():
                continue
            set_phase('strength_endpoint', condition=self.condition, method=method, candidate=i)
            if not (result.parent / 'terminal.pt').exists():
                if direct:
                    self.load_weights(self.clean)
                else:
                    self.reset()
            adapter = self.bind_watermark(self.design, self.condition, method, self.backend, assets)
            marked = self.endpoint(result.parent, method=method, adapter=adapter, strength=strength)
            set_phase('terminal_readout', condition=self.condition, method=method, candidate=i)
            readout_file = result.parent / 'readout.json'
            if not readout_file.exists():
                readout = scored_readout(adapter.recover(), self.target)
                write_json(readout_file, readout)
            set_phase('candidate_validation', condition=self.condition, method=method, candidate=i)
            measured = (self.backend.quality(self.cache) if self.classifier else
                        self.backend.quality(self.cache, progress_dir=result.parent / 'validation'))
            quality = validate_quality(measured, self.classifier, self.quality_count)
            write_json(result, {'condition': self.condition, 'method': method, 'candidate': i,
                                'strength': strength, 'source_updates': self.total,
                                'embedding_updates': 0 if direct else self.total,
                                'C_T': self.clean, 'W_T': marked, 'decoder_assets': asset_link,
                                'readout': read_json(readout_file), 'quality': quality})
            del adapter


def group_results(root, plan, condition, method, *, verify=False):
    records = []
    strengths = plan['strengths'][condition][method]
    for i, strength in enumerate(strengths, 1):
        path = root / condition / method / f'candidate_{i}' / 'result.json'
        if not path.exists():
            continue
        r = read_json(path)
        if (r['condition'], r['method'], r['candidate'], r['strength'], r['source_updates']) != (
                condition, method, i, strength, plan['terminal_updates'][condition]):
            raise ValueError(f'Result disagrees with registered candidate: {path}')
        if r['embedding_updates'] != (0 if method in plan['direct_methods'] else r['source_updates']):
            raise ValueError(f'Incorrect insertion budget at {path}')
        for key, relative in (('W_T', path.parent / 'terminal.pt'), ('C_T', root / condition / 'clean/terminal.pt'),
                              ('decoder_assets', root / condition / method / 'assets/decoder.pt')):
            if r[key]['path'] != relative.relative_to(root).as_posix():
                raise ValueError(f'Incorrect {key} reference at {path}')
            if records and key != 'W_T' and records[0][key] != r[key]:
                raise ValueError(f'Candidates refer to different {key} artifacts at {path}')
            check_bound(root, r[key], full=verify)
        readout = r['readout']
        if readout != read_json(path.parent / 'readout.json'):
            raise ValueError(f'Result disagrees with the saved terminal readout at {path}')
        if readout['valid']:
            bits, target = readout['bits'], plan['payload']
            if (readout['decoder_status'] != 'ok' or not isinstance(bits, str) or set(bits) - {'0', '1'}
                    or readout['returned_bit_count'] != len(target) or len(bits) != len(target)):
                raise ValueError(f'Invalid ranked output at {path}')
            errors = sum(b != t for b, t in zip(bits, target))
            if readout['bit_errors'] != errors or readout['ber'] != errors / len(target):
                raise ValueError(f'Inconsistent terminal BER at {path}')
        elif readout['bit_errors'] is not None or readout['ber'] is not None:
            raise ValueError(f'Invalid readout has a ranked BER at {path}')
        validate_quality(r['quality'], condition.endswith('_cifar100'), plan['validation_size'][condition])
        records.append(r)
    return records


def write_csv(path, rows, columns):
    with atomic_output(path, 'w') as f:
        table = csv.DictWriter(f, fieldnames=columns)
        table.writeheader()
        for row in rows:
            table.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                            for k, v in row.items()})


def update_report(root, plan, *, verify=False):
    # Independent condition workers can finish concurrently. Collect only atomically published results.
    with lock(root / '.report.lock'):
        groups, candidates = [], []
        for condition in plan['conditions']:
            for method in plan['methods']:
                records = group_results(root, plan, condition, method, verify=verify)
                status, winner = 'pending', None
                if len(records) == 5:
                    winner = selection(records)
                    status = 'selected' if winner else 'selection_failure'
                center = plan['strengths'][condition][method][2]
                group = {'condition': condition, 'method': method, 'reference_strength': center,
                         'measured': len(records), 'status': status,
                         'selected_strength': None, 'bit_errors': None, 'ber': None,
                         'W_T': None, 'C_T': None, 'decoder_assets': None}
                if winner:
                    group.update(selected_strength=winner['strength'], bit_errors=winner['readout']['bit_errors'],
                                 ber=winner['readout']['ber'], **{k: winner[k] for k in ('W_T', 'C_T', 'decoder_assets')})
                groups.append(group)
                for record in records:
                    fields = ('condition', 'method', 'candidate', 'strength', 'source_updates', 'embedding_updates')
                    entry = {k: record[k] for k in fields}
                    entry.update(reference_strength=center, multiplier=str(Decimal(record['strength']) / Decimal(center)))
                    entry.update({k: record['readout'][k] for k in ('decoder_status', 'valid', 'reason', 'returned_bit_count',
                                                                    'bit_errors', 'ber', 'diagnostic_bit_errors')})
                    entry.update(record['quality'])
                    entry.update(selected=None if status == 'pending' else record is winner, checkpoint=record['W_T']['path'])
                    candidates.append(entry)
        counts = {s: sum(r['status'] == s for r in groups) for s in ('selected', 'selection_failure', 'pending')}
        result = {'study_id': plan['study_id'], 'counts': counts, 'measured_candidates': len(candidates), 'groups': groups}
        fixed = ['condition', 'method', 'candidate', 'reference_strength', 'multiplier', 'strength',
                 'source_updates', 'embedding_updates',
                 'decoder_status', 'valid', 'reason', 'returned_bit_count', 'bit_errors', 'ber',
                 'diagnostic_bit_errors', 'selected', 'checkpoint']
        quality_columns = sorted({key for row in candidates for key in row} - set(fixed))
        write_csv(root / 'candidates.csv', candidates, fixed + quality_columns)
        selections = [{k: (v['path'] if isinstance(v, dict) else v) for k, v in g.items()} for g in groups]
        write_csv(root / 'selected.csv', selections, list(groups[0]))
        write_json(root / 'selection.json', result)
        return result


def ensure_plan(root, plan):
    with lock(root / '.plan.lock'):
        if (root / 'plan.json').exists():
            if load_saved_plan(root)['study_id'] != plan['study_id']:
                raise ValueError('Output directory is bound to a different design, recipe or implementation. Use a separate output directory.')
        else:
            foreign_files = [p.name for p in root.iterdir() if p.name != '.plan.lock' and not p.name.startswith('.writing-')]
            if foreign_files:
                raise ValueError('An empty or previously registered campaign output directory is required')
            write_json(root / 'plan.json', plan)
        verified = root / 'verified_inputs.json'
        inputs = {'study_id': plan['study_id'], 'data_root': str(data_root().resolve())}
        if verified.exists():
            if read_json(verified) != inputs:
                raise ValueError(f'Input verification receipt disagrees with the registered plan at {verified}')
        else:
            from experiment_2.code import verify_design
            verify_design.main()
            write_json(verified, inputs)


def run(args, design, plan):
    root = args.output.resolve()
    conditions = subset(args.condition, plan['conditions'], 'condition')
    methods = subset(args.method, plan['methods'], 'method')
    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        raise ValueError('checkpoint-every must be positive')
    ensure_plan(root, plan)
    for condition in conditions:
        set_phase('condition_setup', condition=condition)
        with lock(root / condition / '.worker.lock'):
            pending = [m for m in methods if len(group_results(root, plan, condition, m)) < 5]
            if not pending:
                print(f'{condition}: requested candidates are already complete', flush=True)
                continue
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            import torch
            from experiment_2.code.strength_watermarks import prepare_assets, bind_watermark
            device = torch.device(args.device)
            if device.type == 'cuda':
                device = torch.device('cuda', device.index if device.index is not None else torch.cuda.current_device())
                torch.cuda.set_device(device)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            from importlib.metadata import PackageNotFoundError, version
            libraries = {}
            for name in ('numpy', 'torchvision', 'Pillow', 'scipy', 'pyldpc', 'numba', 'diffusers', 'timm', 'torchdiffeq'):
                try:
                    libraries[name] = version(name)
                except PackageNotFoundError:
                    libraries[name] = None
            environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'libraries': libraries,
                           'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
                           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                           'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}
            env_file = root / condition / 'environment.json'
            if env_file.exists() and read_json(env_file) != environment:
                raise ValueError(f'{condition} is registered to a different numerical environment')
            write_json(env_file, environment)
            cls = condition.endswith('_cifar100')
            if cls:
                from experiment_2.code.classification_backend import ClassificationBackend
                backend = ClassificationBackend(design, condition, device, plan['recipes']['classification'])
            else:
                from experiment_2.code.generation_backend import GenerationBackend
                arch = design['embedding']['native_generator_training']['conditions'][condition]['architecture']
                backend = GenerationBackend(design, condition, device, plan['recipes'][arch])
                backend.prepare_training_manifest(root / condition / 'training_inputs.json')
            interval = args.checkpoint_every or (design['checkpoint_schedule']['classification_updates_per_epoch'] if cls else 128)
            job = ConditionRun(root, design, plan, condition, device, interval, backend, prepare_assets, bind_watermark)
            set_phase('clean_endpoint', condition=condition)
            job.ensure_clean()
            for method in pending:
                job.run_method(method)
                set_phase('selection_report', condition=condition, method=method)
                counts = update_report(root, plan)['counts']
                print(f'{condition}/{method}: {counts}', flush=True)
            del job, backend
    set_phase('final_report')
    return update_report(root, plan)


def subset(requested, allowed, kind):
    if not requested:
        return allowed
    unknown = set(requested) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {kind}: {sorted(unknown)}. Expected one of {allowed}')
    return [v for v in allowed if v in requested]


def print_summary(result):
    print(json.dumps({'measured_candidates': result['measured_candidates'], **result['counts']}), flush=True)


def execute_campaign(args):
    set_phase('registered_design')
    design, plan = build_plan()
    print_summary(run(args, design, plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan', help='show the complete five-strength grid without loading a model')
    planning.add_argument('--json', action='store_true')
    execute = commands.add_parser('run', help='execute or resume all five candidates for the chosen combinations')
    execute.add_argument('--output', required=True, type=Path, help='dedicated directory on the checkpoint volume')
    execute.add_argument('--condition', action='append', help='registered condition, repeatable; default is all seven')
    execute.add_argument('--method', action='append', help='registered method ID, repeatable; default is all six')
    execute.add_argument('--device', default='cuda')
    execute.add_argument('--checkpoint-every', type=int, help='save training progress after this many optimizer updates')
    report = commands.add_parser('report', help='rebuild both tables from completed, registered terminal measurements')
    report.add_argument('--output', required=True, type=Path)
    report.add_argument('--verify-artifacts', action='store_true', help='also verify all referenced SHA-256 hashes')
    args = parser.parse_args()
    if args.command == 'run':
        from experiment_2.code.strength_run_monitor import supervise
        raise SystemExit(supervise(args, execute_campaign))
    if args.command == 'report':
        plan = load_saved_plan(args.output)
        print_summary(update_report(args.output, plan, verify=args.verify_artifacts))
    else:
        _, plan = build_plan()
        if args.json:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        else:
            print('condition,method,source_updates,s0_div_4,s0_div_2,s0,s0_x_2,s0_x_4')
            for c, group in plan['strengths'].items():
                for m, strengths in group.items():
                    print(','.join([c, m, str(plan['terminal_updates'][c]), *strengths]))


if __name__ == '__main__':
    main()

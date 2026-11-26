"""Dispatch unmodified strength-selection jobs across isolated CUDA workers."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import gc
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from experiment_2.code import run_strength_selection as study
from experiment_2.code.strength_run_monitor import set_phase, supervise

from experiment_2.parallel_strength.launch import OUTPUT
from runtime.comparison.protocol_v2.process_monitor import process

PREFERRED = ('mobilenet_v3_large_w1p6_cifar100', 'unet_imagenet', 'unet_caltech256', 'resnet18_cifar100')
MEMORY_FRACTION = 0.85


def try_lock(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def completed(root, condition, method):
    base = root / condition / method
    return all((base / f'candidate_{index}' / 'result.json').is_file() for index in range(1, 6))


def marker_path(ops, condition, method):
    return ops / 'claims' / condition / ((method or 'clean') + '.json')


def choices(root, ops, plan, preferred):
    order = [preferred] + [c for c in plan['conditions'] if c != preferred]
    tasks = []
    for pos, condition in enumerate(order):
        ready = (root / condition / 'clean/ready.json').is_file()
        if not ready:
            if not marker_path(ops, condition, None).exists():
                tasks.append((0, pos, condition, None))
            continue
        methods = [m for m in plan['methods'] if not completed(root, condition, m)]
        active = sum(marker_path(ops, condition, m).exists() for m in methods)
        tasks.extend((active, pos, condition, m) for m in methods if not marker_path(ops, condition, m).exists())
    # Stable sort preserves the original method order within each condition.
    return sorted(tasks, key=lambda t: (t[0], t[1]))


@contextmanager
def claim_work(root, ops, plan, preferred, *, slot=None):
    task = None
    with ExitStack() as held:
        # Serialize dispatch decisions only. GPU work occurs after releasing this lock.
        with study.lock(ops / 'dispatch.lock'):
            for _, _, condition, method in choices(root, ops, plan, preferred):
                with ExitStack() as locks:
                    base = root / condition
                    mode = fcntl.LOCK_EX if method is None else fcntl.LOCK_SH
                    gate = try_lock(base / '.worker.lock', mode)
                    if gate is None:
                        continue
                    locks.enter_context(gate)
                    if method is None:
                        if (base / 'clean/ready.json').exists():
                            continue
                    else:
                        own = try_lock(ops / 'locks' / condition / (method + '.lock'), fcntl.LOCK_EX)
                        if own is None:
                            continue
                        locks.enter_context(own)
                        if completed(root, condition, method):
                            continue
                    path = marker_path(ops, condition, method)
                    study.write_json(path, {'slot': slot, 'pid': os.getpid(), 'condition': condition,
                                            'method': method, 'started_unix': time.time()})
                    held.enter_context(locks.pop_all())
                    task = (condition, method)
                    break
        yield task
        # If the process raises or is killed, the durable claim remains for an explicit resume.
        if task:
            path.unlink()


def recover_claims(ops, slot):
    # Called once on startup with this slot's lifetime lock held. A resumed
    # allocation may have different slot IDs. Release claims only after proving
    # the former owner has exited, even if it was on another GPU.
    with study.lock(ops / 'dispatch.lock'):
        for path in (ops / 'claims').glob('*/*.json'):
            previous = study.read_json(path)['slot']
            if previous == slot:
                path.unlink()
            else:
                idle = try_lock(ops / f'slot-{previous}.lock', fcntl.LOCK_EX)
                if idle is not None:
                    with idle:
                        path.unlink()


def all_complete(root, plan):
    return all(completed(root, c, m) for c in plan['conditions'] for m in plan['methods'])


def cuda_environment():
    # Match the registered execution environment. The per-device allocator limit is operational.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or torch.cuda.device_count() != 1:
        raise RuntimeError('Pin this worker to exactly one physical GPU with CUDA_VISIBLE_DEVICES')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    libraries = {}
    for name in ('numpy', 'torchvision', 'Pillow', 'scipy', 'pyldpc', 'numba', 'diffusers', 'timm', 'torchdiffeq'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    env = {'python': platform.python_version(), 'torch': str(torch.__version__), 'libraries': libraries,
           'device': torch.cuda.get_device_name(device), 'cuda': torch.version.cuda,
           'cudnn': torch.backends.cudnn.version(), 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}
    return device, env


def make_job(root, design, plan, condition, device, environment, preparing):
    from experiment_2.code.strength_watermarks import prepare_assets, bind_watermark
    env_file = root / condition / 'environment.json'
    if env_file.exists():
        if study.read_json(env_file) != environment:
            raise ValueError(f'{condition} is registered to a different numerical environment')
    elif preparing:
        study.write_json(env_file, environment)
    else:
        raise ValueError(f'Missing registered numerical environment for {condition}')
    if condition.endswith('_cifar100'):
        from experiment_2.code.classification_backend import ClassificationBackend
        backend = ClassificationBackend(design, condition, device, plan['recipes']['classification'])
        interval = design['checkpoint_schedule']['classification_updates_per_epoch']
    else:
        from experiment_2.code.generation_backend import GenerationBackend
        from experiment_2.parallel_strength.source_pixels import install_source_cache
        arch = design['embedding']['native_generator_training']['conditions'][condition]['architecture']
        backend = GenerationBackend(design, condition, device, plan['recipes'][arch])
        manifest = root / condition / 'training_inputs.json'
        if not preparing and not manifest.is_file():
            raise ValueError(f'Missing prepared generator training manifest for {condition}')
        backend.prepare_training_manifest(manifest)
        install_source_cache(backend, root)
        interval = 128
    return study.ConditionRun(root, design, plan, condition, device, interval, backend, prepare_assets, bind_watermark)


class SlotStatus:
    def __init__(self, ops, slot, device):
        self.path = ops / f'slot-{slot}.json'
        self.device = device
        self.fields = {'slot': slot, 'pid': os.getpid(), 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
                       'memory_fraction': MEMORY_FRACTION}
        identity = process(os.getpid())
        self.fields.update(boot_id=identity['boot_id'], start_ticks=identity['start'])

    def write(self, state, phase, **fields):
        import torch
        self.fields.update(state=state, phase=phase, updated_unix=time.time(), **fields)
        self.fields.update(cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                           cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        study.write_json(self.path, self.fields)


@contextmanager
def observe_updates(job, status):
    backend = job.backend
    step_fn = backend.train_step
    first = True

    def measured_step(step, watermark_loss=None):
        nonlocal first
        measurement = step_fn(step, watermark_loss)
        if first or (step + 1) % job.interval == 0 or step == 0:
            status.write('running', 'training', completed_step=step + 1)
            first = False
        return measurement

    def validation(fn):
        def observed(*args, **kwargs):
            status.write('running', 'validation', completed_step=job.total)
            return fn(*args, **kwargs)
        return observed

    # No change to update order, inputs, gradients or checkpoint cadence.
    backend.train_step = measured_step
    backend.prepare_quality = validation(backend.prepare_quality)
    backend.quality = validation(backend.quality)
    try:
        yield
    finally:
        for name in ('train_step', 'prepare_quality', 'quality'):
            delattr(backend, name)


def execute(args):
    import torch
    root = args.output.resolve()
    ops = root.with_name(root.name + '.parallel')
    design, plan = study.build_plan()
    preferred = args.preferred_condition or PREFERRED[args.slot % len(PREFERRED)]
    if preferred not in plan['conditions']:
        raise ValueError(f'Unknown preferred condition {preferred}')
    study.ensure_plan(root, plan)
    lifetime = try_lock(ops / f'slot-{args.slot}.lock', fcntl.LOCK_EX)
    if lifetime is None:
        raise RuntimeError(f'Another worker already owns slot {args.slot}')
    with lifetime:
        device, environment = cuda_environment()
        status = SlotStatus(ops, args.slot, device)
        with study.lock(ops / 'dispatch.lock'):
            binding = ops / 'study.json'
            if binding.exists() and study.read_json(binding)['study_id'] != plan['study_id']:
                raise ValueError('Parallel dispatcher belongs to another study')
            if not binding.exists():
                study.write_json(binding, {'study_id': plan['study_id']})
        recover_claims(ops, args.slot)
        job = None
        try:
            while True:
                with claim_work(root, ops, plan, preferred, slot=args.slot) as task:
                    if task is not None:
                        condition, method = task
                        set_phase('condition_setup', condition=condition, method=method)
                        status.write('running', 'initialize', condition=condition, method=method,
                                     completed_step=None, terminal_updates=plan['terminal_updates'][condition])
                        if job is None or job.condition != condition:
                            job = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            job = make_job(root, design, plan, condition, device, environment, method is None)
                        with observe_updates(job, status):
                            set_phase('clean_endpoint', condition=condition, method=method)
                            job.ensure_clean()
                            if method is not None:
                                job.run_method(method)
                        if method is not None:
                            set_phase('selection_report', condition=condition, method=method)
                            study.update_report(root, plan)
                        continue
                job = None
                gc.collect()
                torch.cuda.empty_cache()
                if all_complete(root, plan):
                    study.print_summary(study.update_report(root, plan))
                    status.write('complete', 'finished', condition=None, method=None, completed_step=None)
                    return
                status.write('waiting', 'other_workers', condition=None, method=None, completed_step=None)
                time.sleep(20)
        except BaseException as exc:
            status.write('failed', type(exc).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--preferred-condition')
    args = parser.parse_args()
    if args.slot < 0:
        parser.error('--slot must be a nonnegative physical GPU index')
    raise SystemExit(supervise(args, execute))


if __name__ == '__main__':
    main()

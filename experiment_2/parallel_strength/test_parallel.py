"""Exercise GPU admission and concurrent work claims without starting training or services."""
import fcntl
from collections import Counter
from contextlib import redirect_stdout
from importlib.util import find_spec
import io
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from experiment_2.parallel_strength import launch


class WorkClaims(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'campaign'
        self.ops = Path(directory.name) / 'operations'
        self.root.mkdir()
        self.ops.mkdir()
        self.a = 'unet_imagenet'
        self.b = 'resnet18_cifar100'
        self.plan = {'conditions': [self.a], 'methods': ['uchida', 'greedy']}

    def claim(self, preferred=None):
        from experiment_2.parallel_strength import worker
        return worker.claim_work(self.root, self.ops, self.plan, preferred or self.a)

    def marker(self, *parts):
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}')

    def test_legacy_exclusive_condition_lock_skips_only_its_condition(self):
        self.plan['conditions'].append(self.b)
        condition = self.root / self.a
        condition.mkdir()
        with (condition / '.worker.lock').open('a') as legacy:
            fcntl.flock(legacy, fcntl.LOCK_EX)
            with self.claim() as assignment:
                self.assertEqual(assignment, (self.b, None))
                with self.claim() as unavailable:
                    self.assertIsNone(unavailable)
        with self.claim() as resumed:
            self.assertEqual(resumed, (self.a, None))

    def test_only_one_clean_training_before_any_method(self):
        with self.claim() as first:
            self.assertEqual(first, (self.a, None))
            with self.claim() as competitor:
                self.assertIsNone(competitor)
        # A completed claim without a ready.json cannot release any marked method.
        with self.claim() as retry:
            self.assertEqual(retry, (self.a, None))
            self.marker(self.a, 'clean', 'ready.json')
        with self.claim() as next_job:
            self.assertEqual(next_job[0], self.a)
            self.assertIn(next_job[1], self.plan['methods'])

    def test_ready_clean_is_shared_and_methods_are_exclusive(self):
        self.marker(self.a, 'clean', 'ready.json')
        with self.claim() as first, self.claim() as second:
            self.assertEqual({first, second}, {(self.a, 'uchida'), (self.a, 'greedy')})
            # A later legacy whole-condition worker cannot race either method.
            with (self.root / self.a / '.worker.lock').open('a') as legacy:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.claim() as unavailable:
                self.assertIsNone(unavailable)

    def test_incomplete_method_is_resumable_and_completed_methods_are_skipped(self):
        self.marker(self.a, 'clean', 'ready.json')
        for method, count in [('uchida', 5), ('greedy', 4)]:
            for i in range(1, count + 1):
                self.marker(self.a, method, f'candidate_{i}', 'result.json')
        with self.claim() as job:
            self.assertEqual(job, (self.a, 'greedy'))
            self.marker(self.a, 'greedy', 'candidate_5', 'result.json')
        with self.claim() as done:
            self.assertIsNone(done)

    def test_resume_on_different_gpu_recovers_only_abandoned_claims(self):
        from experiment_2.parallel_strength import worker

        self.marker(self.a, 'clean', 'ready.json')
        with self.assertRaises(InterruptedError):
            with worker.claim_work(self.root, self.ops, self.plan, self.a, slot=1) as first:
                raise InterruptedError('simulated worker failure')
        self.assertTrue(worker.marker_path(self.ops, *first).exists())
        with worker.try_lock(self.ops / 'slot-1.lock', fcntl.LOCK_EX):
            worker.recover_claims(self.ops, 2)
            with worker.claim_work(self.root, self.ops, self.plan, self.a, slot=2) as other:
                self.assertNotEqual(first, other)
        worker.recover_claims(self.ops, 2)
        self.assertFalse(worker.marker_path(self.ops, *first).exists())
        with worker.claim_work(self.root, self.ops, self.plan, self.a, slot=2) as resumed:
            self.assertEqual(resumed, first)


class GPUAdmission(unittest.TestCase):
    idle = {'ActiveState': 'inactive', 'ControlGroup': ''}
    managed = {'ActiveState': 'active', 'ControlGroup': '/user.slice/managed-slot.service'}

    def decision(self, slot=0, pids=(), free=24000, managed=None, task=None):
        gpu = launch.GPU('GPU-123456', free, pids)
        return launch.admit(slot, gpu, managed or self.idle, task or {}, 20480)[0]

    def test_exclusive_idle_slot_requires_sufficient_free_memory(self):
        self.assertEqual(self.decision(slot=2), 'start')
        self.assertEqual(self.decision(slot=2, free=100), 'blocked')
        self.assertEqual(self.decision(slot=2, pids=(901,)), 'blocked')

    def test_existing_worker_is_not_launched_twice_and_unknown_process_blocks(self):
        self.assertEqual(self.decision(pids=(41,), task={'pid': 41}), 'existing')
        self.assertEqual(self.decision(pids=(41, 777), task={'pid': 41}), 'blocked')
        with patch.object(launch, 'owns', side_effect=lambda pid, group: pid == 55 and group == self.managed['ControlGroup']):
            self.assertEqual(self.decision(slot=3, pids=(55,), managed=self.managed), 'existing')
            self.assertEqual(self.decision(slot=3, pids=(55, 777), managed=self.managed), 'blocked')

    def test_requested_gpus_must_be_inside_inherited_cuda_allocation(self):
        devices = {2: launch.GPU('GPU-abc', 22000, ()), 7: launch.GPU('GPU-def', 22000, ())}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'LERRA_GPU_IDS': ''}), \
                patch.object(launch, 'visible_uuids', return_value={'GPU-def'}):
            self.assertEqual(launch.selected_slots(devices), [7])
            self.assertEqual(launch.selected_slots(devices, 'GPU-def'), [7])
            for selection in ('7,7', '2', 'GPU-abc', '4'):
                with self.assertRaises(ValueError):
                    launch.selected_slots(devices, selection)

    def test_two_launches_submit_each_available_slot_once(self):
        existing = {}
        devices = {slot: launch.GPU(f'GPU-abc{slot}', 22000, (41,) if slot == 0 else ()) for slot in (0, 5)}
        submissions = []
        def submit(slot, gpu):
            submissions.append((slot, gpu.uuid))
            existing[launch.unit_name(slot)] = {'ActiveState': 'activating', 'ControlGroup': f'/owned/{slot}'}
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(launch, 'unit_state', side_effect=lambda name: existing.get(name, self.idle)), \
                patch.object(launch, 'inventory', return_value=devices), \
                patch.object(launch, 'live_task', return_value={}), \
                patch.object(launch, 'submit', side_effect=submit), redirect_stdout(io.StringIO()):
            self.assertEqual(launch.launch(check=True), 1)
            self.assertEqual(submissions, [])
            self.assertEqual(launch.launch(), 1)
            self.assertEqual(launch.launch(), 1)
        self.assertEqual(submissions, [(5, 'GPU-abc5')])

    def test_foreground_launch_without_user_systemd(self):
        devices = {9: launch.GPU('GPU-abc', 16000, ())}
        children = []
        with patch.dict(os.environ, {'LERRA_MIN_FREE_MIB': '12000'}, clear=True), \
                patch.object(launch, 'inventory', return_value=devices), \
                patch.object(launch, 'live_task', return_value={}), \
                patch.object(launch, 'unit_state', side_effect=AssertionError('systemd accessed')), \
                patch.object(launch.subprocess, 'Popen') as popen, redirect_stdout(io.StringIO()):
            self.assertEqual(launch.launch(check=True, foreground=True), 0)
            popen.assert_not_called()
            self.assertEqual(launch.launch(foreground=True, children=children), 0)
        self.assertEqual(children, [popen.return_value])
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-abc')
        self.assertEqual(popen.call_args.args[0][-4:], ['--output', str(launch.OUTPUT), '--slot', '9'])


def _tiny_job(root, design, plan, condition):
    import torch
    from experiment_2.code import run_strength_selection as study
    from experiment_2.code.test_strength_campaign import TinyBackend, bind_watermark, prepare_assets

    class Backend(TinyBackend):
        def quality(self, directory, *, progress_dir=None):
            metrics = super().quality(directory)
            if condition.endswith('_cifar100'):
                return metrics
            return {'pairs': metrics['examples'], 'mean_mse': 0.0, 'mean_ssim': 1.0}

    return study.ConditionRun(root, design, plan, condition, torch.device('cpu'), 1,
                              Backend(), prepare_assets, bind_watermark)


def _work_counts(job):
    counts = job.backend.calls.copy()
    for step, marked in job.backend.steps:
        counts[f'{"marked" if marked else "clean"}_step_{step}'] += 1
    return dict(counts)


def _drain_tiny_campaign(root, ops, design, plan, slot, first_wave):
    from experiment_2.parallel_strength import worker

    with worker.try_lock(ops / f'slot-{slot}.lock', fcntl.LOCK_EX):
        worker.recover_claims(ops, slot)
        _run_tiny_tasks(root, ops, design, plan, slot, first_wave)


def _run_tiny_tasks(root, ops, design, plan, slot, first_wave):
    from experiment_2.parallel_strength import worker
    preferred = plan['conditions'][slot - 1]
    records = []
    while not worker.all_complete(root, plan):
        with worker.claim_work(root, ops, plan, preferred, slot=slot) as task:
            if not records:
                # Require all three workers to hold distinct live leases before computing.
                if task is None:
                    raise AssertionError('expected work for the initial group of three slots')
                first_wave.wait(timeout=60)
            if task:
                condition, method = task
                job = _tiny_job(root, design, plan, condition)
                with redirect_stdout(io.StringIO()):
                    job.ensure_clean()
                    if method is not None:
                        job.run_method(method)
                records.append([condition, method, _work_counts(job)])
                continue
        time.sleep(0.02)
    (ops / f'test-slot-{slot}.json').write_text(json.dumps(records))


@unittest.skipUnless(find_spec('torch'), 'use the experiment PyTorch interpreter for the checkpoint integration test')
class ParallelCleanReuse(unittest.TestCase):
    def test_one_clean_per_condition_for_all_210_candidates_and_worker_restart(self):
        from experiment_2.code import run_strength_selection as study
        from experiment_2.parallel_strength import worker

        design, plan = study.build_plan()
        conditions = plan['conditions']
        # Exercise the registered 7 x 6 x 5 grid using two tiny CPU updates per learned arm.
        plan['study_id'] = 'parallel-clean-reuse-test'
        plan['terminal_updates'] = dict.fromkeys(conditions, 2)
        plan['validation_size'] = dict.fromkeys(conditions, 4)
        design['data']['source_CIFAR100']['counts']['validation'] = 4
        design['generation_evaluation']['pairs_per_split'] = 4
        with TemporaryDirectory() as directory:
            root, ops = Path(directory) / 'campaign', Path(directory) / 'operations'
            first = conditions[0]
            records = []
            with study.lock(root / first / '.worker.lock'):
                legacy = _tiny_job(root, design, plan, first)
                legacy.ensure_clean()
                legacy.run_method('lerra')
                records.append([first, 'lerra', _work_counts(legacy)])
                original = dict(legacy.clean)

            # Interrupt the first learned candidate just after saving its first update.
            with self.assertRaises(InterruptedError):
                with worker.claim_work(root, ops, plan, first, slot=1) as task:
                    condition, method = task
                    self.assertEqual(condition, first)
                    self.assertNotIn(method, plan['direct_methods'])
                    partial = _tiny_job(root, design, plan, condition)
                    partial.ensure_clean()
                    train_step = partial.backend.train_step

                    def interrupt(step, watermark_loss):
                        if step == 1:
                            raise InterruptedError
                        return train_step(step, watermark_loss)

                    with patch.object(partial.backend, 'train_step', side_effect=interrupt):
                        partial.run_method(method)
            records.append([condition, method, _work_counts(partial)])
            checkpoint = root / first / method / 'candidate_1/progress.pt'
            self.assertEqual(study.load_tensor(checkpoint)['step'], 1)

            ctx = multiprocessing.get_context('spawn')
            first_wave = ctx.Barrier(3)
            processes = [ctx.Process(target=_drain_tiny_campaign,
                                     args=(root, ops, design, plan, slot, first_wave)) for slot in (1, 2, 3)]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 120
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                        process.join()

            initial = []
            for slot in (1, 2, 3):
                run = json.loads((ops / f'test-slot-{slot}.json').read_text())
                initial.append(run[0][0])
                records.extend(run)
            self.assertEqual(set(initial), set(conditions[:3]))
            self.assertEqual(study.read_json(root / first / 'clean/ready.json')['checkpoint'], original)
            self.assertFalse(checkpoint.exists())
            self.assertEqual(len(list(root.glob('*/clean/terminal.pt'))), 7)
            for condition in conditions:
                counts = Counter()
                for name, _, work in records:
                    if name == condition:
                        counts.update(work)
                self.assertEqual([counts['clean_step_0'], counts['clean_step_1'], counts['clean_quality']], [1, 1, 1])
                self.assertEqual([counts['marked_step_0'], counts['marked_step_1'], counts['marked_quality']], [20, 20, 30])
                clean = study.binding(root, root / condition / 'clean/terminal.pt')
                marked_paths = set()
                for method in plan['methods']:
                    self.assertEqual(counts[method + '_assets'], 1)
                    results = study.group_results(root, plan, condition, method, verify=True)
                    self.assertEqual(len(results), 5)
                    for result in results:
                        self.assertEqual(result['C_T'], clean)
                        self.assertEqual(result['decoder_assets'], results[0]['decoder_assets'])
                        marked_paths.add(result['W_T']['path'])
                self.assertEqual(len(marked_paths), 30)
            self.assertTrue(worker.all_complete(root, plan))
            report = study.update_report(root, plan, verify=True)
            self.assertEqual(report['measured_candidates'], 210)
            self.assertEqual(report['counts']['selected'], 42)
            with worker.claim_work(root, ops, plan, first, slot=1) as no_work:
                self.assertIsNone(no_work)


if __name__ == '__main__':
    unittest.main()

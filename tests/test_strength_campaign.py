"""CPU regression tests for fixed-endpoint training, reuse, readout and ranking."""
import csv
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment_2.code import run_strength_selection as campaign
from runtime.comparison.protocol_v2.methods import Recovery
from tests.tiny_backend import TinyBackend, TinyWatermark, bind_watermark, prepare_assets


CONDITION = 'tiny_cifar100'
STRENGTHS = ['0.25', '0.5', '1', '2', '4']
TARGET = [0, 1] * 128
TOTAL = 3
SEED = 100


def new_run(directory):
    design = {'payload': {'physical_payloads_by_recipient': {'292': ''.join(map(str, TARGET))}},
              'data': {'source_CIFAR100': {'counts': {'validation': 4}}}}
    plan = {'study_id': 'test-study', 'recipient_id': 292, 'terminal_updates': {CONDITION: TOTAL},
            'payload': ''.join(map(str, TARGET)), 'validation_size': {CONDITION: 4},
            'recipes': {'shared_arm_seed': SEED}, 'direct_methods': ['lerra'],
            'conditions': [CONDITION], 'methods': ['lerra', 'uchida'],
            'strengths': {CONDITION: {'lerra': STRENGTHS, 'uchida': STRENGTHS}}}
    return campaign.ConditionRun(directory, design, plan, CONDITION, torch.device('cpu'), 1,
                                 TinyBackend(), prepare_assets, bind_watermark)


def candidate_dir(run, method, index):
    return run.base / method / f'candidate_{index}'


def results(run, method):
    return [campaign.read_json(candidate_dir(run, method, i) / 'result.json') for i in range(1, 6)]


def weights(run, link):
    return campaign.load_tensor(campaign.check_bound(run.root, link))['model']['weight']


class StrengthCampaignTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_registered_paths_relocate_without_changing_plan_or_rewriting_manifest(self):
        # A target mount inside /data would expose accidental double translation.
        from experiment_2.code.input_paths import read_registered_json
        source = self.root / 'manifest.json'
        original = b'{"files":[{"path":"/data/images/p.png","id":"p.png"}]}'
        source.write_bytes(original)
        with patch.dict(os.environ, {'LERRA_DATA_ROOT': '/data'}):
            canonical_design, canonical_plan = campaign.build_plan()
        with patch.dict(os.environ, {'LERRA_DATA_ROOT': '/data/server-two'}):
            mounted_design, mounted_plan = campaign.build_plan()
            self.assertEqual(read_registered_json(source)['files'], [
                {'path': '/data/server-two/images/p.png', 'id': 'p.png'}])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(mounted_plan, canonical_plan)
        def first_checkpoint(design):
            return next(iter(design['embedding']['fixed_source_checkpoints'].values()))['checkpoint']
        before = first_checkpoint(canonical_design)
        self.assertEqual(first_checkpoint(mounted_design), {
            **before, 'path': str(Path('/data/server-two') / Path(before['path']).relative_to('/data'))})

    def test_direct_and_learned_share_control_and_reuse_finished_work(self):
        run = new_run(self.root)
        run.ensure_clean()
        run.run_method('uchida')
        run.run_method('lerra')
        backend = run.backend
        clean = weights(run, run.clean)
        self.assertFalse(torch.equal(clean, backend.source))
        self.assertEqual(backend.steps, [(i, False) for i in range(TOTAL)]
                         + [(i, True) for _ in STRENGTHS for i in range(TOTAL)])
        self.assertEqual(backend.calls, {'clean_quality': 1, 'marked_quality': 10,
                                        'uchida_assets': 1, 'lerra_assets': 1,
                                        'loss': 15, 'direct_insert': 5,
                                        'uchida_recover': 5, 'lerra_recover': 5})

        for method in ('uchida', 'lerra'):
            records = results(run, method)
            self.assertEqual([r['strength'] for r in records], STRENGTHS)
            self.assertEqual([r['candidate'] for r in records], list(range(1, 6)))
            for record in records:
                self.assertEqual(record['C_T'], run.clean)
                self.assertEqual(record['source_updates'], TOTAL)
                self.assertEqual(record['embedding_updates'], 0 if method == 'lerra' else TOTAL)
                self.assertEqual(record['quality']['accuracy_drop_pp'], 25)
                self.assertEqual(record['decoder_assets'], records[0]['decoder_assets'])
                strength = float(record['strength'])
                if method == 'lerra':
                    expected = clean.clone()
                    expected[0, 0] += strength
                else:
                    reference = TinyBackend()
                    campaign.seed_arm(SEED, run.device)
                    for step in range(TOTAL):
                        reference.train_step(step, lambda: strength * reference.model.weight.square().sum())
                    expected = reference.model.weight
                self.assertTrue(torch.equal(weights(run, record['W_T']), expected))
            reference_weights = campaign.load_tensor(campaign.check_bound(run.root, records[0]['decoder_assets']))
            self.assertTrue(torch.equal(reference_weights['reference_weight'],
                                        clean if method == 'lerra' else backend.source))

        restarted = new_run(self.root)
        restarted.ensure_clean()
        for method in ('lerra', 'uchida'):
            restarted.run_method(method)
        self.assertEqual(restarted.backend.steps, [])
        self.assertEqual(restarted.backend.calls, {})

    def test_partial_training_resumes_at_committed_step(self):
        whole = new_run(self.root / 'whole')
        whole.ensure_clean()
        whole.run_method('uchida')
        interrupted = new_run(self.root / 'resumed')
        interrupted.ensure_clean()
        backend = interrupted.backend
        original_step = backend.train_step

        def interrupt_second_candidate(step, loss):
            if backend.calls['loss'] == TOTAL + 1:
                raise InterruptedError
            original_step(step, loss)

        with patch.object(backend, 'train_step', side_effect=interrupt_second_candidate):
            with self.assertRaises(InterruptedError):
                interrupted.run_method('uchida')
        self.assertTrue((candidate_dir(interrupted, 'uchida', 1) / 'result.json').is_file())
        partial = campaign.load_tensor(candidate_dir(interrupted, 'uchida', 2) / 'progress.pt')
        self.assertEqual(partial['step'], 1)
        resumed = new_run(interrupted.root)
        resumed.ensure_clean()
        resumed.run_method('uchida')
        self.assertEqual(resumed.backend.steps, [(1, True), (2, True)]
                         + [(i, True) for _ in range(3) for i in range(TOTAL)])
        self.assertEqual(resumed.backend.calls['uchida_assets'], 0)
        self.assertEqual(resumed.backend.calls['marked_quality'], 4)
        for reference, record in zip(results(whole, 'uchida'), results(resumed, 'uchida')):
            self.assertEqual(record['readout'], reference['readout'])
            self.assertTrue(torch.equal(weights(resumed, record['W_T']), weights(whole, reference['W_T'])))
        self.assertFalse((candidate_dir(resumed, 'uchida', 2) / 'progress.pt').exists())

    def test_interruption_after_readout_reuses_endpoint_and_decoder_output(self):
        run = new_run(self.root)
        run.ensure_clean()
        with patch.object(run.backend, 'quality', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                run.run_method('lerra')
        first = candidate_dir(run, 'lerra', 1)
        self.assertTrue((first / 'readout.json').exists())
        self.assertFalse((first / 'result.json').exists())
        restored = new_run(self.root)
        restored.ensure_clean()
        restored.run_method('lerra')
        self.assertEqual(restored.backend.steps, [])
        self.assertEqual(restored.backend.calls, {'direct_insert': 4, 'lerra_recover': 4, 'marked_quality': 5})

    @torch.no_grad()
    def test_nonfinite_classifier_outputs_count_rows_on_both_sides(self):
        from experiment_2.code.classification_backend import ClassificationBackend

        backend = ClassificationBackend.__new__(ClassificationBackend)
        backend.condition, backend.device = CONDITION, torch.device('cpu')
        backend.recipe, backend.split_sha256 = {'validation_batch': 2}, 'test-validation'
        backend.validation = torch.utils.data.TensorDataset(torch.arange(5), torch.zeros(5, dtype=torch.int64))
        backend.model = torch.nn.Embedding(5, 100)
        backend.model.weight.zero_()
        clean_dir, invalid_dir = self.root / 'finite', self.root / 'undefined'
        backend.prepare_quality(clean_dir)
        backend.model.weight[0, :2] = torch.nan
        backend.model.weight[4, 8] = torch.inf
        marked_invalid = backend.quality(clean_dir)
        self.assertEqual(marked_invalid['examples'], 5)
        self.assertEqual(marked_invalid['quality_status'], 'undefined')
        self.assertEqual((marked_invalid['clean_invalid_examples'], marked_invalid['marked_invalid_examples']), (0, 2))
        self.assertEqual(marked_invalid['clean_correct'], 5)
        self.assertEqual(marked_invalid['clean_top1_pct'], 100)
        self.assertIsNone(marked_invalid['marked_correct'])
        self.assertIsNone(marked_invalid['marked_top1_pct'])
        self.assertIsNone(marked_invalid['accuracy_drop_pp'])
        backend.prepare_quality(invalid_dir)
        backend.model.weight.zero_()
        clean_invalid = backend.quality(invalid_dir)
        self.assertEqual((clean_invalid['clean_invalid_examples'], clean_invalid['marked_invalid_examples']), (2, 0))
        self.assertIsNone(clean_invalid['clean_correct'])
        self.assertEqual(clean_invalid['marked_correct'], 5)
        self.assertEqual(backend.quality(clean_dir)['quality_status'], 'ok')

    def test_direct_native_readout_is_used_once_per_candidate(self):
        run = new_run(self.root)
        run.ensure_clean()
        insert = TinyWatermark.insert

        def insert_and_read(adapter, strength):
            insert(adapter, strength)
            return adapter.recover()

        with patch.object(TinyWatermark, 'insert', insert_and_read):
            run.run_method('lerra')
        self.assertEqual(run.backend.calls['lerra_recover'], 5)
        self.assertEqual(run.backend.calls['direct_insert'], 5)
        self.assertTrue(all(r['readout']['valid'] for r in results(run, 'lerra')))

    def test_rgb_validation_records_undefined_pairs_and_resumes(self):
        from experiment_2.code.generation_backend import GenerationBackend

        backend = GenerationBackend.__new__(GenerationBackend)
        backend.condition, backend.device = 'toy_ffhq', torch.device('cpu')
        backend.model = torch.nn.Linear(1, 1)
        backend.quality_config = {'pairs_per_split': 3, 'seed_ranges_inclusive': {'validation': [2, 4]}}
        backend.native = SimpleNamespace(kind='styleswin_256', spec={'checkpoint': {'sha256': 'toy'}})
        clean, marked = self.root / 'clean', self.root / 'marked'
        rgb = torch.zeros(1, 3, 256, 256)
        nonfinite = rgb.clone()
        nonfinite[0, 0, 0, 0] = float('nan')
        backend.native.sample_rgb = lambda seed: nonfinite if seed == 2 else rgb
        backend.prepare_quality(clean)
        backend.native.sample_rgb = lambda seed: nonfinite if seed == 3 else rgb
        output = backend.quality(clean, progress_dir=marked)
        self.assertEqual(output, campaign.validate_quality(output, False, 3))
        self.assertEqual(output['quality_status'], 'undefined')
        self.assertEqual(output['invalid_pairs'], 2)
        self.assertIsNone(output['mean_ssim'])
        self.assertIsNone(output['mean_mse'])
        with patch.object(backend.native, 'sample_rgb', side_effect=AssertionError('unexpected repeated generation')):
            self.assertEqual(backend.quality(clean, progress_dir=marked), output)

    def test_rank_only_valid_complete_vectors_by_error_count_then_strength(self):
        def candidate(strength, bit_errors):
            bits = np.array(TARGET)
            bits[:bit_errors] ^= 1
            return {'strength': strength, 'readout': campaign.scored_readout(Recovery(bits, 'ok'), TARGET)}

        records = [candidate(s, n) for s, n in zip(STRENGTHS, (4, 1, 3, 1, 2))]
        winner = campaign.selection(records[::-1])
        self.assertEqual(winner['strength'], '2')
        self.assertEqual(winner['readout']['ber'], 1 / 256)
        self.assertEqual(winner['readout']['bit_errors'], 1)
        with self.assertRaises(ValueError):
            campaign.selection(records[:-1])
        records[0] = candidate(STRENGTHS[0], 0)
        self.assertEqual(campaign.selection(records)['strength'], '0.25')
        invalid = [Recovery(None, 'invalid'), Recovery(np.array(TARGET), 'erasure'),
                   Recovery(np.ones(255), 'ok'), Recovery(np.ones(257), 'ok'),
                   Recovery(np.full(256, np.nan), 'ok')]
        for record, recovery in zip(records, invalid):
            record['readout'] = campaign.scored_readout(recovery, TARGET)
            self.assertFalse(record['readout']['valid'])
            self.assertIsNone(record['readout']['ber'])
        self.assertEqual(records[1]['readout']['diagnostic_bit_errors'], 0)
        self.assertIsNone(records[1]['readout']['bit_errors'])
        self.assertIsNone(campaign.selection(records))

    def test_report_distinguishes_incomplete_failed_and_selected_groups(self):
        run = new_run(self.root)
        run.ensure_clean()
        valid = Recovery(np.array(TARGET), 'ok')
        invalid = Recovery(None, 'erasure')
        finite_quality = run.backend.quality(run.cache)
        undefined_quality = {**finite_quality, 'quality_status': 'undefined',
                             'clean_invalid_examples': 0, 'marked_invalid_examples': 1,
                             'marked_correct': None, 'marked_top1_pct': None, 'accuracy_drop_pp': None}
        with patch.object(run.backend, 'quality', side_effect=[finite_quality] * 3 + [undefined_quality]), \
             patch.object(TinyWatermark, 'recover', side_effect=[valid] * 4 + [InterruptedError]):
            with self.assertRaises(InterruptedError):
                run.run_method('lerra')
        partial = campaign.update_report(self.root, run.plan, verify=True)
        self.assertEqual(partial['measured_candidates'], 4)
        self.assertEqual(partial['counts'], {'selected': 0, 'pending': 2, 'selection_failure': 0})
        self.assertFalse(any(g['W_T'] for g in partial['groups']))
        with patch.object(TinyWatermark, 'recover', side_effect=[invalid] * 6):
            run.run_method('lerra')
            run.run_method('uchida')
        report = campaign.update_report(self.root, run.plan, verify=True)
        self.assertEqual(report['counts'], {'selected': 1, 'pending': 0, 'selection_failure': 1})
        winner, failed = report['groups']
        self.assertEqual(winner['selected_strength'], '2')
        self.assertEqual(winner['W_T'], results(run, 'lerra')[3]['W_T'])
        self.assertEqual(results(run, 'lerra')[3]['quality']['quality_status'], 'undefined')
        self.assertEqual(winner['ber'], 0.0)
        self.assertIsNone(failed['W_T'])
        self.assertIsNone(failed['ber'])
        self.assertEqual(report, campaign.read_json(self.root / 'selection.json'))
        with (self.root / 'selected.csv').open() as f:
            selections = list(csv.DictReader(f))
        with (self.root / 'candidates.csv').open() as f:
            candidates = list(csv.DictReader(f))
        self.assertEqual(len(selections), 2)
        self.assertEqual(len(candidates), 10)
        self.assertEqual([c['strength'] for c in candidates if c['selected'] == 'True'], ['2'])


if __name__ == '__main__':
    unittest.main()

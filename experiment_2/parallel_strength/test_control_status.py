"""Verify that different methods refer to the same published clean checkpoint."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiment_2.code import run_strength_selection as study
from experiment_2.parallel_strength.status import inspect_controls


class ControlStatusTests(unittest.TestCase):
    def test_cross_method_receipts_and_modified_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = 'test_cifar100'
            methods = ['uchida', 'lerra']
            plan = {'conditions': [condition], 'methods': methods, 'terminal_updates': {condition: 2}}
            study.write_json(root / 'plan.json', {**plan, 'study_id': study.identity(plan)})
            base = root / condition
            terminal = base / 'clean/terminal.pt'
            terminal.parent.mkdir(parents=True)
            terminal.write_bytes(b'clean')
            link = study.binding(root, terminal)
            self.assertEqual(inspect_controls(root, [])[0]['state'], 'validation_pending')
            study.write_json(base / 'clean/ready.json', {'checkpoint': link, 'step': 2})
            for method in methods:
                path = base / method / 'candidate_1/result.json'
                study.write_json(path, {'C_T': link})
            result, = inspect_controls(root, [], verify=True)
            self.assertEqual((result['state'], result['candidate_references'], result['sha256']), ('ready', 2, link['sha256']))
            study.write_json(path, {'C_T': {**link, 'sha256': 'wrong'}})
            with self.assertRaisesRegex(ValueError, 'inconsistent clean endpoint'):
                inspect_controls(root, [])
            study.write_json(path, {'C_T': link})
            terminal.write_bytes(b'other')
            with self.assertRaises(ValueError):
                inspect_controls(root, [], verify=True)

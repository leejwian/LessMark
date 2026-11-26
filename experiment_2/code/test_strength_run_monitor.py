"""Mock the mail transport while exercising real worker lifetimes and exit statuses."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_2.code import strength_run_monitor as monitor


def _exercise(args):
    monitor.set_phase('strength_endpoint', condition='resnet18_cifar100', method='lerra', candidate=3)
    if args.action == 'failure':
        raise RuntimeError('local-only diagnostic fixture')
    if args.action == 'kill':
        os.kill(os.getpid(), signal.SIGKILL)
    if args.action in ('interrupt', 'terminate'):
        Path(args.pid_file).write_text(str(os.getpid()))
        os.kill(os.getppid(), signal.SIGINT if args.action == 'interrupt' else signal.SIGTERM)
        time.sleep(30)
    if args.action == 'exit':
        raise SystemExit(17)


class RunMonitorTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.output = folder / 'campaign'
        self.pid_file = folder / 'pid'
        stack.enter_context(patch.dict(os.environ, {'LERRA_NOTIFICATION_MODE': 'smtp', 'LERRA_MAIL_ACCOUNT': 'test@example.org'}))
        stack.enter_context(patch.object(monitor.notifications, '_password', return_value='smtp-test-value'))
        factory = stack.enter_context(patch.object(monitor.notifications.smtplib, 'SMTP_SSL'))
        self.smtp = factory.return_value.__enter__.return_value
        self.smtp.noop.return_value = 250, b'OK'
        self.smtp.send_message.return_value = {}
        self.scheduler = stack.enter_context(patch.object(monitor.subprocess, 'run'))
        self.scheduler.return_value.returncode = 0
        stack.enter_context(patch.object(monitor.notifications, 'log_diagnostic'))

    def run_case(self, action):
        args = SimpleNamespace(output=self.output, action=action, pid_file=self.pid_file)
        return monitor.supervise(args, _exercise)

    def receipts(self):
        directory = self.output.with_name('campaign.alerts') / 'notifications'
        return [json.loads(path.read_text()) for path in directory.glob('*.json')
                if not path.name.endswith('.delivery_error.json')]

    def test_default_local_run_records_failures_without_smtp(self):
        os.environ.pop('LERRA_NOTIFICATION_MODE')
        os.environ.pop('LERRA_MAIL_ACCOUNT')
        self.assertEqual(self.run_case('success'), 0)
        self.assertEqual(self.run_case('failure'), 1)
        receipt, = self.receipts()
        self.assertEqual(receipt['mode'], 'local')
        self.assertFalse(receipt['pending'])
        self.smtp.login.assert_not_called()
        self.smtp.send_message.assert_not_called()
        self.scheduler.assert_not_called()

    def test_success_authenticates_and_does_not_send_mail(self):
        self.assertEqual(self.run_case('success'), 0)
        self.smtp.noop.assert_called_once()
        self.smtp.send_message.assert_not_called()
        self.scheduler.assert_not_called()
        self.assertEqual(self.receipts(), [])

    def test_worker_exception_and_signal_are_different_attempts(self):
        self.assertEqual(self.run_case('failure'), 1)
        self.assertEqual(self.run_case('kill'), 137)
        records = self.receipts()
        self.assertEqual(self.smtp.send_message.call_count, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(len({r['event']['attempt'] for r in records}), 2)
        events = {r['event']['exit_code']: r['event'] for r in records}
        self.assertEqual(events[1]['error_type'], 'RuntimeError')
        self.assertEqual(events[137]['signal'], 'SIGKILL')
        for event in events.values():
            self.assertEqual(event['condition'], 'resnet18_cifar100')
            self.assertEqual(event['method'], 'lerra')
            self.assertEqual(event['candidate'], 3)
            self.assertNotIn('local-only diagnostic fixture', str(event))
        self.assertFalse(self.output.exists())

    def test_supervisor_ctrl_c_and_sigterm_stop_the_worker(self):
        self.assertEqual(self.run_case('interrupt'), 130)
        self.smtp.send_message.assert_not_called()
        self.assertFalse(Path('/proc', self.pid_file.read_text()).exists())
        self.assertEqual(self.run_case('terminate'), 143)
        self.assertEqual(self.smtp.send_message.call_count, 1)
        self.assertEqual(self.receipts()[0]['event']['signal'], 'SIGTERM')
        self.assertFalse(Path('/proc', self.pid_file.read_text()).exists())

    def test_preflight_failure_reports_without_starting_worker(self):
        self.smtp.noop.return_value = 421, b'unavailable'
        self.assertEqual(self.run_case('terminate'), 1)
        self.assertFalse(self.pid_file.exists())
        record, = self.receipts()
        self.assertEqual(record['event']['stage'], 'smtp_preflight')
        self.assertTrue(record['sent'])

    def test_spawn_failure_reports_before_training(self):
        args = SimpleNamespace(output=self.output)
        self.assertEqual(monitor.supervise(args, lambda _: None), 1)
        record, = self.receipts()
        self.assertEqual(record['event']['stage'], 'worker_start')
        self.assertTrue(record['sent'])

    def test_failed_delivery_retries_at_next_start_without_changing_exit_code(self):
        self.smtp.send_message.side_effect = OSError('test transport failure')
        self.assertEqual(self.run_case('exit'), 17)
        self.assertEqual(self.smtp.send_message.call_count, monitor.notifications.DELIVERY_ATTEMPTS)
        record, = self.receipts()
        self.assertTrue(record['pending'])
        self.scheduler.assert_called_once()
        command = self.scheduler.call_args.args[0]
        self.assertIn('--on-active=5min', command)
        self.assertIn('--property=RestartForceExitStatus=20', command)
        self.assertIn('--property=RestartSec=5min', command)
        self.assertEqual(command[-2:], ['--retry-pending', str(self.output.with_name('campaign.alerts'))])
        self.assertNotIn('smtp-test-value', str(command))
        first_id = self.smtp.send_message.call_args.args[0]['Message-ID']
        self.smtp.send_message.side_effect = None
        self.assertEqual(self.run_case('success'), 0)
        self.assertEqual(self.smtp.send_message.call_args.args[0]['Message-ID'], first_id)
        self.assertTrue(self.receipts()[0]['sent'])

    def test_unavailable_manager_preserves_receipt_and_worker_failure_code(self):
        self.smtp.send_message.side_effect = OSError('temporary failure')
        self.scheduler.side_effect = FileNotFoundError('no systemd-run')
        self.assertEqual(self.run_case('exit'), 17)
        self.assertTrue(self.receipts()[0]['pending'])

    def test_retry_service_restarts_only_for_deliverable_pending_receipts(self):
        self.smtp.send_message.side_effect = OSError('temporary failure')
        self.assertEqual(self.run_case('exit'), 17)
        outbox = self.output.with_name('campaign.alerts')
        self.assertEqual(monitor._retry_batch(outbox), 20)
        self.smtp.send_message.side_effect = None
        self.assertEqual(monitor._retry_batch(outbox), 0)
        (outbox / 'notifications' / 'broken.json').write_text('broken')
        self.assertEqual(monitor._retry_batch(outbox), os.EX_DATAERR)


if __name__ == '__main__':
    unittest.main()

import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dashboard.mining_jobs import MiningJobConfig, MiningJobManager


SNAPSHOT_ID = "a" * 64


class DashboardMiningJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = MiningJobManager(
            project_root=self.root,
            python_executable="/test/venv/bin/python",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_config_rejects_unsafe_or_unbounded_values(self):
        with self.assertRaises(ValueError):
            MiningJobConfig(snapshot_id="../../etc/passwd")
        with self.assertRaises(ValueError):
            MiningJobConfig(snapshot_id=SNAPSHOT_ID, seeds=(1, 1))
        with self.assertRaises(ValueError):
            MiningJobConfig(snapshot_id=SNAPSHOT_ID, steps=100_001)
        with self.assertRaises(ValueError):
            MiningJobConfig(snapshot_id=SNAPSHOT_ID, batch_size=65_537)

    @patch("dashboard.mining_jobs.subprocess.Popen")
    def test_start_uses_argument_list_and_blocks_a_second_active_job(self, popen):
        popen.return_value = Mock(pid=4242)
        with patch.object(self.manager, "_process_start_ticks", return_value="101"):
            first = self.manager.start_job(MiningJobConfig(snapshot_id=SNAPSHOT_ID))

        command = popen.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[:4], [
            "/test/venv/bin/python",
            "-u",
            "-m",
            "dashboard.mining_job_runner",
        ])
        self.assertFalse(popen.call_args.kwargs.get("shell", False))
        self.assertTrue(first["output_dir"].startswith(str(self.root / "runs" / "binance")))

        with patch.object(self.manager, "_is_owned_process", return_value=True):
            with self.assertRaises(RuntimeError):
                self.manager.start_job(MiningJobConfig(snapshot_id=SNAPSHOT_ID, seeds=(6, 7)))

    def test_reconcile_uses_completed_batch_artifact(self):
        job_id = "20260826T000000Z-web-abc123"
        output_dir = self.root / "runs" / "binance" / job_id
        output_dir.mkdir(parents=True)
        (output_dir / "batch_report.json").write_text(json.dumps({"status": "complete"}))
        state = {
            "job_id": job_id,
            "status": "running",
            "pid": 999_999,
            "process_start_ticks": "1",
            "output_dir": str(output_dir),
            "log_path": str(self.manager.job_root / f"{job_id}.log"),
        }
        self.manager._write_json(self.manager._state_path(job_id), state)
        self.assertEqual(self.manager.get_job(job_id)["status"], "complete")

    @patch("dashboard.mining_jobs.os.killpg")
    def test_stop_signals_only_a_verified_owned_process_group(self, killpg):
        job_id = "20260826T000000Z-web-def456"
        state = {
            "job_id": job_id,
            "status": "running",
            "pid": 5151,
            "process_start_ticks": "2",
            "output_dir": str(self.root / "runs" / "binance" / job_id),
            "log_path": str(self.manager.job_root / f"{job_id}.log"),
        }
        self.manager._write_json(self.manager._state_path(job_id), state)
        with patch.object(self.manager, "_is_owned_process", return_value=True):
            stopped = self.manager.stop_job(job_id)
        self.assertEqual(stopped["status"], "stopping")
        killpg.assert_called_once_with(5151, signal.SIGTERM)

        state["status"] = "running"
        self.manager._write_json(self.manager._state_path(job_id), state)
        with patch.object(self.manager, "_is_owned_process", return_value=False):
            with self.assertRaises(RuntimeError):
                self.manager.stop_job(job_id)
        killpg.assert_called_once()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from march_madness_2026.v10.vast_run import (
    VastRunPaths,
    build_runner_script,
    build_status_payload,
    start_run,
    write_status_files,
)


class VastRunWrapperTests(unittest.TestCase):
    def test_build_runner_script_records_pid_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = VastRunPaths.from_run_dir(Path(tmpdir))
            script = build_runner_script(paths, cwd=Path("/repo"), command=["python", "scripts/autobracket_v10.py", "--resume"])
            self.assertIn("tee -a", script)
            self.assertIn(str(paths.pid_file), script)
            self.assertIn(str(paths.exit_code_file), script)
            self.assertIn('if wait "$child"; then', script)
            self.assertIn("python scripts/autobracket_v10.py --resume", script)

    def test_build_status_payload_collects_progress_and_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            paths = VastRunPaths.from_run_dir(run_dir)
            paths.launch_file.write_text(
                json.dumps(
                    {
                        "session_name": "mm-v10",
                        "label": "test-run",
                        "cwd": "/repo",
                        "command": ["python", "scripts/autobracket_v10.py"],
                        "command_shell": "python scripts/autobracket_v10.py",
                        "hostname": "worker-1",
                    }
                ),
                encoding="utf-8",
            )
            paths.pid_file.write_text(str(os.getpid()), encoding="utf-8")
            paths.log_file.write_text("line one\nline two\n", encoding="utf-8")
            (run_dir / "best_v10_params.json").write_text(
                json.dumps({"score": 0.42, "metrics": {"first_place_equity": 0.11}}),
                encoding="utf-8",
            )
            (run_dir / "experiments_v10.tsv").write_text(
                "round\tscore\n1\t0.42\n",
                encoding="utf-8",
            )

            with (
                mock.patch("march_madness_2026.v10.vast_run._tmux_has_session", return_value=True),
                mock.patch(
                    "march_madness_2026.v10.vast_run._gpu_snapshot",
                    return_value=[{"name": "RTX", "utilization_gpu_pct": 12.0}],
                ),
            ):
                payload = build_status_payload(paths, refresh_time="2026-03-18T12:00:00+00:00")

            self.assertEqual(payload["state"], "running")
            self.assertTrue(payload["process_alive"])
            self.assertEqual(payload["last_log_line"], "line two")
            self.assertEqual(payload["artifacts"]["best_score"], 0.42)
            self.assertEqual(payload["artifacts"]["experiment_rows"], 1)
            self.assertEqual(payload["gpu"][0]["name"], "RTX")

            write_status_files(paths, payload)
            heartbeat = json.loads(paths.heartbeat_file.read_text(encoding="utf-8"))
            self.assertEqual(heartbeat["state"], "running")
            self.assertEqual(heartbeat["best_score"], 0.42)

    def test_start_run_writes_launch_metadata_and_invokes_tmux(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "worker"
            with (
                mock.patch("march_madness_2026.v10.vast_run._tmux_available", return_value=True),
                mock.patch("march_madness_2026.v10.vast_run._tmux_has_session", return_value=False),
                mock.patch("march_madness_2026.v10.vast_run._gpu_snapshot", return_value=[]),
                mock.patch("subprocess.run") as run_mock,
                mock.patch("subprocess.Popen") as popen_mock,
            ):
                paths = start_run(
                    run_dir=run_dir,
                    session_name="mm-v10-test",
                    command=["python", "scripts/autobracket_v10.py", "--resume"],
                    cwd=Path("/repo"),
                    interval_s=5,
                    label="worker-a",
                )

            self.assertTrue(paths.launch_file.exists())
            launch = json.loads(paths.launch_file.read_text(encoding="utf-8"))
            self.assertEqual(launch["session_name"], "mm-v10-test")
            self.assertEqual(launch["label"], "worker-a")
            self.assertEqual(launch["command"], ["python", "scripts/autobracket_v10.py", "--resume"])
            self.assertTrue(paths.runner_script.exists())
            run_mock.assert_called_once()
            tmux_args = run_mock.call_args.args[0]
            self.assertEqual(tmux_args[:4], ["tmux", "new-session", "-d", "-s"])
            popen_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

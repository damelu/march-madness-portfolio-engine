from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "autobracket_v10_worker"
WRAPPER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VastRunPaths:
    run_dir: Path
    log_file: Path
    status_file: Path
    heartbeat_file: Path
    launch_file: Path
    pid_file: Path
    exit_code_file: Path
    runner_script: Path
    shell_pid_file: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "VastRunPaths":
        return cls(
            run_dir=run_dir,
            log_file=run_dir / "run.log",
            status_file=run_dir / "status.json",
            heartbeat_file=run_dir / "heartbeat.json",
            launch_file=run_dir / "launch.json",
            pid_file=run_dir / "run.pid",
            exit_code_file=run_dir / "exit_code.txt",
            runner_script=run_dir / "runner.sh",
            shell_pid_file=run_dir / "runner_shell.pid",
        )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_int(path: Path) -> int | None:
    try:
        return int(_read_text(path).strip())
    except (FileNotFoundError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(_read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _tmux_has_session(session_name: str) -> bool:
    if not _tmux_available():
        return False
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _read_log_tail(log_file: Path, max_bytes: int = 4096) -> str:
    if not log_file.exists():
        return ""
    with log_file.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0))
        chunk = handle.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _gpu_snapshot() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    snapshots: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, util, used, total = parts
        try:
            snapshots.append(
                {
                    "name": name,
                    "utilization_gpu_pct": float(util),
                    "memory_used_mib": float(used),
                    "memory_total_mib": float(total),
                }
            )
        except ValueError:
            continue
    return snapshots


def _collect_optimizer_progress(run_dir: Path) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    best_candidates = [
        run_dir / "best_v10_params.json",
        run_dir / "best_params.json",
    ]
    for path in best_candidates:
        payload = _read_json(path)
        if payload:
            progress["best_params_file"] = str(path)
            progress["best_score"] = payload.get("score")
            progress["best_metrics"] = payload.get("metrics")
            break

    experiment_candidates = [
        run_dir / "experiments_v10.tsv",
        run_dir / "experiments_parallel.tsv",
        run_dir / "experiments_gpu.tsv",
    ]
    for path in experiment_candidates:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        progress["experiments_file"] = str(path)
        progress["experiment_rows"] = max(len(lines) - 1, 0)
        progress["last_experiment_row"] = lines[-1] if len(lines) > 1 else ""
        break
    return progress


def _status_state(pid: int | None, exit_code: int | None, tmux_alive: bool) -> str:
    if exit_code is not None:
        return "completed" if exit_code == 0 else "failed"
    if _process_alive(pid) or tmux_alive:
        return "running"
    if pid is None and tmux_alive:
        return "launching"
    return "stopped"


def build_status_payload(paths: VastRunPaths, *, refresh_time: str | None = None) -> dict[str, Any]:
    launch = _read_json(paths.launch_file)
    pid = _read_int(paths.pid_file)
    exit_code = _read_int(paths.exit_code_file)
    tmux_alive = _tmux_has_session(str(launch.get("session_name", "")))
    log_exists = paths.log_file.exists()
    log_size = paths.log_file.stat().st_size if log_exists else 0
    log_mtime = (
        datetime.fromtimestamp(paths.log_file.stat().st_mtime, tz=UTC).replace(microsecond=0).isoformat()
        if log_exists
        else None
    )

    status = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "updated_at": refresh_time or _utc_now(),
        "session_name": launch.get("session_name"),
        "label": launch.get("label"),
        "cwd": launch.get("cwd"),
        "command": launch.get("command", []),
        "command_shell": launch.get("command_shell", ""),
        "hostname": launch.get("hostname"),
        "pid": pid,
        "shell_pid": _read_int(paths.shell_pid_file),
        "process_alive": _process_alive(pid),
        "tmux_session_present": tmux_alive,
        "exit_code": exit_code,
        "state": _status_state(pid, exit_code, tmux_alive),
        "log_file": str(paths.log_file),
        "log_size_bytes": log_size,
        "log_updated_at": log_mtime,
        "last_log_line": _read_log_tail(paths.log_file),
        "gpu": _gpu_snapshot(),
        "artifacts": _collect_optimizer_progress(paths.run_dir),
    }
    return status


def write_status_files(paths: VastRunPaths, payload: dict[str, Any]) -> None:
    heartbeat = {
        "updated_at": payload["updated_at"],
        "state": payload["state"],
        "pid": payload["pid"],
        "process_alive": payload["process_alive"],
        "tmux_session_present": payload["tmux_session_present"],
        "last_log_line": payload["last_log_line"],
        "gpu": payload["gpu"],
        "best_score": payload["artifacts"].get("best_score"),
        "experiment_rows": payload["artifacts"].get("experiment_rows"),
    }
    _write_json(paths.status_file, payload)
    _write_json(paths.heartbeat_file, heartbeat)


def build_runner_script(paths: VastRunPaths, *, cwd: Path, command: list[str]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(cwd))}",
            f"echo $$ > {shlex.quote(str(paths.shell_pid_file))}",
            "export PYTHONUNBUFFERED=1",
            f"{shlex.join(command)} > >(tee -a {shlex.quote(str(paths.log_file))}) 2> >(tee -a {shlex.quote(str(paths.log_file))} >&2) &",
            "child=$!",
            f"echo \"$child\" > {shlex.quote(str(paths.pid_file))}",
            "if wait \"$child\"; then",
            "  rc=0",
            "else",
            "  rc=$?",
            "fi",
            f"echo \"$rc\" > {shlex.quote(str(paths.exit_code_file))}",
            "exit \"$rc\"",
            "",
        ]
    )


def _monitor_command(paths: VastRunPaths, interval_s: int) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "v10" / "run_vast_worker.py"),
        "monitor",
        "--run-dir",
        str(paths.run_dir),
        "--interval",
        str(interval_s),
    ]


def start_run(
    *,
    run_dir: Path,
    session_name: str,
    command: list[str],
    cwd: Path,
    interval_s: int,
    label: str | None,
) -> VastRunPaths:
    if not _tmux_available():
        raise RuntimeError("tmux is required for this wrapper. Install it first, e.g. `apt-get install -y tmux`.")
    if _tmux_has_session(session_name):
        raise RuntimeError(f"tmux session already exists: {session_name}")
    if not command:
        raise RuntimeError("no command provided")

    paths = VastRunPaths.from_run_dir(run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    launch = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "started_at": _utc_now(),
        "session_name": session_name,
        "label": label or session_name,
        "cwd": str(cwd),
        "command": command,
        "command_shell": shlex.join(command),
        "hostname": socket.gethostname(),
    }
    _write_json(paths.launch_file, launch)
    write_status_files(
        paths,
        {
            **launch,
            "updated_at": _utc_now(),
            "pid": None,
            "shell_pid": None,
            "process_alive": False,
            "tmux_session_present": False,
            "exit_code": None,
            "state": "launching",
            "log_file": str(paths.log_file),
            "log_size_bytes": 0,
            "log_updated_at": None,
            "last_log_line": "",
            "gpu": _gpu_snapshot(),
            "artifacts": {},
        },
    )

    paths.runner_script.write_text(build_runner_script(paths, cwd=cwd, command=command), encoding="utf-8")
    paths.runner_script.chmod(0o755)

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, str(paths.runner_script)],
        check=True,
    )
    subprocess.Popen(
        _monitor_command(paths, interval_s),
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return paths


def monitor_run(paths: VastRunPaths, interval_s: int) -> int:
    while True:
        payload = build_status_payload(paths)
        write_status_files(paths, payload)
        if payload["state"] in {"completed", "failed", "stopped"} and not payload["tmux_session_present"]:
            break
        time.sleep(interval_s)
    return 0


def _print_status(paths: VastRunPaths, refresh: bool) -> int:
    if refresh:
        payload = build_status_payload(paths)
        write_status_files(paths, payload)
    else:
        payload = _read_json(paths.status_file) or build_status_payload(paths)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _tail_log(paths: VastRunPaths, lines: int) -> int:
    if not paths.log_file.exists():
        raise RuntimeError(f"log file not found: {paths.log_file}")
    return subprocess.call(["tail", "-n", str(lines), "-f", str(paths.log_file)])


def _attach_tmux(session_name: str) -> int:
    if not _tmux_has_session(session_name):
        raise RuntimeError(f"tmux session not found: {session_name}")
    return subprocess.call(["tmux", "attach", "-t", session_name])


def _stop_run(paths: VastRunPaths, session_name: str) -> int:
    if _tmux_has_session(session_name):
        subprocess.run(["tmux", "send-keys", "-t", session_name, "C-c"], check=False)
        time.sleep(1.0)
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
    payload = build_status_payload(paths)
    write_status_files(paths, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start and inspect V10 Vast worker runs")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    start = subparsers.add_parser("start", help="start a command in tmux with heartbeat/status files")
    start.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    start.add_argument("--session-name", default="mm-v10")
    start.add_argument("--label", default=None)
    start.add_argument("--cwd", type=Path, default=PROJECT_ROOT)
    start.add_argument("--interval", type=int, default=15)
    start.add_argument("command", nargs=argparse.REMAINDER)

    monitor = subparsers.add_parser("monitor", help="internal monitor loop")
    monitor.add_argument("--run-dir", type=Path, required=True)
    monitor.add_argument("--interval", type=int, default=15)

    status = subparsers.add_parser("status", help="print current status json")
    status.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    status.add_argument("--refresh", action="store_true")

    tail = subparsers.add_parser("tail", help="tail the run log")
    tail.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    tail.add_argument("--lines", type=int, default=40)

    attach = subparsers.add_parser("attach", help="attach to the tmux session")
    attach.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    attach.add_argument("--session-name", default=None)

    stop = subparsers.add_parser("stop", help="stop the tmux session and refresh status")
    stop.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    stop.add_argument("--session-name", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command_name == "start":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        paths = start_run(
            run_dir=args.run_dir,
            session_name=args.session_name,
            command=command,
            cwd=args.cwd,
            interval_s=args.interval,
            label=args.label,
        )
        payload = _read_json(paths.status_file)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    paths = VastRunPaths.from_run_dir(args.run_dir)
    launch = _read_json(paths.launch_file)
    session_name = getattr(args, "session_name", None) or launch.get("session_name") or "mm-v10"

    if args.command_name == "monitor":
        return monitor_run(paths, args.interval)
    if args.command_name == "status":
        return _print_status(paths, args.refresh)
    if args.command_name == "tail":
        return _tail_log(paths, args.lines)
    if args.command_name == "attach":
        return _attach_tmux(session_name)
    if args.command_name == "stop":
        return _stop_run(paths, session_name)
    raise RuntimeError(f"unknown command: {args.command_name}")


if __name__ == "__main__":
    raise SystemExit(main())

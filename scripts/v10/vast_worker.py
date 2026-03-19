#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import site
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _readable_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _candidate_cuda_roots() -> list[Path]:
    suffixes = [
        "nvidia/cuda_nvrtc",
        "nvidia/cuda_runtime",
    ]
    candidates: list[Path] = []
    for base in site.getsitepackages():
        for suffix in suffixes:
            path = Path(base) / suffix
            if path.exists() and path not in candidates:
                candidates.append(path)
    return candidates


def _candidate_cuda_library_dirs() -> list[str]:
    suffixes = [
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/cublas/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
    ]
    candidates: list[str] = []
    for base in site.getsitepackages():
        for suffix in suffixes:
            path = (Path(base) / suffix).resolve()
            if path.exists():
                value = str(path)
                if value not in candidates:
                    candidates.append(value)
    return candidates


def _augment_cuda_env(env: dict[str, str], log_dir: Path) -> None:
    if "CUDA_PATH" not in env:
        roots = _candidate_cuda_roots()
        if roots:
            env["CUDA_PATH"] = str(roots[0])
    if "CUDA_HOME" not in env and env.get("CUDA_PATH"):
        env["CUDA_HOME"] = env["CUDA_PATH"]
    candidates = _candidate_cuda_library_dirs()
    existing = env.get("LD_LIBRARY_PATH", "")
    seen: list[str] = []
    for value in candidates + ([existing] if existing else []):
        for item in str(value).split(":"):
            item = item.strip()
            if item and item not in seen:
                seen.append(item)
    if seen:
        env["LD_LIBRARY_PATH"] = ":".join(seen)
    env.setdefault("MARCH_MADNESS_INFERENCE_BACKEND", "auto")
    env.setdefault("CUPY_CACHE_DIR", str((log_dir / "cupy_kernel_cache").resolve()))


def _gpu_snapshot() -> list[dict[str, object]]:
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,pstate"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if not output:
        return []
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_pct": int(parts[2]),
                "utilization_memory_pct": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "memory_total_mib": int(parts[5]),
                "temperature_c": int(parts[6]),
                "pstate": parts[7],
            }
        )
    return rows


@dataclass
class StreamState:
    last_output_at: str | None = None
    last_output_line: str = ""
    output_lines: int = 0
    log_bytes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, line: str, log_path: Path) -> None:
        clean = line.rstrip("\n")
        with self.lock:
            self.last_output_at = _iso_now()
            self.last_output_line = clean[-500:]
            self.output_lines += 1
            try:
                self.log_bytes = log_path.stat().st_size
            except FileNotFoundError:
                self.log_bytes = 0

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "last_output_at": self.last_output_at,
                "last_output_line": self.last_output_line,
                "output_lines": self.output_lines,
                "log_bytes": self.log_bytes,
            }


def _stream_output(
    pipe,
    log_handle,
    log_path: Path,
    state: StreamState,
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            log_handle.write(line)
            log_handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            state.update(line, log_path)
    finally:
        pipe.close()


def _status_payload(
    *,
    name: str,
    session: str | None,
    cwd: Path,
    command: list[str],
    log_path: Path,
    heartbeat_path: Path,
    status_path: Path,
    pid: int | None,
    started_at: str,
    state_name: str,
    stream_state: StreamState,
    exit_code: int | None,
    child_env: dict[str, str],
) -> dict[str, object]:
    stream = stream_state.snapshot()
    started_epoch = datetime.fromisoformat(started_at).timestamp()
    return {
        "name": name,
        "tmux_session": session,
        "state": state_name,
        "pid": pid,
        "exit_code": exit_code,
        "hostname": socket.gethostname(),
        "cwd": str(cwd),
        "command": command,
        "command_display": _readable_command(command),
        "started_at": started_at,
        "updated_at": _iso_now(),
        "uptime_seconds": round(max(time.time() - started_epoch, 0.0), 2),
        "paths": {
            "log": str(log_path),
            "heartbeat": str(heartbeat_path),
            "status": str(status_path),
        },
        "stream": stream,
        "gpu": _gpu_snapshot(),
        "env": {
            "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
            "pythonunbuffered": child_env.get("PYTHONUNBUFFERED"),
            "cuda_path": child_env.get("CUDA_PATH"),
            "cuda_home": child_env.get("CUDA_HOME"),
            "march_madness_inference_backend": child_env.get("MARCH_MADNESS_INFERENCE_BACKEND"),
            "cupy_cache_dir": child_env.get("CUPY_CACHE_DIR"),
        },
    }


def _heartbeat_payload(status: dict[str, object]) -> dict[str, object]:
    return {
        "name": status["name"],
        "tmux_session": status["tmux_session"],
        "state": status["state"],
        "pid": status["pid"],
        "exit_code": status["exit_code"],
        "updated_at": status["updated_at"],
        "uptime_seconds": status["uptime_seconds"],
        "paths": status["paths"],
        "stream": status["stream"],
        "gpu": status["gpu"],
    }


def run_worker(args: argparse.Namespace) -> int:
    log_dir = args.log_dir.resolve()
    cwd = args.cwd.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "run.log"
    status_path = log_dir / "status.json"
    heartbeat_path = log_dir / "heartbeat.json"

    command = list(args.command)
    if not command:
        raise SystemExit("no command provided after --")

    started_at = _iso_now()
    stream_state = StreamState()
    stop_requested = False
    child: subprocess.Popen[str] | None = None

    def _handle_signal(signum, _frame) -> None:
        nonlocal stop_requested, child
        stop_requested = True
        if child and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _augment_cuda_env(env, log_dir)

    launch_payload = {
        "name": args.name,
        "tmux_session": args.tmux_session,
        "cwd": str(cwd),
        "command": command,
        "command_display": _readable_command(command),
        "started_at": started_at,
    }
    _write_json(log_dir / "launch.json", launch_payload)

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{started_at}] launching {_readable_command(command)}\n")
        log_handle.flush()

        child = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        reader = threading.Thread(
            target=_stream_output,
            args=(child.stdout, log_handle, log_path, stream_state),
            daemon=True,
        )
        reader.start()

        state_name = "running"
        exit_code: int | None = None

        while True:
            polled = child.poll()
            if polled is not None:
                exit_code = int(polled)
                state_name = "completed" if exit_code == 0 else "failed"
            elif stop_requested:
                state_name = "stopping"

            status = _status_payload(
                name=args.name,
                session=args.tmux_session,
                cwd=cwd,
                command=command,
                log_path=log_path,
                heartbeat_path=heartbeat_path,
                status_path=status_path,
                pid=child.pid,
                started_at=started_at,
                state_name=state_name,
                stream_state=stream_state,
                exit_code=exit_code,
                child_env=env,
            )
            _write_json(status_path, status)
            _write_json(heartbeat_path, _heartbeat_payload(status))

            if polled is not None:
                break
            time.sleep(args.poll_seconds)

        reader.join(timeout=max(args.poll_seconds, 1))

        final_status = _status_payload(
            name=args.name,
            session=args.tmux_session,
            cwd=cwd,
            command=command,
            log_path=log_path,
            heartbeat_path=heartbeat_path,
            status_path=status_path,
            pid=child.pid,
            started_at=started_at,
            state_name=state_name,
            stream_state=stream_state,
            exit_code=exit_code,
            child_env=env,
        )
        _write_json(status_path, final_status)
        _write_json(heartbeat_path, _heartbeat_payload(final_status))
        log_handle.write(
            f"[{_iso_now()}] process exited with code {exit_code}\n"
        )
        log_handle.flush()

    return 0 if exit_code == 0 else int(exit_code or 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vast worker process runner")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run a command and write status files")
    run_parser.add_argument("--name", required=True, help="Logical run name")
    run_parser.add_argument("--tmux-session", default=None, help="Owning tmux session name")
    run_parser.add_argument("--log-dir", type=Path, required=True, help="Directory for run.log/status files")
    run_parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for the child process")
    run_parser.add_argument("--poll-seconds", type=int, default=10, help="Heartbeat/status refresh interval")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after --")

    args = parser.parse_args(argv)
    if args.command_name == "run":
        command = list(args.command)
        if command and command[0] == "--":
            args.command = command[1:]
        return run_worker(args)
    raise SystemExit(f"unsupported subcommand: {args.command_name}")


if __name__ == "__main__":
    raise SystemExit(main())

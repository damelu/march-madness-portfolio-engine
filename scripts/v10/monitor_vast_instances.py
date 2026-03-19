#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REMOTE_PROBE = r"""
import json
import subprocess
from pathlib import Path


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def gpu_snapshot():
    result = run([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, util, used, total = parts
        try:
            rows.append(
                {
                    "name": name,
                    "utilization_gpu_pct": float(util),
                    "memory_used_mib": float(used),
                    "memory_total_mib": float(total),
                }
            )
        except ValueError:
            continue
    return rows


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


repo_root = Path("/root/mm-v9")
outputs = repo_root / "outputs"
wrapper_statuses = []
if outputs.exists():
    for status_file in sorted(outputs.glob("*/status.json")):
        payload = read_json(status_file)
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload["status_file"] = str(status_file)
        payload["run_dir"] = str(status_file.parent)
        wrapper_statuses.append(payload)

legacy = {}
legacy_dir = outputs / "autobracket"
best_params = legacy_dir / "best_params.json"
experiments_parallel = legacy_dir / "experiments_parallel.tsv"
experiments_gpu = legacy_dir / "experiments_gpu.tsv"
if best_params.exists():
    legacy["best_params"] = read_json(best_params)
for path in (experiments_parallel, experiments_gpu):
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        legacy[path.name] = {
            "rows": max(len(lines) - 1, 0),
            "last_row": lines[-1] if len(lines) > 1 else "",
        }

ps = run(["ps", "-eo", "pid,etimes,pcpu,pmem,cmd"])
processes = []
parallel_workers = 0
if ps.returncode == 0:
    for raw in ps.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(
            token in line
            for token in (
                "scripts/autobracket_parallel.py",
                "scripts/autobracket_v10.py",
                "scripts/build_v10_portfolio.py",
                "scripts/backtest_v10.py",
                "scripts/train_v10_game_model.py",
                "scripts/v10/run_vast_worker.py",
                "scripts/v10/vast_worker.py",
            )
        ) and "grep" not in line and "sshd" not in line:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid, elapsed, pcpu, pmem, cmd = parts
            processes.append(
                {
                    "pid": int(pid),
                    "elapsed_seconds": int(elapsed),
                    "cpu_pct": float(pcpu),
                    "mem_pct": float(pmem),
                    "cmd": cmd,
                }
            )
            if ".venv/bin/python3 scripts/autobracket_parallel.py" in cmd:
                parallel_workers += 1

payload = {
    "host": run(["hostname"]).stdout.strip(),
    "gpu": gpu_snapshot(),
    "wrapper_statuses": wrapper_statuses,
    "legacy_artifacts": legacy,
    "processes": processes,
    "parallel_worker_processes": parallel_workers,
}
print(json.dumps(payload))
"""


@dataclass
class InstanceSummary:
    instance_id: int
    actual_status: str
    cur_state: str
    gpu_name: str
    location: str
    ssh_host: str | None
    ssh_port: int | None
    remote: dict[str, Any] | None = None
    remote_error: str | None = None

    def mode(self) -> str:
        if self.actual_status != "running":
            return self.actual_status
        if self.remote_error:
            return "ssh_error"
        remote = self.remote or {}
        wrappers = remote.get("wrapper_statuses") or []
        if wrappers:
            states = {str(item.get("state", "unknown")) for item in wrappers if isinstance(item, dict)}
            if len(wrappers) == 1:
                return f"wrapper:{next(iter(states)) if states else 'unknown'}"
            return f"wrapper:{len(wrappers)}"
        procs = remote.get("processes") or []
        if any("scripts/autobracket_parallel.py" in str(p.get("cmd", "")) for p in procs):
            return "legacy_parallel"
        if any("scripts/autobracket_v10.py" in str(p.get("cmd", "")) for p in procs):
            return "v10_direct"
        return "running"


def run_local_json(cmd: list[str]) -> Any:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "command failed")
    return json.loads(result.stdout)


def fetch_instances() -> list[dict[str, Any]]:
    return run_local_json(["vastai", "show", "instances", "--raw"])


def remote_probe(instance: dict[str, Any], *, repo_root: str, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if not host or not port:
        return None, "missing_ssh_target"
    encoded = base64.b64encode(REMOTE_PROBE.encode("utf-8")).decode("ascii")
    remote_cmd = (
        "python3 -c \"import base64,os; "
        f"os.chdir({repo_root!r}) if os.path.isdir({repo_root!r}) else None; "
        f"exec(base64.b64decode('{encoded}'))\""
    )
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={timeout}",
        "root@" + str(host),
        "-p",
        str(port),
        remote_cmd,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        return None, err or f"ssh exited {result.returncode}"
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        return None, "empty probe output"
    last = stdout[-1]
    try:
        return json.loads(last), None
    except json.JSONDecodeError:
        return None, f"invalid json: {last[:200]}"


def build_summaries(instances: list[dict[str, Any]], *, repo_root: str, timeout: int, only_ids: set[int] | None) -> list[InstanceSummary]:
    summaries: list[InstanceSummary] = []
    for item in instances:
        instance_id = int(item["id"])
        if only_ids and instance_id not in only_ids:
            continue
        summary = InstanceSummary(
            instance_id=instance_id,
            actual_status=str(item.get("actual_status")),
            cur_state=str(item.get("cur_state")),
            gpu_name=str(item.get("gpu_name")),
            location=str(item.get("geolocation")),
            ssh_host=item.get("ssh_host"),
            ssh_port=item.get("ssh_port"),
        )
        if summary.actual_status == "running":
            remote, error = remote_probe(item, repo_root=repo_root, timeout=timeout)
            summary.remote = remote
            summary.remote_error = error
        summaries.append(summary)
    return summaries


def format_uptime(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def best_score(remote: dict[str, Any] | None) -> str:
    if not remote:
        return "-"
    wrappers = remote.get("wrapper_statuses") or []
    if wrappers:
        scores = []
        for wrapper in wrappers:
            artifacts = wrapper.get("artifacts") or {}
            score = artifacts.get("best_score")
            if score is not None:
                try:
                    scores.append(float(score))
                except Exception:
                    pass
        if scores:
            return f"{max(scores):.6f}"
    legacy = remote.get("legacy_artifacts") or {}
    best = legacy.get("best_params") or {}
    score = best.get("score")
    if score is not None:
        try:
            return f"{float(score):.6f}"
        except Exception:
            return str(score)
    return "-"


def experiment_rows(remote: dict[str, Any] | None) -> str:
    if not remote:
        return "-"
    wrappers = remote.get("wrapper_statuses") or []
    if wrappers:
        rows = []
        for wrapper in wrappers:
            artifacts = wrapper.get("artifacts") or {}
            value = artifacts.get("experiment_rows")
            if value is not None:
                rows.append(str(value))
        if rows:
            return ",".join(rows)
    legacy = remote.get("legacy_artifacts") or {}
    for key in ("experiments_parallel.tsv", "experiments_gpu.tsv"):
        payload = legacy.get(key)
        if isinstance(payload, dict) and "rows" in payload:
            return str(payload["rows"])
    return "-"


def latest_note(summary: InstanceSummary) -> str:
    if summary.remote_error:
        return summary.remote_error[:80]
    remote = summary.remote or {}
    wrappers = remote.get("wrapper_statuses") or []
    if wrappers:
        notes = []
        for wrapper in wrappers:
            state = wrapper.get("state", "unknown")
            last_line = ((wrapper.get("stream") or {}).get("last_output_line") or "")[:60]
            run_dir = Path(str(wrapper.get("run_dir", ""))).name
            notes.append(f"{run_dir}:{state}:{last_line}")
        return " | ".join(notes)[:120]
    legacy = remote.get("legacy_artifacts") or {}
    for key in ("experiments_parallel.tsv", "experiments_gpu.tsv"):
        payload = legacy.get(key)
        if isinstance(payload, dict):
            last_row = str(payload.get("last_row", ""))
            if last_row:
                return last_row[:120]
    procs = remote.get("processes") or []
    if procs:
        return str(procs[0].get("cmd", ""))[:120]
    return "-"


def render_table(summaries: list[InstanceSummary]) -> str:
    headers = ["id", "status", "mode", "location", "gpu", "util", "mem", "best", "rows", "note"]
    rows: list[list[str]] = []
    for summary in summaries:
        gpu_info = ((summary.remote or {}).get("gpu") or [])
        first_gpu = gpu_info[0] if gpu_info else {}
        util = first_gpu.get("utilization_gpu_pct")
        used = first_gpu.get("memory_used_mib")
        total = first_gpu.get("memory_total_mib")
        mem = "-"
        if used is not None and total is not None:
            mem = f"{int(used)}/{int(total)}"
        rows.append(
            [
                str(summary.instance_id),
                summary.actual_status,
                summary.mode(),
                summary.location,
                summary.gpu_name,
                "-" if util is None else f"{int(float(util))}%",
                mem,
                best_score(summary.remote),
                experiment_rows(summary.remote),
                latest_note(summary),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    def fmt(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def render_details(summary: InstanceSummary) -> str:
    lines = [
        f"Instance {summary.instance_id} [{summary.actual_status}/{summary.cur_state}] {summary.location} {summary.gpu_name}",
    ]
    if summary.remote_error:
        lines.append(f"  remote_error: {summary.remote_error}")
        return "\n".join(lines)
    remote = summary.remote or {}
    gpu_info = remote.get("gpu") or []
    if gpu_info:
        for idx, gpu in enumerate(gpu_info):
            lines.append(
                "  gpu[{idx}]: {name} util={util}% mem={used}/{total} MiB".format(
                    idx=idx,
                    name=gpu.get("name", "?"),
                    util=int(float(gpu.get("utilization_gpu_pct", 0))),
                    used=int(float(gpu.get("memory_used_mib", 0))),
                    total=int(float(gpu.get("memory_total_mib", 0))),
                )
            )
    wrappers = remote.get("wrapper_statuses") or []
    if wrappers:
        for wrapper in wrappers:
            stream = wrapper.get("stream") or {}
            artifacts = wrapper.get("artifacts") or {}
            lines.append(
                f"  wrapper: {Path(str(wrapper.get('run_dir', '?'))).name} state={wrapper.get('state')} "
                f"pid={wrapper.get('pid')} best={artifacts.get('best_score')} rows={artifacts.get('experiment_rows')}"
            )
            last_line = stream.get("last_output_line")
            if last_line:
                lines.append(f"    last: {str(last_line)[:140]}")
    procs = remote.get("processes") or []
    if procs:
        lines.append(f"  processes: {len(procs)} tracked, parallel_workers={remote.get('parallel_worker_processes', 0)}")
        for proc in procs[:4]:
            lines.append(
                f"    pid={proc.get('pid')} elapsed={format_uptime(proc.get('elapsed_seconds'))} "
                f"cpu={proc.get('cpu_pct')} cmd={str(proc.get('cmd', ''))[:120]}"
            )
    legacy = remote.get("legacy_artifacts") or {}
    if legacy:
        best = legacy.get("best_params") or {}
        if best:
            lines.append(f"  legacy_best_score: {best.get('score')}")
        for key in ("experiments_parallel.tsv", "experiments_gpu.tsv"):
            payload = legacy.get(key)
            if isinstance(payload, dict):
                lines.append(f"  {key}: rows={payload.get('rows')} last={str(payload.get('last_row', ''))[:120]}")
    return "\n".join(lines)


def write_snapshot(path: Path, summaries: list[InstanceSummary]) -> None:
    payload = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "instances": [
            {
                "instance_id": summary.instance_id,
                "actual_status": summary.actual_status,
                "cur_state": summary.cur_state,
                "gpu_name": summary.gpu_name,
                "location": summary.location,
                "ssh_host": summary.ssh_host,
                "ssh_port": summary.ssh_port,
                "remote": summary.remote,
                "remote_error": summary.remote_error,
            }
            for summary in summaries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor live Vast.ai March Madness workers")
    parser.add_argument("--repo-root", default="/root/mm-v9", help="Remote repo root on each instance")
    parser.add_argument("--timeout", type=int, default=8, help="SSH connect timeout in seconds")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds; 0 prints once")
    parser.add_argument("--instance", action="append", type=int, default=None, help="Restrict to specific instance id")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the latest aggregated snapshot to JSON")
    parser.add_argument("--details", action="store_true", help="Show per-instance detail sections below the table")
    args = parser.parse_args(argv)

    only_ids = set(args.instance or []) or None
    while True:
        instances = fetch_instances()
        summaries = build_summaries(instances, repo_root=args.repo_root, timeout=args.timeout, only_ids=only_ids)
        if args.watch:
            clear_screen()
        print(f"Captured at {datetime.now().astimezone().isoformat(timespec='seconds')}")
        print(render_table(summaries))
        if args.details:
            print()
            for summary in summaries:
                print(render_details(summary))
                print()
        if args.json_out:
            write_snapshot(args.json_out, summaries)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())

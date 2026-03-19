from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark V10 worker throughput on a single host.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Project root on the host.")
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration per experiment in seconds.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint copied into each worker log dir before launch.",
    )
    parser.add_argument(
        "--dataset",
        default="data/models/v10/inference/2026/snapshot.json",
        help="Dataset path forwarded to autobracket_v10.py.",
    )
    parser.add_argument(
        "--game-model-artifact",
        default="data/models/v10/game_model_v10_default.pkl",
        help="Game-model artifact path.",
    )
    parser.add_argument(
        "--public-field-artifact",
        default="data/models/v10/public_field_v10_default.pkl",
        help="Public-field artifact path.",
    )
    parser.add_argument("--iterations", type=int, default=300, help="Optimizer iterations per worker.")
    parser.add_argument("--batch", type=int, default=4, help="Optimizer batch size.")
    parser.add_argument("--sims", type=int, default=120, help="Tournament sims per eval.")
    parser.add_argument("--candidates", type=int, default=80, help="Candidate brackets per eval.")
    parser.add_argument("--training-profile-id", default="default_v10")
    parser.add_argument("--release-seed-count", type=int, default=3)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="NAME:WORKERS[:ENV=VALUE,ENV=VALUE]",
        help="Repeatable experiment specification. Example: cpu16:16:MARCH_MADNESS_INFERENCE_BACKEND=numpy,OMP_NUM_THREADS=1",
    )
    return parser.parse_args()


def parse_experiment(raw: str) -> tuple[str, int, list[str]]:
    parts = raw.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"invalid experiment spec: {raw}")
    name = parts[0]
    workers = int(parts[1])
    envs = []
    if len(parts) == 3 and parts[2]:
        envs = [item for item in parts[2].split(",") if item]
    return name, workers, envs


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def maybe_run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collect_rows(log_dir: Path) -> int:
    tsv = log_dir / "experiments_v10.tsv"
    if not tsv.exists():
        return 0
    with tsv.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def collect_status(log_dir: Path) -> dict[str, object]:
    status_path = log_dir / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def collect_best_score(log_dir: Path) -> float | None:
    path = log_dir / "best_v10_params.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    score = payload.get("best_score")
    return float(score) if score is not None else None


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    args = parse_args()
    project_root = args.cwd.resolve()
    fleet_script = project_root / "scripts/v10/run_vast_worker_fleet.sh"
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None

    if not fleet_script.exists():
        raise FileNotFoundError(f"missing fleet script: {fleet_script}")

    experiments = (
        [
            parse_experiment(raw)
            for raw in args.experiment
        ]
        if args.experiment
        else [
            (
                "cpu16",
                16,
                [
                    "MARCH_MADNESS_INFERENCE_BACKEND=numpy",
                    "MARCH_MADNESS_PAYOUT_BACKEND=cpu",
                    "MARCH_MADNESS_PORTFOLIO_BACKEND=cpu",
                    "OMP_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1",
                    "MKL_NUM_THREADS=1",
                    "NUMEXPR_NUM_THREADS=1",
                ],
            )
        ]
    )

    common = [
        "bash",
        str(fleet_script),
        "start",
        "--cwd",
        str(project_root),
        "--gpu-id",
        args.gpu_id,
        "--iterations",
        str(args.iterations),
        "--batch",
        str(args.batch),
        "--sims",
        str(args.sims),
        "--candidates",
        str(args.candidates),
        "--training-profile-id",
        args.training_profile_id,
        "--dataset",
        args.dataset,
        "--game-model-artifact",
        args.game_model_artifact,
        "--public-field-artifact",
        args.public_field_artifact,
        "--release-seed-count",
        str(args.release_seed_count),
    ]
    if checkpoint:
        common.extend(["--checkpoint", str(checkpoint)])

    results: list[dict[str, object]] = []
    for name, workers, envs in experiments:
        prefix = f"bench-{name}"
        log_prefix = f"outputs/bench_{name}"
        stop_cmd = [
            "bash",
            str(fleet_script),
            "stop",
            "--workers",
            str(workers),
            "--session-prefix",
            prefix,
            "--log-prefix",
            log_prefix,
            "--cwd",
            str(project_root),
        ]
        maybe_run(stop_cmd)

        for index in range(workers):
            shutil.rmtree(project_root / f"{log_prefix}_{index}", ignore_errors=True)

        launch_cmd = common + [
            "--workers",
            str(workers),
            "--session-prefix",
            prefix,
            "--log-prefix",
            log_prefix,
        ]
        for env in envs:
            launch_cmd.extend(["--env", env])

        start_time = time.monotonic()
        run(launch_cmd)
        time.sleep(args.duration)

        rows = 0
        completed = 0
        running = 0
        best_scores: list[float] = []
        for index in range(workers):
            log_dir = project_root / f"{log_prefix}_{index}"
            rows += collect_rows(log_dir)
            status = collect_status(log_dir)
            state = status.get("state")
            if state == "completed":
                completed += 1
            elif state == "running":
                running += 1
            best_score = collect_best_score(log_dir)
            if best_score is not None:
                best_scores.append(best_score)

        results.append(
            {
                "name": name,
                "workers": workers,
                "rows": rows,
                "rows_per_second": rows / max(args.duration, 1),
                "rows_per_worker": rows / max(workers, 1),
                "running": running,
                "completed": completed,
                "best_score_max": max(best_scores) if best_scores else None,
                "gpu_snapshot": gpu_snapshot(),
                "duration_seconds": round(time.monotonic() - start_time, 3),
                "envs": envs,
            }
        )

        maybe_run(stop_cmd)
        time.sleep(5)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

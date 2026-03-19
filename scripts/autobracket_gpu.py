#!/usr/bin/env python3
"""
autobracket_gpu.py — GPU-accelerated bracket parameter optimizer.

Unlike autobracket_parallel.py (which forks workers that can't share GPU),
this runs SEQUENTIAL evaluations in a single process where CuPy has full
GPU access. The GPU accelerates scoring/FPE/overlap 100x, making each
eval ~3s instead of ~10s. No multiprocessing, no CUDA context issues.

For batch efficiency, evaluates multiple mutations per round by running
them sequentially on GPU, then picking the best.

Usage:
    uv run python scripts/autobracket_gpu.py                    # 500 rounds
    uv run python scripts/autobracket_gpu.py --batch 16         # 16 mutations per round
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AUTOBRACKET_SCHEMA_VERSION = 3

from scripts.autobracket import (
    BracketParams,
    apply_params_to_engine,
    composite_score,
    evaluate_params,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU-accelerated AutoBracket optimizer")
    parser.add_argument("--iterations", type=int, default=500, help="Number of rounds")
    parser.add_argument("--batch", type=int, default=8, help="Mutations evaluated per round (sequential on GPU)")
    parser.add_argument("--sims", type=int, default=5000, help="Tournament simulations per eval")
    parser.add_argument("--candidates", type=int, default=400, help="Candidate brackets per eval")
    parser.add_argument("--scale", type=float, default=0.15, help="Mutation scale")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset or PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

    log_dir = PROJECT_ROOT / "outputs" / "autobracket"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiments_gpu.tsv"
    best_params_file = log_dir / "best_params.json"

    # Check GPU
    try:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        vram = cp.cuda.runtime.memGetInfo()[1] / 1024**3
        print(f"[gpu] GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    except Exception:
        print("[gpu] WARNING: No GPU detected. Running on CPU (will be slower).")

    rng = np.random.default_rng(77)

    # Initialize or resume
    if args.resume and best_params_file.exists():
        with best_params_file.open() as f:
            saved = json.load(f)
        if saved.get("schema_version") != AUTOBRACKET_SCHEMA_VERSION:
            print(f"[gpu] Stale checkpoint, starting from baseline")
            args.resume = False
        else:
            best_params = BracketParams(**saved["params"])
            best_score = saved["score"]
            print(f"[gpu] Resumed from best score: {best_score:.6f}")

    if not args.resume or not best_params_file.exists():
        best_params = BracketParams()
        print("[gpu] Evaluating baseline...")
        start = time.perf_counter()
        metrics = evaluate_params(best_params, dataset_path, args.sims, args.candidates)
        baseline_time = time.perf_counter() - start
        best_score = composite_score(metrics)
        print(f"[gpu] Baseline: score={best_score:.6f} ({baseline_time:.1f}s per eval)")

        tmp_fd, tmp_path = tempfile.mkstemp(dir=best_params_file.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump({"schema_version": AUTOBRACKET_SCHEMA_VERSION, "params": best_params.to_dict(), "score": best_score, "metrics": metrics}, f, indent=2)
            os.replace(tmp_path, best_params_file)
        except BaseException:
            os.unlink(tmp_path)
            raise

    print(f"[gpu] Batch: {args.batch} mutations/round (sequential on GPU)")
    print(f"[gpu] {args.iterations} rounds × {args.batch} = {args.iterations * args.batch} total evaluations")
    print()

    # TSV log
    write_header = not log_file.exists()
    tsv = open(log_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(tsv, delimiter="\t")
    if write_header:
        writer.writerow([
            "round", "timestamp", "best_score", "round_best_score", "fpe",
            "small_fpe", "mid_fpe", "large_fpe", "archetypes", "champions",
            "status", "elapsed_s", "evaluated", "description"
        ])
        tsv.flush()

    total_kept = 0
    total_evaluated = 0
    total_time = 0
    stale_rounds = 0

    for round_num in range(1, args.iterations + 1):
        mutations = []
        descriptions = []
        for _ in range(args.batch):
            mutated = best_params.mutate(rng, scale=args.scale)
            changes = []
            for k in best_params.__dict__:
                old = getattr(best_params, k)
                new = getattr(mutated, k)
                if abs(old - new) > 1e-6:
                    changes.append(f"{k}: {old:.3f}→{new:.3f}")
            mutations.append(mutated)
            descriptions.append("; ".join(changes))

        # Evaluate all mutations SEQUENTIALLY (single process = GPU works)
        start = time.perf_counter()
        round_best_score = -1e9
        round_best_idx = -1
        round_best_metrics = {}

        for idx, params in enumerate(mutations):
            try:
                metrics = evaluate_params(params, dataset_path, args.sims, args.candidates)
                score = composite_score(metrics)
                if score > round_best_score:
                    round_best_score = score
                    round_best_idx = idx
                    round_best_metrics = metrics
            except Exception as e:
                print(f"  [crash] mutation {idx}: {e}")

        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_evaluated += args.batch

        if round_best_score > best_score:
            improvement = round_best_score - best_score
            best_score = round_best_score
            best_params = mutations[round_best_idx]
            total_kept += 1
            stale_rounds = 0

            tmp_fd, tmp_path = tempfile.mkstemp(dir=best_params_file.parent, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump({"schema_version": AUTOBRACKET_SCHEMA_VERSION, "params": best_params.to_dict(), "score": best_score, "metrics": round_best_metrics}, f, indent=2)
                os.replace(tmp_path, best_params_file)
            except BaseException:
                os.unlink(tmp_path)
                raise

            m = round_best_metrics
            print(
                f"  Round {round_num:>3}: KEEP  score={best_score:.6f} (+{improvement:.6f}) "
                f"fpe={m.get('first_place_equity',0):.4f} "
                f"({elapsed:.1f}s, {args.batch} evals, {elapsed/args.batch:.1f}s/eval) "
                f"| {descriptions[round_best_idx][:80]}"
            )

            writer.writerow([
                round_num, datetime.now().isoformat(timespec="seconds"),
                f"{best_score:.6f}", f"{round_best_score:.6f}",
                f"{m.get('first_place_equity',0):.4f}",
                f"{m.get('scenario_small_fpe',0):.4f}",
                f"{m.get('scenario_mid_fpe',0):.4f}",
                f"{m.get('scenario_large_fpe',0):.4f}",
                m.get("distinct_archetypes", ""), m.get("unique_champions", ""),
                "KEEP", f"{elapsed:.1f}", args.batch,
                descriptions[round_best_idx][:120],
            ])
        else:
            stale_rounds += 1
            print(
                f"  Round {round_num:>3}: skip  ({elapsed:.1f}s, {args.batch} evals) "
                f"[stale: {stale_rounds}]"
            )
            writer.writerow([
                round_num, datetime.now().isoformat(timespec="seconds"),
                f"{best_score:.6f}", f"{round_best_score:.6f}",
                "", "", "", "", "", "",
                "DISCARD", f"{elapsed:.1f}", args.batch, "",
            ])

        tsv.flush()

        if stale_rounds >= 30:
            print(f"\n[gpu] Converged! No improvement for {stale_rounds} rounds.")
            break

    tsv.close()

    print(f"\n{'='*60}")
    print(f"[gpu] Optimization complete!")
    print(f"  Rounds: {round_num} ({total_evaluated} evals)")
    print(f"  Kept: {total_kept}")
    print(f"  Best score: {best_score:.6f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/max(total_evaluated,1):.1f}s/eval)")
    print(f"  Best params: {best_params_file}")
    print(f"  Log: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
autobracket_parallel.py — Parallel autonomous bracket parameter optimizer.

Evaluates N mutations simultaneously using multiprocessing, keeping the best.
4-6x faster than sequential autobracket.py on a multi-core machine.

Usage:
    uv run python scripts/autobracket_parallel.py                     # 100 iterations, 4 workers
    uv run python scripts/autobracket_parallel.py --iterations 200    # more iterations
    uv run python scripts/autobracket_parallel.py --workers 8         # more parallelism
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
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


def _evaluate_worker(args: tuple) -> tuple[int, BracketParams, dict, float, str]:
    """Worker function for multiprocessing. Evaluates one mutation."""
    worker_id, params, dataset_path, sims, candidates = args
    try:
        metrics = evaluate_params(params, Path(dataset_path), sims, candidates)
        score = composite_score(metrics)
        return (worker_id, params, metrics, score, "OK")
    except Exception as e:
        return (worker_id, params, {}, -1e9, f"CRASH: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel AutoBracket optimizer")
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization rounds")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: CPU count - 1)")
    parser.add_argument("--batch", type=int, default=None, help="Mutations per round (default: workers)")
    parser.add_argument("--sims", type=int, default=5000, help="Tournament simulations per evaluation")
    parser.add_argument("--candidates", type=int, default=400, help="Candidate brackets per evaluation")
    parser.add_argument("--scale", type=float, default=0.12, help="Mutation scale")
    parser.add_argument("--resume", action="store_true", help="Resume from last best params")
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 1)
    batch_size = args.batch or n_workers
    dataset_path = args.dataset or PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

    log_dir = PROJECT_ROOT / "outputs" / "autobracket"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiments_parallel.tsv"
    best_params_file = log_dir / "best_params.json"

    rng = np.random.default_rng(77)

    # Initialize or resume
    if args.resume and best_params_file.exists():
        with best_params_file.open() as f:
            saved = json.load(f)
        if saved.get("schema_version") != AUTOBRACKET_SCHEMA_VERSION:
            print(f"[parallel] Stale checkpoint (schema_version={saved.get('schema_version')}), starting from baseline")
            args.resume = False
        else:
            best_params = BracketParams(**saved["params"])
            best_score = saved["score"]
            print(f"[parallel] Resumed from best score: {best_score:.6f}")
    if not args.resume or not best_params_file.exists():
        best_params = BracketParams()
        print("[parallel] Evaluating baseline...")
        metrics = evaluate_params(best_params, dataset_path, args.sims, args.candidates)
        best_score = composite_score(metrics)
        print(f"[parallel] Baseline: score={best_score:.6f}, capture={metrics['capture_rate']:.4f}")
        tmp_fd, tmp_path = tempfile.mkstemp(dir=best_params_file.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump({"schema_version": AUTOBRACKET_SCHEMA_VERSION, "params": best_params.to_dict(), "score": best_score, "metrics": metrics}, f, indent=2)
            os.replace(tmp_path, best_params_file)
        except BaseException:
            os.unlink(tmp_path)
            raise

    print(f"[parallel] Workers: {n_workers}, Batch: {batch_size}, Scale: {args.scale}")
    print(f"[parallel] {args.iterations} rounds × {batch_size} mutations = {args.iterations * batch_size} total evaluations")
    print(f"[parallel] Estimated time: {args.iterations * 9 / n_workers / 60:.0f} minutes")
    print()

    # TSV log
    write_header = not log_file.exists()
    tsv = open(log_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(tsv, delimiter="\t")
    if write_header:
        writer.writerow([
            "round", "timestamp", "best_score", "round_best_score", "capture_rate",
            "small", "mid", "large", "archetypes", "champions",
            "status", "elapsed_s", "evaluated", "description"
        ])
        tsv.flush()

    total_kept = 0
    total_evaluated = 0
    total_time = 0
    stale_rounds = 0

    with mp.Pool(processes=n_workers) as pool:
        for round_num in range(1, args.iterations + 1):
            # Generate batch of mutations from current best
            mutations = []
            descriptions = []
            for i in range(batch_size):
                mutated = best_params.mutate(rng, scale=args.scale)
                changes = []
                for k in best_params.__dict__:
                    old = getattr(best_params, k)
                    new = getattr(mutated, k)
                    if abs(old - new) > 1e-6:
                        changes.append(f"{k}: {old:.3f}→{new:.3f}")
                mutations.append(mutated)
                descriptions.append("; ".join(changes))

            # Evaluate all mutations in parallel
            start = time.perf_counter()
            work_items = [
                (i, mutations[i], str(dataset_path), args.sims, args.candidates)
                for i in range(batch_size)
            ]
            results = pool.map(_evaluate_worker, work_items)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_evaluated += batch_size

            # Find the best result in this batch
            round_best_score = -1e9
            round_best_idx = -1
            round_best_metrics = {}

            for worker_id, params, metrics, score, status in results:
                if score > round_best_score:
                    round_best_score = score
                    round_best_idx = worker_id
                    round_best_metrics = metrics

            # Keep if improved
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
                    f"capture={m.get('capture_rate',0):.4f} small={m.get('scenario_small',0):.3f} "
                    f"({elapsed:.1f}s, {batch_size} evals) | {descriptions[round_best_idx][:80]}"
                )

                writer.writerow([
                    round_num, datetime.now().isoformat(timespec="seconds"),
                    f"{best_score:.6f}", f"{round_best_score:.6f}",
                    f"{m.get('capture_rate',0):.4f}",
                    f"{m.get('scenario_small',0):.4f}", f"{m.get('scenario_mid',0):.4f}",
                    f"{m.get('scenario_large',0):.4f}",
                    m.get("distinct_archetypes", ""), m.get("unique_champions", ""),
                    "KEEP", f"{elapsed:.1f}", batch_size,
                    descriptions[round_best_idx][:120],
                ])
            else:
                stale_rounds += 1
                print(
                    f"  Round {round_num:>3}: skip  best_in_batch={round_best_score:.6f} "
                    f"(need >{best_score:.6f}) ({elapsed:.1f}s, {batch_size} evals) "
                    f"[stale: {stale_rounds}]"
                )
                writer.writerow([
                    round_num, datetime.now().isoformat(timespec="seconds"),
                    f"{best_score:.6f}", f"{round_best_score:.6f}",
                    "", "", "", "", "", "",
                    "DISCARD", f"{elapsed:.1f}", batch_size, "",
                ])

            tsv.flush()

            # Early stopping: if no improvement for 30 rounds, convergence reached
            if stale_rounds >= 30:
                print(f"\n[parallel] Converged! No improvement for {stale_rounds} rounds.")
                break

    tsv.close()

    avg_time = total_time / max(args.iterations, 1)
    print(f"\n{'='*60}")
    print(f"[parallel] Optimization complete!")
    print(f"  Rounds: {round_num} ({total_evaluated} total evaluations)")
    print(f"  Kept: {total_kept}")
    print(f"  Best score: {best_score:.6f}")
    print(f"  Total time: {total_time:.0f}s ({avg_time:.1f}s/round, {total_time/total_evaluated:.1f}s/eval)")
    print(f"  Best params: {best_params_file}")
    print(f"  Log: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

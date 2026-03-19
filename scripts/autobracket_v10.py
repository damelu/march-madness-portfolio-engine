#!/usr/bin/env python3
"""
autobracket_v10.py — V10.3-native portfolio optimizer.

This script mutates only runtime/search parameters while keeping model artifacts fixed.
It evaluates candidates under the same multi-seed release contract used by final publication.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AUTOBRACKET_V10_SCHEMA_VERSION = 2
CONTEST_MODE_CHOICES = ("blended", "standard_small", "standard_mid", "standard_large")

from march_madness_2026.config import load_selection_sunday_dataset
from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.v10 import DEFAULT_V10_INFERENCE_SNAPSHOT, V10BracketPortfolioEngine
from march_madness_2026.v10.portfolio import ReleaseContractConfig
from march_madness_2026.v10.search import (
    DEFAULT_BLENDED_WEIGHT_FLOOR,
    V10SearchParams,
    apply_params_to_engine,
    resolve_release_seeds,
)


def evaluate_params(
    params: V10SearchParams,
    *,
    engine: V10BracketPortfolioEngine,
    dataset,
    contest_mode: str,
    release_seeds: list[int],
    release_contract: ReleaseContractConfig,
    blended_weight_floor: float,
) -> dict[str, object]:
    apply_params_to_engine(
        engine,
        params,
        contest_mode=contest_mode,
        blended_weight_floor=blended_weight_floor,
    )
    run_bundle = engine.run(
        dataset=dataset,
        release_seeds=release_seeds,
        release_contract=release_contract,
    )
    result = run_bundle["result"]
    selection_metadata = run_bundle["selection_metadata"]
    v10_artifacts = run_bundle["v10_artifacts"]
    portfolio_eval = v10_artifacts["portfolio_release_evaluation"]

    metrics: dict[str, object] = {
        "release_objective_score": float(portfolio_eval.release_objective_score),
        "objective_score": float(portfolio_eval.objective_score),
        "expected_utility": float(portfolio_eval.expected_utility),
        "expected_payout": float(portfolio_eval.expected_payout),
        "first_place_equity": float(portfolio_eval.first_place_equity),
        "capture_rate": float(portfolio_eval.capture_rate),
        "cash_rate": float(portfolio_eval.cash_rate),
        "top3_equity": float(portfolio_eval.top3_equity),
        "average_overlap": float(portfolio_eval.average_overlap),
        "average_duplication_proxy": float(portfolio_eval.average_duplication_proxy),
        "final_four_repeat_penalty": float(portfolio_eval.final_four_repeat_penalty),
        "region_winner_repeat_penalty": float(portfolio_eval.region_winner_repeat_penalty),
        "champion_repeat_penalty": float(portfolio_eval.champion_repeat_penalty),
        "min_individual_fpe": float(portfolio_eval.min_individual_fpe),
        "distinct_archetypes": float(portfolio_eval.distinct_archetypes),
        "unique_champions": float(portfolio_eval.unique_champions),
        "payoff_correlation": float(portfolio_eval.payoff_correlation),
        "passes_release_gates": float(not portfolio_eval.guardrail_failures),
        "guardrail_failure_count": float(len(portfolio_eval.guardrail_failures)),
        "guardrail_failures": list(portfolio_eval.guardrail_failures),
        "contest_small_weight": float(engine.simulation_profile.primary_contests.get("standard_small", 0.0)),
        "contest_mid_weight": float(engine.simulation_profile.primary_contests.get("standard_mid", 0.0)),
        "contest_large_weight": float(engine.simulation_profile.primary_contests.get("standard_large", 0.0)),
        "game_model_artifact": str(v10_artifacts.get("game_model_artifact", "")),
        "public_field_artifact": str(v10_artifacts.get("public_field_artifact", "")),
        "release_objective_name": selection_metadata.get("selection_objective", release_contract.objective_name),
        "evaluation_seed_count": float(len(release_seeds)),
        "release_evaluation_seed_count": int(len(release_seeds)),
    }

    for contest_id in ("standard_small", "standard_mid", "standard_large"):
        suffix = contest_id.split("_", 1)[1]
        scenario = result.scenario_summary.get(contest_id, {})
        metrics[f"scenario_{suffix}_fpe"] = float(scenario.get("portfolio_first_place_equity", 0.0))
        metrics[f"scenario_{suffix}_capture"] = float(scenario.get("portfolio_capture_rate", 0.0))
        metrics[f"scenario_{suffix}_payout"] = float(scenario.get("portfolio_expected_payout", 0.0))
        metrics[f"scenario_{suffix}_utility"] = float(scenario.get("portfolio_expected_utility", 0.0))
        metrics[f"scenario_{suffix}_cash"] = float(scenario.get("portfolio_cash_rate", 0.0))
        metrics[f"scenario_{suffix}_top3"] = float(scenario.get("portfolio_top3_equity", 0.0))
    return metrics


def _checkpoint_compatible(
    checkpoint: dict[str, Any],
    *,
    contest_mode: str,
    release_seeds: list[int],
    release_contract: ReleaseContractConfig,
    blended_weight_floor: float,
) -> bool:
    return (
        checkpoint.get("schema_version") == AUTOBRACKET_V10_SCHEMA_VERSION
        and checkpoint.get("contest_mode") == contest_mode
        and checkpoint.get("evaluation_seeds") == release_seeds
        and checkpoint.get("release_contract") == asdict(release_contract)
        and float(checkpoint.get("blended_weight_floor", DEFAULT_BLENDED_WEIGHT_FLOOR)) == float(blended_weight_floor)
    )


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    params: V10SearchParams,
    score: float,
    metrics: dict[str, object],
    contest_mode: str,
    release_seeds: list[int],
    release_contract: ReleaseContractConfig,
    blended_weight_floor: float,
) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=checkpoint_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as handle:
            json.dump(
                {
                    "schema_version": AUTOBRACKET_V10_SCHEMA_VERSION,
                    "release_objective_name": release_contract.objective_name,
                    "contest_mode": contest_mode,
                    "evaluation_seed_count": len(release_seeds),
                    "evaluation_seeds": release_seeds,
                    "release_contract": asdict(release_contract),
                    "blended_weight_floor": blended_weight_floor,
                    "params": params.to_dict(blended_weight_floor=blended_weight_floor),
                    "score": float(score),
                    "metrics": metrics,
                    "guardrail_failures": list(metrics.get("guardrail_failures", [])),
                },
                handle,
                indent=2,
            )
        os.replace(tmp_path, checkpoint_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V10.3-native AutoBracket optimizer")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sims", type=int, default=120)
    parser.add_argument("--candidates", type=int, default=80)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260318)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V10_INFERENCE_SNAPSHOT)
    parser.add_argument("--use-demo", action="store_true")
    parser.add_argument("--training-profile-id", default="default_v10")
    parser.add_argument("--game-model-artifact", type=Path, default=None)
    parser.add_argument("--public-field-artifact", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "outputs" / "autobracket_v10")
    parser.add_argument("--stale-rounds", type=int, default=30)
    parser.add_argument("--contest-mode", choices=CONTEST_MODE_CHOICES, default="blended")
    parser.add_argument("--release-seed-count", "--num-eval-seeds", dest="release_seed_count", type=int, default=3)
    parser.add_argument("--release-seeds", "--eval-seeds", dest="release_seeds", default="")
    parser.add_argument("--blended-weight-floor", type=float, default=DEFAULT_BLENDED_WEIGHT_FLOOR)
    parser.add_argument("--allow-naive-regression", action="store_true")
    parser.add_argument("--allow-zero-fpe-finalist", action="store_true")
    args = parser.parse_args(argv)

    if not args.use_demo and not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}")

    dataset = build_demo_dataset(seed=args.seed) if args.use_demo else load_selection_sunday_dataset(args.dataset)
    release_seeds = resolve_release_seeds(
        args.seed,
        explicit_seed_string=args.release_seeds,
        seed_count=args.release_seed_count,
    )
    release_contract = ReleaseContractConfig(
        blended_weight_floor=args.blended_weight_floor,
        fail_on_guardrail=False,
        allow_naive_regression=args.allow_naive_regression,
        allow_zero_fpe_finalist=args.allow_zero_fpe_finalist,
    )
    engine = V10BracketPortfolioEngine(
        simulation_overrides={
            "seed": args.seed,
            "num_tournament_simulations": args.sims,
            "num_candidate_brackets": args.candidates,
        },
        training_profile_id=args.training_profile_id,
        model_artifact_path=args.game_model_artifact,
        public_field_artifact_path=args.public_field_artifact,
    )

    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    best_params_file = log_dir / "best_v10_params.json"
    log_file = log_dir / "experiments_v10.tsv"

    rng = np.random.default_rng(args.seed)
    best_params = V10SearchParams()
    best_score = float("-inf")
    if args.resume and best_params_file.exists():
        with best_params_file.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if _checkpoint_compatible(
            saved,
            contest_mode=args.contest_mode,
            release_seeds=release_seeds,
            release_contract=release_contract,
            blended_weight_floor=args.blended_weight_floor,
        ):
            best_params = V10SearchParams(**saved["params"]).normalized(
                blended_weight_floor=args.blended_weight_floor
            )
            best_score = float(saved["score"])

    initial_metrics = evaluate_params(
        best_params,
        engine=engine,
        dataset=dataset,
        contest_mode=args.contest_mode,
        release_seeds=release_seeds,
        release_contract=release_contract,
        blended_weight_floor=args.blended_weight_floor,
    )
    if best_score == float("-inf"):
        best_score = float(initial_metrics["release_objective_score"])
        _save_checkpoint(
            best_params_file,
            params=best_params,
            score=best_score,
            metrics=initial_metrics,
            contest_mode=args.contest_mode,
            release_seeds=release_seeds,
            release_contract=release_contract,
            blended_weight_floor=args.blended_weight_floor,
        )

    write_header = not log_file.exists()
    with log_file.open("a", newline="") as tsv:
        writer = csv.writer(tsv, delimiter="\t")
        if write_header:
            writer.writerow(
                [
                    "round",
                    "timestamp",
                    "release_objective_score",
                    "objective_score",
                    "expected_utility",
                    "expected_payout",
                    "fpe",
                    "capture_rate",
                    "cash_rate",
                    "top3_equity",
                    "min_individual_fpe",
                    "scenario_small_fpe",
                    "scenario_mid_fpe",
                    "scenario_large_fpe",
                    "average_overlap",
                    "average_duplication_proxy",
                    "final_four_repeat_penalty",
                    "region_winner_repeat_penalty",
                    "champion_repeat_penalty",
                    "payoff_correlation",
                    "guardrail_failures",
                    "game_model_artifact",
                    "public_field_artifact",
                    "release_seed_count",
                    "status",
                    "elapsed_s",
                    "description",
                ]
            )

        stale_rounds = 0
        for round_num in range(1, args.iterations + 1):
            mutations: list[V10SearchParams] = []
            descriptions: list[str] = []
            for _ in range(args.batch):
                candidate_params = best_params.mutate(
                    rng,
                    scale=args.scale,
                    optimize_contest_weights=args.contest_mode == "blended",
                    blended_weight_floor=args.blended_weight_floor,
                )
                description = "; ".join(
                    f"{key}: {getattr(best_params, key):.3f}->{getattr(candidate_params, key):.3f}"
                    for key in best_params.__dict__
                    if abs(getattr(best_params, key) - getattr(candidate_params, key)) > 1e-9
                )
                mutations.append(candidate_params)
                descriptions.append(description)

            start = time.perf_counter()
            round_best_score = float("-inf")
            round_best_idx = -1
            round_best_metrics: dict[str, object] = {}

            for idx, candidate_params in enumerate(mutations):
                metrics = evaluate_params(
                    candidate_params,
                    engine=engine,
                    dataset=dataset,
                    contest_mode=args.contest_mode,
                    release_seeds=release_seeds,
                    release_contract=release_contract,
                    blended_weight_floor=args.blended_weight_floor,
                )
                score = float(metrics["release_objective_score"])
                if score > round_best_score:
                    round_best_score = score
                    round_best_idx = idx
                    round_best_metrics = metrics

            elapsed = time.perf_counter() - start
            status = "DISCARD"
            if round_best_score > best_score:
                best_score = round_best_score
                best_params = mutations[round_best_idx]
                status = "KEEP"
                _save_checkpoint(
                    best_params_file,
                    params=best_params,
                    score=best_score,
                    metrics=round_best_metrics,
                    contest_mode=args.contest_mode,
                    release_seeds=release_seeds,
                    release_contract=release_contract,
                    blended_weight_floor=args.blended_weight_floor,
                )
                stale_rounds = 0
            else:
                stale_rounds += 1

            writer.writerow(
                [
                    round_num,
                    datetime.now().isoformat(timespec="seconds"),
                    f"{round_best_score:.6f}",
                    f"{float(round_best_metrics.get('objective_score', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('expected_utility', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('expected_payout', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('first_place_equity', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('capture_rate', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('cash_rate', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('top3_equity', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('min_individual_fpe', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('scenario_small_fpe', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('scenario_mid_fpe', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('scenario_large_fpe', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('average_overlap', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('average_duplication_proxy', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('final_four_repeat_penalty', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('region_winner_repeat_penalty', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('champion_repeat_penalty', 0.0)):.6f}",
                    f"{float(round_best_metrics.get('payoff_correlation', 0.0)):.6f}",
                    ",".join(round_best_metrics.get("guardrail_failures", [])),
                    str(round_best_metrics.get("game_model_artifact", "")),
                    str(round_best_metrics.get("public_field_artifact", "")),
                    int(round_best_metrics.get("release_evaluation_seed_count", len(release_seeds))),
                    status,
                    f"{elapsed:.2f}",
                    descriptions[round_best_idx] if round_best_idx >= 0 else "",
                ]
            )
            tsv.flush()
            print(
                f"[autobracket-v10] round={round_num} status={status} "
                f"score={round_best_score:.6f} utility={float(round_best_metrics.get('expected_utility', 0.0)):.4f} "
                f"payout={float(round_best_metrics.get('expected_payout', 0.0)):.2f} "
                f"fpe={float(round_best_metrics.get('first_place_equity', 0.0)):.4f} "
                f"capture={float(round_best_metrics.get('capture_rate', 0.0)):.4f} "
                f"guards={','.join(round_best_metrics.get('guardrail_failures', [])) or 'pass'} "
                f"contest_mode={args.contest_mode} seeds={len(release_seeds)}"
            )

            if stale_rounds >= args.stale_rounds:
                print(f"[autobracket-v10] converged after {stale_rounds} stale rounds")
                break

    print(f"[autobracket-v10] best_params={best_params_file}")
    print(f"[autobracket-v10] log={log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

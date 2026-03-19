#!/usr/bin/env python3
"""
autobracket.py — Autonomous bracket parameter optimizer.

Inspired by karpathy/autoresearch: modify parameters → run evaluation → keep/revert.
Optimizes the tunable parameters in the bracket portfolio engine to maximize
first-place equity across pool-size scenarios.

Usage:
    uv run python scripts/autobracket.py                    # run 50 iterations
    uv run python scripts/autobracket.py --iterations 200   # run 200 iterations
    uv run python scripts/autobracket.py --resume           # resume from last best
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AUTOBRACKET_SCHEMA_VERSION = 3

from march_madness_2026.config import (
    load_contest_profiles,
    load_scoring_profiles,
    load_selection_sunday_dataset,
    load_simulation_profile,
)
from march_madness_2026.engine import BracketPortfolioEngine
from march_madness_2026.tournament import ARCHETYPE_PROFILES, ArchetypeProfile


# ---------------------------------------------------------------------------
# Tunable parameter space
# ---------------------------------------------------------------------------

@dataclass
class BracketParams:
    """All tunable parameters for the bracket portfolio engine."""

    # _compute_strength weights
    market_weight: float = 1.0
    coaching_weight: float = 0.75
    continuity_weight: float = 0.55
    offense_weight: float = 0.25
    defense_weight: float = 0.20
    tempo_weight: float = 0.10
    injury_weight: float = 1.30

    # win_probability parameters
    sigmoid_scale: float = 6.75

    # Archetype temperatures (5 archetypes)
    temp_high_confidence: float = 0.72
    temp_balanced: float = 0.95
    temp_selective_contrarian: float = 1.12
    temp_underdog_upside: float = 1.28
    temp_high_risk: float = 1.45

    # Archetype favorite biases
    bias_high_confidence: float = 0.16
    bias_balanced: float = 0.04
    bias_selective_contrarian: float = -0.08
    bias_underdog_upside: float = -0.16
    bias_high_risk: float = -0.26

    # Portfolio selection weights
    overlap_penalty: float = 0.18
    champion_penalty: float = 0.05

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.__dict__.items()}

    def mutate(self, rng: np.random.Generator, scale: float = 0.1) -> "BracketParams":
        """Create a mutated copy with gaussian perturbations."""
        new = copy.copy(self)
        fields = list(self.__dict__.keys())
        # Mutate 1-4 parameters at a time
        num_mutations = rng.integers(1, min(5, len(fields) + 1))
        targets = rng.choice(len(fields), size=num_mutations, replace=False)

        for idx in targets:
            name = fields[idx]
            current = getattr(new, name)

            # Scale perturbation relative to parameter magnitude
            if abs(current) < 0.01:
                delta = rng.normal(0, 0.05 * scale)
            else:
                delta = rng.normal(0, abs(current) * scale)

            new_val = current + delta

            # Apply bounds
            if name.startswith("temp_"):
                new_val = max(0.3, min(2.5, new_val))
            elif name.startswith("bias_"):
                new_val = max(-0.5, min(0.5, new_val))
            elif name.endswith("_weight"):
                new_val = max(0.0, min(3.0, new_val))
            elif name == "sigmoid_scale":
                new_val = max(2.0, min(15.0, new_val))
            elif name.endswith("_penalty"):
                new_val = max(0.0, min(1.0, new_val))

            setattr(new, name, round(new_val, 6))

        return new


# ---------------------------------------------------------------------------
# Monkey-patching to apply params to the engine
# ---------------------------------------------------------------------------

def apply_params_to_engine(params: BracketParams) -> None:
    """Monkey-patch the tournament model and archetype profiles with new params."""
    import march_madness_2026.tournament as tournament_mod

    # Patch _compute_strength
    @staticmethod
    def patched_compute_strength(team) -> float:
        return (
            team.rating
            + params.market_weight * team.market_adjustment
            + params.coaching_weight * team.coaching_adjustment
            + params.continuity_weight * team.continuity_adjustment
            + params.offense_weight * team.offense_rating
            + params.defense_weight * team.defense_rating
            + params.tempo_weight * team.tempo_adjustment
            - params.injury_weight * team.injury_penalty
        )

    tournament_mod.TournamentModel._compute_strength = patched_compute_strength

    # Patch win_probability sigmoid scale
    original_win_prob = tournament_mod.TournamentModel.win_probability

    def patched_win_probability(self, team_a_index: int, team_b_index: int) -> float:
        strength_gap = float(self.strength_by_index[team_a_index] - self.strength_by_index[team_b_index])
        base_probability = 1.0 / (1.0 + math.exp(-strength_gap / params.sigmoid_scale))
        volatility = min(
            0.35,
            float(self.volatility_by_index[team_a_index] + self.volatility_by_index[team_b_index]) / 2.0,
        )
        # V6 matchup adjustments (must match production tournament.py)
        team_a = self.teams[team_a_index]
        team_b = self.teams[team_b_index]
        tempo_diff = abs(team_a.tempo_adjustment - team_b.tempo_adjustment)
        if tempo_diff > 0.3:
            slower_is_a = team_a.tempo_adjustment < team_b.tempo_adjustment
            tempo_bonus = 0.015 * tempo_diff
            base_probability += tempo_bonus if slower_is_a else -tempo_bonus
        three_pt_var = abs(team_a.three_point_rate) + abs(team_b.three_point_rate)
        volatility += three_pt_var * 0.03
        ft_advantage = team_a.free_throw_rate - team_b.free_throw_rate
        base_probability += ft_advantage * 0.01
        base_probability = min(max(base_probability, 0.005), 0.995)
        volatility = min(0.35, volatility)
        # Nonlinear volatility dampening (must match production)
        extremity = abs(base_probability - 0.5) * 2.0
        dampened_volatility = volatility * (1.0 - extremity * 0.85)
        return 0.5 + (base_probability - 0.5) * (1.0 - max(0.0, dampened_volatility))

    tournament_mod.TournamentModel.win_probability = patched_win_probability

    # Patch archetype profiles
    tournament_mod.ARCHETYPE_PROFILES["high_confidence"] = ArchetypeProfile(
        name="High Confidence", risk_label="low",
        temperature=params.temp_high_confidence,
        favorite_bias=params.bias_high_confidence,
        late_round_favorite_bias=0.10,
    )
    tournament_mod.ARCHETYPE_PROFILES["balanced"] = ArchetypeProfile(
        name="Balanced", risk_label="medium",
        temperature=params.temp_balanced,
        favorite_bias=params.bias_balanced,
        late_round_favorite_bias=0.05,
    )
    tournament_mod.ARCHETYPE_PROFILES["selective_contrarian"] = ArchetypeProfile(
        name="Selective Contrarian", risk_label="medium_high",
        temperature=params.temp_selective_contrarian,
        favorite_bias=params.bias_selective_contrarian,
        late_round_favorite_bias=0.02,
    )
    tournament_mod.ARCHETYPE_PROFILES["underdog_upside"] = ArchetypeProfile(
        name="Underdog Upside", risk_label="high",
        temperature=params.temp_underdog_upside,
        favorite_bias=params.bias_underdog_upside,
        late_round_favorite_bias=-0.02,
    )
    tournament_mod.ARCHETYPE_PROFILES["high_risk_high_return"] = ArchetypeProfile(
        name="High Risk High Return", risk_label="very_high",
        temperature=params.temp_high_risk,
        favorite_bias=params.bias_high_risk,
        late_round_favorite_bias=-0.08,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_params(params: BracketParams, dataset_path: Path, sims: int = 5000, candidates: int = 300) -> dict:
    """Run the portfolio engine with given params and return metrics."""
    apply_params_to_engine(params)

    sim_profile = load_simulation_profile()
    sim_profile = sim_profile.model_copy(update={
        "num_tournament_simulations": sims,
        "num_candidate_brackets": candidates,
    })

    dataset = load_selection_sunday_dataset(dataset_path)
    engine = BracketPortfolioEngine(simulation_overrides={
        "num_tournament_simulations": sims,
        "num_candidate_brackets": candidates,
    })
    run_bundle = engine.run(dataset=dataset)
    result = run_bundle["result"]

    fpe = getattr(result.summary, "weighted_portfolio_first_place_equity", None)
    if fpe is None:
        fpe = result.summary.weighted_portfolio_capture_rate

    return {
        "capture_rate": result.summary.weighted_portfolio_capture_rate,
        "naive_capture": result.summary.naive_baseline_capture_rate,
        "objective": result.summary.weighted_portfolio_objective,
        "first_place_equity": fpe,
        "overlap": result.summary.average_pairwise_overlap,
        "distinct_archetypes": result.summary.distinct_archetypes,
        "unique_champions": result.summary.unique_champions,
        "scenario_small": result.scenario_summary.get("standard_small", {}).get("portfolio_capture_rate", 0),
        "scenario_mid": result.scenario_summary.get("standard_mid", {}).get("portfolio_capture_rate", 0),
        "scenario_large": result.scenario_summary.get("standard_large", {}).get("portfolio_capture_rate", 0),
        "scenario_small_fpe": result.scenario_summary.get("standard_small", {}).get("portfolio_first_place_equity", 0),
        "scenario_mid_fpe": result.scenario_summary.get("standard_mid", {}).get("portfolio_first_place_equity", 0),
        "scenario_large_fpe": result.scenario_summary.get("standard_large", {}).get("portfolio_first_place_equity", 0),
    }


def composite_score(metrics: dict) -> float:
    """Single scalar objective combining all metrics — FPE-first."""
    return (
        0.55 * metrics["first_place_equity"]
        + 0.20 * metrics.get("scenario_small_fpe", metrics.get("scenario_small", 0))
        + 0.15 * metrics.get("scenario_mid_fpe", metrics.get("scenario_mid", 0))
        + 0.10 * metrics.get("scenario_large_fpe", metrics.get("scenario_large", 0))
        + 0.02 * (metrics["distinct_archetypes"] / 5.0)
        + 0.02 * (metrics["unique_champions"] / 5.0)
        - 0.03 * max(0.0, metrics["overlap"] - 0.5)
    )


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AutoBracket: autonomous parameter optimizer")
    parser.add_argument("--iterations", type=int, default=50, help="Number of optimization iterations")
    parser.add_argument("--sims", type=int, default=5000, help="Tournament simulations per evaluation")
    parser.add_argument("--candidates", type=int, default=300, help="Candidate brackets per evaluation")
    parser.add_argument("--scale", type=float, default=0.15, help="Mutation scale (0.05=conservative, 0.3=aggressive)")
    parser.add_argument("--resume", action="store_true", help="Resume from last best params")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset JSON path")
    args = parser.parse_args()

    dataset_path = args.dataset or PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

    log_dir = PROJECT_ROOT / "outputs" / "autobracket"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiments.tsv"
    best_params_file = log_dir / "best_params.json"

    rng = np.random.default_rng(42)

    # Initialize or resume
    if args.resume and best_params_file.exists():
        with best_params_file.open() as f:
            saved = json.load(f)
        if saved.get("schema_version") != AUTOBRACKET_SCHEMA_VERSION:
            print(f"[autobracket] Stale checkpoint (schema_version={saved.get('schema_version')}), starting from baseline")
            args.resume = False
        else:
            best_params = BracketParams(**saved["params"])
            best_score = saved["score"]
            print(f"[autobracket] Resumed from best score: {best_score:.6f}")
    if not args.resume or not best_params_file.exists():
        best_params = BracketParams()
        print("[autobracket] Evaluating baseline params...")
        start = time.perf_counter()
        baseline_metrics = evaluate_params(best_params, dataset_path, args.sims, args.candidates)
        elapsed = time.perf_counter() - start
        best_score = composite_score(baseline_metrics)
        print(f"[autobracket] Baseline: score={best_score:.6f}, capture={baseline_metrics['capture_rate']:.4f} ({elapsed:.1f}s)")

        # Save baseline (atomic write)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=best_params_file.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump({"schema_version": AUTOBRACKET_SCHEMA_VERSION, "params": best_params.to_dict(), "score": best_score, "metrics": baseline_metrics}, f, indent=2)
            os.replace(tmp_path, best_params_file)
        except BaseException:
            os.unlink(tmp_path)
            raise

    # Write TSV header if new file
    write_header = not log_file.exists()
    tsv = open(log_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(tsv, delimiter="\t")
    if write_header:
        writer.writerow([
            "iteration", "timestamp", "score", "capture_rate", "small", "mid", "large",
            "overlap", "archetypes", "champions", "status", "elapsed_s", "description"
        ])
        tsv.flush()

    kept = 0
    total_time = 0

    for iteration in range(1, args.iterations + 1):
        # Mutate current best
        candidate_params = best_params.mutate(rng, scale=args.scale)

        # Find which params changed
        changes = []
        for k in best_params.__dict__:
            old = getattr(best_params, k)
            new = getattr(candidate_params, k)
            if abs(old - new) > 1e-6:
                changes.append(f"{k}: {old:.4f} → {new:.4f}")
        description = "; ".join(changes)

        print(f"\n[autobracket] Iteration {iteration}/{args.iterations}: {description}")

        start = time.perf_counter()
        try:
            metrics = evaluate_params(candidate_params, dataset_path, args.sims, args.candidates)
            score = composite_score(metrics)
            elapsed = time.perf_counter() - start
            total_time += elapsed

            if score > best_score:
                improvement = score - best_score
                best_score = score
                best_params = candidate_params
                status = "KEEP"
                kept += 1

                # Save new best (atomic write)
                tmp_fd, tmp_path = tempfile.mkstemp(dir=best_params_file.parent, suffix=".tmp")
                try:
                    with os.fdopen(tmp_fd, "w") as f:
                        json.dump({"schema_version": AUTOBRACKET_SCHEMA_VERSION, "params": best_params.to_dict(), "score": best_score, "metrics": metrics}, f, indent=2)
                    os.replace(tmp_path, best_params_file)
                except BaseException:
                    os.unlink(tmp_path)
                    raise

                print(f"  KEEP: score={score:.6f} (+{improvement:.6f}), capture={metrics['capture_rate']:.4f} ({elapsed:.1f}s)")
            else:
                status = "DISCARD"
                print(f"  DISCARD: score={score:.6f} (best={best_score:.6f}) ({elapsed:.1f}s)")

            writer.writerow([
                iteration,
                datetime.now().isoformat(timespec="seconds"),
                f"{score:.6f}",
                f"{metrics['capture_rate']:.4f}",
                f"{metrics['scenario_small']:.4f}",
                f"{metrics['scenario_mid']:.4f}",
                f"{metrics['scenario_large']:.4f}",
                f"{metrics['overlap']:.4f}",
                metrics["distinct_archetypes"],
                metrics["unique_champions"],
                status,
                f"{elapsed:.1f}",
                description,
            ])
            tsv.flush()

        except Exception as e:
            elapsed = time.perf_counter() - start
            total_time += elapsed
            print(f"  CRASH: {e} ({elapsed:.1f}s)")
            writer.writerow([
                iteration, datetime.now().isoformat(timespec="seconds"),
                "", "", "", "", "", "", "", "",
                "CRASH", f"{elapsed:.1f}", f"{description} | ERROR: {e}",
            ])
            tsv.flush()

    tsv.close()

    print(f"\n{'='*60}")
    print(f"[autobracket] Optimization complete!")
    print(f"  Iterations: {args.iterations}")
    print(f"  Kept: {kept}/{args.iterations} ({kept/max(args.iterations,1)*100:.0f}%)")
    print(f"  Best score: {best_score:.6f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/max(args.iterations,1):.1f}s/iter)")
    print(f"  Best params: {best_params_file}")
    print(f"  Experiment log: {log_file}")
    print(f"\nTo generate brackets with optimized params:")
    print(f"  cat {best_params_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

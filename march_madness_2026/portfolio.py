from __future__ import annotations

import math
import os
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from .gpu import (
    GPU_AVAILABLE,
    evaluate_candidates_gpu,
    overlap_matrix_gpu,
    portfolio_capture_gpu,
    portfolio_fpe_gpu,
    to_device,
    to_numpy,
)
from .models import BracketCandidate, CandidateContestMetrics, SimulationProfile

ZERO_EQUITY_EPSILON = 1e-12


def _portfolio_gpu_enabled() -> bool:
    mode = os.environ.get("MARCH_MADNESS_PORTFOLIO_BACKEND", os.environ.get("MARCH_MADNESS_PAYOUT_BACKEND", "auto"))
    mode = mode.strip().lower()
    if mode in {"cpu", "numpy", "off", "disabled"}:
        return False
    return GPU_AVAILABLE


def evaluate_candidates_for_contest(
    candidates: Sequence[BracketCandidate],
    contest_id: str,
    candidate_scores: np.ndarray,
    opponent_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Use GPU-accelerated path when available
    if _portfolio_gpu_enabled():
        fpe_arr, avg_arr = evaluate_candidates_gpu(candidate_scores, opponent_scores)
        top_decile_cutoff = max(1, int(math.ceil(opponent_scores.shape[0] * 0.10)))
        for index, candidate in enumerate(candidates):
            better_count = (opponent_scores > candidate_scores[index]).sum(axis=0)
            top_decile_rate = float(np.mean((better_count + 1) <= top_decile_cutoff))
            candidate.contest_metrics[contest_id] = CandidateContestMetrics(
                contest_id=contest_id,
                first_place_equity=float(fpe_arr[index]),
                average_finish=float(avg_arr[index]),
                top_decile_rate=top_decile_rate,
            )
        return fpe_arr, avg_arr

    # CPU fallback
    first_place_equities = np.zeros(len(candidates), dtype=np.float32)
    average_finishes = np.zeros(len(candidates), dtype=np.float32)
    max_other_scores = opponent_scores.max(axis=0)
    top_decile_cutoff = max(1, int(math.ceil(opponent_scores.shape[0] * 0.10)))

    for index, candidate in enumerate(candidates):
        candidate_score = candidate_scores[index]
        better_count = (opponent_scores > candidate_score).sum(axis=0)
        tied_count = (opponent_scores == candidate_score).sum(axis=0)
        outright_wins = candidate_score > max_other_scores
        tied_for_first = candidate_score == max_other_scores
        shared_win = np.where(
            outright_wins,
            1.0,
            np.where(tied_for_first, 1.0 / (tied_count + 1.0), 0.0),
        )

        first_place_equity = float(shared_win.mean())
        average_finish = float((1.0 + better_count + (0.5 * tied_count)).mean())
        top_decile_rate = float(np.mean((better_count + 1) <= top_decile_cutoff))

        candidate.contest_metrics[contest_id] = CandidateContestMetrics(
            contest_id=contest_id,
            first_place_equity=first_place_equity,
            average_finish=average_finish,
            top_decile_rate=top_decile_rate,
        )
        first_place_equities[index] = first_place_equity
        average_finishes[index] = average_finish

    return first_place_equities, average_finishes


def build_overlap_matrix(bracket_picks: np.ndarray) -> np.ndarray:
    if _portfolio_gpu_enabled():
        return overlap_matrix_gpu(bracket_picks)
    num_games = bracket_picks.shape[1]
    overlap = (bracket_picks[:, None, :] == bracket_picks[None, :, :]).sum(axis=2).astype(np.float32) / num_games
    np.fill_diagonal(overlap, 1.0)
    return overlap


def portfolio_first_place_equity(
    selected_indices: Sequence[int],
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    contest_weights: Dict[str, float],
) -> float:
    if not selected_indices:
        return 0.0
    weighted_total = 0.0
    for contest_id, weight in contest_weights.items():
        selected_scores = candidate_scores_by_contest[contest_id][list(selected_indices)]
        opponent_scores = opponent_scores_by_contest[contest_id]
        if _portfolio_gpu_enabled():
            weighted_total += weight * portfolio_fpe_gpu(selected_scores, opponent_scores)
            continue
        max_selected = selected_scores.max(axis=0)
        max_opponent = opponent_scores.max(axis=0)
        selected_ties = (selected_scores == max_selected[np.newaxis, :]).sum(axis=0)
        opponent_ties = (opponent_scores == max_selected[np.newaxis, :]).sum(axis=0)
        portfolio_share = np.where(
            max_selected > max_opponent,
            1.0,
            np.where(
                max_selected == max_opponent,
                selected_ties / np.maximum(selected_ties + opponent_ties, 1),
                0.0,
            ),
        )
        weighted_total += weight * float(portfolio_share.mean())
    return weighted_total


def portfolio_capture_rate(
    selected_indices: Sequence[int],
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    contest_weights: Dict[str, float],
) -> float:
    """Diagnostic only: probability at least one selected bracket ties or beats field."""
    if not selected_indices:
        return 0.0

    weighted_total = 0.0
    for contest_id, weight in contest_weights.items():
        selected_scores = candidate_scores_by_contest[contest_id][list(selected_indices)]
        opponent_scores = opponent_scores_by_contest[contest_id]
        if _portfolio_gpu_enabled():
            weighted_total += weight * portfolio_capture_gpu(selected_scores, opponent_scores)
            continue
        max_selected_score = selected_scores.max(axis=0)
        max_other_score = opponent_scores.max(axis=0)
        weighted_total += weight * float(np.mean(max_selected_score >= max_other_score))

    return weighted_total


def average_pairwise_overlap(selected_indices: Sequence[int], overlap_matrix: np.ndarray) -> float:
    if len(selected_indices) < 2:
        return 0.0

    overlap_values: List[float] = []
    for offset, left in enumerate(selected_indices):
        for right in selected_indices[offset + 1 :]:
            overlap_values.append(float(overlap_matrix[left, right]))
    return float(sum(overlap_values) / len(overlap_values))


def _champion_repeat_penalty(selected_indices: Sequence[int], candidates: Sequence[BracketCandidate]) -> float:
    champions = [candidates[index].champion_team_id for index in selected_indices]
    if len(champions) < 2:
        return 0.0
    return (len(champions) - len(set(champions))) / len(champions)


def _can_still_hit_archetype_floor(
    selected_indices: Sequence[int],
    candidates: Sequence[BracketCandidate],
    simulation_profile: SimulationProfile,
) -> bool:
    selected_archetypes = {candidates[index].archetype for index in selected_indices}
    remaining_slots = simulation_profile.portfolio_size - len(selected_indices)
    available_new_archetypes = {
        candidate.archetype
        for index, candidate in enumerate(candidates)
        if index not in selected_indices and candidate.archetype not in selected_archetypes
    }
    max_distinct_archetypes = len(selected_archetypes) + min(remaining_slots, len(available_new_archetypes))
    return max_distinct_archetypes >= simulation_profile.min_distinct_archetypes


def _pool_supports_constraints(candidate_indices, candidates, simulation_profile) -> bool:
    if len(candidate_indices) < simulation_profile.portfolio_size:
        return False
    counts = Counter(candidates[i].archetype for i in candidate_indices)
    distinct = len(counts)
    max_fill = sum(min(count, simulation_profile.max_per_archetype) for count in counts.values())
    return (
        distinct >= simulation_profile.min_distinct_archetypes
        and max_fill >= simulation_profile.portfolio_size
    )


def _select_naive_baseline(
    candidates: Sequence[BracketCandidate],
    simulation_profile: SimulationProfile,
    pool_indices: Sequence[int] | None = None,
) -> List[int]:
    search_space = list(pool_indices) if pool_indices is not None else list(range(len(candidates)))
    ranked_indices = sorted(
        search_space,
        key=lambda index: candidates[index].weighted_first_place_equity,
        reverse=True,
    )
    baseline: List[int] = []

    for candidate_index in ranked_indices:
        candidate = candidates[candidate_index]
        if (
            sum(1 for index in baseline if candidates[index].archetype == candidate.archetype)
            >= simulation_profile.max_per_archetype
        ):
            continue

        trial_selection = baseline + [candidate_index]
        if not _can_still_hit_archetype_floor(trial_selection, candidates, simulation_profile):
            continue

        baseline.append(candidate_index)
        if len(baseline) == simulation_profile.portfolio_size:
            break

    if len(baseline) != simulation_profile.portfolio_size:
        raise RuntimeError("unable to build a feasible naive baseline portfolio")

    return baseline


def _selection_rank(
    selected_indices: Sequence[int],
    candidates: Sequence[BracketCandidate],
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    contest_weights: Dict[str, float],
    overlap_matrix: np.ndarray,
) -> tuple:
    fpe = portfolio_first_place_equity(selected_indices, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights)
    capture = portfolio_capture_rate(selected_indices, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights)
    unique_champs = len({candidates[i].champion_team_id for i in selected_indices})
    avg_overlap = average_pairwise_overlap(selected_indices, overlap_matrix)
    champ_pen = _champion_repeat_penalty(selected_indices, candidates)
    individual_equities = [candidates[i].weighted_first_place_equity for i in selected_indices]
    avg_eq = float(np.mean(individual_equities)) if individual_equities else 0.0
    floor_eq = float(min(individual_equities)) if individual_equities else 0.0
    return (fpe, capture, unique_champs, -avg_overlap, -champ_pen, avg_eq, floor_eq, tuple(-i for i in sorted(selected_indices)))


def _refine_with_local_search(
    selected: List[int],
    candidates: Sequence[BracketCandidate],
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    contest_weights: Dict[str, float],
    overlap_matrix: np.ndarray,
    simulation_profile: SimulationProfile,
) -> tuple[List[int], int]:
    swap_count = 0
    improved = True
    while improved:
        improved = False
        current_rank = _selection_rank(selected, candidates, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights, overlap_matrix)
        best_swap = None
        for out_pos, out_idx in enumerate(selected):
            remaining = [s for s in selected if s != out_idx]
            for cand_idx in range(len(candidates)):
                if cand_idx in selected:
                    continue
                # Check constraints
                arch_count = sum(1 for i in remaining if candidates[i].archetype == candidates[cand_idx].archetype)
                if arch_count >= simulation_profile.max_per_archetype:
                    continue
                if remaining and max(float(overlap_matrix[cand_idx, s]) for s in remaining) > 0.85:
                    continue
                trial = remaining + [cand_idx]
                trial_rank = _selection_rank(trial, candidates, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights, overlap_matrix)
                if trial_rank > current_rank:
                    if best_swap is None or trial_rank > best_swap[0]:
                        best_swap = (trial_rank, out_idx, cand_idx, trial)
        if best_swap is not None:
            _, _, _, selected = best_swap
            swap_count += 1
            improved = True
    return selected, swap_count


def select_portfolio(
    candidates: Sequence[BracketCandidate],
    bracket_picks: np.ndarray,
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    simulation_profile: SimulationProfile,
) -> tuple[List[int], List[int], float, float, float, float, dict]:
    overlap_matrix = build_overlap_matrix(bracket_picks)
    contest_weights = simulation_profile.primary_contests

    # Filter candidate pool to positive-equity candidates
    positive_indices = [i for i, c in enumerate(candidates) if c.weighted_first_place_equity > ZERO_EQUITY_EPSILON]
    used_zero_equity_fallback = False
    zero_equity_fallback_reason = None

    # Check if positive_indices can satisfy constraints using proper feasibility check
    if not _pool_supports_constraints(positive_indices, candidates, simulation_profile):
        used_zero_equity_fallback = True
        positive_archetypes = {candidates[i].archetype for i in positive_indices}
        if len(positive_indices) < simulation_profile.portfolio_size:
            zero_equity_fallback_reason = f"only {len(positive_indices)} candidates with positive equity, need {simulation_profile.portfolio_size}"
        else:
            zero_equity_fallback_reason = f"positive pool cannot satisfy archetype constraints (distinct={len(positive_archetypes)}, need={simulation_profile.min_distinct_archetypes})"
        pool_indices = list(range(len(candidates)))
    else:
        pool_indices = positive_indices

    # Build naive baseline from the SAME eligible pool as the optimizer
    naive_top_five = _select_naive_baseline(candidates, simulation_profile, pool_indices=pool_indices)

    def _greedy_objective(selected_indices: Sequence[int]) -> tuple:
        return _selection_rank(
            selected_indices, candidates, candidate_scores_by_contest,
            opponent_scores_by_contest, contest_weights, overlap_matrix,
        )

    selected: List[int] = []

    for _ in range(simulation_profile.portfolio_size):
        selected_archetypes = {candidates[index].archetype for index in selected}
        needed_new_archetypes = max(
            0,
            simulation_profile.min_distinct_archetypes - len(selected_archetypes),
        )
        remaining_slots = simulation_profile.portfolio_size - len(selected)
        require_new_archetype = needed_new_archetypes >= remaining_slots

        best_index = None
        best_rank = None

        for candidate_index in pool_indices:
            if candidate_index in selected:
                continue
            candidate = candidates[candidate_index]
            if sum(1 for index in selected if candidates[index].archetype == candidate.archetype) >= simulation_profile.max_per_archetype:
                continue
            if require_new_archetype and candidate.archetype in selected_archetypes:
                continue
            # Hard overlap cap: reject candidates with >0.85 overlap with any selected bracket
            if selected and max(float(overlap_matrix[candidate_index, s]) for s in selected) > 0.85:
                continue

            trial_selection = selected + [candidate_index]
            trial_rank = _greedy_objective(trial_selection)

            if best_rank is None or trial_rank > best_rank:
                best_rank = trial_rank
                best_index = candidate_index

        if best_index is None and require_new_archetype:
            require_new_archetype = False
            for candidate_index in pool_indices:
                if candidate_index in selected:
                    continue
                candidate = candidates[candidate_index]
                if sum(1 for index in selected if candidates[index].archetype == candidate.archetype) >= simulation_profile.max_per_archetype:
                    continue
                # Hard overlap cap also applies in fallback
                if selected and max(float(overlap_matrix[candidate_index, s]) for s in selected) > 0.85:
                    continue
                trial_selection = selected + [candidate_index]
                trial_rank = _greedy_objective(trial_selection)
                if best_rank is None or trial_rank > best_rank:
                    best_rank = trial_rank
                    best_index = candidate_index

        if best_index is None:
            raise RuntimeError("unable to complete a 5-bracket portfolio with the current constraints")

        selected.append(best_index)

    greedy_selected = list(selected)

    # Local search refinement
    selected, swap_count = _refine_with_local_search(
        selected, candidates, candidate_scores_by_contest,
        opponent_scores_by_contest, contest_weights, overlap_matrix,
        simulation_profile,
    )

    portfolio_fpe = portfolio_first_place_equity(
        selected, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights,
    )
    naive_fpe = portfolio_first_place_equity(
        naive_top_five, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights,
    )
    portfolio_capture = portfolio_capture_rate(
        selected, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights,
    )
    naive_capture = portfolio_capture_rate(
        naive_top_five, candidate_scores_by_contest, opponent_scores_by_contest, contest_weights,
    )

    selection_metadata = {
        "positive_equity_pool_size": len(positive_indices),
        "used_zero_equity_fallback": used_zero_equity_fallback,
        "zero_equity_fallback_reason": zero_equity_fallback_reason,
        "initial_greedy_indices": list(greedy_selected),
        "local_search_swaps": swap_count,
        "local_search_applied": swap_count > 0,
        "candidate_pool_size": len(pool_indices),
        "candidate_pool_indices": list(pool_indices),
        "naive_baseline_pool_size": len(pool_indices),
    }

    return (
        selected, naive_top_five,
        portfolio_fpe, naive_fpe,
        portfolio_capture, naive_capture,
        selection_metadata,
    )

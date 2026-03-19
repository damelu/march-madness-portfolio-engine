from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..gpu import (
    GPU_AVAILABLE,
    portfolio_capture_gpu,
    portfolio_fpe_gpu,
    portfolio_payout_summary_gpu,
    to_device,
)
from ..models import ContestPayoutProfile, PortfolioPayoutSummary
from ..payout import payout_gpu_enabled, portfolio_expected_utility
from ..portfolio import (
    ZERO_EQUITY_EPSILON,
    average_pairwise_overlap,
    build_overlap_matrix,
    portfolio_capture_rate,
    portfolio_first_place_equity,
)

PRACTICAL_ZERO_FPE_EPSILON = 1e-6


@dataclass
class ReleaseContractConfig:
    objective_name: str = "v10_6_release_objective"
    practical_zero_fpe_floor: float = PRACTICAL_ZERO_FPE_EPSILON
    blended_weight_floor: float = 0.10
    fpe_tolerance: float = 0.0
    payout_tolerance: float = 0.0
    capture_tolerance: float = 0.0
    fail_on_guardrail: bool = True
    allow_naive_regression: bool = False
    allow_zero_fpe_finalist: bool = False


@dataclass
class PortfolioReleaseEvaluation:
    selected_indices: tuple[int, ...]
    objective_score: float
    release_objective_score: float
    expected_utility: float
    expected_payout: float
    first_place_equity: float
    capture_rate: float
    cash_rate: float
    top3_equity: float
    average_overlap: float
    average_duplication_proxy: float
    final_four_repeat_penalty: float
    region_winner_repeat_penalty: float
    champion_repeat_penalty: float
    distinct_archetypes: int
    unique_champions: int
    min_individual_fpe: float
    average_individual_fpe: float
    payoff_correlation: float
    used_zero_equity_fallback: bool = False
    guardrail_failures: list[str] = field(default_factory=list)
    payout_summary: PortfolioPayoutSummary = field(default_factory=PortfolioPayoutSummary)
    rank: tuple = field(default_factory=tuple)


def _selection_key(selected_indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in selected_indices))


def _selection_cache_key(
    selected_indices: Sequence[int],
    *,
    used_zero_equity_fallback: bool,
    naive_evaluation: PortfolioReleaseEvaluation | None,
) -> tuple[Any, ...]:
    return (
        _selection_key(selected_indices),
        bool(used_zero_equity_fallback),
        tuple(naive_evaluation.selected_indices) if naive_evaluation is not None else (),
    )


def weighted_payout_summary(
    selected_indices: Sequence[int],
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    contest_weights: Mapping[str, float],
    contest_payout_profiles: Mapping[str, ContestPayoutProfile],
    contest_device_scores: Mapping[str, tuple[Any, Any]] | None = None,
) -> PortfolioPayoutSummary:
    aggregate = PortfolioPayoutSummary()
    total_weight = 0.0
    for contest_id, weight in contest_weights.items():
        payout_profile = contest_payout_profiles.get(contest_id)
        if payout_profile is None:
            continue
        if contest_device_scores and contest_id in contest_device_scores:
            candidate_device, opponent_device = contest_device_scores[contest_id]
            selected_scores = candidate_device[list(selected_indices)]
            opponent_scores = opponent_device
            payout_curve = {int(rank): float(value) for rank, value in payout_profile.payout_curve.items()}
            if payout_curve and sum(payout_curve.values()) <= 1.0 + 1e-9:
                total_pool = max(int(selected_scores.shape[0]) + int(opponent_scores.shape[0]), 1) * payout_profile.entry_fee
                payout_curve = {rank: share * total_pool for rank, share in payout_curve.items()}
            summary = PortfolioPayoutSummary(
                **portfolio_payout_summary_gpu(
                    selected_scores,
                    opponent_scores,
                    payout_curve,
                )
            )
        else:
            selected_scores = candidate_scores_by_contest[contest_id][list(selected_indices)]
            opponent_scores = opponent_scores_by_contest[contest_id]
            summary = portfolio_expected_utility(selected_scores, opponent_scores, payout_profile)
        total_weight += weight
        aggregate.expected_payout += weight * summary.expected_payout
        aggregate.cash_rate += weight * summary.cash_rate
        aggregate.top1_equity += weight * summary.top1_equity
        aggregate.top3_equity += weight * summary.top3_equity
        aggregate.top5_equity += weight * summary.top5_equity
        aggregate.downside_risk += weight * summary.downside_risk
        aggregate.payoff_correlation += weight * summary.payoff_correlation
        aggregate.expected_utility += weight * summary.expected_utility

    if total_weight > 0:
        for key, value in asdict(aggregate).items():
            setattr(aggregate, key, value / total_weight)
    return aggregate


def _can_still_hit_archetype_floor(selected_indices, candidates, simulation_profile) -> bool:
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
    counts = {}
    for index in candidate_indices:
        archetype = candidates[index].archetype
        counts[archetype] = counts.get(archetype, 0) + 1
    distinct = len(counts)
    max_fill = sum(min(count, simulation_profile.max_per_archetype) for count in counts.values())
    return distinct >= simulation_profile.min_distinct_archetypes and max_fill >= simulation_profile.portfolio_size


def _champion_repeat_penalty(selected_indices, candidates) -> float:
    champions = [candidates[index].champion_team_id for index in selected_indices]
    if len(champions) < 2:
        return 0.0
    return (len(champions) - len(set(champions))) / len(champions)


def _final_four_repeat_penalty(selected_indices, candidates) -> float:
    final_four_teams = [
        team_id
        for index in selected_indices
        for team_id in candidates[index].final_four_team_ids
    ]
    if len(final_four_teams) < 2:
        return 0.0
    return (len(final_four_teams) - len(set(final_four_teams))) / len(final_four_teams)


def _region_winner_repeat_penalty(selected_indices, candidates) -> float:
    if not selected_indices:
        return 0.0
    penalties: list[float] = []
    for slot_index in range(4):
        slot_values = [
            candidates[index].final_four_team_ids[slot_index]
            for index in selected_indices
            if len(candidates[index].final_four_team_ids) > slot_index
        ]
        if len(slot_values) < 2:
            penalties.append(0.0)
            continue
        penalties.append((len(slot_values) - len(set(slot_values))) / len(slot_values))
    return float(np.mean(penalties)) if penalties else 0.0


def _average_duplication_proxy(selected_indices, candidates) -> float:
    if not selected_indices:
        return 0.0
    return float(np.mean([candidates[index].duplication_proxy for index in selected_indices]))


def _large_field_pressure(contest_weights: Mapping[str, float]) -> float:
    pressure = 0.0
    for contest_id, weight in contest_weights.items():
        if "large" in contest_id:
            pressure += float(weight)
        elif "mid" in contest_id:
            pressure += 0.5 * float(weight)
    return float(np.clip(pressure, 0.0, 1.0))


def _select_naive_baseline(candidates, simulation_profile, pool_indices) -> list[int]:
    ranked_indices = sorted(
        pool_indices,
        key=lambda index: candidates[index].weighted_first_place_equity,
        reverse=True,
    )
    baseline: list[int] = []

    for candidate_index in ranked_indices:
        candidate = candidates[candidate_index]
        if (
            sum(1 for index in baseline if candidates[index].archetype == candidate.archetype)
            >= simulation_profile.max_per_archetype
        ):
            continue
        trial = baseline + [candidate_index]
        if not _can_still_hit_archetype_floor(trial, candidates, simulation_profile):
            continue
        baseline.append(candidate_index)
        if len(baseline) == simulation_profile.portfolio_size:
            break

    if len(baseline) != simulation_profile.portfolio_size:
        raise RuntimeError("unable to build a feasible V10 naive baseline portfolio")
    return baseline


def portfolio_release_objective(
    evaluation: PortfolioReleaseEvaluation,
    *,
    large_field_pressure: float,
) -> float:
    positive_corr = max(evaluation.payoff_correlation, 0.0)
    return float(
        evaluation.expected_utility
        + (40.0 * evaluation.first_place_equity)
        + (8.0 * evaluation.cash_rate)
        + (6.0 * evaluation.top3_equity)
        + (0.05 * evaluation.expected_payout)
        + (2.0 * evaluation.unique_champions)
        + (1.0 * evaluation.distinct_archetypes)
        - ((12.0 + (8.0 * large_field_pressure)) * evaluation.average_overlap)
        - ((10.0 + (6.0 * large_field_pressure)) * evaluation.average_duplication_proxy)
        - ((8.0 + (6.0 * large_field_pressure)) * evaluation.final_four_repeat_penalty)
        - ((6.0 + (6.0 * large_field_pressure)) * evaluation.region_winner_repeat_penalty)
        - (4.0 * evaluation.champion_repeat_penalty)
        - (4.0 * positive_corr)
    )


def _release_guardrail_failures(
    evaluation: PortfolioReleaseEvaluation,
    *,
    naive_evaluation: PortfolioReleaseEvaluation | None,
    contest_weights: Mapping[str, float],
    release_contract: ReleaseContractConfig,
) -> list[str]:
    failures: list[str] = []

    if not release_contract.allow_zero_fpe_finalist:
        if evaluation.used_zero_equity_fallback:
            failures.append("zero_equity_fallback_used")
        if evaluation.min_individual_fpe <= release_contract.practical_zero_fpe_floor:
            failures.append("zero_fpe_finalist")

    if not release_contract.allow_naive_regression and naive_evaluation is not None:
        if evaluation.first_place_equity + release_contract.fpe_tolerance < naive_evaluation.first_place_equity:
            failures.append("naive_fpe_regression")
        if evaluation.expected_payout + release_contract.payout_tolerance < naive_evaluation.expected_payout:
            failures.append("naive_payout_regression")
        if evaluation.capture_rate + release_contract.capture_tolerance < naive_evaluation.capture_rate:
            failures.append("naive_capture_regression")

    if {"standard_small", "standard_mid", "standard_large"}.issubset(set(contest_weights)):
        if min(float(weight) for weight in contest_weights.values()) < release_contract.blended_weight_floor:
            failures.append("blended_weight_floor")

    return failures


def evaluate_portfolio_release_contract(
    selected_indices,
    candidates,
    candidate_scores_by_contest,
    opponent_scores_by_contest,
    contest_weights,
    contest_payout_profiles,
    overlap_matrix,
    *,
    large_field_pressure: float,
    release_contract: ReleaseContractConfig,
    used_zero_equity_fallback: bool = False,
    naive_evaluation: PortfolioReleaseEvaluation | None = None,
    contest_device_scores: Mapping[str, tuple[Any, Any]] | None = None,
) -> PortfolioReleaseEvaluation:
    key = _selection_key(selected_indices)
    payout_summary = weighted_payout_summary(
        key,
        candidate_scores_by_contest,
        opponent_scores_by_contest,
        contest_weights,
        contest_payout_profiles,
        contest_device_scores,
    )
    if contest_device_scores:
        fpe = 0.0
        capture = 0.0
        for contest_id, weight in contest_weights.items():
            candidate_device, opponent_device = contest_device_scores[contest_id]
            selected_scores = candidate_device[list(key)]
            fpe += weight * portfolio_fpe_gpu(selected_scores, opponent_device)
            capture += weight * portfolio_capture_gpu(selected_scores, opponent_device)
    else:
        fpe = portfolio_first_place_equity(
            key,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            contest_weights,
        )
        capture = portfolio_capture_rate(
            key,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            contest_weights,
        )
    unique_champions = len({candidates[index].champion_team_id for index in key})
    distinct_archetypes = len({candidates[index].archetype for index in key})
    average_overlap = average_pairwise_overlap(key, overlap_matrix)
    champion_penalty = _champion_repeat_penalty(key, candidates)
    final_four_penalty = _final_four_repeat_penalty(key, candidates)
    region_penalty = _region_winner_repeat_penalty(key, candidates)
    duplication_proxy = _average_duplication_proxy(key, candidates)
    individual_equities = [candidates[index].weighted_first_place_equity for index in key]
    min_individual_fpe = float(min(individual_equities)) if individual_equities else 0.0
    average_individual_fpe = float(np.mean(individual_equities)) if individual_equities else 0.0

    evaluation = PortfolioReleaseEvaluation(
        selected_indices=key,
        objective_score=0.0,
        release_objective_score=0.0,
        expected_utility=float(payout_summary.expected_utility),
        expected_payout=float(payout_summary.expected_payout),
        first_place_equity=float(fpe),
        capture_rate=float(capture),
        cash_rate=float(payout_summary.cash_rate),
        top3_equity=float(payout_summary.top3_equity),
        average_overlap=float(average_overlap),
        average_duplication_proxy=float(duplication_proxy),
        final_four_repeat_penalty=float(final_four_penalty),
        region_winner_repeat_penalty=float(region_penalty),
        champion_repeat_penalty=float(champion_penalty),
        distinct_archetypes=distinct_archetypes,
        unique_champions=unique_champions,
        min_individual_fpe=min_individual_fpe,
        average_individual_fpe=average_individual_fpe,
        payoff_correlation=float(payout_summary.payoff_correlation),
        used_zero_equity_fallback=used_zero_equity_fallback,
        payout_summary=payout_summary,
    )
    evaluation.objective_score = portfolio_release_objective(evaluation, large_field_pressure=large_field_pressure)
    evaluation.guardrail_failures = _release_guardrail_failures(
        evaluation,
        naive_evaluation=naive_evaluation,
        contest_weights=contest_weights,
        release_contract=release_contract,
    )
    evaluation.release_objective_score = evaluation.objective_score - (1000.0 * len(evaluation.guardrail_failures))
    evaluation.rank = (
        evaluation.release_objective_score,
        evaluation.first_place_equity,
        evaluation.expected_payout,
        evaluation.cash_rate,
        -evaluation.average_overlap,
        -evaluation.average_duplication_proxy,
        evaluation.min_individual_fpe,
        tuple(-index for index in key),
    )
    return evaluation


def _selection_rank(
    selected_indices,
    candidates,
    candidate_scores_by_contest,
    opponent_scores_by_contest,
    contest_weights,
    contest_payout_profiles,
    overlap_matrix,
    selection_cache,
    large_field_pressure,
    release_contract: ReleaseContractConfig,
    used_zero_equity_fallback: bool = False,
    naive_evaluation: PortfolioReleaseEvaluation | None = None,
    contest_device_scores: Mapping[str, tuple[Any, Any]] | None = None,
) -> PortfolioReleaseEvaluation:
    key = _selection_cache_key(
        selected_indices,
        used_zero_equity_fallback=used_zero_equity_fallback,
        naive_evaluation=naive_evaluation,
    )
    cached = selection_cache.get(key)
    if cached is None:
        cached = evaluate_portfolio_release_contract(
            selected_indices,
            candidates,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            contest_weights,
            contest_payout_profiles,
            overlap_matrix,
            large_field_pressure=large_field_pressure,
            release_contract=release_contract,
            used_zero_equity_fallback=used_zero_equity_fallback,
            naive_evaluation=naive_evaluation,
            contest_device_scores=contest_device_scores,
        )
        selection_cache[key] = cached
    return cached


def _refine_with_local_search(
    selected,
    candidates,
    candidate_scores_by_contest,
    opponent_scores_by_contest,
    contest_weights,
    contest_payout_profiles,
    overlap_matrix,
    simulation_profile,
    selection_cache,
    large_field_pressure,
    release_contract: ReleaseContractConfig,
    used_zero_equity_fallback: bool,
    naive_evaluation: PortfolioReleaseEvaluation,
    contest_device_scores: Mapping[str, tuple[Any, Any]] | None = None,
) -> tuple[list[int], int]:
    swap_count = 0
    improved = True

    while improved:
        improved = False
        current_eval = _selection_rank(
            selected,
            candidates,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            contest_weights,
            contest_payout_profiles,
            overlap_matrix,
            selection_cache,
            large_field_pressure,
            release_contract,
            used_zero_equity_fallback,
            naive_evaluation,
            contest_device_scores,
        )
        current_rank = current_eval.rank
        best_swap = None
        for out_index in selected:
            remaining = [index for index in selected if index != out_index]
            for candidate_index in range(len(candidates)):
                if candidate_index in selected:
                    continue
                if (
                    sum(1 for index in remaining if candidates[index].archetype == candidates[candidate_index].archetype)
                    >= simulation_profile.max_per_archetype
                ):
                    continue
                max_allowed_overlap = 0.85 - (0.10 * large_field_pressure)
                if remaining and max(float(overlap_matrix[candidate_index, index]) for index in remaining) > max_allowed_overlap:
                    continue
                trial = remaining + [candidate_index]
                if len({candidates[index].archetype for index in trial}) < simulation_profile.min_distinct_archetypes:
                    continue
                trial_eval = _selection_rank(
                    trial,
                    candidates,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    contest_weights,
                    contest_payout_profiles,
                    overlap_matrix,
                    selection_cache,
                    large_field_pressure,
                    release_contract,
                    used_zero_equity_fallback,
                    naive_evaluation,
                    contest_device_scores,
                )
                trial_rank = trial_eval.rank
                if trial_rank > current_rank and (best_swap is None or trial_rank > best_swap[0]):
                    best_swap = (trial_rank, trial)
        if best_swap is not None:
            selected = best_swap[1]
            swap_count += 1
            improved = True
    return selected, swap_count


def select_portfolio(
    candidates,
    bracket_picks: np.ndarray,
    candidate_scores_by_contest: Dict[str, np.ndarray],
    opponent_scores_by_contest: Dict[str, np.ndarray],
    simulation_profile,
    contest_payout_profiles: Mapping[str, ContestPayoutProfile],
    release_contract: ReleaseContractConfig | None = None,
) -> dict:
    release_contract = release_contract or ReleaseContractConfig()
    overlap_matrix = build_overlap_matrix(bracket_picks)
    contest_weights = simulation_profile.primary_contests
    selection_cache: dict[tuple[int, ...], PortfolioReleaseEvaluation] = {}
    large_field_pressure = _large_field_pressure(contest_weights)
    contest_device_scores = (
        {
            contest_id: (
                to_device(candidate_scores_by_contest[contest_id]),
                to_device(opponent_scores_by_contest[contest_id]),
            )
            for contest_id in candidate_scores_by_contest
        }
        if GPU_AVAILABLE and payout_gpu_enabled()
        else None
    )

    positive_indices = [
        index for index, candidate in enumerate(candidates)
        if candidate.weighted_first_place_equity > ZERO_EQUITY_EPSILON
    ]
    practical_positive_indices = [
        index for index, candidate in enumerate(candidates)
        if candidate.weighted_first_place_equity > release_contract.practical_zero_fpe_floor
    ]

    used_zero_equity_fallback = False
    zero_equity_fallback_reason = None

    if _pool_supports_constraints(practical_positive_indices, candidates, simulation_profile):
        pool_indices = practical_positive_indices
    elif _pool_supports_constraints(positive_indices, candidates, simulation_profile):
        used_zero_equity_fallback = True
        zero_equity_fallback_reason = "practical_positive_equity_pool_infeasible"
        pool_indices = positive_indices
    else:
        used_zero_equity_fallback = True
        if len(positive_indices) < simulation_profile.portfolio_size:
            zero_equity_fallback_reason = (
                f"only {len(positive_indices)} candidates with positive equity, "
                f"need {simulation_profile.portfolio_size}"
            )
        else:
            positive_archetypes = {candidates[index].archetype for index in positive_indices}
            zero_equity_fallback_reason = (
                "positive pool cannot satisfy archetype constraints "
                f"(distinct={len(positive_archetypes)}, need={simulation_profile.min_distinct_archetypes})"
            )
        pool_indices = list(range(len(candidates)))

    naive_baseline_indices = _select_naive_baseline(candidates, simulation_profile, pool_indices)
    naive_eval = evaluate_portfolio_release_contract(
        naive_baseline_indices,
        candidates,
        candidate_scores_by_contest,
        opponent_scores_by_contest,
        contest_weights,
        contest_payout_profiles,
        overlap_matrix,
        large_field_pressure=large_field_pressure,
        release_contract=release_contract,
        used_zero_equity_fallback=used_zero_equity_fallback,
        contest_device_scores=contest_device_scores,
    )

    selected: list[int] = []
    for _ in range(simulation_profile.portfolio_size):
        selected_archetypes = {candidates[index].archetype for index in selected}
        needed_new_archetypes = max(0, simulation_profile.min_distinct_archetypes - len(selected_archetypes))
        remaining_slots = simulation_profile.portfolio_size - len(selected)
        require_new_archetype = needed_new_archetypes >= remaining_slots

        best_index = None
        best_rank = None
        for candidate_index in pool_indices:
            if candidate_index in selected:
                continue
            candidate = candidates[candidate_index]
            if (
                sum(1 for index in selected if candidates[index].archetype == candidate.archetype)
                >= simulation_profile.max_per_archetype
            ):
                continue
            if require_new_archetype and candidate.archetype in selected_archetypes:
                continue
            max_allowed_overlap = 0.85 - (0.10 * large_field_pressure)
            if selected and max(float(overlap_matrix[candidate_index, index]) for index in selected) > max_allowed_overlap:
                continue
            trial = selected + [candidate_index]
            rank = _selection_rank(
                trial,
                candidates,
                candidate_scores_by_contest,
                opponent_scores_by_contest,
                contest_weights,
                contest_payout_profiles,
                overlap_matrix,
                selection_cache,
                large_field_pressure,
                release_contract,
                used_zero_equity_fallback,
                naive_eval,
                contest_device_scores,
            ).rank
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_index = candidate_index

        if best_index is None and require_new_archetype:
            for candidate_index in pool_indices:
                if candidate_index in selected:
                    continue
                candidate = candidates[candidate_index]
                if (
                    sum(1 for index in selected if candidates[index].archetype == candidate.archetype)
                    >= simulation_profile.max_per_archetype
                ):
                    continue
                max_allowed_overlap = 0.85 - (0.10 * large_field_pressure)
                if selected and max(float(overlap_matrix[candidate_index, index]) for index in selected) > max_allowed_overlap:
                    continue
                trial = selected + [candidate_index]
                rank = _selection_rank(
                    trial,
                    candidates,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    contest_weights,
                    contest_payout_profiles,
                    overlap_matrix,
                    selection_cache,
                    large_field_pressure,
                    release_contract,
                    used_zero_equity_fallback,
                    naive_eval,
                    contest_device_scores,
                ).rank
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_index = candidate_index

        if best_index is None:
            raise RuntimeError("unable to complete a V10 portfolio with the current constraints")
        selected.append(best_index)

    greedy_selected = list(selected)
    selected, swap_count = _refine_with_local_search(
        selected,
        candidates,
        candidate_scores_by_contest,
        opponent_scores_by_contest,
        contest_weights,
        contest_payout_profiles,
        overlap_matrix,
        simulation_profile,
        selection_cache,
        large_field_pressure,
        release_contract,
        used_zero_equity_fallback,
        naive_eval,
        contest_device_scores,
    )
    portfolio_eval = evaluate_portfolio_release_contract(
        selected,
        candidates,
        candidate_scores_by_contest,
        opponent_scores_by_contest,
        contest_weights,
        contest_payout_profiles,
        overlap_matrix,
        large_field_pressure=large_field_pressure,
        release_contract=release_contract,
        used_zero_equity_fallback=used_zero_equity_fallback,
        naive_evaluation=naive_eval,
        contest_device_scores=contest_device_scores,
    )

    selection_metadata = {
        "selection_objective": release_contract.objective_name,
        "positive_equity_pool_size": len(positive_indices),
        "practical_positive_equity_pool_size": len(practical_positive_indices),
        "used_zero_equity_fallback": used_zero_equity_fallback,
        "zero_equity_fallback_reason": zero_equity_fallback_reason,
        "candidate_pool_size": len(pool_indices),
        "candidate_pool_indices": list(pool_indices),
        "initial_greedy_indices": list(greedy_selected),
        "local_search_swaps": swap_count,
        "local_search_applied": swap_count > 0,
        "naive_baseline_pool_size": len(pool_indices),
        "portfolio_expected_utility": portfolio_eval.expected_utility,
        "naive_expected_utility": naive_eval.expected_utility,
        "portfolio_release_objective_score": portfolio_eval.release_objective_score,
        "naive_release_objective_score": naive_eval.release_objective_score,
        "portfolio_average_overlap": portfolio_eval.average_overlap,
        "naive_average_overlap": naive_eval.average_overlap,
        "portfolio_duplication_score": portfolio_eval.average_duplication_proxy,
        "naive_duplication_score": naive_eval.average_duplication_proxy,
        "portfolio_final_four_repeat_penalty": portfolio_eval.final_four_repeat_penalty,
        "naive_final_four_repeat_penalty": naive_eval.final_four_repeat_penalty,
        "portfolio_region_winner_repeat_penalty": portfolio_eval.region_winner_repeat_penalty,
        "naive_region_winner_repeat_penalty": naive_eval.region_winner_repeat_penalty,
        "portfolio_min_individual_fpe": portfolio_eval.min_individual_fpe,
        "naive_min_individual_fpe": naive_eval.min_individual_fpe,
        "portfolio_guardrail_failures": list(portfolio_eval.guardrail_failures),
        "naive_guardrail_failures": list(naive_eval.guardrail_failures),
        "passes_release_gates": len(portfolio_eval.guardrail_failures) == 0,
        "release_contract": asdict(release_contract),
        "large_field_pressure": large_field_pressure,
        "selection_cache_size": len(selection_cache),
    }

    return {
        "selected_indices": selected,
        "naive_baseline_indices": naive_baseline_indices,
        "portfolio_fpe": portfolio_eval.first_place_equity,
        "naive_fpe": naive_eval.first_place_equity,
        "portfolio_capture": portfolio_eval.capture_rate,
        "naive_capture": naive_eval.capture_rate,
        "portfolio_payout_summary": portfolio_eval.payout_summary,
        "naive_payout_summary": naive_eval.payout_summary,
        "portfolio_average_overlap": portfolio_eval.average_overlap,
        "selection_metadata": selection_metadata,
        "portfolio_release_evaluation": portfolio_eval,
        "naive_release_evaluation": naive_eval,
    }


__all__ = [
    "PRACTICAL_ZERO_FPE_EPSILON",
    "PortfolioReleaseEvaluation",
    "ReleaseContractConfig",
    "evaluate_portfolio_release_contract",
    "portfolio_release_objective",
    "select_portfolio",
    "weighted_payout_summary",
]

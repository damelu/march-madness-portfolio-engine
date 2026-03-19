from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from .gpu import GPU_AVAILABLE, portfolio_payout_summary_gpu
from .models import ContestPayoutProfile, PortfolioPayoutSummary


def payout_gpu_enabled() -> bool:
    mode = os.environ.get("MARCH_MADNESS_PAYOUT_BACKEND", "auto").strip().lower()
    if mode in {"cpu", "numpy", "off", "disabled"}:
        return False
    return GPU_AVAILABLE


def _as_2d_scores(values: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError("score inputs must be one- or two-dimensional")
    return array


def _materialize_payout_curve(
    payout_profile: ContestPayoutProfile | Mapping[int, float],
    total_entries: int,
) -> tuple[dict[int, float], str, float]:
    if isinstance(payout_profile, ContestPayoutProfile):
        curve = {int(rank): float(value) for rank, value in payout_profile.payout_curve.items()}
        tie_split_mode = payout_profile.tie_split_mode
        top_heavy_weight = payout_profile.top_heavy_weight
        if curve and sum(curve.values()) <= 1.0 + 1e-9:
            total_pool = max(total_entries, 1) * payout_profile.entry_fee
            return ({rank: share * total_pool for rank, share in curve.items()}, tie_split_mode, top_heavy_weight)
        return (curve, tie_split_mode, top_heavy_weight)
    return ({int(rank): float(value) for rank, value in payout_profile.items()}, "proportional", 1.0)


def tie_split_payout(
    payout_curve: Mapping[int, float],
    start_rank: int,
    tie_size: int,
    tie_split_mode: str = "proportional",
) -> float:
    if tie_size <= 0:
        raise ValueError("tie_size must be positive")
    occupied_ranks = range(start_rank, start_rank + tie_size)
    gross_payout = sum(float(payout_curve.get(rank, 0.0)) for rank in occupied_ranks)
    if tie_split_mode not in {"proportional", "shared_win", "fractional"}:
        raise ValueError(f"unsupported tie_split_mode: {tie_split_mode}")
    return gross_payout / tie_size


def _entry_rank_statistics(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    selected = _as_2d_scores(selected_scores)
    field = _as_2d_scores(field_scores)
    if selected.shape[1] != field.shape[1]:
        raise ValueError("selected_scores and field_scores must have the same simulation count")

    selected_vs_selected = selected[:, None, :]
    selected_vs_field = selected[:, None, :]
    better_count = (
        np.sum(field[None, :, :] > selected_vs_field, axis=1, dtype=np.int32)
        + np.sum(selected[None, :, :] > selected_vs_selected, axis=1, dtype=np.int32)
    )
    tie_count = (
        np.sum(field[None, :, :] == selected_vs_field, axis=1, dtype=np.int32)
        + np.sum(selected[None, :, :] == selected_vs_selected, axis=1, dtype=np.int32)
    )
    return better_count, tie_count


def _payout_curve_prefix(payout_curve: Mapping[int, float]) -> tuple[np.ndarray, int]:
    if not payout_curve:
        return np.zeros(1, dtype=np.float64), 0
    max_rank = max(int(rank) for rank in payout_curve)
    values = np.zeros(max_rank + 1, dtype=np.float64)
    for rank, value in payout_curve.items():
        if rank <= 0:
            continue
        values[int(rank)] = float(value)
    return np.cumsum(values), max_rank


def _payouts_from_rank_stats(
    better_count: np.ndarray,
    tie_count: np.ndarray,
    payout_curve: Mapping[int, float],
    tie_split_mode: str,
) -> np.ndarray:
    if tie_split_mode not in {"proportional", "shared_win", "fractional"}:
        raise ValueError(f"unsupported tie_split_mode: {tie_split_mode}")
    prefix, max_rank = _payout_curve_prefix(payout_curve)
    if max_rank == 0:
        return np.zeros_like(better_count, dtype=np.float64)
    start_rank = better_count + 1
    end_rank = start_rank + tie_count - 1
    clipped_start = np.clip(start_rank - 1, 0, max_rank)
    clipped_end = np.clip(end_rank, 0, max_rank)
    gross = prefix[clipped_end] - prefix[clipped_start]
    return gross / tie_count


def _top_k_from_rank_stats(
    better_count: np.ndarray,
    tie_count: np.ndarray,
    k: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    start_rank = better_count + 1
    end_rank = start_rank + tie_count - 1
    overlap = np.maximum(0, np.minimum(k, end_rank) - start_rank + 1)
    return (overlap / tie_count).mean(axis=1)


def _portfolio_top_k_share_from_rank_stats(
    better_count: np.ndarray,
    tie_count: np.ndarray,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    start_rank = better_count + 1
    end_rank = start_rank + tie_count - 1
    overlap = np.maximum(0, np.minimum(k, end_rank) - start_rank + 1)
    occupied_slots = np.sum(overlap / tie_count, axis=0)
    return float(np.mean(np.clip(occupied_slots / k, 0.0, 1.0)))


def _payout_matrix(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
    payout_profile: ContestPayoutProfile | Mapping[int, float],
) -> tuple[np.ndarray, dict[int, float], float]:
    selected = _as_2d_scores(selected_scores)
    field = _as_2d_scores(field_scores)
    if selected.shape[1] != field.shape[1]:
        raise ValueError("selected_scores and field_scores must have the same simulation count")

    payout_curve, tie_split_mode, top_heavy_weight = _materialize_payout_curve(
        payout_profile,
        total_entries=selected.shape[0] + field.shape[0],
    )
    better_count, tie_count = _entry_rank_statistics(selected, field)
    payouts = _payouts_from_rank_stats(better_count, tie_count, payout_curve, tie_split_mode)
    return payouts, payout_curve, top_heavy_weight


def expected_payout(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
    payout_profile: ContestPayoutProfile | Mapping[int, float],
) -> np.ndarray:
    payouts, _, _ = _payout_matrix(selected_scores, field_scores, payout_profile)
    return payouts.mean(axis=1)


def cash_rate(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
    payout_profile: ContestPayoutProfile | Mapping[int, float],
) -> np.ndarray:
    payouts, _, _ = _payout_matrix(selected_scores, field_scores, payout_profile)
    return (payouts > 0.0).mean(axis=1)


def top_k_equity(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
    k: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    selected = _as_2d_scores(selected_scores)
    field = _as_2d_scores(field_scores)
    if selected.shape[1] != field.shape[1]:
        raise ValueError("selected_scores and field_scores must have the same simulation count")
    better_count, tie_count = _entry_rank_statistics(selected, field)
    return _top_k_from_rank_stats(better_count, tie_count, k)


def portfolio_payoff_correlation(payout_matrix: np.ndarray | list[list[float]]) -> float:
    payouts = _as_2d_scores(payout_matrix)
    if payouts.shape[0] < 2:
        return 0.0

    centered = payouts - payouts.mean(axis=1, keepdims=True)
    std = payouts.std(axis=1)
    denom = std[:, None] * std[None, :]
    sim_count = max(int(payouts.shape[1]), 1)
    corr_matrix = np.zeros((payouts.shape[0], payouts.shape[0]), dtype=np.float64)
    np.divide(
        centered @ centered.T,
        denom * sim_count,
        out=corr_matrix,
        where=denom > 0.0,
    )
    triu = np.triu_indices(payouts.shape[0], k=1)
    upper = corr_matrix[triu]
    if upper.size == 0:
        return 0.0
    upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(upper))


def portfolio_expected_utility(
    selected_scores: np.ndarray | list[list[float]] | list[float],
    field_scores: np.ndarray | list[list[float]] | list[float],
    payout_profile: ContestPayoutProfile | Mapping[int, float],
    correlation_penalty: float = 0.0,
) -> PortfolioPayoutSummary:
    selected = _as_2d_scores(selected_scores)
    field = _as_2d_scores(field_scores)
    if selected.shape[1] != field.shape[1]:
        raise ValueError("selected_scores and field_scores must have the same simulation count")
    payout_curve, tie_split_mode, top_heavy_weight = _materialize_payout_curve(
        payout_profile,
        total_entries=selected.shape[0] + field.shape[0],
    )
    if payout_gpu_enabled() and selected.size and field.size:
        summary = portfolio_payout_summary_gpu(
            selected,
            field,
            payout_curve,
            correlation_penalty=correlation_penalty,
        )
        return PortfolioPayoutSummary(**summary)

    better_count, tie_count = _entry_rank_statistics(selected, field)
    payouts = _payouts_from_rank_stats(better_count, tie_count, payout_curve, tie_split_mode)
    total_payouts = payouts.sum(axis=0)
    payoff_correlation = portfolio_payoff_correlation(payouts)
    top1 = _portfolio_top_k_share_from_rank_stats(better_count, tie_count, 1)
    top3 = _portfolio_top_k_share_from_rank_stats(better_count, tie_count, 3)
    top5 = _portfolio_top_k_share_from_rank_stats(better_count, tie_count, 5)
    expected_payout_value = float(np.mean(total_payouts))
    certainty_equivalent = float(np.expm1(np.mean(np.log1p(np.maximum(total_payouts, 0.0)))))
    if correlation_penalty > 0.0:
        certainty_equivalent *= max(0.0, 1.0 - (correlation_penalty * max(payoff_correlation, 0.0)))
    return PortfolioPayoutSummary(
        expected_payout=expected_payout_value,
        cash_rate=float(np.mean(total_payouts > 0.0)),
        top1_equity=top1,
        top3_equity=top3,
        top5_equity=top5,
        downside_risk=float(np.percentile(total_payouts, 5)),
        payoff_correlation=payoff_correlation,
        expected_utility=certainty_equivalent,
    )

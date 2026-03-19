"""GPU-accelerated computation backend using CuPy.

Drop-in replacements for the hot paths in scoring.py and portfolio.py.
Falls back to numpy when CuPy is not available (CPU-only environments).

Usage:
    from .gpu import xp, to_device, to_numpy, score_brackets_gpu, portfolio_fpe_gpu

The `xp` module is either cupy or numpy depending on availability.
All functions accept numpy arrays and return numpy arrays — GPU transfer
is handled internally.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
    print("[gpu] CuPy detected — GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    xp = np


def to_device(arr: np.ndarray) -> "xp.ndarray":
    """Move numpy array to GPU (no-op if no GPU)."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_numpy(arr) -> np.ndarray:
    """Move array back to CPU numpy (no-op if already numpy)."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def score_brackets_gpu(
    bracket_picks: np.ndarray,
    tournament_outcomes: np.ndarray,
    upset_gaps: np.ndarray,
    game_weights: np.ndarray,
    upset_bonus_per_seed: float = 0.0,
) -> np.ndarray:
    """Score brackets against tournament outcomes on GPU.

    This is the hottest function in the pipeline:
    (num_brackets × num_sims × num_games) comparisons.

    Returns numpy array of shape (num_brackets, num_sims).
    """
    # Move to GPU
    picks = to_device(bracket_picks)
    outcomes = to_device(tournament_outcomes)
    gaps = to_device(upset_gaps)
    weights = to_device(game_weights)

    num_brackets = picks.shape[0]
    num_sims = outcomes.shape[0]

    # Batch to control GPU memory (each batch: batch_size × num_sims × 63)
    batch_size = min(256, num_brackets)  # ~256 brackets × 5000 sims × 63 games × 1 byte ≈ 80MB
    scores = xp.zeros((num_brackets, num_sims), dtype=xp.float32)

    for start in range(0, num_brackets, batch_size):
        stop = min(start + batch_size, num_brackets)
        batch = picks[start:stop]

        # Core comparison: (batch, 1, 63) == (1, sims, 63) → (batch, sims, 63) bool
        correct = batch[:, None, :] == outcomes[None, :, :]

        # Weighted sum across games
        batch_scores = (correct * weights[None, None, :]).sum(axis=2)

        if upset_bonus_per_seed > 0:
            batch_scores += (correct * gaps[None, :, :] * upset_bonus_per_seed).sum(axis=2)

        scores[start:stop] = batch_scores

    return to_numpy(scores.astype(xp.float32))


def evaluate_candidates_gpu(
    candidate_scores: np.ndarray,
    opponent_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-candidate first-place equity and average finish on GPU.

    Returns (first_place_equities, average_finishes) as numpy arrays.
    """
    cand = to_device(candidate_scores)
    opp = to_device(opponent_scores)

    max_opp = opp.max(axis=0)  # (num_sims,)
    num_candidates = cand.shape[0]

    fpe = xp.zeros(num_candidates, dtype=xp.float32)
    avg_finish = xp.zeros(num_candidates, dtype=xp.float32)

    for i in range(num_candidates):
        scores_i = cand[i]  # (num_sims,)
        better = (opp > scores_i).sum(axis=0)
        tied = (opp == scores_i).sum(axis=0)
        outright = scores_i > max_opp
        tied_first = scores_i == max_opp
        shared = xp.where(outright, 1.0, xp.where(tied_first, 1.0 / (tied + 1.0), 0.0))
        fpe[i] = shared.mean()
        avg_finish[i] = (1.0 + better + 0.5 * tied).mean()

    return to_numpy(fpe), to_numpy(avg_finish)


def portfolio_fpe_gpu(
    selected_scores: np.ndarray,
    opponent_scores: np.ndarray,
) -> float:
    """Compute portfolio first-place equity on GPU.

    selected_scores: (num_selected, num_sims)
    opponent_scores: (num_opponents, num_sims)
    """
    sel = to_device(selected_scores)
    opp = to_device(opponent_scores)

    max_sel = sel.max(axis=0)  # (num_sims,)
    max_opp = opp.max(axis=0)  # (num_sims,)

    sel_ties = (sel == max_sel[None, :]).sum(axis=0)  # how many of our brackets tied for best
    opp_ties = (opp == max_sel[None, :]).sum(axis=0)  # how many opponents matched our best

    share = xp.where(
        max_sel > max_opp,
        1.0,
        xp.where(
            max_sel == max_opp,
            sel_ties / xp.maximum(sel_ties + opp_ties, 1),
            0.0,
        ),
    )
    return float(to_numpy(share.mean()))


def portfolio_capture_gpu(
    selected_scores: np.ndarray,
    opponent_scores: np.ndarray,
) -> float:
    """Compute portfolio capture rate on GPU."""
    sel = to_device(selected_scores)
    opp = to_device(opponent_scores)
    max_sel = sel.max(axis=0)
    max_opp = opp.max(axis=0)
    return float(to_numpy((max_sel >= max_opp).mean()))


def _materialize_payout_prefix_gpu(
    payout_curve: Mapping[int, float],
) -> tuple[Any, int]:
    if not payout_curve:
        return to_device(np.zeros(1, dtype=np.float64)), 0
    max_rank = max(int(rank) for rank in payout_curve)
    values = np.zeros(max_rank + 1, dtype=np.float64)
    for rank, value in payout_curve.items():
        if rank <= 0:
            continue
        values[int(rank)] = float(value)
    return to_device(np.cumsum(values)), max_rank


def _entry_rank_statistics_gpu(
    selected_scores: np.ndarray,
    field_scores: np.ndarray,
) -> tuple[Any, Any]:
    sel = to_device(selected_scores)
    field = to_device(field_scores)
    selected_vs_selected = sel[:, None, :]
    selected_vs_field = sel[:, None, :]
    better_count = (
        xp.sum(field[None, :, :] > selected_vs_field, axis=1, dtype=xp.int32)
        + xp.sum(sel[None, :, :] > selected_vs_selected, axis=1, dtype=xp.int32)
    )
    tie_count = (
        xp.sum(field[None, :, :] == selected_vs_field, axis=1, dtype=xp.int32)
        + xp.sum(sel[None, :, :] == selected_vs_selected, axis=1, dtype=xp.int32)
    )
    return better_count, tie_count


def portfolio_payout_summary_gpu(
    selected_scores: np.ndarray,
    field_scores: np.ndarray,
    payout_curve: Mapping[int, float],
    *,
    correlation_penalty: float = 0.0,
) -> dict[str, float]:
    """Compute joint portfolio payout summary on GPU-compatible backend."""
    better_count, tie_count = _entry_rank_statistics_gpu(selected_scores, field_scores)
    prefix, max_rank = _materialize_payout_prefix_gpu(payout_curve)
    if max_rank == 0:
        return {
            "expected_payout": 0.0,
            "cash_rate": 0.0,
            "top1_equity": 0.0,
            "top3_equity": 0.0,
            "top5_equity": 0.0,
            "downside_risk": 0.0,
            "payoff_correlation": 0.0,
            "expected_utility": 0.0,
        }

    start_rank = better_count + 1
    end_rank = start_rank + tie_count - 1
    clipped_start = xp.clip(start_rank - 1, 0, max_rank)
    clipped_end = xp.clip(end_rank, 0, max_rank)
    gross = prefix[clipped_end] - prefix[clipped_start]
    payouts = gross / tie_count
    total_payouts = payouts.sum(axis=0)

    def portfolio_top_k_share(k: int) -> float:
        overlap = xp.maximum(0, xp.minimum(k, end_rank) - start_rank + 1)
        occupied_slots = xp.sum(overlap / tie_count, axis=0)
        return float(to_numpy(xp.mean(xp.clip(occupied_slots / k, 0.0, 1.0))))

    if payouts.shape[0] < 2:
        payoff_correlation = 0.0
    else:
        centered = payouts - payouts.mean(axis=1, keepdims=True)
        std = payouts.std(axis=1)
        denom = std[:, None] * std[None, :]
        sim_count = max(int(payouts.shape[1]), 1)
        corr_matrix = xp.zeros((int(payouts.shape[0]), int(payouts.shape[0])), dtype=xp.float64)
        numerator = centered @ centered.T
        mask = denom > 0
        corr_matrix[mask] = numerator[mask] / (denom[mask] * sim_count)
        triu = xp.triu_indices(int(payouts.shape[0]), k=1)
        upper = corr_matrix[triu]
        payoff_correlation = float(to_numpy(upper.mean())) if int(upper.size) else 0.0

    expected_payout_value = float(to_numpy(total_payouts.mean()))
    certainty_equivalent = float(to_numpy(xp.expm1(xp.mean(xp.log1p(xp.maximum(total_payouts, 0.0))))))
    if correlation_penalty > 0.0:
        certainty_equivalent *= max(0.0, 1.0 - (correlation_penalty * max(payoff_correlation, 0.0)))
    return {
        "expected_payout": expected_payout_value,
        "cash_rate": float(to_numpy((total_payouts > 0.0).mean())),
        "top1_equity": portfolio_top_k_share(1),
        "top3_equity": portfolio_top_k_share(3),
        "top5_equity": portfolio_top_k_share(5),
        "downside_risk": float(to_numpy(xp.percentile(total_payouts, 5))),
        "payoff_correlation": payoff_correlation,
        "expected_utility": certainty_equivalent,
    }


def overlap_matrix_gpu(bracket_picks: np.ndarray) -> np.ndarray:
    """Compute pairwise overlap matrix on GPU."""
    picks = to_device(bracket_picks)
    num_games = picks.shape[1]
    # (N, 1, G) == (1, N, G) → sum over G → (N, N)
    overlap = (picks[:, None, :] == picks[None, :, :]).sum(axis=2).astype(xp.float32) / num_games
    result = to_numpy(overlap)
    np.fill_diagonal(result, 1.0)
    return result

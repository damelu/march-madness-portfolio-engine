from __future__ import annotations

import numpy as np

from .gpu import GPU_AVAILABLE, score_brackets_gpu
from .models import ScoringProfile


def score_brackets(
    bracket_picks: np.ndarray,
    tournament_outcomes: np.ndarray,
    upset_gaps: np.ndarray,
    game_rounds: np.ndarray,
    scoring_profile: ScoringProfile,
    batch_size: int = 64,
) -> np.ndarray:
    """Score bracket picks against tournament outcomes.

    Uses GPU acceleration via CuPy when available, falls back to CPU numpy.
    """
    round_weights = np.array(scoring_profile.round_weights, dtype=np.float32)
    game_weights = round_weights[game_rounds - 1]

    if GPU_AVAILABLE:
        return score_brackets_gpu(
            bracket_picks, tournament_outcomes, upset_gaps,
            game_weights, scoring_profile.upset_bonus_per_seed,
        )

    # CPU fallback
    scores = np.zeros((bracket_picks.shape[0], tournament_outcomes.shape[0]), dtype=np.float32)
    for start in range(0, bracket_picks.shape[0], batch_size):
        stop = min(start + batch_size, bracket_picks.shape[0])
        batch = bracket_picks[start:stop]
        correct = batch[:, None, :] == tournament_outcomes[None, :, :]
        batch_scores = (correct * game_weights[None, None, :]).sum(axis=2, dtype=np.float32)
        if scoring_profile.upset_bonus_per_seed:
            batch_scores += (
                correct * upset_gaps[None, :, :] * scoring_profile.upset_bonus_per_seed
            ).sum(axis=2, dtype=np.float32)
        scores[start:stop] = batch_scores
    return scores


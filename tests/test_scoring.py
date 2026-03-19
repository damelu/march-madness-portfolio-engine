from __future__ import annotations

import unittest

import numpy as np

from march_madness_2026.models import ScoringProfile
from march_madness_2026.scoring import score_brackets


class ScoringProfileTests(unittest.TestCase):
    def test_standard_and_upset_bonus_scoring(self) -> None:
        standard = ScoringProfile(
            profile_id="standard",
            name="Standard",
            round_weights=[1, 2, 4, 8, 16, 32],
        )
        upset_bonus = ScoringProfile(
            profile_id="upset",
            name="Upset bonus",
            round_weights=[1, 2, 4, 8, 16, 32],
            upset_bonus_per_seed=0.5,
        )

        bracket_picks = np.array(
            [
                [1, 2, 3],
                [1, 9, 3],
            ],
            dtype=np.int16,
        )
        tournament_outcomes = np.array(
            [
                [1, 2, 3],
                [1, 9, 7],
            ],
            dtype=np.int16,
        )
        upset_gaps = np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
            ],
            dtype=np.int16,
        )
        game_rounds = np.array([1, 2, 6], dtype=np.int16)

        standard_scores = score_brackets(
            bracket_picks,
            tournament_outcomes,
            upset_gaps,
            game_rounds,
            standard,
        )
        upset_scores = score_brackets(
            bracket_picks,
            tournament_outcomes,
            upset_gaps,
            game_rounds,
            upset_bonus,
        )

        self.assertEqual(standard_scores.tolist(), [[35.0, 1.0], [33.0, 3.0]])
        self.assertEqual(upset_scores.tolist(), [[35.0, 1.0], [33.0, 4.0]])


if __name__ == "__main__":
    unittest.main()


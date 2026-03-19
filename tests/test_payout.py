from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from march_madness_2026.models import ContestPayoutProfile
from march_madness_2026.portfolio import _portfolio_gpu_enabled
from march_madness_2026.payout import (
    cash_rate,
    expected_payout,
    payout_gpu_enabled,
    portfolio_expected_utility,
    portfolio_payoff_correlation,
    tie_split_payout,
    top_k_equity,
)


class PayoutHelperTests(unittest.TestCase):
    def test_backend_env_can_force_cpu_paths(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MARCH_MADNESS_PAYOUT_BACKEND": "cpu",
                "MARCH_MADNESS_PORTFOLIO_BACKEND": "cpu",
            },
            clear=False,
        ):
            self.assertFalse(payout_gpu_enabled())
            self.assertFalse(_portfolio_gpu_enabled())

    def test_winner_take_all_expected_payout_handles_first_place_ties(self) -> None:
        payout_profile = ContestPayoutProfile(
            profile_id="winner_take_all",
            name="Winner take all",
            entry_fee=20.0,
            payout_curve={1: 1.0},
            tie_split_mode="proportional",
            top_heavy_weight=1.0,
        )
        selected_scores = np.array([[100.0, 90.0]], dtype=np.float64)
        field_scores = np.array([[100.0, 80.0]], dtype=np.float64)

        payouts = expected_payout(selected_scores, field_scores, payout_profile)
        cash = cash_rate(selected_scores, field_scores, payout_profile)
        top1 = top_k_equity(selected_scores, field_scores, 1)

        self.assertAlmostEqual(tie_split_payout({1: 40.0}, start_rank=1, tie_size=2), 20.0, places=6)
        self.assertAlmostEqual(payouts[0], 30.0, places=6)
        self.assertAlmostEqual(cash[0], 1.0, places=6)
        self.assertAlmostEqual(top1[0], 0.75, places=6)

    def test_top3_profile_and_portfolio_summary_capture_cash_and_correlation(self) -> None:
        payout_profile = ContestPayoutProfile(
            profile_id="top3",
            name="Top 3",
            entry_fee=10.0,
            payout_curve={1: 0.6, 2: 0.25, 3: 0.15},
            tie_split_mode="proportional",
            top_heavy_weight=0.75,
        )
        selected_scores = np.array(
            [
                [120.0, 110.0, 90.0],
                [118.0, 95.0, 105.0],
            ],
            dtype=np.float64,
        )
        field_scores = np.array(
            [
                [100.0, 108.0, 102.0],
                [99.0, 90.0, 98.0],
                [80.0, 85.0, 85.0],
            ],
            dtype=np.float64,
        )

        payouts = expected_payout(selected_scores, field_scores, payout_profile)
        cash = cash_rate(selected_scores, field_scores, payout_profile)
        top3 = top_k_equity(selected_scores, field_scores, 3)
        summary = portfolio_expected_utility(selected_scores, field_scores, payout_profile, correlation_penalty=0.1)
        correlation = portfolio_payoff_correlation(
            np.array(
                [
                    [30.0, 0.0, 10.0],
                    [0.0, 20.0, 10.0],
                ],
                dtype=np.float64,
            )
        )

        self.assertEqual(payouts.shape, (2,))
        self.assertEqual(cash.shape, (2,))
        self.assertTrue(np.all(top3 >= 0.0))
        self.assertGreater(summary.expected_payout, 0.0)
        self.assertGreaterEqual(summary.cash_rate, 0.0)
        self.assertLessEqual(summary.payoff_correlation, 1.0)
        self.assertGreaterEqual(correlation, -1.0)

    def test_portfolio_summary_uses_joint_topk_share_and_certainty_equivalent(self) -> None:
        payout_profile = ContestPayoutProfile(
            profile_id="joint_top3",
            name="Joint Top 3",
            entry_fee=10.0,
            payout_curve={1: 0.6, 2: 0.3, 3: 0.1},
            tie_split_mode="proportional",
            top_heavy_weight=1.0,
        )
        selected_scores = np.array(
            [
                [120.0],
                [110.0],
            ],
            dtype=np.float64,
        )
        field_scores = np.array(
            [
                [90.0],
                [80.0],
            ],
            dtype=np.float64,
        )

        summary = portfolio_expected_utility(selected_scores, field_scores, payout_profile)

        self.assertAlmostEqual(summary.top1_equity, 1.0, places=6)
        self.assertAlmostEqual(summary.top3_equity, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(summary.cash_rate, 1.0, places=6)
        self.assertGreaterEqual(summary.expected_payout, summary.expected_utility)

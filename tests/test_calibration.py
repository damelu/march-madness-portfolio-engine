from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from march_madness_2026.calibration import (
    build_calibration_report,
    compute_brier_decomposition,
    compute_brier_score,
    compute_expected_calibration_error,
    compute_log_loss,
    compute_reliability_bins,
    plot_reliability_diagram,
    summarize_calibration_by_round,
)


class CalibrationMetricTests(unittest.TestCase):
    def test_calibration_metrics_and_report_are_consistent(self) -> None:
        probabilities = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float64)
        outcomes = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        rounds = ["R64", "R64", "R32", "R32"]

        log_loss = compute_log_loss(probabilities, outcomes)
        brier = compute_brier_score(probabilities, outcomes)
        ece = compute_expected_calibration_error(probabilities, outcomes, num_bins=2)
        bins = compute_reliability_bins(probabilities, outcomes, num_bins=2)
        brier_parts = compute_brier_decomposition(probabilities, outcomes, num_bins=2)
        by_round = summarize_calibration_by_round(probabilities, outcomes, rounds, num_bins=2)
        report = build_calibration_report(
            probabilities,
            outcomes,
            num_bins=2,
            round_labels=rounds,
            baseline_probabilities=np.full_like(probabilities, 0.5),
            model_id="unit_test_model",
        )

        self.assertAlmostEqual(log_loss, 0.1642520335, places=6)
        self.assertAlmostEqual(brier, 0.025, places=6)
        self.assertAlmostEqual(ece, 0.15, places=6)
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(sum(bin_.weight for bin_ in bins), 1.0, places=6)
        self.assertAlmostEqual(
            brier_parts["reliability"] - brier_parts["resolution"] + brier_parts["uncertainty"],
            brier,
            delta=0.01,
        )
        self.assertIn("R64", by_round)
        self.assertIn("R32", by_round)
        self.assertEqual(report.model_id, "unit_test_model")
        self.assertEqual(report.sample_count, 4)
        self.assertGreater(report.baseline_comparison["baseline_log_loss"], report.log_loss)

        diagram = plot_reliability_diagram(report)
        self.assertIn("bin", diagram)
        self.assertIn("0.0-0.5", diagram)

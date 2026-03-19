from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.config import load_selection_sunday_dataset
from march_madness_2026.engine import BracketPortfolioEngine
from march_madness_2026.reporting import write_outputs


TEST_OVERRIDES = {
    "seed": 20260317,
    "num_tournament_simulations": 180,
    "num_candidate_brackets": 50,
}


class BracketPortfolioEngineTests(unittest.TestCase):
    def test_demo_run_is_reproducible(self) -> None:
        first_engine = BracketPortfolioEngine(simulation_overrides=TEST_OVERRIDES)
        second_engine = BracketPortfolioEngine(simulation_overrides=TEST_OVERRIDES)

        first_run = first_engine.run()
        second_run = second_engine.run()

        first_ids = [candidate.bracket_id for candidate in first_run["result"].finalists]
        second_ids = [candidate.bracket_id for candidate in second_run["result"].finalists]

        self.assertEqual(first_ids, second_ids)
        self.assertAlmostEqual(
            first_run["result"].summary.weighted_portfolio_capture_rate,
            second_run["result"].summary.weighted_portfolio_capture_rate,
            places=6,
        )

    def test_portfolio_selection_is_unique_diversified_and_beats_naive(self) -> None:
        engine = BracketPortfolioEngine(simulation_overrides=TEST_OVERRIDES)
        run_bundle = engine.run()
        result = run_bundle["result"]

        finalists = result.finalists
        finalist_ids = [candidate.bracket_id for candidate in finalists]

        self.assertEqual(len(finalists), 5)
        self.assertEqual(len(set(finalist_ids)), 5)
        self.assertGreaterEqual(result.summary.distinct_archetypes, 3)
        self.assertLess(result.summary.average_pairwise_overlap, 1.0)
        self.assertGreaterEqual(
            result.summary.weighted_portfolio_capture_rate,
            result.summary.naive_baseline_capture_rate,
        )
        self.assertGreaterEqual(
            result.summary.weighted_portfolio_first_place_equity,
            result.summary.naive_baseline_first_place_equity,
        )
        # Verify selection_metadata is present in the run bundle
        self.assertIn("selection_metadata", run_bundle)
        self.assertIn("opponent_scores_by_contest", run_bundle)
        self.assertIn("selected_indices", run_bundle)

    def test_alternative_scoring_profile_changes_candidate_ranking(self) -> None:
        engine = BracketPortfolioEngine(simulation_overrides=TEST_OVERRIDES)
        run_bundle = engine.run()
        candidates = run_bundle["candidates"]

        standard_top = [
            candidate.bracket_id
            for candidate in sorted(
                candidates,
                key=lambda item: item.contest_metrics["standard_mid"].first_place_equity,
                reverse=True,
            )[:5]
        ]
        upset_top = [
            candidate.bracket_id
            for candidate in sorted(
                candidates,
                key=lambda item: item.contest_metrics["upset_bonus_large"].first_place_equity,
                reverse=True,
            )[:5]
        ]

        self.assertNotEqual(standard_top, upset_top)

    def test_report_writer_emits_required_files(self) -> None:
        engine = BracketPortfolioEngine(simulation_overrides=TEST_OVERRIDES)
        run_bundle = engine.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = write_outputs(Path(temp_dir), run_bundle)

            self.assertTrue(output_paths["portfolio_json"].exists())
            self.assertTrue(output_paths["report"].exists())
            self.assertTrue(output_paths["dashboard_spec"].exists())
            self.assertEqual(len([key for key in output_paths if key.startswith("bracket_")]), 5)


class V9SnapshotIntegrationTests(unittest.TestCase):
    """V9: Real snapshot integration test."""

    SNAPSHOT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"

    @unittest.skipUnless(
        (PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json").exists(),
        "snapshot.json not found",
    )
    def test_snapshot_run_produces_valid_portfolio(self) -> None:
        """Load snapshot.json, run with sims=60, cands=40, verify portfolio quality."""
        dataset = load_selection_sunday_dataset(self.SNAPSHOT_PATH)
        engine = BracketPortfolioEngine(
            simulation_overrides={
                "num_tournament_simulations": 60,
                "num_candidate_brackets": 40,
            }
        )
        run_bundle = engine.run(dataset=dataset)
        result = run_bundle["result"]

        # All finalists should have positive FPE
        for finalist in result.finalists:
            self.assertGreater(
                finalist.weighted_first_place_equity, 0.0,
                f"Finalist {finalist.bracket_id} has non-positive FPE",
            )

        # Portfolio FPE >= naive FPE
        self.assertGreaterEqual(
            result.summary.weighted_portfolio_first_place_equity,
            result.summary.naive_baseline_first_place_equity,
        )

        # Scenario summaries contain portfolio_first_place_equity
        for contest_id, summary in result.scenario_summary.items():
            self.assertIn(
                "portfolio_first_place_equity", summary,
                f"Scenario {contest_id} missing portfolio_first_place_equity",
            )
            self.assertGreaterEqual(summary["portfolio_first_place_equity"], 0.0)

        # simulation_config contains contest_profiles
        sim_config = run_bundle["simulation_config"]
        self.assertIn("contest_profiles", sim_config)
        self.assertGreater(len(sim_config["contest_profiles"]), 0)

        # Engine version is v9
        self.assertEqual(run_bundle["engine_version"], "v9")


if __name__ == "__main__":
    unittest.main()


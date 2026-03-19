from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.v10.engine import V10BracketPortfolioEngine
from scripts.autobracket_v10 import V10SearchParams, apply_params_to_engine, main


class AutobracketV10SmokeTests(unittest.TestCase):
    def test_main_writes_v10_metrics_and_explicit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            game_model_artifact = temp_root / "artifacts" / "game.pkl"
            public_field_artifact = temp_root / "artifacts" / "public.pkl"

            exit_code = main(
                [
                    "--use-demo",
                    "--iterations",
                    "1",
                    "--batch",
                    "1",
                    "--sims",
                    "4",
                    "--candidates",
                    "6",
                    "--scale",
                    "0.05",
                    "--training-profile-id",
                    "fast_debug",
                    "--num-eval-seeds",
                    "1",
                    "--game-model-artifact",
                    str(game_model_artifact),
                    "--public-field-artifact",
                    str(public_field_artifact),
                    "--log-dir",
                    str(log_dir),
                    "--seed",
                    "13",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(game_model_artifact.exists())
            self.assertTrue(public_field_artifact.exists())

            checkpoint = log_dir / "best_v10_params.json"
            tsv_path = log_dir / "experiments_v10.tsv"
            self.assertTrue(checkpoint.exists())
            self.assertTrue(tsv_path.exists())

            with checkpoint.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["contest_mode"], "blended")
            self.assertEqual(saved["evaluation_seeds"], [13])
            self.assertEqual(saved["release_objective_name"], "v10_6_release_objective")
            self.assertIn("release_contract", saved)
            self.assertIn("guardrail_failures", saved)
            self.assertAlmostEqual(saved["score"], saved["metrics"]["release_objective_score"], places=6)

            with tsv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)

            self.assertIn("fpe", header)
            self.assertIn("expected_payout", header)
            self.assertIn("game_model_artifact", header)
            self.assertIn("public_field_artifact", header)
            self.assertIn("average_overlap", header)
            self.assertIn("average_duplication_proxy", header)
            self.assertIn("guardrail_failures", header)

    def test_apply_params_to_engine_respects_contest_mode_and_normalizes_weights(self) -> None:
        engine = V10BracketPortfolioEngine(
            simulation_overrides={
                "seed": 17,
                "num_tournament_simulations": 4,
                "num_candidate_brackets": 6,
                "portfolio_size": 5,
            },
            training_profile_id="fast_debug",
        )
        params = V10SearchParams(
            small_weight=99.0,
            mid_weight=1.0,
            large_weight=1.0,
            mix_high_confidence=5.0,
            mix_balanced=4.0,
            mix_selective_contrarian=3.0,
            mix_underdog_upside=2.0,
            mix_high_risk_high_return=1.0,
        )

        apply_params_to_engine(engine, params, contest_mode="standard_large")

        self.assertEqual(engine.simulation_profile.primary_contests, {"standard_large": 1.0})
        self.assertAlmostEqual(sum(engine.simulation_profile.archetype_mix.values()), 1.0, places=6)
        self.assertGreaterEqual(min(engine.simulation_profile.archetype_mix.values()), 0.0)

    def test_main_persists_multi_seed_and_contest_mode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"

            exit_code = main(
                [
                    "--use-demo",
                    "--iterations",
                    "1",
                    "--batch",
                    "1",
                    "--sims",
                    "4",
                    "--candidates",
                    "6",
                    "--scale",
                    "0.05",
                    "--training-profile-id",
                    "fast_debug",
                    "--contest-mode",
                    "standard_large",
                    "--num-eval-seeds",
                    "2",
                    "--log-dir",
                    str(log_dir),
                    "--seed",
                    "21",
                ]
            )

            self.assertEqual(exit_code, 0)
            with (log_dir / "best_v10_params.json").open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            self.assertEqual(checkpoint["contest_mode"], "standard_large")
            self.assertEqual(checkpoint["evaluation_seeds"], [21, 22])
            self.assertEqual(checkpoint["metrics"]["evaluation_seed_count"], 2.0)
            self.assertEqual(
                checkpoint["release_contract"]["objective_name"],
                "v10_6_release_objective",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.engine import BracketPortfolioEngine
from march_madness_2026.v10.reporting import write_outputs


class V10ReportingTests(unittest.TestCase):
    def test_v10_report_writer_emits_integrated_outputs(self) -> None:
        engine = BracketPortfolioEngine(
            simulation_overrides={
                "seed": 20260318,
                "num_tournament_simulations": 12,
                "num_candidate_brackets": 10,
            }
        )
        run_bundle = engine.run()
        run_bundle["engine_version"] = "v10"
        run_bundle["simulation_config"] = {
            "seed": 20260318,
            "candidate_generation_seed": 20260318,
            "num_tournament_simulations": 12,
            "num_candidate_brackets": 10,
            "portfolio_size": 5,
            "release_evaluation_seed_count": 2,
            "release_evaluation_seeds": [20260318, 20260319],
            "release_contract": {
                "objective_name": "v10_6_release_objective",
                "blended_weight_floor": 0.1,
            },
            "training_profile_id": "default_v10",
            "model_artifact_path": "data/models/v10/game_model_v10_default.pkl",
            "public_field_artifact_path": "data/models/v10/public_field_v10_default.pkl",
        }
        run_bundle["selection_metadata"] = {
            "selection_objective": "v10_6_release_objective",
            "candidate_pool_size": 10,
            "positive_equity_pool_size": 10,
            "practical_positive_equity_pool_size": 10,
            "used_zero_equity_fallback": False,
            "local_search_swaps": 0,
            "passes_release_gates": True,
            "portfolio_guardrail_failures": [],
            "portfolio_min_individual_fpe": 0.012,
            "portfolio_release_objective_score": 4.2,
            "naive_release_objective_score": 3.1,
        }
        run_bundle["v10_artifacts"] = {
            "game_model_artifact": "data/models/v10/game_model_v10_default.pkl",
            "public_field_artifact": "data/models/v10/public_field_v10_default.pkl",
            "portfolio_release_evaluation": {
                "release_objective_score": 4.2,
                "guardrail_failures": [],
                "min_individual_fpe": 0.012,
                "average_duplication_proxy": 0.15,
                "final_four_repeat_penalty": 0.12,
                "region_winner_repeat_penalty": 0.08,
            },
            "naive_release_evaluation": {
                "release_objective_score": 3.1,
                "guardrail_failures": [],
            },
            "team_posteriors": {
                "team_posteriors": [
                    {
                        "team_id": "demo-east-1",
                        "team_name": "Demo East 1",
                        "mean_win_probability": 0.71,
                        "mean_uncertainty": 0.08,
                        "posterior_lower_10": 0.61,
                        "posterior_upper_90": 0.79,
                    }
                ]
            },
            "public_duplication": {
                "demo-east-1": {
                    "expected_title_duplicates": 42.0,
                    "expected_final_four_duplicates": 101.0,
                    "path_duplication_proxy": 0.27,
                }
            },
        }
        run_bundle["release_guardrail_failures"] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = write_outputs(Path(temp_dir), run_bundle)
            self.assertTrue(output_paths["portfolio_json"].exists())
            self.assertTrue(output_paths["report"].exists())
            self.assertTrue(output_paths["dashboard_spec"].exists())

            with output_paths["portfolio_json"].open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["engine_version"], "v10")
            self.assertEqual(payload["simulation_config"]["model_artifact_path"], "data/models/v10/game_model_v10_default.pkl")
            self.assertEqual(payload["simulation_config"]["public_field_artifact_path"], "data/models/v10/public_field_v10_default.pkl")
            self.assertIn("v10_artifacts", payload)
            self.assertIn("selection_metadata", payload)
            self.assertIn("release_guardrail_failures", payload)
            self.assertEqual(payload["v10_artifacts"]["game_model_artifact"], "data/models/v10/game_model_v10_default.pkl")
            self.assertEqual(payload["v10_artifacts"]["public_field_artifact"], "data/models/v10/public_field_v10_default.pkl")
            self.assertIn("weighted_portfolio_first_place_equity", payload["summary"])
            self.assertIn("naive_baseline_first_place_equity", payload["summary"])
            self.assertEqual(
                payload["weighted_portfolio_first_place_equity"],
                payload["summary"]["weighted_portfolio_first_place_equity"],
            )
            self.assertEqual(
                payload["naive_baseline_first_place_equity"],
                payload["summary"]["naive_baseline_first_place_equity"],
            )
            self.assertIn("portfolio_first_place_equity", payload["scenario_summary"]["standard_small"])
            self.assertEqual(payload["simulation_config"]["release_evaluation_seed_count"], 2)
            self.assertEqual(payload["simulation_config"]["release_contract"]["objective_name"], "v10_6_release_objective")
            self.assertTrue(payload["selection_metadata"]["passes_release_gates"])
            self.assertEqual(len(payload["finalists"]), 5)

            report_text = output_paths["report"].read_text(encoding="utf-8")
            self.assertIn("## Release Contract", report_text)
            self.assertIn("Release evaluation seeds", report_text)
            self.assertIn("Portfolio release objective score", report_text)


if __name__ == "__main__":
    unittest.main()

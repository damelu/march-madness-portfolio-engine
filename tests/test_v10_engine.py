from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.config import load_selection_sunday_dataset
from march_madness_2026.v10.engine import V10BracketPortfolioEngine
from march_madness_2026.v10.inference import materialize_inference_snapshot
from march_madness_2026.v10.portfolio import PRACTICAL_ZERO_FPE_EPSILON, ReleaseContractConfig


class V10InferenceSnapshotTests(unittest.TestCase):
    def test_materialize_inference_snapshot_creates_isolated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / "source.json"
            output_path = temp_root / "inference" / "snapshot.json"
            payload = {
                "season": 2026,
                "tournament": "mens_d1",
                "teams": [
                    {
                        "team_id": "alpha",
                        "team_name": "Alpha",
                        "region": "East",
                        "seed": 1,
                        "rating": 30.0,
                    },
                    {
                        "team_id": "beta",
                        "team_name": "Beta",
                        "region": "East",
                        "seed": 16,
                        "rating": 10.0,
                    },
                ],
                "metadata": {"source": "unit_test"},
            }
            with source_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            written_path = materialize_inference_snapshot(source_path=source_path, output_path=output_path)
            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as handle:
                materialized = json.load(handle)
            self.assertEqual(materialized["metadata"]["source"], "unit_test")
            self.assertEqual(materialized["metadata"]["v10_inference_snapshot"], "true")
            self.assertEqual(materialized["metadata"]["v10_inference_source_snapshot"], str(source_path))


class V10EngineReproducibilityTests(unittest.TestCase):
    def test_v10_run_is_reproducible_with_fixed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset = build_demo_dataset(seed=20260318)
            model_artifact = temp_root / "game_model_v10_6_release.pkl"
            public_artifact = temp_root / "public_field_v10_6_release.pkl"
            overrides = {
                "seed": 20260318,
                "num_tournament_simulations": 12,
                "num_candidate_brackets": 12,
                "portfolio_size": 5,
            }
            bootstrap_engine = V10BracketPortfolioEngine(
                simulation_overrides=overrides,
                training_profile_id="fast_debug",
                model_artifact_path=model_artifact,
                public_field_artifact_path=public_artifact,
            )
            bootstrap_engine.run(dataset=dataset)

            first_engine = V10BracketPortfolioEngine(
                simulation_overrides=overrides,
                training_profile_id="fast_debug",
                model_artifact_path=model_artifact,
                public_field_artifact_path=public_artifact,
            )
            second_engine = V10BracketPortfolioEngine(
                simulation_overrides=overrides,
                training_profile_id="fast_debug",
                model_artifact_path=model_artifact,
                public_field_artifact_path=public_artifact,
            )

            release_seeds = [20260318, 20260319]
            first_run = first_engine.run(dataset=dataset, release_seeds=release_seeds)
            second_run = second_engine.run(dataset=dataset, release_seeds=release_seeds)
            first_ids = [candidate.bracket_id for candidate in first_run["result"].finalists]
            second_ids = [candidate.bracket_id for candidate in second_run["result"].finalists]

            self.assertEqual(first_ids, second_ids)
            self.assertAlmostEqual(
                first_run["result"].summary.weighted_portfolio_expected_payout,
                second_run["result"].summary.weighted_portfolio_expected_payout,
                places=6,
            )
            self.assertEqual(first_run["simulation_config"]["release_evaluation_seeds"], release_seeds)
            self.assertEqual(first_run["selection_metadata"]["selection_objective"], "v10_6_release_objective")
            self.assertEqual(first_run["simulation_config"]["model_artifact_path"], str(model_artifact))
            self.assertEqual(first_run["simulation_config"]["public_field_artifact_path"], str(public_artifact))


class V10EngineArtifactPathPreservationTests(unittest.TestCase):
    @unittest.skipUnless(
        PROJECT_ROOT.exists(),
        "project root not found",
    )
    def test_v10_run_preserves_explicit_artifact_paths_when_creating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            model_artifact = temp_root / "artifacts" / "custom_game_model.pkl"
            public_field_artifact = temp_root / "artifacts" / "custom_public_field.pkl"

            engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 20260318,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="fast_debug",
                model_artifact_path=model_artifact,
                public_field_artifact_path=public_field_artifact,
            )
            run_bundle = engine.run(dataset=build_demo_dataset(seed=20260318))

            self.assertTrue(model_artifact.exists())
            self.assertTrue(public_field_artifact.exists())
            self.assertEqual(run_bundle["simulation_config"]["model_artifact_path"], str(model_artifact))
            self.assertEqual(run_bundle["simulation_config"]["public_field_artifact_path"], str(public_field_artifact))
            self.assertEqual(run_bundle["v10_artifacts"]["game_model_artifact"], str(model_artifact))
            self.assertEqual(run_bundle["v10_artifacts"]["public_field_artifact"], str(public_field_artifact))
            self.assertIn("team_posteriors", run_bundle["v10_artifacts"])
            self.assertIn("public_duplication", run_bundle["v10_artifacts"])

    def test_non_demo_run_rejects_missing_release_game_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset = build_demo_dataset(seed=20260318).model_copy(
                update={"metadata": {"source": "unit_test_non_demo"}}
            )
            engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 20260318,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="release_v10_6",
                model_artifact_path=temp_root / "missing_game.pkl",
                public_field_artifact_path=temp_root / "missing_public.pkl",
            )

            with self.assertRaisesRegex(RuntimeError, "require a prebuilt release-eligible game model artifact"):
                engine.run(dataset=dataset)

    def test_non_demo_run_falls_back_to_snapshot_public_mode_when_public_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset = build_demo_dataset(seed=7).model_copy(update={"metadata": {"source": "unit_test_non_demo"}})
            artifact_dir = temp_root / "models"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            game_artifact = artifact_dir / "game_model_v10_6_release.pkl"
            public_artifact = artifact_dir / "public_field_v10_6_release.pkl"
            engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 7,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="fast_debug",
                model_artifact_path=game_artifact,
                public_field_artifact_path=public_artifact,
            )
            demo_bundle = engine.run(dataset=build_demo_dataset(seed=7))
            game_manifest = game_artifact.with_name(f"{game_artifact.stem}_manifest.json")
            manifest = json.loads(game_manifest.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "training_profile_id": "release_v10_6",
                    "training_data_mode": "real_only",
                    "release_eligible": True,
                }
            )
            game_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            release_engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 7,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="release_v10_6",
                model_artifact_path=game_artifact,
                public_field_artifact_path=public_artifact,
            )
            run_bundle = release_engine.run(dataset=dataset, release_contract=ReleaseContractConfig(fail_on_guardrail=False))

            self.assertIsNone(run_bundle["simulation_config"]["public_field_artifact_path"])
            self.assertEqual(run_bundle["simulation_config"]["public_field_runtime_mode"], "snapshot_fallback")
            self.assertEqual(
                run_bundle["simulation_config"]["artifact_provenance"]["public_field"]["runtime_usage"],
                "snapshot_fallback",
            )


class V10ReleaseContractTests(unittest.TestCase):
    def test_release_contract_is_exposed_and_drives_summary_objective(self) -> None:
        engine = V10BracketPortfolioEngine(
            simulation_overrides={
                "seed": 20260318,
                "num_tournament_simulations": 4,
                "num_candidate_brackets": 8,
                "portfolio_size": 5,
            },
            training_profile_id="fast_debug",
        )

        run_bundle = engine.run(
            dataset=build_demo_dataset(seed=20260318),
            release_seeds=[20260318, 20260319],
        )

        release_eval = run_bundle["v10_artifacts"]["portfolio_release_evaluation"]
        self.assertAlmostEqual(
            run_bundle["result"].summary.weighted_portfolio_objective,
            release_eval.release_objective_score,
            places=6,
        )
        if not run_bundle["selection_metadata"]["used_zero_equity_fallback"]:
            self.assertTrue(
                all(
                    candidate.weighted_first_place_equity > PRACTICAL_ZERO_FPE_EPSILON
                    for candidate in run_bundle["result"].finalists
                )
            )
        self.assertEqual(run_bundle["simulation_config"]["release_evaluation_seed_count"], 2)
        self.assertEqual(run_bundle["simulation_config"]["release_evaluation_seeds"], [20260318, 20260319])
        self.assertIn("release_contract", run_bundle["simulation_config"])

    def test_strict_blended_weight_floor_triggers_release_guardrail_failure(self) -> None:
        engine = V10BracketPortfolioEngine(
            simulation_overrides={
                "seed": 11,
                "num_tournament_simulations": 4,
                "num_candidate_brackets": 8,
                "portfolio_size": 5,
            },
            training_profile_id="fast_debug",
        )
        release_contract = ReleaseContractConfig(blended_weight_floor=0.40)

        run_bundle = engine.run(
            dataset=build_demo_dataset(seed=11),
            release_contract=release_contract,
            release_seeds=[11],
        )

        self.assertIn("blended_weight_floor", run_bundle["release_guardrail_failures"])
        self.assertFalse(run_bundle["selection_metadata"]["passes_release_gates"])


if __name__ == "__main__":
    unittest.main()

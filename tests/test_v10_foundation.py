from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from march_madness_2026.config import load_contest_profiles, load_payout_profiles, load_training_profiles
from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.game_model import build_matchup_row, prepare_historical_rows, train_game_model
from march_madness_2026.historical import (
    build_historical_games_dataset,
    build_historical_selection_sunday_snapshots,
    load_historical_games_dataset,
    load_historical_snapshot_dataset,
    save_historical_games_dataset,
    season_blocked_splits,
)
from march_madness_2026.v10.entrypoints import build_materialized_v10_argv
from march_madness_2026.v10.inference import DEFAULT_SOURCE_SNAPSHOT, materialize_inference_snapshot
from march_madness_2026.models import HistoricalGameRow, TeamSnapshot
from march_madness_2026.v10.simulation import V10TournamentModel


class V10ConfigLoaderTests(unittest.TestCase):
    def test_training_and_payout_profiles_load_from_repo_configs(self) -> None:
        training_profiles = load_training_profiles()
        payout_profiles = load_payout_profiles()
        contest_profiles = load_contest_profiles(payout_profiles=payout_profiles)

        self.assertIn("default_v10", training_profiles)
        self.assertIn("empirical_v10_5", training_profiles)
        self.assertIn("empirical_v10_5_no_context", training_profiles)
        self.assertIn("release_v10_6", training_profiles)
        self.assertIn("release_v10_6_pruned", training_profiles)
        self.assertIn("release_v10_6_1", training_profiles)
        self.assertEqual(training_profiles["default_v10"].calibration_method, "isotonic")
        self.assertEqual(
            list(training_profiles["empirical_v10_5"].feature_sets),
            ["ratings_base", "four_factors", "matchup_interactions", "context", "uncertainty"],
        )
        self.assertEqual(
            list(training_profiles["empirical_v10_5_no_context"].feature_sets),
            ["ratings_base", "four_factors", "matchup_interactions", "uncertainty"],
        )
        self.assertEqual(
            list(training_profiles["release_v10_6"].feature_sets),
            ["ratings_base", "four_factors", "matchup_interactions", "context", "uncertainty"],
        )
        self.assertEqual(
            list(training_profiles["release_v10_6_pruned"].feature_sets),
            ["ratings_base", "matchup_interactions", "context", "uncertainty"],
        )
        self.assertEqual(
            list(training_profiles["release_v10_6_1"].feature_sets),
            ["ratings_base", "four_factors", "matchup_interactions", "context", "uncertainty"],
        )
        self.assertNotIn("public_field", training_profiles["release_v10_6"].feature_sets)
        self.assertEqual(training_profiles["release_v10_6_1"].train_seasons, [2019, 2021, 2022, 2023])
        self.assertEqual(training_profiles["release_v10_6_1"].validation_seasons, [2024])
        self.assertEqual(training_profiles["release_v10_6_1"].holdout_seasons, [2025])
        self.assertIn("winner_take_all_small", payout_profiles)
        self.assertEqual(payout_profiles["winner_take_all_small"].payout_curve[1], 1.0)
        self.assertEqual(contest_profiles["standard_small"].payout_profile, "winner_take_all_small")
        self.assertEqual(contest_profiles["standard_large"].field_behavior_profile, "large_pool_public")


class HistoricalHelpersTests(unittest.TestCase):
    def test_historical_helpers_round_trip_rows_and_snapshots(self) -> None:
        game_rows = build_historical_games_dataset(
            [
                {
                    "season": 2024,
                    "game_date": "2024-03-21",
                    "tournament_round": "R64",
                    "team_a_id": "A",
                    "team_b_id": "B",
                    "team_a_seed": 1,
                    "team_b_seed": 16,
                    "team_a_features": {"rating": 28.0},
                    "team_b_features": {"rating": 11.0},
                    "context_features": {"travel_delta": 150.0},
                    "public_features": {"team_a_pick_rate": 0.97},
                    "result_team_a_win": True,
                }
            ]
        )
        self.assertIsInstance(game_rows[0], HistoricalGameRow)

        snapshot_rows = build_historical_selection_sunday_snapshots(
            [
                {
                    "season": 2024,
                    "teams": [
                        TeamSnapshot(
                            team_id="A",
                            team_name="Alpha",
                            region="East",
                            seed=1,
                            rating=28.0,
                        )
                    ],
                    "metadata": {"source": "unit_test"},
                }
            ]
        )
        self.assertIn(2024, snapshot_rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            games_path = tmp_root / "games.json"
            snapshots_root = tmp_root / "snapshots"

            save_historical_games_dataset(game_rows, games_path)
            loaded_games = load_historical_games_dataset(games_path)
            self.assertEqual(len(loaded_games), 1)
            self.assertEqual(loaded_games[0].team_a_id, "A")

            for row in snapshot_rows.values():
                build_historical_selection_sunday_snapshots([row], output_dir=snapshots_root)
            loaded_snapshot = load_historical_snapshot_dataset(2024, snapshot_dir=snapshots_root)
            self.assertEqual(loaded_snapshot.metadata["source"], "unit_test")
            self.assertEqual(loaded_snapshot.teams[0].team_name, "Alpha")

    def test_season_blocked_splits_are_ordered_and_non_overlapping(self) -> None:
        splits = season_blocked_splits(
            seasons=[2018, 2019, 2021, 2022, 2023, 2024, 2025],
            train_window=3,
            validation_size=1,
            holdout_size=1,
            min_train_seasons=3,
        )

        self.assertGreaterEqual(len(splits), 1)
        first_split = splits[0]
        self.assertEqual(first_split["train"], [2018, 2019, 2021])
        self.assertEqual(first_split["validation"], [2022])
        self.assertEqual(first_split["holdout"], [2023])
        self.assertTrue(set(first_split["train"]).isdisjoint(first_split["validation"]))
        self.assertTrue(set(first_split["validation"]).isdisjoint(first_split["holdout"]))


class V10InferenceMaterializationTests(unittest.TestCase):
    def test_materialize_inference_snapshot_writes_to_isolated_v10_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source_path = tmp_root / "source_snapshot.json"
            output_path = tmp_root / "data" / "models" / "v10" / "inference" / "2026" / "snapshot.json"

            with DEFAULT_SOURCE_SNAPSHOT.open("r", encoding="utf-8") as handle:
                source_payload = handle.read()
            source_path.write_text(source_payload, encoding="utf-8")

            materialized_path = materialize_inference_snapshot(
                source_path=source_path,
                output_path=output_path,
            )

            self.assertEqual(materialized_path, output_path)
            self.assertTrue(materialized_path.exists())
            self.assertEqual(source_path.read_text(encoding="utf-8"), source_payload)

            with materialized_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["metadata"]["v10_inference_snapshot"], "true")
            self.assertEqual(
                payload["metadata"]["v10_inference_source_snapshot"],
                str(source_path),
            )
            self.assertIn("v10_inference_materialized_at", payload["metadata"])

    def test_materialized_v10_argv_prefers_isolated_snapshot(self) -> None:
        with patch("march_madness_2026.v10.entrypoints.materialize_inference_snapshot", return_value=Path("/tmp/v10/snapshot.json")):
            argv = build_materialized_v10_argv(["--seed", "7"])

        self.assertEqual(argv, ["--dataset", "/tmp/v10/snapshot.json", "--seed", "7"])
        self.assertEqual(build_materialized_v10_argv(["--use-demo"]), ["--use-demo"])
        self.assertEqual(build_materialized_v10_argv(["--dataset", "/tmp/custom.json"]), ["--dataset", "/tmp/custom.json"])

    def test_run_materialized_v10_build_forwards_isolated_dataset(self) -> None:
        with patch("march_madness_2026.v10.entrypoints.materialize_inference_snapshot", return_value=Path("/tmp/v10/snapshot.json")) as materialize_mock:
            with patch("march_madness_2026.v10.entrypoints.main_v10", return_value=0) as main_mock:
                from march_madness_2026.v10.entrypoints import run_materialized_v10_build

                exit_code = run_materialized_v10_build(["--seed", "7"])

        self.assertEqual(exit_code, 0)
        materialize_mock.assert_called_once()
        main_mock.assert_called_once_with(["--dataset", "/tmp/v10/snapshot.json", "--seed", "7"])


class V10CalibrationHonestyTests(unittest.TestCase):
    def test_prepare_historical_rows_treats_nan_feature_values_as_missing(self) -> None:
        rows = [
            {
                "season": 2025,
                "round_number": 1,
                "team_a_id": "A",
                "team_b_id": "B",
                "team_a_seed": 1,
                "team_b_seed": 16,
                "team_a_offense_rating": float("nan"),
                "team_b_offense_rating": 0.0,
                "team_a_win": 1,
            }
        ]
        prepared = prepare_historical_rows(rows, feature_names=["seed_diff", "offense_rating_diff"])
        self.assertEqual(prepared.matrix.shape, (1, 2))
        self.assertFalse(np.isnan(prepared.matrix).any())
        self.assertEqual(float(prepared.matrix[0, 0]), -15.0)
        self.assertEqual(float(prepared.matrix[0, 1]), 0.0)

    def test_train_game_model_supports_isotonic_calibration(self) -> None:
        dataset = build_demo_dataset(seed=23)
        teams = [team.model_dump(mode="python") for team in dataset.teams[:4]]
        historical_rows = []
        outcomes = [1, 0, 1, 0]
        for season in (2024, 2025):
            for index, outcome in enumerate(outcomes):
                left = teams[index % 4]
                right = teams[(index + 1) % 4]
                row = build_matchup_row(
                    left,
                    right,
                    season=season,
                    round_number=1 + (index % 2),
                    metadata={"source": "unit_test"},
                )
                row["team_a_win"] = outcome
                historical_rows.append(row)

        artifact = train_game_model(
            historical_rows,
            model_id="unit_test_model",
            ensemble_size=1,
            validation_seasons=[2025],
            holdout_seasons=[],
            calibration_method="isotonic",
            backend="numpy",
        )

        self.assertEqual(artifact["calibrator"]["method"], "isotonic")
        self.assertEqual(artifact["metadata"]["calibration_method"], "isotonic")

    def test_train_game_model_rejects_unsupported_calibration_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported calibration_method"):
            train_game_model(
                [],
                model_id="unit_test_model",
                calibration_method="bogus",
            )


class V10SimulationCacheTests(unittest.TestCase):
    def test_state_cache_reuses_probability_bundles_across_instances(self) -> None:
        dataset = build_demo_dataset(seed=41)
        team_count = len(dataset.teams)
        probability_rows = 6 * team_count * (team_count - 1)
        fake_model_artifact = {"model_id": "unit_test_model", "feature_names": []}
        fake_public_artifact = {"model_id": "unit_test_public"}

        V10TournamentModel._STATE_CACHE.clear()
        with patch(
            "march_madness_2026.v10.simulation.predict_game_probabilities",
            return_value={"probabilities": [0.5] * probability_rows},
        ) as predict_mock, patch(
            "march_madness_2026.v10.simulation.predict_team_posteriors",
            return_value={"team_posteriors": [], "team_count": 0},
        ) as posteriors_mock, patch(
            "march_madness_2026.v10.simulation.predict_public_advancement_rates",
            return_value={"team_a": {"round_of_32": 0.5}},
        ) as public_mock, patch(
            "march_madness_2026.v10.simulation.estimate_path_duplication",
            return_value={"team_a": 0.5},
        ) as duplication_mock:
            first_model = V10TournamentModel(
                dataset,
                game_model_artifact=fake_model_artifact,
                public_field_artifact=fake_public_artifact,
            )
            second_model = V10TournamentModel(
                dataset,
                game_model_artifact=fake_model_artifact,
                public_field_artifact=fake_public_artifact,
            )

        self.assertEqual(predict_mock.call_count, 1)
        self.assertEqual(posteriors_mock.call_count, 1)
        self.assertEqual(public_mock.call_count, 1)
        self.assertEqual(duplication_mock.call_count, 1)
        self.assertEqual(first_model._round_probability_cache.shape, second_model._round_probability_cache.shape)
        self.assertIs(first_model._round_probability_cache, second_model._round_probability_cache)

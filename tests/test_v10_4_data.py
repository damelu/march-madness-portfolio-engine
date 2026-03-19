from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_v10 import main as backtest_main
from scripts.build_historical_dataset import (
    _fetch_mrchmadness_public_rows,
    build_empirical_rows_from_kaggle_bundle,
    main as build_historical_main,
)
from scripts.train_v10_game_model import main as train_game_main
from scripts.train_v10_public_field import main as train_public_main
from march_madness_2026.config import load_training_profiles
from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.game_model import build_matchup_row


class V104HistoricalDataTests(unittest.TestCase):
    def _build_flat_historical_rows(self, seasons: list[int]) -> list[dict[str, object]]:
        dataset = build_demo_dataset(seed=29)
        teams = [team.model_dump(mode="python") for team in dataset.teams[:4]]
        outcomes = [1, 0, 1, 0]
        rows: list[dict[str, object]] = []
        for season in seasons:
            for index, outcome in enumerate(outcomes):
                left = teams[index % 4]
                right = teams[(index + 1) % 4]
                row = build_matchup_row(
                    left,
                    right,
                    season=season,
                    round_number=1 + (index % 2),
                    metadata={"source_type": "unit_test_real"},
                )
                row["team_a_win"] = outcome
                rows.append(row)
        return rows

    def _write_historical_manifest(self, path: Path, season_modes: dict[int, str]) -> None:
        payload = {
            "tables": {
                "games": {
                    "seasons_detail": {
                        str(season): {"mode": mode, "row_count": 4}
                        for season, mode in season_modes.items()
                    }
                }
            }
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_minimal_kaggle_bundle(self, root: Path) -> Path:
        raw_dir = root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"TeamID": 1, "TeamName": "Alpha"},
                {"TeamID": 2, "TeamName": "Beta"},
                {"TeamID": 3, "TeamName": "Gamma"},
                {"TeamID": 4, "TeamName": "Delta"},
            ]
        ).to_csv(raw_dir / "MTeams.csv", index=False)

        pd.DataFrame(
            [
                {"Season": 2019, "Seed": "W01", "TeamID": 1},
                {"Season": 2019, "Seed": "W16", "TeamID": 2},
                {"Season": 2019, "Seed": "X01", "TeamID": 3},
                {"Season": 2019, "Seed": "X16", "TeamID": 4},
            ]
        ).to_csv(raw_dir / "MNCAATourneySeeds.csv", index=False)

        reg_rows = [
            {
                "Season": 2019,
                "DayNum": 120,
                "WTeamID": 1,
                "WScore": 80,
                "LTeamID": 2,
                "LScore": 60,
                "WLoc": "N",
                "NumOT": 0,
                "WFGM": 28,
                "WFGA": 55,
                "WFGM3": 8,
                "WFGA3": 20,
                "WFTM": 16,
                "WFTA": 22,
                "WOR": 10,
                "WDR": 24,
                "WAst": 15,
                "WTO": 11,
                "WStl": 7,
                "WBlk": 4,
                "WPF": 16,
                "LFGM": 20,
                "LFGA": 52,
                "LFGM3": 6,
                "LFGA3": 18,
                "LFTM": 14,
                "LFTA": 20,
                "LOR": 8,
                "LDR": 20,
                "LAst": 10,
                "LTO": 14,
                "LStl": 4,
                "LBlk": 2,
                "LPF": 18,
            },
            {
                "Season": 2019,
                "DayNum": 121,
                "WTeamID": 3,
                "WScore": 78,
                "LTeamID": 4,
                "LScore": 62,
                "WLoc": "N",
                "NumOT": 0,
                "WFGM": 27,
                "WFGA": 54,
                "WFGM3": 9,
                "WFGA3": 24,
                "WFTM": 15,
                "WFTA": 20,
                "WOR": 9,
                "WDR": 23,
                "WAst": 14,
                "WTO": 10,
                "WStl": 6,
                "WBlk": 3,
                "WPF": 17,
                "LFGM": 21,
                "LFGA": 53,
                "LFGM3": 5,
                "LFGA3": 16,
                "LFTM": 15,
                "LFTA": 21,
                "LOR": 7,
                "LDR": 19,
                "LAst": 9,
                "LTO": 15,
                "LStl": 3,
                "LBlk": 2,
                "LPF": 19,
            },
        ]
        pd.DataFrame(reg_rows).to_csv(raw_dir / "MRegularSeasonDetailedResults.csv", index=False)

        pd.DataFrame(
            [
                {"Season": 2019, "DayNum": 136, "WTeamID": 1, "LTeamID": 2, "WScore": 77, "LScore": 58},
                {"Season": 2019, "DayNum": 136, "WTeamID": 3, "LTeamID": 4, "WScore": 75, "LScore": 61},
            ]
        ).to_csv(raw_dir / "MNCAATourneyDetailedResults.csv", index=False)

        pd.DataFrame(
            [
                {"Season": 2019, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 1, "OrdinalRank": 5},
                {"Season": 2019, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 2, "OrdinalRank": 110},
                {"Season": 2019, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 3, "OrdinalRank": 8},
                {"Season": 2019, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 4, "OrdinalRank": 120},
            ]
        ).to_csv(raw_dir / "MMasseyOrdinals.csv", index=False)

        pd.DataFrame(
            [
                {"Season": 2019, "TeamID": 1, "ConfAbbrev": "A"},
                {"Season": 2019, "TeamID": 2, "ConfAbbrev": "A"},
                {"Season": 2019, "TeamID": 3, "ConfAbbrev": "B"},
                {"Season": 2019, "TeamID": 4, "ConfAbbrev": "B"},
            ]
        ).to_csv(raw_dir / "MTeamConferences.csv", index=False)

        pd.DataFrame(
            [
                {"Season": 2019, "TeamID": 1, "FirstDayNum": 0, "LastDayNum": 154, "CoachName": "Coach A"},
                {"Season": 2019, "TeamID": 2, "FirstDayNum": 0, "LastDayNum": 154, "CoachName": "Coach B"},
                {"Season": 2019, "TeamID": 3, "FirstDayNum": 0, "LastDayNum": 154, "CoachName": "Coach C"},
                {"Season": 2019, "TeamID": 4, "FirstDayNum": 0, "LastDayNum": 154, "CoachName": "Coach D"},
            ]
        ).to_csv(raw_dir / "MTeamCoaches.csv", index=False)

        return raw_dir

    def test_build_empirical_rows_from_kaggle_bundle_emits_real_games_and_proxy_public(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = self._write_minimal_kaggle_bundle(Path(tmp_dir))
            game_rows, public_rows, snapshots, coverage = build_empirical_rows_from_kaggle_bundle(
                raw_dir,
                seasons=[2019],
                public_reference={},
                seed_history={},
            )

        self.assertEqual(coverage["game_seasons"], [2019])
        self.assertEqual(coverage["snapshot_seasons"], [2019])
        self.assertTrue(game_rows)
        self.assertTrue(public_rows)
        self.assertEqual({row["source_type"] for row in game_rows}, {"real_kaggle_tournament"})
        self.assertEqual({row["source_type"] for row in public_rows}, {"heuristic_public_from_real_snapshot"})
        self.assertIn(2019, snapshots)

    def test_build_historical_main_writes_table_level_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = self._write_minimal_kaggle_bundle(root)
            output_dir = root / "models"

            exit_code = build_historical_main(
                [
                    "--kaggle-raw-dir",
                    str(raw_dir),
                    "--output-dir",
                    str(output_dir),
                    "--synthetic-seasons",
                    "2019",
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_dir / "historical_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tables"]["games"]["synthetic_fallback_used"], False)
            self.assertEqual(manifest["tables"]["games"]["seasons_detail"]["2019"]["mode"], "real")
            self.assertEqual(manifest["tables"]["public"]["seasons_detail"]["2019"]["mode"], "synthetic")
            self.assertEqual(manifest["provenance"]["games"]["quality_grade"], "empirical")
            self.assertEqual(manifest["provenance"]["public"]["quality_grade"], "proxy")
            self.assertTrue((output_dir / "historical_games.parquet").exists())
            self.assertTrue((output_dir / "historical_public.parquet").exists())
            self.assertTrue((output_dir / "snapshots" / "2019.json").exists())

    def test_build_historical_main_replaces_proxy_public_rows_with_empirical_espn_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = self._write_minimal_kaggle_bundle(root)
            output_dir = root / "models"
            espn_rows = [
                {
                    "season": 2019,
                    "team_id": "1",
                    "team_name": "Alpha",
                    "seed": 1,
                    "round_name": "champion",
                    "public_pick_pct": 0.42,
                    "public_adv_rate": 0.42,
                    "raw_pick_count": 4200,
                    "source_type": "real_espn_public_api",
                    "source_url": "https://example.test/espn",
                    "snapshot_ts": 1710979200000,
                    "challenge_id": 999,
                }
            ]
            with patch("scripts.build_historical_dataset._fetch_espn_public_picks_rows", return_value=(espn_rows, ["https://example.test/espn"])):
                exit_code = build_historical_main(
                    [
                        "--kaggle-raw-dir",
                        str(raw_dir),
                        "--output-dir",
                        str(output_dir),
                        "--synthetic-seasons",
                        "2019",
                        "--download-espn-public-picks",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_dir / "historical_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tables"]["public"]["seasons_detail"]["2019"]["mode"], "real")
            self.assertEqual(manifest["provenance"]["public"]["quality_grade"], "empirical")
            public_rows = pd.read_parquet(output_dir / "historical_public.parquet").to_dict(orient="records")
            self.assertEqual({row["source_type"] for row in public_rows}, {"real_espn_public_api"})

    def test_build_historical_main_replaces_proxy_public_rows_with_empirical_mrchmadness_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = self._write_minimal_kaggle_bundle(root)
            output_dir = root / "models"
            mrch_rows = [
                {
                    "season": 2019,
                    "team_id": "1",
                    "team_name": "Alpha",
                    "seed": 1,
                    "round_name": "Round of 64",
                    "public_pick_pct": 0.91,
                    "public_adv_rate": 0.91,
                    "source_type": "real_mrchmadness_public_distribution",
                    "source_url": "https://example.test/mrch",
                }
            ]
            with patch(
                "scripts.build_historical_dataset._fetch_mrchmadness_public_rows",
                return_value=(mrch_rows, ["https://example.test/mrch"]),
            ):
                exit_code = build_historical_main(
                    [
                        "--kaggle-raw-dir",
                        str(raw_dir),
                        "--output-dir",
                        str(output_dir),
                        "--synthetic-seasons",
                        "2019",
                        "--download-mrchmadness-public-picks",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_dir / "historical_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tables"]["public"]["seasons_detail"]["2019"]["mode"], "real")
            self.assertEqual(manifest["provenance"]["public"]["quality_grade"], "empirical")
            public_rows = pd.read_parquet(output_dir / "historical_public.parquet").to_dict(orient="records")
            self.assertEqual({row["source_type"] for row in public_rows}, {"real_mrchmadness_public_distribution"})

    def test_fetch_mrchmadness_public_rows_uses_canonical_round_keys(self) -> None:
        snapshots = {
            2021: {
                "teams": [
                    {"team_id": "1", "team_name": "Alpha", "seed": 1},
                    {"team_id": "2", "team_name": "Beta", "seed": 2},
                ]
            }
        }
        frame = pd.DataFrame(
            [
                {
                    "name": "Alpha",
                    "round1": 0.91,
                    "round2": 0.73,
                    "round3": 0.52,
                    "round4": 0.31,
                    "round5": 0.18,
                    "round6": 0.11,
                }
            ]
        )
        with patch("scripts.build_historical_dataset._load_mrchmadness_frame", return_value=frame):
            rows, sources = _fetch_mrchmadness_public_rows([2021], snapshots)

        self.assertEqual(sources, ["https://raw.githubusercontent.com/elishayer/mRchmadness/master/data/pred.pop.men.2021.RData"])
        self.assertEqual(
            [row["round_name"] for row in rows],
            [
                "round_of_32",
                "sweet_16",
                "elite_8",
                "final_four",
                "championship_game",
                "champion",
            ],
        )
        self.assertEqual(
            [row["source_round_name"] for row in rows],
            [
                "Round of 64",
                "Round of 32",
                "Sweet 16",
                "Elite 8",
                "Final Four",
                "National Championship",
            ],
        )

    def test_build_historical_main_replaces_synthetic_game_rows_with_empirical_ncaa_scoreboard_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = self._write_minimal_kaggle_bundle(root)
            output_dir = root / "models"
            scoreboard_rows = [
                {
                    "season": 2025,
                    "round_number": 1,
                    "team_a_id": "1",
                    "team_b_id": "2",
                    "team_a_name": "Alpha",
                    "team_b_name": "Beta",
                    "team_a_seed": 1,
                    "team_b_seed": 16,
                    "team_a_win": 1,
                    "winner_team_id": "1",
                    "loser_team_id": "2",
                    "sample_weight": 1.0,
                    "source_type": "real_ncaa_scoreboard_tournament",
                    "source_url": "https://example.test/ncaa",
                    "game_id": "12345",
                    "bracket_round": "First Round",
                    "bracket_region": "East",
                    "team_a_score": 80,
                    "team_b_score": 60,
                }
            ]
            with patch(
                "scripts.build_historical_dataset._fetch_ncaa_scoreboard_game_rows",
                return_value=(scoreboard_rows, ["https://example.test/ncaa"]),
            ):
                exit_code = build_historical_main(
                    [
                        "--kaggle-raw-dir",
                        str(raw_dir),
                        "--output-dir",
                        str(output_dir),
                        "--synthetic-seasons",
                        "2019,2025",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_dir / "historical_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tables"]["games"]["seasons_detail"]["2025"]["mode"], "real")
            self.assertEqual(manifest["provenance"]["games"]["quality_grade"], "empirical")
            game_rows = pd.read_parquet(output_dir / "historical_games.parquet").to_dict(orient="records")
            game_rows_2025 = [row for row in game_rows if int(row["season"]) == 2025]
            self.assertEqual({row["source_type"] for row in game_rows_2025}, {"real_ncaa_scoreboard_tournament"})

    def test_train_public_field_writes_empirical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = self._write_minimal_kaggle_bundle(root)
            data_dir = root / "models"
            build_historical_main(
                [
                    "--kaggle-raw-dir",
                    str(raw_dir),
                    "--output-dir",
                    str(data_dir),
                    "--synthetic-seasons",
                    "2019",
                ]
            )

            exit_code = train_public_main(
                [
                    "--historical-public",
                    str(data_dir / "historical_public.parquet"),
                    "--output-dir",
                    str(data_dir),
                    "--model-id",
                    "v10_4_test",
                    "--dataset",
                    str(PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((data_dir / "public_field_v10_4_test_manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(manifest["historical_public_row_count"], 0)
            self.assertIn("heuristic_public_from_real_snapshot", manifest["historical_public_source_types"])
            self.assertEqual(manifest["public_history_mode"], "proxy_only")
            self.assertEqual(manifest["effective_historical_public_row_count"], 0)
            self.assertFalse(manifest["release_eligible"])

    def test_backtest_fails_closed_on_non_trainable_row_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bad_rows_path = root / "bad_rows.json"
            bad_rows_path.write_text(
                json.dumps(
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
                            "context_features": {},
                            "public_features": {},
                            "result_team_a_win": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "no evaluable holdout folds"):
                backtest_main(
                    [
                        "--historical-rows",
                        str(bad_rows_path),
                        "--output-dir",
                        str(root / "backtests"),
                    ]
                )

    def test_train_script_uses_training_profile_defaults_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            historical_rows = pd.DataFrame(self._build_flat_historical_rows([2019, 2021, 2022, 2023]))
            historical_path = root / "historical_games.parquet"
            historical_rows.to_parquet(historical_path, index=False)
            self._write_historical_manifest(
                root / "historical_manifest.json",
                {2019: "real", 2021: "real", 2022: "real", 2023: "real"},
            )

            exit_code = train_game_main(
                [
                    "--historical-rows",
                    str(historical_path),
                    "--output-dir",
                    str(root / "models"),
                    "--training-profile-id",
                    "empirical_v10_5",
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest_path = root / "models" / "game_model_v10_5_empirical_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            profile = load_training_profiles()["empirical_v10_5"]
            self.assertEqual(manifest["training_profile_id"], "empirical_v10_5")
            self.assertEqual(manifest["model_id"], profile.model_id)
            self.assertEqual(manifest["feature_families"], list(profile.feature_sets))
            self.assertEqual(manifest["validation_seasons"], list(profile.validation_seasons))
            self.assertEqual(manifest["holdout_seasons"], list(profile.holdout_seasons))
            self.assertEqual(manifest["historical_manifest"]["tables"]["games"]["seasons_detail"]["2023"]["mode"], "real")
            self.assertEqual(manifest["training_data_mode"], "real_only")
            self.assertTrue(manifest["release_eligible"])
            self.assertTrue(Path(manifest["artifact_path"]).exists())

    def test_backtest_script_reports_empirical_and_fallback_summaries_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            historical_rows = pd.DataFrame(self._build_flat_historical_rows([2019, 2021, 2022, 2023]))
            historical_path = root / "historical_games.parquet"
            historical_rows.to_parquet(historical_path, index=False)
            self._write_historical_manifest(
                root / "historical_manifest.json",
                {2019: "real", 2021: "real", 2022: "real", 2023: "synthetic"},
            )

            exit_code = backtest_main(
                [
                    "--historical-rows",
                    str(historical_path),
                    "--output-dir",
                    str(root / "backtests"),
                    "--training-profile-id",
                    "empirical_v10_5_no_context",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary_path = next((root / "backtests").glob("*_v10_backtest_summary.json"))
            markdown_path = next((root / "backtests").glob("*_v10_calibration_report.md"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            profile = load_training_profiles()["empirical_v10_5_no_context"]
            self.assertEqual(summary["training_profile_id"], "empirical_v10_5_no_context")
            self.assertEqual(summary["feature_families"], list(profile.feature_sets))
            self.assertEqual(summary["empirical_only_summary"]["holdout_seasons"], [2022])
            self.assertEqual(summary["fallback_holdout_summary"]["holdout_seasons"], [2023])
            self.assertFalse(summary["release_readiness"]["eligible"])
            self.assertIn("non_real_seasons=2023", summary["release_readiness"]["blocking_issues"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("## Coverage Split", markdown)
            self.assertIn("Training profile", markdown)
            self.assertIn("Release ready", markdown)

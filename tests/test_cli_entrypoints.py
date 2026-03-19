from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.cli import main_v10


class V10CliEntrypointTests(unittest.TestCase):
    def test_main_v10_writes_integrated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_dir = temp_root / "outputs"
            game_model_artifact = temp_root / "game.pkl"
            public_field_artifact = temp_root / "public.pkl"

            exit_code = main_v10(
                [
                    "--use-demo",
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "7",
                    "--num-tournament-sims",
                    "6",
                    "--num-candidates",
                    "8",
                    "--training-profile-id",
                    "fast_debug",
                    "--release-seed-count",
                    "1",
                    "--game-model-artifact",
                    str(game_model_artifact),
                    "--public-field-artifact",
                    str(public_field_artifact),
                    "--allow-guardrail-failure",
                ]
            )

            self.assertEqual(exit_code, 0)
            finalists = [path for path in output_dir.iterdir() if path.name.endswith("_finalists.json")]
            self.assertTrue(finalists)
            self.assertTrue(any(path.name.endswith("_report.md") for path in output_dir.iterdir()))
            self.assertTrue(any(path.name.endswith("_dashboard-spec.md") for path in output_dir.iterdir()))
            with finalists[0].open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["engine_version"], "v10.6")
            self.assertIn("simulation_config", payload)
            self.assertIn("v10_artifacts", payload)
            self.assertEqual(payload["simulation_config"]["release_evaluation_seed_count"], 1)
            self.assertIn("release_contract", payload["simulation_config"])


if __name__ == "__main__":
    unittest.main()

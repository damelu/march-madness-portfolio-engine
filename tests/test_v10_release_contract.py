from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.cli import _run_v10
from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.v10.engine import V10BracketPortfolioEngine
from march_madness_2026.v10.portfolio import (
    PortfolioReleaseEvaluation,
    ReleaseContractConfig,
    _release_guardrail_failures,
    _selection_rank,
)
from march_madness_2026.v10.search import DEFAULT_BLENDED_WEIGHT_FLOOR, V10SearchParams


class V10ReleaseContractTests(unittest.TestCase):
    def test_search_params_enforce_blended_weight_floor(self) -> None:
        params = V10SearchParams(small_weight=100.0, mid_weight=0.1, large_weight=0.1)
        normalized = params.normalized(blended_weight_floor=0.10)

        self.assertGreaterEqual(normalized.small_weight, 0.10)
        self.assertGreaterEqual(normalized.mid_weight, 0.10)
        self.assertGreaterEqual(normalized.large_weight, 0.10)
        self.assertAlmostEqual(
            normalized.small_weight + normalized.mid_weight + normalized.large_weight,
            1.0,
            places=6,
        )

    def test_engine_run_exposes_release_seed_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 17,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="fast_debug",
                model_artifact_path=temp_root / "game.pkl",
                public_field_artifact_path=temp_root / "public.pkl",
            )
            run_bundle = engine.run(
                dataset=build_demo_dataset(seed=17),
                release_seeds=[17, 18],
                release_contract=ReleaseContractConfig(fail_on_guardrail=False),
            )

        selection_metadata = run_bundle["selection_metadata"]
        release_eval = run_bundle["v10_artifacts"]["portfolio_release_evaluation"]
        self.assertEqual(run_bundle["simulation_config"]["release_evaluation_seeds"], [17, 18])
        self.assertEqual(run_bundle["simulation_config"]["release_evaluation_seed_count"], 2)
        self.assertIn("-r2-", run_bundle["result"].run_id)
        self.assertAlmostEqual(
            run_bundle["result"].summary.weighted_portfolio_objective,
            release_eval.release_objective_score,
            places=6,
        )
        self.assertAlmostEqual(
            selection_metadata["portfolio_release_objective_score"],
            release_eval.release_objective_score,
            places=6,
        )
        self.assertEqual(
            run_bundle["release_guardrail_failures"],
            selection_metadata["portfolio_guardrail_failures"],
        )

    def test_cli_fails_closed_when_release_guardrails_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            engine = V10BracketPortfolioEngine(
                simulation_overrides={
                    "seed": 19,
                    "num_tournament_simulations": 4,
                    "num_candidate_brackets": 6,
                    "portfolio_size": 5,
                },
                training_profile_id="fast_debug",
                model_artifact_path=temp_root / "game.pkl",
                public_field_artifact_path=temp_root / "public.pkl",
            )
            run_bundle = engine.run(
                dataset=build_demo_dataset(seed=19),
                release_seeds=[19],
                release_contract=ReleaseContractConfig(fail_on_guardrail=False),
            )
            run_bundle["selection_metadata"]["portfolio_guardrail_failures"] = ["naive_fpe_regression"]
            run_bundle["release_guardrail_failures"] = ["naive_fpe_regression"]

            args = argparse.Namespace(
                use_demo=True,
                dataset=temp_root / "unused.json",
                source_snapshot=temp_root / "unused-source.json",
                output_dir=temp_root / "outputs",
                seed=19,
                num_tournament_sims=4,
                num_candidates=6,
                training_profile_id="fast_debug",
                contest_mode="blended",
                params_json=None,
                game_model_artifact=temp_root / "game.pkl",
                public_field_artifact=temp_root / "public.pkl",
                payout_profile="",
                release_seed_count=1,
                release_seeds="",
                blended_weight_floor=DEFAULT_BLENDED_WEIGHT_FLOOR,
                allow_guardrail_failure=False,
                allow_naive_regression=False,
                allow_zero_fpe_finalist=False,
            )

            with patch("march_madness_2026.cli.V10BracketPortfolioEngine.run", return_value=run_bundle):
                exit_code = _run_v10(args)

        self.assertEqual(exit_code, 2)

    def test_zero_equity_fallback_is_a_release_guardrail_failure(self) -> None:
        evaluation = PortfolioReleaseEvaluation(
            selected_indices=(0, 1, 2, 3, 4),
            objective_score=12.0,
            release_objective_score=12.0,
            expected_utility=5.0,
            expected_payout=100.0,
            first_place_equity=0.1,
            capture_rate=0.2,
            cash_rate=0.3,
            top3_equity=0.1,
            average_overlap=0.2,
            average_duplication_proxy=0.2,
            final_four_repeat_penalty=0.1,
            region_winner_repeat_penalty=0.1,
            champion_repeat_penalty=0.1,
            distinct_archetypes=3,
            unique_champions=2,
            min_individual_fpe=0.0,
            average_individual_fpe=0.05,
            payoff_correlation=0.0,
            used_zero_equity_fallback=True,
        )

        failures = _release_guardrail_failures(
            evaluation,
            naive_evaluation=None,
            contest_weights={"standard_small": 1.0},
            release_contract=ReleaseContractConfig(),
        )

        self.assertIn("zero_equity_fallback_used", failures)
        self.assertIn("zero_fpe_finalist", failures)

    def test_selection_cache_is_scoped_by_release_context(self) -> None:
        selection_cache = {}
        base_eval = PortfolioReleaseEvaluation(
            selected_indices=(0, 1),
            objective_score=1.0,
            release_objective_score=1.0,
            expected_utility=1.0,
            expected_payout=1.0,
            first_place_equity=0.1,
            capture_rate=0.1,
            cash_rate=0.1,
            top3_equity=0.1,
            average_overlap=0.1,
            average_duplication_proxy=0.1,
            final_four_repeat_penalty=0.0,
            region_winner_repeat_penalty=0.0,
            champion_repeat_penalty=0.0,
            distinct_archetypes=2,
            unique_champions=2,
            min_individual_fpe=0.1,
            average_individual_fpe=0.1,
            payoff_correlation=0.0,
        )

        calls: list[bool] = []

        def fake_eval(*args, **kwargs):
            used_zero_equity_fallback = bool(kwargs.get("used_zero_equity_fallback", False))
            calls.append(used_zero_equity_fallback)
            evaluation = PortfolioReleaseEvaluation(**{**base_eval.__dict__, "guardrail_failures": []})
            evaluation.used_zero_equity_fallback = used_zero_equity_fallback
            evaluation.rank = (1 if used_zero_equity_fallback else 0,)
            return evaluation

        with patch("march_madness_2026.v10.portfolio.evaluate_portfolio_release_contract", side_effect=fake_eval):
            first = _selection_rank(
                [0, 1],
                [],
                {"standard_small": np.zeros((2, 1), dtype=np.float64)},
                {"standard_small": np.zeros((1, 1), dtype=np.float64)},
                {"standard_small": 1.0},
                {},
                np.zeros((2, 2), dtype=np.float32),
                selection_cache,
                0.0,
                ReleaseContractConfig(),
                False,
                None,
                None,
            )
            second = _selection_rank(
                [0, 1],
                [],
                {"standard_small": np.zeros((2, 1), dtype=np.float64)},
                {"standard_small": np.zeros((1, 1), dtype=np.float64)},
                {"standard_small": 1.0},
                {},
                np.zeros((2, 2), dtype=np.float32),
                selection_cache,
                0.0,
                ReleaseContractConfig(),
                True,
                None,
                None,
            )

        self.assertEqual(calls, [False, True])
        self.assertFalse(first.used_zero_equity_fallback)
        self.assertTrue(second.used_zero_equity_fallback)

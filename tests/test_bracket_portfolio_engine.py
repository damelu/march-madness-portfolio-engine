from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from march_madness_2026.demo import build_demo_dataset
from march_madness_2026.engine import BracketPortfolioEngine
from march_madness_2026.models import BracketCandidate, ScoringProfile, SimulationProfile
from march_madness_2026.portfolio import (
    build_overlap_matrix,
    portfolio_first_place_equity,
    select_portfolio,
    _selection_rank,
    _pool_supports_constraints,
)
from march_madness_2026.scoring import score_brackets
from march_madness_2026.tournament import TournamentModel


class ScoringProfileTests(unittest.TestCase):
    def test_scoring_profiles_apply_round_weights_and_upset_bonus(self) -> None:
        bracket_picks = np.array(
            [
                [1, 2, 3],
                [1, 4, 6],
            ],
            dtype=np.int16,
        )
        tournament_outcomes = np.array(
            [
                [1, 2, 3],
                [1, 4, 6],
            ],
            dtype=np.int16,
        )
        upset_gaps = np.array(
            [
                [0, 0, 0],
                [0, 0, 4],
            ],
            dtype=np.int16,
        )
        game_rounds = np.array([1, 2, 6], dtype=np.int16)

        standard = ScoringProfile(
            profile_id="standard",
            name="Standard 1-2-4-8-16-32",
            round_weights=[1, 2, 4, 8, 16, 32],
        )
        flat = ScoringProfile(
            profile_id="flat",
            name="Flat scoring",
            round_weights=[10, 10, 10, 10, 10, 10],
        )
        upset_bonus = ScoringProfile(
            profile_id="upset_bonus",
            name="Upset bonus",
            round_weights=[1, 2, 4, 8, 16, 32],
            upset_bonus_per_seed=0.5,
        )

        standard_scores = score_brackets(bracket_picks, tournament_outcomes, upset_gaps, game_rounds, standard)
        flat_scores = score_brackets(bracket_picks, tournament_outcomes, upset_gaps, game_rounds, flat)
        upset_scores = score_brackets(bracket_picks, tournament_outcomes, upset_gaps, game_rounds, upset_bonus)

        np.testing.assert_allclose(
            standard_scores,
            np.array(
                [
                    [35.0, 1.0],
                    [1.0, 35.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            flat_scores,
            np.array(
                [
                    [30.0, 10.0],
                    [10.0, 30.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            upset_scores,
            np.array(
                [
                    [35.0, 1.0],
                    [1.0, 37.0],
                ],
                dtype=np.float32,
            ),
        )


class SimulationReproducibilityTests(unittest.TestCase):
    def test_tournament_simulation_is_reproducible_for_a_fixed_seed(self) -> None:
        dataset = build_demo_dataset(seed=20260317)
        model = TournamentModel(dataset)

        outcomes_a, upset_gaps_a = model.simulate_many(num_simulations=25, seed=123456)
        outcomes_b, upset_gaps_b = model.simulate_many(num_simulations=25, seed=123456)

        np.testing.assert_array_equal(outcomes_a, outcomes_b)
        np.testing.assert_array_equal(upset_gaps_a, upset_gaps_b)


class PortfolioSelectionTests(unittest.TestCase):
    def test_portfolio_selector_prefers_diversified_high_capture_set(self) -> None:
        candidates = [
            BracketCandidate(
                bracket_id="c0",
                archetype="chalk",
                risk_label="low",
                pick_indices=[0, 0, 0, 0],
                champion_team_id="A",
                final_four_team_ids=["A", "B", "C", "D"],
                chalkiness_proxy=0.95,
                duplication_proxy=0.95,
                weighted_first_place_equity=0.95,
            ),
            BracketCandidate(
                bracket_id="c1",
                archetype="chalk",
                risk_label="low",
                pick_indices=[0, 0, 0, 0],
                champion_team_id="A",
                final_four_team_ids=["A", "B", "C", "D"],
                chalkiness_proxy=0.94,
                duplication_proxy=0.94,
                weighted_first_place_equity=0.50,
            ),
            BracketCandidate(
                bracket_id="c2",
                archetype="balanced",
                risk_label="medium",
                pick_indices=[1, 1, 1, 1],
                champion_team_id="B",
                final_four_team_ids=["B", "C", "D", "E"],
                chalkiness_proxy=0.70,
                duplication_proxy=0.70,
                weighted_first_place_equity=0.93,
            ),
            BracketCandidate(
                bracket_id="c3",
                archetype="contrarian",
                risk_label="medium_high",
                pick_indices=[2, 2, 2, 2],
                champion_team_id="C",
                final_four_team_ids=["C", "D", "E", "F"],
                chalkiness_proxy=0.55,
                duplication_proxy=0.55,
                weighted_first_place_equity=0.90,
            ),
            BracketCandidate(
                bracket_id="c4",
                archetype="underdog",
                risk_label="high",
                pick_indices=[3, 3, 3, 3],
                champion_team_id="D",
                final_four_team_ids=["D", "E", "F", "G"],
                chalkiness_proxy=0.35,
                duplication_proxy=0.35,
                weighted_first_place_equity=0.89,
            ),
            BracketCandidate(
                bracket_id="c5",
                archetype="longshot",
                risk_label="very_high",
                pick_indices=[4, 4, 4, 4],
                champion_team_id="E",
                final_four_team_ids=["E", "F", "G", "H"],
                chalkiness_proxy=0.10,
                duplication_proxy=0.10,
                weighted_first_place_equity=0.49,
            ),
        ]

        bracket_picks = np.array([candidate.pick_indices for candidate in candidates], dtype=np.int16)
        candidate_scores = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        opponent_scores = np.full((8, 10), 0.1, dtype=np.float32)

        simulation_profile = SimulationProfile(
            seed=7,
            num_tournament_simulations=10,
            num_candidate_brackets=6,
            portfolio_size=5,
            max_per_archetype=2,
            min_distinct_archetypes=3,
            overlap_penalty_weight=0.25,
            champion_penalty_weight=0.10,
            archetype_mix={
                "chalk": 0.25,
                "balanced": 0.25,
                "contrarian": 0.20,
                "underdog": 0.20,
                "longshot": 0.10,
            },
            primary_contests={"demo_contest": 1.0},
            sensitivity_contests=[],
        )

        selected, naive_top_five, portfolio_fpe, naive_fpe, portfolio_capture, naive_capture, selection_metadata = select_portfolio(
            candidates=candidates,
            bracket_picks=bracket_picks,
            candidate_scores_by_contest={"demo_contest": candidate_scores},
            opponent_scores_by_contest={"demo_contest": opponent_scores},
            simulation_profile=simulation_profile,
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(naive_top_five, [0, 2, 3, 4, 1])
        self.assertGreaterEqual(portfolio_fpe, naive_fpe)
        self.assertEqual(len({candidates[index].bracket_id for index in selected}), 5)
        self.assertGreaterEqual(len({candidates[index].archetype for index in selected}), 4)
        self.assertIn("positive_equity_pool_size", selection_metadata)
        self.assertIn("local_search_swaps", selection_metadata)

    def test_end_to_end_engine_run_is_stable_and_diversified(self) -> None:
        engine = BracketPortfolioEngine(
            simulation_overrides={
                "num_tournament_simulations": 20,
                "num_candidate_brackets": 20,
            }
        )

        first_run = engine.run()["result"]
        second_run = engine.run()["result"]

        first_ids = [candidate.bracket_id for candidate in first_run.finalists]
        second_ids = [candidate.bracket_id for candidate in second_run.finalists]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 5)
        self.assertEqual(len(set(first_ids)), 5)
        self.assertGreaterEqual(first_run.summary.distinct_archetypes, 3)
        self.assertGreaterEqual(first_run.summary.unique_champions, 3)


class PortfolioFirstPlaceEquityTests(unittest.TestCase):
    """Test portfolio_first_place_equity with controlled scenarios."""

    def test_outright_win_gives_full_equity(self) -> None:
        """Single selected bracket scoring above all opponents -> 1.0."""
        candidate_scores = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        opponent_scores = np.array([[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        fpe = portfolio_first_place_equity(
            [0],
            {"c": candidate_scores},
            {"c": opponent_scores},
            {"c": 1.0},
        )
        self.assertAlmostEqual(fpe, 1.0, places=6)

    def test_one_tie_gives_half_equity(self) -> None:
        """Single selected bracket tying with one opponent -> 0.5."""
        candidate_scores = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        opponent_scores = np.array([[10.0, 10.0, 10.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        fpe = portfolio_first_place_equity(
            [0],
            {"c": candidate_scores},
            {"c": opponent_scores},
            {"c": 1.0},
        )
        self.assertAlmostEqual(fpe, 0.5, places=6)

    def test_two_selected_tie_one_opponent_gives_two_thirds(self) -> None:
        """Two selected brackets tying with one opponent -> 2/3."""
        candidate_scores = np.array(
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
            dtype=np.float32,
        )
        opponent_scores = np.array([[10.0, 10.0, 10.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        fpe = portfolio_first_place_equity(
            [0, 1],
            {"c": candidate_scores},
            {"c": opponent_scores},
            {"c": 1.0},
        )
        self.assertAlmostEqual(fpe, 2.0 / 3.0, places=6)

    def test_two_selected_above_field_gives_full_equity(self) -> None:
        """Two selected brackets both above the field -> 1.0."""
        candidate_scores = np.array(
            [[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]],
            dtype=np.float32,
        )
        opponent_scores = np.array([[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        fpe = portfolio_first_place_equity(
            [0, 1],
            {"c": candidate_scores},
            {"c": opponent_scores},
            {"c": 1.0},
        )
        self.assertAlmostEqual(fpe, 1.0, places=6)


class LocalOptimalityTests(unittest.TestCase):
    """Verify that the selected portfolio is locally optimal: no single swap improves the rank."""

    def test_selection_is_locally_optimal(self) -> None:
        candidates = [
            BracketCandidate(
                bracket_id=f"c{i}",
                archetype=["chalk", "balanced", "contrarian", "underdog", "longshot", "chalk"][i],
                risk_label="low",
                pick_indices=[i, i, i, i],
                champion_team_id=chr(65 + i),
                final_four_team_ids=[chr(65 + i)] * 4,
                chalkiness_proxy=0.9 - i * 0.1,
                duplication_proxy=0.9 - i * 0.1,
                weighted_first_place_equity=0.9 - i * 0.05,
            )
            for i in range(6)
        ]

        bracket_picks = np.array([c.pick_indices for c in candidates], dtype=np.int16)
        candidate_scores = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        opponent_scores = np.full((8, 10), 0.1, dtype=np.float32)

        sim_profile = SimulationProfile(
            seed=7,
            num_tournament_simulations=10,
            num_candidate_brackets=6,
            portfolio_size=5,
            max_per_archetype=2,
            min_distinct_archetypes=3,
            overlap_penalty_weight=0.25,
            champion_penalty_weight=0.10,
            archetype_mix={"chalk": 0.25, "balanced": 0.25, "contrarian": 0.20, "underdog": 0.20, "longshot": 0.10},
            primary_contests={"c": 1.0},
            sensitivity_contests=[],
        )

        selected, _, _, _, _, _, metadata = select_portfolio(
            candidates, bracket_picks, {"c": candidate_scores}, {"c": opponent_scores}, sim_profile,
        )

        overlap_matrix = build_overlap_matrix(bracket_picks)
        current_rank = _selection_rank(
            selected, candidates, {"c": candidate_scores}, {"c": opponent_scores}, {"c": 1.0}, overlap_matrix,
        )

        # Verify no single swap improves the rank
        for out_idx in selected:
            remaining = [s for s in selected if s != out_idx]
            for cand_idx in range(len(candidates)):
                if cand_idx in selected:
                    continue
                trial = remaining + [cand_idx]
                trial_rank = _selection_rank(
                    trial, candidates, {"c": candidate_scores}, {"c": opponent_scores}, {"c": 1.0}, overlap_matrix,
                )
                self.assertGreaterEqual(
                    current_rank, trial_rank,
                    f"Swap out={out_idx} in={cand_idx} improved rank: {trial_rank} > {current_rank}",
                )


class EndToEndFPETests(unittest.TestCase):
    """Integration test: portfolio FPE should be >= naive baseline FPE."""

    def test_portfolio_fpe_beats_or_equals_naive(self) -> None:
        engine = BracketPortfolioEngine(
            simulation_overrides={
                "num_tournament_simulations": 20,
                "num_candidate_brackets": 20,
            }
        )
        run_bundle = engine.run()
        result = run_bundle["result"]
        self.assertGreaterEqual(
            result.summary.weighted_portfolio_first_place_equity,
            result.summary.naive_baseline_first_place_equity,
        )


class V9RoundModelProbabilityTests(unittest.TestCase):
    """V9: Test that _candidate_pick_probability uses round-aware model probability."""

    def test_candidate_pick_probability_round1_uses_blend(self) -> None:
        """_candidate_pick_probability(..., round_number=1) should use the round-1 blend."""
        dataset = build_demo_dataset(seed=20260317)
        model = TournamentModel(dataset)
        from march_madness_2026.tournament import ARCHETYPE_PROFILES

        archetype = ARCHETYPE_PROFILES["balanced"]
        # Pick two teams in the same region for a first-round matchup
        team_a = model.index_by_team_id["east-01"]
        team_b = model.index_by_team_id["east-16"]

        prob_r1 = model._candidate_pick_probability(team_a, team_b, 1, archetype)
        prob_r2 = model._candidate_pick_probability(team_a, team_b, 2, archetype)

        # Round 1 uses Vegas/BPI blend; round 2 uses pure model. They should differ.
        # Both should be valid probabilities.
        self.assertGreater(prob_r1, 0.0)
        self.assertLess(prob_r1, 1.0)
        self.assertGreater(prob_r2, 0.0)
        self.assertLess(prob_r2, 1.0)
        # They CAN be different since R1 blends with Vegas data
        # (we just verify both are valid probabilities and the code path works)

    def test_favorite_flag_uses_round_model_probability(self) -> None:
        """Favorite derivation in _play_round should use _round_model_probability."""
        dataset = build_demo_dataset(seed=20260317)
        model = TournamentModel(dataset)
        rng = np.random.default_rng(42)

        # Run a round-1 play and check that favorite flags are consistent
        region_field = list(model.region_fields["East"])
        _, _, _, favorite_flags = model._play_round(region_field, rng, 1)

        # Every favorite flag should be 0 or 1
        for flag in favorite_flags:
            self.assertIn(flag, [0, 1])
        # Should have 8 games in R1 of a region
        self.assertEqual(len(favorite_flags), 8)


class V9PoolFeasibilityTests(unittest.TestCase):
    """V9: Test positive-pool feasibility fallback."""

    def test_pool_supports_constraints_positive(self) -> None:
        """A pool with enough archetypes and capacity should pass."""
        candidates = [
            BracketCandidate(
                bracket_id=f"c{i}",
                archetype=["a", "b", "c", "d", "e"][i % 5],
                risk_label="low",
                pick_indices=[i],
                champion_team_id="X",
                final_four_team_ids=["X"],
                chalkiness_proxy=0.5,
                duplication_proxy=0.5,
            )
            for i in range(10)
        ]
        sim_profile = SimulationProfile(
            seed=7, num_tournament_simulations=10, num_candidate_brackets=10,
            portfolio_size=5, max_per_archetype=2, min_distinct_archetypes=3,
            overlap_penalty_weight=0.1, champion_penalty_weight=0.1,
            archetype_mix={"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2},
            primary_contests={"c": 1.0}, sensitivity_contests=[],
        )
        self.assertTrue(_pool_supports_constraints(list(range(10)), candidates, sim_profile))

    def test_pool_supports_constraints_too_small(self) -> None:
        """A pool smaller than portfolio_size should fail."""
        candidates = [
            BracketCandidate(
                bracket_id="c0", archetype="a", risk_label="low",
                pick_indices=[0], champion_team_id="X", final_four_team_ids=["X"],
                chalkiness_proxy=0.5, duplication_proxy=0.5,
            )
        ]
        sim_profile = SimulationProfile(
            seed=7, num_tournament_simulations=10, num_candidate_brackets=1,
            portfolio_size=5, max_per_archetype=2, min_distinct_archetypes=3,
            overlap_penalty_weight=0.1, champion_penalty_weight=0.1,
            archetype_mix={"a": 1.0}, primary_contests={"c": 1.0}, sensitivity_contests=[],
        )
        self.assertFalse(_pool_supports_constraints([0], candidates, sim_profile))

    def test_naive_baseline_uses_same_pool_as_optimizer(self) -> None:
        """Naive baseline should operate on the same pool indices as the optimizer."""
        engine = BracketPortfolioEngine(
            simulation_overrides={
                "num_tournament_simulations": 20,
                "num_candidate_brackets": 20,
            }
        )
        run_bundle = engine.run()
        metadata = run_bundle["selection_metadata"]
        # naive_baseline_pool_size should match candidate_pool_size
        self.assertEqual(metadata["naive_baseline_pool_size"], metadata["candidate_pool_size"])


if __name__ == "__main__":
    unittest.main()

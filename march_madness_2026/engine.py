from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from .config import load_contest_profiles, load_scoring_profiles, load_simulation_profile
from .demo import build_demo_dataset
from .models import BracketCandidate, PortfolioResult, PortfolioSummary, SelectionSundayDataset, SimulationProfile
from .portfolio import (
    average_pairwise_overlap,
    build_overlap_matrix,
    evaluate_candidates_for_contest,
    portfolio_capture_rate,
    portfolio_first_place_equity,
    select_portfolio,
)
from .scoring import score_brackets
from .tournament import TournamentModel


def _allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    running = {}
    assigned = 0
    fractional: List[tuple[str, float]] = []

    for key, weight in weights.items():
        exact = total * weight
        count = int(exact)
        running[key] = count
        assigned += count
        fractional.append((key, exact - count))

    for key, _ in sorted(fractional, key=lambda item: item[1], reverse=True)[: total - assigned]:
        running[key] += 1
    return running


class BracketPortfolioEngine:
    def __init__(self, simulation_overrides: Dict[str, int] | None = None):
        self.scoring_profiles = load_scoring_profiles()
        self.contest_profiles = load_contest_profiles()
        self.simulation_profile = load_simulation_profile()
        if simulation_overrides:
            self.simulation_profile = self.simulation_profile.model_copy(update=simulation_overrides)

    def _generate_unique_candidates(
        self,
        model: TournamentModel,
        simulation_profile: SimulationProfile,
    ) -> List[BracketCandidate]:
        candidate_counts = _allocate_counts(
            simulation_profile.num_candidate_brackets,
            simulation_profile.archetype_mix,
        )
        # Use well-separated seed to decorrelate from tournament simulation RNG
        rng = np.random.default_rng(simulation_profile.seed ^ 0xDEADBEEF)
        candidates: List[BracketCandidate] = []
        seen = set()
        ordinal_by_archetype = {key: 1 for key in candidate_counts}

        for archetype_name, target_count in candidate_counts.items():
            attempts = 0
            while sum(1 for candidate in candidates if candidate.archetype == archetype_name) < target_count:
                candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
                ordinal_by_archetype[archetype_name] += 1
                attempts += 1
                candidate_key = tuple(candidate.pick_indices)
                if candidate_key in seen:
                    if attempts > target_count * 50:
                        break
                    continue
                seen.add(candidate_key)
                candidates.append(candidate)

        fallback_archetypes = list(candidate_counts)
        while len(candidates) < simulation_profile.num_candidate_brackets:
            archetype_name = fallback_archetypes[len(candidates) % len(fallback_archetypes)]
            candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
            ordinal_by_archetype[archetype_name] += 1
            candidate_key = tuple(candidate.pick_indices)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            candidates.append(candidate)

        return candidates

    def _generate_opponent_field(
        self,
        model: TournamentModel,
        contest_id: str,
    ) -> np.ndarray:
        """V6: Generate opponents using a blend of public pick distribution + archetype profiles.

        In real bracket pools, most opponents follow the public pick distribution
        (heavy on chalk, over-picking Duke/Arizona). We simulate this by generating
        60% of opponents from a public-pick-biased model and 40% from archetype profiles.
        """
        contest_profile = self.contest_profiles[contest_id]
        field_size = contest_profile.simulated_field_size
        stable_seed_offset = int(
            hashlib.blake2b(contest_id.encode("utf-8"), digest_size=4).hexdigest(),
            16,
        )
        rng_seed = (self.simulation_profile.seed ^ 0xCAFEBABE) + (stable_seed_offset % 100_000)
        rng = np.random.default_rng(rng_seed)
        picks: List[List[int]] = []

        # V8: 60% from public candidate generator (simulates ESPN bracket challenge users)
        # Uses public_pick_pct to model how casual fans pick brackets.
        public_count = int(field_size * 0.6)
        for i in range(public_count):
            candidate = model.generate_public_candidate(rng, 20_000 + i)
            picks.append(candidate.pick_indices)

        # 40% from archetype mix (sophisticated opponents)
        remaining = field_size - public_count
        counts = _allocate_counts(remaining, contest_profile.opponent_mix)
        ordinal_by_archetype = {key: 30_000 for key in counts}
        for archetype_name, count in counts.items():
            for _ in range(count):
                candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
                ordinal_by_archetype[archetype_name] += 1
                picks.append(candidate.pick_indices)

        return np.array(picks, dtype=np.int16)

    def run(self, dataset: SelectionSundayDataset | None = None) -> dict:
        active_dataset = dataset or build_demo_dataset(seed=self.simulation_profile.seed)
        model = TournamentModel(active_dataset)
        tournament_outcomes, upset_gaps = model.simulate_many(
            self.simulation_profile.num_tournament_simulations,
            self.simulation_profile.seed,
        )

        candidates = self._generate_unique_candidates(model, self.simulation_profile)
        candidate_pick_matrix = np.array([candidate.pick_indices for candidate in candidates], dtype=np.int16)
        scoring_cache: Dict[str, np.ndarray] = {}
        opponent_scores_by_contest: Dict[str, np.ndarray] = {}
        candidate_scores_by_contest: Dict[str, np.ndarray] = {}

        all_contests = list(self.simulation_profile.primary_contests) + list(self.simulation_profile.sensitivity_contests)

        for contest_id in all_contests:
            contest_profile = self.contest_profiles[contest_id]
            scoring_profile = self.scoring_profiles[contest_profile.scoring_profile]
            if scoring_profile.profile_id not in scoring_cache:
                scoring_cache[scoring_profile.profile_id] = score_brackets(
                    candidate_pick_matrix,
                    tournament_outcomes,
                    upset_gaps,
                    model.game_rounds,
                    scoring_profile,
                )
            candidate_scores_by_contest[contest_id] = scoring_cache[scoring_profile.profile_id]

            opponent_pick_matrix = self._generate_opponent_field(model, contest_id)
            opponent_scores_by_contest[contest_id] = score_brackets(
                opponent_pick_matrix,
                tournament_outcomes,
                upset_gaps,
                model.game_rounds,
                scoring_profile,
            )

            first_place_equities, average_finishes = evaluate_candidates_for_contest(
                candidates,
                contest_id,
                candidate_scores_by_contest[contest_id],
                opponent_scores_by_contest[contest_id],
            )
            if contest_id in self.simulation_profile.primary_contests:
                weight = self.simulation_profile.primary_contests[contest_id]
                for index, candidate in enumerate(candidates):
                    candidate.weighted_first_place_equity += float(first_place_equities[index] * weight)
                    candidate.weighted_average_finish += float(average_finishes[index] * weight)

        selected_indices, naive_baseline_indices, portfolio_fpe, naive_fpe, portfolio_capture, naive_capture, selection_metadata = select_portfolio(
            candidates,
            candidate_pick_matrix,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            self.simulation_profile,
        )
        overlap_matrix = build_overlap_matrix(candidate_pick_matrix)

        finalists = [candidates[index] for index in selected_indices]
        naive_finalists = [candidates[index] for index in naive_baseline_indices]

        scenario_summary: Dict[str, Dict[str, float]] = {}
        for contest_id in self.simulation_profile.primary_contests:
            contest_profile = self.contest_profiles[contest_id]
            best_finalist = max(finalists, key=lambda candidate: candidate.contest_metrics[contest_id].first_place_equity)
            scenario_summary[contest_id] = {
                "portfolio_capture_rate": portfolio_capture_rate(
                    selected_indices,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    {contest_id: 1.0},
                ),
                "portfolio_first_place_equity": portfolio_first_place_equity(
                    selected_indices,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    {contest_id: 1.0},
                ),
                "best_finalist_equity": best_finalist.contest_metrics[contest_id].first_place_equity,
                "field_size": float(contest_profile.simulated_field_size),
            }

        sensitivity_summary: Dict[str, Dict[str, float]] = {}
        for contest_id in self.simulation_profile.sensitivity_contests:
            best_finalist = max(finalists, key=lambda candidate: candidate.contest_metrics[contest_id].first_place_equity)
            sensitivity_summary[contest_id] = {
                "portfolio_capture_rate": portfolio_capture_rate(
                    selected_indices,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    {contest_id: 1.0},
                ),
                "portfolio_first_place_equity": portfolio_first_place_equity(
                    selected_indices,
                    candidate_scores_by_contest,
                    opponent_scores_by_contest,
                    {contest_id: 1.0},
                ),
                "best_finalist_equity": best_finalist.contest_metrics[contest_id].first_place_equity,
            }

        selection_method = "greedy+local_search" if selection_metadata.get("local_search_applied") else "greedy"
        local_search_changed = selection_metadata.get("local_search_applied", False)
        avg_overlap = average_pairwise_overlap(selected_indices, overlap_matrix)
        for finalist in finalists:
            best_fit = max(
                finalist.contest_metrics.items(),
                key=lambda item: item[1].first_place_equity,
            )
            champion_name = model.team_name(finalist.champion_team_id)
            finalist.scenario_fit = best_fit[0]
            finalist.why_selected = (
                f"{champion_name} anchors the champion thesis. "
                f"Weighted first-place equity={finalist.weighted_first_place_equity:.4f}, "
                f"portfolio FPE={portfolio_fpe:.4f}, capture={portfolio_capture:.4f}, "
                f"selection={selection_method}, "
                f"overlap={avg_overlap:.3f}, "
                f"local_search_changed={local_search_changed}, "
                f"chalkiness={finalist.chalkiness_proxy:.3f}, "
                f"best-fit scenario={best_fit[0]}."
            )

        summary = PortfolioSummary(
            weighted_portfolio_capture_rate=portfolio_capture,
            naive_baseline_capture_rate=naive_capture,
            weighted_portfolio_objective=portfolio_fpe,
            naive_baseline_objective=naive_fpe,
            weighted_candidate_baseline=float(
                sum(candidate.weighted_first_place_equity for candidate in naive_finalists) / len(naive_finalists)
            ),
            average_pairwise_overlap=average_pairwise_overlap(selected_indices, overlap_matrix),
            distinct_archetypes=len({candidate.archetype for candidate in finalists}),
            unique_champions=len({candidate.champion_team_id for candidate in finalists}),
            weighted_portfolio_first_place_equity=portfolio_fpe,
            naive_baseline_first_place_equity=naive_fpe,
        )

        result = PortfolioResult(
            run_id=f"{active_dataset.season}-{self.simulation_profile.seed}",
            data_source=active_dataset.metadata.get("source", "selection_sunday_input"),
            candidate_count=len(candidates),
            finalists=finalists,
            summary=summary,
            scenario_summary=scenario_summary,
            sensitivity_summary=sensitivity_summary,
            dashboard_spec_sections=[
                "Scenario switcher for small, mid, and large pool environments",
                "Candidate-universe table with archetype, champion, chalkiness, and weighted first-place equity",
                "Finalist overlap heatmap and champion diversification summary",
                "Selection-report panel showing why each of the 5 finalists survived downselection",
            ],
        )

        contest_profiles_dict = {
            cid: {
                "name": cp.name,
                "scoring_profile": cp.scoring_profile,
                "pool_bucket": cp.pool_bucket,
                "simulated_field_size": cp.simulated_field_size,
            }
            for cid, cp in self.contest_profiles.items()
        }
        scoring_profile_mapping = {
            cid: cp.scoring_profile for cid, cp in self.contest_profiles.items()
        }

        return {
            "result": result,
            "dataset": active_dataset,
            "model": model,
            "candidates": candidates,
            "candidate_scores_by_contest": candidate_scores_by_contest,
            "opponent_scores_by_contest": opponent_scores_by_contest,
            "selected_indices": selected_indices,
            "naive_baseline_indices": naive_baseline_indices,
            "selection_metadata": selection_metadata,
            "engine_version": "v9",
            "simulation_config": {
                "seed": self.simulation_profile.seed,
                "num_tournament_simulations": self.simulation_profile.num_tournament_simulations,
                "num_candidate_brackets": self.simulation_profile.num_candidate_brackets,
                "portfolio_size": self.simulation_profile.portfolio_size,
                "contest_profiles": contest_profiles_dict,
                "scoring_profile_mapping": scoring_profile_mapping,
            },
        }

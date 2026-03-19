from __future__ import annotations

from dataclasses import asdict
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..config import load_selection_sunday_dataset
from ..demo import build_demo_dataset
from ..engine import _allocate_counts
from ..game_model import build_matchup_row, save_model_artifact, train_game_model
from ..models import PortfolioResult, PortfolioSummary, SelectionSundayDataset
from ..portfolio import (
    evaluate_candidates_for_contest,
    portfolio_capture_rate,
    portfolio_first_place_equity,
)
from ..public_field import fit_public_round_model, save_public_field_artifact
from ..scoring import score_brackets
from ..tournament import TournamentModel
from .config import (
    load_v10_contest_profiles,
    load_v10_payout_profiles,
    load_v10_scoring_profiles,
    load_v10_simulation_profile,
    load_v10_training_profiles,
)
from .portfolio import ReleaseContractConfig, select_portfolio, weighted_payout_summary
from .provenance import load_adjacent_artifact_manifest
from .simulation import V10TournamentModel


class V10BracketPortfolioEngine:
    def __init__(
        self,
        simulation_overrides: Dict[str, int] | None = None,
        *,
        training_profile_id: str = "default_v10",
        model_artifact_path: Path | None = None,
        public_field_artifact_path: Path | None = None,
    ):
        self.scoring_profiles = load_v10_scoring_profiles()
        self.payout_profiles = load_v10_payout_profiles()
        self.contest_profiles = load_v10_contest_profiles()
        self.training_profiles = load_v10_training_profiles()
        self.simulation_profile = load_v10_simulation_profile()
        if simulation_overrides:
            self.simulation_profile = self.simulation_profile.model_copy(update=simulation_overrides)
        self.training_profile = self.training_profiles[training_profile_id]
        self.model_artifact_path = model_artifact_path
        self.public_field_artifact_path = public_field_artifact_path
        self._prepared_run_cache: dict[tuple, dict[str, Any]] = {}
        self._model_cache: dict[tuple[int, str, str], V10TournamentModel] = {}

    def _artifact_root(self) -> Path:
        return self._model_root() / "artifacts"

    def _model_root(self) -> Path:
        base_root = Path(__file__).resolve().parents[2] / "data" / "models"
        parts = self.training_profile.model_id.split("_")
        if len(parts) >= 2 and parts[0] == "v10" and parts[1].isdigit():
            return base_root / f"v10_{parts[1]}"
        return base_root / "v10"

    def _default_snapshot_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "features" / "selection_sunday" / "snapshot.json"

    def _bootstrap_historical_rows(self, dataset: SelectionSundayDataset) -> list[dict[str, Any]]:
        base_model = TournamentModel(dataset)
        team_payloads = [team.model_dump(mode="python") for team in dataset.teams]
        seasons = (
            self.training_profile.train_seasons
            + self.training_profile.validation_seasons
            + self.training_profile.holdout_seasons
        )
        unique_seasons = sorted(set(seasons or [dataset.season]))
        rng = np.random.default_rng(self.training_profile.seed)
        rows: list[dict[str, Any]] = []

        for season in unique_seasons:
            for _ in range(max(768, len(team_payloads) * 12)):
                left_index, right_index = rng.choice(len(team_payloads), size=2, replace=False)
                round_number = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.48, 0.22, 0.14, 0.08, 0.05, 0.03]))
                row = build_matchup_row(
                    team_payloads[left_index],
                    team_payloads[right_index],
                    season=season,
                    round_number=round_number,
                    metadata={"source": "heuristic_bootstrap"},
                )
                probability = base_model._round_model_probability(left_index, right_index, round_number)
                row["team_a_win"] = float(rng.random() < probability)
                row["sample_weight"] = 1.0 + (0.05 * (round_number - 1))
                rows.append(row)
        return rows

    def _resolve_existing_artifact(self, explicit_path: Path | None, candidates: list[Path]) -> Path | None:
        if explicit_path is not None and explicit_path.exists():
            return explicit_path
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _is_demo_dataset(self, dataset: SelectionSundayDataset) -> bool:
        return str(dataset.metadata.get("source", "")).startswith("demo_")

    def _ensure_game_model_artifact(self, dataset: SelectionSundayDataset) -> tuple[Path, dict[str, Any]]:
        model_root = self._model_root()
        allow_bootstrap = self._is_demo_dataset(dataset)
        if self.model_artifact_path is not None:
            if self.model_artifact_path.exists():
                manifest = load_adjacent_artifact_manifest(self.model_artifact_path) or {}
                if not allow_bootstrap and not manifest.get("release_eligible", False):
                    raise RuntimeError(
                        f"non-demo V10.6 runs require a release-eligible game model artifact: {self.model_artifact_path}"
                    )
                return self.model_artifact_path, manifest
            artifact_path = self.model_artifact_path
        else:
            artifact_path = self._resolve_existing_artifact(
                None,
                [
                    model_root / f"game_model_{self.training_profile.model_id}.pkl",
                    self._artifact_root() / "game_model" / f"{self.training_profile.model_id}.pkl",
                ],
            )
            if artifact_path is not None:
                manifest = load_adjacent_artifact_manifest(artifact_path) or {}
                if not allow_bootstrap and not manifest.get("release_eligible", False):
                    raise RuntimeError(
                        f"non-demo V10.6 runs require a release-eligible game model artifact: {artifact_path}"
                    )
                return artifact_path, manifest
            artifact_path = self._artifact_root() / "game_model" / f"{self.training_profile.model_id}.pkl"
        if not allow_bootstrap:
            raise RuntimeError(
                "non-demo V10.6 runs require a prebuilt release-eligible game model artifact; runtime bootstrap training is disabled"
            )

        training_root = model_root / "training"
        training_root.mkdir(parents=True, exist_ok=True)
        rows = self._bootstrap_historical_rows(dataset)
        bootstrap_path = training_root / f"{self.training_profile.model_id}_bootstrap_rows.json"
        with bootstrap_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        artifact = train_game_model(
            rows,
            model_id=self.training_profile.model_id,
            seed=self.training_profile.seed,
            ensemble_size=self.training_profile.ensemble_size,
            feature_families=self.training_profile.feature_sets,
            validation_seasons=self.training_profile.validation_seasons,
            holdout_seasons=self.training_profile.holdout_seasons,
            calibration_method=self.training_profile.calibration_method,
            backend=self.training_profile.backend,
        )
        save_model_artifact(artifact, artifact_path)
        bootstrap_manifest = {
            "model_id": self.training_profile.model_id,
            "training_profile_id": self.training_profile.profile_id,
            "training_data_mode": "synthetic_only",
            "release_eligible": False,
            "artifact_path": str(artifact_path),
            "bootstrap_source": "demo_runtime_bootstrap",
        }
        manifest_path = artifact_path.with_name(f"{artifact_path.stem}_manifest.json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(bootstrap_manifest, handle, indent=2)
        return artifact_path, bootstrap_manifest

    def _ensure_public_field_artifact(self, dataset: SelectionSundayDataset) -> tuple[Path | None, dict[str, Any]]:
        model_root = self._model_root()
        allow_bootstrap = self._is_demo_dataset(dataset)
        if self.public_field_artifact_path is not None:
            if self.public_field_artifact_path.exists():
                manifest = load_adjacent_artifact_manifest(self.public_field_artifact_path) or {}
                runtime_mode = "artifact_calibrated" if allow_bootstrap or manifest.get("release_eligible", False) else "snapshot_fallback"
                return (
                    self.public_field_artifact_path if runtime_mode == "artifact_calibrated" else None,
                    {
                        **manifest,
                        "artifact_path": str(self.public_field_artifact_path),
                        "runtime_usage": runtime_mode,
                    },
                )
            artifact_path = self.public_field_artifact_path
        else:
            artifact_path = self._resolve_existing_artifact(
                None,
                [
                    model_root / f"public_field_{self.training_profile.model_id}.pkl",
                    self._artifact_root() / "public_field" / f"{self.training_profile.model_id}_public.pkl",
                ],
            )
            if artifact_path is not None:
                manifest = load_adjacent_artifact_manifest(artifact_path) or {}
                runtime_mode = "artifact_calibrated" if allow_bootstrap or manifest.get("release_eligible", False) else "snapshot_fallback"
                return (
                    artifact_path if runtime_mode == "artifact_calibrated" else None,
                    {
                        **manifest,
                        "artifact_path": str(artifact_path),
                        "runtime_usage": runtime_mode,
                    },
                )
            if not allow_bootstrap:
                return None, {"public_history_mode": "reference_only", "runtime_usage": "snapshot_fallback", "release_eligible": False}
            artifact_path = self._artifact_root() / "public_field" / f"{self.training_profile.model_id}_public.pkl"
        if not allow_bootstrap:
            return None, {"artifact_path": str(artifact_path), "public_history_mode": "reference_only", "runtime_usage": "snapshot_fallback", "release_eligible": False}

        team_payloads = [team.model_dump(mode="python") for team in dataset.teams]
        artifact = fit_public_round_model(
            team_payloads,
            model_id=f"{self.training_profile.model_id}_public",
        )
        save_public_field_artifact(artifact, artifact_path)
        manifest = {
            "model_id": artifact["model_id"],
            "public_history_mode": artifact.get("public_history_mode", "reference_only"),
            "release_eligible": artifact.get("release_eligible", False),
            "artifact_path": str(artifact_path),
            "runtime_usage": "artifact_calibrated" if allow_bootstrap or artifact.get("release_eligible", False) else "snapshot_fallback",
        }
        manifest_path = artifact_path.with_name(f"{artifact_path.stem}_manifest.json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        runtime_path = artifact_path if manifest["runtime_usage"] == "artifact_calibrated" else None
        return runtime_path, manifest

    def _generate_unique_candidates(self, model: V10TournamentModel, *, candidate_seed: int | None = None) -> List:
        candidate_counts = _allocate_counts(
            self.simulation_profile.num_candidate_brackets,
            self.simulation_profile.archetype_mix,
        )
        seed = self.simulation_profile.seed if candidate_seed is None else int(candidate_seed)
        rng = np.random.default_rng(seed ^ 0xABCD1234)
        candidates = []
        seen = set()
        ordinal_by_archetype = {name: 1 for name in candidate_counts}

        for archetype_name, target_count in candidate_counts.items():
            while sum(1 for candidate in candidates if candidate.archetype == archetype_name) < target_count:
                candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
                ordinal_by_archetype[archetype_name] += 1
                key = tuple(candidate.pick_indices)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

        fallback_archetypes = list(candidate_counts)
        while len(candidates) < self.simulation_profile.num_candidate_brackets:
            archetype_name = fallback_archetypes[len(candidates) % len(fallback_archetypes)]
            candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
            ordinal_by_archetype[archetype_name] += 1
            key = tuple(candidate.pick_indices)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _generate_opponent_field(self, model: V10TournamentModel, contest_id: str, *, evaluation_seed: int | None = None) -> np.ndarray:
        contest_profile = self.contest_profiles[contest_id]
        field_size = contest_profile.simulated_field_size
        stable_seed_offset = int(hashlib.blake2b(contest_id.encode("utf-8"), digest_size=4).hexdigest(), 16)
        base_seed = self.simulation_profile.seed if evaluation_seed is None else int(evaluation_seed)
        rng_seed = (base_seed ^ 0xFACEB00C) + (stable_seed_offset % 100_000)
        rng = np.random.default_rng(rng_seed)
        picks: List[List[int]] = []

        public_count = int(field_size * 0.6)
        for ordinal in range(public_count):
            picks.append(model.generate_public_candidate(rng, 20_000 + ordinal).pick_indices)

        remaining = field_size - public_count
        counts = _allocate_counts(remaining, contest_profile.opponent_mix)
        ordinal_by_archetype = {name: 30_000 for name in counts}
        for archetype_name, count in counts.items():
            for _ in range(count):
                candidate = model.generate_candidate(archetype_name, rng, ordinal_by_archetype[archetype_name])
                ordinal_by_archetype[archetype_name] += 1
                picks.append(candidate.pick_indices)

        return np.array(picks, dtype=np.int16)

    def _prepared_run_key(
        self,
        dataset: SelectionSundayDataset,
        model_artifact_path: Path,
        public_field_artifact_path: Path | None,
        evaluation_seed: int,
    ) -> tuple:
        contest_signature = tuple(
            (
                contest_id,
                contest.scoring_profile,
                contest.payout_profile,
                contest.simulated_field_size,
                tuple(sorted(contest.opponent_mix.items())),
            )
            for contest_id, contest in sorted(self.contest_profiles.items())
        )
        return (
            dataset.season,
            dataset.metadata.get("source", ""),
            str(model_artifact_path.resolve()),
            str(public_field_artifact_path.resolve()) if public_field_artifact_path is not None else "none",
            int(evaluation_seed),
            self.simulation_profile.num_tournament_simulations,
            tuple(sorted(self.simulation_profile.primary_contests.items())),
            tuple(self.simulation_profile.sensitivity_contests),
            contest_signature,
        )

    def _model_cache_key(
        self,
        dataset: SelectionSundayDataset,
        model_artifact_path: Path,
        public_field_artifact_path: Path | None,
    ) -> tuple[int, str, str]:
        return (
            id(dataset),
            str(Path(model_artifact_path).resolve()),
            str(Path(public_field_artifact_path).resolve()) if public_field_artifact_path is not None else "none",
        )

    def _get_or_create_model(
        self,
        dataset: SelectionSundayDataset,
        model_artifact_path: Path,
        public_field_artifact_path: Path | None,
    ) -> V10TournamentModel:
        cache_key = self._model_cache_key(dataset, model_artifact_path, public_field_artifact_path)
        cached = self._model_cache.get(cache_key)
        if cached is None:
            cached = V10TournamentModel(
                dataset,
                game_model_artifact=model_artifact_path,
                public_field_artifact=public_field_artifact_path,
            )
            self._model_cache[cache_key] = cached
        return cached

    def _prepare_run_context(
        self,
        dataset: SelectionSundayDataset,
        model_artifact_path: Path,
        public_field_artifact_path: Path | None,
        *,
        evaluation_seed: int,
    ) -> dict[str, Any]:
        cache_key = self._prepared_run_key(dataset, model_artifact_path, public_field_artifact_path, evaluation_seed)
        if cache_key in self._prepared_run_cache:
            return self._prepared_run_cache[cache_key]

        model = self._get_or_create_model(dataset, model_artifact_path, public_field_artifact_path)
        tournament_outcomes, upset_gaps = model.simulate_many(
            self.simulation_profile.num_tournament_simulations,
            evaluation_seed,
        )

        opponent_scores_by_contest: Dict[str, np.ndarray] = {}
        scoring_cache: Dict[str, np.ndarray] = {}
        all_contests = list(self.simulation_profile.primary_contests) + list(self.simulation_profile.sensitivity_contests)
        for contest_id in all_contests:
            contest_profile = self.contest_profiles[contest_id]
            scoring_profile = self.scoring_profiles[contest_profile.scoring_profile]
            if scoring_profile.profile_id not in scoring_cache:
                opponent_pick_matrix = self._generate_opponent_field(model, contest_id, evaluation_seed=evaluation_seed)
                scoring_cache[scoring_profile.profile_id] = score_brackets(
                    opponent_pick_matrix,
                    tournament_outcomes,
                    upset_gaps,
                    model.game_rounds,
                    scoring_profile,
                )
            opponent_scores_by_contest[contest_id] = scoring_cache[scoring_profile.profile_id]

        prepared_context = {
            "key": cache_key,
            "model": model,
            "tournament_outcomes": tournament_outcomes,
            "upset_gaps": upset_gaps,
            "opponent_scores_by_contest": opponent_scores_by_contest,
        }
        self._prepared_run_cache[cache_key] = prepared_context
        return prepared_context

    def run(
        self,
        dataset: SelectionSundayDataset | None = None,
        *,
        release_seeds: list[int] | None = None,
        release_contract: ReleaseContractConfig | None = None,
    ) -> dict[str, Any]:
        if dataset is not None:
            active_dataset = dataset
        elif self._default_snapshot_path().exists():
            active_dataset = load_selection_sunday_dataset(self._default_snapshot_path())
        else:
            active_dataset = build_demo_dataset(seed=self.simulation_profile.seed)
        release_contract = release_contract or ReleaseContractConfig()
        resolved_release_seeds = [int(seed) for seed in (release_seeds or [self.simulation_profile.seed])]
        candidate_generation_seed = int(resolved_release_seeds[0])
        model_artifact_path, game_model_provenance = self._ensure_game_model_artifact(active_dataset)
        public_field_artifact_path, public_field_provenance = self._ensure_public_field_artifact(active_dataset)
        candidate_model = self._get_or_create_model(
            active_dataset,
            model_artifact_path,
            public_field_artifact_path,
        )
        candidates = self._generate_unique_candidates(candidate_model, candidate_seed=candidate_generation_seed)
        candidate_pick_matrix = np.array([candidate.pick_indices for candidate in candidates], dtype=np.int16)
        candidate_scores_by_contest_parts: Dict[str, list[np.ndarray]] = {}
        opponent_scores_by_contest_parts: Dict[str, list[np.ndarray]] = {}

        all_contests = list(self.simulation_profile.primary_contests) + list(self.simulation_profile.sensitivity_contests)
        for evaluation_seed in resolved_release_seeds:
            prepared = self._prepare_run_context(
                active_dataset,
                model_artifact_path,
                public_field_artifact_path,
                evaluation_seed=evaluation_seed,
            )
            prepared_model = prepared["model"]
            tournament_outcomes = prepared["tournament_outcomes"]
            upset_gaps = prepared["upset_gaps"]
            opponent_scores_by_contest = prepared["opponent_scores_by_contest"]
            scoring_cache: Dict[str, np.ndarray] = {}

            for contest_id in all_contests:
                contest_profile = self.contest_profiles[contest_id]
                scoring_profile = self.scoring_profiles[contest_profile.scoring_profile]
                if scoring_profile.profile_id not in scoring_cache:
                    scoring_cache[scoring_profile.profile_id] = score_brackets(
                        candidate_pick_matrix,
                        tournament_outcomes,
                        upset_gaps,
                        prepared_model.game_rounds,
                        scoring_profile,
                    )
                candidate_scores_by_contest_parts.setdefault(contest_id, []).append(
                    scoring_cache[scoring_profile.profile_id]
                )
                opponent_scores_by_contest_parts.setdefault(contest_id, []).append(opponent_scores_by_contest[contest_id])

        candidate_scores_by_contest: Dict[str, np.ndarray] = {
            contest_id: np.concatenate(matrices, axis=1)
            for contest_id, matrices in candidate_scores_by_contest_parts.items()
        }
        opponent_scores_by_contest: Dict[str, np.ndarray] = {
            contest_id: np.concatenate(matrices, axis=1)
            for contest_id, matrices in opponent_scores_by_contest_parts.items()
        }

        for candidate in candidates:
            candidate.weighted_first_place_equity = 0.0
            candidate.weighted_average_finish = 0.0

        for contest_id in all_contests:
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

        contest_payout_profiles = {
            contest_id: self.payout_profiles[self.contest_profiles[contest_id].payout_profile]
            for contest_id in self.simulation_profile.primary_contests
            if self.contest_profiles[contest_id].payout_profile in self.payout_profiles
        }

        selection = select_portfolio(
            candidates,
            candidate_pick_matrix,
            candidate_scores_by_contest,
            opponent_scores_by_contest,
            self.simulation_profile,
            contest_payout_profiles,
            release_contract=release_contract,
        )
        selected_indices = selection["selected_indices"]
        naive_indices = selection["naive_baseline_indices"]
        portfolio_fpe = selection["portfolio_fpe"]
        naive_fpe = selection["naive_fpe"]
        portfolio_capture = selection["portfolio_capture"]
        naive_capture = selection["naive_capture"]
        portfolio_payout = selection["portfolio_payout_summary"]
        naive_payout = selection["naive_payout_summary"]
        portfolio_avg_overlap = selection["portfolio_average_overlap"]
        selection_metadata = selection["selection_metadata"]
        portfolio_release_evaluation = selection["portfolio_release_evaluation"]
        naive_release_evaluation = selection["naive_release_evaluation"]

        finalists = [candidates[index] for index in selected_indices]
        selection_method = "greedy+local_search" if selection_metadata.get("local_search_applied") else "greedy"

        for finalist in finalists:
            best_fit = max(finalist.contest_metrics.items(), key=lambda item: item[1].first_place_equity)
            finalist.scenario_fit = best_fit[0]
            finalist.why_selected = (
                f"portfolio_fpe={portfolio_fpe:.4f}, capture={portfolio_capture:.4f}, "
                f"expected_utility={portfolio_payout.expected_utility:.4f}, "
                f"expected_payout={portfolio_payout.expected_payout:.2f}, "
                f"release_score={portfolio_release_evaluation.release_objective_score:.4f}, "
                f"overlap={portfolio_avg_overlap:.3f}, duplication={selection_metadata.get('portfolio_duplication_score', 0.0):.3f}, "
                f"payoff_corr={portfolio_payout.payoff_correlation:.3f}, "
                f"selection={selection_method}, local_search_changed={selection_metadata.get('local_search_applied', False)}, "
                f"best_fit={best_fit[0]}, "
                f"release_gates={'pass' if not portfolio_release_evaluation.guardrail_failures else ','.join(portfolio_release_evaluation.guardrail_failures)}"
            )

        scenario_summary: Dict[str, Dict[str, float]] = {}
        for contest_id in self.simulation_profile.primary_contests:
            payout_summary = weighted_payout_summary(
                selected_indices,
                candidate_scores_by_contest,
                opponent_scores_by_contest,
                {contest_id: 1.0},
                {contest_id: contest_payout_profiles[contest_id]} if contest_id in contest_payout_profiles else {},
            )
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
                "portfolio_expected_payout": payout_summary.expected_payout,
                "portfolio_expected_utility": payout_summary.expected_utility,
                "portfolio_cash_rate": payout_summary.cash_rate,
                "portfolio_top3_equity": payout_summary.top3_equity,
                "field_size": float(self.contest_profiles[contest_id].simulated_field_size),
            }

        sensitivity_summary: Dict[str, Dict[str, float]] = {}
        for contest_id in self.simulation_profile.sensitivity_contests:
            contest_profile = self.contest_profiles[contest_id]
            payout_profile = self.payout_profiles.get(contest_profile.payout_profile or "")
            payout_summary = weighted_payout_summary(
                selected_indices,
                candidate_scores_by_contest,
                opponent_scores_by_contest,
                {contest_id: 1.0},
                {contest_id: payout_profile} if payout_profile is not None else {},
            )
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
                "portfolio_expected_payout": payout_summary.expected_payout,
                "portfolio_expected_utility": payout_summary.expected_utility,
                "portfolio_cash_rate": payout_summary.cash_rate,
                "portfolio_top3_equity": payout_summary.top3_equity,
                "field_size": float(contest_profile.simulated_field_size),
            }

        summary = PortfolioSummary(
            weighted_portfolio_capture_rate=portfolio_capture,
            naive_baseline_capture_rate=naive_capture,
            weighted_portfolio_objective=portfolio_release_evaluation.release_objective_score,
            naive_baseline_objective=naive_release_evaluation.release_objective_score,
            weighted_candidate_baseline=float(
                np.mean([candidates[index].weighted_first_place_equity for index in naive_indices])
            ),
            average_pairwise_overlap=portfolio_avg_overlap,
            distinct_archetypes=len({candidate.archetype for candidate in finalists}),
            unique_champions=len({candidate.champion_team_id for candidate in finalists}),
            weighted_portfolio_first_place_equity=portfolio_fpe,
            naive_baseline_first_place_equity=naive_fpe,
            weighted_portfolio_expected_payout=portfolio_payout.expected_payout,
            naive_baseline_expected_payout=naive_payout.expected_payout,
            weighted_portfolio_cash_rate=portfolio_payout.cash_rate,
            weighted_portfolio_top3_equity=portfolio_payout.top3_equity,
            payoff_correlation_score=portfolio_payout.payoff_correlation,
        )
        release_seed_token = hashlib.blake2b(
            ",".join(str(seed) for seed in resolved_release_seeds).encode("utf-8"),
            digest_size=4,
        ).hexdigest()

        result = PortfolioResult(
            run_id=f"{active_dataset.season}-{candidate_generation_seed}-v10.6-r{len(resolved_release_seeds)}-{release_seed_token}",
            data_source=active_dataset.metadata.get("source", "selection_sunday_input"),
            candidate_count=len(candidates),
            finalists=finalists,
            summary=summary,
            scenario_summary=scenario_summary,
            sensitivity_summary=sensitivity_summary,
            dashboard_spec_sections=[
                "V10 candidate table with posterior win probabilities and uncertainty bands",
                "Contest payout panel with expected payout, cash rate, and top-3 equity",
                "Public-field duplication panel with champion crowding and path duplication proxies",
                "Artifact provenance panel for game-model and public-field model versions",
            ],
        )

        contest_profiles_dict = {
            contest_id: {
                "name": contest.name,
                "scoring_profile": contest.scoring_profile,
                "payout_profile": contest.payout_profile,
                "field_behavior_profile": contest.field_behavior_profile,
                "simulated_field_size": contest.simulated_field_size,
                "entries_submitted": contest.entries_submitted,
            }
            for contest_id, contest in self.contest_profiles.items()
        }

        return {
            "result": result,
            "dataset": active_dataset,
            "model": candidate_model,
            "candidates": candidates,
            "candidate_scores_by_contest": candidate_scores_by_contest,
            "opponent_scores_by_contest": opponent_scores_by_contest,
            "selected_indices": selected_indices,
            "naive_baseline_indices": naive_indices,
            "selection_metadata": selection_metadata,
            "release_guardrail_failures": list(portfolio_release_evaluation.guardrail_failures),
            "passes_release_gates": not portfolio_release_evaluation.guardrail_failures,
            "release_contract": asdict(release_contract),
            "engine_version": "v10.6",
            "simulation_config": {
                "seed": self.simulation_profile.seed,
                "candidate_generation_seed": candidate_generation_seed,
                "num_tournament_simulations": self.simulation_profile.num_tournament_simulations,
                "num_candidate_brackets": self.simulation_profile.num_candidate_brackets,
                "portfolio_size": self.simulation_profile.portfolio_size,
                "primary_contests": self.simulation_profile.primary_contests,
                "sensitivity_contests": self.simulation_profile.sensitivity_contests,
                "release_evaluation_seed_count": len(resolved_release_seeds),
                "release_evaluation_seeds": resolved_release_seeds,
                "release_contract": asdict(release_contract),
                "contest_profiles": contest_profiles_dict,
                "training_profile_id": self.training_profile.profile_id,
                "model_artifact_path": str(model_artifact_path),
                "public_field_artifact_path": str(public_field_artifact_path) if public_field_artifact_path is not None else None,
                "artifact_provenance": {
                    "game_model": game_model_provenance,
                    "public_field": public_field_provenance,
                },
                "public_field_runtime_mode": public_field_provenance.get("runtime_usage", "snapshot_fallback"),
            },
            "v10_artifacts": {
                "game_model_artifact": str(model_artifact_path),
                "public_field_artifact": str(public_field_artifact_path) if public_field_artifact_path is not None else None,
                "team_posteriors": candidate_model.team_posteriors,
                "public_duplication": candidate_model.public_duplication,
                "portfolio_payout_summary": portfolio_payout,
                "naive_payout_summary": naive_payout,
                "portfolio_release_evaluation": portfolio_release_evaluation,
                "naive_release_evaluation": naive_release_evaluation,
                "artifact_provenance": {
                    "game_model": game_model_provenance,
                    "public_field": public_field_provenance,
                },
                "public_field_runtime_mode": public_field_provenance.get("runtime_usage", "snapshot_fallback"),
            },
        }


def load_dataset_or_demo(path: str | Path | None) -> SelectionSundayDataset | None:
    if path is None:
        return None
    return load_selection_sunday_dataset(Path(path))

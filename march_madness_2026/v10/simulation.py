from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..game_model import build_matchup_row, predict_game_probabilities, predict_team_posteriors
from ..game_model import load_model_artifact
from ..public_field import estimate_path_duplication, predict_public_advancement_rates
from ..public_field import load_public_field_artifact
from ..tournament import TournamentModel, _sigmoid

ROUND_TO_PUBLIC_NAME = {
    1: "round_of_32",
    2: "sweet_16",
    3: "elite_8",
    4: "final_four",
    5: "championship_game",
    6: "champion",
}


class V10TournamentModel(TournamentModel):
    _STATE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}

    def __init__(
        self,
        dataset,
        *,
        game_model_artifact: Mapping[str, Any] | str | Path,
        public_field_artifact: Mapping[str, Any] | str | Path | None = None,
    ):
        super().__init__(dataset)
        self.game_model_artifact = load_model_artifact(game_model_artifact)
        self.public_field_artifact = (
            load_public_field_artifact(public_field_artifact)
            if public_field_artifact is not None
            else None
        )
        self._team_payloads = [team.model_dump(mode="python") for team in self.teams]
        cache_key = self._state_cache_key()
        cached = self._STATE_CACHE.get(cache_key)
        if cached is None:
            public_advancement_rates = (
                predict_public_advancement_rates(self.public_field_artifact, self._team_payloads)
                if self.public_field_artifact
                else {}
            )
            public_duplication = (
                estimate_path_duplication(public_advancement_rates)
                if public_advancement_rates
                else {}
            )
            round_probability_cache = self._build_round_probability_cache()
            public_probability_cache = self._build_public_probability_cache(
                public_advancement_rates=public_advancement_rates,
                round_probability_cache=round_probability_cache,
            )
            posterior_rows = []
            for left in range(len(self.teams)):
                for right in range(left + 1, len(self.teams)):
                    posterior_rows.append(self._matchup_row(left, right, 2))
            team_posteriors = predict_team_posteriors(self.game_model_artifact, posterior_rows)
            cached = {
                "public_advancement_rates": public_advancement_rates,
                "public_duplication": public_duplication,
                "round_probability_cache": round_probability_cache,
                "public_probability_cache": public_probability_cache,
                "team_posteriors": team_posteriors,
            }
            self._STATE_CACHE[cache_key] = cached

        self.public_advancement_rates = cached["public_advancement_rates"]
        self.public_duplication = cached["public_duplication"]
        self._round_probability_cache = cached["round_probability_cache"]
        self._public_probability_cache = cached["public_probability_cache"]
        self.team_posteriors = cached["team_posteriors"]

    def _state_cache_key(self) -> tuple[str, str, str]:
        dataset_payload = {
            "season": self.dataset.season,
            "teams": self._team_payloads,
        }
        dataset_signature = hashlib.blake2b(
            json.dumps(dataset_payload, sort_keys=True, default=str).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        game_model_id = str(self.game_model_artifact.get("model_id", "unknown"))
        public_field_id = (
            str(self.public_field_artifact.get("model_id", "none"))
            if self.public_field_artifact is not None
            else "none"
        )
        return (dataset_signature, game_model_id, public_field_id)

    def _matchup_row(self, team_a_index: int, team_b_index: int, round_number: int) -> dict[str, Any]:
        return build_matchup_row(
            self._team_payloads[team_a_index],
            self._team_payloads[team_b_index],
            season=self.dataset.season,
            round_number=round_number,
        )

    def _build_round_probability_cache(self) -> np.ndarray:
        team_count = len(self.teams)
        cache = np.full((7, team_count, team_count), 0.5, dtype=np.float32)
        matchup_rows: list[dict[str, Any]] = []
        lookup: list[tuple[int, int, int]] = []

        for round_number in range(1, 7):
            for team_a_index in range(team_count):
                for team_b_index in range(team_count):
                    if team_a_index == team_b_index:
                        continue
                    matchup_rows.append(self._matchup_row(team_a_index, team_b_index, round_number))
                    lookup.append((round_number, team_a_index, team_b_index))

        probabilities = predict_game_probabilities(self.game_model_artifact, matchup_rows)["probabilities"]
        for probability, (round_number, team_a_index, team_b_index) in zip(probabilities, lookup):
            adjusted = float(probability)
            if round_number == 1:
                market_a = float(getattr(self.teams[team_a_index], "round1_market_win_prob", 0.0) or 0.0)
                market_b = float(getattr(self.teams[team_b_index], "round1_market_win_prob", 0.0) or 0.0)
                if market_a > 0.0 and market_b > 0.0:
                    market_relative = market_a / max(market_a + market_b, 1e-6)
                    adjusted = (0.75 * adjusted) + (0.25 * market_relative)
            cache[round_number, team_a_index, team_b_index] = min(max(adjusted, 0.005), 0.995)

        return cache

    def _fallback_public_probability(
        self,
        team_a_index: int,
        team_b_index: int,
        round_number: int,
        *,
        round_probability_cache: np.ndarray | None = None,
    ) -> float:
        probability_cache = round_probability_cache if round_probability_cache is not None else self._round_probability_cache
        base_prob = float(probability_cache[round_number, team_a_index, team_b_index])
        base_logit = math.log(max(base_prob, 0.001) / max(1.0 - base_prob, 0.001))
        team_a = self.teams[team_a_index]
        team_b = self.teams[team_b_index]
        popularity_gap = team_a.public_pick_pct - team_b.public_pick_pct
        round_multiplier = 1.0 if round_number <= 3 else 1.5
        adjusted_logit = (base_logit + 0.02 * popularity_gap * round_multiplier) / 0.88
        return min(max(float(_sigmoid(adjusted_logit)), 0.005), 0.995)

    def _build_public_probability_cache(
        self,
        *,
        public_advancement_rates: Mapping[str, Mapping[str, float]] | None = None,
        round_probability_cache: np.ndarray | None = None,
    ) -> np.ndarray:
        team_count = len(self.teams)
        cache = np.full((7, team_count, team_count), 0.5, dtype=np.float32)
        advancement_rates = public_advancement_rates if public_advancement_rates is not None else self.public_advancement_rates
        probability_cache = round_probability_cache if round_probability_cache is not None else self._round_probability_cache

        for round_number in range(1, 7):
            round_name = ROUND_TO_PUBLIC_NAME.get(round_number, "champion")
            for team_a_index in range(team_count):
                team_a_id = self.team_id_by_index[team_a_index]
                for team_b_index in range(team_count):
                    if team_a_index == team_b_index:
                        continue
                    probability = self._fallback_public_probability(
                        team_a_index,
                        team_b_index,
                        round_number,
                        round_probability_cache=probability_cache,
                    )
                    if advancement_rates:
                        team_b_id = self.team_id_by_index[team_b_index]
                        a_rate = advancement_rates.get(team_a_id, {}).get(round_name, 0.0)
                        b_rate = advancement_rates.get(team_b_id, {}).get(round_name, 0.0)
                        if a_rate > 0.0 or b_rate > 0.0:
                            relative = a_rate / max(a_rate + b_rate, 1e-6)
                            model_probability = float(probability_cache[round_number, team_a_index, team_b_index])
                            probability = min(max((0.70 * relative) + (0.30 * model_probability), 0.005), 0.995)
                    cache[round_number, team_a_index, team_b_index] = probability

        return cache

    def _round_model_probability(self, team_a_index: int, team_b_index: int, round_number: int) -> float:
        return float(self._round_probability_cache[round_number, team_a_index, team_b_index])

    def _public_pick_probability(self, team_a_index: int, team_b_index: int, round_number: int) -> float:
        return float(self._public_probability_cache[round_number, team_a_index, team_b_index])

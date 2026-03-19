from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v10.provenance import public_history_provenance, empirical_public_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v10"
DEFAULT_PUBLIC_REFERENCE = PROJECT_ROOT / "data" / "reference" / "public_pick_rates.json"
DEFAULT_SEED_HISTORY = PROJECT_ROOT / "data" / "reference" / "seed_matchup_history.json"

ROUND_SEQUENCE = [
    "round_of_32",
    "sweet_16",
    "elite_8",
    "final_four",
    "championship_game",
    "champion",
]

SEED_HISTORY_KEYS = {
    "round_of_32": None,
    "sweet_16": "sweet_16",
    "elite_8": "elite_8",
    "final_four": "final_four",
    "championship_game": "championship_game",
    "champion": "champion",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _clip_probability(value: float) -> float:
    return float(min(1.0 - 1e-6, max(1e-6, value)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _normalize_team_id(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_reference_data(reference_data: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference_data is not None:
        return dict(reference_data)
    if DEFAULT_PUBLIC_REFERENCE.exists():
        return _load_json(DEFAULT_PUBLIC_REFERENCE)
    return {}


def _load_seed_history(seed_history: Mapping[str, Any] | None) -> dict[str, Any]:
    if seed_history is not None:
        return dict(seed_history)
    if DEFAULT_SEED_HISTORY.exists():
        return _load_json(DEFAULT_SEED_HISTORY)
    return {}


def _seed_prior(seed_history: Mapping[str, Any], seed: int, round_name: str) -> float:
    if round_name == "round_of_32":
        round_of_64 = seed_history.get("round_of_64", {})
        lookup = f"{min(seed, 17 - seed)}_vs_{max(seed, 17 - seed)}"
        if lookup in round_of_64:
            record = round_of_64[lookup]
            higher_seed_rate = _safe_float(record.get("higher_seed_wins"), 0.5)
            if seed <= 8:
                return higher_seed_rate
            return 1.0 - higher_seed_rate
        return 0.5
    history_key = SEED_HISTORY_KEYS[round_name]
    if not history_key:
        return 0.5
    advancement_rates = seed_history.get("advancement_rates", {}).get(history_key, {})
    if str(seed) in advancement_rates:
        return _safe_float(advancement_rates[str(seed)], 0.0)
    return 0.0


def _public_popularity_score(team: Mapping[str, Any], champion_reference: Mapping[str, Any]) -> float:
    direct_public = _safe_float(team.get("public_pick_pct"), 0.0)
    if direct_public > 0.0:
        return max(direct_public, 0.01)
    team_id = _normalize_team_id(str(team.get("team_id", "")))
    team_name = _normalize_team_id(str(team.get("team_name", "")))
    for key in (team_id, team_name):
        if key in champion_reference:
            return max(_safe_float(champion_reference[key]), 0.01)
    return 0.01


def fit_public_round_model(
    teams: Sequence[Mapping[str, Any]],
    *,
    historical_public_rows: Iterable[Mapping[str, Any]] | None = None,
    reference_data: Mapping[str, Any] | None = None,
    seed_history: Mapping[str, Any] | None = None,
    model_id: str = "v10_public_default",
) -> dict[str, Any]:
    reference = _load_reference_data(reference_data)
    history = _load_seed_history(seed_history)
    champion_reference = reference.get("champion_picks", {})
    first_round_reference = reference.get("first_round_public_picks", {})

    rating_values = np.asarray([_safe_float(team.get("rating")) for team in teams], dtype=np.float64)
    rating_mean = float(rating_values.mean()) if len(rating_values) else 0.0
    rating_std = float(rating_values.std()) if len(rating_values) else 1.0
    rating_std = rating_std if rating_std >= 1e-6 else 1.0

    round_models: dict[str, dict[str, float]] = {}
    round_defaults = {
        "round_of_32": {"intercept": -0.35, "public_weight": 0.18, "rating_weight": 0.42, "seed_weight": 1.10},
        "sweet_16": {"intercept": -0.95, "public_weight": 0.22, "rating_weight": 0.52, "seed_weight": 1.05},
        "elite_8": {"intercept": -1.55, "public_weight": 0.25, "rating_weight": 0.56, "seed_weight": 0.98},
        "final_four": {"intercept": -2.05, "public_weight": 0.28, "rating_weight": 0.60, "seed_weight": 0.90},
        "championship_game": {"intercept": -2.55, "public_weight": 0.34, "rating_weight": 0.64, "seed_weight": 0.82},
        "champion": {"intercept": -3.10, "public_weight": 0.42, "rating_weight": 0.68, "seed_weight": 0.74},
    }

    historical_rows = list(historical_public_rows or [])
    provenance = public_history_provenance(historical_rows)
    empirical_rows = empirical_public_rows(historical_rows)
    historical_round_scales: dict[str, list[float]] = {round_name: [] for round_name in ROUND_SEQUENCE}
    for row in empirical_rows:
        round_name = str(row.get("round_name", row.get("round", ""))).strip().lower()
        if round_name not in historical_round_scales:
            continue
        public_share = _safe_float(row.get("public_pick_pct", row.get("public_share", 0.0)), 0.0)
        observed_rate = _safe_float(row.get("public_adv_rate", row.get("observed_rate", row.get("pick_rate", 0.0))), 0.0)
        if public_share > 0.0 and observed_rate > 0.0:
            historical_round_scales[round_name].append(observed_rate / public_share)

    for round_name, coefficients in round_defaults.items():
        public_scale = historical_round_scales[round_name]
        if public_scale:
            coefficients = dict(coefficients)
            coefficients["public_weight"] *= float(np.clip(np.mean(public_scale), 0.6, 1.8))
        round_models[round_name] = coefficients

    artifact = {
        "model_id": model_id,
        "engine_version": "v10_sidecar",
        "created_at": _utc_now_iso(),
        "round_models": round_models,
        "reference_source": reference.get("source", "heuristic"),
        "seed_history_source": history.get("source", "seed_history"),
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "first_round_reference": first_round_reference,
        "champion_reference": champion_reference,
        "historical_public_row_count": provenance["historical_public_row_count"],
        "historical_public_seasons": provenance["historical_public_seasons"],
        "historical_public_source_types": provenance["historical_public_source_types"],
        "effective_historical_public_row_count": provenance["effective_historical_public_row_count"],
        "effective_historical_public_source_types": provenance["effective_historical_public_source_types"],
        "public_history_mode": provenance["public_history_mode"],
        "release_eligible": provenance["release_eligible"],
        "historical_round_scale_summary": {
            round_name: {
                "row_count": len(values),
                "mean_scale": float(np.mean(values)) if values else None,
            }
            for round_name, values in historical_round_scales.items()
        },
    }
    return artifact


def predict_public_advancement_rates(
    public_field_artifact: Mapping[str, Any] | str | Path,
    teams: Sequence[Mapping[str, Any]],
    *,
    seed_history: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    artifact = load_public_field_artifact(public_field_artifact)
    history = _load_seed_history(seed_history)
    rating_mean = _safe_float(artifact.get("rating_mean"), 0.0)
    rating_std = max(_safe_float(artifact.get("rating_std"), 1.0), 1e-6)
    champion_reference = artifact.get("champion_reference", {})

    advancement: dict[str, dict[str, float]] = {}
    for team in teams:
        team_id = str(team.get("team_id", team.get("id", "")))
        popularity_pct = _public_popularity_score(team, champion_reference)
        popularity_log = math.log(max(popularity_pct / 100.0, 1e-4))
        seed = max(1, _safe_int(team.get("seed"), 16))
        seed_strength = (17.0 - seed) / 16.0
        rating_z = (_safe_float(team.get("rating"), rating_mean) - rating_mean) / rating_std
        rates: dict[str, float] = {}

        for round_name in ROUND_SEQUENCE:
            model = artifact["round_models"][round_name]
            base = _seed_prior(history, seed, round_name)
            logit = (
                _safe_float(model["intercept"])
                + (_safe_float(model["public_weight"]) * popularity_log)
                + (_safe_float(model["rating_weight"]) * rating_z)
                + (_safe_float(model["seed_weight"]) * seed_strength)
            )
            blended = 0.55 * _sigmoid(logit) + 0.45 * base
            if round_name == "champion":
                blended = max(blended, popularity_pct / 100.0)
            rates[round_name] = _clip_probability(blended)

        monotone: dict[str, float] = {}
        running = 1.0
        for round_name in ROUND_SEQUENCE:
            running = min(running, rates[round_name])
            monotone[round_name] = running
        advancement[team_id] = monotone

    first_round_reference = artifact.get("first_round_reference", {})
    for matchup in first_round_reference.values():
        if not isinstance(matchup, Mapping):
            continue
        favored = _normalize_team_id(str(matchup.get("favored", "")))
        pick_pct = _safe_float(matchup.get("pick_pct"), 0.0) / 100.0
        if favored and favored in advancement:
            advancement[favored]["round_of_32"] = max(advancement[favored]["round_of_32"], _clip_probability(pick_pct))
    return advancement


def sample_public_bracket_paths(
    public_field_artifact: Mapping[str, Any] | str | Path,
    teams: Sequence[Mapping[str, Any]],
    *,
    num_samples: int = 500,
    seed: int = 20260318,
) -> list[dict[str, Any]]:
    rates = predict_public_advancement_rates(public_field_artifact, teams)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []

    for sample_index in range(num_samples):
        sample_paths: dict[str, str] = {}
        champion_scores: list[tuple[str, float]] = []
        for team in teams:
            team_id = str(team.get("team_id", team.get("id", "")))
            team_rates = rates.get(team_id, {})
            furthest = "round_of_64"
            previous = 1.0
            for round_name in ROUND_SEQUENCE:
                marginal = min(1.0, team_rates.get(round_name, 0.0) / max(previous, 1e-6))
                if rng.random() <= marginal:
                    furthest = round_name
                    previous = team_rates.get(round_name, 0.0)
                else:
                    break
            sample_paths[team_id] = furthest
            champion_scores.append((team_id, team_rates.get("champion", 0.0) * rng.random()))
        champion = max(champion_scores, key=lambda item: item[1])[0] if champion_scores else None
        samples.append(
            {
                "sample_index": sample_index,
                "champion_team_id": champion,
                "paths": sample_paths,
            }
        )
    return samples


def estimate_path_duplication(
    public_advancement_rates: Mapping[str, Mapping[str, float]],
    *,
    field_size: int = 1000,
) -> dict[str, dict[str, float]]:
    duplication: dict[str, dict[str, float]] = {}
    for team_id, rates in public_advancement_rates.items():
        final_four_rate = _safe_float(rates.get("final_four"), 0.0)
        title_rate = _safe_float(rates.get("champion"), 0.0)
        title_duplicates = field_size * title_rate
        final_four_duplicates = field_size * final_four_rate
        path_likelihood = max(
            title_rate,
            math.sqrt(max(title_rate * final_four_rate, 1e-9)),
        )
        duplication[team_id] = {
            "expected_title_duplicates": title_duplicates,
            "expected_final_four_duplicates": final_four_duplicates,
            "path_duplication_proxy": float(np.clip(path_likelihood, 0.0, 1.0)),
        }
    return duplication


def public_field_summary(
    public_field_artifact: Mapping[str, Any] | str | Path,
    teams: Sequence[Mapping[str, Any]],
    *,
    field_size: int = 1000,
) -> dict[str, Any]:
    artifact = load_public_field_artifact(public_field_artifact)
    advancement = predict_public_advancement_rates(artifact, teams)
    duplication = estimate_path_duplication(advancement, field_size=field_size)

    champion_distribution = []
    for team in teams:
        team_id = str(team.get("team_id", team.get("id", "")))
        champion_probability = advancement.get(team_id, {}).get("champion", 0.0)
        champion_distribution.append((team_id, champion_probability))
    champion_distribution.sort(key=lambda item: item[1], reverse=True)

    probabilities = np.asarray([probability for _, probability in champion_distribution], dtype=np.float64)
    probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else probabilities
    entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))).sum()) if len(probabilities) else 0.0

    top_public = []
    for team in teams:
        team_id = str(team.get("team_id", team.get("id", "")))
        if team_id not in {team_key for team_key, _ in champion_distribution[:10]}:
            continue
        top_public.append(
            {
                "team_id": team_id,
                "team_name": str(team.get("team_name", team.get("name", team_id))),
                "champion_public_rate": advancement[team_id]["champion"],
                "final_four_public_rate": advancement[team_id]["final_four"],
                **duplication[team_id],
            }
        )
    top_public.sort(key=lambda item: item["champion_public_rate"], reverse=True)

    return {
        "model_id": artifact.get("model_id"),
        "reference_source": artifact.get("reference_source"),
        "field_size": field_size,
        "champion_entropy": entropy,
        "top_public_teams": top_public[:10],
        "advancement_rates": advancement,
        "duplication_estimates": duplication,
    }


def save_public_field_artifact(public_field_artifact: Mapping[str, Any], artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(public_field_artifact), handle)
    metadata_path = path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(public_field_artifact), handle, indent=2)
    return path


def load_public_field_artifact(public_field_artifact: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(public_field_artifact, Mapping):
        return dict(public_field_artifact)
    path = Path(public_field_artifact)
    with path.open("rb") as handle:
        return pickle.load(handle)

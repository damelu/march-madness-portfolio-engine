from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy.optimize import minimize

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cp = None
    _CUPY_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v10"

CORE_FEATURE_FAMILIES: dict[str, list[str]] = {
    "ratings_base": [
        "rating_diff",
        "seed_diff",
        "market_adjustment_diff",
        "coaching_adjustment_diff",
        "continuity_adjustment_diff",
    ],
    "four_factors": [
        "offense_rating_diff",
        "defense_rating_diff",
        "tempo_adjustment_diff",
        "three_point_rate_diff",
        "free_throw_rate_diff",
        "adj_efg_off_diff",
        "adj_efg_def_diff",
    ],
    "matchup_interactions": [
        "offense_vs_defense_gap",
        "defense_vs_offense_gap",
        "orb_vs_drb_gap",
        "drb_vs_orb_gap",
        "turnover_pressure_gap",
        "rim_pressure_gap",
        "foul_pressure_gap",
        "three_point_volatility_gap",
    ],
    "context": [
        "injury_penalty_diff",
        "volatility_diff",
        "travel_miles_diff",
        "timezone_shift_diff",
        "altitude_adjustment_diff",
        "venue_familiarity_diff",
        "region_strength_index_diff",
        "seed_misprice_diff",
        "injury_uncertainty_diff",
        "lineup_uncertainty_diff",
    ],
    "uncertainty": [
        "rating_abs_gap",
        "combined_volatility",
        "combined_injury_uncertainty",
        "combined_lineup_uncertainty",
    ],
    "public_field": [
        "public_pick_pct_diff",
        "public_pick_log_gap",
    ],
}

DEFAULT_FEATURE_FAMILIES = [
    "ratings_base",
    "four_factors",
    "matchup_interactions",
    "context",
    "uncertainty",
]

ROUND_LABELS = {
    1: "round_of_32",
    2: "sweet_16",
    3: "elite_8",
    4: "final_four",
    5: "championship_game",
    6: "champion",
}


@dataclass(frozen=True)
class PreparedRows:
    rows: list[dict[str, Any]]
    matrix: np.ndarray
    labels: np.ndarray
    seasons: np.ndarray
    feature_names: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))


def _clip_probability(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else default
    try:
        resolved = float(str(value).strip())
        return resolved if math.isfinite(resolved) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _safe_logit(probability: np.ndarray | float) -> np.ndarray | float:
    clipped = _clip_probability(probability)
    return np.log(clipped / (1.0 - clipped))


def _resolve_row_value(row: Mapping[str, Any], *candidates: str) -> float:
    for candidate in candidates:
        if candidate in row and row[candidate] not in (None, ""):
            return _safe_float(row[candidate])
    return 0.0


def _team_stat(row: Mapping[str, Any], side: str, stem: str) -> float:
    letter = "a" if side == "team_a" else "b"
    return _resolve_row_value(
        row,
        f"{side}_{stem}",
        f"{stem}_{letter}",
        f"{side}__{stem}",
    )


def _diff_from_row(row: Mapping[str, Any], stem: str) -> float:
    direct = (
        f"{stem}_diff",
        f"diff_{stem}",
    )
    for candidate in direct:
        if candidate in row and row[candidate] not in (None, ""):
            return _safe_float(row[candidate])
    return _team_stat(row, "team_a", stem) - _team_stat(row, "team_b", stem)


def _extract_season(row: Mapping[str, Any]) -> int:
    for key in ("season", "year", "tournament_season"):
        if key in row and row[key] not in (None, ""):
            return _safe_int(row[key], 0)
    return 0


def _extract_round_number(row: Mapping[str, Any]) -> int:
    if "round_number" in row and row["round_number"] not in (None, ""):
        return max(1, _safe_int(row["round_number"], 1))
    round_name = str(row.get("round_name", "")).strip().lower()
    aliases = {
        "round of 64": 1,
        "round_of_32": 1,
        "round of 32": 2,
        "sweet 16": 3,
        "sweet_16": 3,
        "elite 8": 4,
        "elite_8": 4,
        "final four": 5,
        "final_four": 5,
        "championship": 6,
        "national championship": 6,
        "champion": 6,
    }
    return aliases.get(round_name, 1)


def _extract_label(row: Mapping[str, Any]) -> float:
    label_keys = ("team_a_win", "label", "target", "outcome")
    for key in label_keys:
        if key in row and row[key] not in (None, ""):
            value = row[key]
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"team_a", "a", "left", "win", "1", "true"}:
                    return 1.0
                if lowered in {"team_b", "b", "right", "loss", "0", "false"}:
                    return 0.0
            return 1.0 if _safe_float(value) >= 0.5 else 0.0

    winner_id = str(row.get("winner_team_id", "")).strip()
    team_a_id = str(row.get("team_a_team_id", row.get("team_a_id", ""))).strip()
    team_b_id = str(row.get("team_b_team_id", row.get("team_b_id", ""))).strip()
    if winner_id and team_a_id and winner_id == team_a_id:
        return 1.0
    if winner_id and team_b_id and winner_id == team_b_id:
        return 0.0
    return float("nan")


def _row_sample_weight(row: Mapping[str, Any]) -> float:
    if "sample_weight" in row and row["sample_weight"] not in (None, ""):
        return max(0.0, _safe_float(row["sample_weight"]))
    round_number = _extract_round_number(row)
    return 1.0 + 0.05 * (round_number - 1)


def build_matchup_row(
    team_a: Mapping[str, Any],
    team_b: Mapping[str, Any],
    *,
    season: int | None = None,
    round_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "season": season if season is not None else _safe_int(team_a.get("season"), 0),
        "round_number": round_number,
        "team_a_id": team_a.get("team_id", team_a.get("id", "")),
        "team_b_id": team_b.get("team_id", team_b.get("id", "")),
        "team_a_name": team_a.get("team_name", team_a.get("name", "")),
        "team_b_name": team_b.get("team_name", team_b.get("name", "")),
    }

    stems = {
        "rating",
        "seed",
        "market_adjustment",
        "coaching_adjustment",
        "continuity_adjustment",
        "injury_penalty",
        "volatility",
        "offense_rating",
        "defense_rating",
        "tempo_adjustment",
        "three_point_rate",
        "free_throw_rate",
        "adj_efg_off",
        "adj_efg_def",
        "orb_rate_off",
        "drb_rate_def",
        "turnover_rate_off",
        "turnover_rate_def",
        "rim_rate_off",
        "rim_rate_def",
        "ft_rate_off",
        "ft_rate_def",
        "three_rate_off",
        "three_rate_def",
        "travel_miles_round1",
        "timezone_shift_round1",
        "altitude_adjustment",
        "venue_familiarity",
        "region_strength_index",
        "seed_misprice",
        "injury_uncertainty",
        "lineup_uncertainty",
        "public_pick_pct",
    }
    for stem in stems:
        row[f"team_a_{stem}"] = team_a.get(stem, 0.0)
        row[f"team_b_{stem}"] = team_b.get(stem, 0.0)

    if metadata:
        row.update(dict(metadata))
    return row


def _feature_value(feature_name: str, row: Mapping[str, Any]) -> float:
    if feature_name.endswith("_diff"):
        return _diff_from_row(row, feature_name.removesuffix("_diff"))

    if feature_name == "offense_vs_defense_gap":
        return _team_stat(row, "team_a", "offense_rating") - _team_stat(row, "team_b", "defense_rating")
    if feature_name == "defense_vs_offense_gap":
        return _team_stat(row, "team_b", "offense_rating") - _team_stat(row, "team_a", "defense_rating")
    if feature_name == "orb_vs_drb_gap":
        return _team_stat(row, "team_a", "orb_rate_off") - _team_stat(row, "team_b", "drb_rate_def")
    if feature_name == "drb_vs_orb_gap":
        return _team_stat(row, "team_a", "drb_rate_def") - _team_stat(row, "team_b", "orb_rate_off")
    if feature_name == "turnover_pressure_gap":
        return (
            _team_stat(row, "team_b", "turnover_rate_def")
            - _team_stat(row, "team_a", "turnover_rate_off")
        ) - (
            _team_stat(row, "team_a", "turnover_rate_def")
            - _team_stat(row, "team_b", "turnover_rate_off")
        )
    if feature_name == "rim_pressure_gap":
        return (
            _team_stat(row, "team_a", "rim_rate_off")
            - _team_stat(row, "team_b", "rim_rate_def")
        ) - (
            _team_stat(row, "team_b", "rim_rate_off")
            - _team_stat(row, "team_a", "rim_rate_def")
        )
    if feature_name == "foul_pressure_gap":
        return (
            _team_stat(row, "team_a", "ft_rate_off")
            - _team_stat(row, "team_b", "ft_rate_def")
        ) - (
            _team_stat(row, "team_b", "ft_rate_off")
            - _team_stat(row, "team_a", "ft_rate_def")
        )
    if feature_name == "three_point_volatility_gap":
        return (
            abs(_team_stat(row, "team_a", "three_rate_off")) * (1.0 + _team_stat(row, "team_a", "volatility"))
            - abs(_team_stat(row, "team_b", "three_rate_off")) * (1.0 + _team_stat(row, "team_b", "volatility"))
        )
    if feature_name == "rating_abs_gap":
        return abs(_diff_from_row(row, "rating"))
    if feature_name == "combined_volatility":
        return _team_stat(row, "team_a", "volatility") + _team_stat(row, "team_b", "volatility")
    if feature_name == "combined_injury_uncertainty":
        return _team_stat(row, "team_a", "injury_uncertainty") + _team_stat(row, "team_b", "injury_uncertainty")
    if feature_name == "combined_lineup_uncertainty":
        return _team_stat(row, "team_a", "lineup_uncertainty") + _team_stat(row, "team_b", "lineup_uncertainty")
    if feature_name == "public_pick_log_gap":
        public_a = max(_team_stat(row, "team_a", "public_pick_pct"), 0.01)
        public_b = max(_team_stat(row, "team_b", "public_pick_pct"), 0.01)
        return math.log(public_a) - math.log(public_b)
    return _resolve_row_value(row, feature_name)


def _feature_names_for_families(feature_families: Sequence[str] | None) -> list[str]:
    resolved = list(feature_families or DEFAULT_FEATURE_FAMILIES)
    feature_names: list[str] = []
    for family in resolved:
        feature_names.extend(CORE_FEATURE_FAMILIES.get(family, []))
    return list(dict.fromkeys(feature_names))


def prepare_historical_rows(
    historical_rows: Iterable[Mapping[str, Any]],
    *,
    feature_families: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
) -> PreparedRows:
    rows: list[dict[str, Any]] = []
    names = list(feature_names or _feature_names_for_families(feature_families))
    vectors: list[list[float]] = []
    labels: list[float] = []
    seasons: list[int] = []

    for raw_row in historical_rows:
        row = dict(raw_row)
        label = _extract_label(row)
        if math.isnan(label):
            continue
        season = _extract_season(row)
        vector = [_feature_value(name, row) for name in names]
        row.setdefault("sample_weight", _row_sample_weight(row))
        rows.append(row)
        vectors.append(vector)
        labels.append(label)
        seasons.append(season)

    matrix = np.asarray(vectors, dtype=np.float64) if vectors else np.zeros((0, len(names)), dtype=np.float64)
    return PreparedRows(
        rows=rows,
        matrix=matrix,
        labels=np.asarray(labels, dtype=np.float64),
        seasons=np.asarray(seasons, dtype=np.int32),
        feature_names=names,
    )


def _standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(matrix.shape[1], dtype=np.float64)
    scale = matrix.std(axis=0) if len(matrix) else np.ones(matrix.shape[1], dtype=np.float64)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (matrix - mean) / scale, mean, scale


def _train_logistic_member(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
) -> dict[str, Any]:
    if matrix.size == 0:
        raise ValueError("cannot train V10 model with zero rows")

    scaled_matrix, mean, scale = _standardize_matrix(matrix)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    sample_weight = np.where(sample_weight > 0.0, sample_weight, 1.0)
    n_rows, n_features = scaled_matrix.shape

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        weights = params[:n_features]
        intercept = params[-1]
        logits = scaled_matrix @ weights + intercept
        probabilities = _clip_probability(_sigmoid(logits))
        weighted_loss = -np.sum(
            sample_weight * (labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
        )
        penalty = 0.5 * regularization * float(np.dot(weights, weights))
        residual = probabilities - labels
        grad_weights = scaled_matrix.T @ (sample_weight * residual) + (regularization * weights)
        grad_intercept = float(np.sum(sample_weight * residual))
        return weighted_loss + penalty, np.concatenate([grad_weights, np.array([grad_intercept])])

    start = np.zeros(n_features + 1, dtype=np.float64)
    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=start,
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    params = result.x
    weights = params[:n_features]
    intercept = float(params[-1])
    logits = scaled_matrix @ weights + intercept
    probabilities = _clip_probability(_sigmoid(logits))
    return {
        "coef": weights.astype(np.float64),
        "intercept": intercept,
        "scaler_mean": mean.astype(np.float64),
        "scaler_scale": scale.astype(np.float64),
        "train_log_loss": float(
            -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
        ),
        "train_brier": float(np.mean((probabilities - labels) ** 2)),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "iterations": int(result.nit),
    }


def _apply_calibrator(probabilities: np.ndarray, calibrator: Mapping[str, Any] | None) -> np.ndarray:
    if not calibrator:
        return probabilities
    method = calibrator.get("method", "none")
    if method == "platt":
        a = _safe_float(calibrator.get("a"), 1.0)
        b = _safe_float(calibrator.get("b"), 0.0)
        return _clip_probability(_sigmoid(a * _safe_logit(probabilities) + b))
    if method == "isotonic":
        thresholds = np.asarray(calibrator.get("thresholds", []), dtype=np.float64)
        values = np.asarray(calibrator.get("values", []), dtype=np.float64)
        if thresholds.size == 0 or values.size == 0:
            return probabilities
        indices = np.searchsorted(thresholds, np.asarray(probabilities, dtype=np.float64), side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        return _clip_probability(values[indices])
    return probabilities


def _fit_platt_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = _safe_logit(probabilities)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = params
        calibrated = _clip_probability(_sigmoid(a * logits + b))
        loss = -np.mean(labels * np.log(calibrated) + (1.0 - labels) * np.log(1.0 - calibrated))
        residual = calibrated - labels
        grad_a = float(np.mean(residual * logits))
        grad_b = float(np.mean(residual))
        return loss, np.array([grad_a, grad_b], dtype=np.float64)

    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=np.array([1.0, 0.0], dtype=np.float64),
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        options={"maxiter": 200},
    )
    return {
        "method": "platt",
        "a": float(result.x[0]),
        "b": float(result.x[1]),
        "optimizer_success": bool(result.success),
    }


def _fit_isotonic_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probs = np.asarray(probabilities[order], dtype=np.float64)
    sorted_labels = np.asarray(labels[order], dtype=np.float64)

    blocks: list[list[float]] = []
    for index, label in enumerate(sorted_labels):
        blocks.append([float(index), float(index), 1.0, float(label)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[2] + right[2]
            merged_mean = ((left[3] * left[2]) + (right[3] * right[2])) / total_weight
            blocks.append([left[0], right[1], total_weight, merged_mean])

    thresholds: list[float] = []
    values: list[float] = []
    for start, end, _, mean_value in blocks:
        threshold = float(sorted_probs[int(end)])
        if thresholds and np.isclose(threshold, thresholds[-1]):
            values[-1] = float(mean_value)
        else:
            thresholds.append(threshold)
            values.append(float(mean_value))

    return {
        "method": "isotonic",
        "thresholds": thresholds,
        "values": values,
    }


def _member_probabilities(
    member: Mapping[str, Any],
    matrix: np.ndarray,
    *,
    backend: str = "numpy",
) -> np.ndarray:
    centered = (matrix - np.asarray(member["scaler_mean"], dtype=np.float64)) / np.asarray(
        member["scaler_scale"], dtype=np.float64
    )
    weights = np.asarray(member["coef"], dtype=np.float64)
    intercept = _safe_float(member["intercept"])
    if backend == "cupy" and _CUPY_AVAILABLE and matrix.size:
        centered_gpu = cp.asarray(centered)
        weights_gpu = cp.asarray(weights)
        return cp.asnumpy(_clip_probability(_sigmoid(centered_gpu @ weights_gpu + intercept)))
    return _clip_probability(_sigmoid(centered @ weights + intercept))


def _prediction_summary(member_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if member_probabilities.ndim == 1:
        member_probabilities = member_probabilities[None, :]
    mean = member_probabilities.mean(axis=0)
    std = member_probabilities.std(axis=0)
    lower = np.quantile(member_probabilities, 0.1, axis=0)
    upper = np.quantile(member_probabilities, 0.9, axis=0)
    entropy = -(mean * np.log(_clip_probability(mean)) + (1.0 - mean) * np.log(_clip_probability(1.0 - mean)))
    return {
        "mean": mean,
        "std": std,
        "lower_10": lower,
        "upper_90": upper,
        "entropy": entropy,
    }


def calibration_report(
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    bins: int = 10,
    seasons: Sequence[int] | None = None,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return {
            "row_count": 0,
            "log_loss": None,
            "brier_score": None,
            "accuracy": None,
            "ece": None,
            "bins": [],
            "season_metrics": [],
        }

    clipped = _clip_probability(probs)
    assigned = np.clip((clipped * bins).astype(int), 0, bins - 1)
    report_bins: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(bins):
        mask = assigned == bin_index
        if not np.any(mask):
            continue
        bin_prob = float(clipped[mask].mean())
        bin_rate = float(truth[mask].mean())
        bin_count = int(mask.sum())
        ece += abs(bin_prob - bin_rate) * (bin_count / len(clipped))
        report_bins.append(
            {
                "bin": bin_index,
                "count": bin_count,
                "predicted_mean": bin_prob,
                "observed_rate": bin_rate,
            }
        )

    season_metrics: list[dict[str, Any]] = []
    if seasons is not None:
        season_array = np.asarray(seasons)
        for season in sorted({int(value) for value in season_array if int(value) != 0}):
            mask = season_array == season
            season_probs = clipped[mask]
            season_truth = truth[mask]
            season_metrics.append(
                {
                    "season": season,
                    "row_count": int(mask.sum()),
                    "log_loss": float(
                        -np.mean(
                            season_truth * np.log(season_probs) + (1.0 - season_truth) * np.log(1.0 - season_probs)
                        )
                    ),
                    "brier_score": float(np.mean((season_probs - season_truth) ** 2)),
                    "accuracy": float(np.mean((season_probs >= 0.5) == season_truth)),
                }
            )

    return {
        "row_count": int(len(clipped)),
        "log_loss": float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped))),
        "brier_score": float(np.mean((clipped - truth) ** 2)),
        "accuracy": float(np.mean((clipped >= 0.5) == truth)),
        "ece": float(ece),
        "bins": report_bins,
        "season_metrics": season_metrics,
    }


def train_game_model(
    historical_rows: Iterable[Mapping[str, Any]],
    *,
    model_id: str = "v10_default",
    seed: int = 20260318,
    ensemble_size: int = 8,
    feature_families: Sequence[str] | None = None,
    validation_seasons: Sequence[int] | None = None,
    holdout_seasons: Sequence[int] | None = None,
    regularization: float = 1.0,
    max_iter: int = 250,
    calibration_method: str = "platt",
    backend: str = "auto",
) -> dict[str, Any]:
    calibration_method = str(calibration_method or "none").lower()
    if calibration_method not in {"platt", "isotonic", "none"}:
        raise ValueError(f"unsupported calibration_method: {calibration_method}")
    prepared = prepare_historical_rows(historical_rows, feature_families=feature_families)
    if len(prepared.rows) == 0:
        raise ValueError("no trainable historical rows were found")

    seasons = prepared.seasons
    unique_seasons = sorted({int(value) for value in seasons if int(value) != 0})
    validation_set = set(int(value) for value in (validation_seasons or ([] if len(unique_seasons) < 2 else unique_seasons[-1:])))
    holdout_set = set(int(value) for value in (holdout_seasons or []))
    train_mask = np.ones(len(prepared.rows), dtype=bool)
    if validation_set:
        train_mask &= ~np.isin(seasons, list(validation_set))
    if holdout_set:
        train_mask &= ~np.isin(seasons, list(holdout_set))
    if not np.any(train_mask):
        train_mask = np.ones(len(prepared.rows), dtype=bool)

    train_matrix = prepared.matrix[train_mask]
    train_labels = prepared.labels[train_mask]
    train_weights = np.asarray([_row_sample_weight(prepared.rows[index]) for index in np.where(train_mask)[0]], dtype=np.float64)
    rng = np.random.default_rng(seed)
    backend_name = "cupy" if backend in {"gpu", "cupy"} or (backend == "auto" and _CUPY_AVAILABLE) else "numpy"

    members: list[dict[str, Any]] = []
    for member_index in range(max(1, ensemble_size)):
        sampled_indices = rng.choice(len(train_matrix), size=len(train_matrix), replace=True)
        member = _train_logistic_member(
            train_matrix[sampled_indices],
            train_labels[sampled_indices],
            train_weights[sampled_indices],
            regularization=regularization,
            max_iter=max_iter,
        )
        member["member_index"] = member_index
        members.append(member)

    calibrator: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    if validation_set:
        validation_mask = np.isin(seasons, list(validation_set))
        if np.any(validation_mask):
            validation_matrix = prepared.matrix[validation_mask]
            member_predictions = np.vstack(
                [_member_probabilities(member, validation_matrix, backend=backend_name) for member in members]
            )
            validation_mean = member_predictions.mean(axis=0)
            if calibration_method == "platt":
                calibrator = _fit_platt_calibrator(validation_mean, prepared.labels[validation_mask])
            elif calibration_method == "isotonic":
                calibrator = _fit_isotonic_calibrator(validation_mean, prepared.labels[validation_mask])
            if calibrator is not None:
                validation_mean = _apply_calibrator(validation_mean, calibrator)
            validation_report = calibration_report(
                validation_mean,
                prepared.labels[validation_mask],
                seasons=prepared.seasons[validation_mask],
            )

    full_member_predictions = np.vstack(
        [_member_probabilities(member, prepared.matrix, backend=backend_name) for member in members]
    )
    full_mean = _apply_calibrator(full_member_predictions.mean(axis=0), calibrator)
    train_report = calibration_report(full_mean, prepared.labels, seasons=prepared.seasons)

    artifact = {
        "model_id": model_id,
        "engine_version": "v10_sidecar",
        "created_at": _utc_now_iso(),
        "seed": seed,
        "backend": backend_name,
        "feature_families": list(feature_families or DEFAULT_FEATURE_FAMILIES),
        "feature_names": prepared.feature_names,
        "members": members,
        "calibrator": calibrator,
        "metadata": {
            "row_count": len(prepared.rows),
            "train_row_count": int(train_mask.sum()),
            "validation_row_count": int(np.isin(seasons, list(validation_set)).sum()) if validation_set else 0,
            "holdout_row_count": int(np.isin(seasons, list(holdout_set)).sum()) if holdout_set else 0,
            "unique_seasons": unique_seasons,
            "validation_seasons": sorted(validation_set),
            "holdout_seasons": sorted(holdout_set),
            "regularization": regularization,
            "max_iter": max_iter,
            "ensemble_size": len(members),
            "cupy_available": _CUPY_AVAILABLE,
            "calibration_method": calibration_method,
        },
        "training_report": train_report,
        "validation_report": validation_report,
    }
    return artifact


def predict_game_probabilities(
    model_artifact: Mapping[str, Any] | str | Path,
    matchup_rows: Iterable[Mapping[str, Any]],
    *,
    backend: str | None = None,
    return_member_probabilities: bool = False,
) -> dict[str, Any]:
    artifact = load_model_artifact(model_artifact)
    prepared = prepare_historical_rows(
        [{**row, "team_a_win": row.get("team_a_win", 0)} for row in matchup_rows],
        feature_names=artifact["feature_names"],
    )
    runtime_backend = str(backend or artifact.get("backend", "auto")).lower()
    if runtime_backend == "auto":
        runtime_backend = "cupy" if _CUPY_AVAILABLE else "numpy"
    elif runtime_backend == "numpy" and _CUPY_AVAILABLE:
        preferred_backend = os.environ.get("MARCH_MADNESS_INFERENCE_BACKEND", "auto").lower()
        if preferred_backend in {"auto", "gpu", "cupy"}:
            runtime_backend = "cupy"
    member_predictions = np.vstack(
        [
            _member_probabilities(member, prepared.matrix, backend=runtime_backend)
            for member in artifact["members"]
        ]
    )
    summary = _prediction_summary(member_predictions)
    mean_probability = _apply_calibrator(summary["mean"], artifact.get("calibrator"))
    summary["mean"] = np.asarray(mean_probability, dtype=np.float64)
    result: dict[str, Any] = {
        "probabilities": summary["mean"],
        "uncertainty_std": summary["std"],
        "lower_10": summary["lower_10"],
        "upper_90": summary["upper_90"],
        "entropy": summary["entropy"],
        "feature_names": list(artifact["feature_names"]),
        "row_count": len(prepared.rows),
    }
    if return_member_probabilities:
        result["member_probabilities"] = member_predictions
    return result


def predict_team_posteriors(
    model_artifact: Mapping[str, Any] | str | Path,
    matchup_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact = load_model_artifact(model_artifact)
    rows = [dict(row) for row in matchup_rows]
    predictions = predict_game_probabilities(artifact, rows, return_member_probabilities=True)
    team_summary: dict[str, MutableMapping[str, Any]] = {}

    member_predictions = np.asarray(predictions["member_probabilities"], dtype=np.float64)
    mean_probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
    for index, row in enumerate(rows):
        team_a = str(row.get("team_a_id", row.get("team_a_team_id", "")))
        team_b = str(row.get("team_b_id", row.get("team_b_team_id", "")))
        team_a_name = str(row.get("team_a_name", row.get("team_a_team_name", team_a)))
        team_b_name = str(row.get("team_b_name", row.get("team_b_team_name", team_b)))
        team_a_members = member_predictions[:, index]
        team_b_members = 1.0 - team_a_members

        for team_id, team_name, probability, members in (
            (team_a, team_a_name, mean_probabilities[index], team_a_members),
            (team_b, team_b_name, 1.0 - mean_probabilities[index], team_b_members),
        ):
            summary = team_summary.setdefault(
                team_id,
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "games": 0,
                    "probability_sum": 0.0,
                    "uncertainty_sum": 0.0,
                    "member_samples": [],
                },
            )
            summary["games"] += 1
            summary["probability_sum"] += float(probability)
            summary["uncertainty_sum"] += float(np.std(members))
            summary["member_samples"].extend(float(value) for value in members)

    team_posteriors: list[dict[str, Any]] = []
    for team_id, summary in team_summary.items():
        games = max(1, int(summary["games"]))
        member_samples = np.asarray(summary["member_samples"], dtype=np.float64)
        team_posteriors.append(
            {
                "team_id": team_id,
                "team_name": summary["team_name"],
                "games": games,
                "mean_win_probability": summary["probability_sum"] / games,
                "mean_uncertainty": summary["uncertainty_sum"] / games,
                "posterior_lower_10": float(np.quantile(member_samples, 0.1)) if len(member_samples) else 0.0,
                "posterior_upper_90": float(np.quantile(member_samples, 0.9)) if len(member_samples) else 0.0,
            }
        )

    team_posteriors.sort(key=lambda item: item["mean_win_probability"], reverse=True)
    return {
        "model_id": artifact["model_id"],
        "team_posteriors": team_posteriors,
        "team_count": len(team_posteriors),
    }


def run_season_blocked_backtest(
    historical_rows: Iterable[Mapping[str, Any]],
    *,
    model_id: str = "v10_default",
    feature_families: Sequence[str] | None = None,
    min_training_seasons: int = 2,
    seed: int = 20260318,
    ensemble_size: int = 8,
    regularization: float = 1.0,
    max_iter: int = 250,
) -> dict[str, Any]:
    rows = [dict(row) for row in historical_rows]
    prepared = prepare_historical_rows(rows, feature_families=feature_families)
    seasons = sorted({int(value) for value in prepared.seasons if int(value) != 0})
    if not seasons:
        return {
            "model_id": model_id,
            "season_results": [],
            "summary": {
                "holdout_seasons": [],
                "mean_log_loss": None,
                "mean_brier_score": None,
                "mean_accuracy": None,
            },
            "calibration_report": calibration_report([], []),
        }

    season_results: list[dict[str, Any]] = []
    holdout_predictions: list[float] = []
    holdout_labels: list[float] = []
    holdout_seasons: list[int] = []

    for position, holdout_season in enumerate(seasons):
        if position < min_training_seasons:
            continue
        train_seasons = seasons[:position]
        train_rows = [row for row in rows if _extract_season(row) in train_seasons]
        eval_rows = [row for row in rows if _extract_season(row) == holdout_season]
        if not train_rows or not eval_rows:
            continue

        artifact = train_game_model(
            train_rows,
            model_id=f"{model_id}_through_{train_seasons[-1]}",
            seed=seed + holdout_season,
            ensemble_size=ensemble_size,
            feature_families=feature_families,
            regularization=regularization,
            max_iter=max_iter,
        )
        predictions = predict_game_probabilities(artifact, eval_rows)
        labels = np.asarray([_extract_label(row) for row in eval_rows], dtype=np.float64)
        report = calibration_report(predictions["probabilities"], labels, seasons=[holdout_season] * len(eval_rows))
        season_results.append(
            {
                "holdout_season": holdout_season,
                "train_seasons": train_seasons,
                "row_count": len(eval_rows),
                "log_loss": report["log_loss"],
                "brier_score": report["brier_score"],
                "accuracy": report["accuracy"],
            }
        )
        holdout_predictions.extend(float(value) for value in predictions["probabilities"])
        holdout_labels.extend(float(value) for value in labels)
        holdout_seasons.extend([holdout_season] * len(eval_rows))

    mean_log_loss = float(np.mean([result["log_loss"] for result in season_results])) if season_results else None
    mean_brier = float(np.mean([result["brier_score"] for result in season_results])) if season_results else None
    mean_accuracy = float(np.mean([result["accuracy"] for result in season_results])) if season_results else None
    return {
        "model_id": model_id,
        "season_results": season_results,
        "summary": {
            "holdout_seasons": [result["holdout_season"] for result in season_results],
            "mean_log_loss": mean_log_loss,
            "mean_brier_score": mean_brier,
            "mean_accuracy": mean_accuracy,
        },
        "calibration_report": calibration_report(
            holdout_predictions,
            holdout_labels,
            seasons=holdout_seasons,
        ),
    }


def save_model_artifact(model_artifact: Mapping[str, Any], artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(model_artifact), handle)

    metadata_path = path.with_suffix(".json")
    payload = {
        "model_id": model_artifact.get("model_id"),
        "engine_version": model_artifact.get("engine_version", "v10_sidecar"),
        "created_at": model_artifact.get("created_at"),
        "backend": model_artifact.get("backend"),
        "metadata": model_artifact.get("metadata", {}),
        "feature_families": model_artifact.get("feature_families", []),
        "feature_names": model_artifact.get("feature_names", []),
        "training_report": model_artifact.get("training_report"),
        "validation_report": model_artifact.get("validation_report"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_model_artifact(model_artifact: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(model_artifact, Mapping):
        return dict(model_artifact)
    path = Path(model_artifact)
    with path.open("rb") as handle:
        return pickle.load(handle)

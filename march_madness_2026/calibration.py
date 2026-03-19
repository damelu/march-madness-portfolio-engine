from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .models import CalibrationBin, CalibrationReport


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return array


def _validated_inputs(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.clip(_as_float_array(probabilities), 0.0, 1.0)
    obs = _as_float_array(outcomes)
    if probs.shape != obs.shape:
        raise ValueError("probabilities and outcomes must have matching shape")
    if sample_weights is None:
        weights = np.ones_like(probs, dtype=np.float64)
    else:
        weights = _as_float_array(sample_weights)
        if weights.shape != probs.shape:
            raise ValueError("sample_weights must match probabilities shape")
        if np.any(weights < 0):
            raise ValueError("sample_weights must be non-negative")
    if np.all(weights == 0):
        raise ValueError("sample_weights must contain positive mass")
    return probs, obs, weights


def compute_log_loss(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    epsilon: float = 1e-15,
) -> float:
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)
    clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    losses = -(obs * np.log(clipped) + (1.0 - obs) * np.log(1.0 - clipped))
    return float(np.average(losses, weights=weights))


def compute_brier_score(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)
    return float(np.average((probs - obs) ** 2, weights=weights))


def compute_reliability_bins(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    num_bins: int = 10,
    round_labels: Sequence[str] | None = None,
) -> List[CalibrationBin]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)

    if round_labels is None:
        groups = {"global": np.arange(len(probs))}
    else:
        labels = list(round_labels)
        if len(labels) != len(probs):
            raise ValueError("round_labels must match probabilities shape")
        grouped_indices: Dict[str, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped_indices[label].append(index)
        groups = {label: np.asarray(indices, dtype=np.int64) for label, indices in grouped_indices.items()}

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: List[CalibrationBin] = []
    for round_name, indices in groups.items():
        round_probs = probs[indices]
        round_obs = obs[indices]
        round_weights = weights[indices]
        bin_ids = np.digitize(round_probs, edges[1:-1], right=False)
        total_weight = float(round_weights.sum())
        for bin_index in range(num_bins):
            mask = bin_ids == bin_index
            if not np.any(mask):
                bins.append(
                    CalibrationBin(
                        bin_index=bin_index,
                        lower_bound=float(edges[bin_index]),
                        upper_bound=float(edges[bin_index + 1]),
                        round_name=None if round_name == "global" else round_name,
                    )
                )
                continue
            bin_probs = round_probs[mask]
            bin_obs = round_obs[mask]
            bin_weights = round_weights[mask]
            mean_prediction = float(np.average(bin_probs, weights=bin_weights))
            observed_rate = float(np.average(bin_obs, weights=bin_weights))
            bin_weight = float(bin_weights.sum() / total_weight) if total_weight > 0 else 0.0
            bins.append(
                CalibrationBin(
                    bin_index=bin_index,
                    lower_bound=float(edges[bin_index]),
                    upper_bound=float(edges[bin_index + 1]),
                    mean_prediction=mean_prediction,
                    observed_rate=observed_rate,
                    sample_count=int(mask.sum()),
                    weight=bin_weight,
                    absolute_gap=abs(mean_prediction - observed_rate),
                    round_name=None if round_name == "global" else round_name,
                )
            )
    return bins


def compute_expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    num_bins: int = 10,
) -> float:
    bins = compute_reliability_bins(
        probabilities=probabilities,
        outcomes=outcomes,
        sample_weights=sample_weights,
        num_bins=num_bins,
    )
    return float(sum(bin_.weight * bin_.absolute_gap for bin_ in bins))


def compute_brier_decomposition(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    num_bins: int = 10,
) -> Dict[str, float]:
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)
    bins = compute_reliability_bins(
        probabilities=probs,
        outcomes=obs,
        sample_weights=weights,
        num_bins=num_bins,
    )
    overall_rate = float(np.average(obs, weights=weights))
    reliability = 0.0
    resolution = 0.0
    for bin_ in bins:
        reliability += bin_.weight * (bin_.mean_prediction - bin_.observed_rate) ** 2
        resolution += bin_.weight * (bin_.observed_rate - overall_rate) ** 2
    uncertainty = overall_rate * (1.0 - overall_rate)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def summarize_calibration_by_round(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    round_labels: Sequence[str],
    sample_weights: Sequence[float] | np.ndarray | None = None,
    num_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)
    labels = list(round_labels)
    if len(labels) != len(probs):
        raise ValueError("round_labels must match probabilities shape")

    indices_by_round: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_round[label].append(index)

    summary: Dict[str, Dict[str, float]] = {}
    for round_name, indices in indices_by_round.items():
        round_indices = np.asarray(indices, dtype=np.int64)
        round_probs = probs[round_indices]
        round_obs = obs[round_indices]
        round_weights = weights[round_indices]
        summary[round_name] = {
            "sample_count": float(len(indices)),
            "log_loss": compute_log_loss(round_probs, round_obs, round_weights),
            "brier_score": compute_brier_score(round_probs, round_obs, round_weights),
            "expected_calibration_error": compute_expected_calibration_error(
                round_probs,
                round_obs,
                round_weights,
                num_bins=min(num_bins, max(1, len(indices))),
            ),
        }
    return summary


def build_calibration_report(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    num_bins: int = 10,
    round_labels: Sequence[str] | None = None,
    baseline_probabilities: Sequence[float] | np.ndarray | None = None,
    model_id: str = "unknown",
) -> CalibrationReport:
    probs, obs, weights = _validated_inputs(probabilities, outcomes, sample_weights)
    bins = compute_reliability_bins(
        probabilities=probs,
        outcomes=obs,
        sample_weights=weights,
        num_bins=num_bins,
    )
    brier_parts = compute_brier_decomposition(probs, obs, weights, num_bins=num_bins)
    baseline_comparison: Dict[str, float] = {}
    if baseline_probabilities is not None:
        baseline_comparison = {
            "baseline_log_loss": compute_log_loss(baseline_probabilities, obs, weights),
            "baseline_brier_score": compute_brier_score(baseline_probabilities, obs, weights),
        }
    return CalibrationReport(
        model_id=model_id,
        sample_count=int(len(probs)),
        log_loss=compute_log_loss(probs, obs, weights),
        brier_score=compute_brier_score(probs, obs, weights),
        expected_calibration_error=compute_expected_calibration_error(probs, obs, weights, num_bins=num_bins),
        bins=bins,
        by_round={}
        if round_labels is None
        else summarize_calibration_by_round(probs, obs, round_labels, weights, num_bins=num_bins),
        baseline_comparison=baseline_comparison,
        brier_reliability=brier_parts["reliability"],
        brier_resolution=brier_parts["resolution"],
        brier_uncertainty=brier_parts["uncertainty"],
    )


def plot_reliability_diagram(
    report: CalibrationReport | Iterable[CalibrationBin],
    width: int = 20,
) -> str:
    bins = report.bins if isinstance(report, CalibrationReport) else list(report)
    lines = ["bin        pred    obs    gap  chart"]
    for bin_ in bins:
        filled_pred = int(round(max(0.0, min(1.0, bin_.mean_prediction)) * width))
        filled_obs = int(round(max(0.0, min(1.0, bin_.observed_rate)) * width))
        pred_bar = "#" * filled_pred
        obs_bar = "=" * filled_obs
        label = f"{bin_.lower_bound:.1f}-{bin_.upper_bound:.1f}"
        if bin_.round_name:
            label = f"{bin_.round_name}:{label}"
        chart = f"{pred_bar:<{width}} | {obs_bar:<{width}}"
        lines.append(
            f"{label:<10} {bin_.mean_prediction:>5.3f} {bin_.observed_rate:>6.3f} {bin_.absolute_gap:>6.3f}  {chart}"
        )
    return "\n".join(lines)

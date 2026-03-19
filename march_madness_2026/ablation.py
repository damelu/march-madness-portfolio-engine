from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .game_model import DEFAULT_FEATURE_FAMILIES, run_season_blocked_backtest


def run_feature_family_ablation(
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
    families = list(feature_families or DEFAULT_FEATURE_FAMILIES)
    rows = [dict(row) for row in historical_rows]

    baseline = run_season_blocked_backtest(
        rows,
        model_id=model_id,
        feature_families=families,
        min_training_seasons=min_training_seasons,
        seed=seed,
        ensemble_size=ensemble_size,
        regularization=regularization,
        max_iter=max_iter,
    )

    ablations: dict[str, dict[str, Any]] = {}
    for family in families:
        dropped = [candidate for candidate in families if candidate != family]
        ablations[family] = run_season_blocked_backtest(
            rows,
            model_id=f"{model_id}_minus_{family}",
            feature_families=dropped,
            min_training_seasons=min_training_seasons,
            seed=seed,
            ensemble_size=ensemble_size,
            regularization=regularization,
            max_iter=max_iter,
        )

    return {
        "model_id": model_id,
        "baseline_feature_families": families,
        "baseline": baseline,
        "ablations": ablations,
        "lift_summary": summarize_ablation_lift(
            {
                "model_id": model_id,
                "baseline_feature_families": families,
                "baseline": baseline,
                "ablations": ablations,
            }
        ),
    }


def summarize_ablation_lift(ablation_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline_summary = ablation_result.get("baseline", {}).get("summary", {})
    baseline_log_loss = baseline_summary.get("mean_log_loss")
    baseline_brier = baseline_summary.get("mean_brier_score")
    baseline_accuracy = baseline_summary.get("mean_accuracy")

    summary: list[dict[str, Any]] = []
    for family, result in ablation_result.get("ablations", {}).items():
        metrics = result.get("summary", {})
        log_loss = metrics.get("mean_log_loss")
        brier = metrics.get("mean_brier_score")
        accuracy = metrics.get("mean_accuracy")
        summary.append(
            {
                "family": family,
                "baseline_mean_log_loss": baseline_log_loss,
                "baseline_mean_brier_score": baseline_brier,
                "baseline_mean_accuracy": baseline_accuracy,
                "ablated_mean_log_loss": log_loss,
                "ablated_mean_brier_score": brier,
                "ablated_mean_accuracy": accuracy,
                "log_loss_lift": None if baseline_log_loss is None or log_loss is None else log_loss - baseline_log_loss,
                "brier_lift": None if baseline_brier is None or brier is None else brier - baseline_brier,
                "accuracy_lift": None if baseline_accuracy is None or accuracy is None else baseline_accuracy - accuracy,
            }
        )

    summary.sort(
        key=lambda item: (
            item["log_loss_lift"] is None,
            -(item["log_loss_lift"] or 0.0),
        )
    )
    return summary


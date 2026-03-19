#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.config import load_training_profiles  # noqa: E402
from march_madness_2026.ablation import run_feature_family_ablation  # noqa: E402
from march_madness_2026.game_model import DEFAULT_FEATURE_FAMILIES, DEFAULT_MODEL_DIR, run_season_blocked_backtest  # noqa: E402
from march_madness_2026.v10.provenance import backtest_release_readiness  # noqa: E402


DEFAULT_DATASET = DEFAULT_MODEL_DIR / "historical_games.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v10_backtests"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run season-blocked V10 backtests and optional ablations.")
    parser.add_argument("--historical-rows", type=Path, default=DEFAULT_DATASET, help="Normalized historical parquet/CSV/JSON/JSONL rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory under outputs/v10_backtests.")
    parser.add_argument("--training-profile-id", default="", help="Optional training profile id from configs/model/training_profile.yaml.")
    parser.add_argument("--model-id", default="", help="Logical model id for reporting.")
    parser.add_argument(
        "--feature-families",
        default="",
        help="Comma-separated feature family list.",
    )
    parser.add_argument("--min-training-seasons", type=int, default=2, help="Minimum seasons before a holdout fold is evaluated.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Bootstrap ensemble size.")
    parser.add_argument("--regularization", type=float, default=1.0, help="L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=250, help="Max optimizer iterations.")
    parser.add_argument("--run-ablation", action="store_true", help="Also run drop-one-family ablations.")
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("rows", "games", "historical_rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, dict)]
        return []
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    raise ValueError(f"unsupported historical row format: {path.suffix}")


def _load_adjacent_manifest(path: Path) -> dict[str, Any] | None:
    candidate = path.parent / "historical_manifest.json"
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _season_mode_map(manifest: dict[str, Any] | None, table_name: str) -> dict[int, str]:
    if not manifest:
        return {}
    table = manifest.get("tables", {}).get(table_name, {})
    seasons_detail = table.get("seasons_detail", {})
    resolved: dict[int, str] = {}
    for season, payload in seasons_detail.items():
        if isinstance(payload, dict) and "mode" in payload:
            try:
                resolved[int(season)] = str(payload["mode"])
            except (TypeError, ValueError):
                continue
    return resolved


def _summarize_season_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"holdout_seasons": [], "mean_log_loss": None, "mean_brier_score": None, "mean_accuracy": None}
    count = len(results)
    return {
        "holdout_seasons": [int(row["holdout_season"]) for row in results],
        "mean_log_loss": float(sum(float(row["log_loss"]) for row in results) / count),
        "mean_brier_score": float(sum(float(row["brier_score"]) for row in results) / count),
        "mean_accuracy": float(sum(float(row["accuracy"]) for row in results) / count),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.historical_rows)
    if not rows:
        raise SystemExit(f"no rows found in {args.historical_rows}")

    profiles = load_training_profiles()
    profile = profiles.get(args.training_profile_id) if args.training_profile_id else None
    if args.training_profile_id and profile is None:
        raise SystemExit(f"unknown training profile: {args.training_profile_id}")

    model_id = args.model_id or (profile.model_id if profile is not None else "v10_default")
    feature_families = [value for value in args.feature_families.split(",") if value.strip()]
    if not feature_families:
        feature_families = list(profile.feature_sets) if profile is not None and profile.feature_sets else list(DEFAULT_FEATURE_FAMILIES)
    seed = int(args.seed if args.seed is not None else (profile.seed if profile is not None else 20260318))
    ensemble_size = int(
        args.ensemble_size if args.ensemble_size is not None else (profile.ensemble_size if profile is not None else 8)
    )
    backtest = run_season_blocked_backtest(
        rows,
        model_id=model_id,
        feature_families=feature_families,
        min_training_seasons=args.min_training_seasons,
        seed=seed,
        ensemble_size=ensemble_size,
        regularization=args.regularization,
        max_iter=args.max_iter,
    )
    if not backtest["season_results"] or not backtest["summary"]["holdout_seasons"]:
        raise SystemExit("no evaluable holdout folds were produced; check historical row schema and season coverage")

    ablation = None
    if args.run_ablation:
        ablation = run_feature_family_ablation(
            rows,
            model_id=model_id,
            feature_families=feature_families,
            min_training_seasons=args.min_training_seasons,
            seed=seed,
            ensemble_size=ensemble_size,
            regularization=args.regularization,
            max_iter=args.max_iter,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    manifest = _load_adjacent_manifest(args.historical_rows)
    game_season_modes = _season_mode_map(manifest, "games")
    empirical_results = [
        result for result in backtest["season_results"] if game_season_modes.get(int(result["holdout_season"])) == "real"
    ]
    fallback_results = [
        result
        for result in backtest["season_results"]
        if game_season_modes.get(int(result["holdout_season"])) not in {"", None, "real"}
    ]
    empirical_summary = _summarize_season_results(empirical_results)
    fallback_summary = _summarize_season_results(fallback_results)
    release_readiness = backtest_release_readiness(
        manifest=manifest,
        train_seasons=list(profile.train_seasons) if profile is not None else [],
        validation_seasons=list(profile.validation_seasons) if profile is not None else [],
        holdout_seasons=list(profile.holdout_seasons) if profile is not None else [],
        row_count=len(rows),
        empirical_only_holdout_seasons=empirical_summary["holdout_seasons"],
    )
    summary_path = args.output_dir / f"{stamp}_v10_backtest_summary.json"
    calibration_path = args.output_dir / f"{stamp}_v10_calibration_report.json"
    markdown_path = args.output_dir / f"{stamp}_v10_calibration_report.md"
    ablation_path = args.output_dir / f"{stamp}_v10_ablation_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": backtest["model_id"],
                "training_profile_id": args.training_profile_id or None,
                "summary": backtest["summary"],
                "empirical_only_summary": empirical_summary,
                "fallback_holdout_summary": fallback_summary,
                "release_readiness": release_readiness,
                "season_results": backtest["season_results"],
                "feature_families": feature_families,
                "historical_manifest": manifest,
            },
            handle,
            indent=2,
        )
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(backtest["calibration_report"], handle, indent=2)

    lines = [
        "# V10 Calibration Report",
        "",
        f"- Model ID: `{backtest['model_id']}`",
        f"- Holdout seasons: `{', '.join(str(value) for value in backtest['summary']['holdout_seasons']) or 'none'}`",
        f"- Mean log loss: `{backtest['summary']['mean_log_loss']}`",
        f"- Mean Brier score: `{backtest['summary']['mean_brier_score']}`",
        f"- Mean accuracy: `{backtest['summary']['mean_accuracy']}`",
        f"- Training profile: `{args.training_profile_id or 'none'}`",
        f"- Historical manifest present: `{'yes' if manifest else 'no'}`",
        "",
        "## Coverage Split",
        "",
        f"- Empirical-only holdout seasons: `{', '.join(str(value) for value in empirical_summary['holdout_seasons']) or 'none'}`",
        f"- Empirical-only mean log loss: `{empirical_summary['mean_log_loss']}`",
        f"- Empirical-only mean Brier score: `{empirical_summary['mean_brier_score']}`",
        f"- Empirical-only mean accuracy: `{empirical_summary['mean_accuracy']}`",
        f"- Fallback holdout seasons: `{', '.join(str(value) for value in fallback_summary['holdout_seasons']) or 'none'}`",
        f"- Fallback mean log loss: `{fallback_summary['mean_log_loss']}`",
        f"- Fallback mean Brier score: `{fallback_summary['mean_brier_score']}`",
        f"- Fallback mean accuracy: `{fallback_summary['mean_accuracy']}`",
        f"- Release ready: `{release_readiness['eligible']}`",
        f"- Release blockers: `{', '.join(release_readiness['blocking_issues']) or 'none'}`",
        "",
        "## Season Results",
        "",
    ]
    for result in backtest["season_results"]:
        lines.append(
            f"- `{result['holdout_season']}`: rows=`{result['row_count']}`, "
            f"log_loss=`{result['log_loss']:.4f}`, "
            f"brier=`{result['brier_score']:.4f}`, "
            f"accuracy=`{result['accuracy']:.4f}`"
        )
    lines.extend(["", "## Reliability Bins", ""])
    for row in backtest["calibration_report"]["bins"]:
        lines.append(
            f"- bin `{row['bin']}`: count=`{row['count']}`, predicted=`{row['predicted_mean']:.4f}`, observed=`{row['observed_rate']:.4f}`"
        )
    if ablation:
        lines.extend(["", "## Ablation Lift", ""])
        for row in ablation["lift_summary"]:
            lines.append(
                f"- `{row['family']}` removed: "
                f"log_loss_delta=`{row['log_loss_lift']}`, "
                f"brier_delta=`{row['brier_lift']}`, "
                f"accuracy_delta=`{row['accuracy_lift']}`"
            )

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    if ablation:
        with ablation_path.open("w", encoding="utf-8") as handle:
            json.dump(ablation, handle, indent=2)

    print(f"[v10-backtest] summary={summary_path}")
    print(f"[v10-backtest] calibration={calibration_path}")
    print(f"[v10-backtest] markdown={markdown_path}")
    if ablation:
        print(f"[v10-backtest] ablation={ablation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

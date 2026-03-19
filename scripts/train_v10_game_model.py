#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.config import load_training_profiles  # noqa: E402
from march_madness_2026.game_model import (  # noqa: E402
    DEFAULT_FEATURE_FAMILIES,
    DEFAULT_MODEL_DIR,
    save_model_artifact,
    train_game_model,
)
from march_madness_2026.v10.provenance import requested_game_provenance  # noqa: E402


DEFAULT_DATASET = DEFAULT_MODEL_DIR / "historical_games.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the additive V10 game model sidecar.")
    parser.add_argument("--historical-rows", type=Path, default=DEFAULT_DATASET, help="Normalized historical parquet/CSV/JSON/JSONL rows.")
    parser.add_argument("--training-profile-id", default="", help="Optional training profile id from configs/model/training_profile.yaml.")
    parser.add_argument("--model-id", default="", help="Artifact id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Artifact directory under data/models/v10.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Bootstrap ensemble size.")
    parser.add_argument(
        "--feature-families",
        default="",
        help="Comma-separated feature family list.",
    )
    parser.add_argument("--validation-seasons", default="", help="Comma-separated validation seasons.")
    parser.add_argument("--holdout-seasons", default="", help="Comma-separated holdout seasons.")
    parser.add_argument("--regularization", type=float, default=1.0, help="L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=250, help="Max optimizer iterations per member.")
    parser.add_argument("--backend", choices=["auto", "numpy", "cupy", "gpu"], default="", help="Prediction backend preference.")
    parser.add_argument("--calibration-method", choices=["platt", "isotonic", "none"], default="", help="Calibration method.")
    return parser


def _parse_seasons(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


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
    seed = int(args.seed if args.seed is not None else (profile.seed if profile is not None else 20260318))
    ensemble_size = int(
        args.ensemble_size if args.ensemble_size is not None else (profile.ensemble_size if profile is not None else 8)
    )
    feature_families = [value for value in args.feature_families.split(",") if value.strip()]
    if not feature_families:
        feature_families = list(profile.feature_sets) if profile is not None and profile.feature_sets else list(DEFAULT_FEATURE_FAMILIES)
    validation_seasons = _parse_seasons(args.validation_seasons) or (list(profile.validation_seasons) if profile is not None else [])
    holdout_seasons = _parse_seasons(args.holdout_seasons) or (list(profile.holdout_seasons) if profile is not None else [])
    backend = args.backend or (profile.backend if profile is not None else "auto")
    calibration_method = args.calibration_method or (profile.calibration_method if profile is not None else "isotonic")

    artifact = train_game_model(
        rows,
        model_id=model_id,
        seed=seed,
        ensemble_size=ensemble_size,
        feature_families=feature_families,
        validation_seasons=validation_seasons or None,
        holdout_seasons=holdout_seasons or None,
        regularization=args.regularization,
        max_iter=args.max_iter,
        calibration_method=calibration_method,
        backend=backend,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"game_model_{model_id}.pkl"
    save_model_artifact(artifact, artifact_path)
    manifest_path = args.output_dir / f"game_model_{model_id}_manifest.json"
    historical_manifest = _load_adjacent_manifest(args.historical_rows)
    provenance = requested_game_provenance(
        historical_manifest,
        train_seasons=list(profile.train_seasons) if profile is not None else [],
        validation_seasons=validation_seasons,
        holdout_seasons=holdout_seasons,
        row_count=len(rows),
    )
    manifest_payload = {
        "model_id": model_id,
        "training_profile_id": args.training_profile_id or None,
        "seed": seed,
        "ensemble_size": ensemble_size,
        "feature_families": feature_families,
        "validation_seasons": validation_seasons,
        "holdout_seasons": holdout_seasons,
        "backend": backend,
        "calibration_method": calibration_method,
        "historical_rows_path": str(args.historical_rows),
        "historical_manifest_path": str(args.historical_rows.parent / "historical_manifest.json"),
        "historical_manifest": historical_manifest,
        **provenance,
        "artifact_path": str(artifact_path),
        "training_report": artifact.get("training_report", {}),
        "validation_report": artifact.get("validation_report"),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    print(f"[v10-train] backend={artifact['backend']} model_id={artifact['model_id']}")
    print(f"[v10-train] artifact={artifact_path}")
    print(f"[v10-train] manifest={manifest_path}")
    print(f"[v10-train] train_report={json.dumps(artifact.get('training_report', {}), indent=2)}")
    if artifact.get("validation_report"):
        print(f"[v10-train] validation_report={json.dumps(artifact['validation_report'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from march_madness_2026.config import load_selection_sunday_dataset  # noqa: E402
from march_madness_2026.public_field import (  # noqa: E402
    DEFAULT_PUBLIC_REFERENCE,
    DEFAULT_SEED_HISTORY,
    fit_public_round_model,
    save_public_field_artifact,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v10"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the V10 public-field sidecar artifact.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Selection Sunday dataset JSON used for team payloads.")
    parser.add_argument("--historical-public", type=Path, default=DEFAULT_MODEL_DIR / "historical_public.parquet", help="Historical public rows parquet/CSV/JSON/JSONL.")
    parser.add_argument("--public-reference", type=Path, default=DEFAULT_PUBLIC_REFERENCE, help="Champion/round public reference JSON.")
    parser.add_argument("--seed-history", type=Path, default=DEFAULT_SEED_HISTORY, help="Seed-history JSON.")
    parser.add_argument("--model-id", default="v10_public_default", help="Artifact id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Artifact directory.")
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
            for key in ("rows", "public", "historical_public"):
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
    raise ValueError(f"unsupported historical public format: {path.suffix}")


def _load_adjacent_manifest(path: Path) -> dict[str, Any] | None:
    candidate = path.parent / "historical_manifest.json"
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_selection_sunday_dataset(args.dataset)
    public_rows = _load_rows(args.historical_public) if args.historical_public.exists() else []

    reference_data = None
    if args.public_reference.exists():
        with args.public_reference.open("r", encoding="utf-8") as handle:
            reference_data = json.load(handle)
    seed_history = None
    if args.seed_history.exists():
        with args.seed_history.open("r", encoding="utf-8") as handle:
            seed_history = json.load(handle)

    artifact = fit_public_round_model(
        [team.model_dump(mode="python") for team in dataset.teams],
        historical_public_rows=public_rows,
        reference_data=reference_data,
        seed_history=seed_history,
        model_id=args.model_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"public_field_{args.model_id}.pkl"
    save_public_field_artifact(artifact, artifact_path)

    manifest_path = args.output_dir / f"public_field_{args.model_id}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": artifact["model_id"],
                "historical_manifest_path": str(args.historical_public.parent / "historical_manifest.json"),
                "historical_manifest": _load_adjacent_manifest(args.historical_public),
                "public_history_mode": artifact.get("public_history_mode", "reference_only"),
                "historical_public_row_count": artifact.get("historical_public_row_count", 0),
                "effective_historical_public_row_count": artifact.get("effective_historical_public_row_count", 0),
                "historical_public_seasons": artifact.get("historical_public_seasons", []),
                "historical_public_source_types": artifact.get("historical_public_source_types", []),
                "effective_historical_public_source_types": artifact.get("effective_historical_public_source_types", []),
                "historical_round_scale_summary": artifact.get("historical_round_scale_summary", {}),
                "release_eligible": artifact.get("release_eligible", False),
                "artifact_path": str(artifact_path),
            },
            handle,
            indent=2,
        )

    print(f"[v10-public] model_id={artifact['model_id']}")
    print(f"[v10-public] artifact={artifact_path}")
    print(f"[v10-public] manifest={manifest_path}")
    print(f"[v10-public] rows={artifact.get('historical_public_row_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

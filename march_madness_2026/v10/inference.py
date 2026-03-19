from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SNAPSHOT = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
DEFAULT_V10_INFERENCE_SNAPSHOT = PROJECT_ROOT / "data" / "models" / "v10" / "inference" / "2026" / "snapshot.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def materialize_inference_snapshot(
    source_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    resolved_source = source_path or DEFAULT_SOURCE_SNAPSHOT
    resolved_output = output_path or DEFAULT_V10_INFERENCE_SNAPSHOT

    with resolved_source.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "v10_inference_snapshot": "true",
            "v10_inference_source_snapshot": str(resolved_source),
            "v10_inference_materialized_at": _utc_now_iso(),
        }
    )
    payload["metadata"] = metadata

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_output


__all__ = [
    "DEFAULT_SOURCE_SNAPSHOT",
    "DEFAULT_V10_INFERENCE_SNAPSHOT",
    "materialize_inference_snapshot",
]

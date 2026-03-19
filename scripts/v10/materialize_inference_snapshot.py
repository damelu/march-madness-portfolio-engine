#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.v10.inference import (  # noqa: E402
    DEFAULT_SOURCE_SNAPSHOT,
    DEFAULT_V10_INFERENCE_SNAPSHOT,
    materialize_inference_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the isolated V10 inference snapshot.")
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=DEFAULT_SOURCE_SNAPSHOT,
        help="Selection Sunday snapshot to copy into the V10 inference tree.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_V10_INFERENCE_SNAPSHOT,
        help="Target inference snapshot path under data/models/v10/inference/2026.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot_path = materialize_inference_snapshot(
        source_path=args.source_snapshot,
        output_path=args.output_path,
    )
    print(f"[v10-inference] snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


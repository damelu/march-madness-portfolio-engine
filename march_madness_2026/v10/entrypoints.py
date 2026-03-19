from __future__ import annotations

import sys
from collections.abc import Sequence

from ..cli import main_v10
from .inference import materialize_inference_snapshot


def _should_materialize_snapshot(argv: Sequence[str]) -> bool:
    if any(arg in {"-h", "--help", "--use-demo"} for arg in argv):
        return False
    return not any(arg == "--dataset" or arg.startswith("--dataset=") for arg in argv)


def build_materialized_v10_argv(argv: Sequence[str]) -> list[str]:
    resolved_argv = list(argv)
    if not _should_materialize_snapshot(resolved_argv):
        return resolved_argv

    snapshot_path = materialize_inference_snapshot()
    return ["--dataset", str(snapshot_path), *resolved_argv]


def run_materialized_v10_build(argv: Sequence[str] | None = None) -> int:
    resolved_argv = sys.argv[1:] if argv is None else list(argv)
    return main_v10(build_materialized_v10_argv(resolved_argv))


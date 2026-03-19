#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end bracket portfolio pipeline.

Usage:
    uv run python scripts/run_pipeline.py                           # real bracket data
    uv run python scripts/run_pipeline.py --use-demo                # synthetic demo data
    uv run python scripts/run_pipeline.py --num-tournament-sims 50000  # higher fidelity
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.cli import main  # noqa: E402


if __name__ == "__main__":
    # Default to using the real bracket data if no args provided
    argv = sys.argv[1:]
    default_input = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

    if not any(arg in argv for arg in ("--input-json", "--use-demo")):
        if default_input.exists():
            argv = ["--input-json", str(default_input)] + argv
            print(f"[pipeline] Using bracket data: {default_input}")

    if "--output-dir" not in argv:
        output_dir = PROJECT_ROOT / "outputs"
        argv.extend(["--output-dir", str(output_dir)])
        print(f"[pipeline] Output directory: {output_dir}")

    print(f"[pipeline] Starting portfolio engine...")
    start = time.perf_counter()
    exit_code = main(argv)
    elapsed = time.perf_counter() - start
    print(f"[pipeline] Completed in {elapsed:.1f}s")
    sys.exit(exit_code)

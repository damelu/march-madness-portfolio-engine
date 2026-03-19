from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate autobracket_v10 worker outputs.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing worker log dirs.")
    parser.add_argument("--log-prefix", required=True, help="Worker log prefix, e.g. outputs/autobracket_v10_3_opt")
    parser.add_argument("--workers", type=int, required=True, help="Number of workers to scan.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def worker_summary(root: Path, log_prefix: str, worker: int) -> dict[str, Any]:
    log_dir = root / f"{log_prefix}_{worker}"
    status = load_json(log_dir / "status.json") or {}
    best_payload = load_json(log_dir / "best_v10_params.json") or {}
    tsv = log_dir / "experiments_v10.tsv"
    rows = 0
    if tsv.exists():
        with tsv.open("r", encoding="utf-8") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)

    metrics = best_payload.get("metrics") or best_payload.get("best_metrics") or {}
    params = best_payload.get("params") or best_payload.get("best_params") or {}
    score = best_payload.get("score")
    if score is None:
        score = best_payload.get("best_score")
    if score is None:
        score = float("-inf")

    return {
        "worker": worker,
        "log_dir": str(log_dir),
        "rows": rows,
        "state": status.get("state", "missing"),
        "score": float(score),
        "metrics": metrics,
        "params": params,
        "schema_version": best_payload.get("schema_version"),
        "contest_mode": best_payload.get("contest_mode"),
        "evaluation_seeds": best_payload.get("evaluation_seeds"),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    summaries = [worker_summary(root, args.log_prefix, worker) for worker in range(args.workers)]
    ranked = sorted(summaries, key=lambda item: item["score"], reverse=True)
    payload = {
        "root": str(root),
        "log_prefix": args.log_prefix,
        "workers": args.workers,
        "running_workers": sum(1 for item in summaries if item["state"] == "running"),
        "completed_workers": sum(1 for item in summaries if item["state"] == "completed"),
        "total_rows": sum(int(item["rows"]) for item in summaries),
        "best_worker": ranked[0] if ranked else None,
        "top_workers": ranked[:10],
        "workers_detail": summaries,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

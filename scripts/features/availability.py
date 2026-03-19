#!/usr/bin/env python3
"""Build V10 availability, injury, and lineup uncertainty features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INJURIES = PROJECT_ROOT / "data" / "raw" / "injuries" / "injuries_2026.json"
BRACKET_REF = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "availability.json"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_REF.open() as f:
        bracket = json.load(f)

    teams = {t["team_id"]: t for t in bracket["teams"]}

    # Try to load injury data
    injury_data: dict[str, dict] = {}
    if RAW_INJURIES.exists():
        with RAW_INJURIES.open() as f:
            raw = json.load(f)
        injury_data = raw.get("injuries", {})
        print(f"[availability] Loaded injury data for {len(injury_data)} teams")
    else:
        print("[availability] No injury data found, using bracket reference values")

    features = {}
    for team_id, team in teams.items():
        inj = injury_data.get(team_id, {})
        penalty = inj.get("estimated_penalty", team.get("injury_penalty", 0.0))
        injury_count = int(inj.get("injury_count", 0))
        injuries = inj.get("injuries", [])

        severe_statuses = {"out", "doubtful"}
        uncertain_statuses = {"questionable", "day-to-day", "probable"}
        severe = 0
        uncertain = 0
        for item in injuries:
            status = str(item.get("status", "")).lower()
            if status in severe_statuses:
                severe += 1
            elif status in uncertain_statuses:
                uncertain += 1

        injury_uncertainty = min(1.0, (severe * 0.45) + (uncertain * 0.20) + (penalty * 0.15))
        lineup_uncertainty = min(1.0, injury_uncertainty + (0.10 if injury_count > 0 else 0.0))

        features[team_id] = {
            "team_id": team_id,
            "team_name": team["team_name"],
            "injury_penalty": penalty,
            "injury_count": injury_count,
            "injury_uncertainty": round(injury_uncertainty, 4),
            "lineup_uncertainty": round(lineup_uncertainty, 4),
            "source": "espn_injuries" if team_id in injury_data else "bracket_reference",
        }

    output = {
        "feature_family": "availability",
        "season": 2026,
        "team_count": len(features),
        "features": features,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    inj_count = sum(1 for f in features.values() if f["source"] == "espn_injuries")
    print(f"[availability] Generated features for {len(features)} teams ({inj_count} from injury reports)")
    print(f"[availability] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

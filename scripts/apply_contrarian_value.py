#!/usr/bin/env python3
"""
apply_contrarian_value.py — V5 upgrade: integrate public pick rates for contrarian value.

The key insight from top bracket prediction models (Nate Silver, FiveThirtyEight,
Kaggle winners): your edge comes from picking teams the PUBLIC undervalues.

contrarian_value = true_win_probability / public_pick_rate

When contrarian_value > 1.0, the team is undervalued by the public —
picking them gives you more upside per unit of probability.

This script adjusts market_adjustment to reward contrarian picks and
penalize overvalued favorites.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
PICKS_PATH = PROJECT_ROOT / "data" / "reference" / "public_pick_rates.json"


def main() -> int:
    with SNAPSHOT_PATH.open() as f:
        dataset = json.load(f)
    with PICKS_PATH.open() as f:
        picks = json.load(f)

    champion_picks = picks.get("champion_picks", {})
    teams = {t["team_id"]: t for t in dataset["teams"]}

    # BPI championship probabilities — inline the title% for teams with public pick data
    BPI_TITLE = {
        "duke": 24.4, "arizona": 14.0, "michigan": 15.3, "florida": 7.7,
        "houston": 9.4, "iowa-state": 6.0, "purdue": 4.3, "uconn": 2.4,
        "gonzaga": 3.7, "illinois": 4.2, "michigan-state": 1.3, "alabama": 0.9,
        "arkansas": 0.6, "virginia": 0.5, "st-johns": 0.5, "kansas": 0.5,
        "vanderbilt": 0.6, "nebraska": 0.6, "tennessee": 0.9, "louisville": 0.9,
        "texas-tech": 0.3, "kentucky": 0.2, "wisconsin": 0.2, "byu": 0.2,
        "ucla": 0.1, "saint-marys": 0.05, "north-carolina": 0.05,
    }

    print(f"{'Team':>20s} {'BPI Win%':>9s} {'Public%':>8s} {'Contrarian':>11s} {'Adj':>6s}")
    print("-" * 58)

    adjusted = 0
    for team_id, team in teams.items():
        bpi_title = BPI_TITLE.get(team_id, 0)
        public_pct = champion_picks.get(team_id, 0)

        if bpi_title <= 0 or public_pct <= 0:
            continue
        if bpi_title <= 0:
            continue

        # Contrarian value: true probability / public pick rate
        contrarian = bpi_title / public_pct

        # Convert contrarian value to a market adjustment modifier
        # contrarian > 1.0 → undervalued → positive adjustment
        # contrarian < 1.0 → overvalued → negative adjustment
        # Scale: contrarian 3.0 → +0.50, contrarian 0.5 → -0.30
        contrarian_adj = round(math.log(max(0.1, contrarian)) * 0.35, 2)
        contrarian_adj = max(-0.40, min(0.60, contrarian_adj))

        # Idempotent: store base_market_adjustment on first run, always compute from it
        if "base_market_adjustment" not in team:
            team["base_market_adjustment"] = team["market_adjustment"]
        base_mkt = team["base_market_adjustment"]
        team["market_adjustment"] = round(0.70 * base_mkt + 0.30 * contrarian_adj, 2)

        # V6: Populate public pick field for opponent simulation
        team["public_pick_pct"] = public_pct
        adjusted += 1

        print(f"{team['team_name']:>20s} {bpi_title:8.1f}% {public_pct:7.1f}% {contrarian:10.2f} {contrarian_adj:+5.2f}")

    # Update metadata
    dataset["metadata"]["contrarian_calibration"] = "applied"
    dataset["metadata"]["source"] = "kenpom_bpi_vegas_contrarian_v5"

    with SNAPSHOT_PATH.open("w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n[contrarian] Adjusted {adjusted} teams with contrarian value")
    print(f"[contrarian] Saved to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

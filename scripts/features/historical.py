#!/usr/bin/env python3
"""Build V10 historical, coaching, and seed-context features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_COACHING = PROJECT_ROOT / "data" / "raw" / "coaches" / "coaching_records_2026.json"
SEED_HISTORY = PROJECT_ROOT / "data" / "reference" / "seed_matchup_history.json"
BRACKET_REF = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "historical.json"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_REF.open() as f:
        bracket = json.load(f)

    teams = {t["team_id"]: t for t in bracket["teams"]}

    # Load coaching data
    coaching: dict[str, dict] = {}
    if RAW_COACHING.exists():
        with RAW_COACHING.open() as f:
            raw = json.load(f)
        coaching = raw.get("coaches", {})
        print(f"[historical] Loaded coaching data for {len(coaching)} teams")

    # Load seed history
    seed_adv_rates: dict[str, dict] = {}
    if SEED_HISTORY.exists():
        with SEED_HISTORY.open() as f:
            history = json.load(f)
        seed_adv_rates = history.get("advancement_rates", {})
        print(f"[historical] Loaded seed advancement rates")

    features = {}
    for team_id, team in teams.items():
        coach = coaching.get(team_id, {})
        seed = str(team["seed"])

        # Coaching adjustment from records
        coaching_adj = coach.get("coaching_adjustment", team.get("coaching_adjustment", 0.0))

        # Continuity adjustment (from bracket reference, could be enhanced with roster data)
        continuity_adj = team.get("continuity_adjustment", 0.0)

        # Seed-based historical advancement rates
        hist_f4_rate = seed_adv_rates.get("final_four", {}).get(seed, 0.0)
        hist_champ_rate = seed_adv_rates.get("champion", {}).get(seed, 0.0)

        # Volatility adjustment based on seed
        # Lower seeds are historically more volatile in March
        base_vol = 0.07 + (team["seed"] - 1) * 0.01
        volatility = round(min(0.35, max(0.05, base_vol)), 2)
        years = float(coach.get("years", 0))
        tourn_wins = float(coach.get("tourn_wins", 0))
        tourn_losses = float(coach.get("tourn_losses", 0))
        total_games = tourn_wins + tourn_losses
        coach_win_rate = round(tourn_wins / total_games, 4) if total_games else 0.5
        experience_score = min(
            1.0,
            0.35
            + (0.25 * min(years / 20.0, 1.0))
            + (0.25 * hist_f4_rate)
            + (0.15 * coach_win_rate),
        )

        features[team_id] = {
            "team_id": team_id,
            "team_name": team["team_name"],
            "coaching_adjustment": coaching_adj,
            "continuity_adjustment": continuity_adj,
            "volatility": volatility,
            "historical_f4_rate": hist_f4_rate,
            "historical_champ_rate": hist_champ_rate,
            "coach_years": years,
            "coach_tournament_win_rate": coach_win_rate,
            "experience_score": round(experience_score, 4),
            "coach_name": coach.get("coach", "Unknown"),
            "coaching_source": "compiled_records" if team_id in coaching else "bracket_reference",
        }

    output = {
        "feature_family": "historical",
        "season": 2026,
        "team_count": len(features),
        "features": features,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    coach_count = sum(1 for f in features.values() if f["coaching_source"] == "compiled_records")
    print(f"[historical] Generated features for {len(features)} teams ({coach_count} with coaching records)")
    print(f"[historical] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

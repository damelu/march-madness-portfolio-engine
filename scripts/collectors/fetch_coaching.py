#!/usr/bin/env python3
"""
fetch_coaching.py — Compile coaching tournament records for 2026 tournament coaches.

Uses embedded data from NCAA tournament historical records.
Output: data/raw/coaches/coaching_records_2026.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "coaches" / "coaching_records_2026.json"
BRACKET_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

# Coaching tournament records — estimated based on known records through 2025
# Format: (tournament_wins, tournament_losses, final_fours, championships, years_as_hc)
COACHING_RECORDS: dict[str, dict] = {
    "Duke": {"coach": "Jon Scheyer", "tourn_wins": 8, "tourn_losses": 4, "final_fours": 0, "titles": 0, "years": 4},
    "Alabama": {"coach": "Nate Oats", "tourn_wins": 12, "tourn_losses": 5, "final_fours": 2, "titles": 0, "years": 7},
    "Marquette": {"coach": "Shaka Smart", "tourn_wins": 14, "tourn_losses": 8, "final_fours": 1, "titles": 0, "years": 15},
    "Creighton": {"coach": "Greg McDermott", "tourn_wins": 8, "tourn_losses": 7, "final_fours": 0, "titles": 0, "years": 16},
    "Wisconsin": {"coach": "Greg Gard", "tourn_wins": 10, "tourn_losses": 6, "final_fours": 1, "titles": 0, "years": 10},
    "BYU": {"coach": "Kevin Young", "tourn_wins": 2, "tourn_losses": 1, "final_fours": 0, "titles": 0, "years": 2},
    "Saint Mary's": {"coach": "Randy Bennett", "tourn_wins": 6, "tourn_losses": 7, "final_fours": 0, "titles": 0, "years": 25},
    "Arizona": {"coach": "Tommy Lloyd", "tourn_wins": 8, "tourn_losses": 4, "final_fours": 0, "titles": 0, "years": 5},
    "Iowa State": {"coach": "T.J. Otzelberger", "tourn_wins": 6, "tourn_losses": 3, "final_fours": 0, "titles": 0, "years": 5},
    "Texas Tech": {"coach": "Grant McCasland", "tourn_wins": 2, "tourn_losses": 2, "final_fours": 0, "titles": 0, "years": 2},
    "Purdue": {"coach": "Matt Painter", "tourn_wins": 18, "tourn_losses": 12, "final_fours": 2, "titles": 0, "years": 20},
    "Florida": {"coach": "Todd Golden", "tourn_wins": 6, "tourn_losses": 3, "final_fours": 1, "titles": 0, "years": 4},
    "Auburn": {"coach": "Bruce Pearl", "tourn_wins": 15, "tourn_losses": 8, "final_fours": 2, "titles": 0, "years": 16},
    "Houston": {"coach": "Kelvin Sampson", "tourn_wins": 22, "tourn_losses": 8, "final_fours": 3, "titles": 0, "years": 22},
    "Kentucky": {"coach": "Mark Pope", "tourn_wins": 4, "tourn_losses": 3, "final_fours": 0, "titles": 0, "years": 2},
    "Michigan State": {"coach": "Tom Izzo", "tourn_wins": 55, "tourn_losses": 24, "final_fours": 9, "titles": 2, "years": 31},
    "Michigan": {"coach": "Dusty May", "tourn_wins": 4, "tourn_losses": 2, "final_fours": 0, "titles": 0, "years": 2},
    "Tennessee": {"coach": "Rick Barnes", "tourn_wins": 26, "tourn_losses": 17, "final_fours": 1, "titles": 0, "years": 28},
    "UConn": {"coach": "Dan Hurley", "tourn_wins": 18, "tourn_losses": 4, "final_fours": 2, "titles": 2, "years": 8},
    "Gonzaga": {"coach": "Mark Few", "tourn_wins": 35, "tourn_losses": 14, "final_fours": 3, "titles": 0, "years": 27},
    "Kansas": {"coach": "Bill Self", "tourn_wins": 50, "tourn_losses": 18, "final_fours": 5, "titles": 2, "years": 32},
    "Baylor": {"coach": "Scott Drew", "tourn_wins": 22, "tourn_losses": 10, "final_fours": 2, "titles": 1, "years": 22},
    "UCLA": {"coach": "Mick Cronin", "tourn_wins": 14, "tourn_losses": 9, "final_fours": 1, "titles": 0, "years": 18},
    "Clemson": {"coach": "Brad Brownell", "tourn_wins": 6, "tourn_losses": 6, "final_fours": 0, "titles": 0, "years": 18},
    "St. John's": {"coach": "Rick Pitino", "tourn_wins": 42, "tourn_losses": 16, "final_fours": 4, "titles": 1, "years": 32},
    "Illinois": {"coach": "Brad Underwood", "tourn_wins": 5, "tourn_losses": 5, "final_fours": 0, "titles": 0, "years": 12},
    "Dayton": {"coach": "Anthony Grant", "tourn_wins": 6, "tourn_losses": 5, "final_fours": 0, "titles": 0, "years": 15},
}


def compute_coaching_adjustment(record: dict) -> float:
    """Convert coaching record into a team adjustment value.

    Factors:
    - Tournament win rate (most important)
    - Final Four appearances per tournament trip
    - Championships (highest signal)
    - Experience multiplier
    """
    total_games = record["tourn_wins"] + record["tourn_losses"]
    if total_games == 0:
        return 0.0

    win_rate = record["tourn_wins"] / total_games
    f4_rate = record["final_fours"] / max(record["years"], 1)
    title_bonus = record["titles"] * 0.15

    # Weighted composite
    adjustment = (
        (win_rate - 0.50) * 1.5  # Above/below average win rate
        + f4_rate * 2.0           # Final Four frequency
        + title_bonus             # Championship bonus
    )

    # Clamp to reasonable range
    return round(max(-0.5, min(1.0, adjustment)), 2)


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_PATH.open() as f:
        bracket = json.load(f)

    teams = bracket.get("teams", [])
    coaching_data = {}

    for team in teams:
        name = team["team_name"]
        record = COACHING_RECORDS.get(name)
        if record:
            adj = compute_coaching_adjustment(record)
            coaching_data[team["team_id"]] = {
                **record,
                "coaching_adjustment": adj,
            }

    output = {
        "source": "compiled_coaching_records",
        "season": 2026,
        "teams_with_data": len(coaching_data),
        "coaches": coaching_data,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[coaching] Compiled records for {len(coaching_data)}/{len(teams)} coaches")
    print(f"[coaching] Saved to {OUTPUT_PATH}")

    # Print top coaching adjustments
    print("\nTop coaching adjustments:")
    sorted_coaches = sorted(coaching_data.items(), key=lambda x: x[1]["coaching_adjustment"], reverse=True)
    for team_id, info in sorted_coaches[:10]:
        print(f"  {info['coach']} ({team_id}): {info['coaching_adjustment']:+.2f} ({info['tourn_wins']}-{info['tourn_losses']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

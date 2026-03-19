#!/usr/bin/env python3
"""
fetch_bracket.py — Fetch the 2026 NCAA Tournament bracket structure.

Tries ESPN's scoreboard API first, falls back to the reference file.
Output: data/reference/bracket_2026.json (SelectionSundayDataset format)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

# ESPN endpoints for NCAA tournament data
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
ESPN_RANKINGS = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/rankings"


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "").replace("'", "").replace("&", "and")


def fetch_espn_rankings() -> list[dict] | None:
    """Fetch AP/Coaches rankings from ESPN to get team names and IDs."""
    try:
        resp = httpx.get(ESPN_RANKINGS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        teams = []
        for ranking in data.get("rankings", []):
            for rank in ranking.get("ranks", []):
                team_info = rank.get("team", {})
                teams.append({
                    "espn_id": team_info.get("id"),
                    "name": team_info.get("location", team_info.get("name", "")),
                    "abbreviation": team_info.get("abbreviation", ""),
                    "rank": rank.get("current"),
                })
        return teams if teams else None
    except Exception as e:
        print(f"[fetch_bracket] ESPN rankings fetch failed: {e}", file=sys.stderr)
        return None


def fetch_espn_tournament_games() -> list[dict] | None:
    """Try to fetch NCAA tournament games from ESPN scoreboard API."""
    try:
        params = {
            "dates": "20260317-20260406",
            "groups": "100",  # NCAA tournament group
            "limit": "100",
        }
        resp = httpx.get(ESPN_SCOREBOARD, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        if events:
            games = []
            for event in events:
                for comp in event.get("competitions", []):
                    game = {
                        "id": event.get("id"),
                        "name": event.get("name"),
                        "date": event.get("date"),
                    }
                    for team_entry in comp.get("competitors", []):
                        side = "home" if team_entry.get("homeAway") == "home" else "away"
                        game[f"{side}_team"] = team_entry.get("team", {}).get("location", "")
                        game[f"{side}_seed"] = team_entry.get("curatedRank", {}).get("current")
                    games.append(game)
            return games if games else None
        return None
    except Exception as e:
        print(f"[fetch_bracket] ESPN tournament fetch failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    print("[fetch_bracket] Attempting ESPN API fetch...")

    # Try to get tournament data
    games = fetch_espn_tournament_games()
    if games:
        print(f"[fetch_bracket] Found {len(games)} tournament games from ESPN")
        # TODO: Parse games into bracket structure
        # For now, fall through to reference file

    rankings = fetch_espn_rankings()
    if rankings:
        print(f"[fetch_bracket] Found {len(rankings)} ranked teams from ESPN")

    # Check if reference file already exists
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            data = json.load(f)
        team_count = len(data.get("teams", []))
        print(f"[fetch_bracket] Reference file exists with {team_count} teams: {OUTPUT_PATH}")

        # Validate structure
        regions = set()
        for team in data.get("teams", []):
            regions.add(team.get("region"))
        print(f"[fetch_bracket] Regions: {sorted(regions)}")
        print(f"[fetch_bracket] #1 seeds: {', '.join(t['team_name'] for t in data['teams'] if t['seed'] == 1)}")
        return 0

    print("[fetch_bracket] No reference file found. Create data/reference/bracket_2026.json manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

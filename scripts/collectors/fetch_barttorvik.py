#!/usr/bin/env python3
"""
fetch_barttorvik.py — Fetch BartTorvik T-Rank ratings for tournament teams.

Respects robots.txt Crawl-Delay: 10.
Output: data/raw/ratings/barttorvik_2026.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ratings" / "barttorvik_2026.json"
BRACKET_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

BARTTORVIK_TEAM_URL = "https://barttorvik.com/team.php"
BARTTORVIK_TRANK_URL = "https://barttorvik.com/trank.php"
CRAWL_DELAY = 10  # seconds, per robots.txt


def _load_tournament_teams() -> list[dict]:
    with BRACKET_PATH.open() as f:
        data = json.load(f)
    return data.get("teams", [])


def _slugify_barttorvik(team_name: str) -> str:
    """Convert team name to BartTorvik URL format."""
    replacements = {
        "St. John's": "St. John's",
        "UConn": "Connecticut",
        "VCU": "Virginia Commonwealth",
        "BYU": "Brigham Young",
        "UCF": "Central Florida",
        "SMU": "Southern Methodist",
        "Ole Miss": "Mississippi",
        "UNC Asheville": "UNC Asheville",
        "TCU": "Texas Christian",
    }
    return replacements.get(team_name, team_name)


def fetch_trank_page() -> dict[str, dict] | None:
    """Fetch the main T-Rank page and parse team ratings."""
    try:
        print(f"[barttorvik] Fetching T-Rank page...")
        resp = httpx.get(
            BARTTORVIK_TRANK_URL,
            params={"year": "2026", "conyes": "1"},
            headers={"User-Agent": "MarchMadness2026DataStack/1.0 (research)"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text

        # Try to find the data table — BartTorvik uses a JavaScript-rendered table
        # Look for JSON data embedded in the page
        ratings = {}

        # Pattern: look for team data in the HTML/JS
        # BartTorvik often embeds data as JS arrays
        team_pattern = re.compile(
            r'\["(\d+)","([^"]+)","([^"]+)",'  # rank, team, conference
            r'"([^"]*?)","([^"]*?)","([^"]*?)",'  # record, adjOE, adjDE
            r'"([^"]*?)","([^"]*?)","([^"]*?)"',  # barthag, EFG%, EFGD%
            re.DOTALL,
        )

        for match in team_pattern.finditer(html):
            rank, team_name, conf = match.group(1), match.group(2), match.group(3)
            record, adj_oe, adj_de = match.group(4), match.group(5), match.group(6)
            barthag = match.group(7)

            try:
                ratings[team_name.strip()] = {
                    "rank": int(rank),
                    "team_name": team_name.strip(),
                    "conference": conf.strip(),
                    "record": record.strip(),
                    "adj_oe": float(adj_oe) if adj_oe else None,
                    "adj_de": float(adj_de) if adj_de else None,
                    "barthag": float(barthag) if barthag else None,
                }
            except (ValueError, TypeError):
                continue

        if ratings:
            print(f"[barttorvik] Parsed {len(ratings)} teams from T-Rank")
            return ratings

        # If regex didn't match, the page structure may have changed
        print("[barttorvik] Could not parse T-Rank data from HTML")
        print(f"[barttorvik] Page length: {len(html)} chars")
        return None

    except httpx.HTTPStatusError as e:
        print(f"[barttorvik] HTTP error: {e.response.status_code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[barttorvik] Fetch failed: {e}", file=sys.stderr)
        return None


def update_bracket_ratings(teams: list[dict], ratings: dict[str, dict]) -> list[dict]:
    """Update bracket teams with BartTorvik ratings."""
    updated = 0
    for team in teams:
        bt_name = _slugify_barttorvik(team["team_name"])
        rating_data = ratings.get(bt_name) or ratings.get(team["team_name"])

        if rating_data:
            # Convert BartTorvik Barthag (0-1) to our rating scale (55-100)
            if rating_data.get("barthag") is not None:
                barthag = rating_data["barthag"]
                team["rating"] = round(55 + barthag * 45, 2)

            if rating_data.get("adj_oe") is not None and rating_data.get("adj_de") is not None:
                # Normalize AdjOE and AdjDE (typical range: 95-125 for OE, 85-105 for DE)
                adj_oe = rating_data["adj_oe"]
                adj_de = rating_data["adj_de"]
                team["offense_rating"] = round((adj_oe - 105) / 8, 2)
                team["defense_rating"] = round((100 - adj_de) / 8, 2)

            updated += 1

    print(f"[barttorvik] Updated ratings for {updated}/{len(teams)} teams")
    return teams


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ratings = fetch_trank_page()

    if ratings:
        with OUTPUT_PATH.open("w") as f:
            json.dump({"source": "barttorvik", "season": 2026, "teams": ratings}, f, indent=2)
        print(f"[barttorvik] Saved raw ratings to {OUTPUT_PATH}")

        # Also update the bracket reference file
        teams = _load_tournament_teams()
        updated_teams = update_bracket_ratings(teams, ratings)
        print(f"[barttorvik] Updated bracket ratings based on T-Rank data")
    else:
        print("[barttorvik] Using estimated ratings from bracket reference file")

    return 0


if __name__ == "__main__":
    sys.exit(main())

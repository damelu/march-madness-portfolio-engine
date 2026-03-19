#!/usr/bin/env python3
"""
fetch_ncaa_stats.py — Fetch season statistics from ESPN/NCAA API.

Uses ESPN's undocumented API (free, no auth).
Output: data/raw/ncaa/team_season_stats_2026.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ncaa" / "team_season_stats_2026.json"
BRACKET_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

ESPN_TEAM_STATS = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/statistics"
ESPN_TEAM_SEARCH = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"

NAME_ALIASES = {
    "BYU": "Brigham Young Cougars",
    "UConn": "UConn Huskies",
    "Saint Mary's": "Saint Mary's Gaels",
    "St. John's": "St. John's Red Storm",
    "VCU": "VCU Rams",
    "UCF": "UCF Knights",
    "SMU": "SMU Mustangs",
    "Ole Miss": "Ole Miss Rebels",
    "Miami (FL)": "Miami Hurricanes",
    "North Carolina": "North Carolina Tar Heels",
    "Texas A&M": "Texas A&M Aggies",
    "McNeese": "McNeese Cowboys",
    "Saint Louis": "Saint Louis Billikens",
    "Louisville": "Louisville Cardinals",
    "Ohio State": "Ohio State Buckeyes",
    "South Florida": "South Florida Bulls",
    "Cal Baptist": "California Baptist Lancers",
    "High Point": "High Point Panthers",
    "Hawaii": "Hawai'i Rainbow Warriors",
    "LIU": "LIU Sharks",
    "Virginia": "Virginia Cavaliers",
    "Georgia": "Georgia Bulldogs",
    "Santa Clara": "Santa Clara Broncos",
    "Akron": "Akron Zips",
    "Hofstra": "Hofstra Pride",
    "Wright State": "Wright State Raiders",
    "Tennessee State": "Tennessee State Tigers",
    "Howard": "Howard Bison",
    "Penn": "Penn Quakers",
    "Lehigh": "Lehigh Mountain Hawks",
    "Idaho": "Idaho Vandals",
    "Northern Iowa": "Northern Iowa Panthers",
    "Queens": "Queens University Royals",
}


def _normalize_name(value: str) -> str:
    text = value.lower().strip()
    for source, target in (
        ("st.", "saint"),
        ("st ", "saint "),
        ("saint mary's", "saint marys"),
        ("miami (fl)", "miami"),
        ("texas a&m", "texas am"),
        ("unc ", "north carolina "),
        ("uconn", "connecticut"),
    ):
        text = text.replace(source, target)
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _load_team_directory() -> dict[str, str]:
    response = httpx.get(
        ESPN_TEAM_SEARCH,
        params={"limit": 500, "groups": "50"},
        timeout=30,
        headers={"User-Agent": "MarchMadness2026/1.0"},
    )
    response.raise_for_status()
    data = response.json()
    teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    directory: dict[str, str] = {}
    for entry in teams:
        team = entry.get("team", {})
        team_id = team.get("id")
        if not team_id:
            continue
        names = {
            team.get("displayName", ""),
            team.get("shortDisplayName", ""),
            team.get("location", ""),
            team.get("name", ""),
        }
        for name in names:
            if name:
                directory[_normalize_name(name)] = team_id
    return directory


def _resolve_espn_id(team_name: str, directory: dict[str, str]) -> str | None:
    for candidate in (team_name, NAME_ALIASES.get(team_name, "")):
        if candidate:
            team_id = directory.get(_normalize_name(candidate))
            if team_id:
                return team_id
    return None


def fetch_team_stats(team_name: str, espn_id: str) -> dict | None:
    """Fetch season statistics for a team from ESPN."""
    try:
        resp = httpx.get(
            ESPN_TEAM_STATS.format(team_id=espn_id),
            timeout=10,
            headers={"User-Agent": "MarchMadness2026/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        stats = {}
        for category in data.get("results", {}).get("stats", {}).get("categories", []):
            cat_name = category.get("name", "")
            for stat in category.get("stats", []):
                stat_name = stat.get("name", "")
                value = stat.get("value")
                if value is not None:
                    stats[f"{cat_name}_{stat_name}"] = value

        return {
            "team_name": team_name,
            "espn_id": espn_id,
            "stats": stats,
        }
    except Exception:
        return None


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load tournament teams
    with BRACKET_PATH.open() as f:
        bracket = json.load(f)

    teams = bracket.get("teams", [])
    results = []
    fetched = 0
    failed = 0
    directory = _load_team_directory()

    print(f"[ncaa_stats] Fetching stats for {len(teams)} tournament teams...")

    for team in teams:
        name = team["team_name"]
        espn_id = _resolve_espn_id(name, directory)

        if not espn_id:
            print(f"  [skip] No ESPN ID for {name}")
            failed += 1
            continue

        stats = fetch_team_stats(name, espn_id)
        if stats:
            results.append(stats)
            fetched += 1
        else:
            failed += 1

        time.sleep(0.5)  # Rate limiting

    output = {
        "source": "espn_api",
        "season": 2026,
        "fetched": fetched,
        "failed": failed,
        "teams": results,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[ncaa_stats] Fetched stats for {fetched}/{len(teams)} teams")
    print(f"[ncaa_stats] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

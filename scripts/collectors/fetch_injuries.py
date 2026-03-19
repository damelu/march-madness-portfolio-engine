#!/usr/bin/env python3
"""
fetch_injuries.py — Fetch injury reports for tournament teams.

Uses ESPN injury API (free, no auth).
Output: data/raw/injuries/injuries_2026.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "injuries" / "injuries_2026.json"
BRACKET_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/injuries"
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


def fetch_team_injuries(team_name: str, espn_id: str) -> list[dict]:
    """Fetch injury data for a team."""
    try:
        resp = httpx.get(
            ESPN_INJURIES.format(team_id=espn_id),
            timeout=10,
            headers={"User-Agent": "MarchMadness2026/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        injuries = []
        for item in data.get("items", []):
            athlete = item.get("athlete", {})
            injuries.append({
                "player_name": athlete.get("displayName", "Unknown"),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "status": item.get("status", ""),
                "type": item.get("type", {}).get("description", ""),
                "detail": item.get("longComment", item.get("shortComment", "")),
                "date": item.get("date", ""),
            })
        return injuries
    except Exception:
        return []


def compute_injury_penalty(injuries: list[dict]) -> float:
    """Estimate injury impact on team strength."""
    if not injuries:
        return 0.0

    penalty = 0.0
    for injury in injuries:
        status = injury.get("status", "").lower()
        if "out" in status:
            penalty += 0.4  # Significant impact
        elif "doubtful" in status:
            penalty += 0.3
        elif "questionable" in status:
            penalty += 0.15
        elif "day-to-day" in status or "probable" in status:
            penalty += 0.05

    return min(penalty, 1.5)  # Cap total penalty


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_PATH.open() as f:
        bracket = json.load(f)

    teams = bracket.get("teams", [])
    all_injuries = {}
    teams_with_injuries = 0
    directory = _load_team_directory()

    print(f"[injuries] Checking injuries for {len(teams)} tournament teams...")

    for team in teams:
        name = team["team_name"]
        espn_id = _resolve_espn_id(name, directory)

        if not espn_id:
            continue

        injuries = fetch_team_injuries(name, espn_id)
        if injuries:
            penalty = compute_injury_penalty(injuries)
            all_injuries[team["team_id"]] = {
                "team_name": name,
                "injuries": injuries,
                "injury_count": len(injuries),
                "estimated_penalty": round(penalty, 2),
            }
            teams_with_injuries += 1
            print(f"  {name}: {len(injuries)} injuries (penalty: {penalty:.2f})")

        time.sleep(0.3)

    output = {
        "source": "espn_injuries",
        "season": 2026,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "teams_with_injuries": teams_with_injuries,
        "injuries": all_injuries,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[injuries] Found injuries for {teams_with_injuries} teams")
    print(f"[injuries] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

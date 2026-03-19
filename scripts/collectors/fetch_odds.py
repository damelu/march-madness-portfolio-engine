#!/usr/bin/env python3
"""
fetch_odds.py — Fetch betting odds from The Odds API.

Requires THE_ODDS_API_KEY in .env (free tier: 500 requests/month).
Output: data/raw/odds/tournament_odds_2026.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "odds" / "tournament_odds_2026.json"
BRACKET_PATH = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"

load_dotenv(PROJECT_ROOT / ".env")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"


def _get_api_key() -> str | None:
    return os.environ.get("THE_ODDS_API_KEY")


def fetch_championship_futures(api_key: str) -> list[dict] | None:
    """Fetch NCAA tournament championship futures (outrights)."""
    try:
        resp = httpx.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "outrights",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"[odds] API requests remaining: {remaining}")
        return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"[odds] HTTP error: {e.response.status_code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[odds] Futures fetch failed: {e}", file=sys.stderr)
        return None


def fetch_game_odds(api_key: str) -> list[dict] | None:
    """Fetch first-round game spreads and moneylines."""
    try:
        resp = httpx.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads,h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=15,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"[odds] API requests remaining: {remaining}")
        return resp.json()
    except httpx.HTTPStatusError as e:
        print(f"[odds] HTTP error: {e.response.status_code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[odds] Game odds fetch failed: {e}", file=sys.stderr)
        return None


def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def compute_market_adjustments(futures: list[dict], teams: list[dict]) -> dict[str, float]:
    """Compute market adjustment for each team based on championship futures."""
    # Collect implied probabilities from futures
    team_probs: dict[str, list[float]] = {}
    for event in futures:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is not None:
                        prob = american_to_implied_prob(int(price))
                        team_probs.setdefault(name, []).append(prob)

    # Average across bookmakers and convert to market adjustment
    adjustments = {}
    for team_name, probs in team_probs.items():
        avg_prob = sum(probs) / len(probs)
        # Market adjustment: higher implied prob = positive adjustment
        # Scale: championship probability of 0.15 (strong favorite) → +1.0
        adjustments[team_name] = round((avg_prob - 0.05) * 10, 2)

    return adjustments


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    api_key = _get_api_key()
    if not api_key:
        print("[odds] No THE_ODDS_API_KEY found in .env")
        print("[odds] Using estimated market adjustments from bracket reference file")
        return 0

    print("[odds] Fetching championship futures...")
    futures = fetch_championship_futures(api_key)

    print("[odds] Fetching game odds...")
    games = fetch_game_odds(api_key)

    output = {
        "source": "the_odds_api",
        "season": 2026,
        "championship_futures": futures or [],
        "game_odds": games or [],
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[odds] Saved odds data to {OUTPUT_PATH}")

    # Compute market adjustments
    if futures:
        with BRACKET_PATH.open() as f:
            bracket = json.load(f)
        adjustments = compute_market_adjustments(futures, bracket.get("teams", []))
        print(f"[odds] Computed market adjustments for {len(adjustments)} teams")
        for team, adj in sorted(adjustments.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {team}: {adj:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

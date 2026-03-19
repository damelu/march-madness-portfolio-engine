#!/usr/bin/env python3
"""Build V10 market-prior features from futures and round-1 odds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ODDS = PROJECT_ROOT / "data" / "raw" / "odds" / "tournament_odds_2026.json"
BRACKET_REF = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "market_priors.json"

NAME_ALIASES = {
    "Saint Mary's": "Saint Mary's Gaels",
    "St. John's": "St. John's Red Storm",
    "BYU": "BYU Cougars",
    "UConn": "Connecticut Huskies",
    "North Carolina": "North Carolina Tar Heels",
    "Texas A&M": "Texas A&M Aggies",
    "Miami (FL)": "Miami Hurricanes",
    "Ohio State": "Ohio State Buckeyes",
    "South Florida": "South Florida Bulls",
    "Saint Louis": "Saint Louis Billikens",
    "Santa Clara": "Santa Clara Broncos",
    "Lehigh": "Lehigh Mountain Hawks",
}


def _normalize_name(value: str) -> str:
    text = value.lower().strip()
    replacements = {
        "st.": "saint",
        "saint mary's": "saint marys",
        "texas a&m": "texas am",
        "uconn": "connecticut",
        "miami (fl)": "miami",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _american_to_implied_prob(price: int) -> float:
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def _team_keys(team_name: str) -> set[str]:
    keys = {_normalize_name(team_name)}
    alias = NAME_ALIASES.get(team_name)
    if alias:
        keys.add(_normalize_name(alias))
    return keys


def _extract_round1_market_data(raw_odds: dict, teams_by_id: dict[str, dict]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    probability_samples: dict[str, list[float]] = {team_id: [] for team_id in teams_by_id}
    spread_samples: dict[str, list[float]] = {team_id: [] for team_id in teams_by_id}
    team_lookup = {
        key: team_id
        for team_id, team in teams_by_id.items()
        for key in _team_keys(team["team_name"])
    }

    for event in raw_odds.get("game_odds", []):
        seen_ids = {
            team_lookup[_normalize_name(name)]
            for name in (event.get("home_team", ""), event.get("away_team", ""))
            if _normalize_name(name) in team_lookup
        }
        if len(seen_ids) < 2:
            continue
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    normalized_name = _normalize_name(outcome.get("name", ""))
                    team_id = team_lookup.get(normalized_name)
                    if team_id is None:
                        continue
                    if market_key == "h2h" and outcome.get("price") is not None:
                        probability_samples[team_id].append(
                            _american_to_implied_prob(int(outcome["price"]))
                        )
                    elif market_key == "spreads" and outcome.get("point") is not None:
                        spread_samples[team_id].append(float(outcome["point"]))
    return probability_samples, spread_samples


def _extract_futures_adjustments(raw_odds: dict) -> dict[str, float]:
    adjustments: dict[str, list[float]] = {}
    for event in raw_odds.get("championship_futures", []):
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    adjustments.setdefault(_normalize_name(outcome.get("name", "")), []).append(
                        _american_to_implied_prob(int(price))
                    )
    return {
        name: (sum(samples) / len(samples))
        for name, samples in adjustments.items()
        if samples
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_REF.open() as handle:
        bracket = json.load(handle)
    teams_by_id = {team["team_id"]: team for team in bracket["teams"]}

    raw_odds = {}
    if RAW_ODDS.exists():
        with RAW_ODDS.open() as handle:
            raw_odds = json.load(handle)
        print(
            f"[market_priors] Loaded odds: futures={len(raw_odds.get('championship_futures', []))}, "
            f"games={len(raw_odds.get('game_odds', []))}"
        )
    else:
        print("[market_priors] No odds data found, using bracket reference values")

    round1_probs, round1_spreads = _extract_round1_market_data(raw_odds, teams_by_id)
    futures = _extract_futures_adjustments(raw_odds)

    features = {}
    for team_id, team in teams_by_id.items():
        samples = round1_probs.get(team_id, [])
        spreads = round1_spreads.get(team_id, [])
        normalized_name_keys = _team_keys(team["team_name"])
        future_prob = next((futures[key] for key in normalized_name_keys if key in futures), None)

        if future_prob is not None:
            market_adjustment = round((future_prob - 0.05) * 10.0, 2)
            source = "the_odds_api_futures"
        elif samples:
            avg_prob = sum(samples) / len(samples)
            market_adjustment = round((avg_prob - 0.5) * 3.5, 2)
            source = "the_odds_api_round1"
        else:
            market_adjustment = team["market_adjustment"]
            source = "bracket_reference"

        features[team_id] = {
            "team_id": team_id,
            "team_name": team["team_name"],
            "market_adjustment": market_adjustment,
            "round1_market_win_prob": round(sum(samples) / len(samples), 4) if samples else None,
            "avg_round1_spread": round(sum(spreads) / len(spreads), 3) if spreads else None,
            "source": source,
        }

    output = {
        "feature_family": "market_priors",
        "season": 2026,
        "team_count": len(features),
        "features": features,
    }

    with OUTPUT_PATH.open("w") as handle:
        json.dump(output, handle, indent=2)

    source_counts: dict[str, int] = {}
    for feature in features.values():
        source = str(feature["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"[market_priors] Generated features for {len(features)} teams: {source_counts}")
    print(f"[market_priors] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

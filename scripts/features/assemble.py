#!/usr/bin/env python3
"""Assemble all feature families into a V10 SelectionSundayDataset snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "selection_sunday"
BRACKET_REF = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"
OUTPUT_PATH = FEATURES_DIR / "snapshot.json"

FEATURE_FILES = {
    "team_strength": FEATURES_DIR / "team_strength.json",
    "market_priors": FEATURES_DIR / "market_priors.json",
    "availability": FEATURES_DIR / "availability.json",
    "historical": FEATURES_DIR / "historical.json",
}

MERGE_EXCLUDE_KEYS = {
    "team_id",
    "team_name",
    "source",
    "feature_family",
    "season",
    "team_count",
}


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load bracket reference for base team list
    with BRACKET_REF.open() as f:
        bracket = json.load(f)

    teams_by_id = {t["team_id"]: dict(t) for t in bracket["teams"]}

    # Load and merge each feature family
    sources_used = []
    for family_name, path in FEATURE_FILES.items():
        if not path.exists():
            print(f"[assemble] Missing feature file: {path} — using bracket reference values")
            continue

        with path.open() as f:
            data = json.load(f)

        features = data.get("features", {})
        merged_count = 0

        for team_id, feature_values in features.items():
            if team_id not in teams_by_id:
                continue

            team = teams_by_id[team_id]

            for key, value in feature_values.items():
                if key in MERGE_EXCLUDE_KEYS:
                    continue
                if value is None:
                    continue
                team[key] = value

            if source := feature_values.get("source"):
                team[f"{family_name}_source"] = source

            merged_count += 1

        print(f"[assemble] Merged {family_name}: {merged_count} teams")
        sources_used.append(family_name)

    teams_list = list(teams_by_id.values())
    field_mean_rating = sum(team["rating"] for team in teams_list) / len(teams_list)
    region_mean_rating = {
        region: sum(team["rating"] for team in teams_list if team["region"] == region)
        / max(1, sum(1 for team in teams_list if team["region"] == region))
        for region in {team["region"] for team in teams_list}
    }
    rating_order = {
        team["team_id"]: rank
        for rank, team in enumerate(
            sorted(teams_list, key=lambda item: item["rating"], reverse=True),
            start=1,
        )
    }

    for team in teams_list:
        team.setdefault("adj_efg_off", 0.50)
        team.setdefault("adj_efg_def", 0.50)
        team.setdefault("orb_rate_off", 0.28)
        team.setdefault("drb_rate_def", 0.72)
        team.setdefault("turnover_rate_off", 0.16)
        team.setdefault("turnover_rate_def", 0.08)
        team.setdefault("ft_rate_off", 0.28)
        team.setdefault("ft_rate_def", 0.22)
        team.setdefault("rim_rate_off", 0.60)
        team.setdefault("rim_rate_def", 0.05)
        team.setdefault("three_rate_off", 0.35)
        team.setdefault("three_rate_def", 0.30)
        team.setdefault("bench_depth_score", 0.50)
        team.setdefault("lead_guard_continuity", 0.50)
        team.setdefault("experience_score", 0.50)
        team.setdefault("injury_uncertainty", 0.0)
        team.setdefault("lineup_uncertainty", team["injury_uncertainty"])
        team.setdefault("travel_miles_round1", 0.0)
        team.setdefault("timezone_shift_round1", 0.0)
        team.setdefault("altitude_adjustment", 0.0)
        team.setdefault("venue_familiarity", 0.0)
        team["region_strength_index"] = round(region_mean_rating[team["region"]] - field_mean_rating, 4)
        expected_rating_rank = ((team["seed"] - 1) * 4) + 2.5
        team["seed_misprice"] = round((expected_rating_rank - rating_order[team["team_id"]]) / 16.0, 4)
        team["late_season_volatility"] = round(
            min(
                0.45,
                max(
                    0.05,
                    float(team.get("late_season_volatility", team.get("volatility", 0.12)))
                    + (0.20 * float(team.get("lineup_uncertainty", 0.0))),
                ),
            ),
            4,
        )

    # Build the SelectionSundayDataset
    dataset = {
        "season": 2026,
        "tournament": "mens_d1",
        "teams": teams_list,
        "metadata": {
            "source": "assembled_features_v10",
            "feature_families": ",".join(sources_used),
            "note": "Assembled from V10 feature pipeline. Features sourced from collectors with bracket reference fallbacks.",
        },
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[assemble] Assembled snapshot with {len(teams_by_id)} teams")
    print(f"[assemble] Feature families: {sources_used}")
    print(f"[assemble] Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

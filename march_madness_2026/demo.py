from __future__ import annotations

from random import Random

from .models import SelectionSundayDataset, TeamSnapshot


REGIONS = ["East", "West", "South", "Midwest"]
MASCOTS = [
    "Titans",
    "Rangers",
    "Hawks",
    "Bears",
    "Wolves",
    "Storm",
    "Falcons",
    "Knights",
    "Comets",
    "Dragons",
    "Panthers",
    "Raiders",
    "Foxes",
    "Pilots",
    "Lions",
    "Grit",
]


def build_demo_dataset(seed: int = 20260317) -> SelectionSundayDataset:
    rng = Random(seed)
    teams = []

    for region_index, region in enumerate(REGIONS):
        region_bias = (region_index - 1.5) * 0.6
        for team_seed in range(1, 17):
            base_rating = 100.0 - (team_seed * 2.45) + region_bias + rng.uniform(-1.35, 1.35)
            team_name = f"{region} {MASCOTS[team_seed - 1]}"
            teams.append(
                TeamSnapshot(
                    team_id=f"{region.lower()}-{team_seed:02d}",
                    team_name=team_name,
                    region=region,
                    seed=team_seed,
                    rating=round(base_rating, 2),
                    market_adjustment=round(rng.uniform(-1.2, 1.2), 2),
                    coaching_adjustment=round(rng.uniform(-0.9, 1.1), 2),
                    continuity_adjustment=round(rng.uniform(-0.8, 1.0), 2),
                    injury_penalty=round(max(0.0, rng.gauss(0.3 + (team_seed / 45.0), 0.22)), 2),
                    volatility=min(0.42, round(max(0.04, rng.gauss(0.11 + team_seed / 95.0, 0.04)), 2)),
                    offense_rating=round(rng.uniform(-1.5, 1.8), 2),
                    defense_rating=round(rng.uniform(-1.5, 1.8), 2),
                    tempo_adjustment=round(rng.uniform(-0.7, 0.7), 2),
                )
            )

    return SelectionSundayDataset(
        season=2026,
        tournament="mens_d1_demo",
        teams=teams,
        metadata={
            "source": "demo_synthetic_selection_sunday",
            "note": "Synthetic bracket used to verify the portfolio engine end-to-end.",
        },
    )

#!/usr/bin/env python3
"""
team_strength.py — Build V10 team strength and matchup features.

Priority order:
1. ESPN season stats already captured on disk
2. BartTorvik raw ratings if available
3. Bracket reference fallback

Output: data/features/selection_sunday/team_strength.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_RATINGS = PROJECT_ROOT / "data" / "raw" / "ratings" / "barttorvik_2026.json"
ESPN_STATS = PROJECT_ROOT / "data" / "raw" / "ncaa" / "team_season_stats_2026.json"
BRACKET_REF = PROJECT_ROOT / "data" / "reference" / "bracket_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "team_strength.json"


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    stddev = math.sqrt(variance) if variance > 0 else 1.0
    return [(value - mean) / stddev for value in values]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _compute_raw_metrics(stats: dict) -> dict[str, float]:
    fga = float(stats.get("offensive_avgFieldGoalsAttempted", 60.0))
    fgm = float(stats.get("offensive_avgFieldGoalsMade", 25.0))
    three_att = float(stats.get("offensive_avgThreePointFieldGoalsAttempted", 22.0))
    three_made = float(stats.get("offensive_avgThreePointFieldGoalsMade", 7.0))
    two_att = float(stats.get("offensive_avgTwoPointFieldGoalsAttempted", max(fga - three_att, 1.0)))
    fta = float(stats.get("offensive_avgFreeThrowsAttempted", 17.0))
    off_reb = float(stats.get("offensive_avgOffensiveRebounds", 10.0))
    def_reb = float(stats.get("defensive_avgDefensiveRebounds", 24.0))
    turnovers = float(stats.get("offensive_avgTurnovers", 12.0))
    assists = float(stats.get("offensive_avgAssists", 14.0))
    blocks = float(stats.get("defensive_avgBlocks", 3.5))
    steals = float(stats.get("defensive_avgSteals", 6.5))
    fouls = float(stats.get("general_avgFouls", 16.0))
    points = float(stats.get("offensive_avgPoints", 72.0))
    games = float(stats.get("general_gamesPlayed", 32.0))
    ast_to = float(stats.get("general_assistTurnoverRatio", _safe_div(assists, turnovers, 1.0)))

    possessions = fga + (0.44 * fta) + turnovers - off_reb
    total_reb = off_reb + def_reb

    return {
        "fga": fga,
        "fgm": fgm,
        "three_att": three_att,
        "three_made": three_made,
        "two_att": two_att,
        "fta": fta,
        "off_reb": off_reb,
        "def_reb": def_reb,
        "turnovers": turnovers,
        "assists": assists,
        "blocks": blocks,
        "steals": steals,
        "fouls": fouls,
        "points": points,
        "games": games,
        "assists_to": ast_to,
        "possessions": possessions,
        "adj_efg_off": _safe_div(fgm + (0.5 * three_made), fga, 0.5),
        "orb_rate_off": _safe_div(off_reb, total_reb, 0.28),
        "drb_rate_def": _safe_div(def_reb, total_reb, 0.72),
        "turnover_rate_off": _safe_div(turnovers, possessions, 0.16),
        "turnover_rate_def": _safe_div(steals, possessions, 0.08),
        "ft_rate_off": _safe_div(fta, fga, 0.28),
        "ft_rate_def": _safe_div(fouls, possessions, 0.22),
        "rim_rate_off": _safe_div(two_att, fga, 0.60),
        "rim_rate_def": _safe_div(blocks, possessions, 0.05),
        "three_rate_off": _safe_div(three_att, fga, 0.35),
    }


def _fallback_feature(team: dict) -> dict[str, float | str]:
    return {
        "team_id": team["team_id"],
        "team_name": team["team_name"],
        "rating": team["rating"],
        "offense_rating": team["offense_rating"],
        "defense_rating": team["defense_rating"],
        "tempo_adjustment": team.get("tempo_adjustment", 0.0),
        "three_point_rate": team.get("three_point_rate", 0.0),
        "free_throw_rate": team.get("free_throw_rate", 0.0),
        "adj_efg_off": 0.50,
        "adj_efg_def": 0.50,
        "orb_rate_off": 0.28,
        "drb_rate_def": 0.72,
        "turnover_rate_off": 0.16,
        "turnover_rate_def": 0.08,
        "ft_rate_off": 0.28,
        "ft_rate_def": 0.22,
        "rim_rate_off": 0.60,
        "rim_rate_def": 0.05,
        "three_rate_off": 0.35,
        "three_rate_def": 0.30,
        "bench_depth_score": 0.50,
        "lead_guard_continuity": 0.50,
        "experience_score": 0.50,
        "late_season_volatility": team.get("volatility", 0.12),
        "source": "bracket_reference",
    }


def build_from_espn(espn_data: dict, bracket_teams: dict[str, dict]) -> dict[str, dict]:
    espn_by_name = {
        team_entry["team_name"]: team_entry.get("stats", {})
        for team_entry in espn_data.get("teams", [])
    }
    team_ids = list(bracket_teams.keys())
    raw_metrics_by_team: dict[str, dict[str, float]] = {}

    metric_names = [
        "points",
        "possessions",
        "adj_efg_off",
        "orb_rate_off",
        "drb_rate_def",
        "turnover_rate_off",
        "turnover_rate_def",
        "ft_rate_off",
        "ft_rate_def",
        "rim_rate_off",
        "rim_rate_def",
        "three_rate_off",
        "blocks",
        "steals",
        "off_reb",
        "def_reb",
        "assists_to",
        "assists",
    ]
    metric_lists = {name: [] for name in metric_names}

    for team_id in team_ids:
        team = bracket_teams[team_id]
        stats = espn_by_name.get(team["team_name"])
        if stats is None:
            stats = {}
        raw_metrics = _compute_raw_metrics(stats)
        raw_metrics_by_team[team_id] = raw_metrics
        for name in metric_names:
            metric_lists[name].append(float(raw_metrics[name]))

    zscores = {name: _zscore(values) for name, values in metric_lists.items()}
    three_rate_mean = sum(metric_lists["three_rate_off"]) / len(metric_lists["three_rate_off"])
    ft_rate_mean = sum(metric_lists["ft_rate_off"]) / len(metric_lists["ft_rate_off"])

    features: dict[str, dict] = {}
    for index, team_id in enumerate(team_ids):
        team = bracket_teams[team_id]
        stats = espn_by_name.get(team["team_name"])
        if not stats:
            features[team_id] = _fallback_feature(team)
            continue

        raw = raw_metrics_by_team[team_id]
        off_z = (
            0.28 * zscores["points"][index]
            + 0.24 * zscores["adj_efg_off"][index]
            + 0.18 * zscores["assists_to"][index]
            + 0.15 * zscores["orb_rate_off"][index]
            - 0.15 * zscores["turnover_rate_off"][index]
        )
        def_z = (
            0.28 * zscores["blocks"][index]
            + 0.24 * zscores["steals"][index]
            + 0.22 * zscores["drb_rate_def"][index]
            + 0.12 * zscores["turnover_rate_def"][index]
            - 0.14 * zscores["ft_rate_def"][index]
        )
        tempo_z = zscores["possessions"][index]
        overall_z = (0.52 * off_z) + (0.34 * def_z) + (0.14 * tempo_z)

        bench_depth = _clamp(
            0.50 + (0.16 * zscores["off_reb"][index]) + (0.14 * zscores["assists"][index]),
            0.0,
            1.0,
        )
        lead_guard = _clamp(
            0.50 + (0.18 * zscores["assists_to"][index]) + (0.08 * zscores["assists"][index]),
            0.0,
            1.0,
        )
        experience = _clamp(
            0.50 + (0.10 * zscores["drb_rate_def"][index]) + (0.10 * zscores["assists_to"][index]),
            0.0,
            1.0,
        )
        volatility = _clamp(
            float(team.get("volatility", 0.12))
            + abs(raw["three_rate_off"] - three_rate_mean) * 0.35
            + raw["turnover_rate_off"] * 0.18,
            0.05,
            0.45,
        )

        features[team_id] = {
            "team_id": team_id,
            "team_name": team["team_name"],
            "rating": round(team["rating"] + (overall_z * 2.6), 2),
            "offense_rating": round(off_z * 0.95, 2),
            "defense_rating": round(def_z * 0.95, 2),
            "tempo_adjustment": round(tempo_z * 0.22, 2),
            "three_point_rate": round(raw["three_rate_off"] - three_rate_mean, 4),
            "free_throw_rate": round(raw["ft_rate_off"] - ft_rate_mean, 4),
            "adj_efg_off": round(raw["adj_efg_off"], 4),
            "adj_efg_def": round(_clamp(0.49 - (0.025 * def_z), 0.40, 0.57), 4),
            "orb_rate_off": round(raw["orb_rate_off"], 4),
            "drb_rate_def": round(raw["drb_rate_def"], 4),
            "turnover_rate_off": round(raw["turnover_rate_off"], 4),
            "turnover_rate_def": round(raw["turnover_rate_def"], 4),
            "ft_rate_off": round(raw["ft_rate_off"], 4),
            "ft_rate_def": round(raw["ft_rate_def"], 4),
            "rim_rate_off": round(raw["rim_rate_off"], 4),
            "rim_rate_def": round(raw["rim_rate_def"], 4),
            "three_rate_off": round(raw["three_rate_off"], 4),
            "three_rate_def": round(_clamp(0.31 - (0.02 * def_z), 0.20, 0.42), 4),
            "bench_depth_score": round(bench_depth, 4),
            "lead_guard_continuity": round(lead_guard, 4),
            "experience_score": round(experience, 4),
            "late_season_volatility": round(volatility, 4),
            "source": "espn_stats_v10",
        }

    return features


def build_from_barttorvik(barttorvik: dict, bracket_teams: dict[str, dict]) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for team_id, team in bracket_teams.items():
        bt_data = barttorvik.get(team["team_name"], {})
        if bt_data and bt_data.get("barthag") is not None:
            feature = _fallback_feature(team)
            feature.update(
                {
                    "rating": round(55 + (bt_data["barthag"] * 45), 2),
                    "offense_rating": round((bt_data.get("adj_oe", 105) - 105) / 8, 2),
                    "defense_rating": round((100 - bt_data.get("adj_de", 100)) / 8, 2),
                    "source": "barttorvik",
                }
            )
            features[team_id] = feature
        else:
            features[team_id] = _fallback_feature(team)
    return features


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BRACKET_REF.open() as handle:
        bracket = json.load(handle)
    teams = {team["team_id"]: team for team in bracket["teams"]}

    barttorvik = {}
    if RAW_RATINGS.exists():
        with RAW_RATINGS.open() as handle:
            raw = json.load(handle)
        barttorvik = raw.get("teams", {})
        if barttorvik:
            print(f"[team_strength] Loaded BartTorvik data for {len(barttorvik)} teams")

    espn_data = {}
    if ESPN_STATS.exists():
        with ESPN_STATS.open() as handle:
            espn_data = json.load(handle)
        print(f"[team_strength] Loaded ESPN stats for {espn_data.get('fetched', 0)} teams")

    if espn_data and espn_data.get("teams"):
        features = build_from_espn(espn_data, teams)
    elif barttorvik:
        features = build_from_barttorvik(barttorvik, teams)
    else:
        print("[team_strength] No external stats found, using bracket reference fallback")
        features = {team_id: _fallback_feature(team) for team_id, team in teams.items()}

    output = {
        "feature_family": "team_strength",
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
    print(f"[team_strength] Generated features for {len(features)} teams: {source_counts}")
    print(f"[team_strength] Saved to {OUTPUT_PATH}")

    top_teams = sorted(features.values(), key=lambda item: item["rating"], reverse=True)[:10]
    print("\nTop 10 teams by adjusted rating:")
    for team in top_teams:
        print(
            f"  {team['team_name']:20s} rating={team['rating']:6.1f} "
            f"off={team['offense_rating']:+.2f} def={team['defense_rating']:+.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

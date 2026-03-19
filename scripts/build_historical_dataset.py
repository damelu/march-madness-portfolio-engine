#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyreadr
import httpx
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from march_madness_2026.game_model import build_matchup_row  # noqa: E402
from march_madness_2026.historical import save_historical_snapshot_dataset  # noqa: E402
from march_madness_2026.public_field import (  # noqa: E402
    fit_public_round_model,
    predict_public_advancement_rates,
)
from march_madness_2026.v10.provenance import manifest_table_quality  # noqa: E402


DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v10"
DEFAULT_PUBLIC_REF = PROJECT_ROOT / "data" / "reference" / "public_pick_rates.json"
DEFAULT_SEED_HISTORY = PROJECT_ROOT / "data" / "reference" / "seed_matchup_history.json"
DEFAULT_KAGGLE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "historical" / "kaggle_mania"
FIRST_ROUND_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
DEFAULT_KAGGLE_FILE_CANDIDATES = {
    "MTeams.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MTeams.csv",
        "https://raw.githubusercontent.com/rogerfitz/tutorials/3c0d1dd912bb5f01fcc12a5528aa0568453c8142/march_madness_using_futures/data/MTeams.csv",
    ],
    "MNCAATourneySeeds.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MNCAATourneySeeds.csv",
        "https://raw.githubusercontent.com/dusty-turner/ncaa_tournament_2021_beat_navy/main/01_data/MNCAATourneySeeds.csv",
        "https://raw.githubusercontent.com/rogerfitz/tutorials/3c0d1dd912bb5f01fcc12a5528aa0568453c8142/march_madness_using_futures/data/MNCAATourneySeeds.csv",
    ],
    "MNCAATourneyDetailedResults.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MNCAATourneyDetailedResults.csv",
        "https://raw.githubusercontent.com/dusty-turner/ncaa_tournament_2021_beat_navy/main/01_data/MNCAATourneyDetailedResults.csv",
    ],
    "MRegularSeasonDetailedResults.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MRegularSeasonDetailedResults.csv",
        "https://raw.githubusercontent.com/dusty-turner/ncaa_tournament_2021_beat_navy/main/01_data/MRegularSeasonDetailedResults.csv",
    ],
    "MMasseyOrdinals.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MMasseyOrdinals.csv",
        "https://raw.githubusercontent.com/dusty-turner/ncaa_tournament_2021_beat_navy/main/01_data/MMasseyOrdinals.csv",
    ],
    "MTeamConferences.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MTeamConferences.csv",
        "https://raw.githubusercontent.com/dusty-turner/ncaa_tournament_2021_beat_navy/main/01_data/MTeamConferences.csv",
    ],
    "MTeamCoaches.csv": [
        "https://raw.githubusercontent.com/dongkyunk/Kaggle_NCAA/main/data/MTeamCoaches.csv",
    ],
}
DEFAULT_KAGGLE_FILES = {filename: urls[0] for filename, urls in DEFAULT_KAGGLE_FILE_CANDIDATES.items()}
OFFICIAL_BRACKET_URL = "https://www.ncaa.com/brackets/basketball-men/d1/{season}"
NCAA_SCOREBOARD_URL = "https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{season}/{month:02d}/{day:02d}/scoreboard.json"
ESPN_PUBLIC_PICKS_URL = "https://gambit-api.fantasy.espn.com/apis/v1/propositions?challengeId={challenge_id}"
DEFAULT_ESPN_PUBLIC_CHALLENGES = {
    2023: 239,
    2024: 240,
    2025: 257,
}
MRCHMADNESS_RDATA_URL = "https://raw.githubusercontent.com/elishayer/mRchmadness/master/data/{dataset}.RData"
DEFAULT_MRCHMADNESS_PUBLIC_SEASONS = (2021, 2022)
MRCHMADNESS_PUBLIC_SOURCE_TYPE = "real_mrchmadness_public_distribution"
DEFAULT_NCAA_SCOREBOARD_RESULTS_DATES = {
    2024: (
        (3, 19),
        (3, 20),
        (3, 21),
        (3, 22),
        (3, 23),
        (3, 24),
        (3, 28),
        (3, 29),
        (3, 30),
        (3, 31),
        (4, 6),
        (4, 8),
    ),
    2025: (
        (3, 18),
        (3, 19),
        (3, 20),
        (3, 21),
        (3, 22),
        (3, 23),
        (3, 27),
        (3, 28),
        (3, 29),
        (3, 30),
        (4, 5),
        (4, 7),
    ),
}
REGION_CODE_MAP = {
    "EAST": "E",
    "WEST": "W",
    "SOUTH": "S",
    "MIDWEST": "M",
}
OFFICIAL_TEAM_ALIASES = {
    "fau": "fl atlantic",
    "uconn": "connecticut",
    "saint marys": "st marys ca",
    "saint marys ca": "st marys ca",
    "south dakota st": "s dakota st",
    "western ky": "w kentucky",
    "mount st marys": "mt st marys",
    "saint peters": "st peters",
    "saint francis u": "st francis pa",
    "saint francis": "st francis pa",
    "ole miss": "mississippi",
    "florida atlantic": "fl atlantic",
    "grambling st": "grambling",
    "south carolina": "south carolina",
    "washington st": "washington st",
    "northwestern": "northwestern",
    "american university": "american univ",
    "american": "american univ",
    "mcneese state university": "mcneese st",
    "mcneese": "mcneese st",
    "st johns university new york": "st johns",
    "st johns ny": "st johns",
}


def _is_large_file_pointer(payload: bytes) -> bool:
    preview = payload[:256].decode("utf-8", errors="ignore").splitlines()
    if not preview:
        return False
    return (
        preview[0].strip() == "version https://git-lfs.github.com/spec/v1"
        or preview[0].strip().startswith("version https://huggingface.co/")
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_input_paths(explicit_paths: list[Path], *, exclude_roots: list[Path] | None = None) -> list[Path]:
    if explicit_paths:
        return explicit_paths
    discovered: list[Path] = []
    excluded = [path.resolve() for path in (exclude_roots or [])]
    for directory in (
        PROJECT_ROOT / "data" / "models" / "historical",
        PROJECT_ROOT / "data" / "features" / "historical",
        PROJECT_ROOT / "data" / "raw" / "historical",
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("**/*")):
            if path.is_file() and path.suffix.lower() in {".parquet", ".csv", ".json", ".jsonl"}:
                resolved = path.resolve()
                if any(root == resolved or root in resolved.parents for root in excluded):
                    continue
                discovered.append(path)
    return discovered


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "games", "historical_rows", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    if suffix == ".json":
        return _rows_from_payload(_load_json(path))
    return []


def _normalize_team_name(value: str) -> str:
    text = value.lower().strip()
    replacements = {
        "st.": "st",
        "state.": "state",
        "&": "and",
        "uconn": "connecticut",
        "fau": "fl atlantic",
        "saint": "st",
        "mount": "mt",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else " ")
    normalized = " ".join("".join(cleaned).split())
    return OFFICIAL_TEAM_ALIASES.get(normalized, normalized)


def _build_team_id_lookup(teams_frame: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for row in teams_frame.to_dict(orient="records"):
        normalized = _normalize_team_name(str(row.get("TeamName", "")))
        if normalized:
            lookup[normalized] = int(row["TeamID"])
    return lookup


def _resolve_official_team_id(team_name: str, team_lookup: dict[str, int]) -> int | None:
    normalized = _normalize_team_name(team_name)
    if normalized in team_lookup:
        return team_lookup[normalized]

    import difflib

    matches = difflib.get_close_matches(normalized, list(team_lookup), n=1, cutoff=0.86)
    if matches:
        return team_lookup[matches[0]]
    return None


def _resolve_official_team_id_candidates(candidates: Iterable[str], team_lookup: dict[str, int]) -> int | None:
    for candidate in candidates:
        if not str(candidate).strip():
            continue
        resolved = _resolve_official_team_id(str(candidate), team_lookup)
        if resolved is not None:
            return resolved
    return None


def _extract_pod_teams(pod: Any) -> list[dict[str, Any]]:
    teams: list[dict[str, Any]] = []
    for team_node in pod.select(".team"):
        seed_node = team_node.select_one(".seed")
        name_node = team_node.select_one(".name")
        score_node = team_node.select_one(".score")
        if name_node is None or seed_node is None or score_node is None:
            continue
        teams.append(
            {
                "seed": _safe_int(seed_node.get_text(strip=True), 0),
                "name": name_node.get_text(" ", strip=True),
                "score": _safe_int(score_node.get_text(strip=True), 0),
                "winner": "winner" in (team_node.get("class") or []),
            }
        )
    return teams


def _official_day_num(round_number: int) -> int:
    mapping = {
        0: 134,
        1: 136,
        2: 138,
        3: 144,
        4: 145,
        5: 146,
        6: 147,
    }
    return mapping[round_number]


def _fetch_official_bracket_html(season: int) -> str:
    response = httpx.get(
        OFFICIAL_BRACKET_URL.format(season=season),
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.text


def _official_tournament_frames_from_html(
    season: int,
    html: str,
    teams_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    team_lookup = _build_team_id_lookup(teams_frame)
    seed_entries: list[dict[str, Any]] = []
    result_entries: list[dict[str, Any]] = []
    seed_slot_counts: dict[tuple[str, int], int] = defaultdict(int)
    first_round_winner_seed: dict[str, tuple[str, int]] = {}

    for region in soup.select(".region.region-left, .region.region-right"):
        text = region.get_text(" ", strip=True)
        region_name = text.split()[0].upper() if text else ""
        region_code = REGION_CODE_MAP.get(region_name)
        if region_code is None:
            continue
        for round_number in (1, 2, 3, 4):
            round_nodes = region.select(f".round-{round_number}.region-round .game-pod.final")
            for pod in round_nodes:
                teams = _extract_pod_teams(pod)
                if len(teams) != 2:
                    continue
                resolved_ids = [_resolve_official_team_id(team["name"], team_lookup) for team in teams]
                if any(team_id is None for team_id in resolved_ids):
                    continue
                winner_index = 0 if teams[0]["winner"] or teams[0]["score"] >= teams[1]["score"] else 1
                loser_index = 1 - winner_index
                result_entries.append(
                    {
                        "Season": season,
                        "DayNum": _official_day_num(round_number),
                        "WTeamID": int(resolved_ids[winner_index]),
                        "LTeamID": int(resolved_ids[loser_index]),
                        "WScore": int(teams[winner_index]["score"]),
                        "LScore": int(teams[loser_index]["score"]),
                    }
                )
                if round_number == 1:
                    for team, team_id in zip(teams, resolved_ids):
                        slot = (region_code, int(team["seed"]))
                        seed_slot_counts[slot] += 1
                        suffix = "" if seed_slot_counts[slot] == 1 else chr(ord("a") + seed_slot_counts[slot] - 2)
                        seed_entries.append(
                            {
                                "Season": season,
                                "Seed": f"{region_code}{int(team['seed']):02d}{suffix}",
                                "TeamID": int(team_id),
                            }
                        )
                    winning_team = teams[winner_index]
                    first_round_winner_seed[_normalize_team_name(winning_team["name"])] = (region_code, int(winning_team["seed"]))

    final_games = []
    for pod in soup.select(".center-final-games .game-pod.final"):
        teams = _extract_pod_teams(pod)
        if len(teams) == 2:
            final_games.append(teams)
    if len(final_games) == 3:
        for index, teams in enumerate(final_games):
            resolved_ids = [_resolve_official_team_id(team["name"], team_lookup) for team in teams]
            if any(team_id is None for team_id in resolved_ids):
                continue
            winner_index = 0 if teams[0]["winner"] or teams[0]["score"] >= teams[1]["score"] else 1
            loser_index = 1 - winner_index
            result_entries.append(
                {
                    "Season": season,
                    "DayNum": _official_day_num(5 if index < 2 else 6),
                    "WTeamID": int(resolved_ids[winner_index]),
                    "LTeamID": int(resolved_ids[loser_index]),
                    "WScore": int(teams[winner_index]["score"]),
                    "LScore": int(teams[loser_index]["score"]),
                }
            )

    for pod in soup.select(".game.game-1, .game.game-2, .game.game-3, .game.game-4"):
        teams = _extract_pod_teams(pod)
        if len(teams) != 2:
            continue
        resolved_ids = [_resolve_official_team_id(team["name"], team_lookup) for team in teams]
        if any(team_id is None for team_id in resolved_ids):
            continue
        winner_index = 0 if teams[0]["winner"] or teams[0]["score"] >= teams[1]["score"] else 1
        loser_index = 1 - winner_index
        winner_key = _normalize_team_name(teams[winner_index]["name"])
        region_code, seed_number = first_round_winner_seed.get(winner_key, ("P", int(teams[winner_index]["seed"])))
        slot = (region_code, seed_number)
        start_count = seed_slot_counts[slot]
        for offset, team_id in enumerate((resolved_ids[winner_index], resolved_ids[loser_index]), start=1):
            seed_entries.append(
                {
                    "Season": season,
                    "Seed": f"{region_code}{seed_number:02d}{chr(ord('a') + start_count + offset - 1)}",
                    "TeamID": int(team_id),
                }
            )
        result_entries.append(
            {
                "Season": season,
                "DayNum": _official_day_num(0),
                "WTeamID": int(resolved_ids[winner_index]),
                "LTeamID": int(resolved_ids[loser_index]),
                "WScore": int(teams[winner_index]["score"]),
                "LScore": int(teams[loser_index]["score"]),
            }
        )

    seeds_frame = pd.DataFrame(seed_entries).drop_duplicates(subset=["Season", "TeamID"], keep="first")
    results_frame = pd.DataFrame(result_entries).drop_duplicates(
        subset=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
    )
    return seeds_frame, results_frame


def _official_tournament_supplemental_frames(raw_dir: Path, seasons: Iterable[int]) -> dict[str, pd.DataFrame]:
    teams_path = raw_dir / "MTeams.csv"
    if not teams_path.exists():
        return {}
    teams_frame = pd.read_csv(teams_path)
    all_seed_rows: list[pd.DataFrame] = []
    all_result_rows: list[pd.DataFrame] = []
    for season in sorted(set(int(value) for value in seasons)):
        try:
            html = _fetch_official_bracket_html(season)
            seeds_frame, results_frame = _official_tournament_frames_from_html(season, html, teams_frame)
        except Exception:
            continue
        if not seeds_frame.empty:
            all_seed_rows.append(seeds_frame)
        if not results_frame.empty:
            all_result_rows.append(results_frame)
    supplemental: dict[str, pd.DataFrame] = {}
    if all_seed_rows:
        supplemental["MNCAATourneySeeds.csv"] = pd.concat(all_seed_rows, ignore_index=True)
    if all_result_rows:
        supplemental["MNCAATourneyDetailedResults.csv"] = pd.concat(all_result_rows, ignore_index=True)
    return supplemental


def _scoreboard_round_number(value: str) -> int | None:
    mapping = {
        "first four": 1,
        "first round": 1,
        "second round": 2,
        "sweet 16": 3,
        "elite eight": 4,
        "final four": 5,
        "championship": 6,
    }
    normalized = (
        str(value or "")
        .replace("&#174;", "")
        .replace("®", "")
        .strip()
        .lower()
    )
    normalized = " ".join(normalized.split())
    return mapping.get(normalized)


def _fetch_ncaa_scoreboard_game_rows(
    *,
    seasons: Iterable[int],
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    teams_path = raw_dir / "MTeams.csv"
    if not teams_path.exists():
        return [], []
    teams_frame = pd.read_csv(teams_path)
    team_lookup = _build_team_id_lookup(teams_frame)

    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    seen_keys: set[tuple[int, str]] = set()
    for season in sorted(set(int(value) for value in seasons)):
        for month, day in DEFAULT_NCAA_SCOREBOARD_RESULTS_DATES.get(season, ()):
            url = NCAA_SCOREBOARD_URL.format(season=season, month=month, day=day)
            response = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                follow_redirects=True,
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
            sources.append(url)
            for wrapper in payload.get("games", []):
                game = wrapper.get("game", wrapper)
                bracket_id = str(game.get("bracketId") or "").strip()
                if not bracket_id:
                    continue
                round_number = _scoreboard_round_number(str(game.get("bracketRound", "")))
                if round_number is None:
                    continue
                home = game.get("home") or {}
                away = game.get("away") or {}
                home_names = [
                    home.get("names", {}).get("full") or "",
                    home.get("names", {}).get("short") or "",
                    str(home.get("names", {}).get("seo") or "").replace("-", " "),
                ]
                away_names = [
                    away.get("names", {}).get("full") or "",
                    away.get("names", {}).get("short") or "",
                    str(away.get("names", {}).get("seo") or "").replace("-", " "),
                ]
                home_name = next((str(value) for value in home_names if str(value).strip()), "")
                away_name = next((str(value) for value in away_names if str(value).strip()), "")
                home_id = _resolve_official_team_id_candidates(home_names, team_lookup)
                away_id = _resolve_official_team_id_candidates(away_names, team_lookup)
                if home_id is None or away_id is None:
                    continue
                home_score = _safe_int(home.get("score"), 0)
                away_score = _safe_int(away.get("score"), 0)
                if home_score <= 0 and away_score <= 0:
                    continue
                winner_id = home_id if bool(home.get("winner")) or home_score >= away_score else away_id
                loser_id = away_id if winner_id == home_id else home_id
                ordered = sorted(
                    (
                        {
                            "team_id": str(home_id),
                            "team_name": str(home_name),
                            "seed": _safe_int(home.get("seed"), 0),
                            "winner": winner_id == home_id,
                        },
                        {
                            "team_id": str(away_id),
                            "team_name": str(away_name),
                            "seed": _safe_int(away.get("seed"), 0),
                            "winner": winner_id == away_id,
                        },
                    ),
                    key=lambda item: item["team_id"],
                )
                row = {
                    "season": season,
                    "round_number": round_number,
                    "team_a_id": ordered[0]["team_id"],
                    "team_b_id": ordered[1]["team_id"],
                    "team_a_name": ordered[0]["team_name"],
                    "team_b_name": ordered[1]["team_name"],
                    "team_a_seed": ordered[0]["seed"],
                    "team_b_seed": ordered[1]["seed"],
                    "team_a_win": 1 if ordered[0]["winner"] else 0,
                    "winner_team_id": str(winner_id),
                    "loser_team_id": str(loser_id),
                    "sample_weight": 1.0,
                    "source_type": "real_ncaa_scoreboard_tournament",
                    "source_url": url,
                    "game_id": str(game.get("gameID") or ""),
                    "bracket_round": str(game.get("bracketRound") or ""),
                    "bracket_region": str(game.get("bracketRegion") or ""),
                    "team_a_score": home_score if ordered[0]["team_id"] == str(home_id) else away_score,
                    "team_b_score": away_score if ordered[1]["team_id"] == str(away_id) else home_score,
                }
                row_key = (season, str(row["game_id"]))
                if row["game_id"] and row_key in seen_keys:
                    continue
                if row["game_id"]:
                    seen_keys.add(row_key)
                rows.append(row)
    return rows, sorted(set(sources))


def _espn_round_name_from_outcome_count(outcome_count: int) -> str | None:
    mapping = {
        2: "round_of_32",
        4: "sweet_16",
        8: "elite_8",
        16: "final_four",
        32: "championship_game",
        64: "champion",
    }
    return mapping.get(int(outcome_count))


def _espn_mapping_value(mappings: list[dict[str, Any]], key: str) -> str | None:
    wanted = key.strip().upper()
    for item in mappings:
        if str(item.get("type", "")).upper() == wanted:
            value = item.get("value")
            if value is not None:
                return str(value)
    return None


def _fetch_espn_public_picks_rows(
    season_to_challenge: Mapping[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for season, challenge_id in sorted((int(season), int(challenge)) for season, challenge in season_to_challenge.items()):
        url = ESPN_PUBLIC_PICKS_URL.format(challenge_id=challenge_id)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            continue
        sources.append(url)
        for proposition in payload:
            possible_outcomes = proposition.get("possibleOutcomes") or []
            round_name = _espn_round_name_from_outcome_count(len(possible_outcomes))
            if round_name is None:
                continue
            snapshot_ts = proposition.get("date") or proposition.get("availableDate")
            for outcome in possible_outcomes:
                counters = outcome.get("choiceCounters") or []
                if not counters:
                    continue
                counter = counters[0]
                mappings = outcome.get("mappings") or []
                team_id = (
                    _espn_mapping_value(mappings, "COMPETITOR_ID")
                    or outcome.get("regionCompetitorId")
                    or outcome.get("id")
                )
                if team_id is None:
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_id": str(team_id),
                        "team_name": outcome.get("description") or outcome.get("name") or str(team_id),
                        "seed": _safe_int(
                            _espn_mapping_value(mappings, "SEED")
                            or outcome.get("regionSeed"),
                            0,
                        ),
                        "round_name": round_name,
                        "public_pick_pct": _safe_float(counter.get("percentage"), 0.0),
                        "public_adv_rate": _safe_float(counter.get("percentage"), 0.0),
                        "raw_pick_count": _safe_int(counter.get("count"), 0),
                        "source_type": "real_espn_public_api",
                        "source_url": url,
                        "snapshot_ts": snapshot_ts,
                        "challenge_id": challenge_id,
                    }
                )
    return rows, sources


def _mrchmadness_cache_dir() -> Path:
    path = PROJECT_ROOT / "data" / "raw" / "historical" / "mrchmadness"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_mrchmadness_frame(dataset_name: str) -> pd.DataFrame:
    cache_path = _mrchmadness_cache_dir() / f"{dataset_name}.RData"
    if not cache_path.exists():
        urllib.request.urlretrieve(MRCHMADNESS_RDATA_URL.format(dataset=dataset_name), cache_path)
    payload = pyreadr.read_r(cache_path)
    frame = payload.get(dataset_name)
    if frame is None:
        raise ValueError(f"mRchmadness dataset not found in RData: {dataset_name}")
    return frame


def _resolve_snapshot_team(team_name: str, teams_by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    normalized = _normalize_team_name(team_name)
    if normalized in teams_by_name:
        return teams_by_name[normalized]

    import difflib

    matches = difflib.get_close_matches(normalized, list(teams_by_name), n=1, cutoff=0.86)
    if matches:
        return teams_by_name[matches[0]]
    return None


def _fetch_mrchmadness_public_rows(
    seasons: list[int],
    snapshots: Mapping[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    round_columns = {
        "round1": ("round_of_32", "Round of 64"),
        "round2": ("sweet_16", "Round of 32"),
        "round3": ("elite_8", "Sweet 16"),
        "round4": ("final_four", "Elite 8"),
        "round5": ("championship_game", "Final Four"),
        "round6": ("champion", "National Championship"),
    }
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for season in seasons:
        snapshot = snapshots.get(season)
        if not snapshot:
            continue
        dataset_name = f"pred.pop.men.{season}"
        try:
            frame = _load_mrchmadness_frame(dataset_name)
        except Exception:
            continue

        teams_by_name = {
            _normalize_team_name(str(team.get("team_name", ""))): team
            for team in snapshot.get("teams", [])
            if _normalize_team_name(str(team.get("team_name", "")))
        }
        if not teams_by_name:
            continue

        source_url = MRCHMADNESS_RDATA_URL.format(dataset=dataset_name)
        sources.append(source_url)
        for record in frame.to_dict(orient="records"):
            team = _resolve_snapshot_team(str(record.get("name", "")), teams_by_name)
            if team is None:
                continue
            team_id = str(team["team_id"])
            seed = _safe_int(team.get("seed"), 0)
            team_name = str(team.get("team_name", record.get("name", "")))
            for column, (round_name, source_round_name) in round_columns.items():
                rate = _safe_float(record.get(column), 0.0)
                rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "team_name": team_name,
                        "seed": seed,
                        "round_name": round_name,
                        "public_pick_pct": rate,
                        "public_adv_rate": rate,
                        "source_type": MRCHMADNESS_PUBLIC_SOURCE_TYPE,
                        "source_url": source_url,
                        "source_round_name": source_round_name,
                    }
                )
    return rows, sources


def download_kaggle_mirror_bundle(
    output_dir: Path,
    *,
    force: bool = False,
    file_map: dict[str, str] | dict[str, list[str]] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    candidate_map = file_map or DEFAULT_KAGGLE_FILE_CANDIDATES
    for filename, url_value in candidate_map.items():
        urls = [url_value] if isinstance(url_value, str) else list(url_value)
        destination = output_dir / filename
        should_download = force or not destination.exists()
        if not should_download:
            try:
                should_download = _is_large_file_pointer(destination.read_bytes())
            except OSError:
                should_download = True
        if should_download:
            payload: bytes | None = None
            errors: list[str] = []
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        candidate_payload = response.read()
                except Exception as exc:  # pragma: no cover - network variability
                    errors.append(f"{url} ({exc})")
                    continue
                if _is_large_file_pointer(candidate_payload):
                    errors.append(f"{url} (pointer content)")
                    continue
                payload = candidate_payload
                break
            if payload is None:
                joined = "; ".join(errors) if errors else "no candidate URLs"
                raise RuntimeError(
                    f"unable to download a usable historical CSV for {filename}: {joined}"
                )
            with destination.open("wb") as handle:
                handle.write(payload)
        resolved[filename] = destination
    return resolved


def _seed_number(seed_code: Any) -> int:
    digits = "".join(ch for ch in str(seed_code) if ch.isdigit())
    return _safe_int(digits, 0)


def _seed_region(seed_code: Any) -> str:
    token = str(seed_code).strip()
    return token[:1] if token else "UNK"


def _round_number_from_day(day_num: int) -> int:
    if day_num <= 137:
        return 1
    if day_num <= 139:
        return 2
    if day_num <= 145:
        return 3
    if day_num <= 146:
        return 5
    return 6


def _normalize_series(value: pd.Series) -> pd.Series:
    std = float(value.std(ddof=0))
    if std <= 1e-9:
        return pd.Series(np.zeros(len(value)), index=value.index, dtype=np.float64)
    return (value - float(value.mean())) / std


def _build_empirical_team_features(
    bundle_rows: dict[str, pd.DataFrame],
    *,
    seasons: list[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    teams = bundle_rows["MTeams.csv"].copy()
    seeds = bundle_rows["MNCAATourneySeeds.csv"].copy()
    regular = bundle_rows["MRegularSeasonDetailedResults.csv"].copy()
    massey = bundle_rows["MMasseyOrdinals.csv"].copy()
    conferences = bundle_rows.get("MTeamConferences.csv", pd.DataFrame())
    coaches = bundle_rows.get("MTeamCoaches.csv", pd.DataFrame())

    seeds = seeds[seeds["Season"].isin(seasons)].copy()
    seeds["SeedNum"] = seeds["Seed"].map(_seed_number)
    seeds["Region"] = seeds["Seed"].map(_seed_region)
    seeds = seeds[(seeds["SeedNum"] >= 1) & (seeds["SeedNum"] <= 16)].copy()

    regular = regular[regular["Season"].isin(seasons)].copy()

    winner_rows = pd.DataFrame(
        {
            "Season": regular["Season"],
            "TeamID": regular["WTeamID"],
            "OppTeamID": regular["LTeamID"],
            "ScoreFor": regular["WScore"],
            "ScoreAgainst": regular["LScore"],
            "FGM": regular["WFGM"],
            "FGA": regular["WFGA"],
            "FGM3": regular["WFGM3"],
            "FGA3": regular["WFGA3"],
            "FTM": regular["WFTM"],
            "FTA": regular["WFTA"],
            "OR": regular["WOR"],
            "DR": regular["WDR"],
            "Ast": regular["WAst"],
            "TO": regular["WTO"],
            "Stl": regular["WStl"],
            "Blk": regular["WBlk"],
            "PF": regular["WPF"],
            "OppFGM": regular["LFGM"],
            "OppFGA": regular["LFGA"],
            "OppFGM3": regular["LFGM3"],
            "OppFGA3": regular["LFGA3"],
            "OppFTM": regular["LFTM"],
            "OppFTA": regular["LFTA"],
            "OppOR": regular["LOR"],
            "OppDR": regular["LDR"],
            "OppAst": regular["LAst"],
            "OppTO": regular["LTO"],
            "OppStl": regular["LStl"],
            "OppBlk": regular["LBlk"],
            "OppPF": regular["LPF"],
            "Margin": regular["WScore"] - regular["LScore"],
        }
    )
    loser_rows = pd.DataFrame(
        {
            "Season": regular["Season"],
            "TeamID": regular["LTeamID"],
            "OppTeamID": regular["WTeamID"],
            "ScoreFor": regular["LScore"],
            "ScoreAgainst": regular["WScore"],
            "FGM": regular["LFGM"],
            "FGA": regular["LFGA"],
            "FGM3": regular["LFGM3"],
            "FGA3": regular["LFGA3"],
            "FTM": regular["LFTM"],
            "FTA": regular["LFTA"],
            "OR": regular["LOR"],
            "DR": regular["LDR"],
            "Ast": regular["LAst"],
            "TO": regular["LTO"],
            "Stl": regular["LStl"],
            "Blk": regular["LBlk"],
            "PF": regular["LPF"],
            "OppFGM": regular["WFGM"],
            "OppFGA": regular["WFGA"],
            "OppFGM3": regular["WFGM3"],
            "OppFGA3": regular["WFGA3"],
            "OppFTM": regular["WFTM"],
            "OppFTA": regular["WFTA"],
            "OppOR": regular["WOR"],
            "OppDR": regular["WDR"],
            "OppAst": regular["WAst"],
            "OppTO": regular["WTO"],
            "OppStl": regular["WStl"],
            "OppBlk": regular["WBlk"],
            "OppPF": regular["WPF"],
            "Margin": regular["LScore"] - regular["WScore"],
        }
    )
    long_regular = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long_regular["Possessions"] = long_regular["FGA"] - long_regular["OR"] + long_regular["TO"] + (0.475 * long_regular["FTA"])
    long_regular["OppPossessions"] = (
        long_regular["OppFGA"] - long_regular["OppOR"] + long_regular["OppTO"] + (0.475 * long_regular["OppFTA"])
    )
    long_regular["Win"] = (long_regular["Margin"] > 0).astype(np.float64)

    grouped = long_regular.groupby(["Season", "TeamID"], as_index=False).agg(
        games=("TeamID", "size"),
        wins=("Win", "sum"),
        score_for=("ScoreFor", "sum"),
        score_against=("ScoreAgainst", "sum"),
        possessions=("Possessions", "sum"),
        opp_possessions=("OppPossessions", "sum"),
        fgm=("FGM", "sum"),
        fga=("FGA", "sum"),
        fgm3=("FGM3", "sum"),
        fga3=("FGA3", "sum"),
        ftm=("FTM", "sum"),
        fta=("FTA", "sum"),
        orb=("OR", "sum"),
        drb=("DR", "sum"),
        tov=("TO", "sum"),
        opp_fgm=("OppFGM", "sum"),
        opp_fga=("OppFGA", "sum"),
        opp_fgm3=("OppFGM3", "sum"),
        opp_fga3=("OppFGA3", "sum"),
        opp_fta=("OppFTA", "sum"),
        opp_orb=("OppOR", "sum"),
        opp_drb=("OppDR", "sum"),
        opp_tov=("OppTO", "sum"),
        margin_mean=("Margin", "mean"),
        margin_std=("Margin", "std"),
    )

    grouped["win_rate"] = grouped["wins"] / grouped["games"].clip(lower=1)
    grouped["offense_rating"] = 100.0 * grouped["score_for"] / grouped["possessions"].clip(lower=1.0)
    grouped["defense_rating"] = 100.0 * grouped["score_against"] / grouped["opp_possessions"].clip(lower=1.0)
    grouped["rating"] = grouped["offense_rating"] - grouped["defense_rating"]
    grouped["tempo_adjustment"] = grouped["possessions"] / grouped["games"].clip(lower=1.0)
    grouped["three_point_rate"] = grouped["fga3"] / grouped["fga"].clip(lower=1.0)
    grouped["free_throw_rate"] = grouped["fta"] / grouped["fga"].clip(lower=1.0)
    grouped["adj_efg_off"] = (grouped["fgm"] + (0.5 * grouped["fgm3"])) / grouped["fga"].clip(lower=1.0)
    grouped["adj_efg_def"] = (grouped["opp_fgm"] + (0.5 * grouped["opp_fgm3"])) / grouped["opp_fga"].clip(lower=1.0)
    grouped["orb_rate_off"] = grouped["orb"] / (grouped["orb"] + grouped["opp_drb"]).clip(lower=1.0)
    grouped["drb_rate_def"] = grouped["drb"] / (grouped["drb"] + grouped["opp_orb"]).clip(lower=1.0)
    grouped["turnover_rate_off"] = grouped["tov"] / grouped["possessions"].clip(lower=1.0)
    grouped["turnover_rate_def"] = grouped["opp_tov"] / grouped["opp_possessions"].clip(lower=1.0)
    grouped["ft_rate_off"] = grouped["fta"] / grouped["fga"].clip(lower=1.0)
    grouped["ft_rate_def"] = grouped["opp_fta"] / grouped["opp_fga"].clip(lower=1.0)
    grouped["rim_rate_off"] = (grouped["fga"] - grouped["fga3"]) / grouped["fga"].clip(lower=1.0)
    grouped["rim_rate_def"] = (grouped["opp_fga"] - grouped["opp_fga3"]) / grouped["opp_fga"].clip(lower=1.0)
    grouped["three_rate_off"] = grouped["fga3"] / grouped["fga"].clip(lower=1.0)
    grouped["three_rate_def"] = grouped["opp_fga3"] / grouped["opp_fga"].clip(lower=1.0)
    grouped["volatility"] = grouped["margin_std"].fillna(0.0).clip(lower=0.0) / 25.0
    grouped["injury_uncertainty"] = 1.0 / np.sqrt(grouped["games"].clip(lower=1.0))
    grouped["lineup_uncertainty"] = grouped["injury_uncertainty"]

    if not massey.empty:
        massey = massey[massey["Season"].isin(seasons)].copy()
        if "RankingDayNum" in massey.columns:
            massey = massey[massey["RankingDayNum"] <= 133].copy()
        massey.sort_values(["Season", "SystemName", "TeamID", "RankingDayNum"], inplace=True)
        massey_latest = massey.groupby(["Season", "SystemName", "TeamID"], as_index=False).tail(1)
        ordinal = massey_latest.groupby(["Season", "TeamID"], as_index=False)["OrdinalRank"].median()
        ordinal["massey_score"] = -ordinal["OrdinalRank"]
        ordinal["massey_score_norm"] = ordinal.groupby("Season")["massey_score"].transform(_normalize_series)
        grouped = grouped.merge(ordinal[["Season", "TeamID", "massey_score_norm"]], on=["Season", "TeamID"], how="left")
    else:
        grouped["massey_score_norm"] = 0.0

    if not conferences.empty:
        conferences = conferences[conferences["Season"].isin(seasons)].copy()
        conferences = conferences.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")
        grouped = grouped.merge(conferences[["Season", "TeamID", "ConfAbbrev"]], on=["Season", "TeamID"], how="left")
    else:
        grouped["ConfAbbrev"] = ""

    coach_years = None
    if not coaches.empty:
        coaches = coaches[coaches["Season"].isin(seasons)].copy()
        coaches = coaches.sort_values(["TeamID", "Season"])
        coaches["CoachTenureYears"] = (
            coaches.groupby("TeamID")["CoachName"]
            .transform(lambda series: (series != series.shift()).cumsum())
        )
        # Stable proxy: seasons with same coach up to current season.
        tenure_lookup: dict[tuple[int, int], float] = {}
        by_team: dict[int, dict[str, int]] = {}
        for row in coaches.itertuples(index=False):
            state = by_team.setdefault(int(row.TeamID), {})
            coach_name = str(row.CoachName)
            years = 1 if state.get("coach_name") != coach_name else int(state.get("years", 0)) + 1
            state["coach_name"] = coach_name
            state["years"] = years
            tenure_lookup[(int(row.Season), int(row.TeamID))] = float(years)
        coach_years = grouped.apply(lambda row: tenure_lookup.get((int(row["Season"]), int(row["TeamID"])), 1.0), axis=1)
    else:
        coach_years = pd.Series(np.ones(len(grouped)), index=grouped.index, dtype=np.float64)

    grouped["experience_score"] = np.clip(0.35 + (0.1 * coach_years) + (0.2 * grouped["win_rate"]), 0.0, 1.0)
    grouped["bench_depth_score"] = np.clip(0.4 + (0.3 * grouped["win_rate"]) - (0.15 * grouped["volatility"]), 0.0, 1.0)
    grouped["lead_guard_continuity"] = np.clip(0.5 + (0.1 * coach_years) - (0.1 * grouped["injury_uncertainty"]), 0.0, 1.0)
    grouped["coaching_adjustment"] = np.clip((coach_years - 2.0) / 10.0, -0.2, 0.5)
    grouped["continuity_adjustment"] = np.clip((grouped["win_rate"] - 0.5) * 0.5, -0.3, 0.3)
    grouped["injury_penalty"] = 0.0
    grouped["market_adjustment"] = grouped["massey_score_norm"].fillna(0.0)

    grouped["rating"] = grouped["rating"] + (5.0 * grouped["massey_score_norm"].fillna(0.0))

    teams = teams.rename(columns={"TeamID": "TeamID", "TeamName": "TeamName"})
    grouped = grouped.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    grouped = grouped.merge(seeds[["Season", "TeamID", "Seed", "SeedNum", "Region"]], on=["Season", "TeamID"], how="inner")
    grouped["rating_rank"] = grouped.groupby("Season")["rating"].rank(method="first", ascending=False)

    season_feature_map: dict[int, dict[str, dict[str, Any]]] = {}
    for season in sorted(set(int(value) for value in grouped["Season"].tolist())):
        season_rows = grouped[grouped["Season"] == season].copy()
        if season_rows.empty:
            continue
        field_mean_rating = float(season_rows["rating"].mean())
        region_mean = season_rows.groupby("Region")["rating"].mean().to_dict()
        season_map: dict[str, dict[str, Any]] = {}
        for row in season_rows.to_dict(orient="records"):
            expected_rank = ((int(row["SeedNum"]) - 1) * 4) + 2.5
            seed_misprice = (expected_rank - float(row["rating_rank"])) / 16.0
            season_map[str(int(row["TeamID"]))] = {
                "team_id": str(int(row["TeamID"])),
                "team_name": row.get("TeamName") or f"Team {int(row['TeamID'])}",
                "season": int(row["Season"]),
                "region": str(row["Region"]),
                "seed": int(row["SeedNum"]),
                "rating": round(float(row["rating"]), 4),
                "market_adjustment": round(float(row["market_adjustment"]), 4),
                "coaching_adjustment": round(float(row["coaching_adjustment"]), 4),
                "continuity_adjustment": round(float(row["continuity_adjustment"]), 4),
                "injury_penalty": 0.0,
                "volatility": float(np.clip(row["volatility"], 0.05, 0.45)),
                "offense_rating": round(float(row["offense_rating"]), 4),
                "defense_rating": round(float(row["defense_rating"]), 4),
                "tempo_adjustment": round(float(row["tempo_adjustment"]), 4),
                "three_point_rate": round(float(row["three_point_rate"]), 4),
                "free_throw_rate": round(float(row["free_throw_rate"]), 4),
                "public_pick_pct": 0.0,
                "adj_efg_off": round(float(row["adj_efg_off"]), 4),
                "adj_efg_def": round(float(row["adj_efg_def"]), 4),
                "orb_rate_off": round(float(row["orb_rate_off"]), 4),
                "drb_rate_def": round(float(row["drb_rate_def"]), 4),
                "turnover_rate_off": round(float(row["turnover_rate_off"]), 4),
                "turnover_rate_def": round(float(row["turnover_rate_def"]), 4),
                "ft_rate_off": round(float(row["ft_rate_off"]), 4),
                "ft_rate_def": round(float(row["ft_rate_def"]), 4),
                "rim_rate_off": round(float(row["rim_rate_off"]), 4),
                "rim_rate_def": round(float(row["rim_rate_def"]), 4),
                "three_rate_off": round(float(row["three_rate_off"]), 4),
                "three_rate_def": round(float(row["three_rate_def"]), 4),
                "bench_depth_score": round(float(row["bench_depth_score"]), 4),
                "lead_guard_continuity": round(float(row["lead_guard_continuity"]), 4),
                "experience_score": round(float(row["experience_score"]), 4),
                "injury_uncertainty": round(float(row["injury_uncertainty"]), 4),
                "lineup_uncertainty": round(float(row["lineup_uncertainty"]), 4),
                "travel_miles_round1": 0.0,
                "timezone_shift_round1": 0,
                "altitude_adjustment": 0.0,
                "venue_familiarity": 0.0,
                "region_strength_index": round(float(region_mean.get(str(row["Region"]), field_mean_rating) - field_mean_rating), 4),
                "seed_misprice": round(float(seed_misprice), 4),
                "metadata_conf_abbrev": str(row.get("ConfAbbrev") or ""),
                "source_type": "real_kaggle_regular_season",
            }
        season_feature_map[season] = season_map
    return season_feature_map


def build_empirical_rows_from_kaggle_bundle(
    raw_dir: Path,
    *,
    seasons: list[int] | None = None,
    public_reference: dict[str, Any] | None = None,
    seed_history: dict[str, Any] | None = None,
    supplemental_frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    frames = {filename: pd.read_csv(raw_dir / filename) for filename in DEFAULT_KAGGLE_FILES}
    for filename, frame in (supplemental_frames or {}).items():
        if filename not in frames or frame.empty:
            continue
        frames[filename] = pd.concat([frames[filename], frame], ignore_index=True)
    seed_frame = frames["MNCAATourneySeeds.csv"].copy()
    available_seasons = sorted({int(value) for value in seed_frame["Season"].tolist() if int(value) != 2020})
    target_seasons = sorted(set(seasons or [season for season in available_seasons if season >= 2013]))
    team_features = _build_empirical_team_features(frames, seasons=target_seasons)

    tournament = frames["MNCAATourneyDetailedResults.csv"].copy()
    tournament = tournament[tournament["Season"].isin(target_seasons)].copy()
    game_rows: list[dict[str, Any]] = []
    for row in tournament.to_dict(orient="records"):
        season = int(row["Season"])
        season_map = team_features.get(season, {})
        left_id = str(min(int(row["WTeamID"]), int(row["LTeamID"])))
        right_id = str(max(int(row["WTeamID"]), int(row["LTeamID"])))
        if left_id not in season_map or right_id not in season_map:
            continue
        team_a = season_map[left_id]
        team_b = season_map[right_id]
        matchup = build_matchup_row(
            team_a,
            team_b,
            season=season,
            round_number=_round_number_from_day(_safe_int(row.get("DayNum"), 134)),
            metadata={
                "source_type": "real_kaggle_tournament",
                "winner_team_id": str(int(row["WTeamID"])),
                "loser_team_id": str(int(row["LTeamID"])),
                "game_day_num": int(row["DayNum"]),
            },
        )
        matchup["team_a_win"] = 1 if str(team_a["team_id"]) == str(int(row["WTeamID"])) else 0
        matchup["sample_weight"] = 1.0
        game_rows.append(matchup)

    snapshots: dict[int, dict[str, Any]] = {}
    public_rows: list[dict[str, Any]] = []
    for season, season_map in team_features.items():
        teams = list(season_map.values())
        snapshots[season] = {
            "season": season,
            "tournament": "mens_d1",
            "teams": teams,
            "metadata": {
                "source": "kaggle_mania_github_mirror",
                "snapshot_kind": "selection_sunday_proxy",
                "note": "Derived from regular season detailed results, tournament seeds, and end-of-regular-season Massey ordinals.",
            },
        }
        public_model = fit_public_round_model(
            teams,
            reference_data=public_reference,
            seed_history=seed_history,
            model_id=f"v10_4_public_proxy_{season}",
        )
        public_rates = predict_public_advancement_rates(public_model, teams, seed_history=seed_history)
        for team in teams:
            team_id = str(team["team_id"])
            for round_name, rate in public_rates.get(team_id, {}).items():
                public_rows.append(
                        {
                            "season": season,
                            "team_id": team_id,
                            "team_name": team["team_name"],
                            "seed": int(team["seed"]),
                            "round_name": round_name,
                            "public_pick_pct": float(rate),
                            "public_adv_rate": float(rate),
                            "source_type": "heuristic_public_from_real_snapshot",
                        }
                    )

    coverage = {
        "available_seasons": available_seasons,
        "target_seasons": target_seasons,
        "game_seasons": sorted({int(row["season"]) for row in game_rows}),
        "public_seasons": sorted({int(row["season"]) for row in public_rows}),
        "snapshot_seasons": sorted(snapshots),
    }
    return game_rows, public_rows, snapshots, coverage


def _coverage_payload(rows: list[dict[str, Any]], *, fallback_source_types: set[str]) -> dict[str, Any]:
    seasons = sorted({_safe_int(row.get("season"), 0) for row in rows if _safe_int(row.get("season"), 0) != 0})
    by_season: dict[str, Any] = {}
    for season in seasons:
        season_rows = [row for row in rows if _safe_int(row.get("season"), 0) == season]
        source_types = sorted({str(row.get("source_type", "unknown")) for row in season_rows})
        synthetic = any(source in fallback_source_types for source in source_types)
        if not synthetic and season_rows:
            mode = "real"
        elif synthetic and any(source not in fallback_source_types for source in source_types):
            mode = "mixed"
        else:
            mode = "synthetic"
        by_season[str(season)] = {
            "mode": mode,
            "row_count": len(season_rows),
            "source_types": source_types,
        }
    synthetic_fallback_used = any(payload["mode"] != "real" for payload in by_season.values())
    return {
        "row_count": len(rows),
        "seasons": seasons,
        "synthetic_fallback_used": synthetic_fallback_used,
        "seasons_detail": by_season,
    }


def _jitter_team(base_team: dict[str, Any], rng: np.random.Generator, season: int) -> dict[str, Any]:
    team = dict(base_team)
    team["season"] = season
    team["rating"] = round(_safe_float(team.get("rating")) + float(rng.normal(0.0, 1.8)), 3)
    team["market_adjustment"] = round(_safe_float(team.get("market_adjustment")) + float(rng.normal(0.0, 0.18)), 3)
    team["offense_rating"] = round(_safe_float(team.get("offense_rating")) + float(rng.normal(0.0, 0.22)), 3)
    team["defense_rating"] = round(_safe_float(team.get("defense_rating")) + float(rng.normal(0.0, 0.22)), 3)
    team["tempo_adjustment"] = round(_safe_float(team.get("tempo_adjustment")) + float(rng.normal(0.0, 0.08)), 3)
    team["three_point_rate"] = round(_safe_float(team.get("three_point_rate")) + float(rng.normal(0.0, 0.05)), 3)
    team["free_throw_rate"] = round(_safe_float(team.get("free_throw_rate")) + float(rng.normal(0.0, 0.04)), 3)
    team["volatility"] = float(np.clip(_safe_float(team.get("volatility"), 0.12) + rng.normal(0.0, 0.02), 0.03, 0.40))
    team["injury_penalty"] = float(np.clip(_safe_float(team.get("injury_penalty"), 0.0) + rng.normal(0.0, 0.10), 0.0, 1.0))
    team["travel_miles_round1"] = float(np.clip(_safe_float(team.get("travel_miles_round1"), 350.0) + rng.normal(0.0, 120.0), 25.0, 2500.0))
    team["timezone_shift_round1"] = int(np.clip(_safe_int(team.get("timezone_shift_round1"), 0) + rng.integers(-1, 2), -3, 3))
    team["altitude_adjustment"] = round(_safe_float(team.get("altitude_adjustment"), 0.0) + float(rng.normal(0.0, 0.08)), 3)
    team["venue_familiarity"] = float(np.clip(_safe_float(team.get("venue_familiarity"), 0.0) + rng.normal(0.0, 0.10), -1.0, 1.0))
    team["region_strength_index"] = round(_safe_float(team.get("region_strength_index"), 0.0) + float(rng.normal(0.0, 0.12)), 3)
    team["seed_misprice"] = round(_safe_float(team.get("seed_misprice"), 0.0) + float(rng.normal(0.0, 0.25)), 3)
    team["injury_uncertainty"] = float(np.clip(_safe_float(team.get("injury_uncertainty"), 0.12) + rng.normal(0.0, 0.04), 0.0, 1.0))
    team["lineup_uncertainty"] = float(np.clip(_safe_float(team.get("lineup_uncertainty"), 0.10) + rng.normal(0.0, 0.04), 0.0, 1.0))
    team["adj_efg_off"] = round(_safe_float(team.get("adj_efg_off"), _safe_float(team.get("offense_rating")) * 0.42) + float(rng.normal(0.0, 0.05)), 3)
    team["adj_efg_def"] = round(_safe_float(team.get("adj_efg_def"), _safe_float(team.get("defense_rating")) * 0.42) + float(rng.normal(0.0, 0.05)), 3)
    team["orb_rate_off"] = float(np.clip(_safe_float(team.get("orb_rate_off"), 0.28) + rng.normal(0.0, 0.02), 0.15, 0.45))
    team["drb_rate_def"] = float(np.clip(_safe_float(team.get("drb_rate_def"), 0.71) + rng.normal(0.0, 0.02), 0.50, 0.85))
    team["turnover_rate_off"] = float(np.clip(_safe_float(team.get("turnover_rate_off"), 0.16) + rng.normal(0.0, 0.015), 0.08, 0.28))
    team["turnover_rate_def"] = float(np.clip(_safe_float(team.get("turnover_rate_def"), 0.18) + rng.normal(0.0, 0.015), 0.08, 0.30))
    team["rim_rate_off"] = float(np.clip(_safe_float(team.get("rim_rate_off"), 0.34) + rng.normal(0.0, 0.025), 0.18, 0.55))
    team["rim_rate_def"] = float(np.clip(_safe_float(team.get("rim_rate_def"), 0.31) + rng.normal(0.0, 0.025), 0.15, 0.50))
    team["ft_rate_off"] = float(np.clip(_safe_float(team.get("ft_rate_off"), 0.31) + rng.normal(0.0, 0.02), 0.15, 0.50))
    team["ft_rate_def"] = float(np.clip(_safe_float(team.get("ft_rate_def"), 0.26) + rng.normal(0.0, 0.02), 0.10, 0.45))
    team["three_rate_off"] = float(np.clip(_safe_float(team.get("three_rate_off"), 0.38) + rng.normal(0.0, 0.03), 0.20, 0.55))
    team["three_rate_def"] = float(np.clip(_safe_float(team.get("three_rate_def"), 0.35) + rng.normal(0.0, 0.03), 0.18, 0.55))
    return team


def _seed_prior(seed_history: dict[str, Any], seed_a: int, seed_b: int) -> float:
    lookup = f"{min(seed_a, seed_b)}_vs_{max(seed_a, seed_b)}"
    round_history = seed_history.get("round_of_64", {})
    if lookup not in round_history:
        return 0.5
    rate = _safe_float(round_history[lookup].get("higher_seed_wins"), 0.5)
    return rate if seed_a < seed_b else 1.0 - rate


def _latent_matchup_probability(team_a: dict[str, Any], team_b: dict[str, Any], seed_history: dict[str, Any], round_number: int) -> float:
    rating_gap = _safe_float(team_a.get("rating")) - _safe_float(team_b.get("rating"))
    offense_gap = _safe_float(team_a.get("offense_rating")) - _safe_float(team_b.get("offense_rating"))
    defense_gap = _safe_float(team_b.get("defense_rating")) - _safe_float(team_a.get("defense_rating"))
    market_gap = _safe_float(team_a.get("market_adjustment")) - _safe_float(team_b.get("market_adjustment"))
    injury_gap = _safe_float(team_a.get("injury_penalty")) - _safe_float(team_b.get("injury_penalty"))
    volatility_gap = _safe_float(team_a.get("volatility")) - _safe_float(team_b.get("volatility"))
    public_gap = _safe_float(team_a.get("public_pick_pct")) - _safe_float(team_b.get("public_pick_pct"))
    margin = (
        rating_gap
        + 0.35 * offense_gap
        + 0.25 * defense_gap
        + 0.20 * market_gap
        - 0.95 * injury_gap
        - 0.18 * volatility_gap
        + 0.03 * public_gap
    )
    probability = _sigmoid(margin / 7.2)
    if round_number == 1:
        probability = 0.65 * probability + 0.35 * _seed_prior(seed_history, _safe_int(team_a.get("seed")), _safe_int(team_b.get("seed")))
    return float(np.clip(probability, 0.01, 0.99))


def _generate_synthetic_rows(
    snapshot: dict[str, Any],
    public_reference: dict[str, Any],
    seed_history: dict[str, Any],
    seasons: list[int],
    rows_per_season: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    teams = [dict(team) for team in snapshot.get("teams", [])]
    game_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []

    champion_picks = public_reference.get("champion_picks", {})
    for team in teams:
        team_id = str(team.get("team_id", ""))
        if champion_picks.get(team_id) is not None:
            team["public_pick_pct"] = champion_picks[team_id]

    for season in seasons:
        season_teams = [_jitter_team(team, rng, season) for team in teams]
        by_region_seed = {
            (str(team["region"]), _safe_int(team["seed"])): team
            for team in season_teams
        }

        for region in sorted({str(team["region"]) for team in season_teams}):
            for seed_a, seed_b in FIRST_ROUND_PAIRS:
                team_a = by_region_seed[(region, seed_a)]
                team_b = by_region_seed[(region, seed_b)]
                row = build_matchup_row(team_a, team_b, season=season, round_number=1, metadata={"source_type": "synthetic_round1"})
                probability = _latent_matchup_probability(team_a, team_b, seed_history, 1)
                row["team_a_win"] = int(rng.random() <= probability)
                row["sample_weight"] = 1.15
                game_rows.append(row)

        all_pairs = [(season_teams[i], season_teams[j]) for i in range(len(season_teams)) for j in range(i + 1, len(season_teams))]
        rng.shuffle(all_pairs)
        for team_a, team_b in all_pairs[:rows_per_season]:
            round_number = int(rng.integers(2, 7))
            row = build_matchup_row(team_a, team_b, season=season, round_number=round_number, metadata={"source_type": "synthetic_pair"})
            probability = _latent_matchup_probability(team_a, team_b, seed_history, round_number)
            row["team_a_win"] = int(rng.random() <= probability)
            row["sample_weight"] = 1.0 + 0.05 * (round_number - 1)
            game_rows.append(row)

        public_model = fit_public_round_model(
            season_teams,
            reference_data=public_reference,
            seed_history=seed_history,
            model_id=f"synthetic_public_{season}",
        )
        public_rates = predict_public_advancement_rates(public_model, season_teams, seed_history=seed_history)
        for team in season_teams:
            team_id = str(team.get("team_id"))
            for round_name, rate in public_rates.get(team_id, {}).items():
                public_rows.append(
                    {
                        "season": season,
                        "team_id": team_id,
                        "team_name": team.get("team_name"),
                        "seed": _safe_int(team.get("seed")),
                        "round_name": round_name,
                        "public_pick_pct": _safe_float(team.get("public_pick_pct"), 0.0),
                        "public_adv_rate": rate,
                        "source_type": "synthetic_public",
                    }
                )

    return game_rows, public_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or normalize the V10 historical training dataset.")
    parser.add_argument("--input-path", action="append", type=Path, default=[], help="Optional CSV/JSON/JSONL/Parquet historical row source.")
    parser.add_argument(
        "--public-input-path",
        action="append",
        type=Path,
        default=[],
        help="Optional CSV/JSON/JSONL/Parquet historical public-pick row source.",
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT, help="Selection Sunday snapshot used for synthetic fallback.")
    parser.add_argument("--download-kaggle-mirror", action="store_true", help="Download a public Kaggle mirror bundle before building.")
    parser.add_argument(
        "--download-official-brackets",
        action="store_true",
        help="Supplement missing tournament seeds/results with official NCAA bracket HTML when possible.",
    )
    parser.add_argument(
        "--download-espn-public-picks",
        action="store_true",
        help="Fetch historical public pick distributions from ESPN propositions for 2023-2025.",
    )
    parser.add_argument(
        "--download-mrchmadness-public-picks",
        action="store_true",
        help="Fetch historical public pick distributions from mRchmadness pred.pop data for older seasons.",
    )
    parser.add_argument("--kaggle-raw-dir", type=Path, default=DEFAULT_KAGGLE_RAW_DIR, help="Directory for downloaded historical Kaggle-style CSV files.")
    parser.add_argument("--public-reference", type=Path, default=DEFAULT_PUBLIC_REF, help="Public pick reference JSON.")
    parser.add_argument("--seed-history", type=Path, default=DEFAULT_SEED_HISTORY, help="Historical seed matchup JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact directory under data/models/v10.")
    parser.add_argument("--synthetic-seasons", default="2019,2021,2022,2023,2024,2025", help="Comma-separated fallback seasons.")
    parser.add_argument("--rows-per-season", type=int, default=192, help="Additional synthetic non-round1 rows per season.")
    parser.add_argument("--seed", type=int, default=20260318, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    public_reference = _load_json(args.public_reference) if args.public_reference.exists() else {}
    seed_history = _load_json(args.seed_history) if args.seed_history.exists() else {}
    fallback_seasons = [int(value) for value in args.synthetic_seasons.split(",") if value.strip()]

    source_paths: list[str] = []
    snapshots: dict[int, dict[str, Any]] = {}
    game_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []

    if args.download_kaggle_mirror:
        downloaded = download_kaggle_mirror_bundle(args.kaggle_raw_dir)
        source_paths.extend(str(path) for path in downloaded.values())

    empirical_ready = all((args.kaggle_raw_dir / filename).exists() for filename in DEFAULT_KAGGLE_FILES)
    if empirical_ready:
        supplemental_frames = (
            _official_tournament_supplemental_frames(args.kaggle_raw_dir, fallback_seasons)
            if args.download_official_brackets
            else None
        )
        empirical_games, empirical_public, empirical_snapshots, empirical_coverage = build_empirical_rows_from_kaggle_bundle(
            args.kaggle_raw_dir,
            seasons=fallback_seasons,
            public_reference=public_reference,
            seed_history=seed_history,
            supplemental_frames=supplemental_frames,
        )
        game_rows.extend(empirical_games)
        public_rows.extend(empirical_public)
        snapshots.update(empirical_snapshots)
        source_paths.extend(str(args.kaggle_raw_dir / filename) for filename in DEFAULT_KAGGLE_FILES)
        if supplemental_frames:
            source_paths.extend(
                OFFICIAL_BRACKET_URL.format(season=season)
                for season in fallback_seasons
                if any(
                    not frame[frame["Season"] == season].empty
                    for frame in supplemental_frames.values()
                )
            )
        missing_game_seasons = sorted(set(fallback_seasons) - set(empirical_coverage["game_seasons"]))
        missing_public_seasons = sorted(set(fallback_seasons) - set(empirical_coverage["public_seasons"]))
    else:
        missing_game_seasons = list(fallback_seasons)
        missing_public_seasons = list(fallback_seasons)

    candidate_paths = _candidate_input_paths(
        args.input_path,
        exclude_roots=[
            args.kaggle_raw_dir,
            DEFAULT_KAGGLE_RAW_DIR,
        ],
    )
    explicit_game_rows: list[dict[str, Any]] = []
    explicit_public_rows: list[dict[str, Any]] = []
    for path in candidate_paths:
        rows = _load_rows_from_path(path)
        if not rows:
            continue
        path_name = path.name.lower()
        if "public" in path_name or any("round_name" in row for row in rows):
            for row in rows:
                row.setdefault("source_type", "real_external_public")
            explicit_public_rows.extend(rows)
        else:
            for row in rows:
                row.setdefault("source_type", "real_external_game")
            explicit_game_rows.extend(rows)
        source_paths.append(str(path))

    for path in args.public_input_path:
        rows = _load_rows_from_path(path)
        if not rows:
            continue
        for row in rows:
            row.setdefault("source_type", "real_external_public")
        explicit_public_rows.extend(rows)
        source_paths.append(str(path))

    if explicit_game_rows:
        explicit_game_seasons = {_safe_int(row.get("season"), 0) for row in explicit_game_rows if _safe_int(row.get("season"), 0) != 0}
        game_rows = [row for row in game_rows if _safe_int(row.get("season"), 0) not in explicit_game_seasons]
        game_rows.extend(explicit_game_rows)
        missing_game_seasons = sorted(set(missing_game_seasons) - explicit_game_seasons)
    if explicit_public_rows:
        explicit_public_seasons = {_safe_int(row.get("season"), 0) for row in explicit_public_rows if _safe_int(row.get("season"), 0) != 0}
        public_rows = [row for row in public_rows if _safe_int(row.get("season"), 0) not in explicit_public_seasons]
        public_rows.extend(explicit_public_rows)
        missing_public_seasons = sorted(set(missing_public_seasons) - explicit_public_seasons)

    if args.download_espn_public_picks:
        espn_public_rows, espn_sources = _fetch_espn_public_picks_rows(DEFAULT_ESPN_PUBLIC_CHALLENGES)
        espn_seasons = {_safe_int(row.get("season"), 0) for row in espn_public_rows if _safe_int(row.get("season"), 0) != 0}
        if espn_seasons:
            public_rows = [row for row in public_rows if _safe_int(row.get("season"), 0) not in espn_seasons]
            public_rows.extend(espn_public_rows)
            missing_public_seasons = sorted(set(missing_public_seasons) - espn_seasons)
            source_paths.extend(espn_sources)

    if args.download_mrchmadness_public_picks:
        mrch_public_rows, mrch_sources = _fetch_mrchmadness_public_rows(
            list(DEFAULT_MRCHMADNESS_PUBLIC_SEASONS),
            snapshots,
        )
        mrch_seasons = {_safe_int(row.get("season"), 0) for row in mrch_public_rows if _safe_int(row.get("season"), 0) != 0}
        if mrch_seasons:
            public_rows = [row for row in public_rows if _safe_int(row.get("season"), 0) not in mrch_seasons]
            public_rows.extend(mrch_public_rows)
            missing_public_seasons = sorted(set(missing_public_seasons) - mrch_seasons)
            source_paths.extend(mrch_sources)

    scoreboard_game_rows, scoreboard_sources = _fetch_ncaa_scoreboard_game_rows(
        seasons=missing_game_seasons,
        raw_dir=args.kaggle_raw_dir,
    )
    scoreboard_seasons = {
        _safe_int(row.get("season"), 0)
        for row in scoreboard_game_rows
        if _safe_int(row.get("season"), 0) != 0
    }
    if scoreboard_seasons:
        game_rows = [row for row in game_rows if _safe_int(row.get("season"), 0) not in scoreboard_seasons]
        game_rows.extend(scoreboard_game_rows)
        missing_game_seasons = sorted(set(missing_game_seasons) - scoreboard_seasons)
        source_paths.extend(scoreboard_sources)

    synthetic_used = bool(missing_game_seasons or missing_public_seasons)
    if missing_game_seasons or missing_public_seasons:
        snapshot = _load_json(args.snapshot)
        synth_game_rows, synth_public_rows = _generate_synthetic_rows(
            snapshot,
            public_reference,
            seed_history,
            seasons=sorted(set(missing_game_seasons) | set(missing_public_seasons)),
            rows_per_season=args.rows_per_season,
            seed=args.seed,
        )
        if missing_game_seasons:
            game_rows.extend([row for row in synth_game_rows if _safe_int(row.get("season"), 0) in set(missing_game_seasons)])
        if missing_public_seasons:
            public_rows.extend([row for row in synth_public_rows if _safe_int(row.get("season"), 0) in set(missing_public_seasons)])
        source_paths.extend([str(args.snapshot), str(args.public_reference), str(args.seed_history)])

    games_path = output_dir / "historical_games.parquet"
    public_path = output_dir / "historical_public.parquet"
    manifest_path = output_dir / "historical_manifest.json"
    snapshots_root = output_dir / "snapshots"

    pd.DataFrame(game_rows).to_parquet(games_path, index=False)
    pd.DataFrame(public_rows).to_parquet(public_path, index=False)
    for snapshot in snapshots.values():
        save_historical_snapshot_dataset(snapshot, output_dir=snapshots_root)

    season_values = sorted(
        {
            _safe_int(row.get("season"), 0)
            for row in (game_rows + public_rows)
            if _safe_int(row.get("season"), 0) != 0
        }
    )
    games_coverage = _coverage_payload(game_rows, fallback_source_types={"synthetic_round1", "synthetic_pair"})
    public_coverage = _coverage_payload(
        public_rows,
        fallback_source_types={"synthetic_public", "heuristic_public_from_real_snapshot"},
    )
    manifest = {
        "created_on": date.today().isoformat(),
        "synthetic_fallback_used": synthetic_used,
        "row_count": len(game_rows),
        "public_row_count": len(public_rows),
        "seasons": season_values,
        "sources": sorted(set(source_paths)),
        "tables": {
            "games": games_coverage,
            "public": public_coverage,
            "snapshots": {
                "season_count": len(snapshots),
                "seasons": sorted(snapshots),
                "path": str(snapshots_root),
            },
        },
        "provenance": {
            "games": manifest_table_quality({"tables": {"games": games_coverage}}, "games"),
            "public": manifest_table_quality({"tables": {"public": public_coverage}}, "public"),
        },
        "artifacts": {
            "games": str(games_path),
            "public": str(public_path),
            "snapshots_root": str(snapshots_root),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[v10-dataset] games: {games_path}")
    print(f"[v10-dataset] public: {public_path}")
    print(f"[v10-dataset] manifest: {manifest_path}")
    print(f"[v10-dataset] rows={len(game_rows)} public_rows={len(public_rows)} synthetic={synthetic_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

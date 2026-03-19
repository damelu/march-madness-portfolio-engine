from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PUBLIC_PROXY_SOURCE_TYPES = frozenset({"heuristic_public_from_real_snapshot"})
PUBLIC_SYNTHETIC_SOURCE_TYPES = frozenset({"synthetic_public"})
GAME_SYNTHETIC_SOURCE_TYPES = frozenset({"synthetic_round1", "synthetic_pair"})


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def load_adjacent_artifact_manifest(artifact_path: str | Path | None) -> dict[str, Any] | None:
    if artifact_path is None:
        return None
    path = Path(artifact_path)
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def season_mode_map(manifest: Mapping[str, Any] | None, table_name: str) -> dict[int, str]:
    if not manifest:
        return {}
    seasons_detail = manifest.get("tables", {}).get(table_name, {}).get("seasons_detail", {})
    resolved: dict[int, str] = {}
    for season, payload in seasons_detail.items():
        if not isinstance(payload, Mapping):
            continue
        try:
            resolved[int(season)] = str(payload.get("mode", ""))
        except (TypeError, ValueError):
            continue
    return resolved


def season_source_type_map(manifest: Mapping[str, Any] | None, table_name: str) -> dict[int, list[str]]:
    if not manifest:
        return {}
    seasons_detail = manifest.get("tables", {}).get(table_name, {}).get("seasons_detail", {})
    resolved: dict[int, list[str]] = {}
    for season, payload in seasons_detail.items():
        if not isinstance(payload, Mapping):
            continue
        try:
            resolved[int(season)] = sorted(str(value) for value in payload.get("source_types", []))
        except (TypeError, ValueError):
            continue
    return resolved


def requested_game_provenance(
    manifest: Mapping[str, Any] | None,
    *,
    train_seasons: Iterable[int],
    validation_seasons: Iterable[int],
    holdout_seasons: Iterable[int],
    row_count: int,
) -> dict[str, Any]:
    season_modes = season_mode_map(manifest, "games")
    requested_train = [int(value) for value in train_seasons]
    requested_validation = [int(value) for value in validation_seasons]
    requested_holdout = [int(value) for value in holdout_seasons]
    requested_all = sorted(set(requested_train + requested_validation + requested_holdout))
    effective_train = [season for season in requested_train if season in season_modes]
    effective_validation = [season for season in requested_validation if season in season_modes]
    effective_holdout = [season for season in requested_holdout if season in season_modes]
    effective_all = sorted(set(effective_train + effective_validation + effective_holdout))
    missing_requested = [season for season in requested_all if season not in season_modes]
    modes = {season_modes[season] for season in effective_all}

    if effective_all and not missing_requested and modes == {"real"}:
        training_data_mode = "real_only"
    elif effective_all and not missing_requested and modes <= {"synthetic"}:
        training_data_mode = "synthetic_only"
    else:
        training_data_mode = "mixed"

    reasons: list[str] = []
    if missing_requested:
        reasons.append(f"missing_seasons={','.join(str(value) for value in missing_requested)}")
    non_real = [season for season in effective_all if season_modes.get(season) != "real"]
    if non_real:
        reasons.append(f"non_real_seasons={','.join(str(value) for value in non_real)}")

    return {
        "training_data_mode": training_data_mode,
        "requested_train_seasons": requested_train,
        "requested_validation_seasons": requested_validation,
        "requested_holdout_seasons": requested_holdout,
        "effective_train_seasons": effective_train,
        "effective_validation_seasons": effective_validation,
        "effective_holdout_seasons": effective_holdout,
        "effective_row_count": int(row_count),
        "missing_requested_seasons": missing_requested,
        "release_eligible": training_data_mode == "real_only" and not missing_requested,
        "blocking_issues": reasons,
    }


def empirical_public_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        source_type = str(row.get("source_type", "unknown"))
        if source_type in PUBLIC_PROXY_SOURCE_TYPES or source_type in PUBLIC_SYNTHETIC_SOURCE_TYPES:
            continue
        filtered.append(dict(row))
    return filtered


def public_history_provenance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    all_rows = [dict(row) for row in rows]
    filtered = empirical_public_rows(all_rows)
    historical_public_seasons = sorted(
        {
            _safe_int(row.get("season"), 0)
            for row in all_rows
            if _safe_int(row.get("season"), 0) != 0
        }
    )
    source_types = sorted({str(row.get("source_type", "unknown")) for row in all_rows})
    effective_source_types = sorted({str(row.get("source_type", "unknown")) for row in filtered})
    if filtered:
        mode = "empirical"
    elif all_rows:
        mode = "proxy_only"
    else:
        mode = "reference_only"
    return {
        "public_history_mode": mode,
        "effective_historical_public_row_count": len(filtered),
        "historical_public_row_count": len(all_rows),
        "historical_public_seasons": historical_public_seasons,
        "historical_public_source_types": source_types,
        "effective_historical_public_source_types": effective_source_types,
        "release_eligible": mode == "empirical",
    }


def manifest_table_quality(manifest: Mapping[str, Any] | None, table_name: str) -> dict[str, Any]:
    modes = season_mode_map(manifest, table_name)
    source_types = season_source_type_map(manifest, table_name)
    if table_name == "games":
        empirical = [season for season, mode in modes.items() if mode == "real"]
        fallback = [season for season, mode in modes.items() if mode != "real"]
        quality_grade = "empirical" if empirical and not fallback else ("mixed" if empirical else "synthetic")
    else:
        empirical = []
        proxy = []
        synthetic = []
        for season, values in source_types.items():
            sources = set(values)
            if sources and sources.isdisjoint(PUBLIC_PROXY_SOURCE_TYPES | PUBLIC_SYNTHETIC_SOURCE_TYPES):
                empirical.append(season)
            elif sources & PUBLIC_PROXY_SOURCE_TYPES:
                proxy.append(season)
            else:
                synthetic.append(season)
        if empirical and not proxy and not synthetic:
            quality_grade = "empirical"
        elif proxy and not empirical:
            quality_grade = "proxy"
        elif synthetic and not empirical and not proxy:
            quality_grade = "synthetic"
        else:
            quality_grade = "mixed"
        fallback = sorted(proxy + synthetic)
    return {
        "quality_grade": quality_grade,
        "empirical_seasons": sorted(empirical),
        "fallback_seasons": sorted(fallback),
        "season_count": len(modes),
    }


def backtest_release_readiness(
    *,
    manifest: Mapping[str, Any] | None,
    train_seasons: Iterable[int],
    validation_seasons: Iterable[int],
    holdout_seasons: Iterable[int],
    row_count: int,
    empirical_only_holdout_seasons: Iterable[int],
) -> dict[str, Any]:
    game_provenance = requested_game_provenance(
        manifest,
        train_seasons=train_seasons,
        validation_seasons=validation_seasons,
        holdout_seasons=holdout_seasons,
        row_count=row_count,
    )
    empirical_holdouts = [int(value) for value in empirical_only_holdout_seasons]
    blocking_issues = list(game_provenance["blocking_issues"])
    if not empirical_holdouts:
        blocking_issues.append("no_empirical_holdout_seasons")
    return {
        "eligible": bool(game_provenance["release_eligible"] and empirical_holdouts),
        "game_model_provenance": game_provenance,
        "empirical_holdout_seasons": empirical_holdouts,
        "blocking_issues": blocking_issues,
    }

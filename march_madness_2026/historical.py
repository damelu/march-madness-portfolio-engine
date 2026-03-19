from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import HistoricalGameRow, HistoricalTournamentRow, SelectionSundayDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = PROJECT_ROOT / "data" / "features" / "historical"
DEFAULT_GAMES_PATH = HISTORICAL_ROOT / "games.json"
DEFAULT_PUBLIC_PICKS_PATH = HISTORICAL_ROOT / "public_picks.json"
DEFAULT_SNAPSHOTS_ROOT = HISTORICAL_ROOT / "snapshots"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _dump_json(path: Path, payload: object) -> Path:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_historical_games_dataset(
    rows: Iterable[HistoricalGameRow | Mapping[str, object]],
    output_path: Path | None = None,
) -> list[HistoricalGameRow]:
    dataset = [row if isinstance(row, HistoricalGameRow) else HistoricalGameRow(**row) for row in rows]
    if output_path is not None:
        save_historical_games_dataset(dataset, output_path)
    return dataset


def save_historical_games_dataset(
    rows: Sequence[HistoricalGameRow | Mapping[str, object]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or DEFAULT_GAMES_PATH
    dataset = [row if isinstance(row, HistoricalGameRow) else HistoricalGameRow(**row) for row in rows]
    return _dump_json(path, [row.model_dump(mode="json") for row in dataset])


def load_historical_games_dataset(path: Path | None = None) -> list[HistoricalGameRow]:
    payload = _load_json(path or DEFAULT_GAMES_PATH)
    if not isinstance(payload, list):
        raise ValueError("historical games dataset must be a list of rows")
    return [HistoricalGameRow(**row) for row in payload]


def build_historical_selection_sunday_snapshots(
    snapshots: Iterable[HistoricalTournamentRow | SelectionSundayDataset | Mapping[str, object]],
    output_dir: Path | None = None,
) -> dict[int, HistoricalTournamentRow]:
    built: dict[int, HistoricalTournamentRow] = {}
    for snapshot in snapshots:
        if isinstance(snapshot, HistoricalTournamentRow):
            row = snapshot
        elif isinstance(snapshot, SelectionSundayDataset):
            row = HistoricalTournamentRow(
                season=snapshot.season,
                tournament=snapshot.tournament,
                teams=snapshot.teams,
                metadata=dict(snapshot.metadata),
            )
        else:
            row = HistoricalTournamentRow(**snapshot)
        built[row.season] = row
        if output_dir is not None:
            save_historical_snapshot_dataset(row, output_dir=output_dir)
    return built


def save_historical_snapshot_dataset(
    snapshot: HistoricalTournamentRow | SelectionSundayDataset | Mapping[str, object],
    output_dir: Path | None = None,
) -> Path:
    root = output_dir or DEFAULT_SNAPSHOTS_ROOT
    if isinstance(snapshot, HistoricalTournamentRow):
        row = snapshot
    elif isinstance(snapshot, SelectionSundayDataset):
        row = HistoricalTournamentRow(
            season=snapshot.season,
            tournament=snapshot.tournament,
            teams=snapshot.teams,
            metadata=dict(snapshot.metadata),
        )
    else:
        row = HistoricalTournamentRow(**snapshot)
    path = root / f"{row.season}.json"
    return _dump_json(path, row.model_dump(mode="json"))


def load_historical_snapshot_dataset(
    season_or_path: int | str | Path,
    snapshot_dir: Path | None = None,
) -> HistoricalTournamentRow:
    if isinstance(season_or_path, (str, Path)):
        candidate_path = Path(season_or_path)
        path = candidate_path if candidate_path.suffix else (snapshot_dir or DEFAULT_SNAPSHOTS_ROOT) / f"{candidate_path}.json"
    else:
        path = (snapshot_dir or DEFAULT_SNAPSHOTS_ROOT) / f"{season_or_path}.json"
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("historical snapshot dataset must be a mapping")
    return HistoricalTournamentRow(**payload)


def season_blocked_splits(
    seasons: Sequence[int],
    train_window: int | None = None,
    validation_size: int = 1,
    holdout_size: int = 1,
    min_train_seasons: int = 3,
) -> list[dict[str, list[int]]]:
    ordered = sorted(set(int(season) for season in seasons))
    if validation_size <= 0 or holdout_size <= 0:
        raise ValueError("validation_size and holdout_size must be positive")
    if min_train_seasons <= 0:
        raise ValueError("min_train_seasons must be positive")

    splits: list[dict[str, list[int]]] = []
    for holdout_start in range(min_train_seasons + validation_size, len(ordered) - holdout_size + 1):
        train_end = holdout_start - validation_size
        train = ordered[:train_end]
        if train_window is not None:
            train = train[-train_window:]
        if len(train) < min_train_seasons:
            continue
        validation = ordered[train_end:holdout_start]
        holdout = ordered[holdout_start:holdout_start + holdout_size]
        splits.append(
            {
                "train": train,
                "validation": validation,
                "holdout": holdout,
            }
        )
    return splits

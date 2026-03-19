from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yaml

from .models import (
    ContestPayoutProfile,
    ContestProfile,
    ScoringProfile,
    SelectionSundayDataset,
    SimulationProfile,
    TrainingProfile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _profile_mapping(raw: dict) -> dict:
    if raw is None:
        return {}
    return raw.get("profiles", raw)


def load_scoring_profiles(path: Path | None = None) -> Dict[str, ScoringProfile]:
    config_path = path or CONFIG_ROOT / "portfolio" / "scoring_profiles.yaml"
    raw = _profile_mapping(_load_yaml(config_path))
    return {
        profile_id: ScoringProfile(profile_id=profile_id, **payload)
        for profile_id, payload in raw.items()
    }


def load_training_profiles(path: Path | None = None) -> Dict[str, TrainingProfile]:
    config_path = path or CONFIG_ROOT / "model" / "training_profile.yaml"
    raw = _profile_mapping(_load_yaml(config_path))
    return {
        profile_id: TrainingProfile(profile_id=profile_id, **payload)
        for profile_id, payload in raw.items()
    }


def load_payout_profiles(path: Path | None = None) -> Dict[str, ContestPayoutProfile]:
    config_path = path or CONFIG_ROOT / "portfolio" / "payout_profiles.yaml"
    raw = _profile_mapping(_load_yaml(config_path))
    return {
        profile_id: ContestPayoutProfile(profile_id=profile_id, **payload)
        for profile_id, payload in raw.items()
    }


def load_contest_profiles(
    path: Path | None = None,
    payout_profiles: Dict[str, ContestPayoutProfile] | None = None,
) -> Dict[str, ContestProfile]:
    config_path = path or CONFIG_ROOT / "portfolio" / "contest_profiles.yaml"
    raw = _profile_mapping(_load_yaml(config_path))
    profiles = {
        contest_id: ContestProfile(contest_id=contest_id, **payload)
        for contest_id, payload in raw.items()
    }
    if payout_profiles is not None:
        missing = sorted(
            {
                contest.payout_profile
                for contest in profiles.values()
                if contest.payout_profile and contest.payout_profile not in payout_profiles
            }
        )
        if missing:
            raise ValueError(f"unknown payout_profile(s): {', '.join(missing)}")
    return profiles


def load_simulation_profile(path: Path | None = None) -> SimulationProfile:
    config_path = path or CONFIG_ROOT / "portfolio" / "simulation_profile.yaml"
    raw = _load_yaml(config_path)["simulation"]
    return SimulationProfile(**raw)


def load_selection_sunday_dataset(path: Path) -> SelectionSundayDataset:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return SelectionSundayDataset(**raw)

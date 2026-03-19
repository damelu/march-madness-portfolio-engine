from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import (
    CONFIG_ROOT,
    load_contest_profiles,
    load_payout_profiles,
    load_scoring_profiles,
    load_selection_sunday_dataset,
    load_simulation_profile,
    load_training_profiles,
)
from ..models import ContestPayoutProfile, ContestProfile, ScoringProfile, SimulationProfile, TrainingProfile


V10_CONFIG_ROOT = CONFIG_ROOT / "v10"


def load_v10_training_profiles(path: Path | None = None) -> Dict[str, TrainingProfile]:
    return load_training_profiles(path=path)


def load_v10_payout_profiles(path: Path | None = None) -> Dict[str, ContestPayoutProfile]:
    return load_payout_profiles(path=path)


def load_v10_contest_profiles(path: Path | None = None) -> Dict[str, ContestProfile]:
    payout_profiles = load_v10_payout_profiles()
    return load_contest_profiles(path=path, payout_profiles=payout_profiles)


def load_v10_scoring_profiles(path: Path | None = None) -> Dict[str, ScoringProfile]:
    return load_scoring_profiles(path=path)


def load_v10_simulation_profile(path: Path | None = None) -> SimulationProfile:
    return load_simulation_profile(path=path)


__all__ = [
    "V10_CONFIG_ROOT",
    "load_selection_sunday_dataset",
    "load_v10_contest_profiles",
    "load_v10_payout_profiles",
    "load_v10_scoring_profiles",
    "load_v10_simulation_profile",
    "load_v10_training_profiles",
]

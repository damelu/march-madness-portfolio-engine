from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_BLENDED_WEIGHT_FLOOR = 0.10


def _normalize_simplex_with_floor(values: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    floor = max(float(floor), 0.0)
    if floor * values.size >= 1.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        values = np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    else:
        values = values / total
    residual = 1.0 - (floor * values.size)
    return (values * residual) + floor


def normalize_weights(values: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {key: float(value) / total for key, value in values.items()}


def resolve_release_seeds(
    base_seed: int,
    explicit_seeds: Sequence[int] | None = None,
    explicit_seed_string: str | None = None,
    seed_count: int = 1,
) -> list[int]:
    if explicit_seeds:
        return [int(seed) for seed in explicit_seeds]
    if explicit_seed_string:
        parsed = [int(item.strip()) for item in explicit_seed_string.split(",") if item.strip()]
        if parsed:
            return parsed
    return [int(base_seed + offset) for offset in range(max(int(seed_count), 1))]


@dataclass
class V10SearchParams:
    small_weight: float = 0.333333
    mid_weight: float = 0.333333
    large_weight: float = 0.333334

    mix_high_confidence: float = 0.20
    mix_balanced: float = 0.20
    mix_selective_contrarian: float = 0.20
    mix_underdog_upside: float = 0.20
    mix_high_risk_high_return: float = 0.20

    max_per_archetype: float = 2.0
    min_distinct_archetypes: float = 3.0

    def _normalize_group(self, fields: list[str], floor: float) -> None:
        values = np.array([max(float(getattr(self, field)), 0.0) for field in fields], dtype=np.float64)
        values = _normalize_simplex_with_floor(values, floor)
        for field, value in zip(fields, values):
            setattr(self, field, float(value))

    def normalized(self, *, blended_weight_floor: float = DEFAULT_BLENDED_WEIGHT_FLOOR) -> "V10SearchParams":
        normalized = V10SearchParams(**self.__dict__)
        normalized._normalize_group(
            ["small_weight", "mid_weight", "large_weight"],
            floor=max(float(blended_weight_floor), 0.0),
        )
        normalized._normalize_group(
            [
                "mix_high_confidence",
                "mix_balanced",
                "mix_selective_contrarian",
                "mix_underdog_upside",
                "mix_high_risk_high_return",
            ],
            floor=0.05,
        )
        normalized.max_per_archetype = float(int(np.clip(round(normalized.max_per_archetype), 1, 3)))
        normalized.min_distinct_archetypes = float(int(np.clip(round(normalized.min_distinct_archetypes), 2, 5)))
        return normalized

    def to_dict(self, *, blended_weight_floor: float = DEFAULT_BLENDED_WEIGHT_FLOOR) -> dict[str, float]:
        normalized = self.normalized(blended_weight_floor=blended_weight_floor)
        return {key: round(float(value), 6) for key, value in normalized.__dict__.items()}

    def mutate(
        self,
        rng: np.random.Generator,
        scale: float = 0.15,
        *,
        optimize_contest_weights: bool = True,
        blended_weight_floor: float = DEFAULT_BLENDED_WEIGHT_FLOOR,
    ) -> "V10SearchParams":
        mutated = self.normalized(blended_weight_floor=blended_weight_floor)

        def mutate_simplex(fields: list[str], floor: float) -> None:
            values = np.array([max(float(getattr(mutated, field)), floor) for field in fields], dtype=np.float64)
            log_values = np.log(values)
            log_values += rng.normal(0.0, scale, size=len(fields))
            updated = np.exp(log_values - np.max(log_values))
            updated = _normalize_simplex_with_floor(updated, floor)
            for field, value in zip(fields, updated):
                setattr(mutated, field, float(value))

        if optimize_contest_weights and rng.random() < 0.8:
            mutate_simplex(
                ["small_weight", "mid_weight", "large_weight"],
                floor=max(float(blended_weight_floor), 0.0),
            )
        if rng.random() < 0.95:
            mutate_simplex(
                [
                    "mix_high_confidence",
                    "mix_balanced",
                    "mix_selective_contrarian",
                    "mix_underdog_upside",
                    "mix_high_risk_high_return",
                ],
                floor=0.05,
            )
        if rng.random() < 0.35:
            mutated.max_per_archetype = float(int(np.clip(mutated.max_per_archetype + rng.choice([-1, 1]), 1, 3)))
        if rng.random() < 0.35:
            mutated.min_distinct_archetypes = float(
                int(np.clip(mutated.min_distinct_archetypes + rng.choice([-1, 1]), 2, 5))
            )
        return mutated.normalized(blended_weight_floor=blended_weight_floor)


def apply_params_to_engine(
    engine,
    params: V10SearchParams,
    *,
    contest_mode: str = "blended",
    blended_weight_floor: float = DEFAULT_BLENDED_WEIGHT_FLOOR,
) -> None:
    normalized = params.normalized(blended_weight_floor=blended_weight_floor)
    if contest_mode == "blended":
        primary_contests = normalize_weights(
            {
                "standard_small": normalized.small_weight,
                "standard_mid": normalized.mid_weight,
                "standard_large": normalized.large_weight,
            }
        )
    else:
        primary_contests = {contest_mode: 1.0}

    archetype_mix = normalize_weights(
        {
            "high_confidence": normalized.mix_high_confidence,
            "balanced": normalized.mix_balanced,
            "selective_contrarian": normalized.mix_selective_contrarian,
            "underdog_upside": normalized.mix_underdog_upside,
            "high_risk_high_return": normalized.mix_high_risk_high_return,
        }
    )
    engine.simulation_profile = engine.simulation_profile.model_copy(
        update={
            "primary_contests": primary_contests,
            "archetype_mix": archetype_mix,
            "max_per_archetype": int(round(normalized.max_per_archetype)),
            "min_distinct_archetypes": int(round(normalized.min_distinct_archetypes)),
        }
    )


__all__ = [
    "DEFAULT_BLENDED_WEIGHT_FLOOR",
    "V10SearchParams",
    "apply_params_to_engine",
    "normalize_weights",
    "resolve_release_seeds",
]

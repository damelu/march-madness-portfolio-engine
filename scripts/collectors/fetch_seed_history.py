#!/usr/bin/env python3
"""
fetch_seed_history.py — Compile historical seed performance data.

Uses embedded data from NCAA tournament records (1985-2025).
Output: data/reference/seed_matchup_history.json (already created, this validates it)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "seed_matchup_history.json"


def validate_seed_history() -> bool:
    """Validate the seed matchup history file has expected structure."""
    if not OUTPUT_PATH.exists():
        print(f"[seed_history] File not found: {OUTPUT_PATH}")
        return False

    with OUTPUT_PATH.open() as f:
        data = json.load(f)

    # Check required sections
    required_sections = ["round_of_64", "advancement_rates"]
    for section in required_sections:
        if section not in data:
            print(f"[seed_history] Missing section: {section}")
            return False

    # Validate round of 64 matchups
    r64 = data["round_of_64"]
    expected_matchups = [
        "1_vs_16", "2_vs_15", "3_vs_14", "4_vs_13",
        "5_vs_12", "6_vs_11", "7_vs_10", "8_vs_9",
    ]
    for matchup in expected_matchups:
        if matchup not in r64:
            print(f"[seed_history] Missing matchup: {matchup}")
            return False
        win_rate = r64[matchup]["higher_seed_wins"]
        if not 0 < win_rate <= 1:
            print(f"[seed_history] Invalid win rate for {matchup}: {win_rate}")
            return False

    # Validate advancement rates
    adv = data["advancement_rates"]
    expected_rounds = ["sweet_16", "elite_8", "final_four", "championship_game", "champion"]
    for round_name in expected_rounds:
        if round_name not in adv:
            print(f"[seed_history] Missing advancement round: {round_name}")
            return False
        rates = adv[round_name]
        for seed in range(1, 17):
            if str(seed) not in rates:
                print(f"[seed_history] Missing seed {seed} in {round_name}")
                return False

    # Sanity checks
    assert r64["1_vs_16"]["higher_seed_wins"] > 0.98, "1-seed should win >98% vs 16"
    assert r64["8_vs_9"]["higher_seed_wins"] < 0.55, "8-9 should be close to 50-50"
    assert adv["champion"]["1"] > adv["champion"]["2"], "1-seeds should win more titles"

    print("[seed_history] Validation passed!")
    return True


def print_summary():
    """Print a summary of historical seed performance."""
    with OUTPUT_PATH.open() as f:
        data = json.load(f)

    print("\n=== Historical Seed Performance (1985-2025) ===\n")
    print("Round of 64 (higher seed win rates):")
    for matchup, info in data["round_of_64"].items():
        seeds = matchup.replace("_vs_", " vs ")
        print(f"  #{seeds}: {info['higher_seed_wins']:.1%} ({info['upsets']} upsets in {info['sample_size']} games)")

    print("\nAdvancement rates by seed:")
    header = f"{'Seed':>4} {'S16':>6} {'E8':>6} {'F4':>6} {'CG':>6} {'Champ':>6}"
    print(header)
    adv = data["advancement_rates"]
    for seed in range(1, 17):
        s = str(seed)
        print(
            f"  {seed:>2}  "
            f"{adv['sweet_16'][s]:>5.1%}  "
            f"{adv['elite_8'][s]:>5.1%}  "
            f"{adv['final_four'][s]:>5.1%}  "
            f"{adv['championship_game'][s]:>5.1%}  "
            f"{adv['champion'][s]:>5.1%}"
        )


def main() -> int:
    if validate_seed_history():
        print_summary()
        return 0
    else:
        print("[seed_history] Seed history file needs fixing")
        return 1


if __name__ == "__main__":
    sys.exit(main())

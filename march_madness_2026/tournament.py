from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .models import BracketCandidate, SelectionSundayDataset, TeamSnapshot

# V6: BPI advancement rates for ensemble blending (ESPN BPI, March 2026)
# Format: team_id → championship probability (0-1)

# V6: Vegas spread-implied first-round win probabilities
# Spread → probability: P ≈ sigmoid(spread * 0.14) approximately
_VEGAS_FIRST_ROUND: Dict[str, float] = {
    "duke": 0.99, "arizona": 0.99, "michigan": 0.99, "florida": 0.98,
    "uconn": 0.97, "houston": 0.98, "purdue": 0.98, "iowa-state": 0.98,
    "michigan-state": 0.94, "illinois": 0.97, "gonzaga": 0.96, "virginia": 0.93,
    "kansas": 0.93, "nebraska": 0.93, "arkansas": 0.92, "alabama": 0.90,
    "st-johns": 0.85, "wisconsin": 0.76, "vanderbilt": 0.81, "texas-tech": 0.82,
    "louisville": 0.81, "tennessee": 0.79, "north-carolina": 0.62, "byu": 0.63,
    "ucla": 0.72, "kentucky": 0.72, "miami-fl": 0.55, "saint-marys": 0.53,
    "ohio-state": 0.65, "clemson": 0.51, "georgia": 0.51,
}


STANDARD_REGION_ORDER = ["East", "West", "South", "Midwest"]
REGION_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "National Championship",
}


@dataclass(frozen=True)
class ArchetypeProfile:
    name: str
    risk_label: str
    temperature: float
    favorite_bias: float
    late_round_favorite_bias: float


@dataclass(frozen=True)
class GameMeta:
    game_index: int
    round_number: int
    round_name: str
    region: str
    slot_name: str


ARCHETYPE_PROFILES: Dict[str, ArchetypeProfile] = {
    "high_confidence": ArchetypeProfile(
        name="High Confidence",
        risk_label="low",
        temperature=0.72,
        favorite_bias=0.16,
        late_round_favorite_bias=0.10,
    ),
    "balanced": ArchetypeProfile(
        name="Balanced",
        risk_label="medium",
        temperature=0.95,
        favorite_bias=0.04,
        late_round_favorite_bias=0.05,
    ),
    "selective_contrarian": ArchetypeProfile(
        name="Selective Contrarian",
        risk_label="medium_high",
        temperature=1.12,
        favorite_bias=-0.08,
        late_round_favorite_bias=0.02,
    ),
    "underdog_upside": ArchetypeProfile(
        name="Underdog Upside",
        risk_label="high",
        temperature=1.28,
        favorite_bias=-0.16,
        late_round_favorite_bias=-0.02,
    ),
    "high_risk_high_return": ArchetypeProfile(
        name="High Risk High Return",
        risk_label="very_high",
        temperature=1.45,
        favorite_bias=-0.26,
        late_round_favorite_bias=-0.08,
    ),
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class TournamentModel:
    def __init__(self, dataset: SelectionSundayDataset):
        self.dataset = dataset
        self.teams: List[TeamSnapshot] = list(dataset.teams)
        self.regions = self._resolve_regions()
        self._validate_field()

        self.team_id_by_index = [team.team_id for team in self.teams]
        self.team_name_by_index = [team.team_name for team in self.teams]
        self.index_by_team_id = {team.team_id: idx for idx, team in enumerate(self.teams)}
        self.seed_by_index = np.array([team.seed for team in self.teams], dtype=np.int16)
        self.region_by_index = [team.region for team in self.teams]
        self.strength_by_index = np.array(
            [self._compute_strength(team) for team in self.teams], dtype=np.float32
        )
        self.volatility_by_index = np.array([team.volatility for team in self.teams], dtype=np.float32)
        self.region_fields = self._build_region_fields()
        self.game_metadata = self._build_game_metadata()
        self.game_rounds = np.array([meta.round_number for meta in self.game_metadata], dtype=np.int16)

    def _resolve_regions(self) -> List[str]:
        raw_regions = sorted({team.region for team in self.teams})
        ordering = {region: index for index, region in enumerate(STANDARD_REGION_ORDER)}
        return sorted(raw_regions, key=lambda region: ordering.get(region, len(ordering) + 1))

    def _validate_field(self) -> None:
        if len(self.regions) != 4:
            raise ValueError("the bracket engine expects exactly four regions")
        for region in self.regions:
            region_teams = [team for team in self.teams if team.region == region]
            seeds = sorted(team.seed for team in region_teams)
            if len(region_teams) != 16 or seeds != list(range(1, 17)):
                raise ValueError(f"region {region} must contain one team for each seed 1-16")

    def _build_region_fields(self) -> Dict[str, List[int]]:
        fields: Dict[str, List[int]] = {}
        for region in self.regions:
            region_teams = {team.seed: self.index_by_team_id[team.team_id] for team in self.teams if team.region == region}
            fields[region] = [region_teams[seed] for seed in REGION_SEED_ORDER]
        return fields

    def _build_game_metadata(self) -> List[GameMeta]:
        metadata: List[GameMeta] = []
        game_index = 0
        for region in self.regions:
            for round_number, game_count in [(1, 8), (2, 4), (3, 2), (4, 1)]:
                for slot in range(1, game_count + 1):
                    metadata.append(
                        GameMeta(
                            game_index=game_index,
                            round_number=round_number,
                            round_name=ROUND_NAMES[round_number],
                            region=region,
                            slot_name=f"{region.lower()}_r{round_number}_g{slot}",
                        )
                    )
                    game_index += 1
        for slot in range(1, 3):
            metadata.append(
                GameMeta(
                    game_index=game_index,
                    round_number=5,
                    round_name=ROUND_NAMES[5],
                    region="FinalFour",
                    slot_name=f"final_four_{slot}",
                )
            )
            game_index += 1
        metadata.append(
            GameMeta(
                game_index=game_index,
                round_number=6,
                round_name=ROUND_NAMES[6],
                region="Championship",
                slot_name="national_championship",
            )
        )
        return metadata

    @staticmethod
    def _compute_strength(team: TeamSnapshot) -> float:
        return (
            team.rating
            + team.market_adjustment
            + 0.75 * team.coaching_adjustment
            + 0.55 * team.continuity_adjustment
            + 0.25 * team.offense_rating
            + 0.20 * team.defense_rating
            + 0.10 * team.tempo_adjustment
            - 1.30 * team.injury_penalty
        )

    def win_probability(self, team_a_index: int, team_b_index: int) -> float:
        # Calibration target: seed_matchup_history.json provides historical base rates
        # (e.g., 1v16: 99.4%, 5v12: 64.8%, 8v9: 51.2%). The sigmoid scale (6.75) and
        # nonlinear volatility dampening are tuned to approximate these rates.
        strength_gap = float(self.strength_by_index[team_a_index] - self.strength_by_index[team_b_index])
        base_probability = _sigmoid(strength_gap / 6.75)
        volatility = min(
            0.35,
            float(self.volatility_by_index[team_a_index] + self.volatility_by_index[team_b_index]) / 2.0,
        )

        # V6: Matchup-specific adjustments
        team_a = self.teams[team_a_index]
        team_b = self.teams[team_b_index]

        # Tempo mismatch: slower team gets a small edge (fewer possessions = more variance = favors underdog)
        tempo_diff = abs(team_a.tempo_adjustment - team_b.tempo_adjustment)
        if tempo_diff > 0.3:
            slower = team_a if team_a.tempo_adjustment < team_b.tempo_adjustment else team_b
            slower_is_a = slower is team_a
            tempo_bonus = 0.015 * tempo_diff  # ~1.5% per 0.1 tempo unit
            if slower_is_a:
                base_probability += tempo_bonus
            else:
                base_probability -= tempo_bonus

        # 3-point dependent teams have higher variance (cold shooting night = elimination)
        three_pt_var = abs(team_a.three_point_rate) + abs(team_b.three_point_rate)
        volatility += three_pt_var * 0.03  # 3PT dependence adds volatility

        # Free throw rate: teams that get to the line are more consistent in close games
        ft_advantage = team_a.free_throw_rate - team_b.free_throw_rate
        base_probability += ft_advantage * 0.01

        base_probability = min(max(base_probability, 0.005), 0.995)
        volatility = min(0.35, volatility)

        # Nonlinear volatility dampening
        extremity = abs(base_probability - 0.5) * 2.0
        dampened_volatility = volatility * (1.0 - extremity * 0.85)
        model_prob = 0.5 + (base_probability - 0.5) * (1.0 - max(0.0, dampened_volatility))

        return min(max(model_prob, 0.005), 0.995)

    def win_probability_round1(self, team_a_index: int, team_b_index: int) -> float:
        """First-round win probability with Vegas/BPI ensemble blending.

        Only used for R1 games — later rounds use pure model probability
        to avoid contaminating with first-round-specific Vegas lines.
        """
        model_prob = self.win_probability(team_a_index, team_b_index)
        team_a_id = self.team_id_by_index[team_a_index]
        team_b_id = self.team_id_by_index[team_b_index]
        bpi_a = _VEGAS_FIRST_ROUND.get(team_a_id)
        bpi_b = _VEGAS_FIRST_ROUND.get(team_b_id)
        if bpi_a is not None and bpi_b is not None:
            bpi_relative = bpi_a / max(bpi_a + bpi_b, 0.01)
            model_prob = 0.75 * model_prob + 0.25 * bpi_relative
        return min(max(model_prob, 0.005), 0.995)

    def _round_model_probability(self, team_a_index: int, team_b_index: int, round_number: int) -> float:
        if round_number == 1:
            return self.win_probability_round1(team_a_index, team_b_index)
        return self.win_probability(team_a_index, team_b_index)

    def _public_pick_probability(self, team_a_index: int, team_b_index: int, round_number: int) -> float:
        base_prob = self._round_model_probability(team_a_index, team_b_index, round_number)
        base_logit = math.log(max(base_prob, 0.001) / max(1.0 - base_prob, 0.001))
        team_a = self.teams[team_a_index]
        team_b = self.teams[team_b_index]
        popularity_gap = team_a.public_pick_pct - team_b.public_pick_pct
        round_multiplier = 1.0 if round_number <= 3 else 1.5
        adjusted_logit = (base_logit + 0.02 * popularity_gap * round_multiplier) / 0.88
        return min(max(_sigmoid(adjusted_logit), 0.005), 0.995)

    def _candidate_pick_probability(
        self,
        team_a_index: int,
        team_b_index: int,
        round_number: int,
        archetype: ArchetypeProfile,
    ) -> float:
        probability_team_a = self._round_model_probability(team_a_index, team_b_index, round_number)
        favorite_is_team_a = probability_team_a >= 0.5
        favorite_probability = probability_team_a if favorite_is_team_a else 1.0 - probability_team_a
        favorite_probability = min(max(favorite_probability, 0.01), 0.99)
        favorite_seed = int(self.seed_by_index[team_a_index if favorite_is_team_a else team_b_index])
        underdog_seed = int(self.seed_by_index[team_b_index if favorite_is_team_a else team_a_index])
        seed_gap = max(underdog_seed - favorite_seed, 1) / 16.0
        logit = math.log(favorite_probability / (1.0 - favorite_probability))
        # Apply all biases on the logit scale (not probability scale) to preserve
        # probability axioms — per code review finding #3
        adjusted_logit = logit / archetype.temperature
        adjusted_logit += archetype.favorite_bias * seed_gap * 4.0
        if round_number >= 5:
            adjusted_logit += archetype.late_round_favorite_bias * seed_gap * 4.0
        adjusted_favorite = min(max(_sigmoid(adjusted_logit), 0.03), 0.97)
        return adjusted_favorite if favorite_is_team_a else 1.0 - adjusted_favorite

    def _play_round(
        self,
        team_indices: Sequence[int],
        rng: np.random.Generator,
        round_number: int,
        archetype: ArchetypeProfile | None = None,
        public: bool = False,
    ) -> tuple[List[int], List[int], List[int], List[int]]:
        winners: List[int] = []
        round_pick_indices: List[int] = []
        upset_gaps: List[int] = []
        favorite_picks: List[int] = []

        for left, right in zip(team_indices[::2], team_indices[1::2]):
            model_left_probability = self._round_model_probability(left, right, round_number)
            if public:
                left_probability = self._public_pick_probability(left, right, round_number)
            elif archetype is None:
                left_probability = model_left_probability
            else:
                left_probability = self._candidate_pick_probability(left, right, round_number, archetype)
            winner = left if float(rng.random()) < left_probability else right
            loser = right if winner == left else left
            favorite = left if model_left_probability >= 0.5 else right

            winners.append(winner)
            round_pick_indices.append(winner)
            upset_gaps.append(max(int(self.seed_by_index[winner]) - int(self.seed_by_index[loser]), 0))
            favorite_picks.append(1 if winner == favorite else 0)

        return winners, round_pick_indices, upset_gaps, favorite_picks

    def simulate_tournament(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        winners: List[int] = []
        upset_gaps: List[int] = []
        region_champions: List[int] = []

        for region in self.regions:
            current = list(self.region_fields[region])
            for round_number in range(1, 5):
                current, round_winners, round_upsets, _ = self._play_round(current, rng, round_number)
                winners.extend(round_winners)
                upset_gaps.extend(round_upsets)
            region_champions.append(current[0])

        semifinal_a, semifinal_winners_a, semifinal_upsets_a, _ = self._play_round(
            [region_champions[0], region_champions[1]], rng, 5
        )
        semifinal_b, semifinal_winners_b, semifinal_upsets_b, _ = self._play_round(
            [region_champions[2], region_champions[3]], rng, 5
        )
        winners.extend(semifinal_winners_a + semifinal_winners_b)
        upset_gaps.extend(semifinal_upsets_a + semifinal_upsets_b)

        _, final_winners, final_upsets, _ = self._play_round(
            [semifinal_a[0], semifinal_b[0]], rng, 6
        )
        winners.extend(final_winners)
        upset_gaps.extend(final_upsets)

        return np.array(winners, dtype=np.int16), np.array(upset_gaps, dtype=np.int16)

    def simulate_many(self, num_simulations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized Monte Carlo: simulate ALL tournaments simultaneously.

        Instead of looping 5000 times in Python, we process all simulations
        per round in one vectorized operation. This reduces 5000 × 63 Python ops
        to 6 × num_games_per_round vectorized numpy ops.

        Pattern from GPU Monte Carlo research: "vectorize across simulations,
        loop across rounds" (QB2134/GPU-Accelerated-Monte-Carlo).
        """
        try:
            from .gpu import GPU_AVAILABLE, xp, to_numpy
        except ImportError:
            GPU_AVAILABLE = False
            xp = np
            to_numpy = lambda x: x

        N = num_simulations
        num_games = len(self.game_metadata)
        num_teams = len(self.teams)

        # Pre-compute full pairwise probability matrix (deterministic, computed once)
        prob_matrix = np.zeros((num_teams, num_teams), dtype=np.float32)
        prob_matrix_r1 = np.zeros((num_teams, num_teams), dtype=np.float32)
        for i in range(num_teams):
            for j in range(num_teams):
                if i != j:
                    prob_matrix[i, j] = self.win_probability(i, j)
                    prob_matrix_r1[i, j] = self.win_probability_round1(i, j)

        # Move to GPU if available
        if GPU_AVAILABLE:
            prob_mat = xp.asarray(prob_matrix)
            prob_mat_r1 = xp.asarray(prob_matrix_r1)
            seed_arr = xp.asarray(self.seed_by_index)
            rng_gpu = xp.random.default_rng(seed)
        else:
            prob_mat = prob_matrix
            prob_mat_r1 = prob_matrix_r1
            seed_arr = self.seed_by_index.astype(np.int16)
            rng_gpu = np.random.default_rng(seed)

        # Pre-generate ALL random numbers at once: (N, 63)
        all_randoms = rng_gpu.random((N, num_games), dtype=np.float32)

        # Output arrays
        outcomes = xp.zeros((N, num_games), dtype=xp.int16)
        upset_gaps_arr = xp.zeros((N, num_games), dtype=xp.int16)

        game_idx = 0  # tracks position in the 63-game sequence

        # Process 4 regions
        for region in self.regions:
            fields = self.region_fields[region]
            # current_teams: (N, 16) — all sims start with the same 16 teams
            current_teams = xp.tile(xp.array(fields, dtype=xp.int16), (N, 1))

            for round_number in range(1, 5):
                num_matchups = current_teams.shape[1] // 2
                # Split into matchup pairs
                team_a = current_teams[:, 0::2]  # (N, num_matchups)
                team_b = current_teams[:, 1::2]  # (N, num_matchups)

                # Look up probabilities for all sims × all matchups
                prob_round = prob_mat_r1 if round_number == 1 else prob_mat
                probs = prob_round[team_a.ravel().get() if GPU_AVAILABLE else team_a.ravel(),
                                   team_b.ravel().get() if GPU_AVAILABLE else team_b.ravel()]
                if GPU_AVAILABLE:
                    probs = xp.asarray(probs)
                probs = probs.reshape(N, num_matchups)

                # Get random numbers for these games
                randoms = all_randoms[:, game_idx:game_idx + num_matchups]

                # Determine winners vectorized across all sims
                winners = xp.where(randoms < probs, team_a, team_b)
                losers = xp.where(randoms < probs, team_b, team_a)

                # Compute upset gaps
                winner_seeds = seed_arr[winners.ravel()].reshape(N, num_matchups)
                loser_seeds = seed_arr[losers.ravel()].reshape(N, num_matchups)
                gaps = xp.maximum(winner_seeds - loser_seeds, 0)

                # Store results
                outcomes[:, game_idx:game_idx + num_matchups] = winners
                upset_gaps_arr[:, game_idx:game_idx + num_matchups] = gaps

                # Advance to next round
                current_teams = winners
                game_idx += num_matchups

        # Final Four: 2 semifinal games + 1 championship
        # Collect region champions from outcomes
        # Region champion indices: game 14, 29, 44, 59 (last game of each region's 15 games)
        region_champ_games = [14, 29, 44, 59]
        region_champs = xp.stack([outcomes[:, g] for g in region_champ_games], axis=1)  # (N, 4)

        # Semifinal 1: region 0 vs region 1
        sf1_a = region_champs[:, 0:1]  # (N, 1)
        sf1_b = region_champs[:, 1:2]

        sf1_probs_flat = prob_mat[sf1_a.ravel().get() if GPU_AVAILABLE else sf1_a.ravel(),
                                  sf1_b.ravel().get() if GPU_AVAILABLE else sf1_b.ravel()]
        if GPU_AVAILABLE:
            sf1_probs_flat = xp.asarray(sf1_probs_flat)
        sf1_probs = sf1_probs_flat.reshape(N, 1)
        sf1_rand = all_randoms[:, game_idx:game_idx + 1]
        sf1_winners = xp.where(sf1_rand < sf1_probs, sf1_a, sf1_b)
        sf1_losers = xp.where(sf1_rand < sf1_probs, sf1_b, sf1_a)
        sf1_gaps = xp.maximum(seed_arr[sf1_winners.ravel()].reshape(N, 1) - seed_arr[sf1_losers.ravel()].reshape(N, 1), 0)
        outcomes[:, game_idx:game_idx + 1] = sf1_winners
        upset_gaps_arr[:, game_idx:game_idx + 1] = sf1_gaps
        game_idx += 1

        # Semifinal 2: region 2 vs region 3
        sf2_a = region_champs[:, 2:3]
        sf2_b = region_champs[:, 3:4]
        sf2_probs_flat = prob_mat[sf2_a.ravel().get() if GPU_AVAILABLE else sf2_a.ravel(),
                                  sf2_b.ravel().get() if GPU_AVAILABLE else sf2_b.ravel()]
        if GPU_AVAILABLE:
            sf2_probs_flat = xp.asarray(sf2_probs_flat)
        sf2_probs = sf2_probs_flat.reshape(N, 1)
        sf2_rand = all_randoms[:, game_idx:game_idx + 1]
        sf2_winners = xp.where(sf2_rand < sf2_probs, sf2_a, sf2_b)
        sf2_losers = xp.where(sf2_rand < sf2_probs, sf2_b, sf2_a)
        sf2_gaps = xp.maximum(seed_arr[sf2_winners.ravel()].reshape(N, 1) - seed_arr[sf2_losers.ravel()].reshape(N, 1), 0)
        outcomes[:, game_idx:game_idx + 1] = sf2_winners
        upset_gaps_arr[:, game_idx:game_idx + 1] = sf2_gaps
        game_idx += 1

        # Championship
        champ_a = sf1_winners
        champ_b = sf2_winners
        champ_probs_flat = prob_mat[champ_a.ravel().get() if GPU_AVAILABLE else champ_a.ravel(),
                                    champ_b.ravel().get() if GPU_AVAILABLE else champ_b.ravel()]
        if GPU_AVAILABLE:
            champ_probs_flat = xp.asarray(champ_probs_flat)
        champ_probs = champ_probs_flat.reshape(N, 1)
        champ_rand = all_randoms[:, game_idx:game_idx + 1]
        champ_winners = xp.where(champ_rand < champ_probs, champ_a, champ_b)
        champ_losers = xp.where(champ_rand < champ_probs, champ_b, champ_a)
        champ_gaps = xp.maximum(seed_arr[champ_winners.ravel()].reshape(N, 1) - seed_arr[champ_losers.ravel()].reshape(N, 1), 0)
        outcomes[:, game_idx:game_idx + 1] = champ_winners
        upset_gaps_arr[:, game_idx:game_idx + 1] = champ_gaps

        return to_numpy(outcomes), to_numpy(upset_gaps_arr)

    def generate_candidate(self, archetype_name: str, rng: np.random.Generator, ordinal: int) -> BracketCandidate:
        archetype = ARCHETYPE_PROFILES[archetype_name]
        winners: List[int] = []
        favorite_flags: List[int] = []
        region_champions: List[int] = []

        for region in self.regions:
            current = list(self.region_fields[region])
            for round_number in range(1, 5):
                current, round_winners, _, round_favorites = self._play_round(
                    current, rng, round_number, archetype=archetype
                )
                winners.extend(round_winners)
                favorite_flags.extend(round_favorites)
            region_champions.append(current[0])

        semifinal_a, semifinal_winners_a, _, semifinal_favorites_a = self._play_round(
            [region_champions[0], region_champions[1]], rng, 5, archetype=archetype
        )
        semifinal_b, semifinal_winners_b, _, semifinal_favorites_b = self._play_round(
            [region_champions[2], region_champions[3]], rng, 5, archetype=archetype
        )
        winners.extend(semifinal_winners_a + semifinal_winners_b)
        favorite_flags.extend(semifinal_favorites_a + semifinal_favorites_b)

        _, final_winners, _, final_favorites = self._play_round(
            [semifinal_a[0], semifinal_b[0]], rng, 6, archetype=archetype
        )
        winners.extend(final_winners)
        favorite_flags.extend(final_favorites)

        champion_index = final_winners[0]
        final_four_team_ids = [self.team_id_by_index[index] for index in region_champions]
        chalkiness = float(sum(favorite_flags) / len(favorite_flags))
        champion_seed_score = 1.0 - ((int(self.seed_by_index[champion_index]) - 1) / 15.0)
        duplication_proxy = min(1.0, (0.65 * chalkiness) + (0.35 * champion_seed_score))
        pick_digest = hashlib.blake2b(
            ",".join(str(index) for index in winners).encode("utf-8"),
            digest_size=6,
        ).hexdigest()

        return BracketCandidate(
            bracket_id=f"{archetype_name}-{ordinal:03d}-{pick_digest}",
            archetype=archetype_name,
            risk_label=archetype.risk_label,
            pick_indices=[int(index) for index in winners],
            champion_team_id=self.team_id_by_index[champion_index],
            final_four_team_ids=final_four_team_ids,
            chalkiness_proxy=round(chalkiness, 4),
            duplication_proxy=round(duplication_proxy, 4),
        )

    def generate_public_candidate(self, rng: np.random.Generator, ordinal: int) -> BracketCandidate:
        """Generate a bracket that mimics public/ESPN bracket behavior using public_pick_pct."""
        winners: List[int] = []
        favorite_flags: List[int] = []
        region_champions: List[int] = []

        for region in self.regions:
            current = list(self.region_fields[region])
            for round_number in range(1, 5):
                current, round_winners, _, round_favorites = self._play_round(
                    current, rng, round_number, public=True
                )
                winners.extend(round_winners)
                favorite_flags.extend(round_favorites)
            region_champions.append(current[0])

        semifinal_a, semifinal_winners_a, _, semifinal_favorites_a = self._play_round(
            [region_champions[0], region_champions[1]], rng, 5, public=True
        )
        semifinal_b, semifinal_winners_b, _, semifinal_favorites_b = self._play_round(
            [region_champions[2], region_champions[3]], rng, 5, public=True
        )
        winners.extend(semifinal_winners_a + semifinal_winners_b)
        favorite_flags.extend(semifinal_favorites_a + semifinal_favorites_b)

        _, final_winners, _, final_favorites = self._play_round(
            [semifinal_a[0], semifinal_b[0]], rng, 6, public=True
        )
        winners.extend(final_winners)
        favorite_flags.extend(final_favorites)

        champion_index = final_winners[0]
        final_four_team_ids = [self.team_id_by_index[idx] for idx in region_champions]
        chalkiness = float(sum(favorite_flags) / len(favorite_flags))
        champion_seed_score = 1.0 - ((int(self.seed_by_index[champion_index]) - 1) / 15.0)
        duplication_proxy = min(1.0, (0.65 * chalkiness) + (0.35 * champion_seed_score))

        pick_digest = hashlib.blake2b(
            ",".join(str(i) for i in winners).encode("utf-8"),
            digest_size=6,
        ).hexdigest()

        return BracketCandidate(
            bracket_id=f"public_field-{ordinal:03d}-{pick_digest}",
            archetype="public_field",
            risk_label="public",
            pick_indices=[int(i) for i in winners],
            champion_team_id=self.team_id_by_index[champion_index],
            final_four_team_ids=final_four_team_ids,
            chalkiness_proxy=round(chalkiness, 4),
            duplication_proxy=round(duplication_proxy, 4),
        )

    def team_name(self, team_id: str) -> str:
        return self.team_name_by_index[self.index_by_team_id[team_id]]


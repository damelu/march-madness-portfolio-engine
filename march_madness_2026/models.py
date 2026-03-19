from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


PoolBucket = Literal["small", "mid", "large"]
TiebreakMode = Literal["shared_win", "fractional"]
CalibrationMethod = Literal["none", "isotonic", "platt"]
UncertaintyMode = Literal["none", "bootstrap", "deep_ensemble"]
PayoutTieSplitMode = Literal["proportional", "shared_win", "fractional"]


class TeamSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: str
    team_name: str
    region: str
    seed: int = Field(ge=1, le=16)
    rating: float
    market_adjustment: float = 0.0
    coaching_adjustment: float = 0.0
    continuity_adjustment: float = 0.0
    injury_penalty: float = 0.0
    volatility: float = Field(default=0.10, ge=0.0, le=0.45)
    offense_rating: float = 0.0
    defense_rating: float = 0.0
    tempo_adjustment: float = 0.0
    # V6: matchup-specific features
    three_point_rate: float = 0.0       # 3PA/FGA ratio deviation from mean (high = 3PT dependent)
    free_throw_rate: float = 0.0        # FTA/FGA ratio deviation (high = gets to the line)
    # V6: external model probabilities for ensemble
    bpi_championship_pct: float = 0.0   # ESPN BPI championship probability
    bpi_final_four_pct: float = 0.0     # ESPN BPI Final Four probability
    public_pick_pct: float = 0.0        # % of public brackets picking as champion
    round1_market_win_prob: float = 0.0
    avg_round1_spread: float = 0.0
    # V10: additive four-factor and tournament-context fields
    adj_efg_off: float = 0.0
    adj_efg_def: float = 0.0
    orb_rate_off: float = 0.0
    drb_rate_def: float = 0.0
    turnover_rate_off: float = 0.0
    turnover_rate_def: float = 0.0
    ft_rate_off: float = 0.0
    ft_rate_def: float = 0.0
    rim_rate_off: float = 0.0
    rim_rate_def: float = 0.0
    three_rate_off: float = 0.0
    three_rate_def: float = 0.0
    bench_depth_score: float = 0.0
    lead_guard_continuity: float = 0.0
    experience_score: float = 0.0
    late_season_volatility: float = 0.0
    injury_uncertainty: float = 0.0
    lineup_uncertainty: float = 0.0
    travel_miles_round1: float = 0.0
    timezone_shift_round1: float = 0.0
    altitude_adjustment: float = 0.0
    venue_familiarity: float = 0.0
    region_strength_index: float = 0.0
    seed_misprice: float = 0.0


class SelectionSundayDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    season: int = 2026
    tournament: str = "mens_d1"
    teams: List[TeamSnapshot]
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("teams")
    @classmethod
    def validate_teams(cls, value: List[TeamSnapshot]) -> List[TeamSnapshot]:
        team_count = len(value)
        if team_count < 8 or team_count % 2 != 0 or (team_count & (team_count - 1)) != 0:
            raise ValueError("selection data must contain a power-of-two team count")
        if len({team.team_id for team in value}) != team_count:
            raise ValueError("team_id values must be unique")
        return value


class ScoringProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_id: str
    name: str
    round_weights: List[float] = Field(min_length=6, max_length=6)
    upset_bonus_per_seed: float = 0.0
    tiebreak_mode: TiebreakMode = "shared_win"


class ContestProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    contest_id: str
    name: str
    scoring_profile: str
    pool_bucket: PoolBucket
    simulated_field_size: int = Field(gt=1)
    entries_submitted: int = Field(gt=0)
    opponent_mix: Dict[str, float]
    payout_profile: Optional[str] = None
    public_source: Optional[str] = None
    field_behavior_profile: Optional[str] = None

    @field_validator("opponent_mix")
    @classmethod
    def validate_opponent_mix(cls, value: Dict[str, float]) -> Dict[str, float]:
        total = sum(value.values())
        if total <= 0:
            raise ValueError("opponent_mix must have positive weight")
        return {key: weight / total for key, weight in value.items()}


class SimulationProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int
    num_tournament_simulations: int = Field(gt=0)
    num_candidate_brackets: int = Field(gt=0)
    portfolio_size: int = Field(gt=0)
    max_per_archetype: int = Field(gt=0)
    min_distinct_archetypes: int = Field(gt=0)
    overlap_penalty_weight: float = Field(ge=0.0)
    champion_penalty_weight: float = Field(ge=0.0)
    archetype_mix: Dict[str, float]
    primary_contests: Dict[str, float]
    sensitivity_contests: List[str] = Field(default_factory=list)

    @field_validator("archetype_mix")
    @classmethod
    def validate_archetype_mix(cls, value: Dict[str, float]) -> Dict[str, float]:
        total = sum(value.values())
        if total <= 0:
            raise ValueError("archetype_mix must have positive weight")
        return {key: weight / total for key, weight in value.items()}

    @field_validator("primary_contests")
    @classmethod
    def validate_primary_contests(cls, value: Dict[str, float]) -> Dict[str, float]:
        total = sum(value.values())
        if total <= 0:
            raise ValueError("primary_contests must have positive weight")
        return {key: weight / total for key, weight in value.items()}


class TrainingProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_id: str
    model_id: str
    seed: int
    train_seasons: List[int] = Field(default_factory=list)
    validation_seasons: List[int] = Field(default_factory=list)
    holdout_seasons: List[int] = Field(default_factory=list)
    feature_sets: List[str] = Field(default_factory=list)
    backend: str = "numpy"
    device: str = "cpu"
    batch_size: int = Field(default=1024, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_epochs: int = Field(default=1, gt=0)
    early_stopping_patience: int = Field(default=0, ge=0)
    ensemble_size: int = Field(default=1, gt=0)
    bootstrap_replicates: int = Field(default=0, ge=0)
    calibration_method: CalibrationMethod = "none"
    uncertainty_mode: UncertaintyMode = "none"


class ContestPayoutProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_id: str
    name: str
    entry_fee: float = Field(default=0.0, ge=0.0)
    payout_curve: Dict[int, float]
    tie_split_mode: PayoutTieSplitMode = "proportional"
    top_heavy_weight: float = Field(default=1.0, ge=0.0)

    @field_validator("payout_curve")
    @classmethod
    def validate_payout_curve(cls, value: Dict[int, float]) -> Dict[int, float]:
        if not value:
            raise ValueError("payout_curve must not be empty")
        normalized: Dict[int, float] = {}
        for rank, payout in value.items():
            rank_int = int(rank)
            if rank_int <= 0:
                raise ValueError("payout_curve ranks must be positive")
            if payout < 0:
                raise ValueError("payout_curve values must be non-negative")
            normalized[rank_int] = float(payout)
        return dict(sorted(normalized.items()))


class HistoricalGameRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    season: int
    game_date: str
    tournament_round: str
    team_a_id: str
    team_b_id: str
    team_a_seed: int = Field(ge=1, le=16)
    team_b_seed: int = Field(ge=1, le=16)
    team_a_features: Dict[str, float] = Field(default_factory=dict)
    team_b_features: Dict[str, float] = Field(default_factory=dict)
    context_features: Dict[str, float] = Field(default_factory=dict)
    public_features: Dict[str, float] = Field(default_factory=dict)
    result_team_a_win: bool
    sample_weight: float = Field(default=1.0, gt=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoricalTournamentRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    season: int
    tournament: str = "mens_d1"
    teams: List[TeamSnapshot]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PublicRoundPickProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: str
    source: str = "unknown"
    seed: Optional[int] = Field(default=None, ge=1, le=16)
    round_probabilities: Dict[str, float] = Field(default_factory=dict)
    championship_pick_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    path_duplication_rate: float = Field(default=0.0, ge=0.0)


class TeamPosterior(BaseModel):
    model_config = ConfigDict(extra="ignore")

    team_id: str
    mean_rating: float = 0.0
    rating_std: float = Field(default=0.0, ge=0.0)
    round_advancement_probabilities: Dict[str, float] = Field(default_factory=dict)
    championship_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    final_four_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    injury_uncertainty: float = Field(default=0.0, ge=0.0)
    lineup_uncertainty: float = Field(default=0.0, ge=0.0)


class ModelArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str
    model_id: str
    created_at: str
    training_profile_id: Optional[str] = None
    train_seasons: List[int] = Field(default_factory=list)
    validation_seasons: List[int] = Field(default_factory=list)
    holdout_seasons: List[int] = Field(default_factory=list)
    backend: str = "unknown"
    calibration_method: CalibrationMethod = "none"
    uncertainty_mode: UncertaintyMode = "none"
    feature_sets: List[str] = Field(default_factory=list)
    artifact_path: Optional[str] = None
    notes: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class CandidateContestMetrics:
    contest_id: str
    first_place_equity: float
    average_finish: float
    top_decile_rate: float


@dataclass
class BracketCandidate:
    bracket_id: str
    archetype: str
    risk_label: str
    pick_indices: List[int]
    champion_team_id: str
    final_four_team_ids: List[str]
    chalkiness_proxy: float
    duplication_proxy: float
    weighted_first_place_equity: float = 0.0
    weighted_average_finish: float = 0.0
    contest_metrics: Dict[str, CandidateContestMetrics] = field(default_factory=dict)
    why_selected: Optional[str] = None
    scenario_fit: Optional[str] = None


@dataclass
class PortfolioSummary:
    weighted_portfolio_capture_rate: float
    naive_baseline_capture_rate: float
    weighted_portfolio_objective: float
    naive_baseline_objective: float
    weighted_candidate_baseline: float
    average_pairwise_overlap: float
    distinct_archetypes: int
    unique_champions: int
    weighted_portfolio_first_place_equity: float = 0.0
    naive_baseline_first_place_equity: float = 0.0
    weighted_portfolio_expected_payout: float = 0.0
    naive_baseline_expected_payout: float = 0.0
    weighted_portfolio_cash_rate: float = 0.0
    weighted_portfolio_top3_equity: float = 0.0
    payoff_correlation_score: float = 0.0


@dataclass
class CalibrationBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    mean_prediction: float = 0.0
    observed_rate: float = 0.0
    sample_count: int = 0
    weight: float = 0.0
    absolute_gap: float = 0.0
    round_name: Optional[str] = None


@dataclass
class CalibrationReport:
    model_id: str
    sample_count: int
    log_loss: float
    brier_score: float
    expected_calibration_error: float
    bins: List[CalibrationBin] = field(default_factory=list)
    by_round: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    brier_reliability: float = 0.0
    brier_resolution: float = 0.0
    brier_uncertainty: float = 0.0


@dataclass
class BacktestSeasonResult:
    season: int
    sample_count: int = 0
    log_loss: float = 0.0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    expected_payout: float = 0.0
    cash_rate: float = 0.0
    top1_equity: float = 0.0
    top3_equity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestSummary:
    season_results: List[BacktestSeasonResult] = field(default_factory=list)
    mean_log_loss: float = 0.0
    mean_brier_score: float = 0.0
    mean_expected_calibration_error: float = 0.0
    mean_expected_payout: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioPayoutSummary:
    expected_payout: float = 0.0
    cash_rate: float = 0.0
    top1_equity: float = 0.0
    top3_equity: float = 0.0
    top5_equity: float = 0.0
    downside_risk: float = 0.0
    payoff_correlation: float = 0.0
    expected_utility: float = 0.0


@dataclass
class PortfolioResult:
    run_id: str
    data_source: str
    candidate_count: int
    finalists: List[BracketCandidate]
    summary: PortfolioSummary
    scenario_summary: Dict[str, Dict[str, float]]
    sensitivity_summary: Dict[str, Dict[str, float]]
    dashboard_spec_sections: List[str]

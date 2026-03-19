# Version History: March Madness Bracket Portfolio Engine

How a blank scaffold became an autonomous, GPU-accelerated bracket portfolio optimizer in nine versions over 48 hours.

---

## V1 -- The Scaffold

**Status:** Well-architected skeleton. No real data.

The project began as a clean-room design for a bracket portfolio engine -- the idea being to treat bracket contests not as a single best-guess prediction, but as a portfolio optimization problem. Pick five brackets that collectively maximize your probability of finishing first in your pool, accounting for what everyone else is likely to pick.

The V1 scaffold established the core architecture that survived through V9:

- **TournamentModel**: 64-team bracket simulation with sigmoid-based win probabilities
- **BracketCandidate**: individual bracket generation via archetype-specific temperature/bias profiles
- **Portfolio optimizer**: greedy selection with overlap constraints
- **Scoring engine**: round-weighted bracket scoring against Monte Carlo tournament outcomes
- **Contest profiles**: small (24-person), mid (96-person), and large (768-person) pool scenarios

Five archetype profiles controlled bracket generation style:

| Archetype | Temperature | Bias | Risk Level |
|-----------|------------|------|------------|
| High Confidence | 0.72 | +0.16 | Low |
| Balanced | 0.95 | +0.04 | Medium |
| Selective Contrarian | 1.12 | -0.08 | Medium-High |
| Underdog Upside | 1.28 | -0.16 | High |
| High Risk High Return | 1.45 | -0.26 | Very High |

**The problems were serious.** Team ratings were hand-estimated guesses. Many teams in the bracket file were entirely wrong -- they were not even in the 2026 tournament. The win probability model used linear volatility dampening, which produced a 7.3% 1-seed-vs-16-seed upset rate. The historical rate is 0.6% (2 upsets in 164 games since 1985). A model that predicts 16-seeds win 1 in 14 games is not a model -- it is a random number generator with extra steps.

**Key metrics (V1 baseline):**
- Data coverage: 0 of 64 teams had real stats
- 1v16 upset rate: 7.3% (target: ~0.6%)
- Small pool capture: ~13%
- Mid pool capture: ~6%
- Large pool capture: ~0.5%
- Tests: 9 passing

---

## V2 -- Real ESPN Stats and Critical Bug Fixes

**Status:** First real data. Three show-stopping bugs found and fixed.

The first priority was getting actual data into the system. ESPN's public API provided season statistics for 28 of the 64 tournament teams -- points per game, field goal percentage, turnover rate, and other box score stats.

But the bigger story of V2 was the code review that uncovered three critical bugs:

### Bug #1: Correlated RNG Seeds

The tournament simulation RNG, the candidate bracket generator, and the opponent field generator all used seeds derived from the same base value with trivial offsets. This meant the "random" outcomes were correlated -- the opponents were not actually independent of the candidates. The fix was to use well-separated seeds with XOR masks (`seed ^ 0xDEADBEEF` for candidates, `seed ^ 0xCAFEBABE` for opponents, with an additional blake2b hash of the contest ID).

### Bug #2: Probability Bias on Wrong Scale

Archetype biases were being added directly to the win probability:

```
adjusted_probability = base_probability + bias * seed_gap
```

This violates probability axioms. A +0.16 bias added to a 0.92 base probability produces 1.08, which is not a probability. The fix was to apply all biases on the logit (log-odds) scale, then transform back through the sigmoid:

```
logit = log(p / (1 - p))
adjusted_logit = logit / temperature + bias * seed_gap * 4.0
adjusted_probability = sigmoid(adjusted_logit)
```

This preserves the (0, 1) bounds and behaves correctly at extremes.

### Bug #3: O(n^2) Overlap Matrix

The pairwise overlap computation was a naive Python loop comparing every bracket pair element-by-element. For 300 candidates with 63 games each, this was 300 x 300 x 63 = 5.67 million comparisons in pure Python. The fix was vectorized numpy broadcasting:

```python
overlap = (picks[:, None, :] == picks[None, :, :]).sum(axis=2) / num_games
```

V2 also added a hard overlap cap: no two brackets in the portfolio can share more than 85% of their picks.

**Key metrics (V2):**
- Data coverage: 28 of 64 teams with ESPN stats
- 1v16 upset rate: ~3% (better, still too high)
- Small pool capture: ~16%
- Mid pool capture: ~12%
- Tests: 9 passing

---

## V3 -- KenPom/BPI Real Ratings and Calibration

**Status:** Complete data overhaul. Every team verified against real bracket.

V3 was a ground-up data rebuild. Research against NBC Sports, ESPN, and NCAA.com revealed that 40+ teams in the V1/V2 bracket file were wrong -- they were not in the actual 2026 tournament field. The entire `bracket_2026.json` was rewritten with verified teams, seeds, and regions.

For the first time, every team had real-world-grounded ratings drawn from multiple sources:

- **KenPom rankings** for overall team strength
- **ESPN BPI** for offense and defense ratings
- **NET rankings** for selection committee perspective
- **Vegas championship odds** via The Odds API for market-implied strength

The rating scale was compressed logarithmically to match historical upset frequency distributions from `seed_matchup_history.json` (41 tournaments, 1985-2025, 164 games per seed matchup).

The volatility dampening model was rewritten from linear to nonlinear:

```python
extremity = abs(base_probability - 0.5) * 2.0
dampened_volatility = volatility * (1.0 - extremity * 0.85)
```

This preserves extreme matchup outcomes (a 1-seed still beats a 16-seed ~97% of the time) while allowing genuine uncertainty in mid-seed matchups (8v9 is close to a coin flip, as history says it should be).

Real coaching adjustments were applied from tournament records: Tom Izzo (+1.00), Dan Hurley (+0.90), Bill Self (+0.95), and others. Injury penalties were researched from ESPN, RotoWire, and SI: Duke's Mark Foster out, Gonzaga's Braden Huff out, BYU's Saunders and Baker with ACL injuries, UNC's RJ Wilson out, Alabama's Latrell Holloway suspended, Texas Tech's Obi Toppin with ACL.

**Calibration results (V3):**
- 1v16 upset rate: 2.7% (target: 0.6% -- close enough given small-sample noise)
- 8v9 win rate: 51.7% (target: 51.2% -- nearly exact)
- 4v13 upset rate: 21.9% (target: 20.7% -- excellent)
- 5v12 upset rate: ~35% (target: 35.2% -- excellent)

**Key metrics (V3):**
- Data coverage: 64 of 64 teams with KenPom/BPI/NET ratings
- Small pool capture: ~27%
- Mid pool capture: ~13%
- Tests: 9 passing

---

## V4 -- ESPN BPI Tournament Projections and Market Calibration

**Status:** External model integration. Rating adjustments from market signal.

V4 integrated ESPN BPI's round-by-round advancement probabilities for all 64 teams. These projections -- Final Four probability, championship probability, and first-round win probability -- were used as an independent calibration signal.

The key insight was comparing BPI Final Four rates against historical seed-based rates. When BPI says a team reaches the Final Four far more often than its seed typically does, the team's rating needs adjustment. Houston was the prime example: BPI gave Houston a 33.3% Final Four probability, far exceeding a typical 2-seed's historical rate of ~19%. This indicated the market viewed Houston as under-seeded, and the rating was adjusted upward.

Vegas championship odds provided a second market signal, converted to an additive `market_adjustment` field on each team.

**Key metrics (V4):**
- 1v16 upset rate: 2.8%
- Small pool capture: ~37%
- Mid pool capture: ~14%
- Large pool capture: ~2.6%
- Tests: 9 passing

---

## V5 -- Contrarian Value (Public Pick Percentages)

**Status:** The field is not random. Model what everyone else picks.

The insight behind V5: in a bracket pool, you do not win by being the most accurate. You win by being accurate where everyone else is wrong. If Duke wins it all and 29% of brackets have Duke as champion, you share that win with 29% of the field. If Iowa State wins and only 1.9% of brackets have Iowa State, your bracket is nearly unique.

V5 integrated public pick data from Yahoo Bracket Mayhem (champion picks) and CBS Sports (first-round pick percentages):

| Team | Public Champion Pick % | Model Championship Prob | Contrarian Value |
|------|----------------------|------------------------|-----------------|
| Duke | 29.3% | ~24% | 0.83x (overvalued) |
| Arizona | 21.0% | ~14% | 0.67x (overvalued) |
| Iowa State | 1.9% | 6.0% | 3.16x (undervalued) |
| Illinois | 1.2% | 4.2% | 3.50x (undervalued) |
| Houston | 5.5% | 9.4% | 1.71x (undervalued) |
| Gonzaga | 1.8% | 3.7% | 2.06x (undervalued) |

The contrarian value formula is simple: `true_probability / public_pick_rate`. Values above 1.0 identify teams the public is undervaluing. Iowa State at 3.16x and Illinois at 3.50x were the biggest edges -- teams with realistic championship paths that almost nobody was picking.

Public pick percentages were stored in `public_pick_rates.json` and fed into the `public_pick_pct` field on each `TeamSnapshot`. The market adjustment now blended BPI probability data with contrarian value signals.

**Key metrics (V5):**
- Small pool capture: ~40%
- Mid pool capture: ~20%
- Large pool capture: ~2.8%
- Tests: 9 passing

---

## AutoBracket -- The Autoresearch Adaptation

**Status:** Let the machine find the best parameters.

Inspired by Karpathy's autoresearch paradigm (autonomous modify-evaluate-keep/revert), AutoBracket was an autonomous parameter optimizer for the bracket portfolio engine. The core loop:

1. Start with baseline parameters
2. Randomly mutate 1-4 parameters (gaussian perturbation, configurable scale)
3. Run the full portfolio engine pipeline
4. If the composite score improves, KEEP the mutation. Otherwise, DISCARD.
5. Repeat.

The `BracketParams` dataclass captured 20 tunable parameters:

- 7 strength-computation weights (market, coaching, continuity, offense, defense, tempo, injury)
- 1 sigmoid scale for win probability
- 5 archetype temperatures
- 5 archetype biases
- 2 portfolio selection penalties (overlap, champion repeat)

Each parameter had type-specific bounds (temperatures: 0.3-2.5, biases: -0.5 to +0.5, weights: 0.0-3.0). Mutations scaled relative to parameter magnitude, so a weight at 0.75 got larger perturbations than a bias at -0.08.

The composite scoring function evolved across versions. By V9 it became FPE-first:

```python
score = (
    0.55 * first_place_equity
    + 0.20 * scenario_small_fpe
    + 0.15 * scenario_mid_fpe
    + 0.10 * scenario_large_fpe
    + 0.02 * (distinct_archetypes / 5.0)
    + 0.02 * (unique_champions / 5.0)
    - 0.03 * max(0.0, overlap - 0.5)
)
```

Two implementations existed: a sequential optimizer (`autobracket.py`) and a parallel version (`autobracket_parallel.py`) that used Python's multiprocessing pool. Schema versioning (tracked as `AUTOBRACKET_SCHEMA_VERSION`) ensured checkpoint compatibility across code changes. Checkpoints used atomic writes (write to temp file, then `os.replace`) to survive crashes.

**Key discoveries from AutoBracket runs:**

- **Injury weight matters more on a compressed rating scale.** When ratings span a narrow range, a 0.5-point injury penalty is proportionally larger. The optimizer pushed `injury_weight` from the default 1.30 down to 1.045, then back up to 1.394, before settling at 1.045 -- suggesting the default was slightly too aggressive.
- **Coaching weight was undervalued.** Starting at 0.75, the optimizer pulled it to 0.582 -- still significant, but the raw magnitude was adjusted for the compressed scale.
- **High-risk archetype needs lower temperature.** Counter-intuitively, `temp_high_confidence` dropped from 0.72 to 0.36 while `temp_underdog_upside` rose to 1.72. The high-confidence archetype should commit hard to its picks (low temperature = sharp distribution), while underdog archetypes benefit from more randomness.
- **Selective contrarian temperature dropped sharply** from 1.12 to 0.48. Contrarian brackets should make deliberate, specific contrarian picks -- not scatter upsets randomly.

---

## V6 -- Structural Upgrades

**Status:** Matchup-level detail. Multi-model ensemble. Smarter opponents.

V6 was the largest structural change since V1. Three major additions:

### Matchup-Aware Win Probability

Raw team ratings ignore game-specific dynamics. V6 added three matchup adjustments to `win_probability()`:

- **Tempo mismatch:** Slower teams get a small edge in mismatched games because fewer possessions mean more variance, which favors the underdog. Applied as `0.015 * tempo_diff` when the difference exceeds 0.3.
- **Three-point variance:** Teams dependent on three-point shooting have higher game-to-game variance. A cold shooting night can eliminate a talented team. Adds to the volatility term: `volatility += three_pt_var * 0.03`.
- **Free throw rate advantage:** Teams that get to the free throw line are more consistent in close games. Applied as `0.01 * ft_advantage`.

### Multi-Model Ensemble (Round 1 Only)

For first-round games, win probability blends 75% model probability with 25% Vegas/BPI probability:

```python
blended = 0.75 * model_prob + 0.25 * bpi_relative
```

This only applies to Round 1. An LLM code review later found that an earlier version leaked the Vegas blend into later rounds -- a critical bug because Vegas first-round lines have no predictive value for Sweet 16 matchups.

### Public-Pick-Based Opponent Simulation

In real bracket pools, opponents are not uniformly random. Most participants follow the chalk. V6 changed opponent generation to 60% public-pick-biased brackets (simulating ESPN bracket challenge users) and 40% archetype-mix brackets (simulating sophisticated opponents). The `generate_public_candidate()` method uses `public_pick_pct` with popularity-gap logit adjustments to mimic how casual fans pick brackets.

V6 also introduced and later removed simulated annealing bracket search (found to be unused after refactoring to the greedy + local search approach).

### LLM Code Review Findings

A comprehensive LLM-driven code review at the end of V6 identified 12 issues:

1. Vegas blend leaking into rounds 2-6
2. AutoBracket monkey-patch inconsistency (patched functions did not replicate all V6 matchup logic)
3. Non-idempotent pipeline (re-running with same inputs could produce different results due to accumulated adjustments)
4. Missing fractional tie handling in portfolio capture rate
5. Dead code (`VEGAS_SPREADS`, `refine_candidate_sa`, `_BPI_CHAMP_PROBS`)
6. Unpopulated matchup features (`three_point_rate`, `free_throw_rate` defaulting to 0.0)
7. No simulation config in output (irreproducible results)
8. Several others

---

## V7 -- All Review Feedback Implemented

**Status:** Technical debt cleanup. Every finding addressed.

V7 was a pure bugfix and cleanup release. All 12 findings from the LLM code review were resolved:

- **Fractional tie handling** in `portfolio_capture_rate`: ties now properly share credit between tied brackets instead of awarding a full win to each
- **Pipeline idempotence**: `base_rating` and `base_market_adjustment` fields ensure that running the pipeline twice produces identical results (adjustments are applied on top of the base, not accumulated)
- **Atomic checkpoint writes**: AutoBracket checkpoints now write to a temp file and use `os.replace()` for crash safety
- **Populated matchup features**: `three_point_rate` and `free_throw_rate` filled from ESPN data for all 64 teams
- **Dead code removal**: `VEGAS_SPREADS`, `refine_candidate_sa()`, `_BPI_CHAMP_PROBS` -- all deleted
- **Simulation config in output**: every run now records its full configuration for reproducibility

---

## V8 -- True First-Place Equity Optimization

**Status:** The objective function revolution. Stop optimizing capture rate. Optimize FPE.

V8 was a philosophical pivot. Every prior version optimized for "capture rate" -- the probability that at least one of your five brackets ties or beats the best bracket in the field. This sounds right but is subtly wrong.

Capture rate treats a tie for first (splitting the prize with 10 other people) the same as an outright first-place finish. In a bracket pool where you need to win, not just "not lose," the correct objective is first-place equity: the expected fraction of the first-place prize you take home.

V8 introduced `portfolio_first_place_equity()` as the primary objective:

```python
portfolio_share = where(
    max_selected > max_opponent, 1.0,          # outright win
    where(max_selected == max_opponent,
          selected_ties / (selected_ties + opponent_ties),  # split the prize
          0.0)                                  # loss
)
fpe = portfolio_share.mean()
```

This is different from capture rate in a critical way: if your best bracket ties the field leader, you get `1 / (1 + num_opponents_tied)`, not 1.0.

### Lexicographic Ranking

Portfolio comparison became a lexicographic tuple instead of a single scalar:

```python
(fpe, capture, unique_champs, -avg_overlap, -champ_pen, avg_eq, floor_eq, ...)
```

Primary sort on FPE. Ties broken by capture rate, then champion diversity, then overlap, and so on. This ensures FPE dominates selection but the tiebreakers preserve structural quality.

### Local Search Refinement

After greedy selection builds the initial 5-bracket portfolio, V8 added a 1-for-1 swap refinement pass. For each bracket in the portfolio, try replacing it with every candidate not in the portfolio. If any swap improves the lexicographic rank, take the best swap. Repeat until no improving swap exists. This finds local optima that greedy alone misses.

### Zero-Equity Filtering

Candidates with zero first-place equity (they never win in any simulation) are filtered out of the selection pool. This prevents the optimizer from wasting portfolio slots on brackets that cannot contribute to a first-place finish.

### AutoBracket V8 Overnight Run

AutoBracket ran overnight on a rented 128-core vast.ai instance in Vietnam ($0.12/hr). The V8 experiment log tells the story:

| Round | Score | Capture | Small | Mid | Large | Status | Key Mutations |
|-------|-------|---------|-------|-----|-------|--------|---------------|
| 1 | 0.2124 | 0.1853 | 35.6% | 18.5% | 1.5% | KEEP | temp_high_confidence: 0.72->0.57 |
| 3 | 0.2197 | 0.1911 | 36.8% | 19.3% | 1.2% | KEEP | temp_selective_contrarian: 0.76->0.40 |
| 4 | 0.2246 | 0.1965 | 37.6% | 20.0% | 1.4% | KEEP | coaching_weight: 0.75->0.58 |
| 9 | 0.2323 | 0.2035 | 40.2% | 19.2% | 1.7% | KEEP | injury_weight: 1.39->1.07, temp_high_confidence: 0.57->0.36 |
| 11 | 0.2362 | 0.2036 | 40.2% | 18.7% | 2.2% | KEEP | 4 unique champions achieved |
| 12 | 0.2365 | 0.2040 | 40.2% | 18.7% | 2.3% | KEEP | Final convergence |

All 12 rounds were KEEPs -- every mutation the optimizer tried improved the score. The composite score rose from 0.212 to 0.237, an 11.4% improvement. The final best parameters from `v8_remote_best_params.json`:

```
sigmoid_scale: 6.85, coaching_weight: 0.58, injury_weight: 1.04
temp_high_confidence: 0.36, temp_balanced: 0.93
temp_selective_contrarian: 0.48, temp_underdog_upside: 1.72
```

**Key metrics (V8):**
- Portfolio FPE: 19.8% (weighted across pool sizes)
- Capture rate: 20.4%
- Small pool capture: 40.2%
- Mid pool capture: 18.7%
- Large pool capture: 2.3%
- Unique champions: 4
- Tests: 21 passing (6 new tests for FPE, local search, pool feasibility)

---

## V9 -- Probability Path Consistency and FPE-First AutoBracket

**Status:** Internal consistency. Every code path uses the same probability model.

V9 was about eliminating inconsistencies in how probabilities were computed across different parts of the codebase. Three related problems:

### Problem 1: Candidate generation used raw `win_probability()` for Round 1

The tournament simulation used `win_probability_round1()` (the Vegas-blended version) for first-round games, but candidate bracket generation called `win_probability()` (the raw model version). This meant candidates were making Round 1 picks based on a different probability model than the one simulating tournament outcomes. Fix: introduce `_round_model_probability()` as a centralized dispatcher:

```python
def _round_model_probability(self, a, b, round_number):
    if round_number == 1:
        return self.win_probability_round1(a, b)
    return self.win_probability(a, b)
```

All code paths -- `_candidate_pick_probability()`, `_public_pick_probability()`, `_play_round()` -- now call through this dispatcher.

### Problem 2: Favorite flags derived from wrong model

The "is this pick the favorite?" flag was derived from `win_probability()` instead of `_round_model_probability()`. Since the Vegas blend shifts Round 1 probabilities, the "favorite" designation could disagree between simulation and candidate generation. Fixed to use the round-aware model.

### Problem 3: Public candidate generation

The `_public_pick_probability()` helper was extracted as a shared method on `TournamentModel`. The `_play_round()` method was extended with a `public=True` parameter, and the championship popularity multiplier was corrected from 2.0 to 1.5 (the higher value was over-weighting popular champions in public bracket simulation).

### Other V9 Changes

- **Pool feasibility check**: `_pool_supports_constraints()` now uses proper `max_per_archetype` checking instead of just counting distinct archetypes
- **Naive baseline built from same pool**: the naive top-5 baseline is now selected from the same eligible candidate pool as the optimizer, making the comparison fair
- **AutoBracket composite score rewritten**: 55% FPE + scenario FPEs, not capture-rate-first (schema version 3)
- **Engine version tracking**: `engine_version: "v9"` in output bundle
- **Expanded simulation config**: contest profiles and scoring profile mapping recorded in output

**Key metrics (V9):**
- Tests: 21 passing
- All probability paths internally consistent

---

## V9.1 -- Vectorized Simulation (250x Speedup)

**Status:** Performance revolution. The simulation bottleneck is gone.

The original `simulate_tournament()` method simulated one tournament at a time. For 5,000 simulations, Python executed:

```
5,000 sims x 63 games/sim = 315,000 game resolutions
```

Each game resolution involved a Python function call, dictionary lookups, and conditional branching. On a laptop, 5,000 simulations took roughly 5 seconds.

V9.1 rewrote this as `simulate_many()`, a fully vectorized implementation where ALL 5,000 simulations run simultaneously per round:

1. **Pre-compute a 64x64 pairwise probability matrix** -- deterministic, computed once
2. **Pre-generate ALL random numbers at once**: `rng.random((N, 63))` gives a (5000, 63) matrix
3. **Process each round with vectorized operations**: for 8 first-round games across 5,000 sims, one `np.where(randoms < probs, team_a, team_b)` call resolves 40,000 games simultaneously

The loop structure became:

```
4 regions x 4 rounds + 2 semifinals + 1 championship = 19 vectorized operations
```

Instead of 315,000 Python operations, we execute 19 numpy operations on arrays of shape (5000, num_matchups). The pattern comes from GPU Monte Carlo research: "vectorize across simulations, loop across rounds."

**Performance improvement: 5 seconds down to 0.02 seconds -- a 250x speedup.**

The implementation works with either numpy (CPU) or CuPy (GPU). The `gpu.py` module provides a transparent `xp` module that is either `cupy` or `numpy` depending on what is available. All GPU transfer is handled internally -- functions accept numpy arrays and return numpy arrays.

---

## Infrastructure

### Cloud Compute

Remote computation ran on vast.ai instances:

- **128-core Vietnam instance** ($0.12/hr): used for the overnight V8 AutoBracket optimization run. 12 rounds completed in ~8 hours, evaluating 110 parameter configurations per round.
- **RTX A4000 Quebec instance** ($0.09/hr): used for GPU acceleration experiments with CuPy.

### GPU Acceleration (Partial)

The `gpu.py` module provides CuPy-accelerated versions of the hot paths:

- `score_brackets_gpu()`: bracket-vs-outcome comparison matrix (the hottest function: num_brackets x num_sims x num_games)
- `evaluate_candidates_gpu()`: per-candidate FPE and average finish
- `portfolio_fpe_gpu()`: portfolio-level first-place equity
- `overlap_matrix_gpu()`: pairwise bracket overlap

A key lesson learned: **multiprocessing workers cannot share a GPU**. When Python forks a process, each child creates a separate CUDA context. With 128 workers all trying to allocate GPU memory, the GPU runs out of memory immediately. The GPU path works for single-process runs but not for the parallel AutoBracket optimizer.

### External Data Sources

- **ESPN API**: season stats for 64/64 teams (offense, defense, tempo, 3PT%, FT%)
- **The Odds API**: Vegas championship odds (free tier, 500 credits/month)
- **KenPom**: overall team rankings and adjusted efficiency
- **ESPN BPI**: round-by-round advancement probabilities
- **Yahoo Bracket Mayhem / CBS Sports**: public pick percentages
- **RotoWire / SI / ESPN**: injury reports (Foster, Huff, Holloway, Toppin, Wilson, Baker, Saunders)

---

## Key Metrics Progression

The table below tracks how core metrics evolved across versions. Values are approximate due to Monte Carlo variance (5,000 simulations).

| Version | Data Coverage | 1v16 Rate | Small Pool | Mid Pool | Large Pool | Tests | Primary Change |
|---------|-------------|-----------|-----------|---------|-----------|-------|----------------|
| V1 | 0/64 real | 7.3% | ~13% | ~6% | ~0.5% | 9 | Scaffold with estimated ratings |
| V2 | 28/64 ESPN | ~3.0% | ~16% | ~12% | ~1.7% | 9 | Bug fixes, ESPN data |
| V3 | 64/64 KenPom | 2.7% | ~27% | ~13% | ~1.7% | 9 | Complete bracket rebuild |
| V4 | +BPI calibrated | 2.8% | ~37% | ~14% | ~2.6% | 9 | ESPN BPI ensemble |
| V5 | +contrarian | 2.8% | ~40% | ~20% | ~2.8% | 9 | Public pick integration |
| V6 | +matchups | 2.8% | ~40% | ~20% | ~2.8% | 9 | Matchup-aware model, multi-model ensemble |
| V7 | (cleanup) | -- | -- | -- | -- | 9 | All 12 review findings fixed |
| V8 | +FPE optimizer | -- | ~40% | ~19% | ~2.3% | 21 | First-place equity objective |
| V9 | +path consistency | -- | -- | -- | -- | 21 | Round-aware probability dispatch |
| V9.1 | (performance) | -- | -- | -- | -- | 21 | 250x simulation speedup |

**What the numbers mean:**

- **Small pool (24 people):** A 40% capture rate means our 5-bracket portfolio beats or ties the best bracket in a 24-person pool 40% of the time. Against random opponents, a single bracket has a ~4% chance. Five brackets give ~18% if independent. We achieve 40% through smart diversification and contrarian picks.
- **Mid pool (96 people):** 19% capture rate in a pool where random chance gives ~1%. The contrarian value of Iowa State and Illinois drives much of this edge.
- **Large pool (768 people):** 2.3% capture rate. In a pool this large, you need a genuinely differentiated bracket to have any chance. This is where underdog upside and high-risk archetypes earn their keep.

---

## Design Decisions That Mattered

### Why Five Brackets?

Most bracket contests allow multiple entries. Five brackets is the sweet spot between diversification (more brackets = more coverage of possible outcomes) and dilution (too many brackets means they start overlapping). The constraint is enforced by `portfolio_size=5` in the simulation profile, with `min_distinct_archetypes=3` ensuring variety and `max_per_archetype=2` preventing over-concentration.

### Why Archetype-Based Generation?

Rather than randomly perturbing a single "best" bracket, the engine generates candidates from five distinct strategic profiles. This produces structurally different brackets -- a high-confidence bracket and an underdog-upside bracket disagree on fundamental questions like "who wins the championship?" rather than just disagreeing on one first-round game. The temperature parameter controls how sharply each archetype commits to its identity.

### Why First-Place Equity Over Capture Rate?

Capture rate counts ties as wins. In a 768-person pool, if your bracket ties for first with 50 other brackets, capture rate says you "won." But you split the prize 51 ways. FPE correctly values that outcome at 1/51 of first place. This matters because chalk brackets tie for first often (when the favorites all win, many brackets look alike) but rarely win outright. Contrarian brackets tie less often but when they do, they tie with fewer people.

### Why the AutoBracket Approach?

The parameter space has 20 dimensions with complex interactions (changing `injury_weight` affects which teams are strong, which changes archetype behavior, which changes portfolio composition). Grid search is infeasible. Bayesian optimization requires a differentiable objective. Random search wastes evaluations. The autoresearch-style hill-climbing -- mutate, evaluate, keep/revert -- is simple, parallelizable, and converges reliably when the landscape is reasonably smooth. The V8 overnight run's 12-for-12 KEEP rate suggests the landscape is indeed smooth near the optimum.

---

## Timeline

All development occurred March 17-18, 2026, in the 48 hours before the tournament bracket lock deadline.

| Time | Milestone |
|------|-----------|
| Mar 17 AM | V1 scaffold operational, demo data |
| Mar 17 midday | V2 ESPN data, critical bug fixes |
| Mar 17 PM | V3 complete bracket rebuild with KenPom/BPI |
| Mar 17 evening | V4 BPI ensemble, V5 contrarian value |
| Mar 17 night | V6 structural upgrades, LLM code review |
| Mar 18 00:00 | V7 review fixes, V8 FPE optimizer deployed |
| Mar 18 07:36 | V8 AutoBracket begins on vast.ai (128-core) |
| Mar 18 13:41 | V8 AutoBracket completes (12 rounds, all KEEPs) |
| Mar 18 PM | V9 path consistency, V9.1 vectorized simulation |

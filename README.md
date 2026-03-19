# March Madness 2026 Bracket Portfolio Engine

I built this repo to answer one practical question: if you are not a basketball expert but you still want to compete seriously in March Madness pools, what is the best data-driven way to choose your brackets?

The result is a release-tested bracket portfolio engine built from a trained matchup model, a public-field model, and a guarded search process that refuses to ship variants just because they are newer.

The current baseline is **V10.6**. Under the final release contract, that baseline beat the naive portfolio on the numbers that mattered:
- first-place equity: `0.4580` vs `0.3194`
- capture rate: `0.4704` vs `0.3278`
- expected payout: `239.09` vs `186.61`

If you want the exact winning release state, start with [docs/current-release-state.md](docs/current-release-state.md). If you want the final submitted brackets, go straight to [docs/submission-brackets/README.md](docs/submission-brackets/README.md).

## Start Here

If you are seeing this repo for the first time, use this path:

1. [docs/current-release-state.md](docs/current-release-state.md)  
   The fastest summary of what won, what shipped, and what remained incomplete.

2. [docs/submission-baseline.md](docs/submission-baseline.md)  
   The final V10.6 decision and the best three-bracket subset.

3. [docs/submission-brackets/README.md](docs/submission-brackets/README.md)  
   The public copy of the five final submitted brackets.

4. [docs/model-architecture-and-evolution.md](docs/model-architecture-and-evolution.md)  
   The full explanation of the hybrid design, data choices, and version history.

## What This Repo Optimizes

This is not a simple “which team will win the tournament?” model.

That question is too small for a bracket contest. A real pool depends on round scoring, payout shape, field size, how duplicated the obvious picks are, and how a small set of your own entries interact with each other. The project is trying to answer a harder question:

> If you only get a handful of NCAA tournament entries, what set of brackets gives you the best chance to finish first in a specific kind of pool?

That is why the system optimizes a **portfolio of brackets** instead of just predicting a champion.

## What The System Actually Does

The stack has several moving parts because bracket contests are top-heavy and duplication-sensitive:
- a matchup model estimates game-level win probabilities
- a public-field model estimates how opponents are likely to pick and where duplication pressure is high
- tournament simulation turns those probabilities into full-bracket paths
- the portfolio layer scores sets of brackets against simulated opponents and payout structures
- release guardrails reject variants that look interesting in search but fail honest rebuild checks

That means the project is closer to a decision system than a plain forecasting model.

## How The Weighting Works

There is no single master weight knob. The project uses several smaller weight systems at different layers:

| Weight layer | What it controls | Current release_v10_6 values | Where it lives |
| --- | --- | --- |
| Round scoring weights | How many points each round is worth, plus optional upset bonuses | Standard release scoring is `1-2-4-8-16-32` with `0.0` upset bonus | `configs/portfolio/scoring_profiles.yaml` |
| Contest weights | How much the baseline cares about small, mid, and large pool assumptions | `standard_small=0.333333`, `standard_mid=0.333333`, `standard_large=0.333334` | `march_madness_2026/v10/search.py` and `configs/portfolio/simulation_profile.yaml` |
| Payout weights | How top-heavy the pool is: winner-take-all, top-3, top-5, and tie splitting | Small: `1st=100%`; Mid: `1st=55%`, `2nd=30%`, `3rd=15%`; Large: `1st=60%`, `2nd=25%`, `3rd=15%` | `configs/portfolio/payout_profiles.yaml` |
| Opponent archetype mix | How the field is split across high-confidence, balanced, contrarian, and upside-seeking bracket behavior | Release simulation prior is `0.20` each across all five archetypes before contest-specific field behavior adjustments | `configs/portfolio/contest_profiles.yaml` and `march_madness_2026/v10/search.py` |
| Release guardrails | What counts as shippable after search | `3` release seeds, blended-weight floor `0.10`, practical zero-FPE floor `1e-6`, no naive regression allowed, no zero-FPE finalist allowed | `march_madness_2026/v10/portfolio.py` and `march_madness_2026/cli.py` |

The important point is that these layers are deliberate. The project is not arbitrarily piling on weights; it is separating distinct decisions that would otherwise get hidden inside one opaque score.

The release objective itself is also weighted. The current `v10_6_release_objective`:

- heavily rewards first-place equity: `+40.0`
- rewards survivability and payout quality: `+8.0 * cash_rate`, `+6.0 * top3_equity`, `+0.05 * expected_payout`
- rewards portfolio variety: `+2.0 * unique_champions`, `+1.0 * distinct_archetypes`
- penalizes overlap and duplication: overlap starts at `12.0`, duplication at `10.0`, Final Four repeat at `8.0`, region-winner repeat at `6.0`, champion repeat at `4.0`
- scales those diversification penalties upward as contests get larger

Where `large_field_pressure` is derived from the contest blend:
- each `large` contest weight contributes `1.0`
- each `mid` contest weight contributes `0.5`
- `small` contributes `0.0`
- the result is clipped to `[0.0, 1.0]`

So the project is not using one giant arbitrary score. It uses a layered contest model, then a release objective that explicitly trades off upside, survivability, and diversification.

## Current Release_v10_6 Runtime Settings

The public baseline is also specific about search/runtime settings:

- tournament simulations: `3000`
- candidate brackets generated: `240`
- portfolio size: `5`
- max per archetype: `2`
- minimum distinct archetypes: `3`
- overlap penalty weight: `0.18`
- champion penalty weight: `0.05`
- sensitivity contests: `flat_mid`, `upset_bonus_large`

The contest profiles themselves also change by pool size:

| Contest profile | Simulated field size | Opponent mix |
| --- | --- | --- |
| `standard_small` | `24` | `high_confidence=0.40`, `balanced=0.35`, `selective_contrarian=0.15`, `underdog_upside=0.07`, `high_risk_high_return=0.03` |
| `standard_mid` | `96` | `high_confidence=0.26`, `balanced=0.32`, `selective_contrarian=0.20`, `underdog_upside=0.14`, `high_risk_high_return=0.08` |
| `standard_large` | `768` | `high_confidence=0.15`, `balanced=0.25`, `selective_contrarian=0.24`, `underdog_upside=0.20`, `high_risk_high_return=0.16` |

These details are why the repo is more complex than a simple bracket picker, but also why the resulting baseline is more defensible.

## What This Repo Is

This is a local-first research stack for building, backtesting, and publishing a small bracket portfolio. It is designed to be reproducible on one machine, with explicit profiles, explicit artifacts, and a paper trail for what won and what did not.

It includes:
- a Selection Sunday feature snapshot and historical data pipeline
- a trained matchup model and public-field model
- a portfolio selector that optimizes for bracket contests instead of average pick accuracy
- optimizer, backtest, and release tooling for iterating on the stack

It is not a hosted product or a one-command “perfect bracket” app. It is a research codebase with a stable public baseline.

## What Is In The Repo

The core package lives in `march_madness_2026/`. That includes the older engine path and the modern V10+ path under `march_madness_2026/v10/`.

The project configuration lives under `configs/`. The main profiles are:
- `configs/model/training_profile.yaml`: training seasons, feature sets, calibration, and uncertainty settings
- `configs/portfolio/contest_profiles.yaml`: contest-size assumptions and opponent archetype mixes
- `configs/portfolio/payout_profiles.yaml`: prize curves and top-heaviness
- `configs/portfolio/scoring_profiles.yaml`: round-point systems and upset bonuses
- `configs/portfolio/simulation_profile.yaml`: release-time simulation and portfolio-search settings

Small reference data that is worth committing lives under `data/reference/`, and the committed Selection Sunday inference snapshot lives at `data/features/selection_sunday/snapshot.json`.

The main operational scripts are:
- `scripts/build_v10_portfolio.py`
- `scripts/train_v10_game_model.py`
- `scripts/train_v10_public_field.py`
- `scripts/backtest_v10.py`
- `scripts/build_historical_dataset.py`
- `scripts/autobracket_v10.py`

Tests live under `tests/`.

## What Stays Local

This repo is meant to be publishable without dragging along a machine’s local state. That means the heavy build surfaces stay ignored by default:
- `.env`
- `.venv/`
- `node_modules/`
- `logs/`
- `tmp/`
- `outputs/`
- `data/landing/`
- `data/raw/`
- `data/staged/`
- `data/models/`
- most of `data/features/`, except the committed release snapshot

That split is intentional. The committed repo tells the story and contains the reproducible entrypoints. Local outputs, model binaries, remote optimizer logs, and raw ingest mirrors stay local unless they are intentionally curated into `docs/`.

## Quickstart

The setup is simple if you only want to inspect the project or run the release baseline.

Install the dependencies:

```bash
uv sync
npm install
```

Check that the local tooling surface is intact:

```bash
uv run python scripts/verify_project_setup.py
```

Run the test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Build the current release-style portfolio locally:

```bash
uv run python scripts/build_v10_portfolio.py \
  --dataset data/features/selection_sunday/snapshot.json \
  --output-dir outputs/v10_local_preview \
  --training-profile-id release_v10_6
```

Run the backtest path:

```bash
uv run python scripts/backtest_v10.py \
  --training-profile-id release_v10_6 \
  --output-dir outputs/v10_backtest_local
```

## Reproducibility

This project is built around explicit profiles, explicit artifact paths, and explicit rebuild rules. The important operational rule is simple: optimizer checkpoints are not trusted as final winners until they are rebuilt and compared under the exact same release contract.

That rule ended up mattering a lot. More than once, a variant looked better in a raw search or mixed backtest and then lost when rebuilt honestly.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Environment

Optional data integrations use `.env`. The committed example is `.env.example`.

Cloudflare Browser Rendering support expects:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

Other optional source keys are listed in `.env.example`. They are not required to inspect the repo or run the public baseline.

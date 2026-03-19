# March Madness 2026 Bracket Portfolio Engine

The main reason I built this repo was because I do not follow basketball - but I enjoy competing at big sporting events like this. So the question was "how can I use data to choose my bracket?". Using a hybrid of codex (GPT 5.4) and claude (Opus 4.6) back and forth testing, adding weights, and implementing the foundational thinking in https://github.com/karpathy/autoresearch - this was the outcome.

It is a release-tested portfolio built from a trained matchup model, a public-field model, and a guarded search process that refuses to ship variants just because they are newer.

The current baseline is **V10.6**. Under the final release contract, that baseline beat the naive portfolio on the numbers that mattered:
- first-place equity: `0.4580` vs `0.3194`
- capture rate: `0.4704` vs `0.3278`
- expected payout: `239.09` vs `186.61`

If you want the exact winning release state, start with [docs/current-release-state.md](docs/current-release-state.md). If you want the final submitted brackets, go straight to [docs/submission-brackets/README.md](docs/submission-brackets/README.md).

## What This Repo Optimizes

This is not a simple “which team will win the tournament?” model.

That question is too small for a bracket contest. A real pool depends on round scoring, payout shape, field size, how duplicated the obvious picks are, and how a small set of your own entries interact with each other. The project is trying to answer a harder question:

> If you only get a handful of NCAA tournament entries, what set of brackets gives you the best chance to finish first in a specific kind of pool?

That is why the system optimizes a **portfolio of brackets** instead of just predicting a champion.

## Why The Project Is Complex

The stack has several moving parts because bracket contests are top-heavy and duplication-sensitive:
- a matchup model estimates game-level win probabilities
- a public-field model estimates how opponents are likely to pick and where duplication pressure is high
- tournament simulation turns those probabilities into full-bracket paths
- the portfolio layer scores sets of brackets against simulated opponents and payout structures
- release guardrails reject variants that look interesting in search but fail honest rebuild checks

That means the project is closer to a decision system than a plain forecasting model.

## How The Weighting Works

There is no single master weight knob. The project uses several smaller weight systems at different layers:

| Weight layer | What it controls | Where it lives |
| --- | --- | --- |
| Round scoring weights | How many points each round is worth, plus optional upset bonuses | `configs/portfolio/scoring_profiles.yaml` |
| Contest weights | How much the baseline cares about small, mid, and large pool assumptions | `march_madness_2026/v10/search.py` |
| Payout weights | How top-heavy the pool is: winner-take-all, top-3, top-5, and tie splitting | `configs/portfolio/payout_profiles.yaml` |
| Opponent archetype mix | How the field is split across high-confidence, balanced, contrarian, and upside-seeking bracket behavior | `configs/portfolio/contest_profiles.yaml` and `march_madness_2026/v10/search.py` |
| Model/training profile weights | Which seasons, feature families, calibration method, and uncertainty mode define a release profile | `configs/model/training_profile.yaml` |
| Release objective and guardrails | What counts as shippable after search: first-place equity, capture, payout, diversification, and naive-baseline checks | `march_madness_2026/v10/portfolio.py` |

The important point is that these layers are deliberate. The project is not arbitrarily piling on weights; it is separating distinct decisions that would otherwise get hidden inside one opaque score.

## What This Repo Is

This is a local-first research stack for building, backtesting, and publishing a small bracket portfolio. It is designed to be reproducible on one machine, with explicit profiles, explicit artifacts, and a paper trail for what won and what did not.

It includes:
- a Selection Sunday feature snapshot and historical data pipeline
- a trained matchup model and public-field model
- a portfolio selector that optimizes for bracket contests instead of average pick accuracy
- optimizer, backtest, and release tooling for iterating on the stack

It is not a hosted product or a one-command “perfect bracket” app. It is a research codebase with a stable public baseline.

## Why The Project Looks The Way It Does

March Madness is a bad fit for a clean end-to-end dataset. The annual sample is small. The public field matters as much as raw game win probability. Historical public bracket data is incomplete. That pushed the project toward a hybrid design instead of a pure end-to-end learned system.

The short version is:
- game probabilities are learned
- part of the public-field layer is empirical, part is proxy-backed
- final portfolio selection is search-driven
- release guardrails decide what is allowed to ship

That design choice is explained in full in [docs/model-architecture-and-evolution.md](docs/model-architecture-and-evolution.md).

## Start Here

If you are seeing this repo for the first time, read these in order:

1. [docs/current-release-state.md](docs/current-release-state.md)  
   This tells you what the current baseline is, what it beat, and what is still incomplete.

2. [docs/model-architecture-and-evolution.md](docs/model-architecture-and-evolution.md)  
   This explains why the stack is hybrid, how the V10.x line evolved, and which datasets actually matter.

3. [docs/submission-baseline.md](docs/submission-baseline.md)  
   This records the final V10.6 decision and the best three-bracket subset.

4. [docs/submission-brackets/README.md](docs/submission-brackets/README.md)  
   This is the public copy of the five final submission brackets.

5. [docs/publishing-guide.md](docs/publishing-guide.md)  
   Read this only if you are maintaining or publishing the repo. It is a maintainer document, not the main project explainer.

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

## Publishing This Repo

If you are moving this project into its own GitHub repo, read [docs/publishing-guide.md](docs/publishing-guide.md) before you push. The short version is:
- publish this folder, not the whole workspace
- keep local-heavy data and outputs ignored
- use the curated docs bundle, not the raw local output tree
- keep the public framing honest

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Environment

Optional data integrations use `.env`. The committed example is `.env.example`.

Cloudflare Browser Rendering support expects:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

Other optional source keys are listed in `.env.example`. They are not required to inspect the repo or run the public baseline.

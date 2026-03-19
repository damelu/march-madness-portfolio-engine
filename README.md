# March Madness 2026 Bracket Portfolio Engine

This repo exists to answer one practical question: if you only get a handful of NCAA tournament entries, what set of brackets gives you the best shot at finishing first?

The answer in this project is not “pick the five safest brackets.” It is a release-tested portfolio built from a trained matchup model, a public-field model, and a guarded search process that refuses to ship variants just because they are newer.

The current baseline is **V10.6**. Under the final release contract, that baseline beat the naive portfolio on the numbers that mattered:
- first-place equity: `0.4580` vs `0.3194`
- capture rate: `0.4704` vs `0.3278`
- expected payout: `239.09` vs `186.61`

If you want the exact winning release state, start with [docs/current-release-state.md](docs/current-release-state.md). If you want the final submitted brackets, go straight to [docs/submission-brackets/README.md](docs/submission-brackets/README.md).

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
- `configs/model/training_profile.yaml`
- `configs/portfolio/contest_profiles.yaml`
- `configs/portfolio/payout_profiles.yaml`
- `configs/portfolio/scoring_profiles.yaml`
- `configs/portfolio/simulation_profile.yaml`

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

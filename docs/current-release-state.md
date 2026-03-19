# Current Release State

If you only read one document in this repo, read this one.

The current public baseline is **V10.6** with the `release_v10_6` training profile. It is the last version that stayed clean under the release guardrails and still beat the naive baseline on the metrics that mattered at submission time.

If you want the artifact-level answer, read [submission-baseline.md](submission-baseline.md) next. If you want the full final bracket set, go to [submission-brackets/README.md](submission-brackets/README.md).

## The Baseline In One View

V10.6 won because it held together under the full release contract, not because it had the newest label.

The headline numbers were:
- weighted first-place equity: `0.4580`
- weighted capture rate: `0.4704`
- expected payout: `239.09`
- release objective score: `82.5099`
- cash rate: `0.7138`
- top-3 equity: `0.3513`
- guardrail failures: none

Those numbers were good enough to clear the baseline bar, and later variants did not replace it fairly.

## Quick Metric Guide

The four most important terms in this repo are:

- `first-place equity`: how often the portfolio finishes first under the simulated contest model
- `capture rate`: how often the portfolio catches the winning outcome path closely enough to matter
- `release objective`: the guarded scalar used to compare release candidates under the final contract
- `guardrail failures`: reasons a candidate is rejected even if some raw metric looks attractive

## What We Shipped

The winning 5-bracket submission portfolio was:

1. `Michigan` / `selective_contrarian`
2. `Arizona` / `high_confidence`
3. `Duke` / `underdog_upside`
4. `Duke` / `high_confidence`
5. `Florida` / `high_risk_high_return`

If you only had three entries, the best subset was:

1. `Michigan` / `selective_contrarian`
2. `Arizona` / `high_confidence`
3. `Duke` / `underdog_upside`

That three-bracket set won because it gave the highest `3-of-5` release score, kept three distinct champions, and balanced leverage with control. The final public copy of all five brackets lives in [submission-brackets/README.md](submission-brackets/README.md).

## Why V10.6 Stayed The Baseline

Several later variants had something attractive about them. Some looked cleaner in backtests. Some looked better during raw optimizer runs. Some improved data provenance. None of that was enough on its own.

The rule in this repo is simple: a new variant does not become the default unless it wins after rebuild under the exact same release contract.

That rule eliminated:
- `release_v10_6_pruned` as a new default
- later V10.6.1 public-history repair previews as replacement baselines

This is the reason the final baseline is trustworthy. It was not promoted because it was newer. It was promoted because it kept winning when the comparison was fair.

## What Is Empirical In The Current Stack

The current stack has real historical footing, but it is not fully empirical end to end.

What is real in the current baseline:
- real tournament game coverage through `2024`
- empirical ESPN public-history coverage through `2023-2025`

What is still incomplete:
- `2025` game coverage is still not fully empirical end to end
- historical public advancement data is still mixed and partially proxy-backed
- large trained artifacts and optimizer outputs remain local-only

That mix is the reason the repo describes itself as a strong hybrid system instead of a fully empirical forecasting stack.

## What Is Stable Now

The parts of the project that are stable enough to share are:
- the release contract
- the build, train, and backtest tooling
- the V10.6 baseline recommendation
- the curated submission bracket set

The parts that are still open research are:
- fuller empirical historical coverage
- stronger public-field realism
- future optimizer and profile variants beyond `release_v10_6`

## Read Next

If you want the full reasoning behind the hybrid design, the V10.x evolution, and the dataset choices, read [model-architecture-and-evolution.md](model-architecture-and-evolution.md).

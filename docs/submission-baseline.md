# Submission Baseline

This document records the final submission decision from the V10 line.

It answers a narrower question than the rest of the repo: once the build, backtest, and optimizer work were finished, which bracket set actually made it to the line?

## The Winning Submission Baseline

The final submission baseline was `release_v10_6`.

It was chosen because it was the strongest release-eligible artifact at submission time under the guarded release contract. The core numbers were:
- first-place equity: `0.4580`
- capture rate: `0.4704`
- expected payout: `239.09`
- release objective: `82.5099`
- cash rate: `0.7138`
- top-3 equity: `0.3513`
- guardrail failures: none

Later variants were tested and rejected as new defaults:
- `release_v10_6_pruned`
- V10.6.1 public-history repair previews

## Why This Baseline Won

The important point is that V10.6 did not win because it was the most recent version. It won because it survived the most skeptical comparison we had: rebuild the candidate under the same contract, compare it to the incumbent, and only promote it if it still wins.

That standard sounds obvious, but it ruled out several tempting upgrades. Some later variants had cleaner stories or stronger intermediate results. Once rebuilt honestly, they lost.

That is exactly the behavior you want from a release process.

## The Winning 5-Bracket Submission Portfolio

The winning 5-bracket submission portfolio was:

1. `Michigan` / `selective_contrarian`
2. `Arizona` / `high_confidence`
3. `Duke` / `underdog_upside`
4. `Duke` / `high_confidence`
5. `Florida` / `high_risk_high_return`

The public copy of those five brackets lives in [submission-brackets/README.md](submission-brackets/README.md).

## The Best Three-Bracket Subset

If only three entries were available, the best subset from the final five was:

1. `Michigan` / `selective_contrarian`
2. `Arizona` / `high_confidence`
3. `Duke` / `underdog_upside`

That trio won because it had the highest `3-of-5` release score, kept three distinct champions alive, and gave the cleanest split between leverage, control, and upside.

In practical terms:
- `Michigan / selective_contrarian` carried the leverage path
- `Arizona / high_confidence` was the clean control build
- `Duke / underdog_upside` was the best Duke-centered upside entry

The remaining two finalists were still valid portfolio members, but they were not as strong once the problem changed from “best five-entry set” to “best three entries.”

## How To Read The Bracket Docs

The bracket files in [submission-brackets/README.md](submission-brackets/README.md) are meant to be used, not just admired. Each page tells you:
- which champion and Final Four path it commits to
- why it made the final cut
- the full bracket by round
- the suggested tiebreaker number that went with that submission

Those files are intentionally committed into `docs/` instead of left in ignored `outputs/` so the public repo contains the final answer, not just the machinery that produced it.

If you want the one-page project status summary, go back to [current-release-state.md](current-release-state.md). If you want the broader technical rationale behind the stack, read [model-architecture-and-evolution.md](model-architecture-and-evolution.md).

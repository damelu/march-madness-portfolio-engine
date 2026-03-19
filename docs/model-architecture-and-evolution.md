# Model Architecture and Evolution

This document answers the questions that matter once you get past the repo name.

What kind of model is this, really? Why does it look the way it does? Where did the data come from? Why did the project stop at `release_v10_6` instead of stretching for a bigger version label?

The short answer is that this repo is a **hybrid March Madness research stack**. That is not a hedge. It is the accurate description.

## The Honest Label

The project is hybrid because the final bracket portfolio comes from several layers working together:

- a trained matchup model
- engineered basketball and context features
- a public-field and duplication model
- a search-based portfolio optimizer
- release guardrails that decide what is allowed to ship

That is a real modeled system. It is not just a pile of heuristics. But it is also not a pure end-to-end learned bracket model trained on a complete historical bracket-distribution dataset, because that dataset does not exist here in full.

As of `release_v10_6`, the historical manifest still says `synthetic_fallback_used: true`. The current training base contains:

- `796` historical game rows
- `2388` historical public rows
- real game seasons: `2019`, `2021`, `2022`, `2023`, `2024`
- synthetic fallback game season: `2025`
- real public-history seasons: `2023`, `2024`, `2025`
- proxy or synthetic public-history seasons: `2019`, `2021`, `2022`

Those facts are the fastest way to understand why the repo is strong, useful, and still not fully empirical end to end.

## Why The Model Is Hybrid

The game-probability layer is learned. That part lives in [`march_madness_2026/game_model.py`](../march_madness_2026/game_model.py). It uses real historical rows and feature families such as `ratings_base`, `four_factors`, `matchup_interactions`, `context`, and `uncertainty`. That is a real supervised model, not a spreadsheet of pick rules.

The public-field layer is also modeled, but it does not sit on top of a complete observed bracket-distribution archive. That part lives in [`march_madness_2026/public_field.py`](../march_madness_2026/public_field.py). It tries to answer the part of March Madness that raw game models miss: which paths are likely to be crowded, where duplication risk accumulates, and how the field is likely to behave by round. Some of that history is empirical. Some of it is reconstructed from weaker sources or priors.

The final portfolio is not produced by one giant model that emits “the five best brackets.” The project still generates candidates, scores them, compares portfolios, and enforces release gates. That logic lives in [`march_madness_2026/v10/engine.py`](../march_madness_2026/v10/engine.py), [`march_madness_2026/v10/portfolio.py`](../march_madness_2026/v10/portfolio.py), and [`scripts/autobracket_v10.py`](../scripts/autobracket_v10.py). That means the final decision layer is search-driven, not learned end to end.

Some signal is empirical because it can be learned. Some signal is engineered because the data is sparse, incomplete, or only visible through weak proxies. That is especially true for injury uncertainty, lineup continuity, and public duplication pressure. In a domain like this, pretending those gaps do not exist would be less honest than mixing learned and engineered layers carefully.

## What A Fully Empirical Stack Would Require

A fully empirical end-to-end stack would need much more than a stronger classifier.

It would need complete Selection Sunday snapshots for every historical season you want to learn from. It would need complete tournament outcomes for those same seasons. It would need a far better historical record of public picks, round advancement expectations, and path duplication. It would also need enough depth to learn portfolio-level decisions directly instead of leaning so hard on engineered search and release policy.

This repo is not there yet. It is closer to:

> a learned matchup model, a partially empirical public model, engineered features where the data is weak, and a guarded search layer that turns those signals into a five-bracket portfolio.

That is why “hybrid” is not a compromise label. It is the correct one.

## Why That Was The Right Design

March Madness is awkward data.

You only get one tournament a year. Public-field behavior matters as much as raw game win probability. The historical public record is incomplete. The payout structure rewards finishing first, not being mildly accurate across every game.

If the project had waited for a pristine end-to-end historical dataset before trying to ship anything useful, it would still be waiting. The hybrid design let the work progress in the right order:

1. make the system runnable and reproducible
2. make the scoring and release contract honest
3. improve the empirical footing
4. only promote variants that still win under the same contract

That sequencing is why the repo now has a believable public baseline instead of a pile of disconnected experiments.

## How The Project Changed

The earliest version of the project was a local bracket engine. The basic problem was already there: generate candidates, simulate the tournament, score the results, and pick a final set. The older engine shape is still visible in [`march_madness_2026/tournament.py`](../march_madness_2026/tournament.py), [`march_madness_2026/scoring.py`](../march_madness_2026/scoring.py), [`march_madness_2026/portfolio.py`](../march_madness_2026/portfolio.py), and [`march_madness_2026/engine.py`](../march_madness_2026/engine.py).

V8 and V9 were mostly about correctness. The project learned that the optimizer objective, the published metrics, and the actual release artifact had to agree. It also learned that local-search behavior, feasible candidate pools, and reporting honesty mattered more than adding one more feature family too early.

V10 was the architectural split that mattered most. It created the isolated modern path under [`march_madness_2026/v10`](../march_madness_2026/v10), separating the newer artifact-driven stack from the older baseline engine. That change made it possible to train, backtest, search, and report on the modern stack without muddying the contract with legacy behavior.

V10.1 through V10.3 were about honesty under pressure. This was the period where the project learned several expensive lessons:

- a faster optimizer does not help if it is chasing the wrong objective
- the top raw checkpoint is not the same thing as the best rebuilt artifact
- remote compute should match the real bottleneck, not the imagined one
- a release process that lets optimizer logic drift from publish logic will lie to you

V10.3 was the release-discipline release. It aligned optimizer acceptance, selector behavior, and final rebuild under one contract.

V10.4 through V10.6 upgraded the empirical footing. V10.4 introduced the first useful historical rows and season-blocked backtests. V10.5 separated empirical-only holdouts from fallback-heavy aggregates, which stopped the project from making model decisions based on synthetic noise. V10.6 extended real tournament-game coverage through `2024`, added empirical ESPN public history for `2023-2025`, and produced the final submission-ready baseline that beat naive under the guarded release contract.

That is the path that made the current repo worth publishing.

## The Datasets And Why They Belong Here

The repo uses two kinds of data.

The first kind is small, committed, and there to make the project understandable. The second kind is larger, local, and there to make the project trainable.

### Committed Reference Data

The small committed files in `data/reference/` and the committed Selection Sunday snapshot are in the repo because they make the project legible.

[`data/reference/bracket_2026.json`](../data/reference/bracket_2026.json) is the cleanest human-readable view of the field. It gives the repo a stable tournament reference, keeps explanations grounded, and makes it possible to talk about why the finalists keep circling the same teams without forcing a reader through a build pipeline first.

[`data/reference/public_pick_rates.json`](../data/reference/public_pick_rates.json) and [`data/reference/seed_matchup_history.json`](../data/reference/seed_matchup_history.json) are not glamorous, but they matter. They are explicit fallback priors. In this domain, fallback priors are better than pretending the missing data does not matter.

[`data/features/selection_sunday/snapshot.json`](../data/features/selection_sunday/snapshot.json) is committed because it turns the repo into something a fresh clone can actually run. Without that file, the project would read more like a promise than a working artifact.

### Local Historical Training Data

The main historical backbone comes from structured NCAA and Kaggle March Madness tables. The V10.6 manifest names the important ones directly:

- `MNCAATourneyDetailedResults.csv`
- `MRegularSeasonDetailedResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`
- `MTeamCoaches.csv`
- `MTeamConferences.csv`
- `MTeams.csv`

These belong in the build pipeline because they are the fastest and most practical route from synthetic scaffolding to real historical supervision. They are structured, they are rich enough to support multiple feature families, and they are usable in a local workflow.

The project also uses NCAA bracket pages for empirical outcome coverage. In the manifest those are the bracket pages for `2019`, `2021`, `2022`, `2023`, and `2024`. They matter because the repo needed real tournament seasons, not just simulated ones.

For recent public behavior, the key step was the ESPN propositions API:

- `https://gambit-api.fantasy.espn.com/apis/v1/propositions?challengeId=239`
- `https://gambit-api.fantasy.espn.com/apis/v1/propositions?challengeId=240`
- `https://gambit-api.fantasy.espn.com/apis/v1/propositions?challengeId=257`

That source mattered because it moved the public-field layer from “smart proxy” closer to “observed public behavior” for recent seasons. In a bracket contest, that is a real upgrade, not a side detail.

Later passes also used sources like `mRchmadness` to improve historical public-row provenance and expose ingestion bugs. Those passes were still worth doing even when they did not produce a new winning baseline, because they made the data story more honest.

### Planned And Optional Sources

The repo also carries a forward-looking source architecture in:

- [docs/source-catalog.md](source-catalog.md)
- [docs/data-requirements.md](data-requirements.md)
- [`configs/source_registry.yaml`](../configs/source_registry.yaml)

Those files matter because they show the next extension path clearly. The project is explicit about the intended stack: NCAA feeds, `stats.ncaa.org`, hoopR, market priors, official athletics sites, Bart Torvik, KenPom, EvanMiya, and other roster, injury, and portal sources where they can be used responsibly.

That documentation belongs in the repo because a serious research project should make its next data layer visible instead of leaving it trapped in someone’s head.

## Why These Datasets Were Chosen

The dataset choices were pragmatic, not ideological.

Some files were included because they improved reproducibility. Some were included because they improved empirical footing. Some were included because the perfect version of the data did not exist and an explicit fallback prior was still better than silence.

The project kept data when it did one of four useful things:

- made the repo runnable
- made the model more empirical
- made the fallback behavior explicit
- made the next extension path legible

That is a better standard than “collect everything.”

## Why The Final Outcome Is Good

The final outcome is not good because it is perfect. It is good because it is honest and useful at the same time.

The repo now distinguishes clearly between what is empirical, what is proxy-backed, what is synthetic fallback, and what is actually shipped. That sounds basic, but it is one of the hardest things to maintain once a project starts iterating quickly.

The project is also reproducible now in a way it was not early on. There is a committed snapshot, versioned profiles, versioned artifacts, backtests, release guards, and a documented winning baseline. Another engineer can clone the repo and understand the shape of the system without needing the original working session in front of them.

Most importantly, the final baseline actually beat naive on the metrics that mattered: first-place equity, capture, and expected payout. That is the only result that really matters in the end.

The repo also learned to reject false upgrades. Several later or more interesting-looking variants did not survive fair comparison. Some looked better before rebuild. Some looked better in mixed backtests. Some improved provenance without improving the final release artifact. The project did not promote them anyway.

That is why the current result is trustworthy.

## Why We Stopped At V10.6

The project did not stop because it ran out of ideas. It stopped because the next honest step was still a 10.x step, not a `V11` label.

A real V11 would require substantially fuller empirical game coverage, substantially fuller empirical public-history coverage, stronger field realism, and evidence that the new release is more than another profile tweak or data repair. The repo did not have that yet.

Stopping at `release_v10_6` was the disciplined choice. It is strong, releaseable, empirically improved, and still clear about its limits.

That is exactly where a serious research repo should stop.

## Final Characterization

The shortest accurate description of this project is:

> a local-first March Madness bracket-portfolio research engine with a trained matchup model, a partially empirical public-field layer, and a guarded release process

And the shortest honest qualifier is:

> a strong hybrid system with a credible release baseline, not yet a fully empirical end-to-end forecasting stack

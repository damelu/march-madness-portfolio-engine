# Publishing Guide

This project lives inside a larger local workspace. If you want to put it on GitHub, publish **this folder as its own repo**. Do not publish the whole workspace and assume readers will figure out where the real project is.

## What To Keep In The Public Repo

The public repo should show the project, not the machine it was built on.

That means it should include:
- the source code under `march_madness_2026/`
- committed configs under `configs/`
- committed docs under `docs/`
- committed reference data under `data/reference/`
- the committed Selection Sunday snapshot if you want reproducible local release builds
- tests under `tests/`
- packaging metadata such as `pyproject.toml`, `package.json`, and `.env.example`

Those files are enough for someone to understand the project, run the public baseline, and inspect the modeling and release logic.

## What Should Stay Local

The repo should not become a dump of local state.

Keep these local:
- `.env`
- `.venv/`
- `node_modules/`
- `outputs/`
- `logs/`
- `tmp/`
- `data/landing/`
- `data/raw/`
- `data/staged/`
- `data/models/`
- remote-run artifacts and transient optimizer outputs

That split is already part of the repo contract. The public-facing docs explain the project. The ignored local trees are where heavy generation work happens.

## The Curated Exception

There is one important exception to the “don’t publish outputs” rule: if a generated artifact becomes part of the project’s public story, promote a curated copy of it into `docs/`.

That is why the final five submission brackets live in [submission-brackets/README.md](submission-brackets/README.md). They are not generic local output. They are the final public answer from the winning release baseline.

## License

The recommended license for this repo is Apache-2.0, and the project now includes a top-level [LICENSE](../LICENSE) file for that choice.

Apache-2.0 is the right default here because it keeps the repo easy to reuse while still carrying an explicit patent grant. It is permissive without being sloppy.

## What Not To Claim

Do not describe this repo as:
- a fully empirical end-to-end historical model
- a hosted product
- a universally productionized forecasting platform

The honest framing is stronger than that anyway. This is a serious local research stack with a guarded release process and a documented winning baseline.

## The Cleanest Way To Publish It

If you want a standalone public repo, the easiest approach is to copy this project into a clean directory and initialize Git there.

Use a copy step that excludes the heavy local-only trees:

```bash
rsync -a \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude 'node_modules' \
  --exclude 'logs' \
  --exclude 'tmp' \
  --exclude 'outputs' \
  --exclude 'data/landing' \
  --exclude 'data/raw' \
  --exclude 'data/staged' \
  --exclude 'data/models' \
  march-madness-2026/ /path/to/publish/march-madness-2026/
```

Then initialize and inspect it there:

```bash
cd /path/to/publish/march-madness-2026
git init
git add .
git status
```

The important thing is not the exact command. The important thing is that the published repo should look intentional. A reader should see a coherent project, not a workstation snapshot.

If you are still working in the original local workspace, you can also use the helper script in [`scripts/prepare_public_repo.sh`](../scripts/prepare_public_repo.sh). It builds a clean publish copy, removes junk files, runs verification, runs the test suite, initializes a fresh Git repo in the copy, and stages the result for inspection.

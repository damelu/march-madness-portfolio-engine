#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
STAMP="$(date +%F)"
DEFAULT_DEST="${PROJECT_ROOT}/.publish/${STAMP}_${PROJECT_NAME}"
DEST="${1:-${DEFAULT_DEST}}"

if [[ -e "${DEST}" ]]; then
  echo "Destination already exists: ${DEST}" >&2
  echo "Pass a different destination path or remove the existing one first." >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST}")"

echo "Preparing public repo copy..."
echo "Project root: ${PROJECT_ROOT}"
echo "Destination:  ${DEST}"

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
  --exclude '.publish' \
  "${PROJECT_ROOT}/" "${DEST}/"

find "${DEST}" -name '.DS_Store' -delete

mkdir -p \
  "${DEST}/data/landing/api" \
  "${DEST}/data/landing/crawl" \
  "${DEST}/data/landing/manual" \
  "${DEST}/data/raw/ncaa" \
  "${DEST}/data/raw/athletics_sites" \
  "${DEST}/data/raw/ratings" \
  "${DEST}/data/raw/injuries" \
  "${DEST}/data/raw/recruiting" \
  "${DEST}/data/raw/odds" \
  "${DEST}/data/raw/news_sentiment" \
  "${DEST}/data/staged/teams" \
  "${DEST}/data/staged/players" \
  "${DEST}/data/staged/games" \
  "${DEST}/data/staged/coaches" \
  "${DEST}/data/staged/events" \
  "${DEST}/data/models/training" \
  "${DEST}/data/models/inference" \
  "${DEST}/data/models/artifacts" \
  "${DEST}/logs" \
  "${DEST}/outputs" \
  "${DEST}/tmp"

echo
echo "Running project verification in publish copy..."
(
  cd "${DEST}"
  uv run python scripts/verify_project_setup.py
)

echo
echo "Running tests in publish copy..."
(
  cd "${DEST}"
  uv run python -m unittest discover -s tests -v
)

echo
echo "Initializing fresh git repository in publish copy..."
(
  cd "${DEST}"
  rm -rf .git
  git init >/dev/null
  git add .
  git status --short
)

echo
echo "Public repo copy is ready:"
echo "${DEST}"

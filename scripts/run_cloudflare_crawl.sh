#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <start|status> <url-or-job-id> [profile]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
TARGET="$2"
PROFILE="${3:-official-athletics-markdown}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
fi

: "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID in .env}"
: "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN in .env}"

BASE_URL="https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/browser-rendering/crawl"

case "$MODE" in
  start)
    PROFILE_PATH="$ROOT_DIR/configs/crawl_profiles/${PROFILE}.json"
    if [[ ! -f "$PROFILE_PATH" ]]; then
      echo "missing profile: $PROFILE_PATH" >&2
      exit 1
    fi

    jq --arg url "$TARGET" '.url = $url' "$PROFILE_PATH" | curl -sS \
      -X POST "$BASE_URL" \
      -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
      -H "Content-Type: application/json" \
      --data @-
    ;;
  status)
    curl -sS \
      -X GET "${BASE_URL}/${TARGET}" \
      -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}"
    ;;
  *)
    echo "mode must be start or status" >&2
    exit 1
    ;;
esac


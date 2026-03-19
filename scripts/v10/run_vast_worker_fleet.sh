#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER_WRAPPER="${SCRIPT_DIR}/run_vast_worker.sh"

usage() {
  cat <<'EOF'
Usage:
  run_vast_worker_fleet.sh start [options]
  run_vast_worker_fleet.sh status [options]
  run_vast_worker_fleet.sh stop [options]

Options:
  --workers N                 Number of independent workers to launch (default: 4)
  --session-prefix NAME       tmux prefix (default: mm-v10-fleet)
  --log-prefix PATH           log-dir prefix relative to cwd (default: outputs/autobracket_v10_fleet)
  --cwd PATH                  project root / remote repo path (default: project root)
  --gpu-id ID                 CUDA_VISIBLE_DEVICES for all workers (default: 0)
  --env KEY=VALUE             Extra environment variable forwarded to each worker (repeatable)
  --base-seed N               Base optimizer seed; worker i uses base_seed + i (default: 20260318)
  --checkpoint PATH           Optional best_v10_params.json copied into each worker log dir before launch
  --iterations N              Optimizer iterations (default: 300)
  --batch N                   Optimizer batch size (default: 4)
  --sims N                    Tournament sims (default: 120)
  --candidates N              Candidate brackets (default: 80)
  --training-profile-id ID    Training profile id (default: default_v10)
  --dataset PATH              Dataset path (default: data/models/v10/inference/2026/snapshot.json)
  --game-model-artifact PATH  Game-model artifact path
  --public-field-artifact PATH Public-field artifact path
  --release-seed-count N      Number of release seeds (default: 3)
  --contest-mode MODE         Optimizer contest mode (default: blended)
  --blended-weight-floor X    Blended weight floor override (default: 0.1)
  --allow-naive-regression    Forwarded to autobracket_v10.py
  --allow-zero-fpe-finalist   Forwarded to autobracket_v10.py

Examples:
  bash scripts/v10/run_vast_worker_fleet.sh start \
    --workers 4 \
    --session-prefix codex-v10-3 \
    --log-prefix outputs/autobracket_v10_3_fleet \
    --checkpoint outputs/autobracket_v10_3_local/best_v10_params.json

  bash scripts/v10/run_vast_worker_fleet.sh status --workers 4 --session-prefix codex-v10-3 \
    --log-prefix outputs/autobracket_v10_3_fleet

  bash scripts/v10/run_vast_worker_fleet.sh stop --workers 4 --session-prefix codex-v10-3 \
    --log-prefix outputs/autobracket_v10_3_fleet
EOF
}

resolve_path() {
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}

subcommand="${1:-}"
if [[ -z "${subcommand}" ]]; then
  usage
  exit 1
fi
shift || true

workers="4"
session_prefix="mm-v10-fleet"
log_prefix="outputs/autobracket_v10_fleet"
cwd="${PROJECT_ROOT}"
gpu_id="0"
extra_envs=()
base_seed="20260318"
checkpoint=""
iterations="300"
batch="4"
sims="120"
candidates="80"
training_profile_id="default_v10"
dataset="data/models/v10/inference/2026/snapshot.json"
game_model_artifact="data/models/v10/game_model_v10_default.pkl"
public_field_artifact="data/models/v10/public_field_v10_default.pkl"
release_seed_count="3"
contest_mode="blended"
blended_weight_floor="0.1"
allow_naive_regression="0"
allow_zero_fpe_finalist="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) workers="$2"; shift 2 ;;
    --session-prefix) session_prefix="$2"; shift 2 ;;
    --log-prefix) log_prefix="$2"; shift 2 ;;
    --cwd) cwd="$2"; shift 2 ;;
    --gpu-id) gpu_id="$2"; shift 2 ;;
    --env) extra_envs+=("$2"); shift 2 ;;
    --base-seed) base_seed="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --batch) batch="$2"; shift 2 ;;
    --sims) sims="$2"; shift 2 ;;
    --candidates) candidates="$2"; shift 2 ;;
    --training-profile-id) training_profile_id="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --game-model-artifact) game_model_artifact="$2"; shift 2 ;;
    --public-field-artifact) public_field_artifact="$2"; shift 2 ;;
    --release-seed-count) release_seed_count="$2"; shift 2 ;;
    --contest-mode) contest_mode="$2"; shift 2 ;;
    --blended-weight-floor) blended_weight_floor="$2"; shift 2 ;;
    --allow-naive-regression) allow_naive_regression="1"; shift ;;
    --allow-zero-fpe-finalist) allow_zero_fpe_finalist="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cwd="$(resolve_path "${cwd}")"

for ((i = 0; i < workers; i++)); do
  session="${session_prefix}-${i}"
  log_dir="${log_prefix}_${i}"
  seed="$((base_seed + i))"

  case "${subcommand}" in
    start)
      mkdir -p "${cwd}/${log_dir}"
      if [[ -n "${checkpoint}" ]]; then
        cp "${checkpoint}" "${cwd}/${log_dir}/best_v10_params.json"
      fi
      cmd=(
        bash "${WORKER_WRAPPER}" start
        --session "${session}"
        --cwd "${cwd}"
        --gpu-id "${gpu_id}"
        --log-dir "${cwd}/${log_dir}"
        --
        env
      )
      for env_kv in "${extra_envs[@]}"; do
        cmd+=("${env_kv}")
      done
      cmd+=(
        uv run python scripts/autobracket_v10.py
        --iterations "${iterations}"
        --resume
        --batch "${batch}"
        --sims "${sims}"
        --candidates "${candidates}"
        --seed "${seed}"
        --dataset "${dataset}"
        --training-profile-id "${training_profile_id}"
        --game-model-artifact "${game_model_artifact}"
        --public-field-artifact "${public_field_artifact}"
        --log-dir "${log_dir}"
        --release-seed-count "${release_seed_count}"
        --contest-mode "${contest_mode}"
        --blended-weight-floor "${blended_weight_floor}"
      )
      if [[ "${allow_naive_regression}" == "1" ]]; then
        cmd+=(--allow-naive-regression)
      fi
      if [[ "${allow_zero_fpe_finalist}" == "1" ]]; then
        cmd+=(--allow-zero-fpe-finalist)
      fi
      "${cmd[@]}"
      ;;

    status)
      echo "=== ${session} ==="
      if ! bash "${WORKER_WRAPPER}" status --session "${session}" --log-dir "${cwd}/${log_dir}"; then
        echo "status unavailable for ${session}" >&2
      fi
      ;;

    stop)
      echo "=== ${session} ==="
      if ! bash "${WORKER_WRAPPER}" stop --session "${session}"; then
        echo "stop unavailable for ${session}" >&2
      fi
      ;;

    *)
      usage
      exit 1
      ;;
  esac
done

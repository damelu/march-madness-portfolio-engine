#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_vast_worker.sh start [options] -- COMMAND...
  run_vast_worker.sh status [options]
  run_vast_worker.sh attach [options]
  run_vast_worker.sh logs [options]
  run_vast_worker.sh stop [options]

Options:
  --session NAME        tmux session name (default: mm-v10-worker)
  --log-dir PATH        log/status directory (default: outputs/vast_worker)
  --cwd PATH            working directory for the child command (default: project root)
  --gpu-id ID           CUDA_VISIBLE_DEVICES value for the tmux session
  --poll-seconds N      heartbeat/status refresh interval (default: 10)
  --tail N              lines for logs subcommand (default: 100)

Examples:
  scripts/v10/run_vast_worker.sh start --session mm-v10-0 --gpu-id 0 \
    --log-dir outputs/autobracket_v10_worker0 -- \
    uv run python scripts/autobracket_v10.py --resume --batch 4

  scripts/v10/run_vast_worker.sh status --log-dir outputs/autobracket_v10_worker0
  scripts/v10/run_vast_worker.sh attach --session mm-v10-0
  scripts/v10/run_vast_worker.sh logs --log-dir outputs/autobracket_v10_worker0 --tail 50
  scripts/v10/run_vast_worker.sh stop --session mm-v10-0
EOF
}

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
}

resolve_tool() {
  local tool="$1"
  local candidate=""
  if candidate="$(command -v "$tool" 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  for prefix in "${HOME}/.local/bin" "/root/.local/bin" "/usr/local/bin" "/opt/homebrew/bin"; do
    if [[ -x "${prefix}/${tool}" ]]; then
      printf '%s\n' "${prefix}/${tool}"
      return 0
    fi
  done
  return 1
}

resolve_path() {
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}

tmux_has_session() {
  local session="$1"
  tmux has-session -t "$session" >/dev/null 2>&1
}

json_pretty() {
  python3 -m json.tool "$1"
}

subcommand="${1:-}"
if [[ -z "${subcommand}" ]]; then
  usage
  exit 1
fi
shift || true

session="mm-v10-worker"
log_dir="${PROJECT_ROOT}/outputs/vast_worker"
cwd="${PROJECT_ROOT}"
gpu_id=""
poll_seconds="10"
tail_lines="100"

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session)
        session="$2"
        shift 2
        ;;
      --log-dir)
        log_dir="$2"
        shift 2
        ;;
      --cwd)
        cwd="$2"
        shift 2
        ;;
      --gpu-id)
        gpu_id="$2"
        shift 2
        ;;
      --poll-seconds)
        poll_seconds="$2"
        shift 2
        ;;
      --tail)
        tail_lines="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done
  REMAINING_ARGS=("$@")
}

parse_common_args "$@"
log_dir="$(resolve_path "${log_dir}")"
cwd="$(resolve_path "${cwd}")"
status_path="${log_dir}/status.json"
heartbeat_path="${log_dir}/heartbeat.json"
run_log="${log_dir}/run.log"

case "${subcommand}" in
  start)
    require_tool tmux
    mkdir -p "${log_dir}"
    if tmux_has_session "${session}"; then
      echo "tmux session already exists: ${session}" >&2
      exit 1
    fi
    if [[ ${#REMAINING_ARGS[@]} -eq 0 ]]; then
      echo "start requires a command after --" >&2
      exit 1
    fi

    printf '%s\n' "${REMAINING_ARGS[@]}" > "${log_dir}/command.argv"
    printf '%s\n' "$(date -Is 2>/dev/null || date)" > "${log_dir}/launched_at.txt"

    uv_bin="$(resolve_tool uv)" || {
      echo "missing required tool: uv" >&2
      exit 1
    }
    uv_dir="$(dirname "${uv_bin}")"
    printf -v child_cmd '%q ' "${REMAINING_ARGS[@]}"
    tmux_cmd="cd $(printf '%q' "${PROJECT_ROOT}")"
    if [[ -n "${gpu_id}" ]]; then
      tmux_cmd+=" && export CUDA_VISIBLE_DEVICES=$(printf '%q' "${gpu_id}")"
    fi
    tmux_cmd+=" && export PATH=$(printf '%q' "${uv_dir}"):\$PATH"
    tmux_cmd+=" && export PYTHONUNBUFFERED=1"
    tmux_cmd+=" && exec $(printf '%q' "${uv_bin}") run python scripts/v10/vast_worker.py run"
    tmux_cmd+=" --name $(printf '%q' "${session}")"
    tmux_cmd+=" --tmux-session $(printf '%q' "${session}")"
    tmux_cmd+=" --log-dir $(printf '%q' "${log_dir}")"
    tmux_cmd+=" --cwd $(printf '%q' "${cwd}")"
    tmux_cmd+=" --poll-seconds $(printf '%q' "${poll_seconds}")"
    tmux_cmd+=" -- ${child_cmd}"

    tmux new-session -d -s "${session}" "${tmux_cmd}"
    echo "started tmux session ${session}"
    echo "log dir: ${log_dir}"
    echo "status: ${status_path}"
    echo "heartbeat: ${heartbeat_path}"
    echo "run log: ${run_log}"
    ;;

  status)
    if tmux_has_session "${session}"; then
      echo "tmux_session=${session} (running)"
    else
      echo "tmux_session=${session} (not running)"
    fi
    echo "log_dir=${log_dir}"
    if [[ -f "${status_path}" ]]; then
      json_pretty "${status_path}"
    else
      echo "status file not found: ${status_path}" >&2
      exit 1
    fi
    ;;

  attach)
    require_tool tmux
    exec tmux attach -t "${session}"
    ;;

  logs)
    if [[ ! -f "${run_log}" ]]; then
      echo "log file not found: ${run_log}" >&2
      exit 1
    fi
    exec tail -n "${tail_lines}" -F "${run_log}"
    ;;

  stop)
    require_tool tmux
    if tmux_has_session "${session}"; then
      tmux kill-session -t "${session}"
      echo "stopped tmux session ${session}"
    else
      echo "tmux session not running: ${session}" >&2
      exit 1
    fi
    ;;

  *)
    usage
    exit 1
    ;;
esac

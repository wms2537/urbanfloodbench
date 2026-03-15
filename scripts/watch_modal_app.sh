#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
FOLLOW_ACTIVE=0
APP_ID="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  watch_modal_app.sh [app_id]
  watch_modal_app.sh --follow-active

Environment:
  INTERVAL_SECONDS   Poll interval in seconds. Default: 300

Behavior:
  - Without args, monitors the newest active Modal app if one exists.
  - With --follow-active, waits for the next active app and then monitors it.
  - Writes concise status snapshots and decision points into logs/.
EOF
}

strip_ansi() {
  perl -pe 's/\e\[[0-9;]*[A-Za-z]//g'
}

current_time() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

latest_active_app() {
  modal app list | strip_ansi | perl -ne 'if (/ap-[A-Za-z0-9]+/) { /((ap-[A-Za-z0-9]+))/; print "$1\n"; exit }'
}

app_is_active() {
  local app_id="$1"
  modal app list | strip_ansi | rg -q "${app_id}"
}

log_tail() {
  local app_id="$1"
  timeout 15 modal app logs "${app_id}" --timestamps 2>&1 | strip_ansi | tail -n 300 || true
}

extract_signal() {
  rg 'val/std_rmse=|Metric val/std_rmse improved|Epoch [0-9]+:|Validation DataLoader|Traceback|OutOfMemory|NaN|APP_STOPPED|APP_STARTED'
}

monitor_app() {
  local app_id="$1"
  local log_file="${LOG_DIR}/modal_watch_${app_id}_$(date '+%Y%m%d_%H%M%S').log"
  local last_signal=""

  echo "[$(current_time)] APP_STARTED ${app_id}" | tee -a "${log_file}"

  while true; do
    local ts
    ts="$(current_time)"

    if ! app_is_active "${app_id}"; then
      {
        echo "[$ts] APP_STOPPED ${app_id}"
        log_tail "${app_id}" | extract_signal | tail -n 12
      } | tee -a "${log_file}"
      break
    fi

    local signal
    signal="$(log_tail "${app_id}" | extract_signal | tail -n 12 || true)"

    if [[ -n "${signal}" && "${signal}" != "${last_signal}" ]]; then
      {
        echo "=== [${ts}] ==="
        printf '%s\n' "${signal}"
      } | tee -a "${log_file}"
      last_signal="${signal}"
    fi

    sleep "${INTERVAL_SECONDS}"
  done
}

case "${APP_ID}" in
  -h|--help)
    usage
    exit 0
    ;;
  --follow-active)
    FOLLOW_ACTIVE=1
    APP_ID=""
    ;;
esac

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found in PATH" >&2
  exit 1
fi

if (( FOLLOW_ACTIVE )); then
  while true; do
    APP_ID="$(latest_active_app || true)"
    if [[ -n "${APP_ID}" ]]; then
      monitor_app "${APP_ID}"
    fi
    sleep "${INTERVAL_SECONDS}"
  done
fi

if [[ -z "${APP_ID}" ]]; then
  APP_ID="$(latest_active_app || true)"
fi

if [[ -z "${APP_ID}" ]]; then
  echo "No active Modal app found" >&2
  exit 1
fi

monitor_app "${APP_ID}"

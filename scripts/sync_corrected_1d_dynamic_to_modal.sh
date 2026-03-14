#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME_NAME="${1:-urbanfloodbench-data}"
REMOTE_DATA_ROOT="${2:-/data/data}"

cd "$ROOT_DIR"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found in PATH" >&2
  exit 1
fi

count=0
while IFS= read -r -d '' file_path; do
  rel_path="${file_path#${ROOT_DIR}/data/}"
  remote_path="${REMOTE_DATA_ROOT}/${rel_path}"
  count=$((count + 1))
  echo "[$count] ${file_path} -> ${remote_path}"
  modal volume put -f "$VOLUME_NAME" "$file_path" "$remote_path"
done < <(
  find \
    "$ROOT_DIR/data/Model_1" \
    "$ROOT_DIR/data/Model_2" \
    -type f \
    -name '1d_nodes_dynamic_all.csv' \
    -print0 | sort -z
)

echo "Synced $count corrected 1D dynamic files to Modal volume '$VOLUME_NAME' under '$REMOTE_DATA_ROOT'."

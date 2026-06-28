#!/usr/bin/env bash
# Push the working tree Mac -> board. Edit on Mac, sync, then board_build.sh.
set -euo pipefail

BOARD="${BOARD:-rdk}"
DEST="${DEST:-projects/bcdl}"   # relative to board $HOME
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$BOARD" "mkdir -p ${DEST}"
echo ">> sync ${PROJ} -> ${BOARD}:${DEST}"
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'build/' \
  --exclude 'third_party/' \
  --exclude '__pycache__/' \
  --exclude '*.so' \
  --exclude '*.hbm' \
  "${PROJ}/" "${BOARD}:${DEST}/"
echo ">> synced"

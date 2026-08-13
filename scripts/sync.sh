#!/usr/bin/env bash
# Push the working tree Mac -> board. Edit on Mac, sync, then board_build.sh.
#
# This deletes on the far side, so everything fetch_models.sh puts in models/ is
# PROTECTED below rather than excluded by extension. Extensions were the first
# fix and it was the wrong one: the anchors arrived as .npy, then the bench
# samples as .png and .npz, and each new asset type got silently deleted on the
# next sync until someone noticed. Protect the directory, not a file list.
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
  --filter 'protect models/*' \
  "${PROJ}/" "${BOARD}:${DEST}/"
echo ">> synced"

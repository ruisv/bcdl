#!/usr/bin/env bash
# One-time board setup: install Miniforge (aarch64) if absent, then create the
# `bcdl` conda env. Idempotent. Run from the Mac:  scripts/bootstrap_board.sh
set -euo pipefail

BOARD="${BOARD:-rdk}"
DEST="${DEST:-projects/bcdl}"   # relative to $HOME on the board, as in board_build.sh

echo ">> bootstrapping conda env on ${BOARD}"
ssh "$BOARD" "DEST='$DEST' bash -s" <<'REMOTE'
set -euo pipefail

# 1. Locate a conda. A non-interactive ssh shell does NOT source ~/.bashrc, so
# `command -v conda` can be empty even when conda is installed — probe the known
# install dirs directly and prefer an EXISTING one. Only install Miniforge if no
# conda is found anywhere (avoids creating a second, redundant install).
CONDA_SH=""
for _c in "$HOME/conda" "$HOME/miniforge3" "$HOME/miniconda3"; do
  if [ -f "$_c/etc/profile.d/conda.sh" ]; then CONDA_SH="$_c/etc/profile.d/conda.sh"; break; fi
done
if [ -z "$CONDA_SH" ]; then
  echo ">> no conda found; installing Miniforge3 (aarch64) to \$HOME/miniforge3"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  tmp="$(mktemp /tmp/miniforge.XXXXXX.sh)"
  curl -fsSL "$url" -o "$tmp"
  bash "$tmp" -b -p "$HOME/miniforge3"
  rm -f "$tmp"
  CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
fi
echo ">> using conda at $(dirname "$(dirname "$(dirname "$CONDA_SH")")")"

# shellcheck disable=SC1091
source "$CONDA_SH"

# 2. env from spec (created on next sync; fall back to inline if not synced yet).
# DEST comes from the caller, so overriding it actually works — it used to be
# declared and then ignored here, which silently pinned the checkout location.
spec="$HOME/${DEST}/env/environment.yml"
if conda env list | grep -qE '^\s*bcdl\s'; then
  echo ">> env 'bcdl' exists; updating"
  [ -f "$spec" ] && conda env update -n bcdl -f "$spec" --prune || true
else
  if [ -f "$spec" ]; then
    conda env create -f "$spec"
  else
    echo ">> spec not synced yet; creating minimal env"
    conda create -y -n bcdl -c conda-forge python=3.10 cmake ninja nanobind numpy pytest
  fi
fi

conda activate bcdl
echo ">> ready:  python=$(python --version 2>&1)  cmake=$(cmake --version | head -1)"
python -c "import nanobind, numpy; print('nanobind', nanobind.__version__, '| numpy', numpy.__version__)"
REMOTE
echo ">> done. Next: scripts/sync.sh && scripts/board_build.sh"

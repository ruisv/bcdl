#!/usr/bin/env bash
# Configure + build BCDL on the board inside the `bcdl` conda env.
#   scripts/board_build.sh            # build
#   scripts/board_build.sh --run      # build, then run det_infer ($HBM)
#   scripts/board_build.sh --clean    # wipe build/ first
set -euo pipefail

BOARD="${BOARD:-rdk}"
DEST="${DEST:-projects/bcdl}"
HBM="${HBM:-}"                     # path on board to a .hbm for --run
RUN=0; CLEAN=0
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --clean) CLEAN=1 ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

ssh "$BOARD" "RUN=$RUN CLEAN=$CLEAN HBM='$HBM' DEST='$DEST' bash -s" <<'REMOTE'
set -euo pipefail
# Prefer the user's existing conda install; fall back to a bootstrapped miniforge3.
for _c in "$HOME/conda" "$HOME/miniforge3"; do
  if [ -f "$_c/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_c/etc/profile.d/conda.sh"; break
  fi
done
conda activate bcdl
cd "$HOME/$DEST"

[ "$CLEAN" = "1" ] && rm -rf build

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo ">> build ok: $(ls -1 build/*.so build/det_infer 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

if [ "$RUN" = "1" ]; then
  if [ -z "$HBM" ]; then
    echo ">> --run needs HBM=/path/to/model.hbm on the board" >&2; exit 1
  fi
  echo ">> running det_infer $HBM"
  ./build/det_infer "$HBM"
fi
REMOTE

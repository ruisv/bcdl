#!/usr/bin/env bash
# BCDL build script — run this ON the RDK S100 / S100P board (from the repo root
# or anywhere; it cd's to the repo). Configures + builds the C++ library, the
# examples, and the Python module.
#
#   scripts/build.sh                 # configure + build into build/
#   scripts/build.sh --clean         # wipe build/ first
#   scripts/build.sh --no-python     # skip the nanobind Python module
#   scripts/build.sh --no-examples   # library + Python module only
#   scripts/build.sh --install PREFIX# also `cmake --install` to PREFIX (find_package)
#   scripts/build.sh --pip           # pip install . (build the wheel and install)
#
# Env: BCDL_ENV=<conda env name> to activate a conda env first; BUILD_TYPE
# (default Release); JOBS (default nproc).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CLEAN=0; PY=ON; EXAMPLES=ON; INSTALL_PREFIX=""; PIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --clean) CLEAN=1 ;;
    --no-python) PY=OFF ;;
    --no-examples) EXAMPLES=OFF ;;
    --install) INSTALL_PREFIX="${2:?--install needs a PREFIX}"; shift ;;
    --pip) PIP=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

say()  { printf '\033[1;36m>> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------
arch="$(uname -m)"
[ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ] || die \
  "BCDL builds on the RDK board (aarch64) only — current arch: $arch. Build on the board."

[ -n "${BCDL_ENV:-}" ] && {
  for _c in "$HOME/conda" "$HOME/miniforge3" "$HOME/miniconda3" /opt/conda; do
    [ -f "$_c/etc/profile.d/conda.sh" ] && { . "$_c/etc/profile.d/conda.sh"; break; }
  done
  say "activating conda env: $BCDL_ENV"; conda activate "$BCDL_ENV"
}

command -v cmake >/dev/null || die "cmake not found"
command -v ninja >/dev/null || die "ninja not found (install ninja-build)"
[ -e /usr/include/hobot/dnn/hb_dnn.h ] || die \
  "hobot SDK headers not found (/usr/include/hobot). This board needs the D-Robotics SDK."
[ -e /usr/hobot/lib/libdnn.so ] || die "hobot libs not found (/usr/hobot/lib/libdnn.so)."
if [ "$PY" = "ON" ]; then
  python -c "import nanobind" 2>/dev/null || die \
    "nanobind not found in the active Python env (pip/conda install nanobind), or build with --no-python"
fi

# --- configure + build -------------------------------------------------------
[ "$CLEAN" = "1" ] && { say "cleaning build/"; rm -rf build; }

say "configure (BUILD_TYPE=$BUILD_TYPE, python=$PY, examples=$EXAMPLES)"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBCDL_BUILD_PYTHON="$PY" \
  -DBCDL_BUILD_EXAMPLES="$EXAMPLES"

say "build (-j$JOBS)"
cmake --build build -j"$JOBS"

[ -n "$INSTALL_PREFIX" ] && { say "install -> $INSTALL_PREFIX"; cmake --install build --prefix "$INSTALL_PREFIX"; }
[ "$PIP" = "1" ] && { say "pip install ."; pip install .; }

say "build ok: $(ls -1 build/*.so build/det_infer 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
cat <<EOF

Next:
  models:  scripts/fetch_models.sh          # populate models/ (.hbm)
  C++:     ./build/ocr_demo data/images/ocr.jpg
  Python:  PYTHONPATH=build:python python -c "import bcdl; print(bcdl.__version__)"
  tests:   PYTHONPATH=build:python python -m pytest tests/
EOF

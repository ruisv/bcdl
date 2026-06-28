#!/usr/bin/env bash
# Populate the repo-local models/ directory from the board's source locations.
#
# Run this ON THE BOARD once (or after a fresh clone). The application code
# (tests/board_models.py, examples/ocr_demo.cc, tests/*) references models ONLY
# by the repo-relative path models/<name>.hbm — this script is the single place
# the board's absolute source paths live. The .hbm files are gitignored (266 MB
# total, incl. a 116 MB depth model), so they are not committed; re-run this to
# repopulate. Source roots are env-overridable.
#
#   scripts/fetch_models.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO}/models"

# Source roots on the board (override via env if your layout differs).
DEPLOY="${BCDL_SRC_DEPLOY:-/path/to/converted/models}"                           # your offline-converted .hbm models
ZOO="${BCDL_SRC_ZOO:-/path/to/rdk_model_zoo/samples/vision}"                      # rdk_model_zoo checkout
HOBOT="${BCDL_SRC_HOBOT:-/opt/hobot/model/s100/basic}"                            # board-shipped

mkdir -p "${DEST}"

# dest_name <TAB> source_path
copy() {
  local src="$1"
  local name
  name="$(basename "$src")"
  if [[ -f "$src" ]]; then
    cp -f "$src" "${DEST}/${name}"
    printf '  %-46s %6.1f MB\n' "$name" "$(echo "scale=1; $(stat -c%s "$src")/1048576" | bc)"
  else
    printf '  %-46s MISSING (%s)\n' "$name" "$src"
  fi
}

echo ">> fetching models into ${DEST}"
# OCR — PP-OCRv5 (converted offline in rdk-build-dev, see docs/PLAN.md finding #6)
copy "${DEPLOY}/ppocrv5_server_det_960x960.hbm"
copy "${DEPLOY}/ppocrv5_server_rec_48x320.hbm"
copy "${DEPLOY}/ppocrv5_lcnet_cls_80x160.hbm"
# Detection used by det/tracking/async demos + tests
copy "${DEPLOY}/yolo26s_det_nashm_640x640_nv12.hbm"
# YOLO26n nash-m family (cls/det/pose/seg/obb) from rdk_model_zoo
copy "${ZOO}/ultralytics_yolo26/model/nash-m/yolo26n_cls_nashm_224x224_nv12.hbm"
copy "${ZOO}/ultralytics_yolo26/model/nash-m/yolo26n_detect_nashm_640x640_nv12.hbm"
copy "${ZOO}/ultralytics_yolo26/model/nash-m/yolo26n_pose_nashm_640x640_nv12.hbm"
copy "${ZOO}/ultralytics_yolo26/model/nash-m/yolo26n_seg_nashm_640x640_nv12.hbm"
copy "${ZOO}/ultralytics_yolo26/model/nash-m/yolo26n_obb_nashm_640x640_nv12.hbm"
# Depth (Depth-Anything-V2) from rdk_model_zoo
copy "${ZOO}/depth_anything_v2/model/s100/depth_any.hbm"
# Stereo disparity (Lite Any Stereo V2, int16, 480x640) — converted offline.
# Two calibration modes: *_int16 (resize-fit) and *_crop (center-crop-fit).
# The stereo board test prefers the crop model on the repo's center-crop pair.
LAS2_DIR="${BCDL_SRC_LAS2:-/path/to/converted/las2}"
copy "${LAS2_DIR}/las2_m_int16_nashm.hbm"
copy "${LAS2_DIR}/las2_m_crop_nashm.hbm"
# Board-shipped: DFL det (yolov8) + semantic seg (deeplabv3plus)
copy "${HOBOT}/yolov8_640x640_nv12.hbm"
copy "${HOBOT}/deeplabv3plus_dilation1248_1024x2048_nv12.hbm"
echo ">> done. ($(ls -1 "${DEST}"/*.hbm 2>/dev/null | wc -l) models)"

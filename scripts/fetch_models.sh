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

# Models converted on the x86 host are staged into ${DEPLOY} on the board. That
# staging is what silently went stale before 2026-07-19: five M6/M8 models were
# only ever on the convert host, so a re-run produced a models/ dir that could
# not run those tests. Rather than trust the staging, treat the convert host as
# the source of truth and SELF-HEAL: use the board-local copy when it is there,
# otherwise pull it from the convert host and stage it for next time.
CONVERT_HOST="${BCDL_CONVERT_HOST:-convert-host}"
BUILD="${BCDL_SRC_BUILD:-/path/to/convert-project/models}"

# copy_staged <name> <path-under-BUILD-on-convert-host>
copy_staged() {
  local name="$1" remote="$2"
  if [[ ! -f "${DEPLOY}/${name}" ]]; then
    if scp -q -o ConnectTimeout=10 -o BatchMode=yes \
        "${CONVERT_HOST}:${BUILD}/${remote}" "${DEPLOY}/${name}.part" 2>/dev/null; then
      # Rename only after a complete transfer: a half-copied .hbm staged under
      # the real name would be indistinguishable from a good one next run.
      mv -f "${DEPLOY}/${name}.part" "${DEPLOY}/${name}"
      printf '  %-46s staged from %s\n' "$name" "${CONVERT_HOST}"
    else
      rm -f "${DEPLOY}/${name}.part"
      printf '  %-46s MISSING (%s:%s/%s)\n' "$name" "${CONVERT_HOST}" "${BUILD}" "$remote"
      return
    fi
  fi
  copy "${DEPLOY}/${name}"
}

echo ">> fetching models into ${DEST}"
# OCR — PP-OCRv5 (converted offline from ccdl ONNX)
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
# M6 open-vocab (YOLOE) + M8 promptable seg (EdgeSAM) — self-healing from the
# convert host (see copy_staged above). (M9's rtmpose_m_body is deliberately NOT
# fetched: that milestone was shelved and its decoder removed from the repo, so
# nothing here uses the model.)
copy_staged yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm \
            yoloe/yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm
copy_staged yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm \
            yoloe/yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm
copy_staged edge_sam_encoder_nashm.hbm     edgesam/enc_out/edge_sam_encoder_nashm.hbm
copy_staged edge_sam_decoder_sp1_nashm.hbm edgesam/dec_out/edge_sam_decoder_sp1_nashm.hbm
copy_staged edge_sam_decoder_bp2_nashm.hbm edgesam/dec_bp_out/edge_sam_decoder_bp2_nashm.hbm
echo ">> done. ($(ls -1 "${DEST}"/*.hbm 2>/dev/null | wc -l) models)"

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

# Your local source roots go in scripts/local.env (gitignored), e.g.
#   BCDL_SRC_DEPLOY=/your/path/to/converted/models
# Anything not set there falls back to the neutral placeholders below and is
# simply reported as MISSING.
# shellcheck disable=SC1091
[ -f "${REPO}/scripts/local.env" ] && . "${REPO}/scripts/local.env"

# Source roots on the board (override via scripts/local.env or the environment).
DEPLOY="${BCDL_SRC_DEPLOY:-/path/to/converted/models}"                            # offline-converted .hbm
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

# Some models are official PREBUILT downloads rather than anything converted
# here, so there is no local source to copy from — fetch them straight from the
# public model-zoo archive. Downloads land on a .part first for the same reason
# copy_staged does: a truncated transfer under the real name is indistinguishable
# from a good model on the next run.
download() {
  local name="$1" url="$2"
  if [[ -f "${DEST}/${name}" ]]; then
    printf '  %-46s %6.1f MB (already present)\n' "$name" \
      "$(echo "scale=1; $(stat -c%s "${DEST}/${name}")/1048576" | bc)"
    return
  fi
  if command -v curl >/dev/null && \
     curl -fsSL --retry 3 -o "${DEST}/${name}.part" "$url"; then
    mv -f "${DEST}/${name}.part" "${DEST}/${name}"
    printf '  %-46s %6.1f MB (downloaded)\n' "$name" \
      "$(echo "scale=1; $(stat -c%s "${DEST}/${name}")/1048576" | bc)"
  else
    rm -f "${DEST}/${name}.part"
    printf '  %-46s MISSING (download failed: %s)\n' "$name" "$url"
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
# OCR — PP-OCRv6 (converted offline; recipes in bcdl-model-zoo). This is the
# default stack for the demo and tests. The rec build is all-int16: faster and
# more accurate than the compiler's default mixed-precision int8 (see
# docs/MODELS.md). No v6 classifier exists, so the v5 PP-LCNet cls stays below.
copy "${DEPLOY}/ppocrv6_medium_det_960x960.hbm"
copy "${DEPLOY}/ppocrv6_medium_rec_int16_48x320.hbm"
# PP-OCRv5 stack — kept as a fallback and for the wider textline classifier the
# v6 line does not provide.
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
# M11 face: SCRFD detection + ArcFace recognition. The recognition build to take
# is the one calibrated on ALIGNED crops — it is what the on-board accuracy
# figures were measured with, and it is a different file from the same-shaped
# build calibrated on centre crops.
copy_staged scrfd_10g_nashm_640x640_nv12.hbm face/out_det/scrfd_10g_nashm_640x640_nv12.hbm
copy_staged arcface_r50_aligned_nashm_112x112.hbm \
            face/out_rec2/arcface_r50_aligned_nashm_112x112.hbm
# M13 panoptic driving (det + drivable area + lane lines). Take the "_cut"
# build: the published export bakes the anchor decode into the graph via
# ScatterND, which compiles with no error into a model whose objectness and class
# columns are never written. Cutting the graph before that decode and emitting
# the three raw heads is what works.
copy_staged yolop_cut_nashm_640x640_nv12.hbm yolop/out_cut/yolop_cut_nashm_640x640_nv12.hbm
# M14 real-time semantic segmentation — replaces deeplabv3plus for the semseg_rt
# task; the deeplabv3plus copy above is kept as the slower baseline. Take "_v3":
# the earlier builds were calibrated on data that had not been pre-normalized,
# which also compiles cleanly and decodes to garbage.
copy_staged pidnet_s_nashm_1024x2048_nv12_v3.hbm \
            pidnet/out_v3/pidnet_s_nashm_1024x2048_nv12_v3.hbm
# M15 whole-body pose (ViTPose-S, 133 keypoints). Top-down: it needs the YOLO
# person detector above in front of it.
copy_staged vitpose_s_wholebody_nashm_256x192.hbm \
            vitpose/out/vitpose_s_wholebody_nashm_256x192.hbm
# M16 sparse local features (XFeat, 640x480).
copy_staged xfeat_nashm_640x480.hbm xfeat/out/xfeat_nashm_640x480.hbm
# M10 person ReID (OSNet-AIN, 256x128 crops -> 512-d). This is the appearance
# half of tracking: the YOLO detector above supplies the boxes, this turns each
# crop into a vector, and ByteTracker's two-argument update() associates on both.
# Take the "_qat" build and only that one. int8 PTQ does not work on this
# network at all — it compiles cleanly and returns well-formed unit vectors whose
# Market-1501 Rank-1 is 51% against the float model's 85%. The shipped model was
# recovered by QAT self-distillation back to 84.6%.
copy_staged osnet_ain_qat_nashm_256x128.hbm \
            osnet/osnet_ain_qat_nashm_256x128.hbm
# M17 x4 super-resolution. The 128 tile is deliberate: the compiled instruction
# stream scales with tile AREA, so the same net at 256 is a 148 MB .hbm against
# 37 MB here for identical per-pixel throughput.
copy_staged realesr_general_x4v3_nashm_128.hbm \
            superres/out_128/realesr_general_x4v3_nashm_128.hbm
# M17 second upscaler. Not a replacement for the one above: SPAN is
# fidelity-oriented and wins on clean input at a sixth of the size, while the
# Compact model is perceptual and wins on blurred / compressed input.
copy_staged spanx4_ch48_nashm_128.hbm span/out/spanx4_ch48_nashm_128.hbm
# M12 image embeddings (SigLIP). This one is an official PREBUILT model rather
# than anything converted here, so prefer a copy already sitting in the model zoo
# checkout and otherwise pull it from the public archive.
SIGLIP="bpu-siglip-base-patch16-224.hbm"
if [[ -f "${ZOO}/siglip/model/s100/${SIGLIP}" ]]; then
  copy "${ZOO}/siglip/model/s100/${SIGLIP}"
else
  download "${SIGLIP}" \
    "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/${SIGLIP}"
fi
echo ">> done. ($(ls -1 "${DEST}"/*.hbm 2>/dev/null | wc -l) models)"

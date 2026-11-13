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

# copy_or_hf <local-path> <hf-repo> [how-to-build] — prefer a locally staged
# build (a maintainer rebuilding a model wants their new one, and anyone who
# built it from its recipe points BCDL_SRC_* at the output), else fetch the
# published copy. When neither exists, say how to build it.
copy_or_hf() {
  local src="$1" repo="$2" hint="${3:-}" name
  name="$(basename "$src")"
  if [[ -f "$src" ]]; then copy "$src"; else from_hf "$name" "$repo"; fi
  if [[ ! -f "${DEST}/${name}" && -n "$hint" ]]; then
    printf '  %-46s   -> %s\n' "" "$hint"
  fi
}

# from_hf <name> <hf-repo> — a converted model published on Hugging Face.
#
# These are the builds whose WEIGHTS are redistributable, so anyone can have them
# without a conversion host and without running hb_compile. Falls through to
# copy_staged when the download fails, so a maintainer with the convert host
# still gets a model that has not been published (or has just been rebuilt).
from_hf() {
  local name="$1" repo="$2" staged="${3:-}"
  if [[ -f "${DEST}/${name}" ]]; then
    printf '  %-46s %6.1f MB (already present)\n' "$name" \
      "$(echo "scale=1; $(stat -c%s "${DEST}/${name}")/1048576" | bc)"
    return
  fi
  local url="https://huggingface.co/${repo}/resolve/main/${name}?download=true"
  if command -v curl >/dev/null && \
     curl -fsSL --retry 3 -o "${DEST}/${name}.part" "$url" 2>/dev/null; then
    mv -f "${DEST}/${name}.part" "${DEST}/${name}"
    printf '  %-46s %6.1f MB (Hugging Face: %s)\n' "$name" \
      "$(echo "scale=1; $(stat -c%s "${DEST}/${name}")/1048576" | bc)" "$repo"
  else
    rm -f "${DEST}/${name}.part"
    if [[ -n "$staged" ]]; then
      copy_staged "$name" "$staged"
    else
      printf '  %-46s MISSING (%s)\n' "$name" "$repo"
    fi
  fi
}


echo ">> fetching models into ${DEST}"
# OCR — PP-OCRv6 (converted offline; recipes in bcdl-model-zoo). This is the
# default stack for the demo and tests. The rec build is all-int16: faster and
# more accurate than the compiler's default mixed-precision int8 (see
# docs/MODELS.md). No v6 classifier exists, so the v5 PP-LCNet cls stays below.
copy_or_hf "${DEPLOY}/ppocrv6_medium_det_960x960.hbm" ruisv/bcdl-ppocr-v6
copy_or_hf "${DEPLOY}/ppocrv6_medium_rec_int16_48x320.hbm" ruisv/bcdl-ppocr-v6
# PP-OCRv5 stack — kept as a fallback and for the wider textline classifier the
# v6 line does not provide.
copy_or_hf "${DEPLOY}/ppocrv5_server_det_960x960.hbm" ruisv/bcdl-ppocr-v5
copy_or_hf "${DEPLOY}/ppocrv5_server_rec_48x320.hbm" ruisv/bcdl-ppocr-v5
copy_or_hf "${DEPLOY}/ppocrv5_lcnet_cls_80x160.hbm" ruisv/bcdl-ppocr-v6
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
# Two calibration modes: *_crop2 (center-crop-fit) and *_resize2 (resize-fit),
# each in two sizes: M (67 FPS) and L (42 FPS, more accurate indoors).
# The "2" builds are calibrated across the full 0..192 px disparity range; the
# earlier *_crop / *_int16 builds clipped the output at 131 / 47.7 px.
# The stereo board test prefers the M crop model on the repo's center-crop pair.
LAS2_DIR="${BCDL_SRC_LAS2:-/path/to/converted/las2}"
copy_or_hf "${LAS2_DIR}/las2_m_crop2_nashm.hbm" ruisv/bcdl-las2-stereo
copy_or_hf "${LAS2_DIR}/las2_m_resize2_nashm.hbm" ruisv/bcdl-las2-stereo
copy_or_hf "${LAS2_DIR}/las2_l_crop2_nashm.hbm" ruisv/bcdl-las2-stereo
copy_or_hf "${LAS2_DIR}/las2_l_resize2_nashm.hbm" ruisv/bcdl-las2-stereo
# RGB-D depth refinement (LingBot-Depth v0.5, 480x640) — converted offline.
# Refines an existing noisy/holey depth map rather than estimating one; pairs
# with either of the two models above as the depth source.
LINGBOT_DIR="${BCDL_SRC_LINGBOT:-/path/to/converted/lingbot-depth}"
copy_or_hf "${LINGBOT_DIR}/lingbot_depth_v05_int16_nashm.hbm" ruisv/bcdl-lingbot-depth
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
# Indoor scenes: YOLOE-26m with a fixed 99-class indoor vocabulary (navigation
# landmarks + household objects, boxes + instance masks) and its class list, plus
# the Places365 ResNet50 scene classifier. The Places365 class names and the
# per-class indoor/outdoor flag come straight from the upstream repo.
copy_staged yoloe26m_indoor99_seg_nashm_640x640_nv12.hbm \
            yoloe26_indoor/out_int8/yoloe26m_indoor99_seg_nashm_640x640_nv12.hbm
copy_staged yoloe26m_indoor99_labels.txt yoloe26_indoor/yoloe26m_indoor99_labels.txt
copy_staged places365_resnet50_nashm_224x224_nv12.hbm \
            places365/out_int8/places365_resnet50_nashm_224x224_nv12.hbm
download categories_places365.txt \
  https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt
download IO_places365.txt \
  https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt
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
from_hf yolop_cut_nashm_640x640_nv12.hbm ruisv/bcdl-yolop \
            yolop/out_cut/yolop_cut_nashm_640x640_nv12.hbm
# M14 real-time semantic segmentation — replaces deeplabv3plus for the semseg_rt
# task; the deeplabv3plus copy above is kept as the slower baseline. Take "_v3":
# the earlier builds were calibrated on data that had not been pre-normalized,
# which also compiles cleanly and decodes to garbage.
from_hf pidnet_s_nashm_1024x2048_nv12_v3.hbm ruisv/bcdl-pidnet \
        pidnet/out_v3/pidnet_s_nashm_1024x2048_nv12_v3.hbm
# YOLO26n-sem semantic segmentation (Cityscapes 19-class, native 640x640, no
# caller-side upsample needed unlike semseg_rt). Take the int16 build,
# calibrated on real Cityscapes frames: int8 compiles clean and passes every
# cosine gate but only agrees with the float model on 61% of pixel-level class
# decisions; int16 alone gets to 85%, real-domain calibration to 96% (see
# docs/MODELS.md).
copy_staged yolo26n_sem_nashm_640x640_nv12.hbm \
            yolo26_sem/out_int16/yolo26n_sem_nashm_640x640_nv12.hbm
# M15 whole-body pose (ViTPose-S, 133 keypoints). Top-down: it needs the YOLO
# person detector above in front of it.
from_hf vitpose_s_wholebody_nashm_256x192.hbm ruisv/bcdl-vitpose-wholebody \
        vitpose/out/vitpose_s_wholebody_nashm_256x192.hbm
# M16 sparse local features (XFeat, 640x480).
from_hf xfeat_nashm_640x480.hbm ruisv/bcdl-xfeat \
            xfeat/out/xfeat_nashm_640x480.hbm
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
from_hf realesr_general_x4v3_nashm_128.hbm ruisv/bcdl-superres \
        superres/out_128/realesr_general_x4v3_nashm_128.hbm
# M17 second upscaler. Not a replacement for the one above: SPAN is
# fidelity-oriented and wins on clean input at a sixth of the size, while the
# Compact model is perceptual and wins on blurred / compressed input.
from_hf spanx4_ch48_nashm_128.hbm ruisv/bcdl-superres \
            span/out/spanx4_ch48_nashm_128.hbm
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
# M20 lidar 3-D detection (PointPillars). Three compiled stages: the published
# VFE and 2-D backbone are downloaded as-is, the anchor head is compiled HERE
# because the published one bakes its anchor decode into the graph with
# ScatterND — the construct that compiles without an error into a model whose
# outputs are never written (this repo already lost a day to it once, on YOLOP).
# The two .npy are the zoo's own sample sweep; the points are recoverable from
# them, so the whole chain can be driven from a real scan with nothing staged.
PP="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointPillars"
download pointpillars_vfe.hbm              "${PP}/vfe.hbm"
download pointpillars_backbone.hbm         "${PP}/backbone.hbm"
download lidar_voxels0.npy                 "${PP}/voxels0.npy"
download lidar_voxel_num_points0.npy       "${PP}/voxel_num_points0.npy"
copy_staged pointpillars_head_int16_nashm_248x216.hbm \
            pointpillars/out/pointpillars_head_int16_nashm_248x216.hbm

# One CALIBRATED frame, which is what upgrades the check figure from "the boxes
# sit on some points" to "the boxes wrap those cars, and here is what was missed".
# The dataset is not redistributed — KITTI is CC BY-NC-SA — so this copies four
# files out of a KITTI object-detection checkout you already have. Set
# BCDL_SRC_KITTI (and optionally BCDL_KITTI_FRAME, default 000736) in
# scripts/local.env. Without it everything still works; the figure just falls
# back to bird's-eye only. See docs/MODELS.md.
KITTI_SRC="${BCDL_SRC_KITTI:-}"
KITTI_ID="${BCDL_KITTI_FRAME:-000736}"
if [[ -n "$KITTI_SRC" && -d "$KITTI_SRC" ]]; then
  mkdir -p "${DEPLOY}/kitti_frame"
  ok=1
  for pair in "velodyne/${KITTI_ID}.bin:velodyne.bin" \
              "image_2/${KITTI_ID}.png:image_2.png" \
              "calib/${KITTI_ID}.txt:calib.txt" \
              "label_2/${KITTI_ID}.txt:label_2.txt"; do
    src="${KITTI_SRC}/${pair%%:*}"; dst="${DEPLOY}/kitti_frame/${pair##*:}"
    if [[ -f "$src" ]]; then cp -f "$src" "$dst"; else ok=0; fi
  done
  echo "$KITTI_ID" > "${DEPLOY}/kitti_frame/frame_id.txt"
  if [[ $ok -eq 1 ]]; then
    printf '  %-46s staged frame %s\n' "kitti_frame/" "$KITTI_ID"
  else
    # label_2 is absent for the test split; the figure copes, so say what is
    # missing rather than failing.
    printf '  %-46s partial (frame %s; camera view needs velodyne+image_2+calib)\n' \
           "kitti_frame/" "$KITTI_ID"
  fi
else
  printf '  %-46s SKIPPED (set BCDL_SRC_KITTI for the camera view)\n' "kitti_frame/"
fi

# M19 one-class anomaly detection, MVTec-AD "bottle". This one is an EXAMPLE,
# not a general detector: the model is trained on a single product's normal
# images, so deploying to another part means retraining the head. int8 is
# deliberate — the all-int16 build of this net runs 91x slower for no gain on
# image or pixel AUROC.
from_hf glass_bottle_int8_nashm_288x288.hbm ruisv/bcdl-glass-bottle \
        glass/out_int8/glass_bottle_int8_nashm_288x288.hbm
# Bench samples for the two tasks whose input is not a repo image: a one-class
# model only means anything on its own product, and the driving model eats a
# sensor rig rather than a photo. Both samples are dataset-derived (MVTec AD /
# nuScenes, non-redistributable), which is exactly why they stage here next to
# the models instead of being committed under data/images/.
copy_staged anomaly_sample_defect.png samples/anomaly_sample_defect.png
copy_staged anomaly_sample_good.png   samples/anomaly_sample_good.png
copy_staged drive_sample_frame.npz    samples/drive_sample_frame.npz

# M18 dense optical flow. int16 is not a preference here: the int8 build of this
# net adds a flat ~1.9 px noise floor that the compiler's 0.9954 output cosine
# does not report, which is most of the signal on slow-moving pixels.
copy_staged neuflow_v2_int16_nashm_512x384.hbm \
            neuflow/out_int16_named/neuflow_v2_int16_nashm_512x384.hbm

# End-to-end driving (camera + lidar -> trajectory). Another official PREBUILT,
# and the first one published per BOARD rather than per march — take the build
# that matches the board you are on. It ships with no companion assets, so the
# clustered trajectory anchors the diffusion decoder starts from come from the
# upstream release; without them the planner has nothing to denoise.
#
# Licence chain, since these are redistributable files and not just weights we
# run: the upstream code is MIT, but both the anchors and the compiled model
# derive from driving datasets whose terms are non-commercial. Fetching them for
# local use is what this script does; shipping them onward is a separate
# question with a different answer.
DRIVE_HBM="diffusiondrive_r34_256x1024_s100p.hbm"
download "${DRIVE_HBM}" \
  "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/${DRIVE_HBM}"
download "kmeans_navsim_traj_20.npy" \
  "https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy"
# Speech recognition (Paraformer-large, Chinese). An official PREBUILT split
# into three stages with CIF on the CPU between the last two. The published file
# names are generic (encoder_int16.hbm, ...), so each is saved under its internal
# model name — prefixed where that name does not say what it belongs to — so the
# _v2 the compiler stamped on two of them stays visible. The token list comes
# with the build; the CMVN statistics come from the upstream model, which is
# where the build's own front end takes them from.
PARA="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paraformer/nash-e"
download paraformer_encoder_int16.hbm      "${PARA}/encoder_int16.hbm"
download paraformer_predictor_int16_v2.hbm "${PARA}/predictor_int16.hbm"
download paraformer_decoder_int16_v2.hbm   "${PARA}/decoder_int16.hbm"
download paraformer_tokens_8404.json       "${PARA}/tokens.json"
download paraformer_am.mvn \
  "https://modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/resolve/master/am.mvn"
# Voice activity detection: Silero VAD v6.2.3 (16 kHz), converted offline at
# int16 as one fixed-shape step of the streaming model: a 512-sample chunk with
# its 64 samples of context in, the LSTM state in and out. The context and the
# reflection pad are done on the CPU.
VAD_DIR="${BCDL_SRC_VAD:-/path/to/converted/silero_vad}"
copy_or_hf "${VAD_DIR}/silero_vad_16k_int16_nashm.hbm" ruisv/bcdl-silero-vad \
  "build it with bcdl-model-zoo models/silero_vad, then BCDL_SRC_VAD=<that dir> scripts/fetch_models.sh"
# Keyword spotting (wake word): sherpa-onnx's zipformer2 KWS model (zh-en, 3M,
# chunk 16 / left 64). The encoder is converted offline at int16 after a graph
# rewrite (the int64 processed_lens becomes a float pad_mask, the softplus is
# made fixed-point safe, the relative-position gathers become slices); the
# decoder and joiner run on the CPU from one weight table.
KWS_DIR="${BCDL_SRC_KWS:-/path/to/converted/kws}"
copy_or_hf "${KWS_DIR}/kws_zipformer_zh_en_3m_c16_int16_nashm.hbm" ruisv/bcdl-kws-zipformer \
  "build it with bcdl-model-zoo models/kws_zipformer, then BCDL_SRC_KWS=<that dir> scripts/fetch_models.sh"
copy_or_hf "${KWS_DIR}/kws_zipformer_zh_en_3m_decoder_joiner.npy" ruisv/bcdl-kws-zipformer \
  "build it with bcdl-model-zoo models/kws_zipformer, then BCDL_SRC_KWS=<that dir> scripts/fetch_models.sh"
copy_or_hf "${KWS_DIR}/kws_zipformer_zh_en_3m_tokens.txt" ruisv/bcdl-kws-zipformer \
  "build it with bcdl-model-zoo models/kws_zipformer, then BCDL_SRC_KWS=<that dir> scripts/fetch_models.sh"
# Speech synthesis back end: Vocos (16 kHz, n_fft 1024 / hop 256) for the
# Matcha-TTS zh-en acoustic model, converted offline at int16 with a fixed
# 256-frame window; VocosVocoder chunks longer mels with exact seams. The graph
# stops at the head's log-magnitude and phase - exp/cos/sin and the iSTFT run
# on the CPU, because the compiler's quantised cos/sin ruin the phase.
VOCOS_DIR="${BCDL_SRC_VOCOS:-/path/to/converted/vocos}"
copy_or_hf "${VOCOS_DIR}/vocos_16k_t256_int16_nashm.hbm" ruisv/bcdl-vocos \
  "build it with bcdl-model-zoo models/vocos, then BCDL_SRC_VOCOS=<that dir> scripts/fetch_models.sh"
# Speech synthesis acoustic model: Matcha-TTS zh-en, two int16 graphs (encoder
# with the duration predictor, 128 tokens; decoder with three flow-matching
# steps, 512 frames = 8.2 s), plus the front end's token table and lexicon.
# The decoder's GroupNorm and attention mask are rewritten so that padding to
# 512 frames reproduces the unpadded model; a stock fixed-shape export does not.
# The token embedding lookup runs on the CPU (the table is the .npy), because a
# float token-id input gets quantised like any activation.
MATCHA_DIR="${BCDL_SRC_MATCHA:-/path/to/converted/matcha}"
copy_or_hf "${MATCHA_DIR}/matcha_zh_en_enc_l128_int16_nashm.hbm" ruisv/bcdl-matcha-tts \
  "build it with bcdl-model-zoo models/matcha_tts, then BCDL_SRC_MATCHA=<that dir> scripts/fetch_models.sh"
copy_or_hf "${MATCHA_DIR}/matcha_zh_en_dec_t512_int16_nashm.hbm" ruisv/bcdl-matcha-tts \
  "build it with bcdl-model-zoo models/matcha_tts, then BCDL_SRC_MATCHA=<that dir> scripts/fetch_models.sh"
copy_or_hf "${MATCHA_DIR}/matcha_zh_en_embedding.npy" ruisv/bcdl-matcha-tts \
  "build it with bcdl-model-zoo models/matcha_tts, then BCDL_SRC_MATCHA=<that dir> scripts/fetch_models.sh"
copy_or_hf "${MATCHA_DIR}/matcha_zh_en_tokens.txt" ruisv/bcdl-matcha-tts \
  "build it with bcdl-model-zoo models/matcha_tts, then BCDL_SRC_MATCHA=<that dir> scripts/fetch_models.sh"
copy_or_hf "${MATCHA_DIR}/matcha_zh_en_lexicon.txt" ruisv/bcdl-matcha-tts \
  "build it with bcdl-model-zoo models/matcha_tts, then BCDL_SRC_MATCHA=<that dir> scripts/fetch_models.sh"
echo ">> done. ($(ls -1 "${DEST}"/*.hbm 2>/dev/null | wc -l) models)"

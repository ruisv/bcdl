"""Shared on-board real-model harness for bcdl's task heads.

One registry + NV12 preprocessing + per-task decode/draw for the YOLO26n
(nash-m / S100P) model family, which exactly matches bcdl's LTRB decoders:

  cls  : yolo26n_cls   224x224  -> Classifier (softmax top-k)
  det  : yolo26n_detect 640x640 -> YoloLtrbDetector (6 outputs)
  pose : yolo26n_pose   640x640 -> PoseEstimator   (9 outputs, 17 kpts)
  seg  : yolo26n_seg    640x640 -> InstanceSegmenter (10 outputs, 32-coef proto)
  obb  : yolo26n_obb    640x640 -> ObbDetector      (9 outputs, DOTA 15-class)

All are 2-input NV12 (Y plane + interleaved UV). Used by both
tests/test_board_models.py (pytest assertions) and scripts/board_bench.py
(latency / FPS / memory + check figures). Board-only: imports bcdl + cv2 lazily.

Sample images + a CJK font live in the repo (data/images, data/fonts; copied
from ccdl examples/) so the check figures are reproducible without board-only
/app/res/assets; cls (zebra) and semseg (cityscapes) have no ccdl counterpart
and fall back to the board assets. Model dir / images / font / assets are all
env-overridable (BCDL_YOLO26_DIR / BCDL_IMAGES / BCDL_FONT / BCDL_ASSETS).
"""

import os

import numpy as np

import bev_svg as bv
import kitti_frame

# All models live in the repo-local models/ dir (populated by
# scripts/fetch_models.sh from the board's source locations) — no absolute paths.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
YOLO26_DIR = os.environ.get("BCDL_YOLO26_DIR", MODELS)
ASSETS = os.environ.get("BCDL_ASSETS", "/app/res/assets")
# Repo-local sample images + CJK font (copied from ccdl examples/), so the check
# figures are reproducible without depending on board-only /app/res/assets.
IMAGES = os.environ.get("BCDL_IMAGES", os.path.join(_REPO, "data", "images"))
FONT = os.environ.get("BCDL_FONT", os.path.join(_REPO, "data", "fonts", "SourceHanSansSC-Regular.otf"))


def find_image(name):
    """Resolve a test image: repo data/images first, else board /app/res/assets.
    cls (zebra) and semseg (cityscapes) have no ccdl counterpart and stay on the
    board; det/pose/seg/obb/depth/ocr use the repo-local ccdl samples."""
    if os.path.isabs(name):
        return name
    p = os.path.join(IMAGES, name)
    return p if os.path.exists(p) else os.path.join(ASSETS, name)


# OBB is DOTA-trained (aerial view) — a ground photo yields no detections, so use
# the ccdl DOTA sample. Override with BCDL_OBB_IMAGE.
OBB_IMAGE = os.environ.get("BCDL_OBB_IMAGE", "obb.jpg")

# COCO-80 (det/pose/seg) and DOTA-15 (obb) class names for readable figures.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

DOTA_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


SEMSEG_MODEL = os.environ.get(
    "BCDL_SEMSEG_MODEL",
    os.path.join(MODELS, "deeplabv3plus_dilation1248_1024x2048_nv12.hbm"))
# "_v3": earlier builds of this model were calibrated on data that had not been
# pre-normalized, which compiles cleanly and decodes to garbage on the board. The
# version suffix is kept in the file name so it is obvious which build is loaded.
SEMSEG_RT_MODEL = os.environ.get(
    "BCDL_SEMSEG_RT_MODEL",
    os.path.join(MODELS, "pidnet_s_nashm_1024x2048_nv12_v3.hbm"))
# YOLO26n-sem: same Cityscapes 19-class task as the two above, at the detector's
# native 640x640. Full-res dense head (no caller-side upsample needed, unlike
# semseg_rt) -- see bcdl-model-zoo/models/yolo26_sem/. Ships int16, calibrated on
# real Cityscapes frames: the all-int8 build passes every per-tensor cosine gate
# and still only agrees with the float model on 61% of pixel-level argmax
# decisions (whole classes swap); int16 alone recovers to 85%, and switching the
# PTQ calibration set from a stand-in domain to real Cityscapes frames gets to
# 96%. See that recipe's README for the measured numbers.
SEMSEG_YOLO26_MODEL = os.environ.get(
    "BCDL_SEMSEG_YOLO26_MODEL",
    os.path.join(MODELS, "yolo26n_sem_nashm_640x640_nv12.hbm"))
DEPTH_MODEL = os.environ.get(
    "BCDL_DEPTH_MODEL",
    os.path.join(MODELS, "depth_any.hbm"))

# Stereo (LAS2) — two-image F32 NCHW RGB inputs -> disparity. Two calibration
# modes shipped by fetch_models.sh: crop (matches the repo's center-crop pair)
# preferred, else resize. The pair lives in data/images/ (committed losslessly).
STEREO_CROP_MODEL = os.environ.get("BCDL_STEREO_CROP_MODEL",
                                   os.path.join(MODELS, "las2_m_crop_nashm.hbm"))
STEREO_RESIZE_MODEL = os.environ.get("BCDL_STEREO_RESIZE_MODEL",
                                     os.path.join(MODELS, "las2_m_int16_nashm.hbm"))
STEREO_LEFT = os.environ.get("BCDL_STEREO_LEFT", os.path.join(IMAGES, "stereo_left.png"))
STEREO_RIGHT = os.environ.get("BCDL_STEREO_RIGHT", os.path.join(IMAGES, "stereo_right.png"))


def model_path(name):
    return name if os.path.isabs(name) else os.path.join(YOLO26_DIR, name)


# OCR uses the LATEST PP-OCRv5 stack (converted offline from ccdl ONNX and
# deployed to the board), all single-input F32 RGB NCHW featuremap
# (normalisation done in preproc, not fused), env-overridable:
#   det: PP-OCRv6 medium det   [1,3,960,960] -> prob map [1,1,960,960]
#   rec: PP-OCRv6 medium rec (int16) [1,3,48,320] -> [1,T,18710] softmax (CTC)
#   cls: PP-LCNet_x1_0_textline_ori [1,3,80,160] -> [1,2] (0°/180°) — still v5,
#        PP-OCRv6 ships no classifier. The decoder reads the class count from the
#        model, so v5→v6 needs only the model + dict swap, no code change.
OCR_DET = os.environ.get("BCDL_OCR_DET", os.path.join(MODELS, "ppocrv6_medium_det_960x960.hbm"))
OCR_REC = os.environ.get("BCDL_OCR_REC", os.path.join(MODELS, "ppocrv6_medium_rec_int16_48x320.hbm"))
# Direction classifier — OPTIONAL: the pipeline skips the 180° flip when absent.
OCR_CLS = os.environ.get("BCDL_OCR_CLS", os.path.join(MODELS, "ppocrv5_lcnet_cls_80x160.hbm"))
OCR_DICT = os.environ.get("BCDL_OCR_DICT",
                          os.path.join(_REPO, "data", "ppocr_keys_v6_18710.txt"))
OCR_IMAGE = os.environ.get("BCDL_OCR_IMAGE", "ocr.jpg")

# --- whole-body pose: TOP-DOWN, so it needs a person detector in front of it ---
WHOLEBODY_MODEL = os.environ.get(
    "BCDL_WHOLEBODY_MODEL", os.path.join(MODELS, "vitpose_s_wholebody_nashm_256x192.hbm"))
WHOLEBODY_DET = os.environ.get(
    "BCDL_WHOLEBODY_DET", os.path.join(MODELS, "yolo26n_detect_nashm_640x640_nv12.hbm"))
WHOLEBODY_IMAGE = os.environ.get("BCDL_WHOLEBODY_IMAGE", "zidane.jpg")

# --- sparse local features (XFeat): a PAIR task — extract twice, then match ---
FEATURES_MODEL = os.environ.get(
    "BCDL_FEATURES_MODEL", os.path.join(MODELS, "xfeat_nashm_640x480.hbm"))
# The stereo pair doubles as the feature pair: two real views of one scene, and
# already in the repo.
FEATURES_LEFT = os.environ.get("BCDL_FEATURES_LEFT", STEREO_LEFT)
FEATURES_RIGHT = os.environ.get("BCDL_FEATURES_RIGHT", STEREO_RIGHT)

# --- super-resolution: a x4 upscaler applied by tiling -----------------------
# The 128 tile is deliberate. The compiled instruction stream scales with tile
# AREA, so the same network at 256 is a 148 MB .hbm against 37 MB here, for
# identical per-pixel throughput — and since the upscaler tiles in software
# anyway, the smaller tile costs only a little extra overlap redundancy.
#
# Two upscalers ship, and they are NOT redundant — they were trained for
# different inputs, which shows up as a clean split on the board:
#   Compact (realesr-general): perceptual, trained on real-world degradations.
#             Wins on blurred/compressed input; loses to bicubic on PSNR for a
#             clean downscale because it sharpens and invents detail.
#   SPAN:     fidelity-oriented. Beats bicubic on a clean downscale, is 6x
#             smaller and ~2x faster, and gives ground to Compact on degraded
#             input.
# Both are driven by the same SuperResolver — scale and tile come from the
# model — so switching is a path change and nothing else.
SUPERRES_MODEL = os.environ.get(
    "BCDL_SUPERRES_MODEL", os.path.join(MODELS, "realesr_general_x4v3_nashm_128.hbm"))
SUPERRES_SPAN_MODEL = os.environ.get(
    "BCDL_SUPERRES_SPAN_MODEL", os.path.join(MODELS, "spanx4_ch48_nashm_128.hbm"))
SUPERRES_IMAGE = os.environ.get("BCDL_SUPERRES_IMAGE", "bus.jpg")

# --- person ReID: the appearance half of tracking ---------------------------
# Takes a DETECTION CROP, not a frame: the person detector above supplies the
# boxes and this turns each crop into a 512-d vector, which ByteTracker's
# two-argument update() associates on alongside geometry. Input is 256x128
# (squashed, not letterboxed) and ImageNet-normalized — see bcdl.reid_preprocess.
REID_MODEL = os.environ.get(
    "BCDL_REID_MODEL", os.path.join(MODELS, "osnet_ain_qat_nashm_256x128.hbm"))
REID_DET = os.environ.get(
    "BCDL_REID_DET", os.path.join(MODELS, "yolo26n_detect_nashm_640x640_nv12.hbm"))
REID_IMAGE = os.environ.get("BCDL_REID_IMAGE", "bus.jpg")


# --- optical flow: a PAIR task, and the only one whose output is a field ------
# The stereo pair doubles as the flow pair (same two views XFeat matches), so no
# new asset is needed: horizontal disparity IS the displacement field, which
# makes a colour-wheel figure that is easy to sanity-check — the whole scene
# should push one way, with magnitude falling off with distance.
FLOW_MODEL = os.environ.get(
    "BCDL_FLOW_MODEL", os.path.join(MODELS, "neuflow_v2_int16_nashm_512x384.hbm"))
FLOW_LEFT = os.environ.get("BCDL_FLOW_LEFT", STEREO_LEFT)
FLOW_RIGHT = os.environ.get("BCDL_FLOW_RIGHT", STEREO_RIGHT)

# --- one-class anomaly detection --------------------------------------------
# The sample part image lives next to the model rather than in data/images/,
# because it comes from MVTec AD (CC BY-NC-SA) — the model is trained on one
# product and the figure needs THAT product, so both are staged by
# fetch_models.sh and neither is committed here.
ANOMALY_MODEL = os.environ.get(
    "BCDL_ANOMALY_MODEL", os.path.join(MODELS, "glass_bottle_int8_nashm_288x288.hbm"))
ANOMALY_IMAGE = os.environ.get(
    "BCDL_ANOMALY_IMAGE", os.path.join(MODELS, "anomaly_sample_defect.png"))
ANOMALY_GOOD = os.environ.get(
    "BCDL_ANOMALY_GOOD", os.path.join(MODELS, "anomaly_sample_good.png"))

# --- end-to-end driving ------------------------------------------------------
# Four input tensors, not an image, so the sample is an .npz of exactly what the
# model eats (one real nuScenes keyframe, staged with the model for the same
# licence reason as above) plus the clustered trajectory anchors.
DRIVE_MODEL = os.environ.get(
    "BCDL_DRIVE_MODEL", os.path.join(MODELS, "diffusiondrive_r34_256x1024_s100p.hbm"))
DRIVE_ANCHORS = os.environ.get(
    "BCDL_DRIVE_ANCHORS", os.path.join(MODELS, "kmeans_navsim_traj_20.npy"))
DRIVE_SAMPLE = os.environ.get(
    "BCDL_DRIVE_SAMPLE", os.path.join(MODELS, "drive_sample_frame.npz"))


# --- lidar 3-D detection (PointPillars) --------------------------------------
# Compiled stages with CPU work between them, so the paths are separate: the
# sample sweep and the published stages come straight from the model zoo
# (fetch_models.sh), and the head is compiled separately because the published
# one bakes its anchor decode in with ScatterND.
#
# TWO CHAINS run here, and the difference is 45.9x of BPU time:
#
#   published : vfe (3 inputs, expansion in-graph)  290.0 ms
#               + backbone 106.2 + head 7.5         = 403.7 ms
#   re-exported: vfe as a 1x1 convolution over
#               (1,10,P,32)                           1.62 ms
#               + backbone and head as one graph      7.20 ms  = 8.82 ms
#
# The re-exported chain is preferred when it is present, and nothing selects it
# by file name: the stages are dispatched on the shapes the models declare (a
# 1-input VFE wants expanded features; a head that takes the scatter canvas has
# the backbone fused into it). Its weights are not distributed with this library
# — they are trained on a non-commercial dataset — so the conversion recipe is
# published instead and the published chain stays the fetchable default.
def _lidar_stage(env, *names):
    override = os.environ.get(env)
    if override:
        return override
    for n in names:
        p = os.path.join(MODELS, n)
        if os.path.exists(p):
            return p
    return os.path.join(MODELS, names[-1])


LIDAR_VFE = _lidar_stage("BCDL_LIDAR_VFE",
                         "pointpillars_vfe_conv_int8_nashm.hbm", "pointpillars_vfe.hbm")
LIDAR_BACKBONE = os.environ.get("BCDL_LIDAR_BACKBONE",
                                os.path.join(MODELS, "pointpillars_backbone.hbm"))
LIDAR_HEAD = _lidar_stage("BCDL_LIDAR_HEAD", "pointpillars_bbhead_int8_nashm.hbm",
                          "pointpillars_head_int16_nashm_248x216.hbm")
# A calibrated frame — sweep + camera image + calibration + labels — staged by
# scripts/fetch_models.sh from your own KITTI copy. The dataset is not
# redistributable, so this is a path, not a download; see docs/MODELS.md.
LIDAR_FRAME = os.environ.get("BCDL_LIDAR_FRAME", os.path.join(MODELS, "kitti_frame"))
LIDAR_VOXELS = os.environ.get("BCDL_LIDAR_VOXELS", os.path.join(MODELS, "lidar_voxels0.npy"))
LIDAR_NUMPTS = os.environ.get("BCDL_LIDAR_NUMPTS",
                              os.path.join(MODELS, "lidar_voxel_num_points0.npy"))
LIDAR_NAMES = ["Car", "Pedestrian", "Cyclist"]


def asset(name):
    return find_image(name)


# Per-task spec: model file, test image, input (w,h), decode `kind`, class names.
# Images resolve via find_image(): det/pose/seg/obb/depth use the repo-local ccdl
# samples; cls (zebra) and semseg (cityscapes) fall back to the board assets.
TASKS = {
    "cls":  dict(model="yolo26n_cls_nashm_224x224_nv12.hbm",    image="zebra_cls.jpg",   wh=(224, 224),  kind="nv12",  names=None),
    "det":  dict(model="yolo26n_detect_nashm_640x640_nv12.hbm", image="bus.jpg",         wh=(640, 640),  kind="nv12",  names=COCO_NAMES),
    "det_dfl": dict(model=os.path.join(MODELS, "yolov8_640x640_nv12.hbm"), image="bus.jpg", wh=(640, 640), kind="nv12", names=COCO_NAMES),
    "pose": dict(model="yolo26n_pose_nashm_640x640_nv12.hbm",   image="zidane.jpg",      wh=(640, 640),  kind="nv12",  names=COCO_NAMES),
    "seg":  dict(model="yolo26n_seg_nashm_640x640_nv12.hbm",    image="bus.jpg",         wh=(640, 640),  kind="nv12",  names=COCO_NAMES),
    "obb":  dict(model="yolo26n_obb_nashm_640x640_nv12.hbm",    image=OBB_IMAGE,         wh=(640, 640),  kind="nv12",  names=DOTA_NAMES),
    "semseg": dict(model=SEMSEG_MODEL, image="segmentation.png", wh=(2048, 1024), kind="nv12", names=None),
    # Real-time semseg at the same input size. Output is 1/8 resolution
    # ([1,19,128,256]) rather than full-res, so the label map is 128x256 and the
    # caller upsamples — everything else decodes through the same Segmenter.
    "semseg_rt": dict(model=SEMSEG_RT_MODEL, image="segmentation.png", wh=(2048, 1024), kind="nv12", names=None),
    "semseg_yolo26": dict(model=SEMSEG_YOLO26_MODEL, image="segmentation.png", wh=(640, 640), kind="nv12", names=None),
    "depth":  dict(model=DEPTH_MODEL,  image="bus.jpg",          wh=(686, 518),   kind="depth", names=None),
    # --- SOTA additions (2026-07): open-vocab det/seg, transformer det ---------
    "yoloe_det": dict(model="yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm", image="bus.jpg", wh=(640, 640), kind="nv12", names=COCO_NAMES),
    "yoloe_seg": dict(model="yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm", image="bus.jpg", wh=(640, 640), kind="nv12", names=COCO_NAMES),
}


def available(task):
    """True when the model(s) and the test image exist on this board."""
    if task == "ocr":
        return all(os.path.exists(p) for p in (OCR_DET, OCR_REC, OCR_DICT, find_image(OCR_IMAGE)))
    if task == "stereo":
        have_model = os.path.exists(STEREO_CROP_MODEL) or os.path.exists(STEREO_RESIZE_MODEL)
        return have_model and os.path.exists(STEREO_LEFT) and os.path.exists(STEREO_RIGHT)
    if task == "wholebody":
        return all(os.path.exists(p) for p in
                   (WHOLEBODY_MODEL, WHOLEBODY_DET, find_image(WHOLEBODY_IMAGE)))
    if task == "reid":
        return all(os.path.exists(p) for p in
                   (REID_MODEL, REID_DET, find_image(REID_IMAGE)))
    if task == "features":
        return all(os.path.exists(p) for p in
                   (FEATURES_MODEL, FEATURES_LEFT, FEATURES_RIGHT))
    if task == "lidar3d":
        # The backbone is a separate stage only in the published chain; the
        # re-exported head has it fused in. LidarTask decides that from the
        # head's declared input shape — this is only a name-based guess, because
        # it has to answer before anything is loaded.
        need = [LIDAR_VFE, LIDAR_HEAD, LIDAR_VOXELS, LIDAR_NUMPTS]
        if "bbhead" not in os.path.basename(LIDAR_HEAD):
            need.append(LIDAR_BACKBONE)
        return all(os.path.exists(p) for p in need)
    if task == "flow":
        return all(os.path.exists(p) for p in (FLOW_MODEL, FLOW_LEFT, FLOW_RIGHT))
    if task == "anomaly":
        # The sample part is as required as the model: this head only means
        # anything on the product it was trained on.
        return os.path.exists(ANOMALY_MODEL) and os.path.exists(ANOMALY_IMAGE)
    if task == "drive":
        return all(os.path.exists(p) for p in (DRIVE_MODEL, DRIVE_ANCHORS, DRIVE_SAMPLE))
    if task in ("superres", "superres_span"):
        model = SUPERRES_MODEL if task == "superres" else SUPERRES_SPAN_MODEL
        return os.path.exists(model) and os.path.exists(find_image(SUPERRES_IMAGE))
    s = TASKS[task]
    return os.path.exists(model_path(s["model"])) and os.path.exists(find_image(s["image"]))


def preprocess_nv12(bcdl, img_bgr, dst_w, dst_h=None):
    """Letterbox a BGR image into dst_w x dst_h and split into NV12 [Y, UV]
    planes (the 2-input layout the RDK NV12 exports expect). Returns (y, uv, lb)."""
    if dst_h is None:
        dst_h = dst_w
    canvas, lb = bcdl.letterbox_numpy(img_bgr, dst_w, dst_h, pad=114)
    nv12 = bcdl.bgr_to_nv12(canvas)              # (dst_h*3//2, dst_w)
    y = np.ascontiguousarray(nv12[:dst_h])       # (dst_h, dst_w)        -> input0
    uv = np.ascontiguousarray(nv12[dst_h:])      # (dst_h//2, dst_w)     -> input1
    return y, uv, lb


def preprocess_depth_f32(img_bgr, dst_w, dst_h):
    """Depth-Anything-V2 preprocessing: stretch-resize to (dst_w,dst_h), BGR->RGB,
    ImageNet z-score normalize, NCHW float32 -> [1,3,dst_h,dst_w]."""
    import cv2

    r = cv2.resize(img_bgr, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])  # [1,3,H,W]


# ---------------------------------------------------------------------------
# Shared check-figure drawing: a deterministic per-class palette + a cv2.freetype
# text renderer (SourceHanSansSC-Regular.otf) so labels — including OCR Chinese — render with a
# filled background for legibility. Falls back to cv2.putText (ASCII) if the
# font / freetype module is missing. Ported in spirit from ccdl's draw helpers.
# ---------------------------------------------------------------------------

_FT = None  # cached cv2.freetype font, or False once we know it is unavailable


def _ft():
    global _FT
    if _FT is None:
        try:
            import cv2
            if os.path.exists(FONT) and hasattr(cv2, "freetype"):
                f = cv2.freetype.createFreeType2()
                f.loadFontData(FONT, 0)
                _FT = f
            else:
                _FT = False
        except Exception:
            _FT = False
    return _FT or None


def palette(cid):
    """Deterministic vivid BGR color for a class id (stable across figures)."""
    h = (int(cid) * 0x9E3779B1) & 0xFFFFFFFF
    return (64 + h % 192, 64 + (h >> 8) % 192, 64 + (h >> 16) % 192)


def put_label(vis, text, x, y, color, fh=18):
    """Draw `text` at (x, y) (label's bottom-left) with a filled color box behind
    it and white glyphs. Uses cv2.freetype (Unicode) when available, else putText.
    Clamps the box inside the image so top-edge labels stay visible."""
    import cv2

    ft = _ft()
    x = max(0, int(x))
    if ft is not None:
        (tw, th), _ = ft.getTextSize(text, fh, -1)
        top = max(0, int(y) - th - 6)
        cv2.rectangle(vis, (x, top), (x + tw + 6, top + th + 6), color, -1)
        ft.putText(vis, text, (x + 3, top + 3), fh, (255, 255, 255), -1,
                   cv2.LINE_AA, False)
    else:
        scale = fh / 30.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, int(y) - th - 6)
        cv2.rectangle(vis, (x, top), (x + tw + 6, top + th + 8), color, -1)
        cv2.putText(vis, text, (x + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)


# COCO-17 skeleton edges (0-based keypoint indices).
_COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
]


def _draw_skeleton(vis, kpts, thr=0.5):
    """Draw COCO-17 keypoints (red dots) + skeleton bones (green) for one person."""
    import cv2

    pts = list(kpts)
    for a, b in _COCO_SKELETON:
        if a < len(pts) and b < len(pts) and pts[a].score > thr and pts[b].score > thr:
            cv2.line(vis, (int(pts[a].x), int(pts[a].y)),
                     (int(pts[b].x), int(pts[b].y)), (0, 255, 0), 2, cv2.LINE_AA)
    for kp in pts:
        if kp.score > thr:
            cv2.circle(vis, (int(kp.x), int(kp.y)), 3, (0, 0, 255), -1, cv2.LINE_AA)


# COCO-WholeBody 133-keypoint layout. The body block is COCO-17, so the same
# skeleton edges apply; everything after it is dense point sets where bones would
# be noise, drawn as coloured dots instead.
WHOLEBODY_PARTS = (
    ("body", slice(0, 17), (0, 0, 255)),
    ("feet", slice(17, 23), (0, 200, 255)),
    ("face", slice(23, 91), (255, 200, 0)),
    ("lhand", slice(91, 112), (255, 0, 255)),
    ("rhand", slice(112, 133), (0, 255, 255)),
)


def _draw_wholebody(vis, kpts, thr=0.2):
    """Body bones + per-part coloured points for one 133-keypoint person."""
    import cv2

    pts = list(kpts)
    for a, b in _COCO_SKELETON:
        if a < len(pts) and b < len(pts) and pts[a].score > thr and pts[b].score > thr:
            cv2.line(vis, (int(pts[a].x), int(pts[a].y)),
                     (int(pts[b].x), int(pts[b].y)), (0, 255, 0), 2, cv2.LINE_AA)
    for _, sl, color in WHOLEBODY_PARTS:
        # Face and hands are dense and small; a 1-px dot keeps them readable.
        r = 3 if sl.stop - sl.start <= 23 else 1
        for kp in pts[sl]:
            if kp.score > thr:
                cv2.circle(vis, (int(kp.x), int(kp.y)), r, color, -1, cv2.LINE_AA)


def _text_patch_into_box(canvas, box, text, color):
    """Render `text` to fill a (rotated) 4-point box and composite onto `canvas`
    (PaddleOCR draw_box_txt_fine-style). `box` is 4 (x,y) points TL,TR,BR,BL.

    Direction-aware: for a tall (vertical) box the text is laid out horizontally
    along the long edge and then rotated upright, so vertical text lines read
    correctly instead of being squished into a narrow column."""
    import cv2

    box = np.asarray(box, np.float32).reshape(4, 2)
    bw = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[3] - box[2])))
    bh = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if bw < 4 or bh < 4:
        return
    vertical = bh > 2 * bw and bh > 30                # tall box -> vertical text line
    pw, ph = (bh, bw) if vertical else (bw, bh)       # lay text along the long edge
    patch = np.full((ph, pw, 3), 255, np.uint8)
    ft = _ft()
    if ft is not None and text:
        fh = max(8, int(ph * 0.8))
        (tw, _th), _ = ft.getTextSize(text, fh, -1)
        if tw > pw:                                   # shrink to fit the long edge
            fh = max(6, int(fh * pw / max(tw, 1)))
        ft.putText(patch, text, (2, 1), fh, color, -1, cv2.LINE_AA, False)
    if vertical:                                      # stand the text up to match the box
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)  # (bw,bh) -> (bh,bw)
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    M = cv2.getPerspectiveTransform(src, box)
    warped = cv2.warpPerspective(patch, M, (canvas.shape[1], canvas.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    np.minimum(canvas, warped, out=canvas)            # white=neutral -> keep glyphs


class Task:
    """A loaded task: engine + decoder + preprocessed inputs, ready to run/time."""

    def __init__(self, bcdl, key):
        import cv2

        self.bcdl = bcdl
        self.key = key
        spec = TASKS[key]
        self.spec = spec
        self.names = spec["names"]
        self.in_w, self.in_h = spec["wh"]
        self.kind = spec["kind"]
        self.model_file = model_path(spec["model"])
        self.image_file = asset(spec["image"])
        self.img = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"cannot read test image: {self.image_file}")
        self.orig_h, self.orig_w = self.img.shape[:2]
        self.engine = bcdl.Engine(self.model_file)
        self.lb = None
        if self.kind == "depth":                          # single F32 NCHW RGB input (ImageNet norm)
            self.x = preprocess_depth_f32(self.img, self.in_w, self.in_h)
        else:                                             # 2-input NV12
            self.y, self.uv, self.lb = preprocess_nv12(bcdl, self.img, self.in_w, self.in_h)
        self._build_decoder()

    def _build_decoder(self):
        b = self.bcdl
        k = self.key
        if k in ("det", "det_dfl", "yoloe_det"):  # YoloLtrbDetector auto-detects LTRB vs DFL box=64
            cfg = b.YoloLtrbConfig(); cfg.num_classes = 80; cfg.conf_thresh = 0.25
            cfg.strides = [8, 16, 32]
            self.dec = b.YoloLtrbDetector(self.engine, cfg)
        elif k == "pose":
            cfg = b.PoseConfig(); cfg.num_keypoints = 17; cfg.conf_thresh = 0.25
            cfg.strides = [8, 16, 32]
            self.dec = b.PoseEstimator(self.engine, cfg)
        elif k in ("seg", "yoloe_seg"):   # InstanceSegmenter auto-detects LTRB vs DFL box
            cfg = b.InstanceSegConfig(); cfg.conf_thresh = 0.25; cfg.strides = [8, 16, 32]
            self.dec = b.InstanceSegmenter(self.engine, cfg)
        elif k == "obb":
            cfg = b.ObbConfig(); cfg.num_classes = 15; cfg.conf_thresh = 0.25
            cfg.strides = [8, 16, 32]
            self.dec = b.ObbDetector(self.engine, cfg)
        elif k == "cls":
            self.dec = b.Classifier(self.engine)
        elif k == "semseg":
            cfg = b.SegConfig(); cfg.channels_first = False  # deeplabv3plus is NHWC [1,H,W,C]
            self.dec = b.Segmenter(self.engine, cfg)         # argmax over 19 channels
        elif k in ("semseg_rt", "semseg_yolo26"):
            cfg = b.SegConfig(); cfg.channels_first = True   # both NCHW [1,C,H,W]
            self.dec = b.Segmenter(self.engine, cfg)         # same decode, 19 channels
        elif k == "depth":
            self.dec = b.DepthEstimator(self.engine)
        else:
            raise ValueError(k)

    def feed(self):
        """Load the preprocessed inputs into the engine (for infer-only timing)."""
        if self.kind == "depth":
            self.engine._e.set_input(0, self.x)
        else:
            self.engine._e.set_input(0, self.y)
            self.engine._e.set_input(1, self.uv)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        """Run the full path (feed + infer + postprocess) and return results."""
        k = self.key
        if k in ("det", "det_dfl", "yoloe_det", "pose", "obb"):
            return self.dec.detect([self.y, self.uv], self.lb)
        if k in ("seg", "yoloe_seg"):
            return self.dec.detect([self.y, self.uv], self.lb, self.orig_w, self.orig_h)
        if k == "cls":
            return self.dec.classify([self.y, self.uv])
        if k in ("semseg", "semseg_rt", "semseg_yolo26"):   # 2-input NV12 -> feed both, then argmax decode
            self.engine._e.set_input(0, self.y)
            self.engine._e.set_input(1, self.uv)
            self.engine._e.infer(0)
            return self.dec.postprocess()
        if k == "depth":           # single F32 NCHW input
            return self.dec.estimate(self.x)
        raise ValueError(k)

    def postprocess(self):
        """Decode the engine's current output (after feed()+infer())."""
        k = self.key
        if k in ("seg", "yoloe_seg"):
            return self.dec.postprocess(self.lb, self.orig_w, self.orig_h)
        if k in ("cls", "semseg", "semseg_rt", "semseg_yolo26", "depth"):
            return self.dec.postprocess()
        return self.dec.postprocess(self.lb)

    # -- summaries + figures ------------------------------------------------
    def summarize(self, results):
        k = self.key
        if k == "cls":
            return ", ".join(f"{r.class_id}:{r.score:.2f}" for r in results[:3])
        if k == "pose":
            return f"{len(results)} pose(s)"
        if k in ("seg", "yoloe_seg"):
            return f"{len(results)} instance(s)"
        if k == "obb":
            return f"{len(results)} rotated box(es)"
        if k in ("semseg", "semseg_rt", "semseg_yolo26"):
            n = int(np.unique(np.asarray(results.labels)).size)
            return f"{results.width}x{results.height} seg, {n} classes"
        if k == "depth":
            return f"{results.width}x{results.height} depth [{results.vmin:.2f},{results.vmax:.2f}]"
        return f"{len(results)} box(es)"

    def _name(self, cid):
        if self.names and 0 <= cid < len(self.names):
            return self.names[cid]
        return str(cid)

    def draw(self, results):
        import cv2

        vis = self.img.copy()
        k = self.key
        if k == "cls":
            for i, r in enumerate(results[:3]):        # ranked banner, top-3
                put_label(vis, f"{self._name(r.class_id)}  {r.score:.2f}",
                          8, 12 + i * 30 + 24, palette(r.class_id), fh=22)
            return vis
        if k == "obb":
            for d in results:
                r = d.rrect
                box = cv2.boxPoints(((r.cx, r.cy), (r.w, r.h), np.degrees(r.angle)))
                c = palette(d.class_id)
                cv2.polylines(vis, [box.astype(np.int32)], True, c, 2, cv2.LINE_AA)
                put_label(vis, self._name(d.class_id), box[:, 0].min(),
                          box[:, 1].min(), c, fh=16)
            return vis
        if k in ("seg", "yoloe_seg"):
            overlay = vis.copy()
            for d in results:                          # masks first, boxes/labels on top
                c = palette(d.class_id)
                m = d.mask
                if m.size:
                    overlay[m.astype(bool)] = c
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            for d in results:
                c = palette(d.class_id)
                cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
                put_label(vis, f"{self._name(d.class_id)} {d.score:.2f}",
                          d.x1, d.y1, c, fh=18)
            return vis
        if k == "pose":
            for d in results:
                cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                              (0, 255, 0), 2)
                _draw_skeleton(vis, d.keypoints)
            return vis
        if k in ("semseg", "semseg_rt", "semseg_yolo26"):  # palette overlay
            cseg = self.bcdl.seg_colorize(results)     # (segH,segW,3) BGR
            cseg = cv2.resize(cseg, (self.orig_w, self.orig_h),
                              interpolation=cv2.INTER_NEAREST)
            return cv2.addWeighted(vis, 0.5, cseg, 0.5, 0)
        if k == "depth":                               # original | turbo depth
            cd = self.bcdl.depth_colorize(results)     # (dH,dW,3) BGR
            cd = cv2.resize(cd, (self.orig_w, self.orig_h))
            return np.hstack([vis, cd])
        # det / det_dfl
        for d in results:
            c = palette(d.class_id)
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
            put_label(vis, f"{self._name(d.class_id)} {d.score:.2f}", d.x1, d.y1, c, fh=18)
        return vis


def _order_points(p):
    """4 corners -> TL, TR, BR, BL (matches ocr_demo's get_rotate_crop_image)."""
    s = p.sum(1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]],
                    np.float32)


def _order_quad(p):
    """4 corners -> reading order [left-top, right-top, right-bottom, left-bottom],
    PaddleOCR get_mini_boxes style: sort by x, then split each pair by y. Unlike
    the x+y/y-x heuristic this stays correct for SLANTED / rotated text boxes, so
    the warped text patch follows the box's tilt instead of shearing."""
    q = sorted(p.reshape(4, 2).tolist(), key=lambda v: v[0])
    (a, b), (c, d) = q[:2], q[2:]
    lt, lb = (a, b) if a[1] <= b[1] else (b, a)        # left pair: smaller y = top
    rt, rb = (c, d) if c[1] <= d[1] else (d, c)        # right pair
    return np.array([lt, rt, rb, lb], np.float32)


def _crop_and_rotate(img, pts):
    import cv2

    # Reading-order corners (ccdl order_points_clockwise / PaddleOCR get_mini_boxes)
    # so a slanted box rectifies correctly instead of shearing.
    src = _order_quad(np.asarray(pts, np.float32).reshape(4, 2))
    w = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
    h = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
    if w <= 0 or h <= 0:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    if h / w >= 1.5:
        # PaddleOCR get_rotate_crop_image uses np.rot90 (counter-clockwise); a
        # clockwise rotate here leaves vertical lines upside-down (the weak
        # direction-cls can't always recover them, e.g. short Latin "ODM OEM").
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _nchw_imagenet(img_bgr, w, h):
    """BGR -> resized RGB, ImageNet z-score, NCHW float32 [1,3,h,w] (det preproc)."""
    import cv2

    r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])


def _resize_pad_nchw(img_bgr, W, H, mean, std):
    """Aspect-ratio-preserving resize to height H (width capped at W), BGR->RGB,
    normalise, then right-pad to width W with 0 in normalised space. This is the
    PaddleOCR resize_norm_img / ccdl resize_img_with_pad convention used for rec
    and cls — it avoids the character distortion of an anamorphic stretch on short
    or unusually-proportioned text lines. Returns NCHW float32 [1,3,H,W]."""
    import cv2

    h, w = img_bgr.shape[:2]
    ratio = w / max(h, 1)
    rw = W if int(np.ceil(H * ratio)) > W else int(np.ceil(H * ratio))
    rw = max(1, min(rw, W))
    r = cv2.resize(img_bgr, (rw, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    out = np.zeros((1, 3, H, W), np.float32)            # pad = 0 (normalised space)
    out[0, :, :, :rw] = rgb.transpose(2, 0, 1)
    return np.ascontiguousarray(out)


class OcrTask:
    """LATEST PP-OCRv5: server DBNet text detection (960×960) + (optional) PP-LCNet
    0°/180° direction cls + server CRNN/CTC recognition (18385-class v5 dict),
    mirroring examples/ocr_demo.cc through the Python bindings. All three are
    single-input F32 RGB NCHW featuremap models (converted offline from ccdl ONNX).
    The cls stage runs only when its .hbm is deployed (OCR_CLS). Duck-types the
    Task API so the bench / test harness can drive it like a single-model task."""

    key = "ocr"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.in_w = self.in_h = 960                      # det canvas (PP-OCRv5 server)
        self.det_model = OCR_DET
        self.rec_model = OCR_REC
        self.model_file = OCR_DET                      # headline model for the row
        self.image_file = find_image(OCR_IMAGE)
        self.img = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"cannot read OCR test image: {self.image_file}")
        self.orig_h, self.orig_w = self.img.shape[:2]

        self.det = bcdl.Engine(self.det_model)
        self.rec = bcdl.Engine(self.rec_model)
        self.engine = self.det                          # for num_outputs in the row

        # det preprocess: anamorphic resize to 960x960 -> RGB ImageNet NCHW F32.
        self.det_input = _nchw_imagenet(self.img, self.in_w, self.in_h)

        cfg = bcdl.DbConfig()
        cfg.bin_thresh = 0.3
        cfg.box_thresh = 0.6                            # PP-OCRv5 server defaults
        cfg.unclip_ratio = 1.5
        self.detector = bcdl.DbTextDetector(self.det._e, cfg, 0)
        self.recer = bcdl.TextRecognizer(self.rec._e, OCR_DICT, 0)
        self._lb = bcdl.LetterboxInfo()                 # identity -> 960-space boxes

        # Optional 3rd stage: text-line direction classifier (0°/180°). When the
        # cls .hbm is deployed, each crop is checked and flipped before rec so
        # upside-down lines recognise correctly; absent, OCR stays det->rec.
        self.cls = self.angler = None
        if os.path.exists(OCR_CLS):
            self.cls = bcdl.Engine(OCR_CLS)
            self.angler = bcdl.TextAngleClassifier(self.cls._e, 0.9, 0)

    def feed(self):
        self.det._e.set_input(0, self.det_input)

    def infer(self):                                    # det infer (headline number)
        self.det._e.infer(0)

    def _rec_one(self, crop):
        """PP-OCRv5 rec preproc: aspect-preserving resize to H=48 (W<=320) + pad,
        BGR->RGB, (x/255-0.5)/0.5 -> [-1,1] NCHW float32 [1,3,48,320]."""
        x = _resize_pad_nchw(crop, 320, 48, 0.5, 0.5)
        self.rec._e.set_input(0, x)
        self.rec._e.infer(0)
        return self.recer.postprocess()

    def _cls_one(self, crop):
        """PP-LCNet textline direction cls preproc: aspect-preserving resize to
        H=80 (W<=160) + pad, BGR->RGB, ImageNet z-score, [1,3,80,160] -> ClsDirResult."""
        x = _resize_pad_nchw(crop, 160, 80, _IMAGENET_MEAN, _IMAGENET_STD)
        self.cls._e.set_input(0, x)
        self.cls._e.infer(0)
        return self.angler.postprocess()

    def decode(self):
        """Full pipeline: det -> boxes (scaled to original) -> crop -> [cls 180°
        flip] -> rec text."""
        import cv2

        self.feed()
        self.det._e.infer(0)
        boxes = self.detector.postprocess(self._lb)     # 960-space
        sx, sy = self.orig_w / float(self.in_w), self.orig_h / float(self.in_h)
        out = []
        for b in boxes:
            pts = np.array(b.points, np.float32).reshape(4, 2) * (sx, sy)
            crop = _crop_and_rotate(self.img, pts)
            if crop is None or crop.size == 0:
                continue
            if self.angler is not None and self._cls_one(crop).flip180:
                crop = cv2.rotate(crop, cv2.ROTATE_180)   # upside-down -> upright
            rr = self._rec_one(crop)
            if rr.text:
                out.append((pts, rr.text, rr.score))
        return out

    def postprocess(self):
        return self.decode()

    def summarize(self, results):
        if not results:
            return "0 text lines"
        return f"{len(results)} lines; e.g. {results[0][1]}"

    def draw(self, results):
        """PaddleOCR-style side-by-side: left = original with colored text
        polygons; right = blank canvas with the recognized text rendered back
        into each box (cv2.freetype + SourceHanSansSC-Regular.otf for Chinese)."""
        import cv2

        left = self.img.copy()
        blank = np.full_like(left, 255)
        for i, (pts, text, _score) in enumerate(results):
            box = np.asarray(pts, np.float32).reshape(4, 2)
            c = palette(i)                              # distinct color per line
            cv2.polylines(left, [box.astype(np.int32)], True, c, 2, cv2.LINE_AA)
            cv2.polylines(blank, [box.astype(np.int32)], True, c, 1, cv2.LINE_AA)
            # Reading-order corners so the warped text follows a slanted box's tilt.
            _text_patch_into_box(blank, _order_quad(box), text, (0, 0, 0))
        return np.hstack([left, blank])


class StereoTask:
    """LAS2 two-image stereo: left/right F32 NCHW RGB -> disparity (+ depth /
    validity mask), via bcdl.StereoPipeline. Duck-types the Task API so the
    bench / test harness drives it like a single-model task. Prefers the
    crop-calibrated model (matching the repo's center-crop pair); falls back to
    the resize model. fx/baseline are nominal (figure/summary only)."""

    key = "stereo"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.in_w, self.in_h = 640, 480
        if os.path.exists(STEREO_CROP_MODEL):
            self.model_file, self.fit = STEREO_CROP_MODEL, bcdl.StereoFit.Crop
        else:
            self.model_file, self.fit = STEREO_RESIZE_MODEL, bcdl.StereoFit.Resize
        self.image_file = STEREO_LEFT
        self.left = cv2.imread(STEREO_LEFT, cv2.IMREAD_COLOR)
        self.right = cv2.imread(STEREO_RIGHT, cv2.IMREAD_COLOR)
        if self.left is None or self.right is None:
            raise RuntimeError("cannot read stereo pair (stereo_left/right.png)")

        self.engine = bcdl.Engine(self.model_file)
        cfg = bcdl.StereoConfig()
        cfg.fit = self.fit
        cfg.fx, cfg.baseline = 700.0, 0.12       # nominal intrinsics for depth row
        cfg.valid_mask = True
        cfg.disp_min = 2.0
        self.cfg = cfg
        self.pipe = bcdl.StereoPipeline(self.engine, cfg)

    def feed(self):
        """Pre-pack both inputs (for infer-only timing), mirroring the pipeline."""
        l = self.bcdl.pack_stereo_input(self.left, self.in_h, self.in_w, self.fit, True)
        r = self.bcdl.pack_stereo_input(self.right, self.in_h, self.in_w, self.fit, True)
        self.engine._e.set_input(0, l)
        self.engine._e.set_input(1, r)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        return self.pipe.process(self.left, self.right)

    def postprocess(self):
        return self.decode()

    def summarize(self, res):
        d = np.asarray(res.disparity.data)
        s = f"disp [{d.min():.1f},{d.max():.1f}] mean {d.mean():.1f}px"
        v = np.asarray(res.valid)
        if v.size:
            s += f"; {100 * v.mean():.0f}% valid"
        return s

    def _colorize(self, res, mask=None):
        import cv2

        d = np.asarray(res.disparity.data)
        vmax = float(d.max()) or 1.0
        g = np.clip(d / vmax * 255.0, 0, 255).astype(np.uint8)
        c = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
        if mask is not None and mask.size:
            c[~mask.astype(bool)] = (60, 60, 60)
        return c

    _BAR_H = 28               # per-panel title strip height
    _CROP_CLR = (0, 255, 0)   # crop-box / linkage color (BGR green)

    @staticmethod
    def _pad_h(img, H):
        """Vertically center-pad an image to H rows (black) for hstack alignment."""
        import cv2

        h = img.shape[0]
        if h >= H:
            return img
        top = (H - h) // 2
        return cv2.copyMakeBorder(img, top, H - h - top, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def _titled(self, panel, text):
        """Prepend a black title strip with `text` (white)."""
        import cv2

        bar = np.zeros((self._BAR_H, panel.shape[1], 3), np.uint8)
        cv2.putText(bar, text, (6, self._BAR_H - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, panel])

    def _legend(self, body_h):
        """A scale legend panel: TURBO bar labelled near->far."""
        import cv2

        w = 84
        panel = np.full((body_h, w, 3), 30, np.uint8)
        gh = body_h - 30
        grad = np.linspace(255, 0, gh).astype(np.uint8)[:, None].repeat(28, axis=1)
        panel[10:10 + gh, 10:38] = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
        cv2.rectangle(panel, (10, 10), (38, 10 + gh), (200, 200, 200), 1)
        cv2.putText(panel, "near", (42, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "far", (42, 10 + gh), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return self._titled(panel, "scale")

    def draw(self, res):
        """Annotated side-by-side: input | disparity | scale legend. In CROP mode
        the input panel is the FULL original with the center-crop box drawn, and
        the SAME green box is drawn around the disparity — so it reads at a glance
        that the disparity covers exactly the boxed crop (the narrow FOV is by
        design). RESIZE mode shows the whole frame scaled to disparity. (The
        validity mask is still computed — see summarize()'s "valid %" — it is
        just not rendered as a gray panel here.)"""
        import cv2

        disp = self._colorize(res)

        if self.fit == self.bcdl.StereoFit.Crop:
            h, w = self.left.shape[:2]
            t, l = (h - self.in_h) // 2, (w - self.in_w) // 2
            left_vis = self.left.copy()
            cv2.rectangle(left_vis, (l, t), (l + self.in_w, t + self.in_h), self._CROP_CLR, 2)
            cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), self._CROP_CLR, 2)
            body_h = max(h, self.in_h)
            content = [self._pad_h(left_vis, body_h), self._pad_h(disp, body_h)]
            titles = ["input  (green = crop fed to model)", "disparity  near->far"]
        else:
            body_h = self.in_h
            content = [cv2.resize(self.left, (self.in_w, self.in_h)), disp]
            titles = ["input  (full frame)", "disparity  near->far"]

        panels = [self._titled(c, t) for c, t in zip(content, titles)]
        panels.append(self._legend(body_h))
        return np.hstack(panels)


class WholeBodyTask:
    """ViTPose whole-body 133 keypoints, TOP-DOWN: a YOLO person detector feeds
    one crop per person to the pose model. Duck-types the Task API so the bench /
    test harness drives it like a single-model task, but note what the timings
    mean here — infer() is ONE person, while decode() runs every person in the
    frame, so end-to-end cost scales with the head count. That is inherent to
    top-down and is the trade for resolving feet, face and hands."""

    key = "wholebody"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.model_file = WHOLEBODY_MODEL
        self.image_file = asset(WHOLEBODY_IMAGE)
        self.img = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"cannot read test image: {self.image_file}")
        self.orig_h, self.orig_w = self.img.shape[:2]

        # Stage 1: person boxes, once — they are the same every run, so keeping
        # them out of decode() keeps the reported pose cost the pose cost.
        self.det_engine = bcdl.Engine(WHOLEBODY_DET)
        cfg = bcdl.YoloLtrbConfig()
        cfg.num_classes = 80
        cfg.conf_thresh = 0.5
        cfg.strides = [8, 16, 32]
        det = bcdl.YoloLtrbDetector(self.det_engine, cfg)
        y, uv, lb = preprocess_nv12(bcdl, self.img, 640, 640)
        self.boxes = [(d.x1, d.y1, d.x2, d.y2) for d in det.detect([y, uv], lb)
                      if d.class_id == 0]

        # Stage 2: the pose model itself.
        self.engine = bcdl.Engine(self.model_file)
        self.est = bcdl.WholeBodyEstimator(self.engine)
        self.in_w, self.in_h = self.est.in_w, self.est.in_h
        if self.boxes:
            self.x, self.crop = bcdl.wholebody_preprocess(
                self.img, *[float(v) for v in self.boxes[0]], self.in_w, self.in_h,
                self.est.config)
        else:
            self.x = np.zeros((1, 3, self.in_h, self.in_w), np.float32)
            self.crop = bcdl.WholeBodyCrop()

    def feed(self):
        self.engine._e.set_input(0, self.x)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        """Every person in the frame: preprocess + infer + decode, one at a time."""
        return [self.est.estimate(self.img, b) for b in self.boxes]

    def postprocess(self):
        """Decode the engine's current output (after feed()+infer()) — person 0."""
        return self.est.postprocess(self.crop)

    def summarize(self, res):
        if not res:
            return "0 person(s)"
        vis = [sum(1 for kp in person if kp.score > 0.2) for person in res]
        return "%d person(s), %d keypoints, %s visible" % (
            len(res), len(res[0]), "/".join(str(v) for v in vis))

    def draw(self, res):
        import cv2

        vis = self.img.copy()
        for person in res:
            _draw_wholebody(vis, person)
        legend_y = 26          # put_label anchors the label's bottom edge
        for name, sl, color in WHOLEBODY_PARTS:
            put_label(vis, "%s (%d)" % (name, sl.stop - sl.start), 8, legend_y, color, fh=16)
            legend_y += 22
        return vis


class FeaturesTask:
    """XFeat sparse features: extract from a PAIR of images and match them.

    Duck-types the Task API, but note what the numbers mean: infer() is ONE
    image, while decode() extracts both and matches, and matching is
    O(|a|*|b|*64) — at the default 4096 features per side that dominates the
    frame, which is why XfeatConfig.top_k is the knob worth reaching for."""

    key = "features"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.model_file = FEATURES_MODEL
        self.image_file = FEATURES_LEFT
        self.a = cv2.imread(FEATURES_LEFT, cv2.IMREAD_COLOR)
        self.b = cv2.imread(FEATURES_RIGHT, cv2.IMREAD_COLOR)
        if self.a is None or self.b is None:
            raise RuntimeError("cannot read the feature image pair")
        self.orig_h, self.orig_w = self.a.shape[:2]

        self.engine = bcdl.Engine(self.model_file)
        self.ext = bcdl.FeatureExtractor(self.engine)
        self.in_w, self.in_h = self.ext.in_w, self.ext.in_h
        self.x, _, _ = bcdl.xfeat_preprocess(self.a, self.in_w, self.in_h)

    def feed(self):
        self.engine._e.set_input(0, self.x)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        fa = self.ext.extract(self.a)
        fb = self.ext.extract(self.b)
        return fa, fb, self.bcdl.match_features(fa, fb, 0.82)

    def postprocess(self):
        """Decode the engine's current output (after feed()+infer()) — image A."""
        sx = self.orig_w / self.in_w
        sy = self.orig_h / self.in_h
        return self.ext.postprocess(sx, sy)

    def summarize(self, res):
        fa, fb, m = res
        s = f"{len(fa)}+{len(fb)} features, {len(m)} matches"
        try:
            import cv2

            # A FUNDAMENTAL matrix, not a homography: these are two views of a
            # scene with real depth (box, keyboard and desk at different
            # distances), so no single plane explains the correspondences and a
            # homography's inlier rate would read as a failure when nothing has
            # failed. Epipolar geometry is the right constraint for any rigid
            # scene, planar or not.
            if len(m) >= 8:
                pa = fa.xy[[x.a for x in m]]
                pb = fb.xy[[x.b for x in m]]
                _, inl = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 2.0, 0.999, 10000)
                if inl is not None:
                    s += f"; {100.0 * inl.sum() / len(m):.0f}% epipolar-consistent"
        except ImportError:
            pass
        return s

    def draw(self, res):
        """The two frames side by side with match lines drawn across the seam."""
        import cv2

        fa, fb, m = res
        ha, wa = self.a.shape[:2]
        hb, wb = self.b.shape[:2]
        H = max(ha, hb)
        vis = np.zeros((H, wa + wb, 3), np.uint8)
        vis[:ha, :wa] = self.a
        vis[:hb, wa:] = self.b

        pa = fa.xy
        pb = fb.xy
        for kp in pa:
            cv2.circle(vis, (int(kp[0]), int(kp[1])), 1, (0, 180, 255), -1)
        for kp in pb:
            cv2.circle(vis, (int(kp[0]) + wa, int(kp[1])), 1, (0, 180, 255), -1)

        # Colour each line by its cosine so a weak match reads as weak.
        shown = sorted(m, key=lambda x: -x.score)[:300]
        for x in shown:
            t = float(np.clip((x.score - 0.82) / 0.18, 0, 1))
            color = (int(60 + 40 * t), int(80 + 175 * t), int(255 - 120 * t))
            p, q = pa[x.a], pb[x.b]
            cv2.line(vis, (int(p[0]), int(p[1])), (int(q[0]) + wa, int(q[1])),
                     color, 1, cv2.LINE_AA)
        put_label(vis, f"{len(fa)} + {len(fb)} features", 8, 26, (0, 180, 255), fh=18)
        put_label(vis, f"{len(m)} mutual-NN matches ({len(shown)} drawn)", 8, 50,
                  (80, 255, 135), fh=18)
        return vis


class SuperResTask:
    """x4 super-resolution applied by tiling a fixed-size upscaler.

    The LR input is a real image downscaled by the model's own scale factor, so
    the original doubles as ground truth and the result can be scored honestly
    rather than just looked at."""

    names = None

    def __init__(self, bcdl, key="superres"):
        import cv2

        self.bcdl = bcdl
        self.key = key
        self.model_file = SUPERRES_MODEL if key == "superres" else SUPERRES_SPAN_MODEL
        self.image_file = asset(SUPERRES_IMAGE)
        src = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if src is None:
            raise RuntimeError(f"cannot read test image: {self.image_file}")

        self.engine = bcdl.Engine(self.model_file)
        self.sr = bcdl.SuperResolver(self.engine)
        s = self.sr.scale
        # Crop to a multiple of the scale so the downscale is exact and the
        # ground truth lines up pixel for pixel.
        h, w = (src.shape[0] // s) * s, (src.shape[1] // s) * s
        self.hr = src[:h, :w]
        self.img = cv2.resize(self.hr, (w // s, h // s), interpolation=cv2.INTER_AREA)
        self.orig_h, self.orig_w = self.img.shape[:2]
        self.in_w = self.in_h = self.sr.tile
        self.x = np.ascontiguousarray(
            np.zeros((1, 3, self.in_h, self.in_w), np.float32))

    def feed(self):
        self.engine._e.set_input(0, self.x)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        return self.sr.upscale(self.img)

    def postprocess(self):
        return self.decode()

    @staticmethod
    def _psnr(a, b):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    @staticmethod
    def _detail(img):
        """Mean gradient magnitude — a stand-in for how much fine structure an
        image actually carries."""
        g = img.astype(np.float64).mean(axis=2)
        return float(np.hypot(np.gradient(g)[0], np.gradient(g)[1]).mean())

    def summarize(self, res):
        import cv2

        bic = cv2.resize(self.img, (self.hr.shape[1], self.hr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        # BOTH numbers, because they disagree and the disagreement is the point:
        # this is a perceptual, GAN-trained upscaler, so on a clean downscale a
        # conservative blur wins PSNR while carrying visibly less detail.
        d_hr = self._detail(self.hr)
        return ("%dx%d -> %dx%d, %d tiles; PSNR %.1f dB (bicubic %.1f); "
                "detail %.2fx of truth (bicubic %.2fx)"
                % (self.orig_w, self.orig_h, res.shape[1], res.shape[0],
                   self.sr.last_tile_count, self._psnr(res, self.hr),
                   self._psnr(bic, self.hr), self._detail(res) / d_hr,
                   self._detail(bic) / d_hr))

    def draw(self, res):
        """LR (nearest-enlarged) | model | bicubic | ground truth, on one crop —
        a whole-frame view at this scale shows nothing useful."""
        import cv2

        s = self.sr.scale
        cw = min(240, self.orig_w)
        ch = min(160, self.orig_h)
        x = (self.orig_w - cw) // 2
        y = (self.orig_h - ch) // 2
        lr = self.img[y:y + ch, x:x + cw]
        panels = [
            cv2.resize(lr, (cw * s, ch * s), interpolation=cv2.INTER_NEAREST),
            res[y * s:(y + ch) * s, x * s:(x + cw) * s],
            cv2.resize(lr, (cw * s, ch * s), interpolation=cv2.INTER_CUBIC),
            self.hr[y * s:(y + ch) * s, x * s:(x + cw) * s],
        ]
        titles = ["input (nearest)", "model x%d" % s, "bicubic x%d" % s, "ground truth"]
        out = []
        for p, t in zip(panels, titles):
            bar = np.zeros((28, p.shape[1], 3), np.uint8)
            cv2.putText(bar, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out.append(np.vstack([bar, p]))
        return np.hstack(out)


class ReidTask:
    """Person ReID (OSNet): a YOLO detector supplies boxes, the ReID model turns
    each crop into a 512-d appearance vector. Duck-types the Task API.

    WHAT THE TIMINGS MEAN. infer() is ONE crop; decode() embeds every person in
    the frame, so cost scales with the head count exactly like the top-down pose
    task. In a tracker this runs on the high-score detections of every frame,
    which is what makes it the expensive half of appearance tracking.

    WHAT THE FIGURE SHOWS. An embedding has nothing to draw, so the check image
    is the cross-similarity matrix between the people found: the diagonal is 1
    by construction and the off-diagonal entries are what the tracker relies on
    being small. A tower that has collapsed under quantization gives a matrix of
    near-ones while still returning perfectly well-formed unit vectors.
    """

    key = "reid"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.model_file = REID_MODEL
        self.image_file = asset(REID_IMAGE)
        self.img = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"cannot read test image: {self.image_file}")
        self.orig_h, self.orig_w = self.img.shape[:2]

        # Stage 1: person boxes, once — same every run, so keeping them out of
        # decode() keeps the reported ReID cost the ReID cost.
        self.det_engine = bcdl.Engine(REID_DET)
        cfg = bcdl.YoloLtrbConfig()
        cfg.num_classes = 80
        cfg.conf_thresh = 0.5
        cfg.strides = [8, 16, 32]
        det = bcdl.YoloLtrbDetector(self.det_engine, cfg)
        y, uv, lb = preprocess_nv12(bcdl, self.img, 640, 640)
        self.dets = [d for d in det.detect([y, uv], lb) if d.class_id == 0]

        # Stage 2: the ReID model.
        self.engine = bcdl.Engine(self.model_file)
        self.ex = bcdl.ReIDExtractor(self.engine)
        self.in_w, self.in_h = 128, 256
        self.x = (self._crop_input(self.dets[0]) if self.dets
                  else np.zeros((1, 3, self.in_h, self.in_w), np.float32))

    def _crop_input(self, d):
        x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
        x2, y2 = min(self.orig_w, int(d.x2)), min(self.orig_h, int(d.y2))
        return self.bcdl.reid_preprocess(self.img[y1:y2, x1:x2])

    def feed(self):
        self.engine._e.set_input(0, self.x)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        """Every person in the frame: crop + preprocess + infer + read-out."""
        return self.ex.embed_detections(self.img, self.dets)

    def postprocess(self):
        """Read the engine's current output (after feed()+infer()) — person 0."""
        return self.ex.postprocess()

    def summarize(self, res):
        vecs = [np.asarray(v, np.float32) for v in res if len(v)]
        if len(vecs) < 2:
            return "%d person(s), %d-d embeddings" % (len(vecs), self.ex.dim)
        sim = np.stack(vecs) @ np.stack(vecs).T
        off = sim[~np.eye(len(vecs), dtype=bool)]
        return "%d person(s), %d-d, cross-similarity max %.3f mean %.3f" % (
            len(vecs), self.ex.dim, off.max(), off.mean())

    def draw(self, res):
        import cv2

        vecs = [np.asarray(v, np.float32) for v in res if len(v)]
        vis = self.img.copy()
        for i, d in enumerate(self.dets[:len(vecs)]):
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          (0, 200, 255), 2)
            put_label(vis, "#%d" % i, int(d.x1), int(d.y1) - 4, (0, 200, 255))
        if len(vecs) < 2:
            return vis

        # Similarity matrix panel, scaled to the frame height.
        sim = np.stack(vecs) @ np.stack(vecs).T
        n = len(vecs)
        cell = max(40, min(90, self.orig_h // (n + 1)))
        panel = np.full(((n + 1) * cell, (n + 1) * cell, 3), 30, np.uint8)
        for i in range(n):
            put_label(panel, "#%d" % i, 6, (i + 2) * cell - cell // 3, (0, 200, 255))
            put_label(panel, "#%d" % i, (i + 1) * cell + 6, cell - cell // 3,
                      (0, 200, 255))
        for i in range(n):
            for j in range(n):
                v = float(sim[i, j])
                # Green = distinct (what we want off-diagonal), red = alike.
                color = (0, int(255 * (1 - max(0.0, v))), int(255 * max(0.0, v)))
                y0, x0 = (i + 1) * cell, (j + 1) * cell
                cv2.rectangle(panel, (x0 + 2, y0 + 2),
                              (x0 + cell - 2, y0 + cell - 2), color, -1)
                cv2.putText(panel, "%.2f" % v, (x0 + 6, y0 + cell // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)
        h = max(vis.shape[0], panel.shape[0])
        def _pad(a):
            return np.vstack([a, np.full((h - a.shape[0], a.shape[1], 3), 30,
                                         np.uint8)]) if a.shape[0] < h else a
        return np.hstack([_pad(vis), _pad(panel)])


class FlowTask:
    """Dense optical flow over a frame pair.

    The pair is the repo's stereo pair, so what comes back is DISPARITY — one
    consistent direction with magnitude falling off with distance. That makes an
    easy figure to sanity-check, but do not read its ~90 px mean as typical of
    consecutive video frames.

    infer() is one forward pass; decode() is the same pass plus preprocessing of
    BOTH frames and the de-planarize, so the gap between the two numbers is the
    CPU side — which for this model is small enough that the BPU dominates."""

    key = "flow"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.model_file = FLOW_MODEL
        self.image_file = FLOW_LEFT
        self.a = cv2.imread(FLOW_LEFT, cv2.IMREAD_COLOR)
        self.b = cv2.imread(FLOW_RIGHT, cv2.IMREAD_COLOR)
        if self.a is None or self.b is None:
            raise RuntimeError("cannot read the flow image pair")
        self.engine = bcdl.Engine(self.model_file)
        self.est = bcdl.OpticalFlowEstimator(self.engine)
        self.in_w, self.in_h = self.est.input_size
        self.x0 = np.ascontiguousarray(bcdl.flow_preprocess(self.a, self.in_w, self.in_h))
        self.x1 = np.ascontiguousarray(bcdl.flow_preprocess(self.b, self.in_w, self.in_h))

    def feed(self):
        self.engine._e.set_input(0, self.x0)
        self.engine._e.set_input(1, self.x1)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        return self.est.estimate(self.a, self.b)

    def summarize(self, res):
        mag = np.linalg.norm(res.data, axis=-1)
        return "%dx%d field, |flow| mean %.2f px, p95 %.2f px, max %.2f px" % (
            res.width, res.height, mag.mean(), np.percentile(mag, 95), mag.max())

    def draw(self, res):
        """Frame, colour wheel and magnitude, stacked — the wheel alone reads as
        pretty noise unless you can see the frame it came from."""
        import cv2

        wheel = self.bcdl.flow_colorize(res)
        mag = np.linalg.norm(res.data, axis=-1)
        heat = cv2.applyColorMap(
            np.clip(mag / max(np.percentile(mag, 99), 1e-6) * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO)
        frame = cv2.resize(self.a, (res.width, res.height))
        vis = np.vstack([frame, wheel, heat])
        # Keep captions short enough to fit the panel width — the label box is
        # not clipped to the image, it just runs off the edge.
        put_label(vis, "frame 0 - stereo pair, so this is disparity", 8, 22,
                  (60, 60, 60))
        put_label(vis, "flow (Middlebury wheel: hue=direction, sat=magnitude)",
                  8, res.height + 22, (60, 60, 60))
        put_label(vis, "|flow| mean %.2f px, max %.2f px" % (mag.mean(), mag.max()),
                  8, 2 * res.height + 22, (60, 60, 60))
        return vis


class AnomalyTask:
    """One-class anomaly detection on the product the model was trained on.

    Runs BOTH a defective and a normal sample when the pair is staged, because
    the number that matters is not the defect's score but the GAP between the
    two — a model that scores everything high is useless and looks fine on a
    single defective part."""

    key = "anomaly"
    names = None

    def __init__(self, bcdl):
        import cv2

        self.bcdl = bcdl
        self.model_file = ANOMALY_MODEL
        self.image_file = ANOMALY_IMAGE
        self.img = cv2.imread(ANOMALY_IMAGE, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"cannot read the anomaly sample {ANOMALY_IMAGE}")
        self.good = cv2.imread(ANOMALY_GOOD, cv2.IMREAD_COLOR) if os.path.exists(ANOMALY_GOOD) else None
        self.engine = bcdl.Engine(self.model_file)
        self.det = bcdl.AnomalyDetector(self.engine)
        self.in_w, self.in_h = self.det.input_size
        self.x = np.ascontiguousarray(bcdl.anomaly_preprocess(self.img, self.in_w, self.in_h))

    def feed(self):
        self.engine._e.set_input(0, self.x)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        bad = self.det.detect(self.img)
        good = self.det.detect(self.good) if self.good is not None else None
        return bad, good

    def summarize(self, res):
        bad, good = res
        s = "defect score %.3f" % bad.score
        if good is not None:
            s += ", normal %.3f (gap %.3f)" % (good.score, bad.score - good.score)
        return s

    def draw(self, res):
        import cv2

        bad, good = res
        panels = []
        for img, m, tag in ((self.img, bad, "defective"),
                            (self.good, good, "normal")):
            if m is None:
                continue
            base = cv2.resize(img, (m.width, m.height))
            heat = cv2.applyColorMap((np.clip(m.data, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            over = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
            mask = self.bcdl.anomaly_mask(m, self.det.config.threshold)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, (255, 255, 255), 1)
            put_label(over, "%s  score %.3f" % (tag, m.score), 6, 20,
                      (0, 0, 170) if tag == "defective" else (0, 130, 0))
            panels.append(np.vstack([base, over]))
        return np.hstack(panels) if panels else self.img


class DriveTask:
    """End-to-end driving: camera + lidar + ego status -> a planned trajectory.

    The sample is a stored .npz of the four input tensors from one real frame,
    because the inputs are a sensor rig, not an image — reconstructing them from
    a photo would produce a confident trajectory for a scene the model never
    saw."""

    key = "drive"
    names = None
    BEV_COLORS = [(40, 40, 40), (90, 90, 90), (150, 120, 60), (60, 200, 200),
                  (200, 200, 60), (60, 60, 220), (220, 60, 220)]

    def __init__(self, bcdl):
        self.bcdl = bcdl
        self.model_file = DRIVE_MODEL
        self.image_file = DRIVE_SAMPLE
        d = np.load(DRIVE_SAMPLE)
        # Accept the tensors with or without a leading batch axis: the planner
        # only counts elements, but drawing needs to know which axis is channels.
        cam, lid = d["camera"], d["lidar"]
        self.camera = np.ascontiguousarray((cam[0] if cam.ndim == 4 else cam).astype(np.float32))
        self.lidar = np.ascontiguousarray((lid[0] if lid.ndim == 4 else lid).astype(np.float32))
        self.velocity = d["velocity"].astype(np.float32)
        self.gt = d["gt"].astype(np.float32) if "gt" in d else None

        self.engine = bcdl.Engine(self.model_file)
        acfg = bcdl.DriveAnchorConfig()
        acfg.noise_scale = 0.0            # deterministic, so the figure is stable
        self.planner = bcdl.DrivePlanner(
            self.engine, np.load(DRIVE_ANCHORS).astype(np.float32), anchor_config=acfg)
        self.in_h, self.in_w = self.camera.shape[-2], self.camera.shape[-1]
        self.status = bcdl.DriveStatus()
        self.status.velocity_x = float(self.velocity[0])
        self.status.velocity_y = float(self.velocity[1])

    def feed(self):
        self.planner.plan(self.camera, self.lidar, self.status)

    def infer(self):
        self.engine._e.infer(0)

    def decode(self):
        return self.planner.plan(self.camera, self.lidar, self.status)

    def summarize(self, res):
        p = np.array([[q.x, q.y] for q in res.trajectory])
        s = "%d poses, %.1f m ahead, %d vehicle(s)" % (
            len(res.trajectory), np.linalg.norm(np.diff(p, axis=0), axis=1).sum(),
            len(res.agents))
        if self.gt is not None:
            s += ", %.2f m from the realized future" % np.linalg.norm(
                p - self.gt[:, :2], axis=1).mean()
        return s

    def draw(self, res):
        """One fixed-size dashboard frame — also the unit the sequence video is
        made of, so the check image is literally one frame of it."""
        return draw_drive_frame(self.camera, self.lidar, res, self.velocity, self.gt)



# --- driving dashboard ------------------------------------------------------
# Kept as a free function rather than a method so the sequence video and the
# single check image are rendered by the SAME code: a check image that is not a
# frame of what you actually ship is a check of something else.
#
# The canvas is FIXED at 1024x768 — a hardware encoder wants a constant size
# (width %32, height %8), and a figure that resizes itself per frame also makes
# consecutive frames incomparable by eye.
DRIVE_BEV_CLASSES = ["background", "road", "walkway", "centreline",
                     "static", "vehicle", "pedestrian"]
# Drivable surface stays neutral so the overlays own the saturated colours; the
# three thin classes (centreline, vehicle, pedestrian) are the ones that have to
# survive being one pixel wide at this scale. Hex, because these go into a
# stylesheet — bev_svg.bgr() converts for the numpy raster layers.
DRIVE_BEV_COLORS = ["#12151a", "#3d434d", "#4a5566", "#e0d24a",
                    "#6b7280", "#e2574c", "#d968c4"]
DRIVE_FWD_M = 36.0       # forward window
DRIVE_BACK_M = 4.0       # ... plus a little behind, so the ego box is not clipped
DRIVE_LAT_M = 16.0       # +/- lateral window
DRIVE_PPM = 12.8         # px per metre (isotropic)
DRIVE_PANEL_W = int(2 * DRIVE_LAT_M * DRIVE_PPM)                  # 410
DRIVE_PANEL_H = int((DRIVE_FWD_M + DRIVE_BACK_M) * DRIVE_PPM)     # 512
DRIVE_FRAME_W, DRIVE_FRAME_H = 1024, 768


def _drive_window(grid, src_ppm, x0_row, y0_col, fill=0):
    """Crop a metric BEV array to the drawing window and orient it north-up.

    `grid` is indexed [x_bin, y_bin] with both indices INCREASING (forward,
    left) — the convention both the semantic map and the lidar histogram use.
    `x0_row`/`y0_col` are the array indices of x=0 and y=0; the window starts
    DRIVE_BACK_M behind that."""
    import cv2

    rows = int((DRIVE_FWD_M + DRIVE_BACK_M) * src_ppm)
    half = int(DRIVE_LAT_M * src_ppm)
    out = np.full((rows, 2 * half), fill, grid.dtype)
    x0_row = int(x0_row - DRIVE_BACK_M * src_ppm)
    pad = max(0, -x0_row)                     # window may start before the array
    x0_row = max(0, x0_row)
    r1 = min(grid.shape[0], x0_row + rows - pad)
    c0, c1 = max(0, y0_col - half), min(grid.shape[1], y0_col + half)
    out[pad:pad + (r1 - x0_row), (c0 - (y0_col - half)):(c1 - (y0_col - half))] = grid[x0_row:r1, c0:c1]
    # Flip both axes: forward must go UP the image and +y (left) must go LEFT.
    return out[::-1, ::-1]


def _drive_view():
    """The window both BEV panels use. Same class as the lidar detector's view,
    so "metres" means the same thing in every figure this file draws."""
    return bv.BevView(fwd_m=DRIVE_FWD_M, back_m=DRIVE_BACK_M, lat_m=DRIVE_LAT_M,
                      ppm=DRIVE_PPM)


def draw_drive_frame(camera, lidar, res, velocity, gt=None, tag=""):
    """1024x768: the camera strip the model is fed, the BEV map it produced, the
    lidar it was given, and the numbers — one frame of the check video.

    The two BEV panels share a window, a scale and a grid on purpose: the lidar
    is the model's other input, so putting it beside the map it produced is what
    makes the frame self-checking rather than decorative. Anything drawn in
    metres goes through the same `BevView` as the lidar detector's figure, so the
    two cannot drift apart.
    """
    import cv2

    view = _drive_view()
    pad = 12
    cam_h = DRIVE_FRAME_H - view.h - 3 * pad
    doc = bv.SVG(DRIVE_FRAME_W, DRIVE_FRAME_H)

    # --- camera strip --------------------------------------------------------
    cam = np.ascontiguousarray((camera.transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8))
    cam = cv2.resize(cam, (DRIVE_FRAME_W - 2 * pad, cam_h))
    cam_rgba = np.dstack([cam[..., ::-1], np.full(cam.shape[:2], 255, np.uint8)])
    doc.add('<clipPath id="camclip"><rect x="%d" y="%d" width="%d" height="%d" rx="8"/>'
            '</clipPath>' % (pad, pad, cam.shape[1], cam_h))
    doc.add('<g clip-path="url(#camclip)">')
    doc.raster(cam_rgba, pad, pad, cam.shape[1], cam_h)
    # The strip is three cameras concatenated; an unmarked seam looks like a lens
    # artefact. The thirds are equal by construction upstream.
    for k in (1, 2):
        x = pad + cam.shape[1] * k / 3.0
        doc.line(x, pad, x, pad + cam_h, "", 'stroke="%s" stroke-opacity=".55"' % bv.BG)
    doc.add('</g>')
    bv.panel_title(doc, pad, pad, cam.shape[1], "camera — three views, stitched",
                   tag or None)

    # --- the two BEV panels --------------------------------------------------
    lab = _drive_window(res.bev.labels, 4.0, 0, res.bev.width // 2)
    occ = _drive_window(lidar[0], 4.0, lidar.shape[1] // 2, lidar.shape[2] // 2)
    sx, sy = pad, cam_h + 2 * pad
    lx = sx + view.w + pad

    for i, (x0, title) in enumerate(((sx, "BEV semantic + plan"),
                                     (lx, "lidar BEV — model input"))):
        bv.panel_bg(doc, x0, sy, view.w, view.h)
        doc.add('<clipPath id="bev%d"><rect x="%d" y="%d" width="%d" height="%d" rx="8"/>'
                '</clipPath>' % (i, x0, sy, view.w, view.h))
        doc.add('<g clip-path="url(#bev%d)">' % i)
        if i == 0:
            doc.raster(bv.labels_rgba(lab, DRIVE_BEV_COLORS), x0, sy, view.w, view.h,
                       pixelated=True)
        else:
            # Occupancy is drawn in a neutral: it means "the sensor says
            # something is here", and an accent would make it argue with the
            # overlays it sits next to.
            doc.raster(bv.tint_rgba(np.clip(occ, 0, 1), bv.NEUTRAL), x0, sy,
                       view.w, view.h, pixelated=True)
        bv.polar_grid(doc, view, x0, sy, (10, 20, 30), spokes=False)
        # Both inputs stop at 32 m; a plan drawn past that line is extrapolation.
        y32 = sy + view.px(32.0, 0.0)[1]
        doc.line(x0, y32, x0 + view.w, y32, "grid2")
        doc.text(x0 + view.w - 10, y32 - 6, "input horizon", "sm", anchor="end")
        if i == 0:
            for a_ in res.agents:
                bv.box(doc, view, x0, sy, a_.x, a_.y, a_.heading, a_.length, a_.width,
                       bv.NEUTRAL, sw=1.2)
            bv.ribbon(doc, view, x0, sy,
                      [(0.0, 0.0)] + [(q.x, q.y) for q in res.trajectory], bv.ACCENT)
            if gt is not None:
                # AFTER the plan: the two coincide when the model is right, and
                # whichever goes down first is the one that disappears.
                bv.dashed(doc, view, x0, sy,
                          [(0.0, 0.0)] + [(float(q[0]), float(q[1])) for q in gt], bv.AMBER)
        bv.ego(doc, view, x0, sy)
        doc.add('</g>')
        bv.panel_title(doc, x0, sy, view.w, title)

    # --- readout -------------------------------------------------------------
    p = np.array([[q.x, q.y] for q in res.trajectory])
    plan_m = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    facts = [("speed", None, "%.1f m/s" % float(np.hypot(*velocity[:2]))),
             ("plan, 4 s", bv.ACCENT, "%.0f m" % plan_m),
             ("agents", bv.NEUTRAL, "%d" % len(res.agents))]
    if gt is not None:
        facts.append(("mean error", bv.AMBER,
                      "%.2f m" % float(np.linalg.norm(p - gt[:, :2], axis=1).mean())))
    legend = [("plan", bv.ACCENT, None)]
    if gt is not None:
        legend.append(("realized", bv.AMBER, None))
    legend += [("agent", bv.NEUTRAL, None), ("ego", bv.CYAN, None), (None,),
               ("lidar return", bv.NEUTRAL, None), (None,)]
    legend += [(DRIVE_BEV_CLASSES[i], DRIVE_BEV_COLORS[i], None) for i in range(1, 7)]

    ix = lx + view.w + pad
    iw = DRIVE_FRAME_W - ix - pad
    top_h = 33 + 20 + 21 * len(facts)
    y = bv.card(doc, ix, sy, iw, top_h, "frame")
    bv.rows(doc, ix, y, iw, facts)
    y = bv.card(doc, ix, sy + top_h + pad, iw, view.h - top_h - pad, "legend")
    bv.rows(doc, ix, y, iw, legend)
    return doc.render()


class LidarTask:
    """PointPillars: a sweep in, 3-D boxes out, through compiled stages.

    infer() times the BPU stages only; decode() is the whole chain including the
    CPU voxelize/scatter/decode, which is why the two numbers differ by more here
    than anywhere else in this suite.

    Handles both VFE forms and both head forms, dispatched on what the models
    declare rather than on their file names — see the note at LIDAR_VFE."""

    key = "lidar3d"
    names = LIDAR_NAMES

    def __init__(self, bcdl):
        self.bcdl = bcdl
        self.model_file = LIDAR_VFE
        # A calibrated frame if one is staged, else the published sample. The
        # published one is a bare point cloud — no image, no calibration, no
        # labels — so it can only support the bird's-eye half of the figure.
        self.frame = None
        if kitti_frame.KittiFrame.available(LIDAR_FRAME):
            self.frame = kitti_frame.KittiFrame(LIDAR_FRAME)
            self.points = np.ascontiguousarray(self.frame.points)
            self.image_file = os.path.join(LIDAR_FRAME, "velodyne.bin")
        else:
            # The published sample stores pillars, not points; the points are
            # still in there, so the chain can be driven from a real sweep.
            v = np.load(LIDAR_VOXELS)
            n = np.load(LIDAR_NUMPTS)
            self.points = np.ascontiguousarray(
                np.concatenate([v[i, :int(n[i])] for i in range(len(v))]).astype(np.float32))
            self.image_file = LIDAR_VOXELS
        self.cfg = bcdl.VoxelConfig()
        self.vfe = bcdl.Engine(LIDAR_VFE)
        self.head = bcdl.Engine(LIDAR_HEAD)
        # A convolution-form VFE takes one expanded tensor; the published one
        # takes the three raw pillar tensors.
        self.vfe_conv = self.vfe.num_inputs == 1
        # A head whose input is the scatter canvas has the backbone fused in.
        canvas = [1, 64, self.cfg.grid_y, self.cfg.grid_x]
        self.fused_head = self.head.input_shape(0) == canvas
        self.backbone = None if self.fused_head else bcdl.Engine(LIDAR_BACKBONE)
        self.det = bcdl.LidarDetector3d(self.head)
        self.in_w, self.in_h = self.cfg.grid_x, self.cfg.grid_y
        self.engine = self.head
        self.vox, self.npts, self.coords, self.stats = bcdl.voxelize_pillars(
            self.points, self.cfg)
        self._spatial = None

    def _vfe_inputs(self, vox, npts, coords, num_pillars):
        if not self.vfe_conv:
            return [vox, npts, coords]
        return [self.bcdl.pillar_point_features(vox, npts, coords, num_pillars, self.cfg)]

    def _pillar_features(self, out, num_pillars):
        """Both forms end up as (P, C); the convolution form emits (1, C, P, 1)."""
        if not self.vfe_conv:
            return out[:num_pillars]
        return np.ascontiguousarray(out.reshape(out.shape[1], -1).T[:num_pillars])

    def feed(self):
        for i, arr in enumerate(self._vfe_inputs(self.vox, self.npts, self.coords,
                                                 self.stats.num_pillars)):
            self.vfe._e.set_input(i, np.ascontiguousarray(arr))

    def infer(self):
        """The BPU stages, with the CPU scatter between them (the scatter cannot
        be hoisted out: the backbone's input IS its result)."""
        n = self.stats.num_pillars
        out = self.vfe.infer(self._vfe_inputs(self.vox, self.npts, self.coords, n))[0]
        spatial = self.bcdl.scatter_pillars(self._pillar_features(out, n),
                                            self.coords[:n], n, self.cfg)
        self._spatial = spatial
        if self.backbone is not None:
            spatial = self.backbone.infer([spatial])[0]
        self.head.infer([spatial])

    def decode(self):
        vox, npts, coords, stats = self.bcdl.voxelize_pillars(self.points, self.cfg)
        out = self.vfe.infer(self._vfe_inputs(vox, npts, coords, stats.num_pillars))[0]
        feats = self._pillar_features(out, stats.num_pillars)
        spatial = self.bcdl.scatter_pillars(feats,
                                            coords[:stats.num_pillars],
                                            stats.num_pillars, self.cfg)
        if self.backbone is not None:
            spatial = self.backbone.infer([spatial])[0]
        self.head.infer([spatial])
        return self.det.postprocess()

    def summarize(self, res):
        counts = [sum(1 for b in res if b.label == i) for i in range(3)]
        return "%d pillars from %d points; %d boxes (%dC/%dP/%dY)" % (
            self.stats.num_pillars, len(self.points), len(res), *counts)

    def draw(self, res):
        """The check figure.

        With a calibrated frame staged (docs/MODELS.md) this is the pair of views
        the field settles 3-D detection with: the boxes projected into the camera
        image, where a box either wraps the car or does not, and the same boxes
        over the sweep in bird's-eye view. Ground truth turns it from "plausible"
        into matched / missed / invented.

        Without one it degrades to the bird's-eye half, which is all an
        uncalibrated point cloud can support.
        """
        if self.frame is not None:
            return self._draw_calibrated(res)
        return self._draw_bev_only(res)

    # --- the calibrated figure ------------------------------------------
    # Colour carries the verdict, not the class: a detection that matched a
    # label, one that matched nothing, and a label nothing found are three
    # different things, and a figure that paints them all "car green" cannot be
    # used to judge the detector.
    MATCHED, FALSE_POS, MISSED = bv.ACCENT, bv.PINK, bv.AMBER

    def _draw_calibrated(self, res):
        fr = self.frame
        gts = fr.labels()
        # Only predictions the camera could see are scored. The labels exist only
        # for the camera's frustum, so a car detected behind the vehicle is not a
        # false positive — it is an object nobody wrote a label for.
        scored = [i for i, b in enumerate(res) if fr.in_camera(b)]
        unscored = len(res) - len(scored)
        sub = [res[i] for i in scored]
        sub_hit, gt_hit = kitti_frame.match(sub, gts, bev_iou=self.bcdl.bev_iou)
        pred_hit = [-2] * len(res)          # -2: outside the camera, not judged
        for k, i in enumerate(scored):
            pred_hit[i] = sub_hit[k]

        img = fr.image()
        pad = 16
        cam_w = 1216
        scale = cam_w / img.shape[1]
        cam_h = int(round(img.shape[0] * scale))
        view = bv.BevView(fwd_m=self.cfg.max_x, back_m=4.0, lat_m=self.cfg.max_y, ppm=7.6)
        card_w = cam_w - view.w - pad
        doc = bv.SVG(cam_w + 2 * pad, cam_h + view.h + 3 * pad)

        # -- camera panel: the boxes over the photograph they describe --------
        import cv2

        rgba = np.dstack([cv2.resize(img, (cam_w, cam_h))[..., ::-1],
                          np.full((cam_h, cam_w), 255, np.uint8)])
        doc.add('<clipPath id="camc"><rect x="%d" y="%d" width="%d" height="%d" rx="8"/>'
                '</clipPath>' % (pad, pad, cam_w, cam_h))
        doc.add('<g clip-path="url(#camc)">')
        doc.raster(rgba, pad, pad, cam_w, cam_h)

        def to_img(box):
            """8 projected corners, or None if the box is not in front of the
            camera — a box behind it still projects, to a mirrored ghost."""
            c3 = fr.box_corners(box)
            uv, depth = fr.project(c3)
            if depth.min() < 0.5:
                return None
            uv = uv * scale + pad
            if uv[:, 0].max() < pad or uv[:, 0].min() > pad + cam_w:
                return None
            return uv

        # Labels first and thin, so a prediction drawn over one still reads.
        for gi, g in enumerate(gts):
            uv = to_img(g)
            if uv is not None and gt_hit[gi] < 0:
                bv.cuboid(doc, uv, self.MISSED, dashed_edges=True, sw=1.3, front=False)
        for pi in sorted(range(len(res)), key=lambda i: -res[i].z):
            uv = to_img(res[pi])
            if uv is None:
                continue
            hit = pred_hit[pi] >= 0
            bv.cuboid(doc, uv, self.MATCHED if hit else self.FALSE_POS,
                      dashed_edges=not hit, sw=1.7)
        doc.add('</g>')
        bv.panel_title(doc, pad, pad, cam_w, "camera — the model's 3-D boxes, projected",
                       "KITTI %s" % fr.name)

        # -- bird's-eye panel -------------------------------------------------
        bx, by = pad, cam_h + 2 * pad
        bv.panel_bg(doc, bx, by, view.w, view.h)
        doc.add('<clipPath id="bevc"><rect x="%d" y="%d" width="%d" height="%d" rx="8"/>'
                '</clipPath>' % (bx, by, view.w, view.h))
        doc.add('<g clip-path="url(#bevc)">')
        bv.polar_grid(doc, view, bx, by, (10, 20, 30, 40, 50, 60))
        cloud, z_lo, z_hi = bv.points_rgba(view, self.points[:, :2], self.points[:, 2])
        doc.raster(cloud, bx, by, view.w, view.h)
        for gi, g in enumerate(gts):
            if gt_hit[gi] < 0:
                bv.box(doc, view, bx, by, g.x, g.y, g.heading, g.dx, g.dy,
                       self.MISSED, sw=1.2)
        for pi, p in enumerate(res):
            col = (self.MATCHED if pred_hit[pi] >= 0 else
                   bv.DIM if pred_hit[pi] == -2 else self.FALSE_POS)
            bv.box(doc, view, bx, by, p.x, p.y, p.heading, p.dx, p.dy, col)
        bv.ego(doc, view, bx, by)
        bv.scale_bar(doc, view, bx, by)
        doc.add('</g>')
        bv.panel_title(doc, bx, by, view.w, "bird's-eye — sweep and boxes",
                       "ego at origin · +x up")

        # -- scorecard --------------------------------------------------------
        sx = bx + view.w + pad
        tp = sum(1 for h in pred_hit if h >= 0)
        fp = sum(1 for h in pred_hit if h == -1)
        fn = sum(1 for h in gt_hit if h < 0)
        y = bv.card(doc, sx, by, card_w, view.h, "against the labels")
        y = bv.rows(doc, sx, y, card_w, [
            ("matched", self.MATCHED, "%d" % tp),
            ("false positive", self.FALSE_POS, "%d" % fp),
            ("missed", self.MISSED, "%d" % fn),
            (None,),
            ("outside the camera", bv.DIM, "%d" % unscored),
            (None,),
            ("recall", None, "%.0f%%" % (100.0 * tp / max(len(gts), 1))),
            ("precision", None, "%.0f%%" % (100.0 * tp / max(tp + fp, 1))),
        ])
        for k, line in enumerate((
                "Matched at KITTI's own BEV IoU thresholds:",
                "0.7 for cars, 0.5 for pedestrians and cyclists.",
                "Labels cover the camera's view only, so boxes",
                "outside it are shown but not scored.")):
            doc.text(sx + 14, y + 2 + 14 * k, line, "sm")
        y += 72
        counts = [sum(1 for b in res if b.label == i) for i in range(3)]
        gcounts = [sum(1 for g in gts if g.label == i) for i in range(3)]
        y = bv.rows(doc, sx, y, card_w, [("detected / labelled", None, None)] +
                    [(LIDAR_NAMES[i], None, "%d / %d" % (counts[i], gcounts[i]))
                     for i in range(3)] + [
            (None,),
            ("returns", None, "%d" % len(self.points)),
            ("pillars", None, "%d" % self.stats.num_pillars),
            ("out of range", None, "%d" % self.stats.points_out_of_range),
            ("pillar full", None, "%d" % self.stats.points_dropped_full),
            ("no slot left", None, "%d" % self.stats.pillars_dropped),
        ])
        doc.text(sx + 14, y + 10, "RETURN HEIGHT", "h2")
        bv.height_ramp(doc, sx + 14, y + 22, 13,
                       max(60, view.h - (y - by) - 46), z_lo, z_hi)
        return doc.render()

    def _draw_bev_only(self, res):
        """Bird's-eye only: the sweep, the boxes, and the counts.

        The point cloud is drawn at full density and coloured by height because
        the only check available here is "does this box sit on a car-shaped
        cluster", and a figure that thins the cloud or flattens it to one colour
        cannot be used to make it.
        """
        cfg = self.cfg
        view = bv.BevView(fwd_m=cfg.max_x, back_m=4.0, lat_m=cfg.max_y, ppm=9.0)
        pad, side_w = 16, 232
        doc = bv.SVG(pad * 3 + view.w + side_w, view.h + 2 * pad)
        px0, py0 = pad, pad

        bv.panel_bg(doc, px0, py0, view.w, view.h)
        doc.add('<clipPath id="bevclip"><rect x="%d" y="%d" width="%d" height="%d" rx="8"/>'
                '</clipPath>' % (px0, py0, view.w, view.h))
        doc.add('<g clip-path="url(#bevclip)">')
        bv.polar_grid(doc, view, px0, py0, (10, 20, 30, 40, 50, 60))
        rgba, z_lo, z_hi = bv.points_rgba(view, self.points[:, :2], self.points[:, 2])
        doc.raster(rgba, px0, py0, view.w, view.h)

        # Boxes over the cloud they are meant to be checked against; weakest
        # first, so a confident box is never hidden by a marginal one.
        for b in sorted(res, key=lambda b: b.score):
            bv.box(doc, view, px0, py0, b.x, b.y, b.heading, b.dx, b.dy,
                   bv.CLASS_COLORS[b.label % len(bv.CLASS_COLORS)])
        bv.ego(doc, view, px0, py0)
        placer = bv.ChipPlacer((px0 + 4, py0 + 44, px0 + view.w - 4, py0 + view.h - 26))
        for b in sorted(res, key=lambda b: -b.score):
            x, y = view.px(b.x, b.y)
            placer.add(doc, "%s %.2f" % (LIDAR_NAMES[b.label], b.score),
                       x + px0, y + py0 - b.dx / 2 * view.ppm,
                       bv.CLASS_COLORS[b.label % len(bv.CLASS_COLORS)])
        bv.scale_bar(doc, view, px0, py0)
        doc.add('</g>')
        bv.panel_title(doc, px0, py0, view.w, "lidar BEV — PointPillars",
                       "ego at origin · +x up · height-coloured returns")

        sx = px0 + view.w + pad
        counts = [sum(1 for b in res if b.label == i) for i in range(3)]
        y = bv.card(doc, sx, py0, side_w, view.h, "sweep")
        y = bv.rows(doc, sx, y, side_w, [
            ("returns", None, "%d" % len(self.points)),
            ("pillars", None, "%d" % self.stats.num_pillars),
            ("boxes", None, "%d" % len(res)),
            (None,),
        ] + [(LIDAR_NAMES[i], bv.CLASS_COLORS[i], "%d" % counts[i]) for i in range(3)] + [
            (None,),
            ("out of range", None, "%d" % self.stats.points_out_of_range),
            ("pillar full", None, "%d" % self.stats.points_dropped_full),
            ("no slot left", None, "%d" % self.stats.pillars_dropped),
        ])
        doc.text(sx + 14, y + 14, "RETURN HEIGHT", "h2")
        bv.height_ramp(doc, sx + 14, y + 26, 13, view.h - (y - py0) - 52, z_lo, z_hi)
        return doc.render()


def make_task(bcdl, key):
    """Factory: OcrTask for 'ocr', StereoTask for 'stereo', WholeBodyTask for
    'wholebody', FeaturesTask for 'features', SuperResTask for 'superres',
    ReidTask for 'reid', FlowTask / AnomalyTask / DriveTask for the 2026-08
    additions, else single-model Task."""
    if key == "lidar3d":
        return LidarTask(bcdl)
    if key == "flow":
        return FlowTask(bcdl)
    if key == "anomaly":
        return AnomalyTask(bcdl)
    if key == "drive":
        return DriveTask(bcdl)
    if key == "reid":
        return ReidTask(bcdl)
    if key == "features":
        return FeaturesTask(bcdl)
    if key in ("superres", "superres_span"):
        return SuperResTask(bcdl, key)
    if key == "ocr":
        return OcrTask(bcdl)
    if key == "stereo":
        return StereoTask(bcdl)
    if key == "wholebody":
        return WholeBodyTask(bcdl)
    return Task(bcdl, key)

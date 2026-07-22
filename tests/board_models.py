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
#   det: PP-OCRv5_server_det   [1,3,960,960] -> prob map [1,1,960,960]
#   rec: PP-OCRv5_server_rec   [1,3,48,320]  -> [1,T,18385] softmax (CTC)
#   cls: PP-LCNet_x1_0_textline_ori [1,3,80,160] -> [1,2] (0°/180°)
OCR_DET = os.environ.get("BCDL_OCR_DET", os.path.join(MODELS, "ppocrv5_server_det_960x960.hbm"))
OCR_REC = os.environ.get("BCDL_OCR_REC", os.path.join(MODELS, "ppocrv5_server_rec_48x320.hbm"))
# Direction classifier — OPTIONAL: the pipeline skips the 180° flip when absent.
OCR_CLS = os.environ.get("BCDL_OCR_CLS", os.path.join(MODELS, "ppocrv5_lcnet_cls_80x160.hbm"))
OCR_DICT = os.environ.get("BCDL_OCR_DICT",
                          os.path.join(_REPO, "data", "ppocr_keys_v5_18385.txt"))
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
        elif k == "semseg_rt":
            cfg = b.SegConfig(); cfg.channels_first = True   # PIDNet is NCHW [1,C,H,W]
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
        if k in ("semseg", "semseg_rt"):   # 2-input NV12 -> feed both, then argmax decode
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
        if k in ("cls", "semseg", "semseg_rt", "depth"):
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
        if k in ("semseg", "semseg_rt"):
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
        if k == "semseg":                              # palette overlay
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


def make_task(bcdl, key):
    """Factory: OcrTask for 'ocr', StereoTask for 'stereo', WholeBodyTask for
    'wholebody', FeaturesTask for 'features', SuperResTask for 'superres',
    ReidTask for 'reid', else single-model Task."""
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

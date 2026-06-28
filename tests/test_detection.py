"""Detection post-processing tests.

These are PURE-NUMPY tests of the decode + NMS reference (the oracle that the
C++ ``bcdl::decode`` / ``bcdl::nms`` mirror). They need only numpy and run
anywhere — no board, no .hbm model.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_detection.py

The module-level ``ref_*`` functions double as the documented "numpy Python
path": given a raw model-output tensor and a letterbox, they produce the same
Detections the C++ Detector does. If the C++ bindings later expose ``decode`` /
``nms`` they are exercised opportunistically (guarded import), but the core
assertions never depend on them.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Letterbox geometry — mirrors include/bcdl/preproc/geometry.h                 #
# --------------------------------------------------------------------------- #
class Letterbox:
    def __init__(self, scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h):
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.src_w = src_w
        self.src_h = src_h
        self.dst_w = dst_w
        self.dst_h = dst_h

    def inv_x(self, x):
        return (x - self.pad_x) / self.scale

    def inv_y(self, y):
        return (y - self.pad_y) / self.scale

    def clamp_x(self, x):
        return min(max(x, 0.0), float(self.src_w))

    def clamp_y(self, y):
        return min(max(y, 0.0), float(self.src_h))


def compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True):
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    pad_x = (dst_w - new_w) * 0.5 if center_pad else 0.0
    pad_y = (dst_h - new_h) * 0.5 if center_pad else 0.0
    return Letterbox(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)


# --------------------------------------------------------------------------- #
# Reference decode + NMS oracle                                               #
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = area_a + area_b - inter
    return inter / uni if uni > 0.0 else 0.0


def ref_nms(dets, iou_thresh, max_dets):
    """dets: list of dicts {x1,y1,x2,y2,score,class_id}. Per-class greedy NMS.
    Returns kept indices (into dets), truncated to max_dets."""
    order = sorted(range(len(dets)), key=lambda i: dets[i]["score"], reverse=True)
    suppressed = [False] * len(dets)
    keep = []
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        if max_dets > 0 and len(keep) >= max_dets:
            break
        bi = (dets[i]["x1"], dets[i]["y1"], dets[i]["x2"], dets[i]["y2"])
        for j in order[oi + 1:]:
            if suppressed[j] or dets[j]["class_id"] != dets[i]["class_id"]:
                continue
            bj = (dets[j]["x1"], dets[j]["y1"], dets[j]["x2"], dets[j]["y2"])
            if ref_iou(bi, bj) > iou_thresh:
                suppressed[j] = True
    return keep


def ref_decode(data, shape, cfg, lb):
    """Pure-numpy mirror of bcdl::decode. `data` is a flat row-major float array.

    cfg keys: num_classes, conf_thresh, iou_thresh, max_dets, layout
              ('yolov8'|'yolov5'), channels_first (bool), apply_sigmoid (bool).
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    nc = cfg["num_classes"]
    has_obj = cfg["layout"] == "yolov5"
    attrs = (5 if has_obj else 4) + nc
    total = int(np.prod(shape))
    N = total // attrs

    def at(a, j):
        off = a * N + j if cfg["channels_first"] else j * attrs + a
        return data[off]

    cls_start = 5 if has_obj else 4
    dets = []
    for j in range(N):
        cls_vals = np.array([at(cls_start + k, j) for k in range(nc)])
        best_k = int(np.argmax(cls_vals))
        best_raw = float(cls_vals[best_k])
        cls_score = _sigmoid(best_raw) if cfg["apply_sigmoid"] else best_raw
        if has_obj:
            obj = at(4, j)
            if cfg["apply_sigmoid"]:
                obj = _sigmoid(obj)
            score = float(obj * cls_score)
        else:
            score = float(cls_score)
        if score < cfg["conf_thresh"]:
            continue
        cx, cy, w, h = at(0, j), at(1, j), at(2, j), at(3, j)
        mx1, my1 = cx - w / 2, cy - h / 2
        mx2, my2 = cx + w / 2, cy + h / 2
        dets.append({
            "x1": lb.clamp_x(lb.inv_x(mx1)),
            "y1": lb.clamp_y(lb.inv_y(my1)),
            "x2": lb.clamp_x(lb.inv_x(mx2)),
            "y2": lb.clamp_y(lb.inv_y(my2)),
            "score": score,
            "class_id": best_k,
        })
    keep = ref_nms(dets, cfg["iou_thresh"], cfg["max_dets"])
    return [dets[i] for i in keep]


# --------------------------------------------------------------------------- #
# Synthetic tensor builders                                                   #
# --------------------------------------------------------------------------- #
def _default_cfg(**kw):
    cfg = dict(num_classes=3, conf_thresh=0.25, iou_thresh=0.45, max_dets=300,
               layout="yolov8", channels_first=True, apply_sigmoid=False)
    cfg.update(kw)
    return cfg


def build_yolov8(boxes, channels_first, num_classes, bg=0.0):
    """boxes: list of (cx,cy,w,h, class_id, class_score). Builds a [1, attrs, N]
    or [1, N, attrs] tensor (no objectness). `bg` fills the non-target class
    slots (use a strong negative when feeding logits + apply_sigmoid)."""
    N = len(boxes)
    attrs = 4 + num_classes
    t = np.zeros((attrs, N), dtype=np.float32)  # channels_first natural form
    if bg != 0.0:
        t[4:, :] = bg
    for j, (cx, cy, w, h, cid, sc) in enumerate(boxes):
        t[0, j], t[1, j], t[2, j], t[3, j] = cx, cy, w, h
        t[4 + cid, j] = sc
    if channels_first:
        return t.reshape(1, attrs, N), [1, attrs, N]
    return t.T.reshape(1, N, attrs), [1, N, attrs]


def build_yolov5(boxes, channels_first, num_classes):
    """boxes: list of (cx,cy,w,h, class_id, obj, class_score). [1,N,5+nc] etc."""
    N = len(boxes)
    attrs = 5 + num_classes
    t = np.zeros((attrs, N), dtype=np.float32)
    for j, (cx, cy, w, h, cid, obj, sc) in enumerate(boxes):
        t[0, j], t[1, j], t[2, j], t[3, j] = cx, cy, w, h
        t[4, j] = obj
        t[5 + cid, j] = sc
    if channels_first:
        return t.reshape(1, attrs, N), [1, attrs, N]
    return t.T.reshape(1, N, attrs), [1, N, attrs]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("channels_first", [True, False])
def test_yolov8_geometry_and_argmax(channels_first):
    # Identity letterbox (input == source, no scale/pad) makes the math trivial.
    lb = compute_letterbox(640, 640, 640, 640)
    assert lb.scale == 1.0 and lb.pad_x == 0.0

    # one box: center (100,200), size 40x60, class 2 strong.
    boxes = [(100, 200, 40, 60, 2, 0.9)]
    data, shape = build_yolov8(boxes, channels_first, num_classes=3)
    cfg = _default_cfg(channels_first=channels_first)
    out = ref_decode(data, shape, cfg, lb)

    assert len(out) == 1
    d = out[0]
    assert d["class_id"] == 2
    assert d["score"] == pytest.approx(0.9, abs=1e-5)
    assert d["x1"] == pytest.approx(80.0, abs=1e-4)   # 100 - 20
    assert d["y1"] == pytest.approx(170.0, abs=1e-4)  # 200 - 30
    assert d["x2"] == pytest.approx(120.0, abs=1e-4)
    assert d["y2"] == pytest.approx(230.0, abs=1e-4)


def test_threshold_filters_low_score():
    lb = compute_letterbox(640, 640, 640, 640)
    boxes = [
        (100, 100, 20, 20, 0, 0.90),  # keep
        (300, 300, 20, 20, 1, 0.10),  # drop (< 0.25)
    ]
    data, shape = build_yolov8(boxes, True, 3)
    out = ref_decode(data, shape, _default_cfg(), lb)
    assert len(out) == 1
    assert out[0]["class_id"] == 0


def test_nms_dedup_same_class():
    lb = compute_letterbox(640, 640, 640, 640)
    # two heavily overlapping same-class boxes -> NMS keeps only the higher.
    boxes = [
        (100, 100, 50, 50, 0, 0.9),
        (104, 104, 50, 50, 0, 0.8),
    ]
    data, shape = build_yolov8(boxes, True, 3)
    out = ref_decode(data, shape, _default_cfg(), lb)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.9)


def test_nms_keeps_different_classes():
    lb = compute_letterbox(640, 640, 640, 640)
    # same location, different class -> per-class NMS keeps both.
    boxes = [
        (100, 100, 50, 50, 0, 0.9),
        (100, 100, 50, 50, 1, 0.8),
    ]
    data, shape = build_yolov8(boxes, True, 3)
    out = ref_decode(data, shape, _default_cfg(), lb)
    assert len(out) == 2
    assert {d["class_id"] for d in out} == {0, 1}


def test_letterbox_inverse_mapping():
    # 1280x720 source into 640x640 canvas: scale 0.5, pad_y = (640-360)/2 = 140.
    lb = compute_letterbox(1280, 720, 640, 640)
    assert lb.scale == pytest.approx(0.5)
    assert lb.pad_x == pytest.approx(0.0)
    assert lb.pad_y == pytest.approx(140.0)

    # model-space center (320, 320) -> source (640, 360).
    boxes = [(320, 320, 100, 100, 0, 0.9)]
    data, shape = build_yolov8(boxes, True, 3)
    out = ref_decode(data, shape, _default_cfg(), lb)
    assert len(out) == 1
    d = out[0]
    # x1 model 270 -> src (270-0)/0.5 = 540 ; y1 model 270 -> (270-140)/0.5 = 260
    assert d["x1"] == pytest.approx(540.0, abs=1e-3)
    assert d["y1"] == pytest.approx(260.0, abs=1e-3)
    assert d["x2"] == pytest.approx(740.0, abs=1e-3)
    # y2 model 370 -> (370-140)/0.5 = 460 (< src_h 720, no clamp).
    assert d["y2"] == pytest.approx(460.0, abs=1e-3)


def test_letterbox_clamps_to_source():
    lb = compute_letterbox(1280, 720, 640, 640)
    # huge box centered top-left -> negative coords clamp to 0.
    boxes = [(10, 10, 400, 400, 0, 0.9)]
    data, shape = build_yolov8(boxes, True, 3)
    out = ref_decode(data, shape, _default_cfg(), lb)
    d = out[0]
    assert d["x1"] == 0.0
    assert d["y1"] == 0.0
    assert 0.0 <= d["x2"] <= lb.src_w
    assert 0.0 <= d["y2"] <= lb.src_h


@pytest.mark.parametrize("channels_first", [True, False])
def test_yolov5_objectness(channels_first):
    lb = compute_letterbox(640, 640, 640, 640)
    # score = obj * cls. obj=0.8, cls=0.5 -> 0.40 (>0.25 keep). obj=0.2,cls=0.9->0.18 drop.
    boxes = [
        (100, 100, 20, 20, 1, 0.8, 0.5),  # 0.40 keep, class 1
        (300, 300, 20, 20, 2, 0.2, 0.9),  # 0.18 drop
    ]
    data, shape = build_yolov5(boxes, channels_first, num_classes=3)
    cfg = _default_cfg(layout="yolov5", channels_first=channels_first)
    out = ref_decode(data, shape, cfg, lb)
    assert len(out) == 1
    assert out[0]["class_id"] == 1
    assert out[0]["score"] == pytest.approx(0.4, abs=1e-5)


def test_apply_sigmoid_logits():
    lb = compute_letterbox(640, 640, 640, 640)
    # logit 2.0 -> sigmoid ~0.880 (keep). logit -2.0 -> 0.119 (drop).
    boxes = [
        (100, 100, 20, 20, 0, 2.0),
        (300, 300, 20, 20, 1, -2.0),
    ]
    data, shape = build_yolov8(boxes, True, 3, bg=-10.0)
    cfg = _default_cfg(apply_sigmoid=True)
    out = ref_decode(data, shape, cfg, lb)
    assert len(out) == 1
    assert out[0]["class_id"] == 0
    assert out[0]["score"] == pytest.approx(_sigmoid(2.0), abs=1e-6)


def test_max_dets_truncation():
    lb = compute_letterbox(640, 640, 640, 640)
    # 5 well-separated boxes, max_dets=2 -> keep top 2 by score.
    boxes = [(50 + 80 * i, 50, 20, 20, 0, 0.5 + 0.05 * i) for i in range(5)]
    data, shape = build_yolov8(boxes, True, 3)
    cfg = _default_cfg(max_dets=2)
    out = ref_decode(data, shape, cfg, lb)
    assert len(out) == 2
    scores = sorted((d["score"] for d in out), reverse=True)
    assert scores[0] == pytest.approx(0.70, abs=1e-5)  # i=4
    assert scores[1] == pytest.approx(0.65, abs=1e-5)  # i=3


# Opportunistic check against the C++ bindings if they are ever exposed. The
# core tests above never depend on this.
def test_cpp_bindings_if_available():
    try:
        import bcdl  # noqa: F401
        has = hasattr(bcdl, "decode") and hasattr(bcdl, "nms")
    except Exception:
        has = False
    if not has:
        pytest.skip("bcdl.decode/nms bindings not exposed")
    # If/when bound, a parity check would go here.

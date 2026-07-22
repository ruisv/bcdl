"""Anchor-based (YOLOv5-style) detection decode tests, numpy path.

Exercises the Engine-free `bcdl.decode_yolov5_anchor` binding against an
independent numpy implementation of the same formula, plus hand-computed cases
whose answer is known by construction. Needs only the compiled bcdl extension
(no .hbm / board); the real-model path is in tests/test_board_models.py.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_panoptic_drive.py

WHY A SEPARATE DECODE FROM THE REST OF THE LIBRARY. Everything else here decodes
anchor-FREE heads (LTRB / DFL), where a cell predicts distances to the box
edges. An anchor-BASED head instead predicts, per prior box, an offset from the
cell centre and a multiplier on that prior's size — a different formula that
cannot be expressed in the LTRB decoder's terms.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not all(hasattr(bcdl, n) for n in ("decode_yolov5_anchor", "AnchorDetectConfig",
                                          "Anchor")):
        pytest.skip("anchor decode bindings not exposed")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _numpy_decode(raw, stride, anchors, num_classes):
    """Independent reference: the published anchor decode, written out plainly.

    raw: [na*(5+nc), H, W] unactivated. Returns (boxes_xywh, scores, class_ids)
    in MODEL-INPUT pixels, with no thresholding or NMS."""
    C, H, W = raw.shape
    na = len(anchors)
    no = 5 + num_classes
    assert C == na * no
    t = raw.reshape(na, no, H, W).transpose(0, 2, 3, 1)      # (na,H,W,no)
    y = _sigmoid(t)
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.stack((gx, gy), -1).astype(np.float32)         # (H,W,2)
    a = np.array([[p.w, p.h] for p in anchors], np.float32).reshape(na, 1, 1, 2)
    xy = (y[..., 0:2] * 2 - 0.5 + grid) * stride
    wh = (y[..., 2:4] * 2) ** 2 * a
    obj = y[..., 4]
    cls = y[..., 5:]
    return (np.concatenate([xy, wh], -1).reshape(-1, 4),
            (obj * cls.max(-1)).reshape(-1),
            cls.argmax(-1).reshape(-1))


def _corners_clamped(boxes, w, h):
    """xywh -> corners, clamped to the source extent — decodeYoloV5Anchor maps
    boxes back through the letterbox and clamps, so the reference must too."""
    x1 = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, w)
    y1 = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, h)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, w)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, h)
    return np.stack([x1, y1, x2, y2], -1)


def _identity_lb():
    """A no-op letterbox: model pixels == original pixels, so decoded boxes can
    be compared directly against the formula's model-space output."""
    return bcdl.compute_letterbox(64, 64, 64, 64, True)


def _cfg(num_classes=1, conf=0.0, iou=1.0, strides=(8,), anchors=None):
    cfg = bcdl.AnchorDetectConfig()
    cfg.num_classes = num_classes
    cfg.conf_thresh = conf
    cfg.iou_thresh = iou            # 1.0 => NMS suppresses nothing
    cfg.max_dets = 10000
    cfg.strides = list(strides)
    cfg.anchors = anchors if anchors is not None else [[bcdl.Anchor(10.0, 20.0)]]
    return cfg


def test_single_cell_known_by_construction():
    """One anchor, one cell, all raw logits zero => sigmoid = 0.5 everywhere, so
    every term of the formula collapses to a value that can be worked out by
    hand: centre = (0.5*2 - 0.5 + 0) * stride, size = (0.5*2)^2 * anchor."""
    _have()
    stride, aw, ah = 8, 10.0, 20.0
    raw = np.zeros((1 * 6, 1, 1), np.float32)   # na=1, nc=1 -> 6 channels
    cfg = _cfg(strides=(stride,), anchors=[[bcdl.Anchor(aw, ah)]])
    dets = bcdl.decode_yolov5_anchor([raw], cfg, _identity_lb())

    assert len(dets) == 1
    d = dets[0]
    cx = (0.5 * 2 - 0.5 + 0) * stride           # = 4.0
    cy = cx
    w = (0.5 * 2) ** 2 * aw                     # = 10.0
    h = (0.5 * 2) ** 2 * ah                     # = 20.0
    # Boxes are clamped to the source extent (here 64x64), which bites on the
    # left/top edge: cx - w/2 = -1 becomes 0.
    assert d.x1 == pytest.approx(max(0.0, cx - w / 2), abs=1e-4)
    assert d.y1 == pytest.approx(max(0.0, cy - h / 2), abs=1e-4)
    assert d.x2 == pytest.approx(min(64.0, cx + w / 2), abs=1e-4)
    assert d.y2 == pytest.approx(min(64.0, cy + h / 2), abs=1e-4)
    assert d.score == pytest.approx(0.25, abs=1e-4)   # obj 0.5 * cls 0.5
    assert d.class_id == 0


def test_matches_numpy_reference_random():
    """Random logits, no threshold and no suppression: every candidate must come
    back, and match the independent numpy formula box-for-box."""
    _have()
    rng = np.random.default_rng(7)
    stride, H, W = 16, 5, 4
    anchors = [bcdl.Anchor(3.0, 9.0), bcdl.Anchor(5.0, 11.0), bcdl.Anchor(4.0, 20.0)]
    raw = rng.normal(0, 1.5, (len(anchors) * 6, H, W)).astype(np.float32)

    cfg = _cfg(strides=(stride,), anchors=[anchors])
    dets = bcdl.decode_yolov5_anchor([raw], cfg, _identity_lb())

    boxes, scores, _ = _numpy_decode(raw, stride, anchors, 1)
    assert len(dets) == boxes.shape[0] == len(anchors) * H * W

    # Both sides are score-ordered after NMS's sort, so compare as sorted sets.
    got = sorted([(round(d.score, 5), round(d.x1, 3), round(d.y1, 3),
                   round(d.x2, 3), round(d.y2, 3)) for d in dets])
    corners = _corners_clamped(boxes, 64.0, 64.0)
    want = sorted([(round(float(s), 5), round(float(c[0]), 3), round(float(c[1]), 3),
                    round(float(c[2]), 3), round(float(c[3]), 3))
                   for c, s in zip(corners, scores)])
    for g, w_ in zip(got, want):
        assert g[0] == pytest.approx(w_[0], abs=1e-4)
        for i in range(1, 5):
            assert g[i] == pytest.approx(w_[i], abs=2e-3)


def test_multi_scale_pools_all_candidates():
    """Three scales are decoded into one pooled candidate list."""
    _have()
    anchors = [[bcdl.Anchor(3, 9)], [bcdl.Anchor(7, 18)], [bcdl.Anchor(19, 50)]]
    raws = [np.zeros((6, 8, 8), np.float32),
            np.zeros((6, 4, 4), np.float32),
            np.zeros((6, 2, 2), np.float32)]
    cfg = _cfg(strides=(8, 16, 32), anchors=anchors)
    dets = bcdl.decode_yolov5_anchor(raws, cfg, _identity_lb())
    assert len(dets) == 8 * 8 + 4 * 4 + 2 * 2


def test_objectness_threshold_filters():
    """Score is objectness * class probability, so a low objectness kills the
    candidate regardless of how confident the class score is."""
    _have()
    raw = np.zeros((6, 1, 1), np.float32)
    raw[4, 0, 0] = -10.0            # objectness logit -> sigmoid ~ 4.5e-5
    raw[5, 0, 0] = 10.0             # class logit -> sigmoid ~ 1.0
    cfg = _cfg(conf=0.01, strides=(8,))
    assert bcdl.decode_yolov5_anchor([raw], cfg, _identity_lb()) == []

    raw[4, 0, 0] = 10.0             # confident objectness -> survives
    assert len(bcdl.decode_yolov5_anchor([raw], cfg, _identity_lb())) == 1


def test_class_argmax_across_classes():
    _have()
    nc = 3
    raw = np.zeros((1 * (5 + nc), 1, 1), np.float32)
    raw[4, 0, 0] = 10.0             # objectness high
    raw[5 + 2, 0, 0] = 10.0         # class 2 wins
    cfg = _cfg(num_classes=nc, conf=0.1, strides=(8,))
    dets = bcdl.decode_yolov5_anchor([raw], cfg, _identity_lb())
    assert len(dets) == 1 and dets[0].class_id == 2


def test_letterbox_inverse_is_applied():
    """Boxes must come back in ORIGINAL-image pixels. Letterboxing a 128x64
    source into a 64x64 canvas halves the scale, so a decoded model-space box
    maps back to twice its coordinates."""
    _have()
    lb_id = _identity_lb()
    lb = bcdl.compute_letterbox(128, 64, 64, 64, True)   # scale 0.5, pad_y 16
    raw = np.zeros((6, 1, 1), np.float32)
    cfg = _cfg(strides=(8,), anchors=[[bcdl.Anchor(10.0, 20.0)]])

    d_model = bcdl.decode_yolov5_anchor([raw], cfg, lb_id)[0]
    d_orig = bcdl.decode_yolov5_anchor([raw], cfg, lb)[0]
    assert d_orig.x1 == pytest.approx((d_model.x1 - lb.pad_x) / lb.scale, abs=1e-3)
    assert d_orig.y1 == pytest.approx(
        max(0.0, (d_model.y1 - lb.pad_y) / lb.scale), abs=1e-3)


def test_length_mismatch_raises():
    _have()
    cfg = _cfg(strides=(8, 16), anchors=[[bcdl.Anchor(3, 9)]])   # 2 strides, 1 anchor set
    with pytest.raises(Exception):
        bcdl.decode_yolov5_anchor([np.zeros((6, 2, 2), np.float32)], cfg, _identity_lb())

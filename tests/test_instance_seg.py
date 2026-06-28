"""Instance-seg decode binding tests (C++ decodeInstanceSeg via nanobind).

Exercises ``bcdl.decode_instance_seg`` — the Engine-free numpy entry point that
mirrors InstanceSegmenter — with synthetic per-scale cls/box/mc tensors + a
prototype. Inputs are chosen so the full process_mask flow (proto·coef ->
sigmoid -> resize -> crop -> de-pad -> resize -> threshold) is hand-computable:
an IDENTITY letterbox (no pad, scale 1) and a constant prototype make every
resize an identity and the box crop the only discriminating step. Needs only the
compiled bcdl extension (no .hbm / board); skips cleanly otherwise.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_instance_seg.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _have():
    if not hasattr(bcdl, "decode_instance_seg"):
        pytest.skip("decode_instance_seg not exposed in this build")


def test_decode_instance_seg_left_half_mask():
    _have()
    O, S, NC, NP = 16, 8, 2, 1          # 16x16 image, stride 8 -> 2x2 grid
    H = W = O // S
    cls = np.full((H, W, NC), -10.0, np.float32)
    box = np.zeros((H, W, 4), np.float32)
    mc = np.zeros((H, W, NP), np.float32)
    proto = np.full((O, O, NP), 10.0, np.float32)   # sigmoid(1*10) ~ 1 everywhere

    # One active cell whose LTRB box spans the LEFT HALF [0,0,8,16] (model px):
    #   x1=(0.5-l)*8=0 -> l=0.5 ; x2=(0.5+r)*8=8 -> r=0.5
    #   y1=(0.5-t)*8=0 -> t=0.5 ; y2=(0.5+b)*8=16 -> b=1.5
    cls[0, 0] = [-10.0, 5.0]            # argmax class 1
    box[0, 0] = [0.5, 0.5, 0.5, 1.5]
    mc[0, 0] = [1.0]

    cfg = bcdl.InstanceSegConfig()
    cfg.conf_thresh = 0.25
    cfg.strides = [S]
    lb = bcdl.compute_letterbox(O, O, O, O)   # identity: scale=1, pad=0

    masks = bcdl.decode_instance_seg([cls], [box], [mc], proto, cfg, lb, O, O)
    assert len(masks) == 1
    m = masks[0]
    assert m.class_id == 1
    assert m.score == pytest.approx(_sigmoid(5.0), abs=1e-4)
    assert (m.x1, m.y1, m.x2, m.y2) == pytest.approx((0, 0, 8, 16), abs=1e-3)
    assert (m.mask_w, m.mask_h) == (O, O)

    mask = m.mask
    assert mask.shape == (O, O)
    assert mask.dtype == np.uint8
    # Mask = 1 over the cropped left half (x < 8), 0 elsewhere.
    assert mask[:, :8].all()
    assert not mask[:, 8:].any()
    assert int(mask.sum()) == O * 8


def test_decode_instance_seg_compute_masks_false_empty():
    _have()
    O, S, NC, NP = 16, 16, 1, 1         # 1x1 grid
    cls = np.full((1, 1, NC), 5.0, np.float32)
    box = np.array([[[0.5, 0.5, 0.5, 0.5]]], np.float32)   # box [0,0,16,16]
    mc = np.zeros((1, 1, NP), np.float32)
    proto = np.zeros((O, O, NP), np.float32)

    cfg = bcdl.InstanceSegConfig()
    cfg.conf_thresh = 0.25
    cfg.strides = [S]
    cfg.compute_masks = False
    lb = bcdl.compute_letterbox(O, O, O, O)

    masks = bcdl.decode_instance_seg([cls], [box], [mc], proto, cfg, lb, O, O)
    assert len(masks) == 1
    assert masks[0].mask.size == 0          # no mask assembled
    assert (masks[0].mask_w, masks[0].mask_h) == (O, O)


def test_decode_instance_seg_below_threshold_empty():
    _have()
    O, S, NC, NP = 16, 16, 1, 1
    cls = np.full((1, 1, NC), -10.0, np.float32)   # below conf_thresh
    box = np.zeros((1, 1, 4), np.float32)
    mc = np.zeros((1, 1, NP), np.float32)
    proto = np.zeros((O, O, NP), np.float32)
    cfg = bcdl.InstanceSegConfig()
    cfg.conf_thresh = 0.25
    cfg.strides = [S]
    lb = bcdl.compute_letterbox(O, O, O, O)
    masks = bcdl.decode_instance_seg([cls], [box], [mc], proto, cfg, lb, O, O)
    assert len(masks) == 0

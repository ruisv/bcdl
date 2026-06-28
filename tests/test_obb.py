"""OBB decode binding tests (C++ decodeObb via nanobind, numpy path).

Exercises ``bcdl.decode_obb`` — the Engine-free numpy entry point that mirrors
ObbDetector — with synthetic per-scale cls/box/angle tensors whose single active
cell has a hand-computed expected rotated box. Needs only the compiled bcdl
extension (no .hbm, no board hardware); skips cleanly where it can't import.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_obb.py

Decode contract mirrored here (src/tasks/obb.cc decodeObb):
  a_rad = (sigmoid(angle) - 0.5) * pi * sign + offset
  l,t,r,b = abs(box);  xf = (r-l)/2,  yf = (b-t)/2
  cx = (gx+0.5 + xf*cos a - yf*sin a) * S ;  cy = (gy+0.5 + xf*sin a + yf*cos a) * S
  w = (l+r)*S ;  h = (t+b)*S ;  if regularize and w<h: swap(w,h), a += pi/2
  class_id = argmax(cls), score = sigmoid(max logit); then rotated NMS + inv-LB.
"""

import math

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lb_identity(src=1000):
    return bcdl.compute_letterbox(src, src, src, src)  # scale=1, pad=0


def _have_decode_obb():
    if not hasattr(bcdl, "decode_obb"):
        pytest.skip("decode_obb not exposed in this build")


def test_decode_obb_single_cell_axis_aligned():
    _have_decode_obb()
    H, W, S, NC = 2, 2, 8, 3
    cls = np.full((H, W, NC), -10.0, np.float32)
    box = np.zeros((H, W, 4), np.float32)
    angle = np.zeros((H, W, 1), np.float32)        # sigmoid(0)=0.5 -> a_rad = 0

    gy, gx = 0, 1
    cls[gy, gx] = [-10.0, 8.0, -10.0]              # argmax class 1
    box[gy, gx] = [0.5, 0.5, 0.5, 0.5]             # l=t=r=b -> xf=yf=0, w=h

    cfg = bcdl.ObbConfig()
    cfg.num_classes = NC
    cfg.conf_thresh = 0.25
    cfg.strides = [S]
    cfg.regularize = False

    dets = bcdl.decode_obb([cls], [box], [angle], cfg, _lb_identity())
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 1
    assert d.score == pytest.approx(_sigmoid(8.0), abs=1e-4)

    r = d.rrect
    assert (r.cx, r.cy) == pytest.approx(((gx + 0.5) * S, (gy + 0.5) * S), abs=1e-3)
    assert (r.w, r.h) == pytest.approx((S, S), abs=1e-3)   # (l+r)*S, (t+b)*S
    assert r.angle == pytest.approx(0.0, abs=1e-4)


def test_decode_obb_regularize_swaps_wh_and_rotates():
    _have_decode_obb()
    # w < h triggers regularize: swap(w,h) and angle += pi/2. With angle raw 0
    # (a_rad=0) and a symmetric box (xf=yf=0) the center stays on the grid.
    H, W, S, NC = 1, 1, 8, 1
    cls = np.full((H, W, NC), 5.0, np.float32)
    box = np.array([[[0.25, 1.0, 0.25, 1.0]]], np.float32)  # w=(0.5)S < h=(2.0)S
    angle = np.zeros((H, W, 1), np.float32)

    cfg = bcdl.ObbConfig()
    cfg.num_classes = NC
    cfg.conf_thresh = 0.25
    cfg.strides = [S]
    cfg.regularize = True

    dets = bcdl.decode_obb([cls], [box], [angle], cfg, _lb_identity())
    assert len(dets) == 1
    r = dets[0].rrect
    assert (r.w, r.h) == pytest.approx((2.0 * S, 0.5 * S), abs=1e-3)  # swapped
    assert r.angle == pytest.approx(math.pi / 2, abs=1e-4)
    assert (r.cx, r.cy) == pytest.approx((0.5 * S, 0.5 * S), abs=1e-3)


def test_decode_obb_below_threshold_empty():
    _have_decode_obb()
    H, W, NC = 1, 1, 2
    cls = np.full((H, W, NC), -10.0, np.float32)   # sigmoid(-10) << 0.25
    box = np.zeros((H, W, 4), np.float32)
    angle = np.zeros((H, W, 1), np.float32)
    cfg = bcdl.ObbConfig()
    cfg.num_classes = NC
    cfg.conf_thresh = 0.25
    cfg.strides = [8]
    dets = bcdl.decode_obb([cls], [box], [angle], cfg, _lb_identity())
    assert len(dets) == 0

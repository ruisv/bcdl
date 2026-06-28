"""Pose decode binding tests (C++ decodePose via nanobind, numpy path).

Exercises ``bcdl.decode_pose`` — the Engine-free numpy entry point that mirrors
PoseEstimator — with synthetic per-scale cls/box/kpt tensors whose single active
cell has hand-computed expected box + keypoints. Needs only the compiled bcdl
extension (no .hbm, no board hardware); skips cleanly where it can't import.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_pose.py

Decode contract mirrored here (src/tasks/pose.cc decodePose):
  cx = gx+0.5, cy = gy+0.5
  box LTRB d -> model px [(cx-d0)*S, (cy-d1)*S, (cx+d2)*S, (cy+d3)*S], then inv-LB
  kpt k     -> model px [(raw_x+cx)*S, (raw_y+cy)*S], score = sigmoid(raw_s)
  person score = sigmoid(cls), class_id = 0, then box NMS.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lb_identity(src=1000):
    # scale=1, pad=0, src_w/h=src -> invX/Y are identity, no clamp for small coords.
    return bcdl.compute_letterbox(src, src, src, src)


def _have_decode_pose():
    if not hasattr(bcdl, "decode_pose"):
        pytest.skip("decode_pose not exposed in this build")


def test_decode_pose_single_cell():
    _have_decode_pose()
    H, W, S, K = 2, 2, 8, 2
    cls = np.full((H, W, 1), -10.0, np.float32)   # sigmoid(-10) ~ 0 -> below thresh
    box = np.zeros((H, W, 4), np.float32)
    kpt = np.zeros((H, W, K * 3), np.float32)

    gy, gx = 0, 1                                  # the one active cell
    cls[gy, gx, 0] = 10.0                          # score ~ 1
    box[gy, gx] = [0.5, 0.5, 1.5, 1.0]             # l, t, r, b
    kpt[gy, gx] = [0.0, 0.0, 5.0, 1.0, -0.5, -5.0]  # k0(x,y,s), k1(x,y,s)

    cfg = bcdl.PoseConfig()
    cfg.num_keypoints = K
    cfg.conf_thresh = 0.25
    cfg.strides = [S]

    dets = bcdl.decode_pose([cls], [box], [kpt], cfg, _lb_identity())
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 0
    assert d.score == pytest.approx(_sigmoid(10.0), abs=1e-4)

    cx, cy = gx + 0.5, gy + 0.5
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx(
        ((cx - 0.5) * S, (cy - 0.5) * S, (cx + 1.5) * S, (cy + 1.0) * S), abs=1e-3)

    assert len(d.keypoints) == K
    k0, k1 = d.keypoints[0], d.keypoints[1]
    assert (k0.x, k0.y) == pytest.approx(((0.0 + cx) * S, (0.0 + cy) * S), abs=1e-3)
    assert k0.score == pytest.approx(_sigmoid(5.0), abs=1e-4)
    assert (k1.x, k1.y) == pytest.approx(((1.0 + cx) * S, (-0.5 + cy) * S), abs=1e-3)
    assert k1.score == pytest.approx(_sigmoid(-5.0), abs=1e-4)


def test_decode_pose_below_threshold_empty():
    _have_decode_pose()
    H, W, K = 1, 1, 1
    cls = np.full((H, W, 1), -10.0, np.float32)    # sigmoid(-10) ~ 0
    box = np.zeros((H, W, 4), np.float32)
    kpt = np.zeros((H, W, K * 3), np.float32)
    cfg = bcdl.PoseConfig()
    cfg.num_keypoints = K
    cfg.conf_thresh = 0.25
    cfg.strides = [8]
    dets = bcdl.decode_pose([cls], [box], [kpt], cfg, _lb_identity())
    assert len(dets) == 0


def test_decode_pose_nms_dedups_overlap():
    _have_decode_pose()
    # Two adjacent active cells whose boxes overlap heavily -> NMS keeps the
    # higher-scoring one only.
    H, W, S, K = 1, 2, 8, 1
    cls = np.array([[[2.0], [3.0]]], np.float32)   # both > thresh; cell1 stronger
    # Wide boxes so the two cells' boxes overlap almost entirely.
    box = np.zeros((H, W, 4), np.float32)
    box[0, 0] = [2.0, 1.0, 2.0, 1.0]
    box[0, 1] = [3.0, 1.0, 1.0, 1.0]
    kpt = np.zeros((H, W, K * 3), np.float32)
    cfg = bcdl.PoseConfig()
    cfg.num_keypoints = K
    cfg.conf_thresh = 0.25
    cfg.iou_thresh = 0.5
    cfg.strides = [S]
    dets = bcdl.decode_pose([cls], [box], [kpt], cfg, _lb_identity())
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(_sigmoid(3.0), abs=1e-4)  # the stronger cell

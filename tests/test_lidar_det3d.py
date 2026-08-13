"""Lidar pillar encoding + 3-D detection: numpy path, no model required.

    PYTHONPATH=build:python pytest -s tests/test_lidar_det3d.py

Two things here are contracts rather than choices, and both fail silently when
broken: the pillar caps (a dense sweep overflows them and the detector goes blind
in exactly the busy scene it was bought for) and the anchor grid (a permuted or
half-pillar-shifted grid decodes into well-formed boxes in the wrong places).
"""
import numpy as np
import pytest

import bcdl


def _cfg(**kw):
    c = bcdl.VoxelConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --- pillar encoding -------------------------------------------------------

def test_points_land_in_the_right_pillar():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=16, max_points_per_pillar=4)
    assert (cfg.grid_x, cfg.grid_y) == (4, 4)
    pts = np.array([[0.5, -1.5, 0.0, 0.1],     # cell x=0, y=0
                    [3.5, 1.5, 0.0, 0.2]], np.float32)
    vox, npts, coords, st = bcdl.voxelize_pillars(pts, cfg)
    assert st.num_pillars == 2
    np.testing.assert_array_equal(coords[0], [0, 0, 0, 0])
    np.testing.assert_array_equal(coords[1], [0, 0, 3, 3])
    np.testing.assert_allclose(vox[0, 0], pts[0])
    assert npts[0] == 1 and npts[1] == 1


def test_points_in_one_column_share_a_pillar():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=16, max_points_per_pillar=4)
    pts = np.array([[0.1, -1.9, -0.5, 0.0],
                    [0.9, -1.1, 0.5, 0.0]], np.float32)
    vox, npts, coords, st = bcdl.voxelize_pillars(pts, cfg)
    assert st.num_pillars == 1
    assert npts[0] == 2


def test_out_of_range_points_are_counted_not_silently_lost():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, min_z=-1, max_z=1,
               pillar_x=1.0, pillar_y=1.0, max_pillars=16)
    pts = np.array([[1.0, 0.0, 0.0, 0.0],
                    [99.0, 0.0, 0.0, 0.0],     # beyond max_x
                    [1.0, 0.0, 5.0, 0.0],      # above max_z
                    [np.nan, 0.0, 0.0, 0.0]], np.float32)
    _, _, _, st = bcdl.voxelize_pillars(pts, cfg)
    assert st.points_used == 1
    assert st.points_out_of_range == 3


def test_caps_are_reported():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=2, max_points_per_pillar=2)
    # three distinct columns, and four points in the first
    pts = np.array([[0.5, -1.5, 0, 0], [0.5, -1.5, 0, 0], [0.5, -1.5, 0, 0],
                    [1.5, -1.5, 0, 0], [2.5, -1.5, 0, 0]], np.float32)
    _, npts, _, st = bcdl.voxelize_pillars(pts, cfg)
    assert st.num_pillars == 2                 # the third column had no slot
    assert st.pillars_dropped == 1
    assert st.points_dropped_full == 1         # the first column's third point
    assert npts[0] == 2


def test_scatter_puts_features_where_the_coords_say():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=8)
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    coords = np.array([[0, 0, 1, 2], [0, 0, 3, 0]], np.float32)   # (b, z, y, x)
    canvas = bcdl.scatter_pillars(feats, coords, 2, cfg)
    assert canvas.shape == (1, 2, 4, 4)
    assert canvas[0, 0, 1, 2] == 1.0 and canvas[0, 1, 1, 2] == 2.0
    assert canvas[0, 0, 3, 0] == 3.0 and canvas[0, 1, 3, 0] == 4.0
    assert canvas.sum() == pytest.approx(10.0)   # everything else is zero


# --- per-point feature expansion (the 1x1-convolution VFE's input) ----------
#
# A VFE that takes (pillars, points, 4) computes this inside its own graph. One
# re-exported as a 1x1 convolution takes it precomputed, which is what makes it
# 179x faster — so this arithmetic has to match the in-graph version exactly or
# the whole chain is quietly wrong.

def _reference_features(vox, npts, coords, cfg):
    """The upstream PillarVFE expansion, written straight out."""
    v = vox.astype(np.float64)
    n = np.maximum(npts.astype(np.float64), 1.0)[:, None, None]
    mean = v[:, :, :3].sum(1, keepdims=True) / n
    cluster = v[:, :, :3] - mean
    centre = np.empty_like(v[:, :, :3])
    pz = cfg.max_z - cfg.min_z
    centre[:, :, 0] = v[:, :, 0] - (coords[:, 3:4] * cfg.pillar_x + cfg.pillar_x / 2 + cfg.min_x)
    centre[:, :, 1] = v[:, :, 1] - (coords[:, 2:3] * cfg.pillar_y + cfg.pillar_y / 2 + cfg.min_y)
    centre[:, :, 2] = v[:, :, 2] - (coords[:, 1:2] * pz + pz / 2 + cfg.min_z)
    feats = np.concatenate([v, cluster, centre], axis=2)
    live = (np.arange(v.shape[1])[None, :] < npts.astype(np.int64)[:, None])[:, :, None]
    return (feats * live).transpose(2, 0, 1)[None]


def test_point_features_match_the_upstream_expansion():
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=8, max_points_per_pillar=4)
    rng = np.random.default_rng(3)
    pts = np.concatenate([rng.uniform([0.05, -1.95, -2.9], [3.95, 1.95, 0.9], (40, 3)),
                          rng.random((40, 1))], 1).astype(np.float32)
    vox, npts, coords, st = bcdl.voxelize_pillars(pts, cfg)
    got = bcdl.pillar_point_features(vox, npts, coords, st.num_pillars, cfg)
    assert got.shape == (1, 10, 8, 4)
    want = _reference_features(vox, npts, coords, cfg)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_unfilled_slots_and_unused_pillars_stay_zero():
    """They go through the MLP too. Left at the centroid arithmetic's output they
    would be a real point as far as the max over points is concerned."""
    cfg = _cfg(min_x=0, max_x=4, min_y=-2, max_y=2, pillar_x=1.0, pillar_y=1.0,
               max_pillars=8, max_points_per_pillar=4)
    # Deliberately off the pillar's centre, so the centre offsets are nonzero.
    pts = np.array([[0.3, -1.7, 0.5, 0.7]], np.float32)     # one point, one pillar
    vox, npts, coords, st = bcdl.voxelize_pillars(pts, cfg)
    f = bcdl.pillar_point_features(vox, npts, coords, st.num_pillars, cfg)
    assert not f[0, :, 0, 1:].any()      # the pillar's other 3 point slots
    assert not f[0, :, 1:, :].any()      # every unused pillar
    # A lone point is its own centroid, so the cluster offsets are exactly zero
    # while the centre offsets are not.
    np.testing.assert_allclose(f[0, 4:7, 0, 0], 0.0, atol=1e-6)
    assert abs(f[0, 7, 0, 0]) > 1e-6


def test_point_features_reject_a_pillar_count_past_the_cap():
    cfg = _cfg(max_pillars=8)
    vox = np.zeros((8, cfg.max_points_per_pillar, 4), np.float32)
    npts = np.zeros(8, np.float32)
    coords = np.zeros((8, 4), np.float32)
    with pytest.raises(Exception, match="out of range"):
        bcdl.pillar_point_features(vox, npts, coords, 9, cfg)


# --- anchors ---------------------------------------------------------------

def test_anchor_grid_spans_the_range_inclusively():
    """The reference generator lays anchors on a CLOSED linspace, not on cell
    centres — half a pillar of offset moves every decoded box."""
    a = bcdl.generate_anchors_3d(bcdl.Anchor3dConfig())
    assert a.shape == (248 * 216 * 3 * 2, 7)
    assert a[:, 0].min() == pytest.approx(0.0)
    assert a[:, 0].max() == pytest.approx(69.12, abs=1e-3)
    assert a[:, 1].min() == pytest.approx(-39.68, abs=1e-3)
    assert a[:, 1].max() == pytest.approx(39.68, abs=1e-3)


def test_anchor_ordering_is_class_major_then_rotation():
    a = bcdl.generate_anchors_3d(bcdl.Anchor3dConfig())
    # The first six rows are one cell: three classes x two yaws, in that order.
    np.testing.assert_allclose(a[0, 3:6], [3.9, 1.6, 1.56])
    np.testing.assert_allclose(a[1, 3:6], [3.9, 1.6, 1.56])
    np.testing.assert_allclose(a[2, 3:6], [0.8, 0.6, 1.73])
    np.testing.assert_allclose(a[4, 3:6], [1.76, 0.6, 1.73])
    assert a[0, 6] == pytest.approx(0.0)
    assert a[1, 6] == pytest.approx(1.57)
    # ... and all six share one position.
    assert len(set(map(tuple, np.round(a[:6, :2], 6)))) == 1


# --- decode ----------------------------------------------------------------

def _head(h=2, w=2, num_classes=3, num_anchors=6):
    cls = np.full((1, num_anchors * num_classes, h, w), -10.0, np.float32)
    box = np.zeros((1, num_anchors * 7, h, w), np.float32)
    dirp = np.zeros((1, num_anchors * 2, h, w), np.float32)
    return cls, box, dirp


def _small_cfg(h=2, w=2, **kw):
    a = bcdl.Anchor3dConfig()
    a.grid_x, a.grid_y = w, h
    c = bcdl.Det3dConfig()
    c.anchors = a
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_zero_residual_returns_the_anchor_itself():
    cls, box, dirp = _head()
    cls[0, 0, 0, 0] = 10.0                       # anchor slot 0, class 0
    cfg = _small_cfg(score_thresh=0.5)
    anchors = bcdl.generate_anchors_3d(cfg.anchors)
    boxes = bcdl.decode_det3d(cls, box, None, anchors, cfg)
    assert len(boxes) == 1
    b = boxes[0]
    a0 = anchors[0]
    # exp(0)=1 and 0*diag=0, so a zero residual must reproduce the anchor.
    assert (b.x, b.y, b.z) == pytest.approx((a0[0], a0[1], a0[2]))
    assert (b.dx, b.dy, b.dz) == pytest.approx((a0[3], a0[4], a0[5]))
    assert b.label == 0 and b.score > 0.99


def test_residuals_decode_through_the_anchor_diagonal():
    cls, box, dirp = _head()
    cls[0, 0, 0, 0] = 10.0
    box[0, 0, 0, 0] = 1.0        # x residual
    box[0, 3, 0, 0] = np.log(2)  # dx residual
    cfg = _small_cfg(score_thresh=0.5)
    anchors = bcdl.generate_anchors_3d(cfg.anchors)
    a0 = anchors[0]
    diag = np.hypot(a0[3], a0[4])
    b = bcdl.decode_det3d(cls, box, None, anchors, cfg)[0]
    assert b.x == pytest.approx(a0[0] + diag)
    assert b.dx == pytest.approx(a0[3] * 2, rel=1e-5)


def test_direction_classifier_flips_the_yaw_by_pi():
    cls, box, dirp = _head()
    cls[0, 0, 0, 0] = 10.0
    cfg = _small_cfg(score_thresh=0.5)
    anchors = bcdl.generate_anchors_3d(cfg.anchors)
    a = bcdl.decode_det3d(cls, box, dirp, anchors, cfg)[0]
    dirp[0, 1, 0, 0] = 5.0                        # "other half" wins
    b = bcdl.decode_det3d(cls, box, dirp, anchors, cfg)[0]
    assert abs(abs(b.heading - a.heading) - np.pi) < 1e-4


def test_class_channels_are_read_per_anchor_slot():
    cls, box, dirp = _head()
    cls[0, 1 * 3 + 2, 0, 0] = 10.0               # anchor slot 1, class 2
    cfg = _small_cfg(score_thresh=0.5)
    anchors = bcdl.generate_anchors_3d(cfg.anchors)
    b = bcdl.decode_det3d(cls, box, None, anchors, cfg)[0]
    assert b.label == 2
    np.testing.assert_allclose([b.dx, b.dy, b.dz], anchors[1, 3:6], rtol=1e-5)


# --- NMS -------------------------------------------------------------------

def _box(x, y, score, label=0, yaw=0.0):
    b = bcdl.Box3D()
    b.x, b.y, b.z = x, y, 0.0
    b.dx, b.dy, b.dz = 4.0, 2.0, 1.5
    b.heading, b.label, b.score = yaw, label, score
    return b


def test_bev_iou_ignores_height():
    a, b = _box(0, 0, 1.0), _box(0, 0, 1.0)
    b.z, b.dz = 50.0, 0.1                      # far above, different height
    assert bcdl.bev_iou(a, b) == pytest.approx(1.0, abs=1e-3)


def test_nms_suppresses_overlap_but_not_across_classes():
    boxes = [_box(0, 0, 0.9), _box(0.2, 0, 0.8), _box(0, 0, 0.7, label=1)]
    kept = bcdl.nms3d(boxes, 0.1, 10)
    assert len(kept) == 2
    assert sorted(b.label for b in kept) == [0, 1]


def test_nms_keeps_separated_boxes():
    kept = bcdl.nms3d([_box(0, 0, 0.9), _box(20, 0, 0.8)], 0.1, 10)
    assert len(kept) == 2

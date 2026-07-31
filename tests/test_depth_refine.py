"""RGB-D depth-refinement preprocessing / decode tests.

These are PURE-NUMPY tests of the references that ``src/tasks/depth_refine.cc``
mirrors. They need only numpy and run anywhere — no board, no .hbm model. When
the C++ bindings are importable they are exercised against the same oracles, so
a divergence between the two implementations fails here rather than on the board.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_depth_refine.py

Why the preprocessing is worth testing at all: the RGB downscale into the
encoder grid is AREA-AVERAGED, matching cv2.INTER_AREA to within uint8 rounding.
Measured against the reference implementation's antialiased bilinear across the
reference scenes, area averaging costs 6.1e-4 mean absolute relative depth error
where a plain bilinear resize costs 9.1e-4.
"""

import numpy as np
import pytest

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float64).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float64).reshape(3, 1, 1)

try:  # board / built-module path; the numpy assertions never depend on it
    import bcdl as _bcdl
except Exception:  # pragma: no cover - exercised only where the module is built
    _bcdl = None

_HAS_CPP = _bcdl is not None and hasattr(_bcdl, "preprocess_refine_image")


# --------------------------------------------------------------------------- #
# References (mirror src/tasks/depth_refine.cc)                                #
# --------------------------------------------------------------------------- #
def ref_area_table(src, dst):
    """cv::resize(INTER_AREA)'s weight table: overlap of [d*s,(d+1)*s] with cells."""
    scale = src / dst
    taps = []
    for d in range(dst):
        f0 = d * scale
        f1 = min(f0 + scale, float(src))
        s0 = int(np.floor(f0))
        s1 = max(s0 + 1, int(np.ceil(f1)))
        w = np.array([max(0.0, min(f1, s + 1) - max(f0, s)) for s in range(s0, s1)])
        total = w.sum()
        taps.append((s0, w / total if total > 0 else np.array([1.0])))
    return taps


def ref_area_resize(img, out_h, out_w):
    """Area-average resize of an (H, W, C) float image."""
    h, w = img.shape[:2]
    xt, yt = ref_area_table(w, out_w), ref_area_table(h, out_h)
    tmp = np.zeros((h, out_w, img.shape[2]), np.float64)
    for x, (s0, wts) in enumerate(xt):
        cols = np.clip(np.arange(s0, s0 + len(wts)), 0, w - 1)
        tmp[:, x] = np.tensordot(img[:, cols], wts, axes=([1], [0]))
    out = np.zeros((out_h, out_w, img.shape[2]), np.float64)
    for y, (s0, wts) in enumerate(yt):
        rows = np.clip(np.arange(s0, s0 + len(wts)), 0, h - 1)
        out[y] = np.tensordot(tmp[rows], wts, axes=([0], [0]))
    return out


def ref_preprocess_image(bgr, eh, ew):
    """BGR uint8 (H,W,3) -> normalized planar RGB (1,3,eh,ew)."""
    small = ref_area_resize(bgr.astype(np.float64), eh, ew)
    rgb = small[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return ((rgb - IMAGE_MEAN) / IMAGE_STD)[None]


def ref_preprocess_depth(depth_m, eh, ew, min_valid=0.01):
    """Metric depth (H,W) -> log depth (1,1,eh,ew), 0 where invalid."""
    h, w = depth_m.shape
    rows = np.minimum(h - 1, (np.arange(eh) * (h / eh)).astype(int))
    cols = np.minimum(w - 1, (np.arange(ew) * (w / ew)).astype(int))
    d = depth_m[np.ix_(rows, cols)]
    valid = np.isfinite(d) & (d > min_valid)
    out = np.zeros_like(d, dtype=np.float64)
    out[valid] = np.log(d[valid])
    return out[None, None]


def ref_decode(depth, mask_logit, threshold=0.5, apply_mask=True):
    keep = np.ones(depth.shape, bool) if mask_logit is None else (
        1.0 / (1.0 + np.exp(-mask_logit)) > threshold)
    usable = keep & np.isfinite(depth)
    out = np.where(usable | (not apply_mask), depth, 0.0)
    vals = depth[usable]
    return {
        "depth": out,
        "mask": keep.astype(np.uint8),
        "vmin": float(vals.min()) if vals.size else 0.0,
        "vmax": float(vals.max()) if vals.size else 0.0,
    }


def ref_pointcloud(depth, mask, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where((mask > 0) & np.isfinite(depth) & (depth > 0), depth, 0.0)
    x = np.where(z > 0, (u - cx) * z / fx, 0.0)
    y = np.where(z > 0, (v - cy) * z / fy, 0.0)
    return np.stack([x, y, z], -1)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def scene():
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    depth = rng.uniform(0.4, 6.0, (480, 640)).astype(np.float32)
    depth[100:180, 200:300] = 0.0          # a hole, as a real sensor leaves
    depth[300, 400] = np.inf               # and a non-finite reading
    return bgr, depth


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_area_table_weights_sum_to_one():
    for src, dst in ((640, 560), (480, 420), (1920, 560), (33, 7)):
        for start, w in ref_area_table(src, dst):
            assert w.sum() == pytest.approx(1.0)
            assert (w >= 0).all()
            assert start >= 0


def test_area_resize_preserves_a_constant_image():
    img = np.full((480, 640, 3), 37.0)
    out = ref_area_resize(img, 420, 560)
    assert np.allclose(out, 37.0)


def test_area_resize_differs_from_naive_subsampling():
    """The point of area averaging: it is not nearest-neighbour."""
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 255, (480, 640, 3))
    area = ref_area_resize(img, 420, 560)
    rows = (np.arange(420) * (480 / 420)).astype(int)
    cols = (np.arange(560) * (640 / 560)).astype(int)
    nearest = img[np.ix_(rows, cols)]
    assert np.abs(area - nearest).mean() > 1.0


def test_preprocess_depth_marks_invalid_as_zero(scene):
    _, depth = scene
    out = ref_preprocess_depth(depth, 420, 560)
    assert out.shape == (1, 1, 420, 560)
    # The hole spans a large region, so plenty of encoder pixels must be zero.
    assert (out == 0.0).sum() > 1000
    assert np.isfinite(out).all()


def test_preprocess_depth_is_log_of_valid_readings():
    depth = np.full((420, 560), 2.5, np.float32)
    out = ref_preprocess_depth(depth, 420, 560)
    assert np.allclose(out, np.log(2.5))


def test_decode_applies_mask_and_reports_range():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    logits = np.array([[5.0, -5.0], [5.0, 5.0]])  # reject the 2.0
    r = ref_decode(depth, logits)
    assert r["mask"].tolist() == [[1, 0], [1, 1]]
    assert r["depth"][0, 1] == 0.0
    assert (r["vmin"], r["vmax"]) == (1.0, 4.0)


def test_decode_without_mask_trusts_everything():
    depth = np.array([[1.0, 2.0]])
    r = ref_decode(depth, None)
    assert r["mask"].tolist() == [[1, 1]]
    assert r["depth"].tolist() == [[1.0, 2.0]]


def test_pointcloud_centre_pixel_is_on_the_axis():
    depth = np.full((5, 5), 2.0)
    mask = np.ones((5, 5), np.uint8)
    pts = ref_pointcloud(depth, mask, fx=100.0, fy=100.0, cx=2.0, cy=2.0)
    assert pts[2, 2].tolist() == [0.0, 0.0, 2.0]
    assert pts[2, 4, 0] == pytest.approx(2 * 2.0 / 100.0)
    assert pts[4, 2, 1] == pytest.approx(2 * 2.0 / 100.0)


def test_pointcloud_zeroes_rejected_pixels():
    depth = np.full((3, 3), 2.0)
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 1
    pts = ref_pointcloud(depth, mask, 100.0, 100.0, 1.0, 1.0)
    assert (pts[0, 0] == 0).all()
    assert pts[1, 1, 2] == 2.0


# --------------------------------------------------------------------------- #
# C++ parity (only where the extension is built)                               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CPP, reason="bcdl C++ module not built here")
def test_cpp_image_preprocess_matches_reference(scene):
    bgr, _ = scene
    cfg = _bcdl.DepthRefineConfig()
    got = np.asarray(_bcdl.preprocess_refine_image(np.ascontiguousarray(bgr), cfg))
    want = ref_preprocess_image(bgr, cfg.encoder_height, cfg.encoder_width)
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-4


@pytest.mark.skipif(not _HAS_CPP, reason="bcdl C++ module not built here")
def test_cpp_depth_preprocess_matches_reference(scene):
    _, depth = scene
    cfg = _bcdl.DepthRefineConfig()
    got = np.asarray(_bcdl.preprocess_refine_depth(np.ascontiguousarray(depth), cfg))
    want = ref_preprocess_depth(depth, cfg.encoder_height, cfg.encoder_width,
                                cfg.min_valid_depth)
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5


@pytest.mark.skipif(not _HAS_CPP, reason="bcdl C++ module not built here")
def test_cpp_decode_matches_reference():
    rng = np.random.default_rng(3)
    depth = rng.uniform(0.5, 8.0, (48, 64)).astype(np.float32)
    logits = rng.normal(0.0, 3.0, (48, 64)).astype(np.float32)
    cfg = _bcdl.DepthRefineConfig()
    r = _bcdl.decode_refined_depth(np.ascontiguousarray(depth),
                                   np.ascontiguousarray(logits), cfg)
    want = ref_decode(depth.astype(np.float64), logits.astype(np.float64),
                      cfg.mask_threshold, cfg.apply_mask)
    assert np.array_equal(np.asarray(r.mask), want["mask"])
    assert np.abs(np.asarray(r.depth) - want["depth"]).max() < 1e-5
    assert r.vmin == pytest.approx(want["vmin"], abs=1e-5)
    assert r.vmax == pytest.approx(want["vmax"], abs=1e-5)


@pytest.mark.skipif(not _HAS_CPP, reason="bcdl C++ module not built here")
def test_cpp_pointcloud_matches_reference():
    rng = np.random.default_rng(5)
    depth = rng.uniform(0.5, 8.0, (24, 32)).astype(np.float32)
    logits = np.full((24, 32), 5.0, np.float32)
    cfg = _bcdl.DepthRefineConfig()
    r = _bcdl.decode_refined_depth(np.ascontiguousarray(depth),
                                   np.ascontiguousarray(logits), cfg)
    k = _bcdl.Intrinsics(200.0, 210.0, 16.0, 12.0)
    got = np.asarray(_bcdl.depth_to_pointcloud(r, k))
    want = ref_pointcloud(np.asarray(r.depth).astype(np.float64),
                          np.asarray(r.mask), k.fx, k.fy, k.cx, k.cy)
    assert np.abs(got - want).max() < 1e-4


@pytest.mark.skipif(not _HAS_CPP, reason="bcdl C++ module not built here")
def test_cpp_scale_intrinsics():
    k = _bcdl.Intrinsics(460.0, 460.0, 320.0, 240.0)
    s = _bcdl.scale_intrinsics(k, 640, 480, 320, 240)
    assert (s.fx, s.fy, s.cx, s.cy) == (230.0, 230.0, 160.0, 120.0)

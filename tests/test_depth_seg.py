"""Dense-prediction post-processing tests (depth + segmentation).

These are PURE-NUMPY tests of the decode references (the oracles that the C++
``bcdl::decodeDepth`` / ``bcdl::decodeSeg`` mirror). They need only numpy and
run anywhere — no board, no .hbm model.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_depth_seg.py

The module-level ``ref_*`` functions double as the documented "numpy Python
path": given a raw model-output tensor + config they produce the same DepthMap /
SegMask the C++ tasks do. If the C++ bindings later expose ``decodeDepth`` /
``decodeSeg`` they are exercised opportunistically (guarded import), but the
core assertions never depend on them.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Reference: depth decode (mirrors src/tasks/depth.cc decodeDepth)            #
# --------------------------------------------------------------------------- #
def _resolve_hw(shape):
    """Last two non-unit dims are H,W (e.g. {1,H,W}, {1,1,H,W}, {H,W})."""
    dims = [d for d in shape if d > 1]
    if len(dims) >= 2:
        return dims[-2], dims[-1]
    n = len(shape)
    H = shape[n - 2] if n >= 2 else (shape[n - 1] if n >= 1 else 0)
    W = shape[n - 1] if n >= 1 else 0
    return H, W


def ref_decode_depth(data, shape, cfg):
    """Pure-numpy mirror of bcdl::decodeDepth. Returns dict
    {width,height,data(1d float),vmin,vmax}.

    cfg keys: width, height (0 => infer), normalize (bool), clip_lo, clip_hi.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    H, W = _resolve_hw(shape)
    if cfg.get("height", 0) > 0:
        H = cfg["height"]
    if cfg.get("width", 0) > 0:
        W = cfg["width"]
    n = H * W
    out = {"width": W, "height": H, "data": np.zeros(0), "vmin": 0.0, "vmax": 0.0}
    if n <= 0:
        return out

    vals = data[:n].astype(np.float64).copy()
    do_clip = cfg.get("clip_hi", 0.0) > cfg.get("clip_lo", 0.0)
    if do_clip:
        vals = np.clip(vals, cfg["clip_lo"], cfg["clip_hi"])
    vmin = float(vals.min())
    vmax = float(vals.max())
    out["vmin"], out["vmax"] = vmin, vmax

    if cfg.get("normalize", True):
        rng = vmax - vmin
        if rng > 0.0:
            vals = (vals - vmin) / rng
        else:
            vals = np.zeros_like(vals)
    out["data"] = vals
    return out


# --------------------------------------------------------------------------- #
# Reference: segmentation decode (mirrors src/tasks/segmentation.cc)          #
# --------------------------------------------------------------------------- #
def ref_decode_seg(data, shape, cfg):
    """Pure-numpy mirror of bcdl::decodeSeg. Returns dict
    {width,height,num_classes,labels(1d int)}.

    cfg keys: num_classes (0 => infer), channels_first (bool), argmaxed (bool).
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)

    if cfg.get("argmaxed", False):
        H, W = _resolve_hw(shape)
        n = H * W
        labels = np.rint(data[:n]).astype(np.int32)
        nc = cfg.get("num_classes", 0)
        if nc <= 0:
            nc = int(labels.max()) + 1 if n > 0 else 0
        return {"width": W, "height": H, "num_classes": nc, "labels": labels}

    dims = list(shape)
    cf = cfg.get("channels_first", True)
    C = H = W = 0
    if cf:
        if len(dims) >= 4:
            C, H, W = dims[-3], dims[-2], dims[-1]
        elif len(dims) == 3:
            C, H, W = dims[0], dims[1], dims[2]
    else:
        if len(dims) >= 4:
            H, W, C = dims[-3], dims[-2], dims[-1]
        elif len(dims) == 3:
            H, W, C = dims[0], dims[1], dims[2]
    if cfg.get("num_classes", 0) > 0:
        C = cfg["num_classes"]

    npix = H * W
    if npix <= 0 or C <= 0:
        return {"width": W, "height": H, "num_classes": C,
                "labels": np.zeros(npix, dtype=np.int32)}

    if cf:
        # data[c*npix + p]
        grid = data[:C * npix].reshape(C, npix)
        labels = np.argmax(grid, axis=0).astype(np.int32)
    else:
        # data[p*C + c]
        grid = data[:npix * C].reshape(npix, C)
        labels = np.argmax(grid, axis=1).astype(np.int32)
    return {"width": W, "height": H, "num_classes": C, "labels": labels}


# --------------------------------------------------------------------------- #
# Depth tests                                                                  #
# --------------------------------------------------------------------------- #
def test_depth_normalize_known_range():
    # 4x4 ramp 0..15 -> normalized 0..1, vmin=0 vmax=15.
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    out = ref_decode_depth(raw, [1, 4, 4], {"normalize": True})
    assert out["width"] == 4 and out["height"] == 4
    assert out["vmin"] == pytest.approx(0.0)
    assert out["vmax"] == pytest.approx(15.0)
    assert out["data"][0] == pytest.approx(0.0)
    assert out["data"][-1] == pytest.approx(1.0)
    assert out["data"][3] == pytest.approx(3.0 / 15.0)


def test_depth_no_normalize_copies_raw():
    raw = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = ref_decode_depth(raw, [2, 2], {"normalize": False})
    assert out["vmin"] == pytest.approx(10.0)
    assert out["vmax"] == pytest.approx(40.0)
    np.testing.assert_allclose(out["data"], [10, 20, 30, 40])


def test_depth_4d_shape_inference():
    raw = np.linspace(0, 1, 6, dtype=np.float32).reshape(1, 1, 2, 3)
    out = ref_decode_depth(raw, [1, 1, 2, 3], {"normalize": True})
    assert (out["height"], out["width"]) == (2, 3)
    assert out["data"][0] == pytest.approx(0.0)
    assert out["data"][-1] == pytest.approx(1.0)


def test_depth_clip_before_normalize():
    # values 0..15, clip to [4,8] -> after clip range vmin=4 vmax=8.
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    out = ref_decode_depth(raw, [1, 4, 4],
                           {"normalize": True, "clip_lo": 4.0, "clip_hi": 8.0})
    assert out["vmin"] == pytest.approx(4.0)
    assert out["vmax"] == pytest.approx(8.0)
    # value 0 -> clipped to 4 -> normalized 0; value 15 -> clipped to 8 -> 1.
    assert out["data"][0] == pytest.approx(0.0)
    assert out["data"][-1] == pytest.approx(1.0)
    # value 6 -> (6-4)/(8-4) = 0.5
    assert out["data"][6] == pytest.approx(0.5)


def test_depth_flat_map_guard():
    raw = np.full((1, 3, 3), 7.0, dtype=np.float32)
    out = ref_decode_depth(raw, [1, 3, 3], {"normalize": True})
    assert out["vmin"] == pytest.approx(7.0)
    assert out["vmax"] == pytest.approx(7.0)
    np.testing.assert_allclose(out["data"], np.zeros(9))  # no divide-by-zero


def test_depth_gray8_equivalent():
    # Mirror depthToGray8 on an already-normalized map: round(t*255).
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    out = ref_decode_depth(raw, [1, 4, 4], {"normalize": True})
    gray = np.rint(np.clip(out["data"], 0, 1) * 255).astype(np.uint8)
    assert gray[0] == 0
    assert gray[-1] == 255


# --------------------------------------------------------------------------- #
# Segmentation tests                                                           #
# --------------------------------------------------------------------------- #
def test_seg_argmax_channels_first():
    # [1,2,3,3]: channel-0 high in a known pattern, channel-1 high elsewhere.
    C, H, W = 2, 3, 3
    t = np.zeros((1, C, H, W), dtype=np.float32)
    # make a checkerboard: class 1 where (y+x) even, else class 0.
    for y in range(H):
        for x in range(W):
            cls = 1 if (y + x) % 2 == 0 else 0
            t[0, cls, y, x] = 5.0
    out = ref_decode_seg(t, [1, C, H, W], {"channels_first": True})
    assert (out["height"], out["width"], out["num_classes"]) == (3, 3, 2)
    expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).reshape(-1)
    np.testing.assert_array_equal(out["labels"], expected)


def test_seg_argmax_channels_last():
    C, H, W = 2, 3, 3
    t = np.zeros((1, H, W, C), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            cls = 1 if (y + x) % 2 == 0 else 0
            t[0, y, x, cls] = 5.0
    out = ref_decode_seg(t, [1, H, W, C], {"channels_first": False})
    assert (out["height"], out["width"], out["num_classes"]) == (3, 3, 2)
    expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).reshape(-1)
    np.testing.assert_array_equal(out["labels"], expected)


def test_seg_channels_first_matches_numpy_argmax():
    rng = np.random.default_rng(0)
    C, H, W = 5, 8, 6
    t = rng.standard_normal((1, C, H, W)).astype(np.float32)
    out = ref_decode_seg(t, [1, C, H, W], {"channels_first": True})
    expected = np.argmax(t[0], axis=0).reshape(-1)
    np.testing.assert_array_equal(out["labels"], expected)


def test_seg_channels_last_matches_numpy_argmax():
    rng = np.random.default_rng(1)
    C, H, W = 5, 8, 6
    t = rng.standard_normal((1, H, W, C)).astype(np.float32)
    out = ref_decode_seg(t, [1, H, W, C], {"channels_first": False})
    expected = np.argmax(t[0], axis=2).reshape(-1)
    np.testing.assert_array_equal(out["labels"], expected)


def test_seg_passthrough_ids():
    # Already-argmaxed float id map -> rounded ints.
    ids = np.array([[0, 3, 3], [2, 2, 1]], dtype=np.float32)
    out = ref_decode_seg(ids, [1, 2, 3], {"argmaxed": True})
    assert (out["height"], out["width"]) == (2, 3)
    np.testing.assert_array_equal(out["labels"], [0, 3, 3, 2, 2, 1])
    assert out["num_classes"] == 4  # max id + 1


def test_seg_passthrough_rounds_float():
    ids = np.array([0.0, 1.4, 1.6, 2.9], dtype=np.float32)
    out = ref_decode_seg(ids, [1, 4], {"argmaxed": True})
    np.testing.assert_array_equal(out["labels"], [0, 1, 2, 3])


def test_seg_num_classes_override():
    # 3-channel tensor but caller forces inference over all 3.
    C, H, W = 3, 2, 2
    t = np.zeros((1, C, H, W), dtype=np.float32)
    t[0, 2, 0, 0] = 9.0  # pixel (0,0) -> class 2
    out = ref_decode_seg(t, [1, C, H, W],
                         {"channels_first": True, "num_classes": 3})
    assert out["num_classes"] == 3
    assert out["labels"][0] == 2


# --------------------------------------------------------------------------- #
# Opportunistic checks against C++ bindings if exposed. Core tests never       #
# depend on these.                                                             #
# --------------------------------------------------------------------------- #
def test_cpp_bindings_if_available():
    try:
        import bcdl  # noqa: F401
        has = hasattr(bcdl, "decodeDepth") and hasattr(bcdl, "decodeSeg")
    except Exception:
        has = False
    if not has:
        pytest.skip("bcdl.decodeDepth/decodeSeg bindings not exposed")
    # If/when bound, a parity check would go here.

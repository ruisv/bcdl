"""On-board end-to-end RGB-D refinement test: DepthRefiner on a real .hbm.

The interesting part of this head is that it needs a depth map to refine, so the
test builds one the way the repo actually would: run the LAS2 stereo model on
the bundled rectified pair, convert disparity to metric depth, punch holes in it
the way a real sensor does, and hand that plus the left image to the refiner.
When the stereo model is absent it falls back to a synthetic depth map, which
still exercises the whole path (preprocess -> BPU -> decode -> point cloud).

Runs only on the board (needs the compiled bcdl extension, cv2, and the
refinement .hbm); skips cleanly when the model or images are absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_depth_refine_board_py.py

Models come from scripts/fetch_models.sh. Env-overridable:
BCDL_REFINE_MODEL / BCDL_STEREO_MODEL / BCDL_IMAGES.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_IMAGES = os.environ.get("BCDL_IMAGES", os.path.join(_REPO, "data", "images"))
_IN_W, _IN_H = 640, 480


def _refine_model():
    env = os.environ.get("BCDL_REFINE_MODEL")
    if env:
        return env if os.path.exists(env) else None
    p = os.path.join(_MODELS, "lingbot_depth_v05_int16_nashm.hbm")
    return p if os.path.exists(p) else None


def _left_image():
    p = os.path.join(_IMAGES, "stereo_left.png")
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    # Centre-crop to the model's aspect, then resize — same convention the
    # stereo test uses so the two tests see the same framing.
    y0, x0 = max(0, (h - _IN_H) // 2), max(0, (w - _IN_W) // 2)
    img = img[y0:y0 + _IN_H, x0:x0 + _IN_W]
    if img.shape[:2] != (_IN_H, _IN_W):
        img = cv2.resize(img, (_IN_W, _IN_H))
    return np.ascontiguousarray(img)


def _sensor_like_depth(bgr):
    """A depth map with the defects a real sensor produces: holes and noise.

    Uses the LAS2 stereo model when it is present (the realistic pairing), else a
    smooth synthetic ramp. Either way the holes are what the refiner must fill.
    """
    depth = None
    stereo = os.environ.get("BCDL_STEREO_MODEL") or os.path.join(
        _MODELS, "las2_m_crop_nashm.hbm")
    right = os.path.join(_IMAGES, "stereo_right.png")
    if os.path.exists(stereo) and os.path.exists(right):
        try:
            eng = bcdl.Engine(stereo)
            cfg = bcdl.StereoConfig()
            cfg.fit = bcdl.StereoFit.Crop
            cfg.fx, cfg.baseline = 700.0, 0.12
            pipe = bcdl.StereoPipeline(eng, cfg)
            rimg = cv2.imread(right, cv2.IMREAD_COLOR)
            h, w = rimg.shape[:2]
            y0, x0 = max(0, (h - _IN_H) // 2), max(0, (w - _IN_W) // 2)
            rimg = np.ascontiguousarray(rimg[y0:y0 + _IN_H, x0:x0 + _IN_W])
            if rimg.shape[:2] != (_IN_H, _IN_W):
                rimg = cv2.resize(rimg, (_IN_W, _IN_H))
            res = pipe.process(bgr, rimg)
            depth = np.asarray(res.depth, np.float32)
        except Exception:
            depth = None
    if depth is None or depth.shape != (_IN_H, _IN_W):
        yy, xx = np.mgrid[0:_IN_H, 0:_IN_W].astype(np.float32)
        depth = 1.0 + 3.0 * (yy / _IN_H) + 0.5 * np.sin(xx / 40.0)
    depth = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 20.0)
    depth[120:200, 260:380] = 0.0          # a dropout, as a sensor leaves
    depth[::37, :] = 0.0                   # scattered missing rows
    return np.ascontiguousarray(depth.astype(np.float32))


@pytest.fixture(scope="module")
def refiner():
    model = _refine_model()
    if model is None:
        pytest.skip("lingbot_depth refinement .hbm not present (scripts/fetch_models.sh)")
    engine = bcdl.Engine(model)
    return bcdl.DepthRefiner(engine)


@pytest.fixture(scope="module")
def frame():
    bgr = _left_image()
    if bgr is None:
        pytest.skip("data/images/stereo_left.png missing")
    return bgr, _sensor_like_depth(bgr)


def test_refined_depth_is_dense_and_metric(refiner, frame):
    bgr, depth = frame
    r = refiner.run(bgr, depth)
    out = np.asarray(r.depth)
    mask = np.asarray(r.mask)

    assert (r.height, r.width) == out.shape
    assert np.isfinite(out).all(), "refined depth must not contain NaN/Inf"
    kept = mask > 0
    assert kept.mean() > 0.5, f"only {kept.mean():.1%} of pixels trusted"
    vals = out[kept]
    assert (vals > 0).all(), "trusted depth must be positive"
    assert 0.05 < vals.min() < 100.0 and vals.max() < 200.0, \
        f"implausible metric range [{vals.min():.2f}, {vals.max():.2f}] m"
    assert r.vmin == pytest.approx(float(vals.min()), rel=1e-5)
    assert r.vmax == pytest.approx(float(vals.max()), rel=1e-5)


def test_holes_in_the_input_get_filled(refiner, frame):
    """The point of the model: pixels with no reading come back with one."""
    bgr, depth = frame
    r = refiner.run(bgr, depth)
    out = np.asarray(r.depth)
    mask = np.asarray(r.mask)
    hole = np.zeros(depth.shape, bool)
    hole[120:200, 260:380] = True
    filled = hole & (mask > 0) & (out > 0)
    assert filled.mean() > 0.5 * hole.mean(), \
        "the injected dropout came back mostly empty"


def test_refinement_tracks_the_input_where_the_input_is_valid(refiner, frame):
    """Refinement is not free invention: where the sensor had a reading, the
    output should stay close to it rather than drift to an unrelated scale."""
    bgr, depth = frame
    r = refiner.run(bgr, depth)
    out = np.asarray(r.depth)
    valid = (depth > 0.05) & (np.asarray(r.mask) > 0) & (out > 0)
    assert valid.sum() > 1000
    ratio = np.median(out[valid] / depth[valid])
    assert 0.5 < ratio < 2.0, f"median scale drift {ratio:.3f}"


def test_pointcloud_geometry(refiner, frame):
    bgr, depth = frame
    r = refiner.run(bgr, depth)
    k = bcdl.Intrinsics(700.0, 700.0, r.width / 2.0, r.height / 2.0)
    pts = np.asarray(bcdl.depth_to_pointcloud(r, k))
    assert pts.shape == (r.height, r.width, 3)
    assert np.isfinite(pts).all()
    z = pts[..., 2]
    kept = np.asarray(r.mask) > 0
    assert np.allclose(z[kept], np.asarray(r.depth)[kept])
    # The principal point sits on the optical axis.
    cy, cx = r.height // 2, r.width // 2
    if kept[cy, cx]:
        assert abs(pts[cy, cx, 0]) < 1e-3 and abs(pts[cy, cx, 1]) < 1e-3


def test_cpp_preprocessing_matches_the_python_reference(frame):
    """Guards the one thing the board can get wrong on its own: the host-side
    resize/normalize/log that the .hbm was calibrated against."""
    bgr, depth = frame
    cfg = bcdl.DepthRefineConfig()
    img = np.asarray(bcdl.preprocess_refine_image(bgr, cfg))
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                       (cfg.encoder_width, cfg.encoder_height),
                       interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    want = ((small.astype(np.float32).transpose(2, 0, 1) / 255.0) - mean) / std
    # cv2 rounds its resized output to uint8; the C++ keeps the average in float.
    # The whole gap is that half-level, carried through the normalization:
    # 0.5/255/min(std) = 0.5/255/0.224 = 8.8e-3.
    assert np.abs(img[0] - want).max() < 9e-3

    dep = np.asarray(bcdl.preprocess_refine_depth(depth, cfg))
    d = cv2.resize(depth, (cfg.encoder_width, cfg.encoder_height),
                   interpolation=cv2.INTER_NEAREST)
    want_d = np.where(d > cfg.min_valid_depth, np.log(np.maximum(d, 1e-6)), 0.0)
    assert np.abs(dep[0, 0] - want_d).max() < 1e-5

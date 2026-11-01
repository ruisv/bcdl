"""On-board end-to-end stereo test: StereoPipeline on a real LAS2 .hbm.

Drives the BPU with the repo-local rectified stereo pair (data/images/
stereo_{left,right}.png) and asserts sane disparity + depth + validity, plus a
self-contained cross-check that the pipeline's C++ preproc matches a cv2
reference (BGR->RGB + fit + planar F32). Runs only on the board (needs the
compiled bcdl extension + cv2 + a LAS2 .hbm); skips cleanly when the model or
images are absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_stereo_board_py.py

Models come from scripts/fetch_models.sh (las2_m_crop2_nashm.hbm preferred — its
center-crop calibration matches this pair; las2_m_resize2_nashm.hbm = resize).
test_full_disparity_range runs against every LAS2 build present (M and L).
Env-overridable: BCDL_STEREO_MODEL / BCDL_STEREO_MODE / BCDL_STEREO_LEFT/RIGHT.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_IMAGES = os.environ.get("BCDL_IMAGES", os.path.join(_REPO, "data", "images"))
_LEFT = os.environ.get("BCDL_STEREO_LEFT", os.path.join(_IMAGES, "stereo_left.png"))
_RIGHT = os.environ.get("BCDL_STEREO_RIGHT", os.path.join(_IMAGES, "stereo_right.png"))
_IN_W, _IN_H = 640, 480

# Every LAS2 build fetch_models.sh ships, with the letterbox mode it was
# calibrated for. The first two (M) are the defaults for the single-model tests.
_BUILDS = (("las2_m_crop2_nashm.hbm", bcdl.StereoFit.Crop),
           ("las2_m_resize2_nashm.hbm", bcdl.StereoFit.Resize),
           ("las2_l_crop2_nashm.hbm", bcdl.StereoFit.Crop),
           ("las2_l_resize2_nashm.hbm", bcdl.StereoFit.Resize))


def _resolve_model():
    """Pick (model_path, StereoFit) — crop-calibrated preferred, else resize.
    Honors BCDL_STEREO_MODEL (+ BCDL_STEREO_MODE = crop|resize)."""
    env = os.environ.get("BCDL_STEREO_MODEL")
    if env:
        mode = os.environ.get("BCDL_STEREO_MODE", "crop")
        fit = bcdl.StereoFit.Crop if mode == "crop" else bcdl.StereoFit.Resize
        return (env, fit) if os.path.exists(env) else (None, None)
    for name, fit in _BUILDS[:2]:
        p = os.path.join(_MODELS, name)
        if os.path.exists(p):
            return p, fit
    return None, None


def _load_pair():
    if not (os.path.exists(_LEFT) and os.path.exists(_RIGHT)):
        pytest.skip("stereo test images not present")
    return cv2.imread(_LEFT, cv2.IMREAD_COLOR), cv2.imread(_RIGHT, cv2.IMREAD_COLOR)


def _ref_input(img_bgr, fit):
    """cv2 reference preproc: fit (crop/resize) + BGR->RGB + planar F32 (3,H,W)."""
    if fit == bcdl.StereoFit.Crop:
        h, w = img_bgr.shape[:2]
        t, l = (h - _IN_H) // 2, (w - _IN_W) // 2
        im = img_bgr[t:t + _IN_H, l:l + _IN_W]
    else:
        im = cv2.resize(img_bgr, (_IN_W, _IN_H))
    rgb = im[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(rgb)


@pytest.fixture(scope="module")
def setup():
    model, fit = _resolve_model()
    if model is None:
        pytest.skip("no LAS2 .hbm on this board")
    left, right = _load_pair()
    return model, fit, left, right


def test_disparity_sane(setup):
    model, fit, left, right = setup
    eng = bcdl.Engine(model)
    cfg = bcdl.StereoConfig()
    cfg.fit = fit
    disp = np.asarray(bcdl.StereoPipeline(eng, cfg).process(left, right).disparity.data)
    assert disp.shape == (_IN_H, _IN_W)
    assert np.isfinite(disp).all()
    assert disp.min() >= 0.0                 # disparity is non-negative
    assert disp.max() > 5.0                  # a real scene has meaningful disparity
    # plausible band: the indoor pair sits well inside LAS2's [0, ~192] range
    assert 5.0 < float(disp.mean()) < 160.0


def test_preproc_matches_cv2_reference(setup):
    """The pipeline's C++ preproc (fit + BGR->RGB + planar pack) must reproduce
    the cv2 reference exactly — so the disparity is identical to feeding the
    reference tensors through the same Engine."""
    model, fit, left, right = setup
    eng = bcdl.Engine(model)

    # reference path: cv2 preproc -> Engine.infer
    outs = eng.infer([_ref_input(left, fit), _ref_input(right, fit)])
    disp_ref = np.asarray(outs[0]).reshape(_IN_H, _IN_W)

    # pipeline path
    cfg = bcdl.StereoConfig()
    cfg.fit = fit
    disp_pipe = np.asarray(bcdl.StereoPipeline(eng, cfg).process(left, right).disparity.data)

    a, b = disp_ref.ravel(), disp_pipe.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999
    assert np.abs(a - b).max() < 1e-2


def test_depth_and_valid_mask(setup):
    model, fit, left, right = setup
    eng = bcdl.Engine(model)
    cfg = bcdl.StereoConfig()
    cfg.fit = fit
    cfg.fx = 700.0          # plausible focal length at 640-wide input
    cfg.baseline = 0.12     # 12 cm baseline
    cfg.valid_mask = True
    cfg.disp_min = 2.0      # drop far/sub-pixel
    res = bcdl.StereoPipeline(eng, cfg).process(left, right)

    depth = np.asarray(res.depth)
    valid = np.asarray(res.valid)
    assert depth.shape == (_IN_H, _IN_W)
    assert valid.shape == (_IN_H, _IN_W)
    assert valid.dtype == np.uint8

    # depth is positive & finite wherever the mask says the pixel is trustworthy
    vd = depth[valid.astype(bool)]
    assert vd.size > 0
    assert np.isfinite(vd).all() and (vd > 0).all()

    # left-border geometry: a pixel at column x needs disparity <= x to match the
    # right image, so the very-left columns can never be valid.
    disp = np.asarray(res.disparity.data)
    xx = np.arange(_IN_W)[None, :]
    assert valid[xx < disp].sum() == 0      # x - disp < 0 => always masked out

    # the mask drops *some* pixels (occlusions/range) but keeps a usable fraction
    cov = float(valid.mean())
    assert 0.2 < cov < 1.0


def _shift_left(img, k):
    """Move an image k px left, replicating the last column. Applied to the
    right view it adds exactly k px to every disparity."""
    if k <= 0:
        return img
    out = np.empty_like(img)
    out[:, :-k] = img[:, k:]
    out[:, -k:] = img[:, -1:]
    return out


def test_full_disparity_range():
    """The output must reach the top of LAS2's 0..192 px range.

    int16 output thresholds come from the calibration data, so a calibration
    set with only near-field-free scenes silently caps the disparity: the
    earlier builds clipped at 131 px (crop) and 47.7 px (resize), and the
    repo pair (44..129 px) sits just under the first cap, so nothing else here
    noticed. Shifting the right view left by k adds k to every disparity; push
    the pair up to ~175 px and check the model follows.
    """
    builds = [(os.path.join(_MODELS, n), f) for n, f in _BUILDS
              if os.path.exists(os.path.join(_MODELS, n))]
    if not builds:
        pytest.skip("no LAS2 .hbm on this board")
    left, right = _load_pair()
    for model, fit in builds:
        eng = bcdl.Engine(model)
        cfg = bcdl.StereoConfig()
        cfg.fit = fit
        pipe = bcdl.StereoPipeline(eng, cfg)
        d0 = np.asarray(pipe.process(left, right).disparity.data)

        scale = 1.0 if fit == bcdl.StereoFit.Crop else _IN_W / left.shape[1]
        k_src = int(round((175.0 - np.percentile(d0, 99)) / scale))
        k_in = k_src * scale
        d1 = np.asarray(pipe.process(left, _shift_left(right, k_src)).disparity.data)

        name = os.path.basename(model)
        p99 = float(np.percentile(d1, 99))
        assert p99 > 160.0, f"{name}: disparity capped at {p99:.1f} px"
        # interior columns only: the left edge loses its match, the right edge
        # of the shifted view is replicated padding
        x0, x1 = 200, _IN_W - int(np.ceil(k_in)) - 8
        err = np.median(np.abs(d1[:, x0:x1] - d0[:, x0:x1] - k_in))
        assert err < 2.0, f"{name}: shifted disparity off by {err:.2f} px (k={k_in:.1f})"

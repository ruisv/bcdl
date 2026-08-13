"""On-board optical flow test: OpticalFlowEstimator on a real .hbm.

Flow is one of the few tasks where ground truth can be manufactured exactly:
shift an image by a known number of pixels and the correct answer is that
constant field everywhere. That gives a real error number without a dataset, and
it also exposes the failure mode quantisation actually causes here — a noise
floor that is invisible on fast pixels and dominates slow ones.

The identical-frames case is the same idea at zero displacement: whatever the
model reports there is pure noise, so it measures the floor directly.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_optical_flow_board_py.py

Model from scripts/fetch_models.sh. Env-overridable: BCDL_FLOW_MODEL / BCDL_IMAGES.
"""
import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_IMAGES = os.environ.get("BCDL_IMAGES", os.path.join(_REPO, "data", "images"))


def _model():
    p = os.environ.get("BCDL_FLOW_MODEL") or os.path.join(
        _MODELS, "neuflow_v2_int16_nashm_512x384.hbm")
    return p if os.path.exists(p) else None


pytestmark = pytest.mark.skipif(_model() is None,
                                reason="flow .hbm missing (scripts/fetch_models.sh)")


@pytest.fixture(scope="module")
def rig():
    engine = bcdl.Engine(_model())
    est = bcdl.OpticalFlowEstimator(engine)
    w, h = est.input_size
    img = None
    for name in ("bus.jpg", "zidane.jpg"):
        p = os.path.join(_IMAGES, name)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            break
    if img is None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return est, cv2.resize(img, (w, h))


def _interior(field, margin=48):
    """Drop the border: a shifted frame has no correspondence there, so the
    model is being asked an unanswerable question in that band."""
    d = field.data
    return d[margin:-margin, margin:-margin]


def test_shapes_and_types(rig):
    est, img = rig
    f = est.estimate(img, img)
    assert (f.width, f.height) == (img.shape[1], img.shape[0])
    assert f.data.shape == (img.shape[0], img.shape[1], 2)
    assert np.isfinite(f.data).all()


def test_identical_frames_measure_the_noise_floor(rig):
    est, img = rig
    d = _interior(est.estimate(img, img))
    mag = np.linalg.norm(d, axis=-1)
    # Zero displacement is the answer; whatever comes back is the floor. Two
    # pixels is loose on purpose — this asserts "not broken", and the actual
    # value is what the docs quote.
    assert mag.mean() < 2.0, f"static-pair noise floor {mag.mean():.2f} px"


@pytest.mark.parametrize("shift", [(6, 0), (0, 4), (-8, 3)])
def test_recovers_a_known_translation(rig, shift):
    est, img = rig
    dx, dy = shift
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    d = _interior(est.estimate(img, moved))
    # Flow maps frame0 pixels onto frame1, so shifting the content by (dx, dy)
    # is what the field must report.
    err = np.linalg.norm(d - np.array([dx, dy], np.float32), axis=-1)
    assert np.median(err) < 2.5, (
        f"median EPE {np.median(err):.2f} px on a known {shift} shift; "
        f"reported median flow {np.median(d.reshape(-1, 2), axis=0)}")


def test_colorize_runs_on_a_real_field(rig):
    est, img = rig
    f = est.estimate(img, img)
    vis = bcdl.flow_colorize(f)
    assert vis.shape == (img.shape[0], img.shape[1], 3)
    assert vis.dtype == np.uint8


def test_deterministic(rig):
    est, img = rig
    m = np.float32([[1, 0, 5], [0, 1, 0]])
    moved = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    a = est.estimate(img, moved).data
    b = est.estimate(img, moved).data
    np.testing.assert_array_equal(a, b)

"""On-board anomaly detection test: AnomalyDetector on a real .hbm.

The shipped model is trained on ONE product's normal images, so a generic test
photo is meaningless to it — there is no "correct" answer for a bus. What can be
asserted without the product's dataset is the contract: the map covers the input
resolution, the score is the map's maximum, the smoothing behaves, and the same
image twice gives the same answer.

The accuracy claim (image / pixel AUROC against the official test split) lives in
docs/MODELS.md and is measured on the product's own data, which is where a
one-class model can be scored at all.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_anomaly_board_py.py

Model from scripts/fetch_models.sh. Env-overridable: BCDL_ANOMALY_MODEL / BCDL_IMAGES.
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
    p = os.environ.get("BCDL_ANOMALY_MODEL") or os.path.join(
        _MODELS, "glass_bottle_int8_nashm_288x288.hbm")
    return p if os.path.exists(p) else None


pytestmark = pytest.mark.skipif(_model() is None,
                                reason="anomaly .hbm missing (scripts/fetch_models.sh)")


@pytest.fixture(scope="module")
def rig():
    det = bcdl.AnomalyDetector(bcdl.Engine(_model()))
    w, h = det.input_size
    p = os.path.join(_IMAGES, "bus.jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR) if os.path.exists(p) else None
    if img is None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return det, img


def test_map_covers_the_input_resolution(rig):
    det, img = rig
    m = det.detect(img)
    w, h = det.input_size
    # Not the head's coarse grid: a caller has to be able to overlay this.
    assert (m.width, m.height) == (w, h)
    assert m.data.shape == (h, w)
    assert np.isfinite(m.data).all()


def test_score_is_the_map_maximum(rig):
    det, img = rig
    m = det.detect(img)
    assert m.score == pytest.approx(float(m.data.max()), abs=1e-6)


def test_mask_follows_the_threshold(rig):
    det, img = rig
    m = det.detect(img)
    lo = int((bcdl.anomaly_mask(m, float(m.data.min()) - 1e-3) > 0).sum())
    hi = int((bcdl.anomaly_mask(m, float(m.data.max()) + 1e-3) > 0).sum())
    assert lo == m.data.size
    assert hi == 0


def test_disabling_the_blur_sharpens_the_map(rig):
    det, img = rig
    smooth = det.detect(img)
    cfg = bcdl.AnomalyConfig()
    cfg.blur_kernel = 0
    raw = bcdl.AnomalyDetector(det.engine, cfg).detect(img)
    # Smoothing can only reduce the extreme, never raise it.
    assert raw.score >= smooth.score - 1e-6


def test_deterministic(rig):
    det, img = rig
    np.testing.assert_array_equal(det.detect(img).data, det.detect(img).data)

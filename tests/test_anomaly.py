"""Industrial anomaly detection: numpy path, no model required.

    PYTHONPATH=build:python pytest -s tests/test_anomaly.py

The post-processing here is not cosmetic — the reference implementation upsamples
the coarse patch grid and Gaussian-smooths it before anything is read off, and
both steps move pixel-level agreement. These tests pin that the smoothing is a
real normalised Gaussian with replicated borders (an unnormalised kernel would
scale every score, and a zero border would fake a clean frame edge), and that the
image-level score stays a maximum rather than drifting toward a mean.
"""
import numpy as np
import pytest

import bcdl


def _grid(h=4, w=4, value=0.0):
    return np.full((1, h, w), value, np.float32)


def _cfg(**kw):
    c = bcdl.AnomalyConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_raw_grid_passthrough():
    g = _grid()
    g[0, 1, 2] = 0.9
    m = bcdl.decode_anomaly(g, _cfg(blur_kernel=0))
    assert (m.height, m.width) == (4, 4)
    assert m.data[1, 2] == pytest.approx(0.9)
    assert m.score == pytest.approx(0.9)


def test_upsamples_to_the_requested_size():
    m = bcdl.decode_anomaly(_grid(4, 4, 0.25), _cfg(out_width=32, out_height=16,
                                                    blur_kernel=0))
    assert (m.height, m.width) == (16, 32)
    # A constant grid must stay constant through the resize.
    np.testing.assert_allclose(m.data, 0.25, atol=1e-6)


def test_blur_preserves_a_constant_field():
    """A normalised kernel with replicated borders leaves a flat field alone —
    including at the edges, where a zero-padded blur would darken it."""
    m = bcdl.decode_anomaly(_grid(16, 16, 0.7), _cfg(blur_kernel=33, blur_sigma=4.0))
    np.testing.assert_allclose(m.data, 0.7, atol=1e-5)
    assert m.score == pytest.approx(0.7, abs=1e-5)


def test_blur_spreads_and_lowers_a_spike():
    g = _grid(32, 32)
    g[0, 16, 16] = 1.0
    sharp = bcdl.decode_anomaly(g, _cfg(blur_kernel=0))
    blurred = bcdl.decode_anomaly(g, _cfg(blur_kernel=9, blur_sigma=2.0))
    assert sharp.score == pytest.approx(1.0)
    assert blurred.score < 0.2                     # energy spread over the kernel
    assert blurred.data[16, 17] > 0                # ... into the neighbourhood
    assert blurred.data.sum() == pytest.approx(sharp.data.sum(), rel=1e-3)


def test_score_is_the_max_not_the_mean():
    g = _grid(16, 16, 0.1)
    g[0, 3, 3] = 0.95                              # one small defect
    m = bcdl.decode_anomaly(g, _cfg(blur_kernel=0))
    assert m.score == pytest.approx(0.95)
    assert m.data.mean() < 0.2                     # the mean would have hidden it


def test_mask_thresholds():
    g = _grid(8, 8, 0.2)
    g[0, 2:4, 2:4] = 0.8
    m = bcdl.decode_anomaly(g, _cfg(blur_kernel=0))
    mask = bcdl.anomaly_mask(m, 0.5)
    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    assert int((mask > 0).sum()) == 4
    assert int((bcdl.anomaly_mask(m, 0.9) > 0).sum()) == 0


def test_decode_rejects_multichannel():
    with pytest.raises(Exception):
        bcdl.decode_anomaly(np.zeros((1, 3, 8, 8), np.float32))


def test_preprocess_normalisation_matches_the_reference():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)
    out = bcdl.anomaly_preprocess(img, 12, 12)
    assert out.shape == (3, 12, 12)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rgb = img[..., ::-1].astype(np.float32) / 255.0     # BGR in, RGB expected out
    expected = ((rgb - mean) / std).transpose(2, 0, 1)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_preprocess_resizes():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)
    out = bcdl.anomaly_preprocess(img, 16, 8)
    assert out.shape == (3, 8, 16)
    assert np.isfinite(out).all()

"""Dense optical flow: numpy path, no model required.

    PYTHONPATH=build:python pytest -s tests/test_optical_flow.py

The properties worth pinning here are the ones a running model cannot reveal:
that (u, v) stay paired with the right pixel through the de-planarization, that
the model-to-source pixel scale is actually applied, and that preprocessing does
not quietly convert BGR or pre-normalise — each of those produces a field that
looks entirely reasonable and is wrong.
"""
import numpy as np
import pytest

import bcdl


def _planar(h=3, w=4):
    """A tensor whose u and v planes are distinguishable per pixel."""
    u = np.arange(h * w, dtype=np.float32).reshape(h, w)
    v = -np.arange(h * w, dtype=np.float32).reshape(h, w)
    return np.stack([u, v])[None]          # (1, 2, H, W)


def test_decode_keeps_uv_paired_with_their_pixel():
    t = _planar()
    f = bcdl.decode_flow(t)
    assert (f.height, f.width) == (3, 4)
    d = f.data
    assert d.shape == (3, 4, 2)
    np.testing.assert_allclose(d[..., 0], t[0, 0])
    np.testing.assert_allclose(d[..., 1], t[0, 1])
    # A transposed de-planarization would still be self-consistent here, so pin
    # one off-diagonal pixel explicitly.
    assert d[1, 2, 0] == pytest.approx(6.0)
    assert d[1, 2, 1] == pytest.approx(-6.0)


def test_decode_applies_the_pixel_scale():
    cfg = bcdl.FlowConfig()
    cfg.scale_x, cfg.scale_y = 2.0, 3.0
    f = bcdl.decode_flow(_planar(), cfg)
    assert f.data[1, 2, 0] == pytest.approx(12.0)
    assert f.data[1, 2, 1] == pytest.approx(-18.0)


def test_decode_channels_last():
    t = np.moveaxis(_planar()[0], 0, -1)[None]     # (1, H, W, 2)
    cfg = bcdl.FlowConfig()
    cfg.channels_first = False
    f = bcdl.decode_flow(np.ascontiguousarray(t), cfg)
    assert f.data[1, 2, 0] == pytest.approx(6.0)
    assert f.data[1, 2, 1] == pytest.approx(-6.0)


@pytest.mark.parametrize("bad", [np.zeros((1, 3, 4, 4), np.float32),
                                 np.zeros((4, 4), np.float32)])
def test_decode_rejects_non_flow_tensors(bad):
    with pytest.raises(Exception):
        bcdl.decode_flow(bad)


def test_endpoint_error():
    a = bcdl.decode_flow(_planar())
    b = bcdl.decode_flow(_planar() + 3.0)
    # every vector moved by (3, 3) -> EPE = sqrt(18)
    assert bcdl.flow_endpoint_error(a, b) == pytest.approx(np.sqrt(18.0), rel=1e-5)
    assert bcdl.flow_endpoint_error(a, a) == pytest.approx(0.0)


def test_endpoint_error_rejects_size_mismatch():
    with pytest.raises(Exception):
        bcdl.flow_endpoint_error(bcdl.decode_flow(_planar(3, 4)),
                                 bcdl.decode_flow(_planar(4, 4)))


def test_colorize_shape_and_direction():
    h, w = 8, 8
    zero = bcdl.decode_flow(np.zeros((1, 2, h, w), np.float32))
    img = bcdl.flow_colorize(zero, 1.0)
    assert img.shape == (h, w, 3)
    # No motion sits at the centre of the wheel, which is white.
    assert img.min() > 240

    def solid(u, v):
        t = np.zeros((1, 2, h, w), np.float32)
        t[0, 0] = u
        t[0, 1] = v
        return bcdl.flow_colorize(bcdl.decode_flow(t), 1.0)

    right, left, down = solid(1, 0), solid(-1, 0), solid(0, 1)
    assert not np.array_equal(right, left)
    assert not np.array_equal(right, down)
    # A uniform field colours every pixel the same (per channel — the three
    # channels of that one colour naturally differ from each other).
    assert right.reshape(-1, 3).std(axis=0).max() == 0


def test_preprocess_keeps_bgr_and_0_255():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 10      # B
    img[..., 1] = 20      # G
    img[..., 2] = 30      # R
    out = bcdl.flow_preprocess(img, 6, 4)
    assert out.shape == (3, 4, 6)
    # Channel order untouched, range untouched: plane 0 is still blue at 10.
    np.testing.assert_allclose(out[0], 10.0)
    np.testing.assert_allclose(out[1], 20.0)
    np.testing.assert_allclose(out[2], 30.0)


def test_preprocess_resizes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (16, 32, 3), dtype=np.uint8)
    out = bcdl.flow_preprocess(img, 16, 8)
    assert out.shape == (3, 8, 16)
    assert 0.0 <= out.min() and out.max() <= 255.0
    # Same size in, same pixels out.
    same = bcdl.flow_preprocess(img, 32, 16)
    np.testing.assert_allclose(same, img.astype(np.float32).transpose(2, 0, 1))

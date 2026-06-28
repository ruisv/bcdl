"""Depth + segmentation binding tests (C++ decode via nanobind).

These exercise the compiled ``bcdl`` extension's decode_depth / decode_seg /
colorize bindings, so they only run where the extension imports (on the board,
inside the `bcdl` conda env). They need NO .hbm model — the decode path is pure
CPU. Guarded so the file skips cleanly off-board.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_depth_seg_py.py

The numpy oracles these mirror live in tests/test_depth_seg.py.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have(*names):
    return all(hasattr(bcdl, n) for n in names)


# --------------------------------------------------------------------------- #
# Depth                                                                        #
# --------------------------------------------------------------------------- #
def test_decode_depth_normalize():
    if not _have("decode_depth", "DepthConfig"):
        pytest.skip("decode_depth not exposed")
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    cfg = bcdl.DepthConfig()
    cfg.normalize = True
    dm = bcdl.decode_depth(raw, cfg)
    assert (dm.height, dm.width) == (4, 4)
    assert dm.vmin == pytest.approx(0.0)
    assert dm.vmax == pytest.approx(15.0)
    data = dm.data
    assert data.shape == (4, 4)
    assert data.dtype == np.float32
    assert data.flat[0] == pytest.approx(0.0)
    assert data.flat[-1] == pytest.approx(1.0)


def test_decode_depth_clip_before_normalize():
    if not _have("decode_depth", "DepthConfig"):
        pytest.skip("decode_depth not exposed")
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    cfg = bcdl.DepthConfig()
    cfg.normalize = True
    cfg.clip_lo = 4.0
    cfg.clip_hi = 8.0
    dm = bcdl.decode_depth(raw, cfg)
    assert dm.vmin == pytest.approx(4.0)
    assert dm.vmax == pytest.approx(8.0)
    assert dm.data.flat[6] == pytest.approx(0.5)


def test_depth_colorize_and_gray8():
    if not _have("decode_depth", "depth_colorize", "depth_to_gray8", "DepthConfig"):
        pytest.skip("depth colorize bindings not exposed")
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    cfg = bcdl.DepthConfig()
    dm = bcdl.decode_depth(raw, cfg)
    bgr = bcdl.depth_colorize(dm)
    assert bgr.shape == (4, 4, 3)
    assert bgr.dtype == np.uint8
    gray = bcdl.depth_to_gray8(dm)
    assert gray.shape == (4, 4)
    assert gray.dtype == np.uint8
    assert gray.flat[0] == 0
    assert gray.flat[-1] == 255


# --------------------------------------------------------------------------- #
# Segmentation                                                                 #
# --------------------------------------------------------------------------- #
def test_decode_seg_channels_first():
    if not _have("decode_seg", "SegConfig"):
        pytest.skip("decode_seg not exposed")
    rng = np.random.default_rng(0)
    C, H, W = 5, 8, 6
    t = rng.standard_normal((1, C, H, W)).astype(np.float32)
    cfg = bcdl.SegConfig()
    cfg.channels_first = True
    sm = bcdl.decode_seg(t, cfg)
    assert (sm.height, sm.width, sm.num_classes) == (H, W, C)
    labels = sm.labels
    assert labels.shape == (H, W)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, np.argmax(t[0], axis=0))


def test_decode_seg_channels_last():
    if not _have("decode_seg", "SegConfig"):
        pytest.skip("decode_seg not exposed")
    rng = np.random.default_rng(1)
    C, H, W = 5, 8, 6
    t = rng.standard_normal((1, H, W, C)).astype(np.float32)
    cfg = bcdl.SegConfig()
    cfg.channels_first = False
    sm = bcdl.decode_seg(t, cfg)
    np.testing.assert_array_equal(sm.labels, np.argmax(t[0], axis=2))


def test_decode_seg_passthrough_ids():
    if not _have("decode_seg", "SegConfig"):
        pytest.skip("decode_seg not exposed")
    ids = np.array([[0, 3, 3], [2, 2, 1]], dtype=np.float32).reshape(1, 2, 3)
    cfg = bcdl.SegConfig()
    cfg.argmaxed = True
    sm = bcdl.decode_seg(ids, cfg)
    assert (sm.height, sm.width) == (2, 3)
    np.testing.assert_array_equal(sm.labels.reshape(-1), [0, 3, 3, 2, 2, 1])
    assert sm.num_classes == 4


def test_seg_colorize():
    if not _have("decode_seg", "seg_colorize", "SegConfig"):
        pytest.skip("seg_colorize not exposed")
    C, H, W = 3, 4, 4
    t = np.zeros((1, C, H, W), dtype=np.float32)
    t[0, 2, 0, 0] = 9.0
    cfg = bcdl.SegConfig()
    cfg.channels_first = True
    sm = bcdl.decode_seg(t, cfg)
    bgr = bcdl.seg_colorize(sm)
    assert bgr.shape == (H, W, 3)
    assert bgr.dtype == np.uint8

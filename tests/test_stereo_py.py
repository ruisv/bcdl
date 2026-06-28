"""Stereo pipeline binding tests (C++ preproc/decode via nanobind).

These exercise the compiled ``bcdl`` extension's stereo helpers — input packing
(BGR->RGB + fit), disparity->depth, and the validity mask — against numpy
oracles. They need NO .hbm model (the StereoPipeline itself needs the board +
LAS2 model and is covered by the on-board suite). Guarded so the file skips
cleanly off-board.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_stereo_py.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have(*names):
    return all(hasattr(bcdl, n) for n in names)


def _disp_map(arr):
    """Build a DepthMap holding raw disparity `arr` (HxW float32) via decode_depth
    with normalize off — the only Python route to a DepthMap with set data."""
    cfg = bcdl.DepthConfig()
    cfg.normalize = False
    return bcdl.decode_depth(np.ascontiguousarray(arr, dtype=np.float32), cfg)


# --------------------------------------------------------------------------- #
# pack_stereo_input: fit (resize/crop) + BGR->RGB + planar (3,H,W) float       #
# --------------------------------------------------------------------------- #
def test_pack_crop_channel_swap():
    if not _have("pack_stereo_input", "StereoFit"):
        pytest.skip("pack_stereo_input not exposed")
    # 4x4 BGR image; each pixel = (B,G,R) = (x, 100+y, 200+x). Center-crop 2x2.
    h, w = 4, 4
    bgr = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            bgr[y, x] = (x, 100 + y, 200 + x)
    out = bcdl.pack_stereo_input(bgr, 2, 2, bcdl.StereoFit.Crop, True)  # to_rgb
    assert out.shape == (3, 2, 2) and out.dtype == np.float32
    # center crop top/left = (4-2)//2 = 1 -> source rows/cols [1,2]
    # plane 0 = R = 200+x, plane 1 = G = 100+y, plane 2 = B = x
    for j, sx in enumerate((1, 2)):
        for i, sy in enumerate((1, 2)):
            assert out[0, i, j] == 200 + sx  # R
            assert out[1, i, j] == 100 + sy  # G
            assert out[2, i, j] == sx        # B


def test_pack_no_swap_keeps_bgr():
    if not _have("pack_stereo_input", "StereoFit"):
        pytest.skip("pack_stereo_input not exposed")
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[:, :] = (10, 20, 30)  # B,G,R
    out = bcdl.pack_stereo_input(bgr, 2, 2, bcdl.StereoFit.Crop, False)  # keep BGR
    assert out[0].mean() == 10 and out[1].mean() == 20 and out[2].mean() == 30


def test_pack_resize_matches_full_frame():
    if not _have("pack_stereo_input", "StereoFit"):
        pytest.skip("pack_stereo_input not exposed")
    # Resize to the same size is identity (up to channel swap).
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = bcdl.pack_stereo_input(bgr, 8, 8, bcdl.StereoFit.Resize, True)
    assert np.allclose(out[0], bgr[:, :, 2])  # R
    assert np.allclose(out[2], bgr[:, :, 0])  # B


# --------------------------------------------------------------------------- #
# disparity_to_depth: z = fx*baseline/disp, 0 where disp<=eps                   #
# --------------------------------------------------------------------------- #
def test_disparity_to_depth():
    if not _have("disparity_to_depth", "decode_depth"):
        pytest.skip("disparity_to_depth not exposed")
    disp = np.array([[10.0, 20.0], [0.0, 5.0]], np.float32)
    dm = _disp_map(disp)
    fx, base = 700.0, 0.12
    depth = bcdl.disparity_to_depth(dm, fx, base)
    assert depth.shape == (2, 2)
    assert depth[0, 0] == pytest.approx(fx * base / 10.0)
    assert depth[0, 1] == pytest.approx(fx * base / 20.0)
    assert depth[1, 0] == 0.0  # disp 0 -> invalid
    assert depth[1, 1] == pytest.approx(fx * base / 5.0)


# --------------------------------------------------------------------------- #
# stereo_valid_mask: range + left-border + optional LR consistency             #
# --------------------------------------------------------------------------- #
def test_valid_mask_range_and_border():
    if not _have("stereo_valid_mask", "decode_depth"):
        pytest.skip("stereo_valid_mask not exposed")
    # column index x; disp constant 3 -> x-d>=0 fails for x<3.
    disp = np.full((1, 6), 3.0, np.float32)
    dm = _disp_map(disp)
    mask = bcdl.stereo_valid_mask(dm, disp_min=1.0, max_disp=192.0, left_margin=0)
    # x-d>=0  => x>=3 valid
    assert mask.tolist()[0] == [0, 0, 0, 1, 1, 1]
    # disp below disp_min invalidates everything
    dm2 = _disp_map(np.full((1, 6), 0.5, np.float32))
    assert bcdl.stereo_valid_mask(dm2, disp_min=1.0).sum() == 0


def test_valid_mask_lr_check():
    if not _have("stereo_valid_mask", "decode_depth"):
        pytest.skip("stereo_valid_mask not exposed")
    # Left disp all 2; right disp agrees except one column -> that pixel rejected.
    W = 8
    dl = np.full((1, W), 2.0, np.float32)
    dr = np.full((1, W), 2.0, np.float32)
    dr[0, 5] = 6.0  # disagree at the matched right column (x-2 for x=7 -> 5)
    dml, dmr = _disp_map(dl), _disp_map(dr)
    mask = bcdl.stereo_valid_mask(dml, disp_min=0.0, max_disp=192.0, left_margin=0,
                                  disp_right=dmr, lr_thresh=1.5)
    # x>=2 satisfies the border; x=7 matches right col 5 (disp 6) -> |2-6|>1.5 reject
    assert mask[0, 7] == 0
    assert mask[0, 6] == 1  # matches right col 4 (disp 2) -> ok

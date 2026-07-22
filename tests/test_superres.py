"""Super-resolution tiling: the geometry and the blend, without a model.

Both pieces are pure and both are easy to get subtly wrong in ways that only
show up as a visible artefact: a tile plan that leaves a gap or hangs off the
edge, or a weight ramp that hits zero and makes the normalized blend divide by
nothing at the image border.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_superres.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not all(hasattr(bcdl, n) for n in ("plan_tiles", "tile_weight", "SuperResConfig")):
        pytest.skip("superres bindings not exposed")


def _covered(w, h, tile_w, tile_h, overlap):
    """How many times each pixel is covered by the plan."""
    cov = np.zeros((h, w), np.int32)
    for t in bcdl.plan_tiles(w, h, tile_w, tile_h, overlap):
        cov[t.y:t.y + tile_h, t.x:t.x + tile_w] += 1
    return cov


# --------------------------------------------------------------------------- #
# Tile plan                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w,h", [(640, 480), (1000, 700), (256, 256), (257, 256),
                                 (255, 130), (1024, 256), (300, 1200)])
def test_plan_covers_every_pixel_and_never_hangs_off(w, h):
    _have()
    tile, ov = 256, 16
    pw, ph = max(w, tile), max(h, tile)      # the caller pads up to tile size
    plan = bcdl.plan_tiles(pw, ph, tile, tile, ov)
    for t in plan:
        assert 0 <= t.x and t.x + tile <= pw
        assert 0 <= t.y and t.y + tile <= ph
    assert (_covered(pw, ph, tile, tile, ov) >= 1).all(), "a pixel is not covered"


def test_last_tile_is_flush_with_the_far_edge():
    """Pushing the final tile flush rather than letting it overhang is what keeps
    every tile a legal read; the price is a larger overlap on that one seam."""
    _have()
    plan = bcdl.plan_tiles(600, 300, 256, 256, 16)
    assert max(t.x for t in plan) == 600 - 256
    assert max(t.y for t in plan) == 300 - 256


def test_smaller_than_a_tile_is_a_single_placement():
    _have()
    plan = bcdl.plan_tiles(256, 256, 256, 256, 16)
    assert len(plan) == 1 and (plan[0].x, plan[0].y) == (0, 0)


def test_overlap_reduces_stride_and_so_adds_tiles():
    _have()
    few = bcdl.plan_tiles(1024, 1024, 256, 256, 0)
    many = bcdl.plan_tiles(1024, 1024, 256, 256, 64)
    assert len(few) == 16                      # exactly 4x4, no overlap needed
    assert len(many) > len(few)


def test_non_square_tiles_use_each_axis_own_size():
    _have()
    plan = bcdl.plan_tiles(800, 400, 256, 128, 16)
    for t in plan:
        assert t.x + 256 <= 800 and t.y + 128 <= 400
    assert (_covered(800, 400, 256, 128, 16) >= 1).all()


def test_bad_overlap_is_rejected():
    _have()
    with pytest.raises(Exception):
        bcdl.plan_tiles(640, 480, 256, 256, 256)      # overlap == tile
    with pytest.raises(Exception):
        bcdl.plan_tiles(640, 480, 256, 256, -1)


# --------------------------------------------------------------------------- #
# Blend weights                                                                #
# --------------------------------------------------------------------------- #
def test_weight_is_flat_in_the_middle_and_ramps_at_the_ends():
    _have()
    n, ramp = 100, 20
    w = np.array([bcdl.tile_weight(i, n, ramp) for i in range(n)])
    assert w[n // 2] == pytest.approx(1.0)
    np.testing.assert_allclose(w[ramp:n - ramp], 1.0, atol=1e-6)
    assert w[0] < w[1] < w[2]                       # rising into the tile
    assert w[-1] < w[-2] < w[-3]
    np.testing.assert_allclose(w, w[::-1], atol=1e-6)   # symmetric


def test_weight_is_never_zero():
    """The blend divides by the accumulated weight. A zero at the tile edge means
    a border pixel — covered by exactly one tile — divides by zero."""
    _have()
    for n, ramp in ((64, 16), (256, 64), (10, 9)):
        w = [bcdl.tile_weight(i, n, ramp) for i in range(n)]
        assert min(w) > 0.0


def test_zero_ramp_disables_the_fade():
    _have()
    assert all(bcdl.tile_weight(i, 50, 0) == 1.0 for i in range(50))


def test_normalized_blend_reconstructs_a_constant_image():
    """The property the whole scheme rests on: whatever the weights, sum(w*v) /
    sum(w) over overlapping tiles must return v exactly — including at the
    borders, where only one tile contributes."""
    _have()
    W = H = 600
    tile, ov = 256, 16
    ramp = ov                                  # scale 1 for this pure check
    acc = np.zeros((H, W))
    wsum = np.zeros((H, W))
    for t in bcdl.plan_tiles(W, H, tile, tile, ov):
        wy = np.array([bcdl.tile_weight(i, tile, ramp) for i in range(tile)])
        wx = wy
        w = np.outer(wy, wx)
        acc[t.y:t.y + tile, t.x:t.x + tile] += w * 7.5      # constant "image"
        wsum[t.y:t.y + tile, t.x:t.x + tile] += w
    assert (wsum > 0).all()
    np.testing.assert_allclose(acc / wsum, 7.5, atol=1e-5)

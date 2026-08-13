"""Engine output-buffer layout: a padded model output must not be read flat.

The mirror of tests/test_engine_input_stride.py, and it went unnoticed for the
same reason — most RDK outputs are contiguous, so taking the valid prefix of the
device buffer happened to be right. It is not universal. ViTPose's
[1,133,64,48] heatmap has 48 f32 columns (192 B/row) padded to a 256 B stride;
PointPillars' [1,64,11546,1] pillar features pad each channel by 6 float slots.

The failure mode is what makes this worth a test: a flat read of a padded output
displaces row r by r*pad and returns a tensor that still looks like data. A
re-compiled PointPillars VFE was scored at cosine 0.70 against its own float
reference and written off as a quantization failure for a day; the artifact was
right to 0.9988 and the reader was wrong.

Runs on the board; each case skips when its model is absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_engine_output_stride.py
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

import board_models as bm  # noqa: E402  (tests/ is on sys.path under pytest)

VITPOSE = os.path.join(bm.MODELS, "vitpose_s_wholebody_nashm_256x192.hbm")
# Contiguous on both sides; small and always fetched by scripts/fetch_models.sh.
ANY_MODEL = os.path.join(bm.MODELS, "yolo26n_detect_nashm_640x640_nv12.hbm")


def _engine(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not on this board")
    return bcdl.Engine(path)


def _random_input(e, i):
    dt = np.dtype(e.input_dtype(i))
    n = e._e.input_packed_bytes(i) // dt.itemsize
    rng = np.random.default_rng(0)
    if dt.kind == "f":
        return rng.standard_normal(n).astype(dt)
    return rng.integers(0, 128, size=n).astype(dt)


def test_vitpose_output_is_padded_and_strides_are_exposed():
    """The precondition. If a future rebuild of this model comes out packed, the
    test below stops proving anything — so assert the padding is really there."""
    e = _engine(VITPOSE)
    assert e.output_shape(0) == [1, 133, 64, 48]
    assert e.output_stride(0) == [2179072, 16384, 256, 4]   # 48*4 = 192 B/row -> 256
    assert e._e.output_packed_bytes(0) == 1 * 133 * 64 * 48 * 4


def test_infer_returns_a_packed_tensor_not_the_sheared_prefix():
    """Read the same inference three ways: the flat prefix (the old behaviour),
    a stride walk, and Engine.infer(). The last two must agree, and the first
    must not — otherwise this model no longer exercises the bug."""
    e = _engine(VITPOSE)
    x = _random_input(e, 0)
    got = e.infer([x])[0]

    shape, stride = e.output_shape(0), e.output_stride(0)
    dt = np.dtype(e.output_dtype(0))
    raw = e._e.output_bytes(0)
    # Stride walk over one row of each of the first few heatmaps is enough to
    # pin the layout; walking all 408k elements in python is needlessly slow.
    buf = np.frombuffer(raw, dtype=dt)
    row_floats = stride[2] // dt.itemsize
    for k in (0, 1, 66, 132):
        for r in (0, 1, 63):
            base = (k * stride[1] + r * stride[2]) // dt.itemsize
            np.testing.assert_array_equal(got[0, k, r], buf[base:base + shape[3]])

    flat = np.frombuffer(raw, dtype=dt, count=int(np.prod(shape))).reshape(shape)
    assert row_floats != shape[3], "model is packed; this test proves nothing"
    assert not np.array_equal(flat, got), "flat read must differ from the packed read"


def test_output_packed_matches_infer():
    """Engine.infer() is a thin wrapper over output_packed(); keep them honest."""
    e = _engine(VITPOSE)
    got = e.infer([_random_input(e, 0)])[0]
    dt = np.dtype(e.output_dtype(0))
    packed = np.frombuffer(e._e.output_packed(0), dtype=dt).reshape(e.output_shape(0))
    np.testing.assert_array_equal(got, packed)


def test_packed_equals_device_for_a_contiguous_model():
    """The common case stays a plain memcpy — no behaviour change for the NV12
    models the whole detection path runs on."""
    e = _engine(ANY_MODEL)
    for i in range(e.num_outputs):
        assert e._e.output_packed_bytes(i) <= len(e._e.output_bytes(i))
        shape, stride = e.output_shape(i), e.output_stride(i)
        esz = np.dtype(e.output_dtype(i)).itemsize
        inner = esz
        for ax in range(len(shape) - 1, -1, -1):
            assert stride[ax] == inner, f"output {i} axis {ax} is padded"
            inner = stride[ax] * shape[ax]

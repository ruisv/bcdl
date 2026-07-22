"""Engine input-buffer layout: a padded model input must not be fed a packed
array by plain memcpy.

Most RDK models have contiguous input rows, so a flat memcpy into the device
buffer happened to be correct and the gap went unnoticed for a long time. It is
not universal: ArcFace R50's 112 f32 columns (448 B/row) are padded to a 512 B
stride, and memcpy'ing a packed 1x3x112x112 array shifted every row after the
first — the model then saw the same scrambled image whatever face was fed in, so
all its embeddings came out near-collinear (board-vs-float cosine 0.015) while
every per-layer quantization metric still looked healthy.

Runs on the board; each case skips when its model is absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_engine_input_stride.py
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

import board_models as bm  # noqa: E402  (tests/ is on sys.path under pytest)

ARCFACE = os.path.join(bm.MODELS, "arcface_r50_aligned_nashm_112x112.hbm")
# Any model works for the layout-agnostic checks; this one is small and always
# fetched by scripts/fetch_models.sh.
ANY_MODEL = os.path.join(bm.MODELS, "yolo26n_detect_nashm_640x640_nv12.hbm")


def _engine(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not on this board")
    return bcdl.Engine(path)


def test_arcface_input_is_padded_and_strides_are_exposed():
    """The regression's precondition. If a future rebuild of this model comes out
    packed, the test below stops proving anything — so assert the padding is
    really there."""
    e = _engine(ARCFACE)
    stride = e.input_stride(0)
    assert e.input_shape(0) == [1, 3, 112, 112]
    assert stride == [172032, 57344, 512, 4]      # 112*4 = 448 B/row, padded to 512
    assert e.input_packed_bytes(0) == 150528
    assert e.input_bytes(0) == 172032


def test_packed_array_lands_row_aligned_in_a_padded_input():
    """Feed a packed array and read the device buffer back through the model's own
    strides: every row must be where the model expects it, and the pad columns
    must be zero rather than the next row's pixels."""
    e = _engine(ARCFACE)
    x = np.arange(1 * 3 * 112 * 112, dtype=np.float32).reshape(1, 3, 112, 112)
    e._e.set_input(0, np.ascontiguousarray(x))

    raw = np.frombuffer(e._e.input_buffer_bytes(0), dtype=np.float32)
    # Device layout: 3 channels x 112 rows x 128 floats (112 valid + 16 pad).
    view = raw.reshape(3, 112, 128)
    np.testing.assert_array_equal(view[:, :, :112], x[0])
    assert not view[:, :, 112:].any(), "pad columns must be zeroed, not row-shifted data"


def test_device_layout_array_is_taken_as_is():
    """The other accepted form: data already laid out in the device stride."""
    e = _engine(ARCFACE)
    buf = np.full(e.input_bytes(0) // 4, 1.5, dtype=np.float32)
    e._e.set_input(0, buf)
    raw = np.frombuffer(e._e.input_buffer_bytes(0), dtype=np.float32)
    np.testing.assert_array_equal(raw, buf)


def test_a_size_that_is_neither_packed_nor_device_is_rejected():
    """The old code silently accepted any short buffer, which is how the ArcFace
    bug survived: a packed array is short for a padded input."""
    e = _engine(ANY_MODEL)
    short = np.zeros(e.input_bytes(0) // 2, dtype=np.uint8)
    with pytest.raises(Exception, match="size mismatch"):
        e._e.set_input(0, short)


def test_packed_equals_device_for_a_contiguous_model():
    """The common case stays a plain memcpy — no behaviour change for the NV12
    models the whole detection path runs on."""
    e = _engine(ANY_MODEL)
    for i in range(e.num_inputs):
        assert e.input_packed_bytes(i) == e.input_bytes(i)

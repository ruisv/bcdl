"""AsyncDetectionPipeline binding test (threaded preproc‖infer via nanobind).

Exercises the compiled ``bcdl.AsyncDetectionPipeline`` on a real NV12 YOLO
``.hbm``. This is the Python view of ``examples/async_check.cc``: we push an
interleaved sequence of two DISTINCT frames — a real image (frame A, which
detects objects) and a solid-black frame (frame B, which detects nothing) — and
assert that ``next()`` returns results in submission order, one per submitted
frame, with A-positions non-empty and B-positions empty. That proves the
threaded slot/queue handoff neither loses, duplicates, nor reorders frames, and
that submit()/next() (which release the GIL) round-trip correctly from Python.

The test image is loaded with bcdl's own JPU ``JpegDecoder`` + a numpy NV12->BGR
so there is NO cv2 dependency — it runs on the board (inside the `bcdl` conda
env) with just a model + JPEG present; skips cleanly otherwise.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_async_detection_py.py

Override the model / image via BCDL_DET_HBM / BCDL_TEST_IMG.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_HBM = os.environ.get(
    "BCDL_DET_HBM",
    os.path.join(_REPO, "models", "yolo26s_det_nashm_640x640_nv12.hbm"),
)
TEST_IMG = os.environ.get("BCDL_TEST_IMG", os.path.join(_REPO, "data", "images", "bus.jpg"))


def _require(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} not found: {path}")


def _nv12_to_bgr(flat, w, h):
    """Full-range BT.601 NV12 -> BGR (inverse of bcdl's bgrToNv12Cpu). `flat` is
    the packed (W*H*3/2,) uint8 buffer VpImage.to_numpy() returns for NV12."""
    ysize = w * h
    y = flat[:ysize].reshape(h, w).astype(np.float32)
    uv = flat[ysize:].reshape(h // 2, w // 2, 2).astype(np.float32)
    # Upsample chroma 2x2 to full res.
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1) - 128.0
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1) - 128.0
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)


def _load_bgr(path):
    """Decode a JPEG to a uint8 HxWx3 BGR array via bcdl's JPU decoder (no cv2)."""
    with open(path, "rb") as f:
        data = f.read()
    dec = bcdl.JpegDecoder(bcdl.ImageFormat.NV12)
    im = dec.decode(data)
    return _nv12_to_bgr(im.to_numpy(), im.width, im.height)


def _make_pipeline(depth=3):
    if not hasattr(bcdl, "AsyncDetectionPipeline"):
        pytest.skip("AsyncDetectionPipeline not exposed")
    _require(DET_HBM, "detection model")
    _require(TEST_IMG, "test image")

    img = _load_bgr(TEST_IMG)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8

    engine = bcdl.Engine(DET_HBM)
    cfg = bcdl.PipelineConfig()
    cfg.detect.num_classes = 80
    cfg.detect.conf_thresh = 0.25
    pipe = bcdl.AsyncDetectionPipeline(engine, cfg, depth)
    # Hold the engine ref for the test's duration (binding also keep_alive's it).
    return engine, pipe, img


def test_async_fifo_order_and_no_loss():
    engine, pipe, img = _make_pipeline(depth=3)
    black = np.zeros_like(img)  # distinct frame: no detections

    # Interleaved A/B pattern (A = real image, B = black) repeated. The result
    # sequence MUST follow this pattern in order: A non-empty, B empty.
    pattern = "ABAABBAB"
    seq = pattern * 6
    depth = 3

    results = []  # (expected_char, detections)
    out_idx = 0
    for i, c in enumerate(seq):
        frame = img if c == "A" else black
        assert pipe.submit(frame) is True
        if i >= depth:  # keep ~depth in flight, drain in order
            dets = pipe.next()
            assert dets is not None
            results.append((seq[out_idx], dets))
            out_idx += 1

    pipe.finish()
    while True:  # drain the last in-flight frames
        dets = pipe.next()
        if dets is None:
            break
        results.append((seq[out_idx], dets))
        out_idx += 1

    # No frame lost or duplicated.
    assert len(results) == len(seq), (
        f"produced {len(results)} != submitted {len(seq)} (frames lost/duplicated)"
    )

    # Order preserved: every A-position detected something, every B-position
    # (solid black) detected nothing.
    a_total = 0
    for c, dets in results:
        if c == "A":
            assert len(dets) > 0, "real frame produced no detections (order broken?)"
            a_total += len(dets)
            for d in dets:
                assert d.x2 > d.x1 and d.y2 > d.y1
        else:
            assert len(dets) == 0, "black frame produced detections (order broken?)"
    assert a_total > 0


def test_async_submit_rejected_after_finish():
    engine, pipe, img = _make_pipeline(depth=2)

    assert pipe.submit(img) is True
    pipe.finish()
    # After finish(), submit() must refuse new frames.
    assert pipe.submit(img) is False

    # The one in-flight frame still drains, then next() returns None forever.
    seen = 0
    while pipe.next() is not None:
        seen += 1
    assert seen == 1
    assert pipe.next() is None  # idempotent drained state


def test_async_rejects_bad_shape():
    engine, pipe, img = _make_pipeline(depth=2)
    with pytest.raises(Exception):
        pipe.submit(np.zeros((10, 10), dtype=np.uint8))  # not HxWx3
    pipe.finish()

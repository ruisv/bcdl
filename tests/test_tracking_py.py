"""TrackingPipeline binding test (detect + ByteTrack via nanobind).

Exercises the compiled ``bcdl.TrackingPipeline`` end-to-end on a real NV12 YOLO
``.hbm``: a still image is panned to synthesise motion, and we assert the tracker
emits stable track_ids across frames. Runs only on the board (inside the `bcdl`
conda env) with a model + image present; skips cleanly otherwise.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_tracking_py.py

Override the model / image via BCDL_DET_HBM / BCDL_TEST_IMG.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_HBM = os.environ.get(
    "BCDL_DET_HBM",
    os.path.join(_REPO, "models", "yolo26s_det_nashm_640x640_nv12.hbm"),
)
TEST_IMG = os.environ.get("BCDL_TEST_IMG", os.path.join(_REPO, "data", "images", "bus.jpg"))


def _require(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} not found: {path}")


def test_tracking_pipeline_stable_ids():
    if not hasattr(bcdl, "TrackingPipeline"):
        pytest.skip("TrackingPipeline not exposed")
    _require(DET_HBM, "detection model")
    _require(TEST_IMG, "test image")

    img = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)
    assert img is not None and img.ndim == 3 and img.shape[2] == 3

    engine = bcdl.Engine(DET_HBM)
    cfg = bcdl.PipelineConfig()
    cfg.detect.num_classes = 80
    cfg.detect.conf_thresh = 0.25
    pipe = bcdl.TrackingPipeline(engine, cfg, bcdl.ByteTrackConfig())

    frames = 8
    per_frame_ids = []
    for f in range(frames):
        M = np.float32([[1, 0, 10.0 * f], [0, 1, 0]])
        frame = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tracks = pipe.process(frame)
        per_frame_ids.append({t.track_id for t in tracks})
        # Boxes must be well-formed. (They are Kalman estimates, so they may
        # extend modestly past the frame edge — don't assert strict in-bounds.)
        for t in tracks:
            assert t.x2 > t.x1 and t.y2 > t.y1

    # Frame 0 must detect + spawn at least one track on a busy street scene.
    assert per_frame_ids[0], "no tracks on first frame — detector produced nothing"

    # At least one id must survive across >=4 consecutive frames (stable identity
    # under motion is the whole point of tracking).
    longest = _max_consecutive_persistence(per_frame_ids)
    assert longest >= 4, f"no track id persisted >=4 frames (max run={longest})"


def _max_consecutive_persistence(per_frame_ids):
    """Longest run of consecutive frames in which some single id appears throughout."""
    best = 0
    active = {}  # id -> current run length
    for ids in per_frame_ids:
        nxt = {}
        for i in ids:
            nxt[i] = active.get(i, 0) + 1
            best = max(best, nxt[i])
        active = nxt
    return best


def test_tracking_pipeline_reset():
    if not hasattr(bcdl, "TrackingPipeline"):
        pytest.skip("TrackingPipeline not exposed")
    _require(DET_HBM, "detection model")
    _require(TEST_IMG, "test image")

    img = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)
    engine = bcdl.Engine(DET_HBM)
    pipe = bcdl.TrackingPipeline(engine)

    pipe.process(img)
    first = {t.track_id for t in pipe.process(img)}
    pipe.reset()
    # After reset the id counter restarts, so the next created track is id 1.
    after = {t.track_id for t in pipe.process(img)}
    assert after, "no tracks after reset"
    assert min(after) == 1, f"reset should restart ids at 1, got {sorted(after)}"
    # last_detections reflects the most recent frame and is non-empty here.
    assert len(pipe.last_detections) >= 1

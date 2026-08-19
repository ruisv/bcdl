"""Bounds / memory-safety regression tests for the Engine-free numpy decoders.

These pin the fixes for out-of-bounds reads in the pure `decode_*` entry points:
the decoders used to trust caller-supplied config (num_classes / num_keypoints /
height / width) and only derive the grid from the `cls` tensor, so a mismatched
shape would index past the sibling buffers. Each case below would previously
over-read (garbage / crash); now it either raises a clean error or is clamped to
the buffer. Needs the compiled bcdl module; pure numpy otherwise (no board/model).

    PYTHONPATH=build:python pytest tests/test_memory_safety.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _lb():
    return bcdl.LetterboxInfo()


def test_decode_obb_rejects_too_few_class_channels():
    # cls has 2 channels but config asks for 80 -> would stride past cls; now raises.
    cls = [np.zeros((4, 4, 2), np.float32)]
    box = [np.zeros((4, 4, 4), np.float32)]
    ang = [np.zeros((4, 4, 1), np.float32)]
    cfg = bcdl.ObbConfig()
    cfg.num_classes = 80
    cfg.strides = [8]                          # one stride per scale
    with pytest.raises(Exception):
        bcdl.decode_obb(cls, box, ang, cfg, _lb())


def test_decode_obb_rejects_mismatched_grid():
    cls = [np.zeros((4, 4, 15), np.float32)]
    box = [np.zeros((2, 2, 4), np.float32)]  # smaller grid than cls
    ang = [np.zeros((4, 4, 1), np.float32)]
    cfg = bcdl.ObbConfig()
    cfg.num_classes = 15
    cfg.strides = [8]
    with pytest.raises(Exception):
        bcdl.decode_obb(cls, box, ang, cfg, _lb())


def test_decode_pose_rejects_too_few_kpt_channels():
    cls = [np.zeros((4, 4, 1), np.float32)]
    box = [np.zeros((4, 4, 4), np.float32)]
    kpt = [np.zeros((4, 4, 3), np.float32)]  # 1 keypoint, but config wants 17
    cfg = bcdl.PoseConfig()
    cfg.num_keypoints = 17
    cfg.strides = [8]
    with pytest.raises(Exception):
        bcdl.decode_pose(cls, box, kpt, cfg, _lb())


def test_decode_instance_seg_rejects_mismatched_grid():
    cls = [np.zeros((4, 4, 1), np.float32)]
    box = [np.zeros((2, 2, 4), np.float32)]   # smaller grid
    mc = [np.zeros((4, 4, 32), np.float32)]
    proto = np.zeros((8, 8, 32), np.float32)
    cfg = bcdl.InstanceSegConfig()
    cfg.strides = [8]
    with pytest.raises(Exception):
        bcdl.decode_instance_seg(cls, box, mc, proto, cfg, _lb(), 16, 16)


def test_decode_seg_clamps_oversized_num_classes():
    # tensor has 3 channels but cfg asks for 1000 -> clamp, no OOB, valid result.
    H = W = 8
    arr = np.random.rand(1, H, W, 3).astype(np.float32)
    cfg = bcdl.SegConfig()
    cfg.channels_first = False
    cfg.num_classes = 1000
    m = bcdl.decode_seg(arr, cfg)
    assert (m.width, m.height) == (W, H)
    labels = np.asarray(m.labels)
    assert labels.shape == (H, W)
    assert int(labels.max()) < 3  # clamped to the real channel count


def test_decode_depth_bounds_oversized_hw():
    # 64x64 buffer but cfg says 1024x1024 -> bounded read, no OOB / crash.
    arr = np.random.rand(64, 64).astype(np.float32)
    cfg = bcdl.DepthConfig()
    cfg.height = 1024
    cfg.width = 1024
    dm = bcdl.decode_depth(arr, cfg)
    assert (dm.width, dm.height) == (1024, 1024)
    assert dm.vmax >= dm.vmin                 # finite, well-formed range


def test_decode_dbnet_rejects_1d_input():
    with pytest.raises(Exception):
        bcdl.decode_dbnet(np.zeros(10, np.float32), bcdl.DbConfig(), _lb())


def test_valid_inputs_still_decode():
    # Sanity: the new guards must not reject well-formed inputs.
    cls = [np.zeros((4, 4, 15), np.float32)]
    box = [np.zeros((4, 4, 4), np.float32)]
    ang = [np.zeros((4, 4, 1), np.float32)]
    cfg = bcdl.ObbConfig()
    cfg.num_classes = 15
    cfg.strides = [8]
    bcdl.decode_obb(cls, box, ang, cfg, _lb())     # no raise


def _det(x1, y1, x2, y2, score=0.9, class_id=0):
    d = bcdl.Detection()
    d.x1, d.y1, d.x2, d.y2 = float(x1), float(y1), float(x2), float(y2)
    d.score, d.class_id = float(score), int(class_id)
    return d


def test_bytetrack_survives_degenerate_detection_boxes():
    """Degenerate boxes reach the tracker: `decodeDet` clamps a box that lies
    entirely above the frame to y1 == y2 == 0, and `makeStrack` passes that
    h == 0 straight through.

    The Kalman state is [cx, cy, w/h, h], so h == 0 makes the aspect term
    infinite and tlwh() then computes w = inf * 0 = NaN. That NaN box reaches
    shapeSimilarity(), which has no NaN guard, so a whole row of the assignment
    cost goes NaN; in the Hungarian solver both comparisons in the scan are
    false for NaN, `j1` stays -1, and the code reads p[-1] and writes used[-1].
    ASAN reports it as a heap-buffer-overflow READ in linearAssignment, reached
    from this very call with the BoostTrack cost paths on. Two things fix it:
    the box-side clamp removes the NaN at the source, and the solver no longer
    indexes with -1 whatever it is handed.

    A 4-byte OOB read does not reliably crash, so treat this as a smoke check
    that the input is accepted; the sharp gate is running it under ASAN.
    """
    cfg = bcdl.ByteTrackConfig()
    cfg.boost.rich_similarity = True
    cfg.boost.soft_biou = True
    cfg.boost.boost_detections = True
    t = bcdl.ByteTracker(cfg)
    for i in range(25):
        t.update([_det(10, 0, 60, 0),                    # zero height
                  _det(100, 100, 100, 160),              # zero width
                  _det(200 + i, 200, 260 + i, 280),      # a normal moving box
                  _det(400 - i, 300, 470 - i, 390, 0.4)])  # low-score box
    tracks = t.update([_det(226, 200, 286, 280)])
    assert any(tr.x2 > tr.x1 and tr.y2 > tr.y1 for tr in tracks)


def test_bytetrack_still_tracks_normal_boxes_across_frames():
    """Sanity: the box-side clamp must not disturb ordinary tracking."""
    t = bcdl.ByteTracker(bcdl.ByteTrackConfig())
    ids = []
    for i in range(6):
        tracks = t.update([_det(100 + 4 * i, 100, 160 + 4 * i, 200)])
        ids.append([tr.track_id for tr in tracks])
    settled = [i for i in ids if i]
    assert settled, "the tracker never emitted a confirmed track"
    assert len(set(tuple(i) for i in settled)) == 1, "track id was not stable: %s" % ids

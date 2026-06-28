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

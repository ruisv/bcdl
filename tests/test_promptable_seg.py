"""SAM mask-decoder tail tests (C++ decodeSamMasks via nanobind, numpy path).

Exercises ``bcdl.decode_sam_masks`` — the Engine-free tail of an EdgeSAM/SAM
mask decoder: pick the best candidate (argmax predicted IoU) and binarize its
low-resolution logits at the mask threshold. Needs only the compiled bcdl
extension (no .hbm, no board), skips cleanly otherwise.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_promptable_seg.py

Decode contract (src/tasks/promptable_seg.cc decodeSamMasks):
  best = argmax(iou_pred) if multimask else 0
  mask = (low_res_logits[best] > mask_threshold) as 0/1
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not hasattr(bcdl, "decode_sam_masks"):
        pytest.skip("decode_sam_masks not exposed in this build")


def test_decode_sam_masks_picks_best_iou_and_binarizes():
    _have()
    N, H, W = 3, 4, 4
    logits = np.full((N, H, W), -5.0, np.float32)
    # candidate 1 is the highest-IoU: a 2x2 positive block in its top-left.
    logits[1, 0:2, 0:2] = 3.0
    iou = np.array([0.10, 0.90, 0.30], np.float32)

    cfg = bcdl.SamConfig()          # mask_threshold 0.0, multimask True
    sm = bcdl.decode_sam_masks(logits, iou, cfg)
    assert sm.index == 1
    assert sm.iou == pytest.approx(0.90, abs=1e-6)
    assert (sm.mask_h, sm.mask_w) == (H, W)

    mask = np.asarray(sm.mask)
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8
    expected = np.zeros((H, W), np.uint8)
    expected[0:2, 0:2] = 1
    assert np.array_equal(mask, expected)


def test_decode_sam_masks_single_mask_mode_uses_index0():
    _have()
    N, H, W = 2, 2, 2
    logits = np.full((N, H, W), -1.0, np.float32)
    logits[0, 0, 0] = 2.0           # only candidate 0 has a positive pixel
    iou = np.array([0.2, 0.99], np.float32)
    cfg = bcdl.SamConfig()
    cfg.multimask = False           # ignore iou, take candidate 0
    sm = bcdl.decode_sam_masks(logits, iou, cfg)
    assert sm.index == 0
    assert int(np.asarray(sm.mask).sum()) == 1


def test_decode_sam_masks_threshold():
    _have()
    N, H, W = 1, 1, 3
    logits = np.array([[[-1.0, 0.5, 2.0]]], np.float32)
    iou = np.array([0.5], np.float32)
    cfg = bcdl.SamConfig()
    cfg.mask_threshold = 1.0        # only the 2.0 pixel survives
    sm = bcdl.decode_sam_masks(logits, iou, cfg)
    assert list(np.asarray(sm.mask).ravel()) == [0, 0, 1]

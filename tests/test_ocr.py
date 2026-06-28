"""OCR pure-function tests (CTC greedy decode + DBNet, numpy path).

Exercises the Engine-free `bcdl.decode_ctc` / `bcdl.decode_dbnet` bindings with
synthetic inputs whose expected output is known by construction — the Python
mirror of the C++ checks in examples/tracks_ocr_check.cc. Needs only the compiled
bcdl extension (no .hbm / board hardware); skips cleanly where it can't import.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_ocr.py

The end-to-end OCR path on the real PP-OCRv3 models is covered by
tests/test_board_models.py::test_ocr_two_stage and examples/ocr_demo.cc.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have(*names):
    if not all(hasattr(bcdl, n) for n in names):
        pytest.skip(f"{names} not exposed in this build")


# --------------------------------------------------------------------------- #
# CTC greedy decode                                                           #
# --------------------------------------------------------------------------- #
def test_decode_ctc_collapses_repeats_and_blanks():
    _have("decode_ctc")
    # dict[0] = blank, then a,b,c,d.
    dictionary = ["<blank>", "a", "b", "c", "d"]
    T, C = 6, 5
    # per-step argmax: a, a(repeat), blank, b, b(repeat), c  -> "abc"
    peaks = [1, 1, 0, 2, 2, 3]
    logits = np.zeros((T, C), np.float32)
    for t, p in enumerate(peaks):
        logits[t, p] = 5.0                       # softmax-ish peak

    r = bcdl.decode_ctc(logits, dictionary)
    assert r.text == "abc"                       # repeats collapsed, blank dropped
    assert r.score > 0.9                          # mean max-prob over emitted steps


def test_decode_ctc_all_blank_empty():
    _have("decode_ctc")
    dictionary = ["<blank>", "a", "b"]
    logits = np.zeros((4, 3), np.float32)
    logits[:, 0] = 5.0                            # every step argmax = blank
    r = bcdl.decode_ctc(logits, dictionary)
    assert r.text == ""


# --------------------------------------------------------------------------- #
# DBNet detection                                                             #
# --------------------------------------------------------------------------- #
def test_decode_dbnet_two_blocks():
    _have("decode_dbnet", "DbConfig", "compute_letterbox")
    H, W = 40, 60
    prob = np.full((H, W), 0.1, np.float32)
    prob[5:15, 10:25] = 0.9                       # block 1, center ~ (x=17, y=10)
    prob[20:33, 35:55] = 0.9                      # block 2, center ~ (x=45, y=26)

    cfg = bcdl.DbConfig()                         # bin 0.3 / box 0.6 / unclip 1.5
    lb = bcdl.compute_letterbox(W, H, W, H)       # identity map (scale 1, pad 0)
    boxes = bcdl.decode_dbnet(prob, cfg, lb)

    assert len(boxes) == 2                         # two separated foreground blocks

    def covers(cx, cy):
        return any(b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2 for b in boxes)

    assert covers(17, 10)                          # a box over block 1's center
    assert covers(45, 26)                          # a box over block 2's center
    for b in boxes:
        assert b.x2 > b.x1 and b.y2 > b.y1
        assert np.asarray(b.points).shape == (4, 2)


def test_decode_dbnet_empty_below_threshold():
    _have("decode_dbnet", "DbConfig", "compute_letterbox")
    H, W = 32, 32
    prob = np.full((H, W), 0.1, np.float32)        # all background, none > bin_thresh
    cfg = bcdl.DbConfig()
    lb = bcdl.compute_letterbox(W, H, W, H)
    assert len(bcdl.decode_dbnet(prob, cfg, lb)) == 0


# --------------------------------------------------------------------------- #
# Text-line orientation classifier (0 / 180)                                  #
# --------------------------------------------------------------------------- #
def test_decode_cls_dir_upright():
    _have("decode_cls_dir")
    r = bcdl.decode_cls_dir(np.array([0.95, 0.05], np.float32), 0.9)
    assert r.label == 0 and not r.flip180          # class 0 wins -> no flip
    assert r.score == pytest.approx(0.95, abs=1e-4)


def test_decode_cls_dir_flip180():
    _have("decode_cls_dir")
    r = bcdl.decode_cls_dir(np.array([0.08, 0.92], np.float32), 0.9)
    assert r.label == 1 and r.flip180              # class 1 (180) above thresh -> flip
    assert r.score == pytest.approx(0.92, abs=1e-4)


def test_decode_cls_dir_180_below_thresh_no_flip():
    _have("decode_cls_dir")
    r = bcdl.decode_cls_dir(np.array([0.4, 0.6], np.float32), 0.9)
    assert r.label == 1 and not r.flip180          # 180 wins but below thresh -> keep

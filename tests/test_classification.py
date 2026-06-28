"""Classification decode tests (softmax + top-k, numpy path).

Exercises the Engine-free `bcdl.decode_classification` binding with synthetic
logits whose ranking is known by construction. Needs only the compiled bcdl
extension (no .hbm / board). The end-to-end real-model path is covered by
tests/test_board_models.py::test_classification_zebra.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_classification.py
"""

import math

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not all(hasattr(bcdl, n) for n in ("decode_classification", "ClsConfig")):
        pytest.skip("decode_classification / ClsConfig not exposed")


# logits over 5 classes; ranking by value: 1 (4.0) > 3 (2.0) > 2 (1.0) > 0 (0.5) > 4 (0.0)
LOGITS = np.array([0.5, 4.0, 1.0, 2.0, 0.0], np.float32)


def test_decode_classification_topk_softmax():
    _have()
    cfg = bcdl.ClsConfig()
    cfg.top_k = 3
    cfg.apply_softmax = True
    res = bcdl.decode_classification(LOGITS, cfg)

    assert len(res) == 3                                  # truncated to top_k
    assert [r.class_id for r in res] == [1, 3, 2]         # ranked by logit
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)         # descending
    assert all(0.0 < s < 1.0 for s in scores)             # softmax probabilities
    # top-1 softmax prob matches the closed-form value.
    denom = float(np.exp(LOGITS).sum())
    assert res[0].score == pytest.approx(math.exp(4.0) / denom, abs=1e-4)


def test_decode_classification_no_softmax_raw_logits():
    _have()
    cfg = bcdl.ClsConfig()
    cfg.top_k = 2
    cfg.apply_softmax = False
    res = bcdl.decode_classification(LOGITS, cfg)

    assert [r.class_id for r in res] == [1, 3]
    assert res[0].score == pytest.approx(4.0, abs=1e-4)   # raw logit, no softmax
    assert res[1].score == pytest.approx(2.0, abs=1e-4)


def test_decode_classification_top1():
    _have()
    cfg = bcdl.ClsConfig()
    cfg.top_k = 1
    res = bcdl.decode_classification(LOGITS, cfg)
    assert len(res) == 1
    assert res[0].class_id == 1

"""ReID embedding primitive tests (bcdl.normalize_embedding / cosine_similarity).

Board-independent BoT-SORT association primitives: L2-normalize an appearance
vector and compare two vectors by cosine similarity. Needs only the compiled
bcdl extension.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_reid.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not hasattr(bcdl, "normalize_embedding"):
        pytest.skip("reid primitives not exposed in this build")


def test_normalize_unit_norm():
    _have()
    v = np.array([3.0, 4.0], np.float32)            # norm 5
    out = np.asarray(bcdl.normalize_embedding(v), np.float32)
    assert out == pytest.approx([0.6, 0.8], abs=1e-5)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_normalize_zero_vector_is_noop():
    _have()
    out = np.asarray(bcdl.normalize_embedding(np.zeros(4, np.float32)), np.float32)
    assert out == pytest.approx([0, 0, 0, 0])


def test_cosine_similarity():
    _have()
    a = [1.0, 0.0, 0.0]
    assert bcdl.cosine_similarity(a, [1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-5)
    assert bcdl.cosine_similarity(a, [0.0, 1.0, 0.0]) == pytest.approx(0.0, abs=1e-5)
    assert bcdl.cosine_similarity(a, [-1.0, 0.0, 0.0]) == pytest.approx(-1.0, abs=1e-5)
    # unnormalized inputs are normalized internally
    assert bcdl.cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_length_mismatch_zero():
    _have()
    assert bcdl.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

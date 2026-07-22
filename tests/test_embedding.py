"""Embedding decode + bank search tests (numpy path, no model).

Exercises the Engine-free `bcdl.decode_embedding` / `bcdl.EmbeddingBank`
bindings with vectors whose similarity ordering is known by construction. Needs
only the compiled bcdl extension (no .hbm / board). The real-model path — a
dual-encoder image tower, its embedding width, and retrieval quality against a
host reference — is covered by tests/test_embedding_py.py.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_embedding.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not all(hasattr(bcdl, n) for n in ("decode_embedding", "EmbeddingBank",
                                          "EmbedConfig")):
        pytest.skip("embedding bindings not exposed")


def _norm(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_decode_embedding_normalizes():
    _have()
    raw = np.array([3.0, 4.0], np.float32)          # norm 5
    out = np.asarray(bcdl.decode_embedding(raw, bcdl.EmbedConfig()))

    assert out.shape == (2,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    assert out == pytest.approx([0.6, 0.8], abs=1e-6)


def test_decode_embedding_raw_passthrough():
    _have()
    cfg = bcdl.EmbedConfig()
    cfg.l2_normalize = False
    raw = np.array([3.0, 4.0], np.float32)
    out = np.asarray(bcdl.decode_embedding(raw, cfg))
    assert out == pytest.approx([3.0, 4.0], abs=1e-6)


def test_decode_embedding_zero_vector_is_not_nan():
    """A zero vector has no direction; normalizing it must not produce NaNs that
    would then poison every dot product it takes part in."""
    _have()
    out = np.asarray(bcdl.decode_embedding(np.zeros(4, np.float32),
                                           bcdl.EmbedConfig()))
    assert np.all(np.isfinite(out))
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_bank_search_ranks_by_cosine():
    _have()
    bank = bcdl.EmbeddingBank()
    # Four unit vectors in the plane at 0, 45, 90 and 180 degrees.
    bank.add([1.0, 0.0], "east")
    bank.add([0.7071, 0.7071], "northeast")
    bank.add([0.0, 1.0], "north")
    bank.add([-1.0, 0.0], "west")

    assert len(bank) == 4
    assert bank.dim == 2

    res = bank.search([1.0, 0.0], 3)
    assert [r.label for r in res] == ["east", "northeast", "north"]
    assert [r.index for r in res] == [0, 1, 2]
    assert res[0].score == pytest.approx(1.0, abs=1e-4)      # identical
    assert res[1].score == pytest.approx(0.7071, abs=1e-4)   # 45 degrees
    assert res[2].score == pytest.approx(0.0, abs=1e-4)      # orthogonal
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)


def test_bank_normalizes_on_insert_and_query():
    """Neither side has to arrive pre-normalized: scale is divided out, so only
    direction decides the score."""
    _have()
    bank = bcdl.EmbeddingBank()
    bank.add([100.0, 0.0], "east")       # same direction, wildly different norm
    res = bank.search([0.001, 0.0], 1)
    assert res[0].score == pytest.approx(1.0, abs=1e-4)


def test_bank_search_k_bounds():
    _have()
    bank = bcdl.EmbeddingBank()
    bank.add([1.0, 0.0])
    bank.add([0.0, 1.0])

    assert len(bank.search([1.0, 0.0], 0)) == 2      # k <= 0 => all
    assert len(bank.search([1.0, 0.0], 99)) == 2     # k > size => all
    assert len(bank.search([1.0, 0.0], 1)) == 1
    assert bank.search([1.0, 0.0], 1)[0].label == ""  # added without a label


def test_bank_dimension_mismatch_raises():
    """A silent dimension mismatch would surface as meaningless scores rather
    than an error, so both add() and search() check it."""
    _have()
    bank = bcdl.EmbeddingBank()
    bank.add([1.0, 0.0, 0.0])
    with pytest.raises(Exception):
        bank.add([1.0, 0.0])
    with pytest.raises(Exception):
        bank.search([1.0, 0.0], 1)


def test_bank_empty_search_returns_empty():
    _have()
    assert bcdl.EmbeddingBank().search([1.0, 0.0], 5) == []


def test_zero_shot_argmax_against_a_text_table():
    """The shape zero-shot classification actually takes: a table of offline
    text vectors, one per class name, and an image vector whose argmax over the
    table is the prediction."""
    _have()
    classes = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "car": [0.0, 0.0, 1.0]}
    bank = bcdl.EmbeddingBank()
    for name, vec in classes.items():
        bank.add(vec, name)

    # An image vector leaning towards "dog" with some noise on the other axes.
    img_vec = list(_norm([0.2, 0.9, 0.1]))
    top = bank.search(img_vec, 1)[0]
    assert top.label == "dog"
    assert 0.0 < top.score <= 1.0


def test_embed_preprocess_shape_and_range():
    """embed_preprocess squashes to a square (no letterbox bars) and maps u8 to
    [-1,1]; needs cv2, skipped where it is unavailable."""
    _have()
    pytest.importorskip("cv2")
    if not hasattr(bcdl, "embed_preprocess"):
        pytest.skip("embed_preprocess not exposed")

    img = np.full((100, 300, 3), 255, np.uint8)      # 3:1 aspect, pure white
    out = bcdl.embed_preprocess(img, 224)

    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32
    # White maps to +1 everywhere: a letterbox would have left grey bars behind.
    assert out.min() == pytest.approx(1.0, abs=1e-3)
    assert out.max() == pytest.approx(1.0, abs=1e-3)

    black = bcdl.embed_preprocess(np.zeros((50, 50, 3), np.uint8), 224)
    assert black.min() == pytest.approx(-1.0, abs=1e-3)

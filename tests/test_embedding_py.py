"""Image-embedding tests against a REAL dual-encoder image tower on the board.

Needs an embedding ``.hbm`` in ``models/`` and the ``bcdl`` extension, so every
test skips cleanly when either is missing (off-board, or before the model is
fetched). The Engine-free decode/bank maths is covered by tests/test_embedding.py.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_embedding_py.py

WHY THESE ASSERTIONS. An embedding cannot be eyeballed the way a bounding box
can, and a quantized tower that has genuinely degraded still returns plausible-
looking unit vectors — top-1 on an easy gallery keeps passing long after
retrieval has gotten worse. So the checks here are the ones that actually bite:
the vector is unit-length and deterministic, the pooled submodel (not the patch
one) is bound, semantically identical input scores ~1.0, and retrieval over a
gallery ranks an image's own degraded copy first. The quantitative bar — cosine
against a host float reference, and recall@K parity with it — is measured
offline against the model author's implementation, since it needs a host.
"""

import glob
import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
EMBED_MODEL = os.environ.get(
    "BCDL_EMBED_MODEL", os.path.join(MODELS, "bpu-siglip-base-patch16-224.hbm"))
# Submodel holding the pooled whole-image vector (vs the per-patch feature grid).
EMBED_SUBMODEL = os.environ.get("BCDL_EMBED_SUBMODEL", "pooler_output")
IMAGES = os.path.join(_REPO, "data", "images")


def _embedder():
    if not hasattr(bcdl, "ImageEmbedder"):
        pytest.skip("ImageEmbedder not exposed")
    if not os.path.exists(EMBED_MODEL):
        pytest.skip(f"embedding model not deployed: {EMBED_MODEL}")
    eng = bcdl.Engine(EMBED_MODEL, EMBED_SUBMODEL)
    return bcdl.ImageEmbedder(eng)


def _img(name):
    p = os.path.join(IMAGES, name)
    if not os.path.exists(p):
        pytest.skip(f"missing test image: {p}")
    im = cv2.imread(p)
    if im is None:
        pytest.skip(f"unreadable test image: {p}")
    return im


def test_submodel_enumeration_lists_the_pooled_head():
    """A packed .hbm hides its submodels; model_names() is how you find the
    pooled one instead of guessing (or binding the patch head by accident)."""
    if not os.path.exists(EMBED_MODEL):
        pytest.skip(f"embedding model not deployed: {EMBED_MODEL}")
    names = bcdl.Engine.model_names(EMBED_MODEL)
    assert isinstance(names, list) and names
    assert EMBED_SUBMODEL in names


def test_pooled_head_is_bound_not_the_patch_grid():
    """The pooled head is one vector per image. Binding the patch-feature head
    instead flattens to num_patches * width — orders of magnitude wider, and
    silently meaningless as an embedding."""
    emb = _embedder()
    assert 128 <= emb.dim <= 4096, f"dim {emb.dim} looks like a patch grid, not a pooled vector"


def test_embedding_is_unit_length_and_deterministic():
    emb = _embedder()
    img = _img("bus.jpg")
    a = np.asarray(emb.embed_image(img), np.float32)
    b = np.asarray(emb.embed_image(img), np.float32)

    assert a.shape == (emb.dim,)
    assert np.all(np.isfinite(a))
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-4)
    # Same pixels through the same engine must give the same vector: any drift
    # here is a cache-discipline or buffer-reuse bug, not model behaviour.
    assert float(a @ b) == pytest.approx(1.0, abs=1e-6)


def test_identical_image_scores_far_above_unrelated_ones():
    """The signal retrieval depends on: self-similarity must clear the rest by a
    wide margin, not merely edge it out."""
    emb = _embedder()
    names = ["bus.jpg", "bird.jpg", "zidane.jpg", "ocr.jpg"]
    vecs = [np.asarray(emb.embed_image(_img(n)), np.float32) for n in names]

    self_sim = float(vecs[0] @ vecs[0])
    others = [float(vecs[0] @ v) for v in vecs[1:]]
    assert self_sim == pytest.approx(1.0, abs=1e-4)
    assert max(others) < 0.9, f"unrelated images too close: {others}"


def test_bank_retrieves_a_degraded_copy_of_the_right_image():
    """End-to-end retrieval: re-encode each image as a low-quality JPEG and ask
    the bank for it. Rank 1 must be the original. This is the check that fails
    when quantization has eaten the descriptor, while unit-length and
    determinism both still pass."""
    emb = _embedder()
    names = ["bus.jpg", "bird.jpg", "zidane.jpg", "obb.jpg", "ocr.jpg"]
    imgs = {n: _img(n) for n in names}

    bank = bcdl.EmbeddingBank()
    for n in names:
        bank.add(np.asarray(emb.embed_image(imgs[n]), np.float32).tolist(), n)

    for n in names:
        ok, buf = cv2.imencode(".jpg", imgs[n], [cv2.IMWRITE_JPEG_QUALITY, 40])
        assert ok
        degraded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        q = np.asarray(emb.embed_image(degraded), np.float32)
        top = bank.search(q.tolist(), 2)
        assert top[0].label == n, f"{n} degraded copy retrieved {top[0].label}"
        # And it should win clearly, not by a hair.
        assert top[0].score - top[1].score > 0.05


def test_preprocess_does_not_letterbox():
    """A letterbox would feed the tower grey bars it was never trained on. The
    squashing resize is deliberate; guard it against a well-meaning 'fix'."""
    if not hasattr(bcdl, "embed_preprocess"):
        pytest.skip("embed_preprocess not exposed")
    wide = np.full((80, 400, 3), 200, np.uint8)
    x = bcdl.embed_preprocess(wide, 224)
    assert x.shape == (1, 3, 224, 224)
    # Uniform input stays uniform: padding bars would show up as a second value.
    assert float(x.max() - x.min()) < 1e-3

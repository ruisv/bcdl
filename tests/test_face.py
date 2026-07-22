"""Face detection decode + alignment tests, numpy path.

Exercises the Engine-free `bcdl.decode_scrfd`, `bcdl.similarity_transform` and
`bcdl.align_face` bindings with cases whose answer is known by construction.
Needs only the compiled bcdl extension (no .hbm / board); the real-model path is
in tests/test_board_models.py.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_face.py

WHY ALIGNMENT GETS THIS MUCH ATTENTION. The recognition model was trained solely
on faces warped so the eyes, nose and mouth corners sit on fixed template
positions. A plain crop of the detected box degrades its embeddings badly while
erroring nowhere, so the warp is a correctness requirement, not a refinement —
and a silently mirrored or sheared warp is the subtle way to break it.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not all(hasattr(bcdl, n) for n in ("decode_scrfd", "FaceDetectConfig",
                                          "similarity_transform", "align_face",
                                          "arcface_template")):
        pytest.skip("face bindings not exposed")


def _lb(src_w=640, src_h=640):
    """SCRFD-style letterbox: no padding, content at the top-left."""
    lb = bcdl.compute_letterbox(src_w, src_h, 640, 640, True)
    lb.pad_x = 0.0
    lb.pad_y = 0.0
    lb.scale = min(640 / src_w, 640 / src_h)
    return lb


def _cfg(conf=0.5, iou=0.4, strides=(8,), na=2):
    cfg = bcdl.FaceDetectConfig()
    cfg.conf_thresh = conf
    cfg.iou_thresh = iou
    cfg.max_faces = 100
    cfg.strides = list(strides)
    cfg.num_anchors = na
    return cfg


# --------------------------------------------------------------------------- #
# Detection decode                                                             #
# --------------------------------------------------------------------------- #
def test_single_prediction_known_by_construction():
    """One 2x2 grid, one active cell. The head places the anchor centre ON the
    grid point (no half-cell offset) and regresses distances in STRIDE units, so
    every number here can be worked out by hand."""
    _have()
    stride, g, na = 8, 2, 2
    n = g * g * na
    score = np.zeros((n, 1), np.float32)
    bbox = np.zeros((n, 4), np.float32)
    kps = np.zeros((n, 10), np.float32)

    # cell index 3 = (gy=1, gx=1) -> centre (8, 8); anchor 0 -> flat index 6.
    i = 3 * na
    score[i] = 0.9
    bbox[i] = [1.0, 2.0, 3.0, 4.0]          # l,t,r,b in stride units
    kps[i] = [0.5, 0.5, 1.0, 0.5, 0.75, 1.0, 0.5, 1.5, 1.0, 1.5]

    dets = bcdl.decode_scrfd([score], [bbox], [kps], _cfg(strides=(stride,)), _lb())
    assert len(dets) == 1
    d = dets[0]
    assert d.score == pytest.approx(0.9, abs=1e-5)
    assert d.x1 == pytest.approx(8 - 1 * stride, abs=1e-4)     # 0
    assert d.y1 == pytest.approx(max(0.0, 8 - 2 * stride), abs=1e-4)
    assert d.x2 == pytest.approx(8 + 3 * stride, abs=1e-4)     # 32
    assert d.y2 == pytest.approx(8 + 4 * stride, abs=1e-4)     # 40
    assert len(d.landmarks) == 5
    assert d.landmarks[0] == pytest.approx((8 + 0.5 * stride, 8 + 0.5 * stride), abs=1e-4)
    assert d.landmarks[1] == pytest.approx((8 + 1.0 * stride, 8 + 0.5 * stride), abs=1e-4)


def test_anchor_ordering_is_anchor_fastest():
    """Predictions are laid out (row, column, anchor) with anchors varying
    fastest. Getting this backwards puts faces in the wrong cell — which looks
    plausible on a centred portrait and wrong everywhere else."""
    _have()
    stride, g, na = 8, 4, 2
    n = g * g * na
    score = np.zeros((n, 1), np.float32)
    bbox = np.full((n, 4), 1.0, np.float32)
    kps = np.zeros((n, 10), np.float32)

    # flat index 18 -> cell 9 -> (gy=2, gx=1) -> centre (8, 16). Chosen away from
    # the border so the +-1*stride box stays inside the frame and the clamp does
    # not move the centre (which would mask an ordering error).
    score[18] = 0.99
    dets = bcdl.decode_scrfd([score], [bbox], [kps], _cfg(strides=(stride,), na=na), _lb())
    assert len(dets) == 1
    cx = (dets[0].x1 + dets[0].x2) / 2
    cy = (dets[0].y1 + dets[0].y2) / 2
    assert cx == pytest.approx(8.0, abs=1e-3)
    assert cy == pytest.approx(16.0, abs=1e-3)


def test_threshold_and_nms():
    _have()
    stride, g, na = 8, 2, 2
    n = g * g * na
    score = np.zeros((n, 1), np.float32)
    bbox = np.full((n, 4), 2.0, np.float32)
    kps = np.zeros((n, 10), np.float32)

    score[0] = 0.9      # cell 0, anchor 0
    score[1] = 0.8      # cell 0, anchor 1 -> identical box, must be suppressed
    dets = bcdl.decode_scrfd([score], [bbox], [kps], _cfg(strides=(stride,)), _lb())
    assert len(dets) == 1                      # NMS collapsed the duplicate
    assert dets[0].score == pytest.approx(0.9, abs=1e-5)

    score[:] = 0.1                              # everything below threshold
    assert bcdl.decode_scrfd([score], [bbox], [kps], _cfg(strides=(stride,)), _lb()) == []


def test_multi_scale_pooled():
    _have()
    outs = []
    for g in (8, 4, 2):
        n = g * g * 2
        s = np.zeros((n, 1), np.float32)
        s[0] = 0.9
        outs.append((s, np.full((n, 4), 1.0, np.float32), np.zeros((n, 10), np.float32)))
    dets = bcdl.decode_scrfd([o[0] for o in outs], [o[1] for o in outs],
                             [o[2] for o in outs], _cfg(strides=(8, 16, 32)), _lb())
    # All three fire at cell 0 with different strides -> different box sizes, so
    # NMS keeps them apart rather than collapsing them into one.
    assert len(dets) >= 2


def test_non_square_grid_raises():
    """The grid side is recovered from the flattened count, so a count that is
    not num_anchors * a perfect square means num_anchors is wrong — better to
    say so than to mis-index every cell."""
    _have()
    n = 10                                     # 10/2 = 5, not a square
    score = np.zeros((n, 1), np.float32)
    with pytest.raises(Exception):
        bcdl.decode_scrfd([score], [np.zeros((n, 4), np.float32)],
                          [np.zeros((n, 10), np.float32)], _cfg(), _lb())


# --------------------------------------------------------------------------- #
# Alignment                                                                    #
# --------------------------------------------------------------------------- #
def test_similarity_transform_recovers_scale_rotation_translation():
    _have()
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    m = bcdl.similarity_transform(src, [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)])
    assert [round(v, 4) for v in m] == [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]

    shifted = [(x + 5, y - 3) for x, y in src]
    m = bcdl.similarity_transform(src, shifted)
    assert m[0] == pytest.approx(1.0, abs=1e-4) and m[4] == pytest.approx(1.0, abs=1e-4)
    assert m[2] == pytest.approx(5.0, abs=1e-4) and m[5] == pytest.approx(-3.0, abs=1e-4)

    # 90 degrees: (x,y) -> (-y,x)
    rot = [(-y, x) for x, y in src]
    m = bcdl.similarity_transform(src, rot)
    assert m[0] == pytest.approx(0.0, abs=1e-4)
    assert m[1] == pytest.approx(-1.0, abs=1e-4)
    assert m[3] == pytest.approx(1.0, abs=1e-4)


def test_similarity_transform_is_shear_free():
    """A general affine could stretch one axis to fit; a similarity must not.
    Given deliberately non-uniform targets the solution stays uniformly scaled,
    which is what keeps face proportions intact for the embedder."""
    _have()
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    stretched = [(0.0, 0.0), (4.0, 0.0), (0.0, 1.0), (4.0, 1.0)]   # 4x on x only
    a, b, _, c, d, _ = bcdl.similarity_transform(src, stretched)
    assert a == pytest.approx(d, abs=1e-4)        # same scale on both axes
    assert b == pytest.approx(-c, abs=1e-4)       # rotation, not shear


def test_align_face_puts_landmarks_on_the_template():
    """The whole point of the warp: after it, the five landmarks must land on the
    template positions. Verified by transforming them with the same solved
    matrix rather than by eyeballing the crop."""
    _have()
    tmpl = bcdl.arcface_template()
    # A face whose landmarks are the template scaled 2x and shifted — the warp
    # must undo exactly that.
    lm = [(x * 2 + 30, y * 2 + 20) for x, y in tmpl]
    m = bcdl.similarity_transform(lm, tmpl)
    for (x, y), (tx, ty) in zip(lm, tmpl):
        px = m[0] * x + m[1] * y + m[2]
        py = m[3] * x + m[4] * y + m[5]
        assert px == pytest.approx(tx, abs=1e-3)
        assert py == pytest.approx(ty, abs=1e-3)


def test_align_face_output_shape_and_content():
    _have()
    img = np.zeros((200, 200, 3), np.uint8)
    img[60:140, 60:140] = 200                      # a bright square "face"
    tmpl = bcdl.arcface_template()
    lm = [(x * 0.7 + 62, y * 0.7 + 62) for x, y in tmpl]

    out = bcdl.align_face(img, lm, 112)
    assert out.shape == (112, 112, 3)
    assert out.dtype == np.uint8
    # The template's eye/nose region falls inside the bright square, so the
    # centre of the aligned crop must be bright, not the black background.
    assert out[56, 56, 0] > 100


def test_align_face_size_parameter_scales_the_template():
    _have()
    img = np.full((200, 200, 3), 128, np.uint8)
    lm = [(x + 40, y + 40) for x, y in bcdl.arcface_template()]
    assert bcdl.align_face(img, lm, 112).shape == (112, 112, 3)
    assert bcdl.align_face(img, lm, 224).shape == (224, 224, 3)


def test_align_face_rejects_wrong_landmark_count():
    _have()
    img = np.zeros((50, 50, 3), np.uint8)
    with pytest.raises(Exception):
        bcdl.align_face(img, [(1.0, 1.0), (2.0, 2.0)], 112)

"""ReID tests: the embedding primitives and the appearance-gated association.

Board-independent BoT-SORT machinery — L2-normalize an appearance vector,
compare two vectors by cosine similarity, and hand a parallel list of vectors to
``ByteTracker.update`` so that association can use appearance as well as
geometry. No model and no board are needed: the tracker is fed synthetic boxes
and hand-written embeddings, which is the only way to state what SHOULD happen
when motion and appearance disagree.

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


# ---------------------------------------------------------------------------
# Appearance-gated association (ByteTracker.update(detections, embeddings))
# ---------------------------------------------------------------------------
#
# The scenario below is built so that geometry and appearance give DIFFERENT
# answers, because that is the only case where appearance can be shown to do
# anything. Two same-size boxes sit 25 px apart with 100 px width, so they
# overlap at IoU 0.6 — close enough that every cross pair passes the proximity
# gate (raw IoU distance 0.4 <= proximity_thresh 0.5) and appearance is allowed
# to speak at all. Then the two objects trade places in one frame. Motion says
# "each id keeps its position"; appearance says "each id follows its vector".

RED = [1.0, 0.0, 0.0, 0.0]
BLUE = [0.0, 1.0, 0.0, 0.0]


def _det(x, score=0.9, cls=0, w=100.0, h=200.0):
    d = bcdl.Detection()
    d.x1, d.y1, d.x2, d.y2 = float(x), 0.0, float(x + w), float(h)
    d.score, d.class_id = float(score), int(cls)
    return d


def _ids_left_to_right(tracks):
    """Track ids ordered by box position, left first.

    Deliberately ORDER rather than absolute position: a matched track's box is
    a Kalman estimate that sits partway between prediction and measurement, so
    asserting on exact pixels would be asserting on the filter's gain. Which id
    ended up on which side is the thing under test.
    """
    return [t.track_id for t in sorted(tracks, key=lambda t: t.x1)]


def _run_swap(use_appearance, cfg=None, final_embs=None):
    """Two overlapping objects hold still, then trade places on the last frame.

    Returns (ids_before, ids_after) keyed by position. Whether the ids follow
    the positions or follow the embeddings is the whole question. `final_embs`
    overrides the swapped pair, to test what happens when the appearance a
    tracklet is looking for has drifted.
    """
    tracker = bcdl.ByteTracker(cfg if cfg is not None else bcdl.ByteTrackConfig())
    before = None
    for _ in range(4):  # settle: both tracks confirmed, velocities ~0
        dets = [_det(100), _det(125)]
        embs = [RED, BLUE]
        before = _ids_left_to_right(
            tracker.update(dets, embs) if use_appearance else tracker.update(dets))

    # Same two boxes, embeddings swapped: the RED object is now the right one.
    dets = [_det(100), _det(125)]
    embs = final_embs if final_embs is not None else [BLUE, RED]
    after = _ids_left_to_right(
        tracker.update(dets, embs) if use_appearance else tracker.update(dets))
    return before, after


def _drifted(base, cos_to_base):
    """A unit vector at a chosen cosine from `base` (which must be an axis)."""
    v = np.array(base, np.float32) * cos_to_base
    v[3] = np.sqrt(max(0.0, 1.0 - cos_to_base ** 2))  # spare axis, unused by RED/BLUE
    return [float(x) for x in v]


def _have_reid_update():
    _have()
    try:
        bcdl.ByteTracker(bcdl.ByteTrackConfig()).update([], [])
    except TypeError:
        pytest.skip("ByteTracker appearance overload not exposed in this build")


def test_motion_only_ids_follow_position():
    """Baseline: without embeddings the ids stay pinned to the boxes."""
    _have_reid_update()
    before, after = _run_swap(use_appearance=False)
    assert len(before) == 2
    assert after == before, "motion alone keeps each id on its side"


def test_appearance_decides_ambiguous_pairing():
    """With embeddings, the ids follow the appearance instead of the position.

    This is the entire point of BoT-SORT association: when two candidates are
    both geometrically plausible, the one that LOOKS right wins.
    """
    _have_reid_update()
    before, after = _run_swap(use_appearance=True)
    assert len(before) == 2
    assert after == before[::-1], "each id should have crossed over to its own colour"


def test_proximity_gate_blocks_distant_appearance_match():
    """Appearance must not reach across space: tightening the proximity gate
    below the pair's IoU distance (0.4) puts the ids back on the boxes."""
    _have_reid_update()
    cfg = bcdl.ByteTrackConfig()
    cfg.proximity_thresh = 0.1  # 0.4 > 0.1 -> every cross pair is excluded
    before, after = _run_swap(use_appearance=True, cfg=cfg)
    assert after == before


def test_appearance_gate_blocks_dissimilar_match():
    """The cosine gate must actually gate. The swapped objects come back with
    their appearance DRIFTED to cosine 0.9 (distance (1-0.9)/2 = 0.05), which
    the default 0.25 accepts and a 0.01 threshold rejects — so the same frames
    give opposite answers on either side of the gate."""
    _have_reid_update()
    drifted = [_drifted(BLUE, 0.9), _drifted(RED, 0.9)]

    _, loose = _run_swap(use_appearance=True, final_embs=drifted)
    cfg = bcdl.ByteTrackConfig()
    cfg.appearance_thresh = 0.01
    before, tight = _run_swap(use_appearance=True, cfg=cfg, final_embs=drifted)

    assert loose == before[::-1], "0.05 < 0.25: drifted appearance still matches"
    assert tight == before, "0.05 > 0.01: gated out, geometry decides"


def test_class_gate_blocks_cross_class_appearance():
    """Appearance is never compared across classes — there is no shared
    appearance space between a person model's output and anything else."""
    _have_reid_update()
    tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
    for _ in range(4):
        before = _ids_left_to_right(tracker.update([_det(100, cls=0), _det(125, cls=1)],
                                          [RED, BLUE]))
    # Swap the embeddings AND keep the classes pinned to the positions: with the
    # class gate on, appearance is ignored and the ids stay put.
    after = _ids_left_to_right(tracker.update([_det(100, cls=0), _det(125, cls=1)],
                                              [BLUE, RED]))
    assert after == before


def test_empty_embeddings_fall_back_to_geometry():
    """An empty entry means 'no appearance for this detection' — the intended
    way to skip the ReID model on cheap crops."""
    _have_reid_update()
    tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
    for _ in range(4):
        before = _ids_left_to_right(tracker.update([_det(100), _det(125)], [[], []]))
    after = _ids_left_to_right(tracker.update([_det(100), _det(125)], [[], []]))
    assert after == before


def test_embedding_count_must_match_detections():
    _have_reid_update()
    tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
    with pytest.raises(Exception):
        tracker.update([_det(100), _det(125)], [RED])


def test_embedding_width_must_not_change_mid_stream():
    _have_reid_update()
    tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
    tracker.update([_det(100)], [RED])
    with pytest.raises(Exception):
        tracker.update([_det(100)], [[1.0, 0.0]])


def test_ema_alpha_must_be_a_fraction():
    _have()
    cfg = bcdl.ByteTrackConfig()
    cfg.ema_alpha = 1.5
    with pytest.raises(Exception):
        bcdl.ByteTracker(cfg)


# ---------------------------------------------------------------------------
# Crop preprocessing (bcdl.reid_crop_preprocess)
# ---------------------------------------------------------------------------


def _numpy_reference(crop_bgr, width=128, height=256):
    """cv2 + numpy statement of what a ReID crop is supposed to be."""
    cv2 = pytest.importorskip("cv2")
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    resized = cv2.resize(crop_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - mean) / std).transpose(2, 0, 1)[None]


def _have_preprocess():
    if not hasattr(bcdl, "reid_crop_preprocess"):
        pytest.skip("reid_crop_preprocess not exposed in this build")


def test_reid_preprocess_matches_the_numpy_reference():
    """The C++ crop path must reproduce cv2.resize + ImageNet z-score.

    Worth pinning because every one of its conventions fails SILENTLY: BGR left
    unswapped, a letterbox instead of a squash, or the pixel-center offset
    dropped all yield a well-formed array and a plausible-looking embedding that
    simply matches the wrong people.
    """
    _have_preprocess()
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (97, 43, 3), dtype=np.uint8)  # odd size on purpose

    got = np.asarray(bcdl.reid_preprocess(crop))
    assert got.shape == (1, 3, 256, 128) and got.dtype == np.float32
    assert np.abs(got - _numpy_reference(crop)).max() < 2e-2


def test_reid_preprocess_swaps_to_rgb():
    """A pure-blue crop must land in the BLUE channel of the output, i.e. index
    2 after the BGR->RGB swap. Catches the swap being dropped, which nothing
    about the array's shape or range would reveal."""
    _have_preprocess()
    blue = np.zeros((40, 20, 3), np.uint8)
    blue[..., 0] = 255                       # BGR: blue channel
    out = np.asarray(bcdl.reid_preprocess(blue))[0]
    assert out[2].mean() > out[0].mean() and out[2].mean() > out[1].mean()


def test_reid_preprocess_squashes_rather_than_pads():
    """A very wide crop must fill the whole canvas, not sit in a letterbox: an
    aspect-preserving resize would leave constant bars, and constant bars have
    zero variance."""
    _have_preprocess()
    rng = np.random.default_rng(1)
    wide = rng.integers(0, 256, (20, 400, 3), dtype=np.uint8)
    out = np.asarray(bcdl.reid_preprocess(wide))[0, 0]
    # Every output row and column must carry real content.
    assert out.std(axis=1).min() > 0.05, "rows look like padding bars"
    assert out.std(axis=0).min() > 0.05, "columns look like padding bars"


def test_reid_preprocess_clips_a_box_hanging_off_the_frame():
    """Detector boxes routinely run past the edge; the crop must clip instead of
    reading out of bounds."""
    _have_preprocess()
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    out = np.asarray(bcdl.reid_crop_preprocess(frame, -30.0, -20.0, 60.0, 90.0))
    assert out.shape == (1, 3, 256, 128) and np.isfinite(out).all()
    with pytest.raises(Exception):  # entirely outside -> empty after clipping
        bcdl.reid_crop_preprocess(frame, 200.0, 200.0, 300.0, 300.0)


# ---------------------------------------------------------------------------
# BoostTrack++ additions and camera-motion compensation
# ---------------------------------------------------------------------------
#
# Each test isolates ONE switch and picks a scenario where that switch is the
# only thing that can change the outcome — the same discipline as the appearance
# tests above, and the reason the flags are individually switchable at all.


def _have_boost():
    _have()
    if not hasattr(bcdl, "BoostConfig"):
        pytest.skip("BoostConfig not exposed in this build")


def test_camera_motion_preserves_id_across_a_pan():
    """A camera pan moves every box at once. Motion alone reads that as every
    target teleporting and drops the tracks; warping the tracklets by the
    measured camera transform first keeps them."""
    _have_boost()

    def run(with_gmc):
        t = bcdl.ByteTracker(bcdl.ByteTrackConfig())
        for _ in range(5):                       # settle a track at x=100
            ids = [x.track_id for x in t.update([_det(100)])]
        shift = 260.0                            # well beyond the box width
        if with_gmc:
            # 2x3 affine mapping the previous frame onto this one.
            t.apply_camera_motion(np.array([[1, 0, shift], [0, 1, 0]], np.float32))
        after = [x.track_id for x in t.update([_det(100 + shift)])]
        return ids, after

    before_no, after_no = run(False)
    before_yes, after_yes = run(True)
    assert before_no == before_yes and len(before_no) == 1
    assert after_no != before_no, "without GMC the pan should break the track"
    assert after_yes == before_yes, "with GMC the id should survive the pan"


def test_detection_boost_lets_a_weak_detection_refind_a_lost_track():
    """ByteTrack's low-score pool can only rescue tracks that are still TRACKED;
    a lost track can only be re-found from the high pool. So a weak detection
    sitting exactly where a lost track was predicted is wasted — unless the
    tracklet vouches for it and the boost promotes it."""
    _have_boost()

    def run(boost):
        cfg = bcdl.ByteTrackConfig()
        cfg.boost.boost_detections = boost
        t = bcdl.ByteTracker(cfg)
        for _ in range(6):
            ids = [x.track_id for x in t.update([_det(100)])]
        t.update([])                              # miss a frame -> the track is lost
        after = [x.track_id for x in t.update([_det(100, score=0.35)])]
        return ids, after

    before_off, after_off = run(False)
    before_on, after_on = run(True)
    assert len(before_off) == 1 and before_off == before_on
    assert after_off != before_off, "unboosted, the weak detection cannot re-find it"
    assert after_on == before_on, "boosted, the same detection re-finds the track"


def test_duo_boost_starts_a_track_from_an_unexplained_detection():
    """A weak detection that no tracklet can explain is more likely a new object
    than noise, so DUO promotes it past the track-creation threshold."""
    _have_boost()

    def run(boost):
        cfg = bcdl.ByteTrackConfig()
        cfg.boost.boost_detections = boost
        t = bcdl.ByteTracker(cfg)
        for _ in range(5):
            t.update([_det(100)])
        # A second object, far away, detected weakly (below high_thresh 0.6).
        for _ in range(3):
            tracks = t.update([_det(100), _det(900, score=0.45)])
        return len(tracks)

    assert run(False) == 1, "unboosted, the weak far detection should not start a track"
    assert run(True) == 2, "DUO should promote it into a track"


def test_boost_flags_are_off_by_default():
    """The whole ablation depends on `{}` meaning 'none of this'."""
    _have_boost()
    b = bcdl.ByteTrackConfig().boost
    assert not b.rich_similarity and not b.soft_biou and not b.boost_detections


def test_boost_switches_do_not_break_plain_tracking():
    """Turning everything on must still track a simple moving object."""
    _have_boost()
    cfg = bcdl.ByteTrackConfig()
    cfg.boost.rich_similarity = True
    cfg.boost.soft_biou = True
    cfg.boost.boost_detections = True
    t = bcdl.ByteTracker(cfg)
    ids = set()
    for f in range(10):
        for trk in t.update([_det(100 + 8 * f)]):
            ids.add(trk.track_id)
    assert len(ids) == 1, f"a single object produced {len(ids)} ids: {ids}"

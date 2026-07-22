"""On-board real-model tests for bcdl's task heads (YOLO26n nash-m family).

End-to-end on the BPU: preprocess a real image -> NV12 -> Engine.infer ->
bcdl decoder, asserting sane structured output for each task. Runs only on the
board (needs the compiled bcdl extension + cv2 + the official YOLO26n models +
test images); each task skips cleanly when its model or image is absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_board_models.py

Perf / memory numbers + check figures for these same models come from
scripts/board_bench.py (benchmarks/RESULTS.md). Model dir / assets are
env-overridable via BCDL_YOLO26_DIR / BCDL_ASSETS / BCDL_OBB_IMAGE.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
pytest.importorskip("cv2")

import board_models as bm  # noqa: E402  (tests/ is on sys.path under pytest)


def _task(key):
    if not bm.available(key):
        pytest.skip(f"{key}: model(s) or image not on this board")
    return bm.make_task(bcdl, key)


def _in_image(box, w, h, slack=2.0):
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1 and x1 >= -slack and y1 >= -slack \
        and x2 <= w + slack and y2 <= h + slack


def test_classification_zebra():
    t = _task("cls")
    res = t.decode()
    assert len(res) >= 1
    top = res[0]
    assert top.score > 0.5
    # zebra_cls.jpg is a zebra; ImageNet class 340 == "zebra".
    assert top.class_id == 340, f"expected zebra (340), got {top.class_id}"


def test_detection_bus():
    t = _task("det")
    res = t.decode()
    # bus.jpg: a bus + several persons.
    assert len(res) >= 4
    for d in res:
        assert 0.0 <= d.score <= 1.0 and d.score >= 0.25
        assert _in_image((d.x1, d.y1, d.x2, d.y2), t.orig_w, t.orig_h)
    classes = {d.class_id for d in res}
    assert 0 in classes      # person
    assert 5 in classes      # bus


def test_detection_dfl_bus():
    # YOLOv8 DFL head (box=64 = 4 sides x 16 bins) — YoloLtrbDetector auto-detects
    # reg_max and softmax-reduces each side to an LTRB distance.
    t = _task("det_dfl")
    res = t.decode()
    assert len(res) >= 4
    classes = {d.class_id for d in res}
    assert 0 in classes and 5 in classes        # person + bus
    for d in res:
        assert d.score >= 0.25
        assert _in_image((d.x1, d.y1, d.x2, d.y2), t.orig_w, t.orig_h)


def test_pose_keypoints():
    t = _task("pose")
    res = t.decode()
    assert len(res) >= 1     # at least one person
    for p in res:
        assert len(p.keypoints) == 17       # COCO keypoints
        assert _in_image((p.x1, p.y1, p.x2, p.y2), t.orig_w, t.orig_h)


def test_instance_seg_masks():
    t = _task("seg")
    res = t.decode()
    assert len(res) >= 1
    for m in res:
        assert (m.mask_w, m.mask_h) == (t.orig_w, t.orig_h)
        mask = m.mask
        assert mask.shape == (t.orig_h, t.orig_w)
        assert mask.dtype == np.uint8
        assert mask.any()                   # non-empty instance mask
        assert _in_image((m.x1, m.y1, m.x2, m.y2), t.orig_w, t.orig_h)


def test_obb_rotated_boxes():
    t = _task("obb")
    res = t.decode()
    # obb.jpg is a DOTA aerial scene (many vehicles) -> many rotated boxes.
    assert len(res) >= 1
    for d in res:
        r = d.rrect
        assert r.w > 0 and r.h > 0
        assert d.score >= 0.25
        assert 0 <= d.class_id < len(bm.DOTA_NAMES)


def test_semantic_seg_cityscapes():
    t = _task("semseg")
    seg = t.decode()
    assert (seg.width, seg.height) == (t.in_w, t.in_h)   # full-res 2048x1024
    labels = np.asarray(seg.labels)
    assert labels.shape == (t.in_h, t.in_w)
    ids = np.unique(labels)
    assert ids.min() >= 0 and ids.max() < 19             # Cityscapes 19 classes
    assert ids.size >= 3                                  # a real street scene


def test_semantic_seg_cityscapes_realtime():
    """Real-time semseg head (PIDNet-S) at the same 2048x1024 input.

    Unlike the deeplabv3plus head this one emits logits at 1/8 resolution, so the
    label map is 128x256 and the caller upsamples. That is the only difference:
    the same Segmenter argmax decodes it, which is what this test pins."""
    t = _task("semseg_rt")
    seg = t.decode()
    assert (seg.width, seg.height) == (t.in_w // 8, t.in_h // 8)   # 256x128
    labels = np.asarray(seg.labels)
    assert labels.shape == (t.in_h // 8, t.in_w // 8)
    ids = np.unique(labels)
    assert ids.min() >= 0 and ids.max() < 19             # Cityscapes 19 classes
    assert ids.size >= 3                                  # a real street scene


def test_face_detect_align_embed():
    """The whole face chain on real models: detect -> align -> embed.

    Recognition is not a new task here — an aligned crop goes through
    ImageEmbedder and matching is EmbeddingBank. What is face-specific is the
    landmark decode and the warp, and the warp is the part that silently ruins
    recognition when it is wrong (the model only ever saw faces on the canonical
    template), so this checks landmark geometry and the aligned crop, not just
    that a box came back."""
    import os

    import cv2

    det_model = os.environ.get("BCDL_FACE_DET_MODEL",
                               os.path.join(bm.MODELS, "scrfd_10g_nashm_640x640_nv12.hbm"))
    if not os.path.exists(det_model):
        pytest.skip(f"face detector not deployed: {det_model}")
    for name in ("FaceDetector", "FaceDetectConfig", "align_face", "face_letterbox"):
        if not hasattr(bcdl, name):
            pytest.skip(f"{name} not exposed")

    img_path = bm.find_image("zidane.jpg")      # two people, faces clearly visible
    if not os.path.exists(img_path):
        pytest.skip(f"missing test image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        pytest.skip(f"unreadable test image: {img_path}")

    eng = bcdl.Engine(det_model)
    assert eng.num_outputs == 9                 # 3 scales x (score, bbox, kps)
    cfg = bcdl.FaceDetectConfig(); cfg.conf_thresh = 0.5
    det = bcdl.FaceDetector(eng, cfg, 0)

    canvas, lb = bcdl.face_letterbox(img, 640)
    assert lb.pad_x == 0.0 and lb.pad_y == 0.0  # bottom-right padding convention
    nv12 = np.asarray(bcdl.bgr_to_nv12(canvas)).reshape(-1)
    eng._e.set_input(0, np.ascontiguousarray(nv12[:640 * 640].reshape(1, 640, 640, 1)))
    eng._e.set_input(1, np.ascontiguousarray(nv12[640 * 640:].reshape(1, 320, 320, 2)))
    eng._e.infer(0)

    faces = det.postprocess(lb)
    assert len(faces) >= 1
    for f in faces:
        assert f.score >= 0.5
        assert f.x2 > f.x1 and f.y2 > f.y1
        assert len(f.landmarks) == 5
        le, re_, nose, ml, mr = f.landmarks
        # Geometry, not just presence: eyes above the nose, nose above the mouth,
        # right of each pair to the right. A transposed or mis-strided landmark
        # read still yields five plausible-looking points that fail this.
        assert re_[0] > le[0], "right eye must be right of left eye"
        assert nose[1] > le[1] and nose[1] > re_[1], "nose below the eye line"
        assert ml[1] > nose[1] and mr[1] > nose[1], "mouth below the nose"
        assert mr[0] > ml[0], "mouth corners ordered left to right"

    crop = bcdl.align_face(img, faces[0].landmarks, 112)
    assert crop.shape == (112, 112, 3) and crop.dtype == np.uint8
    assert crop.std() > 5, "aligned crop is flat — the warp landed off the face"

    rec_model = os.environ.get(
        "BCDL_FACE_REC_MODEL",
        os.path.join(bm.MODELS, "arcface_r50_aligned_nashm_112x112.hbm"))
    if not os.path.exists(rec_model) or not hasattr(bcdl, "ImageEmbedder"):
        pytest.skip("face recognition model not deployed")
    rec = bcdl.Engine(rec_model)
    emb = bcdl.ImageEmbedder(rec)
    assert emb.dim == 512

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
    v = np.asarray(emb.embed(np.ascontiguousarray(((rgb - 127.5) / 127.5)[None])),
                   np.float32)
    assert v.shape == (512,)
    assert np.all(np.isfinite(v))
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


def test_panoptic_driving_three_heads():
    """Panoptic driving model: one inference, three decoders.

    This is the multi-head shape the rest of the suite does not cover — an
    anchor-BASED detector reading three raw head tensors, plus two segmentation
    masks, off a single Engine. The published export bakes the anchor decode
    into the graph, but that decode is built from ScatterND and does not survive
    BPU compilation (the compiled tensor's objectness/class columns are never
    written, so a detector finds nothing at any threshold). The graph is cut
    before it and AnchorDetector does the arithmetic on the CPU — that path is
    what this pins."""
    import os

    import cv2

    # "_cut": the published export bakes the anchor decode into the graph via
    # ScatterND, which compiles without error and produces a tensor whose
    # objectness and class columns are never written. The usable build is the one
    # cut before that decode, emitting the three raw heads.
    model = os.environ.get("BCDL_YOLOP_MODEL",
                           os.path.join(bm.MODELS, "yolop_cut_nashm_640x640_nv12.hbm"))
    if not os.path.exists(model):
        pytest.skip(f"panoptic driving model not deployed: {model}")
    for name in ("AnchorDetector", "AnchorDetectConfig", "Anchor"):
        if not hasattr(bcdl, name):
            pytest.skip(f"{name} not exposed")

    img_path = bm.find_image("bus.jpg")
    if not os.path.exists(img_path):
        pytest.skip(f"missing test image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        pytest.skip(f"unreadable test image: {img_path}")

    eng = bcdl.Engine(model)
    assert eng.num_outputs == 5           # 3 raw det heads + drivable + lane

    cfg = bcdl.AnchorDetectConfig()
    cfg.num_classes = 1                   # vehicles only
    # A LOW threshold on purpose. This model is trained on dashcam driving
    # frames, and bus.jpg is a close-up street photo — far out of domain, and its
    # bus fills the frame while the priors top out at 68x157 px, so confidences
    # stay low (nothing at 0.35, one box at 0.1). That is fine for what this
    # pins: the broken in-graph-decode tensor returned ZERO at every threshold
    # down to 0.01, so any detection at all distinguishes the two. Detection
    # QUALITY is measured off-board against the float reference on real driving
    # frames, not here.
    cfg.conf_thresh = 0.05
    cfg.strides = [8, 16, 32]
    cfg.anchors = [
        [bcdl.Anchor(3, 9), bcdl.Anchor(5, 11), bcdl.Anchor(4, 20)],
        [bcdl.Anchor(7, 18), bcdl.Anchor(6, 39), bcdl.Anchor(12, 31)],
        [bcdl.Anchor(19, 50), bcdl.Anchor(38, 81), bcdl.Anchor(68, 157)],
    ]
    det = bcdl.AnchorDetector(eng, cfg, 0)
    scfg = bcdl.SegConfig(); scfg.num_classes = 2; scfg.channels_first = True
    drive = bcdl.Segmenter(eng, scfg, 3)
    lane = bcdl.Segmenter(eng, scfg, 4)

    lbimg, lb = bcdl.letterbox_numpy(img, 640, 640)
    nv12 = np.asarray(bcdl.bgr_to_nv12(lbimg)).reshape(-1)
    eng._e.set_input(0, np.ascontiguousarray(nv12[:640 * 640].reshape(1, 640, 640, 1)))
    eng._e.set_input(1, np.ascontiguousarray(nv12[640 * 640:].reshape(1, 320, 320, 2)))
    eng._e.infer(0)

    dets = det.postprocess(lb)
    assert len(dets) >= 1, "zero detections at any threshold = the in-graph-decode symptom"
    for d in dets:
        assert 0.05 <= d.score <= 1.0
        assert d.class_id == 0
        assert d.x2 > d.x1 and d.y2 > d.y1
        assert 0 <= d.x1 <= img.shape[1] and 0 <= d.y1 <= img.shape[0]

    for seg, name in ((drive.postprocess(), "drivable"), (lane.postprocess(), "lane")):
        assert (seg.width, seg.height) == (640, 640)
        labels = np.asarray(seg.labels)
        assert labels.shape == (640, 640)
        ids = np.unique(labels)
        assert ids.min() >= 0 and ids.max() < 2, f"{name} is a 2-class mask"


def test_depth_estimation():
    t = _task("depth")
    dm = t.decode()
    assert (dm.width, dm.height) == (t.in_w, t.in_h)      # 686x518
    assert np.asarray(dm.data).shape == (t.in_h, t.in_w)
    assert dm.vmax > dm.vmin                              # a real depth range


def test_ocr_two_stage():
    t = _task("ocr")
    lines = t.decode()
    assert len(lines) >= 1                                # ocr.jpg has text
    assert any(text for _pts, text, _score in lines)     # at least one non-empty
    # boxes are 4-point polygons inside the image
    for pts, _text, _score in lines:
        assert np.asarray(pts).shape == (4, 2)


def test_ocr_direction_cls():
    """Direction classifier (3rd OCR stage): an upright text line must NOT be
    flagged for a 180° flip, while the same line rotated 180° MUST be — proving
    the cls .hbm distinguishes orientation. Skips if the cls model isn't deployed."""
    import os
    import cv2

    t = _task("ocr")
    if t.angler is None or not os.path.exists(bm.OCR_CLS):
        pytest.skip("OCR direction-cls model not deployed (BCDL_OCR_CLS)")

    # A real, clear text-line crop from the detector (det-space -> original).
    t.feed()
    t.det._e.infer(0)
    boxes = t.detector.postprocess(t._lb)
    assert boxes, "no text detected to classify"
    sx, sy = t.orig_w / float(t.in_w), t.orig_h / float(t.in_h)
    pts = np.array(boxes[0].points, np.float32).reshape(4, 2) * (sx, sy)
    crop = bm._crop_and_rotate(t.img, pts)
    assert crop is not None and crop.size > 0

    up = t._cls_one(crop)
    down = t._cls_one(cv2.rotate(crop, cv2.ROTATE_180))
    assert not up.flip180, f"upright line flagged for flip (label={up.label}, {up.score:.2f})"
    assert down.flip180, f"upside-down line not flagged (label={down.label}, {down.score:.2f})"


def test_all_tasks_have_a_model_or_skip():
    # Visibility: report which tasks are runnable on this board (never fails).
    keys = list(bm.TASKS) + ["ocr"]
    have = [k for k in keys if bm.available(k)]
    print(f"\nrunnable board tasks: {have}")
    assert isinstance(have, list)


def test_wholebody_133_keypoints_on_the_real_model():
    """Top-down whole-body pose end to end: person detector -> crop -> ViTPose.

    Asserts the part LAYOUT rather than just the count, because the 133 slots are
    a fixed COCO-WholeBody ordering (body/feet/face/hands) and getting it wrong
    puts face landmarks on hands without changing any shape.
    """
    t = _task("wholebody")
    res = t.decode()
    assert len(res) >= 1, "no people detected in the test image"

    for person in res:
        assert len(person) == 133
        body = person[0:17]
        face = person[23:91]
        assert sum(kp.score > 0.2 for kp in body) >= 8, "body block barely resolved"
        assert sum(kp.score > 0.2 for kp in face) >= 40, "face block barely resolved"

        # Confident keypoints must land in the frame; low-confidence ones are
        # allowed to be anywhere, which is exactly why callers threshold them.
        for kp in person:
            if kp.score > 0.5:
                assert -2 <= kp.x <= t.orig_w + 2 and -2 <= kp.y <= t.orig_h + 2

        # The face block is a dense set on one face, so it must be far tighter
        # than the whole-body spread — the check that catches a scrambled layout.
        fx = np.array([kp.x for kp in face if kp.score > 0.2])
        fy = np.array([kp.y for kp in face if kp.score > 0.2])
        bx = np.array([kp.x for kp in body if kp.score > 0.2])
        assert np.ptp(fx) < np.ptp(bx), "face landmarks are not clustered on a face"
        assert np.ptp(fy) < t.orig_h * 0.5


def test_wholebody_crop_geometry_inverts():
    """A keypoint at the centre of the heatmap must map back into the person box
    the crop came from — the inverse map is the part with no shape to check it."""
    t = _task("wholebody")
    box = t.boxes[0]
    _, crop = bcdl.wholebody_preprocess(t.img, *[float(v) for v in box],
                                        t.in_w, t.in_h, t.est.config)
    hm = np.zeros((133, 64, 48), np.float32)
    hm[:, 32, 24] = 1.0
    kp = bcdl.decode_wholebody(hm, crop)
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    assert abs(kp[0].x - cx) < 0.15 * (box[2] - box[0])
    assert abs(kp[0].y - cy) < 0.15 * (box[3] - box[1])


def test_xfeat_features_match_across_a_real_pair():
    """Sparse features end to end on the real model, then matched across two
    views of one scene.

    The assertion that actually means something is EPIPOLAR CONSISTENCY, not the
    match count: a matcher that returns thousands of pairs is worthless if they
    are not geometrically possible, and a fundamental matrix is the constraint
    that holds for a rigid scene whether or not it is planar.
    """
    t = _task("features")
    fa, fb, m = t.decode()
    assert len(fa) > 500 and len(fb) > 500
    assert len(m) > 100, "almost nothing matched between two views of one scene"

    # Descriptors are unit length by construction — matching treats a dot
    # product as a cosine, so this is a precondition, not a nicety.
    d = fa.descriptors
    assert d.shape[1] == 64
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-4)

    pa = fa.xy[[x.a for x in m]]
    pb = fb.xy[[x.b for x in m]]
    assert (pa[:, 0] <= t.orig_w + 1).all() and (pa[:, 1] <= t.orig_h + 1).all()

    import cv2

    _, inl = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 2.0, 0.999, 10000)
    assert inl is not None, "could not fit epipolar geometry to the matches"
    assert inl.sum() / len(m) > 0.6, "matches are not geometrically consistent"


def test_xfeat_top_k_bounds_the_feature_count():
    """top_k is the knob that decides whether matching costs milliseconds or
    seconds, so it has to actually bound the output."""
    t = _task("features")
    cfg = bcdl.XfeatConfig()
    cfg.top_k = 256
    ext = bcdl.FeatureExtractor(t.engine, cfg)
    f = ext.extract(t.a)
    assert len(f) <= 256
    scores = [k.score for k in f.keypoints]
    assert scores == sorted(scores, reverse=True), "features are not score-ordered"


def test_superres_upscales_and_beats_its_input():
    """x4 upscaling end to end on the real model.

    The bar is deliberately not "beats bicubic on PSNR" — this is a perceptual
    GAN-trained upscaler, and on a CLEAN downscale a conservative blurry
    interpolation wins that metric by construction. What must hold is that the
    output is the right size, is not degenerate, and reconstructs the ground
    truth far better than its own low-resolution input does.
    """
    t = _task("superres")
    out = t.decode()
    s = t.sr.scale
    assert out.shape == (t.orig_h * s, t.orig_w * s, 3)
    assert out.dtype == np.uint8
    assert t.sr.last_tile_count >= 1

    import cv2

    nearest = cv2.resize(t.img, (t.hr.shape[1], t.hr.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    assert t._psnr(out, t.hr) > t._psnr(nearest, t.hr)
    assert 20.0 < t._psnr(out, t.hr) < 60.0, "output is degenerate"
    assert out.std() > 20, "output is nearly flat"


def test_superres_drives_a_second_model_with_no_code_change():
    """The tiler reads scale and tile size from the model, so a different
    upscaler is a path change and nothing else. Both shipped models go through
    the same SuperResolver — this is what makes that claim testable."""
    a = _task("superres")
    b = _task("superres_span")
    assert a.model_file != b.model_file
    assert a.sr.scale == b.sr.scale
    out = b.decode()
    assert out.shape == (b.orig_h * b.sr.scale, b.orig_w * b.sr.scale, 3)
    assert b._psnr(out, b.hr) > 20.0


def test_superres_overlap_does_not_change_the_image_size():
    """Tiling is an implementation detail: the result must not depend on it."""
    t = _task("superres")
    base = t.decode()
    cfg = bcdl.SuperResConfig()
    cfg.overlap = 0
    butt = bcdl.SuperResolver(t.engine, cfg).upscale(t.img)
    assert butt.shape == base.shape
    # Same picture, not the same pixels: the overlap changes how border pixels
    # are computed, but not by much.
    assert t._psnr(butt, base) > 30.0


def test_engine_model_names_lists_packed_models():
    """Engine.model_names() enumerates an .hbm package without loading it.

    A .hbm is a PACKAGE: the official SigLIP encoders ship a global-embedding
    submodel and a patch-feature submodel in one file, and Engine(path,
    model_name) needs a name to pick one. Before this there was no way to
    discover the names short of running `hrt_model_exec model_info` by hand.
    """
    if not bm.available("det"):
        pytest.skip("no detection model on this board")
    path = bm.model_path(bm.TASKS["det"]["model"])

    names = bcdl.Engine.model_names(path)
    assert isinstance(names, list) and names, "package reported no models"
    assert all(isinstance(n, str) and n for n in names)

    # The name the Engine actually resolves to must be one it advertised, and
    # the default (empty model_name) must be the first entry.
    eng = bcdl.Engine(path)
    assert eng.model_name in names
    assert eng.model_name == names[0]

    # Selecting that name explicitly gives the same model.
    assert bcdl.Engine(path, names[0]).model_name == names[0]


def test_reid_embeddings_separate_identities_on_the_real_model():
    """Person ReID end to end: detector boxes -> crops -> 512-d vectors.

    The assertion that means something is SEPARATION, not well-formedness. A
    tower degraded by quantization still emits unit-length, deterministic,
    correctly-shaped vectors — what it loses is the gap between "same person"
    and "different person", which is the only property the tracker consumes. So
    the checks are: a crop matches a shifted/darkened version of itself far more
    than it matches anyone else, and no two distinct people look identical.
    """
    import cv2

    t = _task("reid")
    embs = [np.asarray(v, np.float32) for v in t.decode() if len(v)]
    assert len(embs) >= 2, "need at least two people in the test image"
    assert all(e.shape == (t.ex.dim,) for e in embs)
    assert np.allclose([np.linalg.norm(e) for e in embs], 1.0, atol=1e-3)

    # Determinism: the same crop twice must give bit-identical vectors.
    again = np.asarray(t.ex.embed(t.x), np.float32)
    assert np.array_equal(again, np.asarray(t.ex.embed(t.x), np.float32))

    # Self-match under a nuisance change vs. cross-person similarity.
    d = t.dets[0]
    x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
    x2, y2 = min(t.orig_w, int(d.x2)), min(t.orig_h, int(d.y2))
    crop = t.img[y1:y2, x1:x2]
    # A 2 px shift plus a mild exposure change — the sort of difference the same
    # person produces between consecutive frames.
    nuisance = np.clip(
        cv2.warpAffine(crop, np.float32([[1, 0, 2], [0, 1, 2]]),
                       (crop.shape[1], crop.shape[0]),
                       borderMode=cv2.BORDER_REPLICATE) * 0.85, 0, 255).astype(np.uint8)
    self_sim = float(embs[0] @ np.asarray(t.ex.embed_crop(nuisance), np.float32))

    sim = np.stack(embs) @ np.stack(embs).T
    cross = sim[~np.eye(len(embs), dtype=bool)]
    print(f"\nreid: {len(embs)} people, self-similarity {self_sim:.3f}, "
          f"cross max {cross.max():.3f} mean {cross.mean():.3f}")

    assert self_sim > 0.9, "the same person under a 2 px shift stopped matching"
    assert cross.max() < self_sim - 0.15, \
        "two different people are as similar as one person is to themselves"

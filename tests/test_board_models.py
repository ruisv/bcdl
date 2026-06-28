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

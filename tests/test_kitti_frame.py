"""Calibration geometry for the lidar check figure.

Every failure this file guards against produces a figure that looks completely
convincing: boxes land somewhere on the image, just not on the objects. So the
tests check the geometry against things that are independently known — a
synthetic calibration whose answer can be worked out by hand, and, when a real
frame is staged, the fact that ground truth and the detector must agree with
each other about where the cars are.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_kitti_frame.py
"""
import os
import textwrap

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

import board_models as bm  # noqa: E402  (tests/ is on sys.path under pytest)
import kitti_frame as kf  # noqa: E402


def _synthetic(tmp_path):
    """A frame whose calibration is the identity apart from the axis swap, so
    the projection of a known point can be derived on paper."""
    d = tmp_path / "frame"
    d.mkdir()
    # Lidar (x fwd, y left, z up) -> camera (x right, y down, z fwd).
    tr = "0 -1 0 0  0 0 -1 0  1 0 0 0"
    calib = textwrap.dedent("""\
        P2: 500 0 320 0 0 500 240 0 0 0 1 0
        R0_rect: 1 0 0 0 1 0 0 0 1
        Tr_velo_to_cam: %s
        """ % tr)
    (d / "calib.txt").write_text(calib)
    np.zeros((4, 4), np.float32).tofile(str(d / "velodyne.bin"))
    import cv2
    cv2.imwrite(str(d / "image_2.png"), np.zeros((480, 640, 3), np.uint8))
    return str(d)


def test_a_point_straight_ahead_lands_at_the_principal_point(tmp_path):
    fr = kf.KittiFrame(_synthetic(tmp_path))
    uv, depth = fr.project(np.array([[10.0, 0.0, 0.0]]))
    assert depth[0] == pytest.approx(10.0)
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-6)


def test_left_is_left_and_up_is_up(tmp_path):
    """The sign errors that matter. A mirrored projection still puts boxes on
    cars — the ones on the other side of the road."""
    fr = kf.KittiFrame(_synthetic(tmp_path))
    uv, _ = fr.project(np.array([[10.0, 1.0, 0.0],      # 1 m to the LEFT
                                 [10.0, 0.0, 1.0]]))    # 1 m UP
    assert uv[0][0] < 320.0, "a point to the left must project left of centre"
    assert uv[1][1] < 240.0, "a point above must project above centre"


def test_box_corners_are_centred_on_z(tmp_path):
    """`z` is the box centre, not its base. Getting this wrong sinks every box
    half a car into the road, which looks like a plausible detector."""
    b = bcdl.Box3D()
    b.x, b.y, b.z, b.dx, b.dy, b.dz, b.heading = 10.0, 0.0, -1.0, 4.0, 2.0, 1.6, 0.0
    c = kf.KittiFrame.box_corners(b)
    assert c[:, 2].min() == pytest.approx(-1.8)
    assert c[:, 2].max() == pytest.approx(-0.2)
    assert c[:, 0].min() == pytest.approx(8.0)
    assert c[:, 0].max() == pytest.approx(12.0)


def test_heading_rotates_the_footprint(tmp_path):
    b = bcdl.Box3D()
    b.x, b.y, b.z, b.dx, b.dy, b.dz = 10.0, 0.0, 0.0, 4.0, 2.0, 1.6
    b.heading = np.pi / 2
    c = kf.KittiFrame.box_corners(b)
    # Turned a quarter turn, the 4 m length now runs along y.
    assert (c[:, 1].max() - c[:, 1].min()) == pytest.approx(4.0)
    assert (c[:, 0].max() - c[:, 0].min()) == pytest.approx(2.0)


# --- against a real staged frame -------------------------------------------

real = pytest.mark.skipif(not kf.KittiFrame.available(bm.LIDAR_FRAME),
                          reason="no calibrated frame staged (scripts/fetch_models.sh)")


@real
def test_labels_and_the_detector_agree_about_the_cars():
    """The strongest available check on the camera->lidar label conversion: the
    labels and the detector arrive at the same places by completely different
    routes, so if they agree, both are right."""
    fr = kf.KittiFrame(bm.LIDAR_FRAME)
    gts = fr.labels()
    assert gts, "the staged frame has no labels to check against"
    task = bm.LidarTask(bcdl)
    preds = task.decode()
    matched = 0
    for g in gts:
        for p in preds:
            if p.label == g.label and np.hypot(p.x - g.x, p.y - g.y) < 1.0:
                assert abs(g.dz - p.dz) < 1.0, "heights disagree by more than a metre"
                matched += 1
                break
    assert matched >= 3, ("only %d labels line up with a detection; the camera-to-"
                          "lidar conversion is the first suspect" % matched)


@real
def test_labels_project_inside_the_image():
    """A label describes something the camera photographed, so its projection
    has to land on the photograph."""
    fr = kf.KittiFrame(bm.LIDAR_FRAME)
    h, w = fr.image_size()
    inside = 0
    for g in fr.labels():
        uv, depth = fr.project(fr.box_corners(g))
        if depth.min() < 0.5:
            continue
        cu, cv_ = uv[:, 0].mean(), uv[:, 1].mean()
        if -w * 0.1 < cu < w * 1.1 and -h * 0.2 < cv_ < h * 1.2:
            inside += 1
    assert inside >= 3, "labels are not projecting onto the image they came from"


@real
def test_boxes_behind_the_car_are_not_scored_as_mistakes():
    """KITTI labels only the camera's view. Scoring a detection behind the
    vehicle against labels nobody wrote for it invents false positives."""
    fr = kf.KittiFrame(bm.LIDAR_FRAME)
    behind = bcdl.Box3D()
    behind.x, behind.y, behind.z = -20.0, 0.0, -1.0
    behind.dx, behind.dy, behind.dz, behind.heading = 4.0, 2.0, 1.6, 0.0
    assert not fr.in_camera(behind)
    ahead = bcdl.Box3D()
    ahead.x, ahead.y, ahead.z = 15.0, 0.0, -1.0
    ahead.dx, ahead.dy, ahead.dz, ahead.heading = 4.0, 2.0, 1.6, 0.0
    assert fr.in_camera(ahead)


@real
def test_the_figure_reports_a_score():
    """The figure renders and the scoring path runs end to end."""
    task = bm.LidarTask(bcdl)
    assert task.frame is not None
    fig = task.draw(task.decode())
    assert fig.shape[0] > 600 and fig.shape[1] > 1000

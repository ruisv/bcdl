"""On-board end-to-end driving test: DrivePlanner on the real .hbm.

There is no driving dataset on the board, so this cannot check the planned
trajectory against ground truth — and a trajectory that merely looks smooth
proves nothing, since the model emits a smooth curve for any input at all. What
it CAN check is that we are driving the model correctly, by asserting on the
relationships the contract implies:

  * path length tracks commanded speed (the ego status vector is packed right),
  * the route command steers the plan (its one-hot slots land where claimed),
  * blanking a sensor moves the output (both branches are actually wired in),
  * the BEV head produces a structured map rather than one flat class.

Those are exactly the properties that break when a tensor is transposed, a field
is off by one, or an input is quantized as raw float bytes — the failure modes
that otherwise produce a confident, wrong plan.

Runs only on the board; skips cleanly when the model or anchors are absent.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_drive_board_py.py

Assets come from scripts/fetch_models.sh. Env-overridable: BCDL_DRIVE_MODEL /
BCDL_DRIVE_ANCHORS / BCDL_MODELS / BCDL_IMAGES.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_IMAGES = os.environ.get("BCDL_IMAGES", os.path.join(_REPO, "data", "images"))


def _asset(env_key, name, root=_MODELS):
    p = os.environ.get(env_key) or os.path.join(root, name)
    return p if os.path.exists(p) else None


_MODEL = _asset("BCDL_DRIVE_MODEL", "diffusiondrive_r34_256x1024_s100p.hbm")
_ANCHORS = _asset("BCDL_DRIVE_ANCHORS", "kmeans_navsim_traj_20.npy")

pytestmark = pytest.mark.skipif(
    _MODEL is None or _ANCHORS is None,
    reason="diffusiondrive .hbm or anchor file missing (scripts/fetch_models.sh)")


def _road_image():
    """A driving-ish frame for the camera canvas. The board's Cityscapes sample
    is a real street scene; anything else still exercises the path."""
    for p in (os.path.join(_IMAGES, "bus.jpg"), "/app/res/assets/segmentation.png"):
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return np.full((512, 1024, 3), 96, np.uint8)


@pytest.fixture(scope="module")
def rig():
    engine = bcdl.Engine(_MODEL)
    anchors = np.load(_ANCHORS).astype(np.float32)
    cfg = bcdl.DriveAnchorConfig()
    cfg.noise_scale = 0.0  # deterministic: every comparison below is input-driven
    planner = bcdl.DrivePlanner(engine, anchors, anchor_config=cfg)

    # One frame stands in for all three cameras. The stitch geometry is wrong for
    # a real rig, which is fine here: every assertion compares two runs of the
    # same canvas, so the geometry cancels.
    img = _road_image()
    cam_h, cam_w = engine.input_shape(0)[2], engine.input_shape(0)[3]
    third = cv2.resize(img, (cam_w // 2, cam_h + 2))
    ccfg = bcdl.DriveCameraConfig()
    ccfg.crop_top = ccfg.crop_bottom = 1
    ccfg.crop_side = cam_w // 8
    ccfg.out_width, ccfg.out_height = cam_w, cam_h
    camera = bcdl.stitch_drive_cameras(third, third, third, ccfg)

    lidar_shape = engine.input_shape(1)[1:]
    return planner, engine, camera, np.zeros(lidar_shape, np.float32)


def _status(command=0, vx=5.0):
    s = bcdl.DriveStatus()
    s.driving_command = command
    s.velocity_x = vx
    return s


def _path_length(result):
    p = np.array([[q.x, q.y] for q in result.trajectory])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def test_shapes_and_plausible_trajectory(rig):
    planner, engine, camera, lidar = rig
    r = planner.plan(camera, lidar, _status())

    assert len(r.trajectory) == engine.output_shape(0)[1]
    # The horizon is the future, so x (forward) must advance, not wander.
    xs = [p.x for p in r.trajectory]
    assert all(b >= a - 0.5 for a, b in zip(xs, xs[1:])), xs
    assert 0.0 < xs[-1] < 100.0
    assert abs(r.trajectory[-1].y) < 30.0
    assert r.bev.height > 0 and r.bev.width > 0


def test_path_length_tracks_commanded_speed(rig):
    planner, _, camera, lidar = rig
    lengths = [_path_length(planner.plan(camera, lidar, _status(vx=v)))
               for v in (0.0, 3.0, 8.0, 15.0)]
    assert all(b > a for a, b in zip(lengths, lengths[1:])), lengths
    # 4 s of horizon: the fastest plan has to cover meaningfully more ground than
    # the standing one, or the velocity slots are not being read.
    assert lengths[-1] > 3 * max(lengths[0], 1e-3)


def test_route_command_changes_the_plan(rig):
    planner, _, camera, lidar = rig
    plans = [planner.plan(camera, lidar, _status(command=c)) for c in range(4)]
    base = np.array([[p.x, p.y] for p in plans[0].trajectory])
    deltas = [float(np.abs(np.array([[p.x, p.y] for p in plan.trajectory]) - base).max())
              for plan in plans[1:]]
    assert max(deltas) > 0.1, deltas


def test_both_sensor_branches_are_wired(rig):
    planner, _, camera, lidar = rig
    base = planner.plan(camera, lidar, _status())
    base_xy = np.array([[p.x, p.y] for p in base.trajectory])

    def moved(result):
        return float(np.abs(np.array([[p.x, p.y] for p in result.trajectory]) - base_xy).max())

    blank_cam = planner.plan(np.zeros_like(camera), lidar, _status())
    rng = np.random.default_rng(0)
    noisy_lidar = planner.plan(camera, rng.random(lidar.shape).astype(np.float32), _status())
    assert moved(blank_cam) > 0.1, "camera branch does not affect the plan"
    assert moved(noisy_lidar) > 0.1, "lidar branch does not affect the plan"


def test_bev_map_has_structure(rig):
    planner, _, camera, lidar = rig
    r = planner.plan(camera, lidar, _status())
    counts = np.bincount(r.bev.labels.ravel(), minlength=r.bev.num_classes)
    assert (counts > 0).sum() >= 2, counts
    assert counts.max() < 0.99 * counts.sum(), counts


def test_deterministic_for_identical_input(rig):
    planner, _, camera, lidar = rig
    a = planner.plan(camera, lidar, _status())
    b = planner.plan(camera, lidar, _status())
    np.testing.assert_array_equal([[p.x, p.y, p.heading] for p in a.trajectory],
                                  [[p.x, p.y, p.heading] for p in b.trajectory])

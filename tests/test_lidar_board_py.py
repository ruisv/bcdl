"""On-board lidar 3-D detection: the whole PointPillars chain on a real sweep.

The numpy tests next door pin the arithmetic (pillar encoding, the anchor grid,
the decode). This one answers the question they cannot: does the compiled chain
put boxes on the objects. The gate is deliberately geometric rather than a
golden box list — the two supported chains are different weights and disagree by
a couple of boxes, but neither is allowed to hallucinate a car in empty air.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_lidar_board_py.py

Models and the sample sweep come from scripts/fetch_models.sh; the faster
re-exported chain is picked up automatically when it is present (see
docs/MODELS.md). Env-overridable: BCDL_LIDAR_VFE / BCDL_LIDAR_HEAD.
"""
import math

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

import board_models as bm  # noqa: E402  (tests/ is on sys.path under pytest)

pytestmark = pytest.mark.skipif(not bm.available("lidar3d"),
                                reason="lidar .hbm or sample sweep missing "
                                       "(scripts/fetch_models.sh)")


@pytest.fixture(scope="module")
def rig():
    task = bm.LidarTask(bcdl)
    return task, task.decode()


def test_the_sample_sweep_encodes_into_pillars(rig):
    task, _ = rig
    st = task.stats
    assert st.num_pillars > 1000
    assert st.points_used > 10000
    # Whatever the caps dropped, they were reported rather than swallowed.
    assert st.points_used + st.points_out_of_range + st.points_dropped_full <= len(task.points)


def test_boxes_are_found_and_are_car_shaped(rig):
    _, boxes = rig
    assert len(boxes) >= 8, "a KITTI street sweep should not come back nearly empty"
    cars = [b for b in boxes if b.label == 0]
    assert len(cars) >= 6
    for b in cars:
        assert 2.5 < b.dx < 6.0, f"car length {b.dx:.2f} m"
        assert 1.2 < b.dy < 2.4, f"car width {b.dy:.2f} m"
        assert 1.0 < b.dz < 2.5, f"car height {b.dz:.2f} m"
        assert -3.0 < b.z < 1.0, f"car centre {b.z:.2f} m above the sensor origin"


def test_every_box_sits_on_points(rig):
    """The failure this catches is a decode that is internally consistent and
    geometrically wrong — a permuted or half-pillar-shifted anchor grid produces
    well-formed boxes in the wrong places, and only the point cloud says so.

    The threshold has to fall off with range or it is not a test of the decode
    but of the sensor: a spinning lidar's returns thin out as 1/r², and the
    published chain's furthest detection here is a car at 62 m with 8 points
    under it, which is what a car at 62 m looks like."""
    task, boxes = rig
    pts = task.points
    for b in boxes:
        c, s = math.cos(-b.heading), math.sin(-b.heading)
        dx, dy = pts[:, 0] - b.x, pts[:, 1] - b.y
        # into the box's own frame, then a slightly generous footprint
        u, v = c * dx - s * dy, s * dx + c * dy
        inside = int(((np.abs(u) < b.dx / 2 + 0.3) & (np.abs(v) < b.dy / 2 + 0.3)).sum())
        floor = 40 if math.hypot(b.x, b.y) < 20.0 else 5
        assert inside >= floor, (
            f"box at ({b.x:.1f}, {b.y:.1f}) score {b.score:.2f} has only "
            f"{inside} points under it (needs {floor} at this range)")


def test_scores_are_probabilities_and_sorted_by_confidence(rig):
    _, boxes = rig
    assert all(0.0 <= b.score <= 1.0 for b in boxes)
    assert max(b.score for b in boxes) > 0.5


def test_the_chain_reports_which_form_it_is_running(rig):
    """Both a published and a re-exported chain are supported, dispatched on the
    shapes the models declare. Whichever is installed, the two flags have to
    agree with the models actually loaded — a mismatch means the VFE was fed the
    wrong tensors, which produces plausible garbage rather than an error."""
    task, _ = rig
    assert task.vfe_conv == (task.vfe.num_inputs == 1)
    assert (task.backbone is None) == task.fused_head
    if task.vfe_conv:
        assert task.vfe.input_shape(0)[1] == task.cfg.num_features + 6

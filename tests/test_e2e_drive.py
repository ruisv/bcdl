"""End-to-end driving task: numpy path, no model required.

    PYTHONPATH=build:python pytest -s tests/test_e2e_drive.py

Covers the two halves that a running model cannot check for you: the sensor
tensors are geometric contracts (a transposed BEV grid or a mis-ordered status
vector still infers, it just plans for a different world), and the anchor
normalization is part of the trained contract rather than a preference.
"""
import numpy as np
import pytest

import bcdl


# --- decode ----------------------------------------------------------------

def test_decode_trajectory_passes_through():
    traj = np.array([[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]]], dtype=np.float32)
    r = bcdl.decode_drive(trajectory=traj)
    assert len(r.trajectory) == 2
    assert r.trajectory[0].x == pytest.approx(1.0)
    assert r.trajectory[1].y == pytest.approx(4.0)
    assert r.trajectory[1].heading == pytest.approx(0.2)


def test_decode_agents_filtered_by_label_logits():
    states = np.array([[[1.0, 2.0, 0.5, 4.0, 2.0],
                        [9.0, 9.0, 0.0, 4.5, 2.0],
                        [3.0, 1.0, -0.5, 4.2, 1.9]]], dtype=np.float32)
    # sigmoid: +4 -> 0.98 (keep), -4 -> 0.018 (drop), 0 -> 0.5 (keep at thresh 0.5)
    labels = np.array([[4.0, -4.0, 0.0]], dtype=np.float32)
    r = bcdl.decode_drive(agent_states=states, agent_labels=labels)
    assert len(r.agents) == 2
    assert r.agents[0].x == pytest.approx(1.0)
    assert r.agents[0].length == pytest.approx(4.0)
    assert r.agents[0].score > 0.9
    assert r.agents[1].x == pytest.approx(3.0)

    cfg = bcdl.DriveConfig()
    cfg.agent_score_thresh = 0.9
    assert len(bcdl.decode_drive(agent_states=states, agent_labels=labels, config=cfg).agents) == 1


def test_decode_agents_without_labels_keeps_every_slot():
    states = np.zeros((1, 3, 5), dtype=np.float32)
    assert len(bcdl.decode_drive(agent_states=states).agents) == 3


def test_decode_bev_is_a_segmask():
    bev = np.zeros((1, 3, 2, 4), dtype=np.float32)
    bev[0, 2, 1, 1] = 5.0  # class 2 wins that one pixel
    r = bcdl.decode_drive(bev=bev)
    assert (r.bev.height, r.bev.width) == (2, 4)
    assert r.bev.labels[1, 1] == 2
    assert r.bev.labels[0, 0] == 0


def test_decode_rejects_mismatched_label_count():
    with pytest.raises(Exception):
        bcdl.decode_drive(agent_states=np.zeros((1, 3, 5), np.float32),
                          agent_labels=np.zeros((1, 2), np.float32))


def test_decode_rejects_wrong_trajectory_width():
    with pytest.raises(Exception):
        bcdl.decode_drive(trajectory=np.zeros((1, 8, 2), np.float32))


# --- status vector ---------------------------------------------------------

def test_status_vector_layout():
    s = bcdl.DriveStatus()
    s.driving_command = 2
    s.velocity_x, s.velocity_y = 5.0, 0.5
    s.accel_x, s.accel_y = -1.0, 0.25
    v = bcdl.drive_status_vector(s)
    assert v.shape == (8,)
    np.testing.assert_allclose(v[:4], [0, 0, 1, 0])
    np.testing.assert_allclose(v[4:], [5.0, 0.5, -1.0, 0.25])


def test_status_vector_rejects_out_of_range_command():
    s = bcdl.DriveStatus()
    s.driving_command = 4
    with pytest.raises(Exception):
        bcdl.drive_status_vector(s, 4)


# --- anchors ---------------------------------------------------------------

def _norm_odo_xy(a):
    out = np.empty_like(a, dtype=np.float64)
    out[..., 0] = 2 * (a[..., 0] + 1.2) / 56.9 - 1
    out[..., 1] = 2 * (a[..., 1] + 20) / 46 - 1
    return out


def _anchors():
    rng = np.random.default_rng(7)
    a = np.empty((4, 8, 2), np.float32)
    a[..., 0] = rng.uniform(0.0, 48.0, (4, 8))
    a[..., 1] = rng.uniform(-10.0, 12.0, (4, 8))
    return a


def _alpha_bar(t=8, num_train=1000, beta_start=1e-4, beta_end=0.02):
    """diffusers' "scaled_linear" schedule: betas are linear in sqrt space."""
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train) ** 2
    return float(np.prod(1.0 - betas[:t + 1]))


def test_anchor_noise_deterministic_matches_upstream_normalization():
    # noise_scale=0 is the sampler with its noise term zeroed — the schedule's
    # own scaling stays, so this checks the normalization AND the beta schedule.
    a = _anchors()
    cfg = bcdl.DriveAnchorConfig()
    cfg.noise_scale = 0.0
    expected = _norm_odo_xy(a) * np.sqrt(_alpha_bar())
    np.testing.assert_allclose(bcdl.drive_anchor_noise(a, cfg), expected, rtol=0, atol=1e-6)


def test_anchor_noise_is_seeded_and_bounded():
    a = _anchors()
    base = _norm_odo_xy(a)

    cfg = bcdl.DriveAnchorConfig()
    cfg.seed = 42
    first = bcdl.drive_anchor_noise(a, cfg)
    second = bcdl.drive_anchor_noise(a, cfg)
    np.testing.assert_array_equal(first, second)  # same seed => same tensor

    cfg.seed = 43
    assert not np.array_equal(first, bcdl.drive_anchor_noise(a, cfg))

    # Truncating the schedule at t=8 of 1000 is a small perturbation: the anchors
    # must still be recognizably themselves, or the decoder is being started from
    # the wrong distribution.
    assert np.abs(first - base).max() < 0.2
    assert np.corrcoef(first.ravel(), base.ravel())[0, 1] > 0.99


def test_anchor_noise_rejects_bad_timestep():
    cfg = bcdl.DriveAnchorConfig()
    cfg.truncated_timestep = 1000
    with pytest.raises(Exception):
        bcdl.drive_anchor_noise(_anchors(), cfg)


# --- lidar BEV -------------------------------------------------------------

def _bev_cfg(**kw):
    cfg = bcdl.LidarBevConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_lidar_bev_axis_order_is_x_then_y():
    # x forward is grid axis 0, y left is axis 1. Getting this backwards is the
    # single easiest way to build a model that runs and steers into the scenery.
    cfg = _bev_cfg(min_x=-2, max_x=2, min_y=-2, max_y=2, pixels_per_meter=1)
    pts = np.array([[1.5, -1.5, 1.0]], np.float32)  # x=+1.5 (front), y=-1.5 (right)
    g = bcdl.lidar_bev_histogram(pts, cfg)
    assert g.shape == (1, 4, 4)
    assert g[0, 3, 0] > 0  # last x row, first y column
    assert g.sum() == pytest.approx(g[0, 3, 0])


def test_lidar_bev_counts_clamp_and_scale():
    cfg = _bev_cfg(min_x=-2, max_x=2, min_y=-2, max_y=2, pixels_per_meter=1,
                   hist_max_per_pixel=5)
    pts = np.tile(np.array([[0.5, 0.5, 1.0]], np.float32), (3, 1))
    np.testing.assert_allclose(bcdl.lidar_bev_histogram(pts, cfg).max(), 3 / 5)
    pts = np.tile(np.array([[0.5, 0.5, 1.0]], np.float32), (9, 1))
    np.testing.assert_allclose(bcdl.lidar_bev_histogram(pts, cfg).max(), 1.0)


def test_lidar_bev_height_split_and_ground_plane():
    cfg = _bev_cfg(min_x=-2, max_x=2, min_y=-2, max_y=2, pixels_per_meter=1,
                   split_height=0.2)
    pts = np.array([[0.5, 0.5, 0.1],    # ground band
                    [0.5, 0.5, 1.0]], np.float32)  # above
    single = bcdl.lidar_bev_histogram(pts, cfg)
    assert single.shape[0] == 1
    assert single.sum() == pytest.approx(1 / 5)  # only the above point survives

    cfg.use_ground_plane = True
    both = bcdl.lidar_bev_histogram(pts, cfg)
    assert both.shape[0] == 2
    assert both[0].sum() == pytest.approx(1 / 5)  # channel 0 = below
    assert both[1].sum() == pytest.approx(1 / 5)  # channel 1 = above


def test_lidar_bev_drops_out_of_box_and_too_high():
    cfg = _bev_cfg(min_x=-2, max_x=2, min_y=-2, max_y=2, pixels_per_meter=1,
                   max_height=3.0)
    pts = np.array([[99.0, 0.0, 1.0],
                    [0.0, -99.0, 1.0],
                    [0.5, 0.5, 9.0]], np.float32)
    assert bcdl.lidar_bev_histogram(pts, cfg).sum() == 0.0


def test_lidar_bev_right_edge_is_inclusive():
    cfg = _bev_cfg(min_x=-2, max_x=2, min_y=-2, max_y=2, pixels_per_meter=1)
    g = bcdl.lidar_bev_histogram(np.array([[2.0, 2.0, 1.0]], np.float32), cfg)
    assert g[0, 3, 3] > 0


# --- camera stitch ---------------------------------------------------------

def test_stitch_crops_concatenates_and_converts_to_rgb():
    # 8x8 cameras, 1px top/bottom crop, 2px side crop on the angled pair:
    # widths 4 | 8 | 4 = 16, height 6 — sized so the resize is an identity and
    # the colors survive exactly.
    cfg = bcdl.DriveCameraConfig()
    cfg.crop_top = cfg.crop_bottom = 1
    cfg.crop_side = 2
    cfg.out_width, cfg.out_height = 16, 6

    def solid(bgr):
        img = np.zeros((8, 8, 3), np.uint8)
        img[:, :] = bgr
        return img

    out = bcdl.stitch_drive_cameras(solid((255, 0, 0)),    # blue  (BGR)
                                    solid((0, 255, 0)),    # green
                                    solid((0, 0, 255)),    # red
                                    cfg)
    assert out.shape == (3, 6, 16)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # RGB planes: left segment is pure blue => plane 2, front green => plane 1,
    # right red => plane 0.
    np.testing.assert_allclose(out[2, :, :4], 1.0)
    np.testing.assert_allclose(out[1, :, 4:12], 1.0)
    np.testing.assert_allclose(out[0, :, 12:], 1.0)


def test_stitch_rejects_impossible_crop():
    img = np.zeros((8, 8, 3), np.uint8)
    cfg = bcdl.DriveCameraConfig()
    cfg.crop_top = cfg.crop_bottom = 8
    with pytest.raises(Exception):
        bcdl.stitch_drive_cameras(img, img, img, cfg)

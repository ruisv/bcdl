"""Cross-check bcdl.decode_mono3d (SMOKE monocular-3D numpy path) against the
REAL SMOKE coder.

A host bundle (mono3d_ref_bundle.npz) carries, for N KITTI val frames: the board
RAW cls/reg logits ([N,3,96,320] / [N,8,96,320]), each frame's camera intrinsics
+ original (W,H), and the reference boxes decoded by upstream SMOKE's
SMOKECoder/PostProcessor (the ground truth). This test re-decodes the SAME raw
logits with bcdl.decode_mono3d and asserts the 3D boxes match.

Pure numpy path — needs only the compiled bcdl extension (no .hbm, no board), so
it runs anywhere the bundle is present; skips cleanly otherwise.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_mono3d_py.py

Bundle path: $BCDL_MONO3D_BUNDLE, else <repo>/mono3d_ref_bundle.npz. Produced by
host_toolchain dump_ref_bundle.py (upstream SMOKE, env dinov3-export).
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLE = os.environ.get("BCDL_MONO3D_BUNDLE", os.path.join(_REPO, "mono3d_ref_bundle.npz"))

# ref row layout (upstream SMOKE PostProcessor):
#   0:cls 1:alpha 2..6:box2d 6:h 7:w 8:l 9:x 10:y 11:z 12:roty 13:score
R_CLS, R_ALPHA, R_BOX, R_H, R_W, R_L, R_X, R_Y, R_Z, R_ROTY, R_SCORE = (
    0, 1, slice(2, 6), 6, 7, 8, 9, 10, 11, 12, 13)

_FEAT_W, _FEAT_H = 320, 96


@pytest.fixture(scope="module")
def bundle():
    if not os.path.exists(_BUNDLE):
        pytest.skip("mono3d reference bundle not found: %s" % _BUNDLE)
    return np.load(_BUNDLE, allow_pickle=True)


def _bcdl_decode(cls, reg, fx, fy, cx, cy, W, H):
    cfg = bcdl.Mono3dConfig()
    cfg.conf_thresh = 0.25
    cfg.pred_2d = True
    K = bcdl.CameraIntrinsics(float(fx), float(fy), float(cx), float(cy))
    xform = bcdl.compute_mono3d_feature_xform(int(W), int(H), _FEAT_W, _FEAT_H)
    return bcdl.decode_mono3d(np.ascontiguousarray(cls, np.float32),
                              np.ascontiguousarray(reg, np.float32), cfg, xform, K)


def test_feature_xform_matches_smoke_geometry(bundle):
    # SMOKE get_transfrom_matrix: scale-to-width, centered height. Verify the
    # forward map sends the image center -> feature center and width fills.
    W, H = int(bundle["size"][0][0]), int(bundle["size"][0][1])
    x = bcdl.compute_mono3d_feature_xform(W, H, _FEAT_W, _FEAT_H)
    assert x.scale == pytest.approx(_FEAT_W / W, rel=1e-6)
    assert x.pad_x == pytest.approx(0.0, abs=1e-6)
    # image center maps to feature center on x; y centered.
    assert x.fwd_x(W / 2.0) == pytest.approx(_FEAT_W / 2.0, abs=1e-4)
    assert x.fwd_y(H / 2.0) == pytest.approx(_FEAT_H / 2.0, abs=1e-4)


def test_decode_matches_host_reference(bundle):
    cls, reg = bundle["cls"], bundle["reg"]
    K, size, ref, count = bundle["K"], bundle["size"], bundle["ref"], bundle["ref_count"]
    ids = bundle["ids"]
    N = cls.shape[0]
    total = 0
    for i in range(N):
        fx, fy, cx, cy = K[i]
        W, H = size[i]
        got = _bcdl_decode(cls[i], reg[i], fx, fy, cx, cy, W, H)
        nref = int(count[i])

        # same number of detections above threshold
        assert len(got) == nref, (
            "id %s: bcdl %d boxes vs host %d" % (ids[i], len(got), nref))
        if nref == 0:
            continue

        # both lists are score-descending; match index-by-index
        got = sorted(got, key=lambda b: -b.score)
        rrows = ref[i][:nref]
        rrows = rrows[np.argsort(-rrows[:, R_SCORE])]

        for b, r in zip(got, rrows):
            assert b.class_id == int(round(r[R_CLS])), "id %s class mismatch" % ids[i]
            assert b.score == pytest.approx(float(r[R_SCORE]), abs=2e-3)
            # 3D location (meters) and dims
            assert b.x == pytest.approx(float(r[R_X]), abs=2e-2)
            assert b.y == pytest.approx(float(r[R_Y]), abs=2e-2)
            assert b.z == pytest.approx(float(r[R_Z]), abs=3e-2)
            assert b.h == pytest.approx(float(r[R_H]), abs=1e-2)
            assert b.w == pytest.approx(float(r[R_W]), abs=1e-2)
            assert b.l == pytest.approx(float(r[R_L]), abs=1e-2)
            # angles wrap at +-pi; compare via sin/cos distance
            dyaw = abs(np.arctan2(np.sin(b.yaw - r[R_ROTY]), np.cos(b.yaw - r[R_ROTY])))
            assert dyaw < 1.5e-2, "id %s yaw %.4f vs %.4f" % (ids[i], b.yaw, r[R_ROTY])
            dal = abs(np.arctan2(np.sin(b.alpha - r[R_ALPHA]), np.cos(b.alpha - r[R_ALPHA])))
            assert dal < 1.5e-2, "id %s alpha mismatch" % ids[i]
            # projected 2D box (pixels)
            for k, rb in enumerate(r[R_BOX]):
                assert b.box2d[k] == pytest.approx(float(rb), abs=1.0)
            total += 1
    assert total > 0
    print("\n[mono3d] %d frames, %d boxes match upstream SMOKE within tol" % (N, total))

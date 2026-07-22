"""ViTPose whole-body decode: bcdl's C++ against an independent numpy + cv2
oracle.

The decode is DARK-UDP — Gaussian-blur the heatmap, log it, and take one Newton
step from the argmax. bcdl blurs only a 13x13 window around each peak instead of
all 133 full maps, which is ~300x less arithmetic but only correct if it lands on
exactly the same numbers OpenCV's full-map blur would. That equivalence is what
most of this file checks, so the oracle below deliberately does it the slow,
obvious way with cv2.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_wholebody.py
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")
cv2 = pytest.importorskip("cv2")

IN_W, IN_H = 192, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _have():
    if not all(hasattr(bcdl, n) for n in ("decode_wholebody", "wholebody_preprocess",
                                          "WholeBodyConfig", "WholeBodyCrop")):
        pytest.skip("wholebody bindings not exposed")


# --------------------------------------------------------------------------- #
# Oracles — the reference maths, written the slow way                          #
# --------------------------------------------------------------------------- #
def oracle_decode(hm, crop, kernel=11):
    """argmax + DARK-UDP + the UDP inverse map, straight off the reference."""
    K, H, W = hm.shape
    flat = hm.reshape(K, -1)
    idx = flat.argmax(1)
    maxvals = flat.max(1)

    blurred = np.stack([cv2.GaussianBlur(hm[k], (kernel, kernel), 0) for k in range(K)])
    log = np.log(np.clip(blurred, 0.001, 50))

    def at(k, x, y):
        return log[k, min(max(y, 0), H - 1), min(max(x, 0), W - 1)]

    out = np.zeros((K, 3), np.float32)
    for k in range(K):
        if maxvals[k] > 0:
            cx, cy = int(idx[k] % W), int(idx[k] // W)
            x, y = float(cx), float(cy)
            c0 = at(k, cx, cy)
            dx = 0.5 * (at(k, cx + 1, cy) - at(k, cx - 1, cy))
            dy = 0.5 * (at(k, cx, cy + 1) - at(k, cx, cy - 1))
            dxx = at(k, cx + 1, cy) - 2 * c0 + at(k, cx - 1, cy)
            dyy = at(k, cx, cy + 1) - 2 * c0 + at(k, cx, cy - 1)
            dxy = 0.5 * (at(k, cx + 1, cy + 1) - at(k, cx + 1, cy) - at(k, cx, cy + 1)
                         + c0 + c0 - at(k, cx - 1, cy) - at(k, cx, cy - 1)
                         + at(k, cx - 1, cy - 1))
            eps = np.finfo(np.float32).eps
            hess = np.array([[dxx + eps, dxy], [dxy, dyy + eps]])
            if abs(np.linalg.det(hess)) > 1e-12:
                d = np.linalg.solve(hess, np.array([dx, dy]))
                x -= d[0]
                y -= d[1]
        else:
            x = y = -1.0

        sx = crop.padded_w / (W - 1)
        sy = crop.padded_h / (H - 1)
        out[k, 0] = x * sx + (crop.padded_w // 2) - crop.padded_w * 0.5 + crop.x1 - crop.pad_left
        out[k, 1] = y * sy + (crop.padded_h // 2) - crop.padded_h * 0.5 + crop.y1 - crop.pad_top
        out[k, 2] = max(0.0, maxvals[k])
    return out


def oracle_preprocess(bgr, box, box_pad=10):
    """Crop -> zero-pad to 3:4 -> resize -> /255 -> ImageNet z-score, the way the
    reference does it (which resizes the uint8 crop, not a float one)."""
    h, w = bgr.shape[:2]
    x1 = min(max(int(round(box[0])) - box_pad, 0), w)
    y1 = min(max(int(round(box[1])) - box_pad, 0), h)
    x2 = min(max(int(round(box[2])) + box_pad, 0), w)
    y2 = min(max(int(round(box[3])) + box_pad, 0), h)
    crop = bgr[y1:y2, x1:x2, ::-1]                      # BGR -> RGB
    ch, cw = crop.shape[:2]
    aspect = IN_W / IN_H
    pad_left = pad_top = 0
    if cw / ch < aspect:
        tw = int(aspect * ch)
        pad_left = (tw - cw) // 2
        crop = np.pad(crop, ((0, 0), (pad_left, tw - cw - pad_left), (0, 0)))
    else:
        th = int(cw / aspect)
        pad_top = (th - ch) // 2
        crop = np.pad(crop, ((pad_top, th - ch - pad_top), (0, 0), (0, 0)))
    padded_h, padded_w = crop.shape[:2]

    x = cv2.resize(crop, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    c = bcdl.WholeBodyCrop()
    c.x1, c.y1 = x1, y1
    c.pad_left, c.pad_top = pad_left, pad_top
    c.padded_w, c.padded_h = padded_w, padded_h
    return x, c


def _gauss_map(h, w, cx, cy, sigma=1.6, amp=1.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _crop(x1=0, y1=0, pad_left=0, pad_top=0, padded_w=48, padded_h=64):
    c = bcdl.WholeBodyCrop()
    c.x1, c.y1 = x1, y1
    c.pad_left, c.pad_top = pad_left, pad_top
    c.padded_w, c.padded_h = padded_w, padded_h
    return c


# --------------------------------------------------------------------------- #
# Decode                                                                       #
# --------------------------------------------------------------------------- #
# A padded extent of exactly 2*(grid-1) makes the UDP map a clean doubling:
# scale = padded/(grid-1) = 2, and the centre offset (padded//2 - padded/2) is 0
# because the extent is even. Both numbers below are then exact by construction.
# (An ODD extent carries a real -0.5 offset — that is the reference's integer
# division of the centre, not a rounding slip, so don't "simplify" to grid-1.)
_X2 = dict(padded_w=94, padded_h=126)


def test_known_by_construction_peak_maps_through_the_udp_scale():
    _have()
    hm = _gauss_map(64, 48, 20.0, 30.0)[None]
    kp = bcdl.decode_wholebody(hm.astype(np.float32), _crop(**_X2))
    assert len(kp) == 1
    assert kp[0].x == pytest.approx(40.0, abs=0.05)   # 20 * 2
    assert kp[0].y == pytest.approx(60.0, abs=0.05)   # 30 * 2
    assert kp[0].score == pytest.approx(1.0, abs=1e-5)


def test_subpixel_beats_plain_argmax():
    """The whole reason DARK exists: a peak sitting BETWEEN cells must decode
    nearer the true position than the cell it snapped to."""
    _have()
    hm = _gauss_map(64, 48, 20.4, 30.4)[None].astype(np.float32)
    kp = bcdl.decode_wholebody(hm, _crop(**_X2))
    assert abs(kp[0].x - 40.8) < abs(40.0 - 40.8)     # argmax alone gives 40.0
    assert abs(kp[0].y - 60.8) < abs(60.0 - 60.8)


def test_matches_the_cv2_oracle_on_random_heatmaps():
    """The equivalence that matters: bcdl's 13x13 windowed blur against OpenCV
    blurring every map in full."""
    _have()
    rng = np.random.default_rng(7)
    hm = np.stack([_gauss_map(64, 48, rng.uniform(2, 45), rng.uniform(2, 61),
                              sigma=rng.uniform(1.0, 3.0))
                   + 0.02 * rng.standard_normal((64, 48))
                   for _ in range(24)]).astype(np.float32)
    crop = _crop(x1=37, y1=12, pad_left=5, pad_top=0, padded_w=200, padded_h=266)

    got = bcdl.decode_wholebody(hm, crop)
    ref = oracle_decode(hm, crop)
    for i, kp in enumerate(got):
        assert kp.x == pytest.approx(float(ref[i, 0]), abs=2e-3)
        assert kp.y == pytest.approx(float(ref[i, 1]), abs=2e-3)
        assert kp.score == pytest.approx(float(ref[i, 2]), abs=1e-6)


def test_peaks_on_the_border_match_the_oracle():
    """Border peaks exercise the two DIFFERENT border rules that meet in this
    decode: reflect inside the blur, clamp when reading around the peak."""
    _have()
    corners = [(0, 0), (47, 0), (0, 63), (47, 63), (0, 30), (47, 30), (20, 0), (20, 63)]
    hm = np.stack([_gauss_map(64, 48, cx, cy) for cx, cy in corners]).astype(np.float32)
    crop = _crop(padded_w=47, padded_h=63)
    got = bcdl.decode_wholebody(hm, crop)
    ref = oracle_decode(hm, crop)
    for i, kp in enumerate(got):
        assert kp.x == pytest.approx(float(ref[i, 0]), abs=2e-3)
        assert kp.y == pytest.approx(float(ref[i, 1]), abs=2e-3)


def test_crop_geometry_maps_into_original_image_pixels():
    """A keypoint's original-image position must follow the crop's offsets: shift
    the box, and every keypoint shifts with it."""
    _have()
    hm = _gauss_map(64, 48, 24.0, 32.0)[None].astype(np.float32)
    a = bcdl.decode_wholebody(hm, _crop(x1=0, y1=0, padded_w=47, padded_h=63))
    b = bcdl.decode_wholebody(hm, _crop(x1=100, y1=50, padded_w=47, padded_h=63))
    assert b[0].x == pytest.approx(a[0].x + 100, abs=1e-3)
    assert b[0].y == pytest.approx(a[0].y + 50, abs=1e-3)


def test_padding_offset_is_subtracted_not_added():
    """The zero padding sits OUTSIDE the person box, so a keypoint in the padded
    frame maps to a SMALLER original-image coordinate."""
    _have()
    hm = _gauss_map(64, 48, 24.0, 32.0)[None].astype(np.float32)
    a = bcdl.decode_wholebody(hm, _crop(padded_w=47, padded_h=63))
    b = bcdl.decode_wholebody(hm, _crop(pad_left=9, pad_top=4, padded_w=47, padded_h=63))
    assert b[0].x == pytest.approx(a[0].x - 9, abs=1e-3)
    assert b[0].y == pytest.approx(a[0].y - 4, abs=1e-3)


def test_all_non_positive_heatmap_scores_zero():
    _have()
    hm = np.full((1, 64, 48), -0.5, np.float32)
    kp = bcdl.decode_wholebody(hm, _crop(padded_w=47, padded_h=63))
    assert kp[0].score == 0.0


def test_even_kernel_is_rejected():
    _have()
    cfg = bcdl.WholeBodyConfig()
    cfg.blur_kernel = 10
    with pytest.raises(Exception):
        bcdl.decode_wholebody(_gauss_map(64, 48, 10, 10)[None].astype(np.float32),
                              _crop(), cfg)


def test_batched_and_unbatched_heatmaps_agree():
    _have()
    hm = _gauss_map(64, 48, 20.0, 30.0)[None].astype(np.float32)
    c = _crop(padded_w=47, padded_h=63)
    a = bcdl.decode_wholebody(hm, c)
    b = bcdl.decode_wholebody(hm[None], c)
    assert a[0].x == pytest.approx(b[0].x) and a[0].y == pytest.approx(b[0].y)


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("box", [(60, 40, 140, 260),    # tall  -> pads horizontally
                                 (20, 100, 300, 200),   # wide  -> pads vertically
                                 (10, 10, 30, 40)])     # small, hits the image edge
def test_preprocess_geometry_and_pixels_match_the_oracle(box):
    _have()
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (320, 400, 3), dtype=np.uint8)

    got, crop = bcdl.wholebody_preprocess(img, *box)
    ref, refcrop = oracle_preprocess(img, box)

    assert (crop.x1, crop.y1) == (refcrop.x1, refcrop.y1)
    assert (crop.pad_left, crop.pad_top) == (refcrop.pad_left, refcrop.pad_top)
    assert (crop.padded_w, crop.padded_h) == (refcrop.padded_w, refcrop.padded_h)
    assert got.shape == (1, 3, IN_H, IN_W)

    # bcdl resizes in float; the reference resizes the uint8 crop, whose
    # fixed-point INTER_LINEAR rounds to whole grey levels first. The gap is
    # therefore sub-quantization-noise, not a geometry difference — which is what
    # the tight correlation bound below is really asserting.
    assert np.abs(got - ref).max() < 0.05
    assert np.corrcoef(got.ravel(), ref.ravel())[0, 1] > 0.9999


def test_preprocess_pads_with_zeros_not_edge_pixels():
    """A tall box on a saturated image: the padded columns must come back as the
    normalized value of BLACK, not a smear of the person."""
    _have()
    img = np.full((300, 300, 3), 255, np.uint8)
    got, crop = bcdl.wholebody_preprocess(img, 100, 20, 140, 280)
    assert crop.pad_left > 0 and crop.pad_top == 0
    black = (0.0 - MEAN) / STD
    assert got[0, 0, 128, 0] == pytest.approx(black[0], abs=1e-4)   # left pad column
    white = (1.0 - MEAN) / STD
    assert got[0, 0, 128, IN_W // 2] == pytest.approx(white[0], abs=1e-4)


def test_preprocess_rejects_an_empty_box():
    _have()
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(Exception):
        bcdl.wholebody_preprocess(img, 200, 200, 260, 280)

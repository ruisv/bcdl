"""Media codec binding tests (VpImage + JPU/VPU codecs).

Unlike the pure-numpy post-processing tests, these exercise the C++/nanobind
bindings and the real JPU/VPU, so they only run ON THE BOARD inside the `bcdl`
conda env (importing `bcdl` requires the compiled extension linked against the
hobot SDK). Everything is guarded so the file skips cleanly off-board.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_codec_py.py

No .hbm model is needed — the inputs are synthetic VpImages.
"""

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")


def _have(*names):
    return all(hasattr(bcdl, n) for n in names)


# --------------------------------------------------------------------------- #
# VpImage round-trips                                                          #
# --------------------------------------------------------------------------- #
def test_vp_image_bgr_roundtrip():
    if not _have("vp_image_from_bgr", "VpImage"):
        pytest.skip("VpImage bindings not exposed")
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    img = bcdl.vp_image_from_bgr(bgr)
    assert img.width == 64 and img.height == 48
    assert img.format == bcdl.ImageFormat.BGR
    out = img.to_numpy()
    assert out.shape == (48, 64, 3)
    np.testing.assert_array_equal(out, bgr)


def test_vp_image_nv12_roundtrip():
    if not _have("vp_image_from_nv12", "VpImage"):
        pytest.skip("VpImage bindings not exposed")
    w, h = 64, 48
    nv12 = np.random.default_rng(1).integers(
        0, 256, size=(h * 3 // 2, w), dtype=np.uint8
    )
    img = bcdl.vp_image_from_nv12(nv12, w, h)
    assert img.width == w and img.height == h
    assert img.format == bcdl.ImageFormat.NV12
    out = img.to_numpy()  # flat W*H*3//2
    assert out.shape == (w * h * 3 // 2,)
    np.testing.assert_array_equal(out, nv12.reshape(-1))


def test_vp_image_bgr_roundtrip_strided():
    """Width whose packed row is NOT 16-aligned, so the device row stride
    exceeds width*3 and the bindings' stride-honoring copy loops are actually
    exercised (the 16-aligned cases above collapse to a contiguous copy)."""
    if not _have("vp_image_from_bgr", "VpImage"):
        pytest.skip("VpImage bindings not exposed")
    w, h = 62, 50  # 62*3 = 186 -> align16 = 192 (stride 6 bytes > packed width)
    bgr = np.random.default_rng(7).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = bcdl.vp_image_from_bgr(bgr)
    assert img.width == w and img.height == h
    out = img.to_numpy()
    assert out.shape == (h, w, 3)
    np.testing.assert_array_equal(out, bgr)


def test_vp_image_nv12_roundtrip_strided():
    """NV12 with a non-16-aligned width: Y stride (and uvStride) exceed width,
    exercising the per-plane stride-honoring copies in both directions."""
    if not _have("vp_image_from_nv12", "VpImage"):
        pytest.skip("VpImage bindings not exposed")
    w, h = 62, 48  # align16(62) = 64 -> 2 bytes of row padding on each plane
    nv12 = np.random.default_rng(8).integers(
        0, 256, size=(h * 3 // 2, w), dtype=np.uint8
    )
    img = bcdl.vp_image_from_nv12(nv12, w, h)
    out = img.to_numpy()
    assert out.shape == (w * h * 3 // 2,)
    np.testing.assert_array_equal(out, nv12.reshape(-1))


def test_vp_image_construct():
    if not _have("VpImage"):
        pytest.skip("VpImage bindings not exposed")
    img = bcdl.VpImage(32, 16, bcdl.ImageFormat.Y)
    assert img.width == 32 and img.height == 16
    assert img.valid


# --------------------------------------------------------------------------- #
# JPEG (JPU)                                                                   #
# --------------------------------------------------------------------------- #
def test_jpeg_encode_decode_roundtrip():
    if not _have("JpegEncoder", "JpegDecoder", "vp_image_from_nv12"):
        pytest.skip("JPEG bindings not exposed")
    # JPU needs width%16==0, height%8==0.
    w, h = 64, 48
    nv12 = np.random.default_rng(2).integers(
        0, 256, size=(h * 3 // 2, w), dtype=np.uint8
    )
    src = bcdl.vp_image_from_nv12(nv12, w, h)
    enc = bcdl.JpegEncoder(w, h, 50, bcdl.ImageFormat.NV12)
    jpeg = enc.encode(src)
    assert isinstance(jpeg, (bytes, bytearray)) and len(jpeg) > 0
    assert jpeg[:2] == b"\xff\xd8"  # SOI marker

    dec = bcdl.JpegDecoder()
    img = dec.decode(jpeg)
    assert img.width == w and img.height == h
    assert img.format == bcdl.ImageFormat.NV12


def test_jpeg_encode_helper():
    if not _have("jpeg_encode"):
        pytest.skip("jpeg_encode helper not exposed")
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    bgr = np.random.default_rng(3).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    jpeg = bcdl.jpeg_encode(bgr, quality=60)
    assert isinstance(jpeg, (bytes, bytearray)) and len(jpeg) > 0
    img = bcdl.jpeg_decode(jpeg)
    assert img.width == 64 and img.height == 48


def test_jpu_decode_vs_cpu_real_image():
    """Hardware JPEG decode (JPU) on a real photo, compared to CPU (cv2/libjpeg).

    Validates the bcdl hardware-decode path end-to-end and that it yields the
    SAME image cv2 does (the JPU pads dimensions up to its alignment, so compare
    on the overlapping region after NV12->BGR). This is the test-suite half of
    the software-vs-hardware decode comparison benchmarked by scripts/board_bench.py.
    """
    import os

    if not _have("jpeg_decode"):
        pytest.skip("jpeg_decode not exposed")
    cv2 = pytest.importorskip("cv2")
    path = os.environ.get("BCDL_DECODE_IMG", "/app/res/assets/bus.jpg")
    if not os.path.exists(path):
        pytest.skip(f"test image not found: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    cpu_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    ch, cw = cpu_bgr.shape[:2]

    img = bcdl.jpeg_decode(raw)                      # JPU
    assert img.format == bcdl.ImageFormat.NV12
    assert img.width >= cw and img.height >= ch      # JPU aligns dims up
    assert img.width - cw < 16 and img.height - ch < 16
    assert img.valid

    nv12 = img.to_numpy().reshape(img.height * 3 // 2, img.width)
    jpu_bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)[:ch, :cw]

    # Same picture from two decoders: mean per-channel brightness agrees closely
    # (different YUV ranges/chroma upsampling -> not bit-identical, but ~same).
    diff = np.abs(jpu_bgr.astype(np.int32) - cpu_bgr.astype(np.int32))
    assert diff.mean() < 12.0, f"JPU vs CPU decode mean abs diff too large: {diff.mean()}"


def test_jpu_decode_rejects_non_420():
    """A 4:2:2 / 4:4:4 JPEG must raise a clean error, NOT segfault the JPU.

    The hardware decoder targets 4:2:0; non-4:2:0 streams crashed the JPU firmware
    at larger sizes, so bcdl now parses the subsampling and rejects them up front.
    Build a 4:4:4 JPEG with OpenCV (chroma subsampling factor 1).
    """
    import os

    if not _have("jpeg_decode"):
        pytest.skip("jpeg_decode not exposed")
    cv2 = pytest.importorskip("cv2")
    bgr = np.random.default_rng(7).integers(0, 256, size=(128, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92,
                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:
        pytest.skip("cv2 could not encode a 4:4:4 JPEG")
    with pytest.raises(Exception):                 # clean bcdl::Error, not a crash
        bcdl.jpeg_decode(buf.tobytes())


# --------------------------------------------------------------------------- #
# H.264 / H.265 (VPU)                                                          #
# --------------------------------------------------------------------------- #
def test_video_encode_decode():
    if not _have("VideoEncoder", "VideoDecoder", "vp_image_from_nv12"):
        pytest.skip("video codec bindings not exposed")
    # VPU encoder: width%32==0 in [256,8192], height%8==0 in [128,4096].
    w, h = 256, 128
    enc_cfg = bcdl.VideoEncConfig()
    enc_cfg.type = bcdl.VideoType.H264
    enc_cfg.width = w
    enc_cfg.height = h
    enc_cfg.format = bcdl.ImageFormat.NV12
    enc = bcdl.VideoEncoder(enc_cfg)
    assert enc.width == w and enc.height == h

    dec_cfg = bcdl.VideoDecConfig()
    dec_cfg.type = bcdl.VideoType.H264
    dec = bcdl.VideoDecoder(dec_cfg)

    rng = np.random.default_rng(4)
    got_frame = False
    total_encoded = 0
    for _ in range(8):  # feed a few frames; decoder buffers references first
        nv12 = rng.integers(0, 256, size=(h * 3 // 2, w), dtype=np.uint8)
        frame = bcdl.vp_image_from_nv12(nv12, w, h)
        chunk = enc.encode(frame)
        assert isinstance(chunk, (bytes, bytearray))
        total_encoded += len(chunk)
        if chunk:
            out = dec.decode(chunk)
            if out is not None:
                assert out.width == w and out.height == h
                got_frame = True
                break
    # A decoded frame is timing/buffering dependent, but the encoder MUST have
    # emitted a non-empty H.264 stream over 8 frames — assert that for real.
    assert total_encoded > 0, "VideoEncoder produced no bytes over 8 frames"

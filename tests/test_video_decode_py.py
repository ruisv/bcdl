"""VPU hardware video-decode test on a REAL Annex-B elementary stream.

This is the test-suite half of `examples/video_decode.cc` — the first step of
the "live camera -> pipeline" path (a video file standing in for live capture).
It splits a board `.h264` into access units, feeds them to the VPU decoder, and
asserts real NV12 frames come back. Runs ON THE BOARD only (needs the compiled
`bcdl` extension + /dev/vpu).

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_video_decode_py.py

Set BCDL_VIDEO to override the input clip.
"""

import os

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

# Board-shipped Annex-B H.264 clips, in preference order.
_CANDIDATES = [
    "/app/res/assets/1080P_test.h264",
    "/app/multimedia_samples/sample_codec/640x480_30fps.h264",
    "/app/cdev_demo/vps/input_1080p.h264",
]


def _have(*names):
    return all(hasattr(bcdl, n) for n in names)


def _find_clip():
    p = os.environ.get("BCDL_VIDEO")
    if p:
        return p if os.path.exists(p) else None
    for c in _CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _start_codes(s):
    """Offsets of every Annex-B start code (00 00 01) in the byte stream."""
    out = []
    i = 0
    n = len(s)
    while i + 3 <= n:
        if s[i] == 0 and s[i + 1] == 0 and s[i + 2] == 1:
            out.append(i)
            i += 3
        else:
            i += 1
    return out


def _access_units(s, h265=False):
    """Split into [begin, end) byte ranges, one coded picture each.

    New AU starts at a VCL NAL when the current AU already has one; leading
    param-sets/SEI/AUD group with the following slice (single-slice streams).
    """
    sc = _start_codes(s)
    if not sc:
        return []
    aus = []
    au_begin = sc[0]
    au_has_vcl = False
    for pos in sc:
        hdr = pos + 3
        if hdr >= len(s):
            break
        b = s[hdr]
        t = (b >> 1) & 0x3F if h265 else b & 0x1F
        vcl = (0 <= t <= 31) if h265 else (1 <= t <= 5)
        if vcl and au_has_vcl:
            aus.append((au_begin, pos))
            au_begin = pos
            au_has_vcl = False
        if vcl:
            au_has_vcl = True
    aus.append((au_begin, len(s)))
    return aus


def test_vpu_decode_real_h264():
    if not _have("VideoDecoder", "VideoDecConfig", "VideoType"):
        pytest.skip("video codec bindings not exposed")
    clip = _find_clip()
    if clip is None:
        pytest.skip("no board .h264 clip found (set BCDL_VIDEO)")

    with open(clip, "rb") as f:
        stream = f.read()

    aus = _access_units(stream, h265=False)
    assert len(aus) > 1, f"expected multiple access units, got {len(aus)}"

    cfg = bcdl.VideoDecConfig()
    cfg.type = bcdl.VideoType.H264
    dec = bcdl.VideoDecoder(cfg)

    decoded = 0
    first = None
    for b, e in aus:
        out = dec.decode(stream[b:e])
        if out is not None:
            decoded += 1
            if first is None:
                first = out

    # The VPU may hold a small reorder tail (no flush API), but the vast majority
    # of frames must come back.
    assert decoded > 0, "VPU decoded no frames from a real H.264 clip"
    assert decoded >= len(aus) - 4, f"decoded {decoded} of {len(aus)} AUs — too few"

    # The first frame must be a real, non-degenerate NV12 image.
    assert first.format == bcdl.ImageFormat.NV12
    assert first.width >= 16 and first.height >= 16
    nv12 = first.to_numpy().reshape(first.height * 3 // 2, first.width)
    y = nv12[: first.height]
    assert y.std() > 1.0, "decoded Y plane is flat — likely not real pixels"

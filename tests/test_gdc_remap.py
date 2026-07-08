#!/usr/bin/env python3
"""GdcRemap correctness vs cv2.remap + latency bench (board-only).

Validates the CUSTOM-grid warp semantics empirically:
  1. identity map      -> output == input (Y max diff <= 1)
  2. +32px x-shift map  -> output == shifted input
  3. smooth synthetic warp + a real stereo-rectify map -> compare against
     cv2.remap ground truth on the Y plane (interior; borders differ by
     clamp-vs-constant policy)
  4. steady-state latency at 2448x2048 (BCDL_GDC_TIMING=1 for the breakdown)

    conda activate bcdl && python tests/test_gdc_remap.py [--image x.jpg] [--calib maps.npz]
"""
import argparse
import time

import cv2
import numpy as np

import bcdl


def to_nv12(bgr):
    h, w = bgr.shape[:2]
    nv12 = bcdl.bgr_to_nv12(bgr)
    return bcdl.vp_image_from_nv12(nv12.reshape(-1), w, h)


def y_plane(vp):
    a = vp.to_numpy()
    w, h = vp.width, vp.height
    return a[: w * h].reshape(h, w)


def compare(tag, got_y, ref_y, margin=8):
    g = got_y[margin:-margin, margin:-margin].astype(np.int16)
    r = ref_y[margin:-margin, margin:-margin].astype(np.int16)
    d = np.abs(g - r)
    print(f"  [{tag}] max={d.max()}  p99={np.percentile(d, 99):.1f}  mean={d.mean():.3f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="test image (else synthetic texture)")
    ap.add_argument("--calib", default="", help="npz with map_x/map_y (real rectify maps)")
    ap.add_argument("--size", default="2448x2048")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.split("x"))

    if args.image:
        img = cv2.imread(args.image)
        img = cv2.resize(img, (W, H))
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (H, W, 3), np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 2)  # texture without aliasing traps
    src = to_nv12(img)
    src_y = y_plane(src)

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # 1) identity
    g = bcdl.GdcRemap(xs, ys, W, H)
    out = g.run(src)
    compare("identity", y_plane(out), src_y)
    del g

    # 2) +32 px x-shift: out(x,y) = in(x+32, y)
    g = bcdl.GdcRemap(xs + 32, ys, W, H)
    out = g.run(src)
    ref = cv2.remap(src_y, xs + 32, ys, cv2.INTER_LINEAR)
    compare("shift+32x", y_plane(out), ref, margin=48)
    del g

    # 3) smooth synthetic warp (barrel-ish) vs cv2.remap
    cx, cy = W / 2, H / 2
    r2 = ((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2
    mx = (xs - cx) * (1 + 0.03 * r2) + cx
    my = (ys - cy) * (1 + 0.03 * r2) + cy
    mx = np.clip(mx, 0, W - 1).astype(np.float32)
    my = np.clip(my, 0, H - 1).astype(np.float32)
    g = bcdl.GdcRemap(mx, my, W, H)
    out = g.run(src)
    ref = cv2.remap(src_y, mx, my, cv2.INTER_LINEAR)
    # wide margin: at the borders the clipped map makes cv2 (BORDER_CONSTANT)
    # and the GDC (clamp) legitimately differ
    compare("barrel", y_plane(out), ref, margin=128)

    # 4) real rectify maps, if provided
    if args.calib:
        z = np.load(args.calib)
        mx, my = z["map_x"].astype(np.float32), z["map_y"].astype(np.float32)
        gr = bcdl.GdcRemap(mx, my, W, H)
        out = gr.run(src)
        ref = cv2.remap(src_y, mx, my, cv2.INTER_LINEAR)
        compare("rectify", y_plane(out), ref, margin=16)
        g = gr

    # 5) bench (last constructed map)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = g.run(src)
    dt = (time.perf_counter() - t0) / args.iters * 1e3
    print(f"  bench: {dt:.2f} ms/frame ({W}x{H} NV12, incl. numpy<->VpImage copies)")


if __name__ == "__main__":
    main()

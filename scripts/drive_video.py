"""Render a driving sequence to H.264 on the board, with bcdl's own VPU encoder.

Every frame is the SAME dashboard the benchmark check image uses
(`tests/board_models.draw_drive_frame`), so what you watch is what the figure
claims — a preview rendered by different code would be a preview of that code.

The input is an .npz holding one scene's frames: `camera` (N,3,256,1024),
`lidar` (N,1,256,256), `velocity`/`acceleration` (N,2), and optionally `gt`
(N,8,3) with a `has_gt` mask for the realized future. Sensor rigs are not
images, so this cannot be built from a video file — see docs/MODELS.md for how
the sample is staged.

Run from the repo root:

    PYTHONPATH=build:python python scripts/drive_video.py seq.npz out.h264 [fps]

Keyframe sequences are typically 2 Hz, so a higher `fps` plays them back faster
than real time; the default 5 is 2.5x. Wrap the raw stream for a player with:

    ffmpeg -r <fps> -i out.h264 -c copy out.mp4
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tests"))   # run from the repo root
import bcdl  # noqa: E402
import board_models as bm  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 5

d = np.load(src)
cam, lid, vel, acc, gt, has_gt = (d["camera"], d["lidar"], d["velocity"],
                                  d["acceleration"], d["gt"], d["has_gt"])
n = len(cam)

acfg = bcdl.DriveAnchorConfig()
acfg.noise_scale = 0.0                       # deterministic frame to frame
planner = bcdl.DrivePlanner(
    bcdl.Engine(os.path.join("models", "diffusiondrive_r34_256x1024_s100p.hbm")),
    np.load(os.path.join("models", "kmeans_navsim_traj_20.npy")).astype(np.float32),
    anchor_config=acfg)

cfg = bcdl.VideoEncConfig()
cfg.width, cfg.height = 1024, 768            # width %32, height %8
cfg.framerate, cfg.bitrate_kbps, cfg.intra_period = fps, 6000, fps * 2
enc = bcdl.VideoEncoder(cfg)

out = open(dst, "wb")
plan_ms = draw_ms = enc_ms = 0.0
errs = []
for i in range(n):
    st = bcdl.DriveStatus()
    st.velocity_x, st.velocity_y = float(vel[i][0]), float(vel[i][1])
    st.accel_x, st.accel_y = float(acc[i][0]), float(acc[i][1])

    t = time.perf_counter()
    res = planner.plan(np.ascontiguousarray(cam[i].astype(np.float32)),
                       np.ascontiguousarray(lid[i].astype(np.float32)), st)
    plan_ms += (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    frame = bm.draw_drive_frame(cam[i].astype(np.float32), lid[i].astype(np.float32),
                                res, vel[i], gt[i] if has_gt[i] else None,
                                tag="frame %d/%d" % (i + 1, n))
    draw_ms += (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    nv12 = bcdl.bgr_to_nv12(np.ascontiguousarray(frame))
    img = bcdl.vp_image_from_nv12(nv12, cfg.width, cfg.height)
    # Feed then drain: the encoder buffers, so packets trail the frames that
    # produced them (the cadence the header warns about).
    if not enc.feed(img, i * int(1e6 / fps)):
        while True:
            pkt = enc.receive(20)
            if pkt is None:
                break
            out.write(bytes(pkt))
        enc.feed(img, i * int(1e6 / fps))
    while True:
        pkt = enc.receive(0)
        if pkt is None:
            break
        out.write(bytes(pkt))
    enc_ms += (time.perf_counter() - t) * 1e3

    if has_gt[i]:
        p = np.array([[q.x, q.y] for q in res.trajectory])
        errs.append(float(np.linalg.norm(p - gt[i][:, :2], axis=1).mean()))

enc.flush()
while True:
    pkt = enc.receive(50)
    if pkt is None:
        break
    out.write(bytes(pkt))
out.close()

print("%d frames -> %s (%.1f KB)" % (n, dst, os.path.getsize(dst) / 1024))
print("per frame: plan %.1f ms, draw %.1f ms, encode %.1f ms" % (
    plan_ms / n, draw_ms / n, enc_ms / n))
print("plan vs realized future: mean %.2f m over %d frames with ground truth" % (
    float(np.mean(errs)), len(errs)))

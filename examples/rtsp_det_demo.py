#!/usr/bin/env python3
"""Live RTSP object detection — validates PURE HARDWARE decode of a network
stream, with all the heavy lifting in the C++ AsyncVideoDetectionPipeline.

  rtsp://cam  --ffmpeg -c copy (RTSP session + RTP depacketize, NO decode)-->
     Annex-B bytes  --pipe--> bcdl.AsyncVideoDetectionPipeline (C++):
        segment AUs → VPU decode ‖ NV12→BGR ‖ BPU preproc ‖ infer+NMS
     --> detections

Python does almost nothing: it pumps the ffmpeg byte stream into the pipeline and
reads results. All decode/convert/detect threads live in C++ (GIL released), so
this thin driver reaches the C++ decode-bound ceiling (~233 FPS @1080p H.264),
not the ~81 FPS a Python-orchestrated loop is capped at.

Only the VPU decodes: ffmpeg runs `-c:v copy` (no software decode); do NOT use
cv2.VideoCapture(rtsp) (that software-decodes on the CPU).

  python rtsp_det_demo.py <det.hbm> <rtsp://url> [max_frames] [depth] [h264|h265] [tcp|udp]

Validate on the board without a camera by looping a file over RTSP (needs an RTSP
server such as mediamtx; ffmpeg's own RTSP listen mode is unreliable).
"""

import subprocess
import sys
import time

import bcdl


def main(argv):
    if len(argv) < 3:
        print("usage: rtsp_det_demo.py <det.hbm> <rtsp://url> [max_frames] [depth] "
              "[h264|h265] [tcp|udp]", file=sys.stderr)
        return 1
    hbm, url = argv[1], argv[2]
    max_frames = int(argv[3]) if len(argv) > 3 else 300
    depth = int(argv[4]) if len(argv) > 4 else 4
    h265 = (len(argv) > 5 and argv[5].lower() in ("h265", "hevc"))
    transport = argv[6] if len(argv) > 6 else "tcp"

    engine = bcdl.Engine(hbm)
    cfg = bcdl.PipelineConfig()
    cfg.detect.num_classes = 80
    cfg.detect.conf_thresh = 0.25
    cfg.detect.iou_thresh = 0.45
    codec = bcdl.VideoType.H265 if h265 else bcdl.VideoType.H264
    pipe = bcdl.AsyncVideoDetectionPipeline(engine, cfg, codec, depth)

    # ffmpeg: RTSP session + RTP depacketize ONLY (-c:v copy => no software decode).
    ff = ["ffmpeg", "-hide_banner", "-loglevel", "error",
          "-rtsp_transport", transport, "-i", url,
          "-an", "-c:v", "copy", "-flush_packets", "1",
          "-f", "hevc" if h265 else "h264", "pipe:1"]
    print(f"model={engine.model_name}  rtsp={url} ({transport}, "
          f"{'H.265' if h265 else 'H.264'})  depth={depth}")
    print("  feeder: ffmpeg -c:v copy (NO software decode) -> C++ AsyncVideoDetectionPipeline")
    proc = subprocess.Popen(ff, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=0)

    recv = total_dets = 0
    t_first = None

    def drain(fn):
        nonlocal recv, total_dets, t_first
        while (dets := fn()) is not None:
            if t_first is None:
                t_first = time.perf_counter()
            recv += 1
            total_dets += len(dets)
            if recv % 30 == 0:
                fps = recv / (time.perf_counter() - t_first)
                print(f"  frame {recv:4d}  dets={len(dets):2d}  {fps:.1f} FPS")
            if recv >= max_frames:
                return True
        return False

    try:
        done = False
        while not done:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break                       # stream ended
            pipe.submit(chunk)              # C++ segments AUs + decodes + detects
            done = drain(pipe.next_nowait)  # collect whatever is ready
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        proc.terminate()
        pipe.finish()
        drain(pipe.next)                    # blocking drain of the last in-flight frames

    wall = time.perf_counter() - (t_first or time.perf_counter())
    sp = pipe.profile()
    print(f"\n=== live RTSP summary ({recv} frames) ===")
    print(f"  decode   {sp.decode_per_frame():.2f} ms/f  (VPU hardware decode)")
    print(f"  preproc  {sp.preproc_per_frame():.2f} ms/f  (CPU letterbox)")
    print(f"  infer    {sp.infer_per_frame():.2f} ms/f  (BPU)")
    print(f"  postproc {sp.postproc_per_frame():.2f} ms/f  (CPU NMS)")
    print(f"end-to-end {recv / wall:.1f} FPS | total dets {total_dets}")
    print("OK: rtsp -> ffmpeg(-c copy, no decode) -> C++ VPU decode -> BPU detect")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

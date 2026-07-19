#!/usr/bin/env python3
"""End-to-end video object detection in Python — the counterpart to
examples/video_det_demo_async.cc.

  in.(h264|h265|mp4|mov)  --VPU decode-->  NV12  --cv2-->  BGR
     --AsyncDetectionPipeline (CPU preproc || BPU infer+NMS)-->  boxes
     --draw--> BGR --cv2--> NV12 --VPU encode--> H.264/H.265 elem. stream
     --ffmpeg -c copy (container mux only)--> out.mp4

Every heavy step runs on dedicated silicon: the VPU decodes and encodes, the BPU
detects. ffmpeg is used ONLY as a (de)muxer with `-c copy` — it never decodes or
re-encodes pixels, so the VPU still does all the actual video coding.

  python video_det_demo.py <det.hbm> <in.(h264|h265|mp4|mov)> [out.mp4] [max_frames] [depth]

Notes
- MP4/MOV input is demuxed to an Annex-B elementary stream with ffmpeg first
  (raw .h264/.h265 needs no demux). Output MP4 is produced by muxing the VPU's
  elementary stream with ffmpeg `-c copy`.
- Requires the `bcdl` conda env (bcdl + opencv) and ffmpeg/ffprobe on PATH.
"""

import collections
import os
import subprocess
import sys
import time

import cv2
import numpy as np

import bcdl


# --- Annex-B parsing (mirrors the C++ demo) --------------------------------
def _nal_type(b, h265):
    return ((b >> 1) & 0x3F) if h265 else (b & 0x1F)


def _is_vcl(t, h265):
    return (0 <= t <= 31) if h265 else (1 <= t <= 5)


def access_units(s, h265):
    """Split an Annex-B byte stream into access units (one coded picture each)."""
    sc = [i for i in range(len(s) - 2) if s[i] == 0 and s[i + 1] == 0 and s[i + 2] == 1]
    if not sc:
        return []
    aus, au_begin, au_has_vcl = [], sc[0], False
    for pos in sc:
        hdr = pos + 3
        if hdr >= len(s):
            break
        vcl = _is_vcl(_nal_type(s[hdr], h265), h265)
        if vcl and au_has_vcl:
            aus.append((au_begin, pos))
            au_begin, au_has_vcl = pos, False
        if vcl:
            au_has_vcl = True
    aus.append((au_begin, len(s)))
    return aus


# --- container helpers (ffmpeg used only for muxing, never for coding) -----
def probe_codec(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return out
    except Exception:
        return ""


def load_annexb(path):
    """Return (elementary_stream_bytes, is_h265). Demux MP4/MOV via ffmpeg."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h264", ".264"):
        return open(path, "rb").read(), False
    if ext in (".h265", ".hevc", ".265"):
        return open(path, "rb").read(), True
    # Container (mp4/mov/mkv...): demux to Annex-B without touching the pixels.
    codec = probe_codec(path)
    h265 = codec in ("hevc", "h265")
    bsf = "hevc_mp4toannexb" if h265 else "h264_mp4toannexb"
    print(f"  demux {ext} ({codec or 'unknown'}) -> Annex-B via ffmpeg -c copy")
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-c:v", "copy",
         "-bsf:v", bsf, "-f", "hevc" if h265 else "h264", "-"],
        capture_output=True, check=True).stdout
    return data, h265


def mux_mp4(elem_path, out_mp4, fps):
    """Wrap the VPU's elementary stream into an MP4 (container only, -c copy)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-r", str(fps), "-i", elem_path,
         "-c", "copy", out_mp4], check=True)


# --- pixel-format conversion ------------------------------------------------
def nv12_to_bgr(flat, w, h):
    """VpImage.to_numpy() NV12 -> (H,W,3) BGR."""
    return cv2.cvtColor(flat.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)


def bgr_to_nv12_flat(bgr):
    """(H,W,3) BGR -> packed NV12 (W*H*3/2) uint8 [Y then interleaved UV]."""
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # (H*3/2, W) planar Y,U,V
    y = i420[:h]
    u = i420[h:h + h // 4].reshape(h // 2, w // 2)
    v = i420[h + h // 4:].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u   # NV12 = U,V interleaved
    uv[:, 1::2] = v
    return np.ascontiguousarray(np.vstack([y, uv])).reshape(-1)


COLORS = [(0, 255, 0), (0, 200, 255), (255, 128, 0), (200, 0, 255), (0, 128, 255)]


def draw(bgr, dets):
    for d in dets:
        c = COLORS[d.class_id % len(COLORS)]
        cv2.rectangle(bgr, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
        cv2.putText(bgr, f"c{d.class_id} {d.score:.2f}", (int(d.x1), int(d.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)


def main(argv):
    if len(argv) < 3:
        print("usage: video_det_demo.py <det.hbm> <in.(h264|h265|mp4|mov)> "
              "[out.mp4] [max_frames] [depth]", file=sys.stderr)
        return 1
    hbm, in_path = argv[1], argv[2]
    out_mp4 = argv[3] if len(argv) > 3 and argv[3] else \
        os.path.splitext(os.path.basename(in_path))[0] + "_det.mp4"
    max_frames = int(argv[4]) if len(argv) > 4 else 0
    depth = int(argv[5]) if len(argv) > 5 else 4
    fps = 30

    stream, h265 = load_annexb(in_path)
    aus = access_units(stream, h265)
    if not aus:
        print("no NAL start codes — not an Annex-B stream?", file=sys.stderr)
        return 2

    engine = bcdl.Engine(hbm)
    cfg = bcdl.PipelineConfig()
    cfg.detect.num_classes = 80          # COCO
    cfg.detect.conf_thresh = 0.25
    cfg.detect.iou_thresh = 0.45
    pipe = bcdl.AsyncDetectionPipeline(engine, cfg, depth)

    dcfg = bcdl.VideoDecConfig()
    dcfg.type = bcdl.VideoType.H265 if h265 else bcdl.VideoType.H264
    dec = bcdl.VideoDecoder(dcfg)

    print(f"model={engine.model_name}  video={in_path} "
          f"({'H.265' if h265 else 'H.264'}, {len(aus)} AUs)  depth={depth} -> {out_mp4}")

    enc = None
    enc_w = enc_h = 0
    elem = bytearray()
    pending = collections.deque()        # decoded BGR frames awaiting their result
    dec_ms = cvt_ms = encprep_ms = enc_ms = 0.0
    total_dets = frames = 0

    elem_path = os.path.splitext(out_mp4)[0] + (".h265" if h265 else ".h264")

    def consume(dets):
        nonlocal enc, enc_w, enc_h, total_dets, encprep_ms, enc_ms
        frame = pending.popleft()
        draw(frame, dets)
        total_dets += len(dets)
        if enc is None:                  # lazily size the encoder (w%32, h%8)
            h, w = frame.shape[:2]
            enc_w, enc_h = (w // 32) * 32, (h // 8) * 8
            ecfg = bcdl.VideoEncConfig()
            ecfg.type = dcfg.type
            ecfg.width, ecfg.height = enc_w, enc_h
            ecfg.framerate = fps
            enc = bcdl.VideoEncoder(ecfg)
        crop = frame[:enc_h, :enc_w]
        t = time.perf_counter()
        vp = bcdl.vp_image_from_nv12(bgr_to_nv12_flat(crop), enc_w, enc_h)  # CPU
        t2 = time.perf_counter()
        # The encoder is a QUEUE, not a function: a packet does not necessarily
        # come out per frame fed in. Feed, then drain to empty; if the input
        # queue is momentarily full, drain and RETRY (at 1080p we generate
        # frames far faster than the VPU encodes them, so this does happen).
        # The tail is flushed after the loop.
        for _ in range(100):
            if enc.feed(vp):
                break
            pkt = enc.receive(5)
            if pkt:
                elem.extend(pkt)
        else:
            raise RuntimeError("VPU encoder never accepted a frame")
        while True:
            pkt = enc.receive(0)
            if pkt is None:
                break
            elem.extend(pkt)                                               # VPU
        enc_ms += (time.perf_counter() - t2) * 1e3
        encprep_ms += (t2 - t) * 1e3

    t_wall = time.perf_counter()
    made = 0

    def submit_frame(vp):
        """Convert one decoded NV12 frame to BGR and push it into the detector."""
        nonlocal made, cvt_ms, frames
        t = time.perf_counter()
        bgr = nv12_to_bgr(vp.to_numpy(), vp.width, vp.height)
        cvt_ms += (time.perf_counter() - t) * 1e3
        pipe.submit(bgr)
        pending.append(bgr)
        made += 1
        if made > depth:
            dets = pipe.next()
            if dets is not None:
                consume(dets)
                frames += 1

    # Feed one AU, then drain every frame the decoder already has. Feeding faster
    # than we dequeue fills the VPU's frame buffers and the next hardware decode
    # stalls (INTERRUPT TIMEOUT); reorder streams also emit frames out of step with
    # the AUs, so the tail only appears via flush() at end-of-stream.
    for b, e in aus:
        if max_frames and made >= max_frames:
            break
        t = time.perf_counter()
        while not dec.feed(bytes(stream[b:e])):     # input queue full: drain, retry
            vp = dec.receive(5)
            dec_ms += (time.perf_counter() - t) * 1e3
            if vp is None:
                break
            submit_frame(vp)
            t = time.perf_counter()
        while (vp := dec.receive(0)) is not None:   # drain all frames ready now
            dec_ms += (time.perf_counter() - t) * 1e3
            submit_frame(vp)
            if max_frames and made >= max_frames:
                break
            t = time.perf_counter()
        if max_frames and made >= max_frames:
            break

    while not max_frames or made < max_frames:      # reorder tail at EOS
        t = time.perf_counter()
        vp = dec.flush()
        dec_ms += (time.perf_counter() - t) * 1e3
        if vp is None:
            break
        submit_frame(vp)

    pipe.finish()
    while (dets := pipe.next()) is not None:
        consume(dets)
        frames += 1
    wall_ms = (time.perf_counter() - t_wall) * 1e3

    # Drain the encoder's tail: packets for frames already fed but not yet
    # emitted. Without this the last frames are silently missing from the mp4.
    if enc is not None:
        while True:
            pkt = enc.flush()
            if pkt is None:
                break
            elem.extend(pkt)

    # Mux the VPU elementary stream into an MP4 (container only).
    with open(elem_path, "wb") as f:
        f.write(elem)
    mux_mp4(elem_path, out_mp4, fps)
    os.remove(elem_path)

    # Per-stage service time (stages overlap; slowest bounds throughput).
    sp = pipe.profile()
    n = max(frames, 1)
    rows = [
        ("decode",    dec_ms / n,               "VPU  hardware decode"),
        ("nv12->bgr", cvt_ms / n,               "CPU  cvtColor (main)"),
        ("preproc",   sp.preproc_per_frame(),   "CPU  letterbox BGR->NV12"),
        ("infer",     sp.infer_per_frame(),     "BPU  feed + submit/wait"),
        ("postproc",  sp.postproc_per_frame(),  "CPU  decode + NMS"),
        ("bgr->nv12", encprep_ms / n,           "CPU  I420->NV12 for encoder"),
        ("encode",    enc_ms / n,               "VPU  hardware encode"),
    ]
    slowest = max(r[1] for r in rows)
    ssum = sum(r[1] for r in rows)
    print(f"\n=== per-stage service time ({frames} frames, stages OVERLAP) ===")
    for name, per_f, unit in rows:
        mark = "  <== bottleneck" if per_f == slowest else ""
        print(f"  {name:<10} {per_f:7.2f} ms/f   ({unit}){mark}")
    print(f"  serial sum = {ssum:.2f} ms/f  ->  overlapped wall = {wall_ms / n:.2f} ms/f  "
          f"(overlap {ssum / (wall_ms / n):.2f}x)")
    print(f"detect+encode {frames} frames | {frames * 1000.0 / wall_ms:.1f} FPS end-to-end "
          f"| total dets {total_dets} -> {out_mp4}")
    print("OK: mp4/h264 -> VPU decode -> BPU detect -> VPU encode -> mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

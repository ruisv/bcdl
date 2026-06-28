#!/usr/bin/env python3
"""On-board benchmark + check-figure generator for bcdl's real-model task heads.

Runs each available YOLO26n (nash-m) task through bcdl end-to-end on a real
image, measures BPU infer latency / FPS, full decode latency, and the engine's
DRAM footprint, draws an annotated check image, and writes results.

Each task is benchmarked in a FRESH SUBPROCESS so the memory number (RSS delta
around loading + running exactly one model) is isolated and reproducible.

    # on the board, inside the bcdl conda env:
    cd ~/projects/bcdl
    PYTHONPATH=build:python python scripts/board_bench.py

Outputs (under benchmarks/, override with BCDL_BENCH_OUT):
    benchmarks/results.json     structured metrics
    benchmarks/RESULTS.md       markdown table
    benchmarks/figures/<task>.jpg   annotated check images
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tests"))
import board_models as bm  # noqa: E402

TASK_ORDER = ["cls", "det", "det_dfl", "pose", "seg", "obb", "semseg", "depth", "stereo", "ocr"]


def read_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def board_info():
    info = {}
    for path, key, div in [
        ("/sys/class/bpu/bpu_core0/devfreq/devfreq0/cur_freq", "bpu_freq_mhz", 1_000_000),
        ("/sys/class/bpu/bpu_core0/devfreq/cur_freq", "bpu_freq_mhz", 1_000_000),
    ]:
        try:
            with open(path) as f:
                info[key] = int(f.read().strip()) // div
                break
        except OSError:
            continue
    try:
        v = open("/etc/version").read().strip().splitlines()
        if v:
            info["sw_version"] = v[0]
    except OSError:
        pass
    return info


def bench_one(key, figdir, n_infer=50, n_e2e=20):
    """Bench a single task in THIS process (called inside the subprocess)."""
    import bcdl
    import cv2

    rss0 = read_rss_mb()
    t = bm.make_task(bcdl, key)
    t.feed()
    for _ in range(3):
        t.infer()                          # warmup
    rss1 = read_rss_mb()

    t0 = time.perf_counter()
    for _ in range(n_infer):
        t.infer()                          # BPU infer only (inputs preset)
    infer_ms = (time.perf_counter() - t0) / n_infer * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_e2e):
        results = t.decode()               # set_input + infer + postprocess
    e2e_ms = (time.perf_counter() - t0) / n_e2e * 1000.0

    results = t.decode()
    os.makedirs(figdir, exist_ok=True)
    fig = t.draw(results)
    # Cap very large figures (e.g. the 3714px DOTA obb scene) so committed check
    # images stay a sane size; preserve aspect ratio.
    longest = max(fig.shape[:2])
    if longest > 1600:
        s = 1600.0 / longest
        fig = cv2.resize(fig, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(figdir, f"{key}.jpg"), fig)

    return dict(
        task=key,
        model=os.path.basename(t.model_file),
        image=os.path.basename(t.image_file),
        input=f"{t.in_w}x{t.in_h}",
        outputs=int(t.engine.num_outputs),
        model_mb=round(os.path.getsize(t.model_file) / 1e6, 2),
        mem_mb=round(rss1 - rss0, 1),
        infer_ms=round(infer_ms, 3),
        infer_fps=round(1000.0 / infer_ms, 1),
        e2e_ms=round(e2e_ms, 3),
        e2e_fps=round(1000.0 / e2e_ms, 1),
        n_results=len(results) if hasattr(results, "__len__") else 1,
        summary=t.summarize(results),
    )


# Use 4:2:0 JPEGs — the JPU decoder only supports 4:2:0 (it rejects 4:2:2/4:4:4
# with a clean error now; e.g. office_desk.jpg / kite.jpg are 4:4:4). Override with
# BCDL_DECODE_IMAGES (comma-separated names).
DECODE_IMAGES = os.environ.get("BCDL_DECODE_IMAGES", "bus.jpg,gt_2322.jpg").split(",")


def draw_decode_overlay(bgr, cv_ms, jpu_ms, ratio):
    """Draw a cv2-vs-JPU decode speed comparison onto the decoded frame.

    A bare decoded photo is visually identical to a software-decoded one — the
    JPU's value is throughput, so we annotate the measured numbers (two bars +
    speed-up badge) directly onto the check image."""
    import cv2

    h, w = bgr.shape[:2]
    s = w / 820.0                                  # scale everything to image width
    pad = int(16 * s)
    bar_h = int(24 * s)
    gap = int(12 * s)
    panel_h = int(34 * s) + 2 * bar_h + gap + int(58 * s)
    y0 = h - panel_h
    overlay = bgr.copy()                            # translucent dark panel
    cv2.rectangle(overlay, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.60, bgr, 0.40, 0, bgr)

    def put(text, org, sc, color, th=1):           # outlined text for legibility
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2,
                    cv2.LINE_AA)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th,
                    cv2.LINE_AA)

    y = y0 + pad + int(16 * s)
    put("Hardware JPEG decode (JPU)  vs  cv2 / libjpeg (CPU)", (pad, y),
        0.62 * s, (255, 255, 255), 2)

    bar_x = pad + int(70 * s)
    bar_max = w - bar_x - int(150 * s)
    mmax = max(cv_ms, jpu_ms, 1e-6)
    y += int(18 * s)
    for label, ms, col in [("cv2", cv_ms, (150, 150, 150)),
                           ("JPU", jpu_ms, (90, 210, 120))]:
        bw = max(int(3 * s), int(bar_max * ms / mmax))
        cv2.rectangle(bgr, (bar_x, y), (bar_x + bw, y + bar_h), col, -1)
        put(label, (pad, y + int(bar_h * 0.72)), 0.5 * s, (255, 255, 255), 1)
        put(f"{ms:.2f} ms", (bar_x + bw + int(8 * s), y + int(bar_h * 0.72)),
            0.5 * s, (255, 255, 255), 1)
        y += bar_h + gap

    put(f"{ratio:.1f}x faster   -   frees the CPU, NV12 in device memory -> "
        "zero-copy to BPU", (pad, y + int(20 * s)), 0.52 * s, (90, 210, 120), 2)
    return bgr


def decode_compare(figdir):
    """Compare JPEG->NV12 via software (cv2/libjpeg, CPU) vs hardware (JPU).

    The JPU decodes straight to NV12 (what the BPU wants); cv2 needs imdecode +
    cvtColor. Runs in its own process. Also writes a check figure proving the JPU
    decode is a valid image. Returns a list of per-image rows.
    """
    import bcdl
    import cv2
    import numpy as np

    def timeit(fn, n=30):
        for _ in range(3):
            fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1000.0

    # Reuse ONE JpegDecoder across calls (its input device buffer is reused) —
    # the steady-state path, and it avoids churning JPU contexts (creating a new
    # decoder per call exhausts them and hard-crashes the process).
    dec = bcdl.JpegDecoder()
    rows = []
    saved_fig = False
    for name in DECODE_IMAGES:
        path = os.path.join(bm.ASSETS, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            raw = f.read()
        try:
            img = dec.decode(raw)              # JPU; aligns dims up, 4:2:0 only
        except Exception as e:                 # non-4:2:0 -> cleanly rejected; skip
            print(f"  (skip {name}: {str(e)[:60]})")
            continue
        w, h = img.width, img.height

        def cv_nv12(_raw=raw):
            bgr = cv2.imdecode(np.frombuffer(_raw, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

        def jpu_nv12(_raw=raw):
            return dec.decode(_raw).to_numpy()

        cv_ms = timeit(cv_nv12)
        jpu_ms = timeit(jpu_nv12)
        rows.append(dict(
            image=name, w=w, h=h, jpeg_kb=round(len(raw) / 1024, 1),
            cv2_ms=round(cv_ms, 2), jpu_ms=round(jpu_ms, 2),
            ratio=round(cv_ms / jpu_ms, 2)))

        if not saved_fig:                       # JPU NV12 -> BGR, as a check image
            nv12 = img.to_numpy().reshape(h * 3 // 2, w)
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            draw_decode_overlay(bgr, cv_ms, jpu_ms, cv_ms / jpu_ms)
            os.makedirs(figdir, exist_ok=True)
            cv2.imwrite(os.path.join(figdir, "decode_jpu.jpg"), bgr)
            saved_fig = True
    return rows


def track_figure(figdir):
    """Detect + ByteTrack over a synthesised camera pan of a still image, then
    draw each track's box, id, and motion trail on the final frame ->
    figures/track.jpg. Tracking has no single-frame photo of its own, so we
    synthesise motion exactly like tests/test_tracking_py (pan a still image)
    and show that ids stay stable across frames."""
    import bcdl
    import cv2
    import numpy as np

    model = os.path.join(bm.MODELS, "yolo26n_detect_nashm_640x640_nv12.hbm")
    img = cv2.imread(bm.find_image("bus.jpg"), cv2.IMREAD_COLOR)
    if not os.path.exists(model) or img is None:
        print("  (skip track: detect model or bus.jpg missing)")
        return False

    engine = bcdl.Engine(model)
    pipe = bcdl.TrackingPipeline(engine)

    H, W = img.shape[:2]
    win_w = int(W * 0.82)
    n = 16
    max_dx = W - win_w
    palette = [(66, 135, 245), (80, 220, 100), (60, 200, 240), (200, 120, 240),
               (240, 180, 60), (240, 90, 90), (120, 230, 200), (250, 130, 200)]
    trails, colors = {}, {}
    last_frame, last_tracks = None, []
    for f in range(n):
        x0 = int(round(max_dx * f / (n - 1)))
        frame = np.ascontiguousarray(img[:, x0:x0 + win_w])
        last_tracks = pipe.process(frame)
        last_frame = frame
        for t in last_tracks:
            trails.setdefault(t.track_id, []).append(
                (0.5 * (t.x1 + t.x2), 0.5 * (t.y1 + t.y2)))

    fig = last_frame.copy()
    for tid, pts in trails.items():
        colors.setdefault(tid, palette[len(colors) % len(palette)])
        if len(pts) >= 2:
            cv2.polylines(fig, [np.int32(pts)], False, colors[tid], 2, cv2.LINE_AA)
    for t in last_tracks:
        col = colors.get(t.track_id, palette[0])
        cv2.rectangle(fig, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), col, 2)
        org = (int(t.x1), max(int(14), int(t.y1) - 6))
        cv2.putText(fig, f"ID {t.track_id}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(fig, f"ID {t.track_id}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    col, 1, cv2.LINE_AA)
    cap = "ByteTrack: stable IDs + motion trails (synthesised camera pan)"
    cv2.putText(fig, cap, (10, fig.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(fig, cap, (10, fig.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(figdir, exist_ok=True)
    longest = max(fig.shape[:2])
    if longest > 1600:
        sc = 1600.0 / longest
        fig = cv2.resize(fig, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(figdir, "track.jpg"), fig)
    print(f"track: {len(trails)} ids over {n} frames -> figures/track.jpg")
    return True


def write_decode_md(rows):
    if not rows:
        return []
    lines = [
        "",
        "## JPEG decode — software (cv2/libjpeg, CPU) vs hardware (JPU)",
        "",
        "Decode straight to NV12 (what the BPU consumes). With a **reused** "
        "`JpegDecoder` (steady state — the way a real video / camera pipeline "
        "runs it) the JPU is **several times faster** (≈3.6–5.3× on these "
        "samples; see the table) than cv2/libjpeg, AND it "
        "**offloads the CPU** (frees the 6×A78AE cores for infer / post-process) "
        "and lands the result in a device NV12 buffer for **zero-copy NV12→BPU**. "
        "Caveat: the one-shot `bcdl.jpeg_decode` *helper* creates a fresh decoder "
        "per call (~5 ms JPU context setup), so it is NOT faster — reuse a "
        "`JpegDecoder` instance for the win. (Format note: the JPU decoder only "
        "supports **4:2:0** JPEGs; 4:2:2 / 4:4:4 streams are now rejected with a "
        "clean `bcdl::Error` — they used to segfault the JPU firmware — so decode "
        "those in software.) Check image: `benchmarks/figures/decode_jpu.jpg` "
        "(a JPU-decoded frame, annotated with the measured cv2-vs-JPU decode time).",
        "",
        "| image | size | jpeg KB | cv2 ms | JPU ms | cv2/JPU |",
        "|-------|------|---------|--------|--------|---------|",
    ]
    for r in rows:
        lines.append(f"| {r['image']} | {r['w']}x{r['h']} | {r['jpeg_kb']} | "
                     f"{r['cv2_ms']:.2f} | {r['jpu_ms']:.2f} | {r['ratio']:.2f} |")
    return lines


def write_md(rows, info, path, decode_rows=None):
    meta = " · ".join(f"{k}={v}" for k, v in info.items())
    lines = [
        "# BCDL on-board benchmark (real models)",
        "",
        f"RDK S100P (BPU){' · ' + meta if meta else ''}. Models: official "
        "YOLO26n **nash-m** from rdk_model_zoo (the LTRB family bcdl's decoders "
        "target). `infer` = BPU inference only; `decode` = infer + post-process "
        "(NMS / mask assembly / keypoints); `mem` = engine DRAM footprint "
        "(RSS delta around load+run, isolated per subprocess); `model` = `.hbm` "
        "size on disk.",
        "",
        "| task | model | input | outs | infer ms | infer FPS | decode ms | "
        "decode FPS | model MB | mem MB | result |",
        "|------|-------|-------|------|----------|-----------|-----------|"
        "------------|----------|--------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | `{r['model']}` | {r['input']} | {r['outputs']} | "
            f"{r['infer_ms']:.2f} | {r['infer_fps']:.0f} | {r['e2e_ms']:.2f} | "
            f"{r['e2e_fps']:.0f} | {r['model_mb']} | {r['mem_mb']} | "
            f"{r['summary']} |")
    lines += [
        "",
        "Check figures: `benchmarks/figures/<task>.jpg` (boxes / keypoints / "
        "instance masks / rotated boxes / top-k).",
    ]
    lines += write_decode_md(decode_rows or [])
    lines += [
        "",
        "Reproduce on the board (in the `bcdl` conda env): "
        "`PYTHONPATH=build:python python scripts/board_bench.py`.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_child(key, figdir):
    """Spawn a fresh process to bench one task (isolated memory)."""
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--task", key, "--figdir", figdir],
        capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    sys.stderr.write(proc.stdout + proc.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="bench a single task in-process (internal)")
    ap.add_argument("--decode", action="store_true", help="run decode compare (internal)")
    ap.add_argument("--track", action="store_true", help="generate tracking figure (internal)")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    outdir = os.environ.get("BCDL_BENCH_OUT", "benchmarks")
    figdir = args.figdir or os.path.join(outdir, "figures")

    if args.task:  # child: bench one task, emit JSON
        row = bench_one(args.task, figdir)
        print("RESULT_JSON:" + json.dumps(row))
        return
    if args.decode:  # child: JPEG decode software-vs-hardware
        print("DECODE_JSON:" + json.dumps(decode_compare(figdir)))
        return
    if args.track:  # child: detect + ByteTrack figure
        track_figure(figdir)
        return

    info = board_info()
    rows = []
    for key in TASK_ORDER:
        if not bm.available(key):
            print(f"skip {key}: model or image missing")
            continue
        row = run_child(key, figdir)
        if row is None:
            print(f"FAIL {key}: subprocess produced no result")
            continue
        rows.append(row)
        print(f"{key:5s} infer {row['infer_ms']:6.2f}ms / {row['infer_fps']:6.0f} FPS  "
              f"decode {row['e2e_ms']:6.2f}ms  mem {row['mem_mb']:5.1f}MB  "
              f"model {row['model_mb']:5.1f}MB  -> {row['summary']}")

    # JPEG decode: software (cv2) vs hardware (JPU), in its own process.
    decode_rows = []
    dproc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--decode", "--figdir", figdir],
        capture_output=True, text=True, env=dict(os.environ))
    for line in dproc.stdout.splitlines():
        if line.startswith("DECODE_JSON:"):
            decode_rows = json.loads(line[len("DECODE_JSON:"):])
    for r in decode_rows:
        print(f"decode {r['image']:16s} cv2 {r['cv2_ms']:5.2f}ms  JPU {r['jpu_ms']:5.2f}ms  "
              f"(cv2/JPU x{r['ratio']})")

    # Tracking check figure (detect + ByteTrack on a synthesised pan), own process.
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--track", "--figdir", figdir],
        text=True, env=dict(os.environ))

    if rows:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump({"board": info, "results": rows, "decode": decode_rows}, f, indent=2)
        write_md(rows, info, os.path.join(outdir, "RESULTS.md"), decode_rows)
        print(f"\nwrote {outdir}/results.json + RESULTS.md + figures/")
    else:
        print("no tasks ran (no models found)")


if __name__ == "__main__":
    main()

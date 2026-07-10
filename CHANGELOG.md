# Changelog

All notable changes to BCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-07

Hardening of the video path, and NV12-native preprocessing. The compressed-video
pipeline is 1.8x faster and no longer detours through BGR.

### Changed (breaking)
- **`GdcLetterbox` builds its warp LUT at construction** — the `bin_path`
  argument is gone, along with the need for an offline `.bin`. The affine
  letterbox is expressed as a CUSTOM sampling grid and generated at runtime.
- **`VideoDecoder::feed()` / `decode()` take exactly one access unit.** The
  decoder now runs in `MC_FEEDING_MODE_FRAME_SIZE`; feeding arbitrary byte chunks
  is no longer supported. (`AsyncVideoDetectionPipeline::submit()` still takes
  arbitrary Annex-B bytes and does the reassembly for you.)
- **`StageProfile` gained `cvt_ms`**, which `totalMs()` now includes.

### Added
- **`AsyncDetectionPipeline::submitNv12()`** — letterboxes a decoded NV12 frame
  straight into a model-input slot, on the GDC hardware engine when present and
  on the CPU otherwise. No BGR round-trip and no copy of the frame. Split into
  `acquireSlot()` / `letterboxIntoSlot()` / `commitSlot()` for callers that must
  not hold a recycled buffer while waiting for pipeline capacity.
- **`letterboxNv12Cpu()`** — NV12→NV12 letterbox with no colour conversion; the
  fallback when GDC is unavailable. `BCDL_NO_GDC=1` forces it, for A/B.
- **`YuvRange`** (`preproc/geometry.h`) — makes the studio→full-range expansion
  explicit. Video carries studio-swing NV12 (Y in [16,235]) while models are
  calibrated on full-range pixels; the old BGR round-trip did this conversion by
  accident, as a side effect of `cv::cvtColor`.
- `StageProfile::cvt_ms` / `cvtPerFrame()` (Python: `cvt_ms`, `cvt_per_frame`).

### Fixed
- **VPU `INTERRUPT TIMEOUT` hang.** Feeding faster than dequeuing fills the
  decoder's frame buffers and the next hardware decode stalls. Every decode path
  now feeds one AU, drains every ready frame, then flushes the reorder tail.
- **Heap corruption / random `SIGSEGV`.** The decoder ran in
  `MC_FEEDING_MODE_STREAM_SIZE`, whose bitstream-ring update path corrupts the
  glibc heap from inside the codec. Single-threaded decode alone crashed 6 runs
  in 20, with no BPU in the process; `FRAME_SIZE` is clean and also faster (pure
  decode 452 vs 281 FPS) and no longer drops a frame.
- **H.265 stalled at 16 FPS and lost frames.** A blocking output dequeue holds
  the codec's internal lock, so a feed thread racing a receive thread stalls
  whenever the decoder has no frame ready. One thread now owns the codec context.
- **Deadlock under backpressure.** The caller-facing results queue was bounded,
  so a full queue blocked the infer stage, which stopped slots recycling, which
  stalled every upstream stage — while the caller, blocked inside `submit()`, was
  the only one who could drain it.
- **`~VideoDecoder` hung** when the decoder still held decoded frames:
  `hb_mm_mc_stop()` waits for the codec to empty its output port, and only the
  application can empty it. The destructor now returns every pending buffer first.
- **One `Component vdec_render isn't ready!` error line per decoded frame.** A
  dequeue that finds no frame prints it (to stdout), and a drain-to-empty cadence
  ends every access unit with such a call. `recv()` now asks
  `hb_mm_mc_get_status()` first, which is silent.
- **`AsyncVideoDetectionPipeline` dropped the last frame** at end of stream: the
  trailing buffer can hold two access units, and under `FRAME_SIZE` only the
  first of them decodes.
- **Per-frame device allocation** in the video pipeline (334 allocations per 319
  frames → 22, all warm-up).
- **`nv12ToBgrCpu()`'s two build configurations produced different pixels** — the
  OpenCV fast path expanded studio swing, the hand-written fallback did not. It
  now takes a `YuvRange` and both branches implement each range.

### Performance
- `AsyncVideoDetectionPipeline`, 1080p H.264, yolo26n: **249 → 441 FPS**
  (1385/1385 frames), decode-bound. H.265: **439–451 FPS** (300/300 frames).
- Pure VPU decode: **452 FPS** (H.264) / **481 FPS** (H.265).
- GDC hardware letterbox, 1920×1080 → 640×640: 0.97 ms/frame wall, ~0.3 ms of it
  CPU, against 4.71 ms of pure CPU for the BGR chain it replaces.

## [0.2.0] — 2026-07

### Added
- **Video object detection — `AsyncVideoDetectionPipeline`** — the whole
  compressed-video → detections path in C++: `submit(Annex-B bytes)` → internal
  AU segmentation → four overlapped stages (VPU decode ‖ NV12→BGR ‖ CPU preproc
  ‖ BPU infer+NMS) → `next()`/`next_nowait()`. A thin caller that only pumps an
  `ffmpeg -c copy` byte stream reaches the decode-bound ceiling (yolo26n 1080p
  H.264 ~240 FPS); the Python binding releases the GIL. New demos
  `examples/video_det_demo.{cc,py}`, `examples/video_det_demo_async.cc`, and the
  thin RTSP driver `examples/rtsp_det_demo.py`.
- **H.265 decode** — `VideoDecoder` rewritten on the `media_codec` (`hb_mm_mc_*`)
  streaming API with reorder support and a decoupled `feed()` / `receive()` /
  `flush()` interface (the old per-AU model timed out on HEVC). Caveat: a
  hierarchical-GOP HEVC stream (camera SVC / smart-codec) decodes its base
  temporal layer only.
- **Per-stage `StageProfile` timing** — `decode` / `preproc` / `infer` /
  `postproc` ms per frame from `DetectionPipeline` and `AsyncDetectionPipeline`
  (in the async path the slowest stage bounds throughput), plus
  `AsyncDetectionPipeline::tryNext()` (non-blocking pop) and `nv12ToBgrCpu()`
  (NV12→BGR, OpenCV SIMD + BT.601 fallback).
- **`bcdl::GdcRemap` — hardware dense remap on the VPS GDC engine** (+ Python
  binding `bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)`). An arbitrary
  FIXED warp with cv2.remap semantics (built for stereo rectification) compiled
  at construction into a GDC CUSTOM-grid warp LUT — generated **at runtime** by
  driving `libgdcbin` directly (`gdc_init`/`gdc_set_custom_points`/
  `gdc_calculate`); no offline bin file. The public `hbn_gen_gdc_cfg` cannot do
  this (it never forwards custom grid points — verified by disassembly). Key
  semantics recovered from `transform_custom` (the SDK header comments are
  wrong/misleading): `custom.w/h` are TILE counts (points = `(w+1)*(h+1)`
  nodes) and `centerx/centery` are in GRID-INDEX units (`w/2, h/2`), not input
  pixels. Measured (2448×2048 NV12, real stereo-rectify maps, S100P):
  **6.3 ms/frame wall, ~1 ms of it CPU** (copy-in 0.5 + copy-out 0.5; the
  5.3 ms GDC op leaves the CPU free) vs 14.7 ms on all 6 cores for `cv2.remap`;
  matches cv2.remap to mean 0.25 / p99 2 grey-levels over the valid map region.
  Correctness/bench suite: `tests/test_gdc_remap.py`. `GdcRemap.run` releases
  the GIL, so two instances (stereo top/bottom) overlap one camera's CPU
  conversions with the other's hardware op from Python threads (adam_fast
  measured 24.3 → 18.6 ms for both cams).
- **Open-vocabulary detection / segmentation (YOLOE)** — prompt-free YOLOE reuses
  the existing LTRB / DFL decode with a `bcdl::LabelMap` (COCO-80 by default,
  `from_file` / `from_list`); no new post-process math. Board bench + figures:
  `yoloe_det`, `yoloe_seg` (~520 / ~397 infer FPS on S100P).
- **Promptable segmentation (EdgeSAM / SAM mask-decoder tail)** — `SamConfig`,
  `SamMask`, `SamMaskDecoder`, `decode_sam_masks`, and the two-stage
  `SamSession` Python wrapper (RepViT encoder → cached embedding → point/box
  prompt decoder). `tests/test_promptable_seg.py`.
- **ReID appearance embeddings** — `normalize_embedding` (L2) + `cosine_similarity`
  BoT-SORT association primitives. `tests/test_reid.py`.
- **QAT float-input detection path** — `YoloLtrbDetector.detect_float()` and
  `letterbox_chw_float()` feed a `[1,3,H,W]` float32 (QAT-exported) model; the
  LTRB/DFL decode is identical to the NV12 PTQ path.
- **Instance-seg DFL box heads** — `InstanceSegmenter` auto-detects plain-LTRB
  (4 ch) vs DFL (`4*reg_max`, e.g. 64) box channels, so yolov8/v11/YOLOE seg
  heads decode without config; `InstanceSegConfig.reg_max` exposed for the raw
  `decode_instance_seg()` entry point.

### Changed
- **Semantic-seg argmax 73× faster** — `Segmenter::postprocess()` argmaxes over
  the channel axis directly on the F32 device buffer (byte-stride aware, OpenMP),
  instead of materializing the whole logit tensor element-by-element first.
  Full-res deeplabv3plus `[1,1024,2048,19]` decode ~660 ms → ~8.7 ms, labels
  byte-identical. No model change (the materialize fallback covers F16/quantized).
- **Docs overhaul** — reorganized README around a 12-task check-image Gallery,
  a Documentation hub, badges/TOC, a fuller private-conda install (env pin,
  channel config, smoke test), and a Quickstart. Documented platform support as
  RDK **S100 / S100P / S600** (same S-series runtime; a compiled `.hbm` is
  march-specific). `scripts/board_bench.py` now annotates the JPU check image
  with the measured cv2-vs-JPU bars and generates a ByteTrack check figure
  (`benchmarks/figures/track.jpg`).

### Added
- **Bilingual README** — Simplified Chinese as the default (`README.md`) with a
  full English translation (`README.en.md`) and a language switcher
  (`English | 简体中文`) at the top of both files.
- **Stereo** — `StereoPipeline`: two-image (left/right) F32 NCHW RGB input
  (resize **or** center-crop fit) → disparity, with optional metric depth
  (`z=fx·baseline/disp`) and a geometry validity mask (disparity range +
  left-border + optional left-right consistency). Pure `pack_stereo_input` /
  `disparity_to_depth` / `stereo_valid_mask` helpers + nanobind bindings +
  `examples/stereo_demo.cc`. Validated on LAS2-M (cosine 1.0 / EPE 0 px vs the
  reference preproc, 70 FPS @480×640).
- Packaging: `find_package(bcdl)` CMake export (`bcdlConfig.cmake` +
  `bcdlTargets.cmake`, bundled `FindHobot.cmake`), `install()` rules, and a
  generated `bcdl/version.h` (`BCDL_VERSION*` macros).
- Python: `pip install .` via scikit-build-core (`pyproject.toml`), `py.typed`
  marker, and `bcdl.__version__`.
- Docs: `CONTRIBUTING.md`, plus full Python (`docs/API.md`) and C++
  (`docs/CPP_API.md`) API references — every class, config, and decoder with a
  usage snippet per task.

### Fixed
- Docs: corrected stale benchmark numbers (semseg decode `660 → 58 ms` after the
  argmax speedup; JPU decode `2.6–3.5× → ≈3.6–5.3×`, now consistent with the
  measured table), the tracking module location (`tracks/`, not `tasks/`), and
  API-reference signature mismatches (the `Detector` wrapper has no
  `postprocess()`; the `decode_*` letterbox parameter name).

## [0.1.0] — 2026-06

First tagged release. On-board validated on RDK S100P (79 passed, 1 skipped).

### Added
- **Core** — `SysMem` / `Task` / `MemPool` (RAII over `hbUCP`), `Engine` over
  `hbDNN` with automatic cache discipline, zero-copy / dequantising output readers.
- **Tasks** — detection (LTRB multi-scale + DFL: YOLO26 / YOLOv8 / v5 / v11),
  classification, pose, instance segmentation, oriented boxes (OBB), semantic
  segmentation, monocular depth, and a 3-stage **PP-OCRv5** OCR pipeline
  (DBNet detect → PP-LCNet direction cls → CRNN/CTC recognize). Each decoder is
  also exposed as an Engine-free pure `decode_*` function.
- **Tracking** — ByteTrack multi-object tracker (Kalman + Hungarian + two-stage
  association).
- **Media** — hardware JPEG (JPU) and H.264 / H.265 (VPU) encode/decode.
- **Pipelines** — synchronous buffer-reuse `DetectionPipeline`, threaded
  `AsyncDetectionPipeline` (preproc ‖ infer overlap, GIL released in Python),
  `TrackingPipeline`, and a video-file → decode → detect path.
- **Python** — nanobind bindings (NumPy in/out) for every decoder + pipeline.
- **Tooling** — on-board benchmark + check-figure generator
  (`scripts/board_bench.py`), repo-local model dir (`scripts/fetch_models.sh`),
  two-tier test suite (Engine-free NumPy + on-board real-model).

[Unreleased]: https://example.com/compare/v0.1.0...HEAD
[0.1.0]: https://example.com/releases/tag/v0.1.0

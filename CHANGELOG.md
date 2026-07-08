# Changelog

All notable changes to BCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

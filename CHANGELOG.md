# Changelog

All notable changes to BCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

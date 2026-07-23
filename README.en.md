# BCDL — BPU Computational Deep Learning

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-RDK%20S100%20%2F%20S100P%20%2F%20S600%20(aarch64)-0A7BBB.svg)](#requirements)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](CHANGELOG.md)

**English** | [简体中文](README.md)

A C++17 inference & media library for **D-Robotics RDK S100 / S100P / S600** (the
BPU "Nash" accelerator), with NumPy-friendly Python bindings. BCDL is the BPU-native
counterpart to a CUDA/TensorRT vision stack: load a compiled `.hbm` model, run
inference, pre/post-process, hardware codec, and stream — all on the board, with
zero-copy across the media and compute units by default.

## Contents

- [Gallery](#gallery)
- [Why BCDL](#why-bcdl) · [Features](#features) · [Architecture](#architecture)
- [Install](#install) · [Quickstart](#quickstart) · [Build from source](#build-from-source)
- [Documentation](#documentation)
- [Benchmarks](#benchmarks) · [Tests](#tests)
- [Community](#community) · [Contributing](#contributing) · [Acknowledgments](#acknowledgments) · [License](#license)

## Gallery

Annotated outputs from the on-board benchmark suite (real models on RDK S100P).
Reproduce with `scripts/board_bench.py`; full numbers in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Classification (top-k) | Detection (YOLO26 LTRB) | Detection — DFL head (YOLOv8) |
|:---:|:---:|:---:|
| <img src="benchmarks/figures/cls.jpg" width="250"> | <img src="benchmarks/figures/det.jpg" width="250"> | <img src="benchmarks/figures/det_dfl.jpg" width="250"> |
| **Oriented boxes (OBB)** | **Multi-object tracking (ByteTrack)** | **Pose (17-keypoint)** |
| <img src="benchmarks/figures/obb.jpg" width="250"> | <img src="benchmarks/figures/track.jpg" width="250"> | <img src="benchmarks/figures/pose.jpg" width="250"> |
| **Instance segmentation** | **Semantic segmentation** | **Monocular depth** |
| <img src="benchmarks/figures/seg.jpg" width="250"> | <img src="benchmarks/figures/semseg.jpg" width="250"> | <img src="benchmarks/figures/depth.jpg" width="250"> |
| **Stereo depth (LAS2)** | **OCR (PP-OCRv5)** | **Hardware JPEG decode (JPU)** |
| <img src="benchmarks/figures/stereo.jpg" width="250"> | <img src="benchmarks/figures/ocr.jpg" width="250"> | <img src="benchmarks/figures/decode_jpu.jpg" width="250"> |

## Why BCDL

The whole S-series compute + media stack is unified on two hobot primitives:

- **`hbUCPSysMem`** — one shared-memory buffer (`phyAddr` + `virAddr`) used alike
  by BPU tensors, JPU/VPU codec images, and the VP preprocessing units.
- **`hbUCPTaskHandle_t`** — one task/queue model; BPU inference, JPEG/H.264/H.265
  codec, and resize/cvtColor all submit & wait through the same `hbUCP` scheduler.

So **zero-copy across JPU → VP → BPU → VPU is the default, not an optimization.**
BCDL is a thin, RAII-clean C++ layer over that fabric plus portable
post-processing (CUDA kernels rewritten as CPU/NEON).

## Features

- **Backend** — `Engine` over `hbDNN` (`hbDNNInferV2`); automatic cache discipline
  (clean before infer, invalidate before read); zero-copy / dequantising output
  readers.
- **Tasks** (CPU/NEON post-process, Engine-free pure `decode_*` functions too):
  - **Detection** — anchor-free LTRB multi-scale + DFL heads (YOLO26 / YOLOv8 /
    v5 / v11), per-class NMS.
  - **Classification, Pose** (17-keypoint), **Instance segmentation** (proto ×
    mask-coef), **Oriented boxes** (OBB, rotated-IoU NMS), **Semantic
    segmentation**, **Monocular depth**, **Stereo depth** (two-image disparity),
    **Monocular 3D detection** (SMOKE — 3D box + orientation from a single image).
  - **OCR** — full 3-stage **PP-OCRv5** pipeline: DBNet detect → PP-LCNet
    direction classifier (0°/180°) → CRNN/CTC recognize (18385-class dict).
  - **Open-vocabulary detection / segmentation** — **YOLOE** (prompt-free, ships a
    COCO-80 label table `LabelMap`, reuses the LTRB / DFL decode — name classes
    without retraining).
  - **Promptable segmentation** — **EdgeSAM** interactive segmentation (point / box
    prompts; RepViT image encoder → cached embedding → prompt decoder two-stage,
    `SamSession`).
  - **Panoptic driving perception** — YOLOP emits three heads from one
    inference: vehicle detection (anchor-based `AnchorDetector`) plus drivable
    area and lane lines (both through `Segmenter`). Note that the published ONNX
    bakes the anchor decode into the graph via ScatterND, and **that decode does
    not survive BPU compilation** — the objectness/class columns are never
    written, giving zero detections at any threshold — so the graph is cut before
    it and `decodeYoloV5Anchor` runs the arithmetic on the CPU.
  - **Image embeddings** — `ImageEmbedder` + `EmbeddingBank` (retrieval and
    zero-shot classification: pooled read-out + L2 normalize + cosine top-k).
    Note that an embedding `.hbm` commonly packs several submodels (a pooled
    whole-image vector and a per-patch feature grid) — pick the pooled one with
    `Engine::modelNames`; preprocessing is a **squashing resize, not a
    letterbox** (these towers never saw padding bars).
  - **Multi-object tracking** — ByteTrack (Kalman + two-stage association); ReID
    appearance embeddings with L2-normalize + cosine similarity (BoT-SORT
    association primitives).
  - Detection heads take both **PTQ (NV12 two-plane) and QAT-exported float-input**
    models (`detect_float` / `letterbox_chw_float`).
- **Hardware preprocessing** — fixed geometric transforms on the VPS GDC engine:
  hardware letterbox and arbitrary dense remap `GdcRemap` (cv2.remap semantics, for
  stereo rectification; 2448×2048 ≈ 6.3 ms, CPU mostly idle).
- **Media** — hardware **JPEG** (JPU) and **H.264 / H.265** (VPU) encode/decode.
- **Pipelines** — synchronous buffer-reuse `DetectionPipeline`, threaded
  `AsyncDetectionPipeline` (preproc ‖ infer overlap), `TrackingPipeline`,
  `StereoPipeline`, and a video-file → decode → detect path.
- **Python** — nanobind bindings: NumPy in/out, every decoder + pipeline, GIL
  released around blocking infer.

## Architecture

```
python/    nanobind bindings (NumPy <-> tensors), GIL-released infer
tasks/     det · cls · pose · seg · obb · semseg · depth · mono3d · ocr · open-vocab · sam · embed · drive
tracks/    ByteTrack multi-object tracker · ReID appearance embeddings
pipeline/  sync / async detection · tracking · stereo  (JPU -> VP -> BPU -> CPU/VPU)
media/     JpegCodec (JPU) · VideoCodec H.264/H.265 (VPU)
backend/   Engine, output readers          (libdnn  -> hbDNN*)
preproc/   CPU letterbox + BGR->NV12 (OpenCV/OpenMP); GDC HW letterbox + dense remap (VPS); VP (hb_vp)
core/      SysMem · Task · Status · MemPool (libhbucp -> hbUCP*)
```

## Requirements

- An **RDK S100 / S100P / S600** board (Ubuntu 22.04, aarch64) with the D-Robotics
  hobot SDK present (`/usr/include/hobot`, `/usr/hobot/lib`: `libdnn`, `libhbucp`,
  `libhbvp`) — or the `hobot-dnn` / `hobot-media` conda packages below. BCDL's
  source is the same across the S-series; a compiled `.hbm` is **march-specific**,
  so run a model built for your target board (see [Models](#models)).
- CMake ≥ 3.22, GCC 11, Ninja.
- OpenCV 5 (image ops; guarded by `BCDL_HAVE_OPENCV` with hand-written fallbacks).
- For the Python module: a Python env with **nanobind** (and NumPy, OpenCV).

## Install

Prebuilt **linux-aarch64** packages (Python 3.9–3.14) are published to a conda
channel, so on the board you can skip the source build entirely:

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bcdl
python -c "import bcdl; print(bcdl.__version__)"
```

Prefer a clean, reproducible environment? Create one and pin a version:

```bash
conda create -n bcdl -c https://mirrors.ruis.ai/conda -c conda-forge \
    python=3.12 bcdl
conda activate bcdl
# or pin an exact build:   conda install ... "bcdl=0.1.0"
```

To avoid passing `-c` every time, add the channel to the env (it must sit
**above** conda-forge so the `hobot-*` packages resolve from here):

```bash
conda config --env --add channels https://mirrors.ruis.ai/conda
conda config --env --add channels conda-forge
```

Then `conda install bcdl` / `conda update bcdl` resolve against the channel
directly. Verify the install end-to-end (prints the version and the loaded
native extension path):

```bash
python -c "import bcdl, bcdl_py; print('bcdl', bcdl.__version__); print(bcdl_py.__file__)"
```

That resolves the whole stack as four packages — the Python bindings, the C++
library, and the packaged D-Robotics hobot SDK they link against:

| package | ships |
|---------|-------|
| **bcdl** | the `bcdl` / `bcdl_py` Python bindings (one build per Python 3.9–3.14) |
| **libbcdl** | the C++ library — `libbcdl.so`, public headers, and the `find_package(bcdl)` CMake config |
| **hobot-dnn** | the BPU/DNN runtime SDK (`libdnn`, `libhbucp`, `libhbvp`, …) + `hobot/` headers |
| **hobot-media** | the media/codec line (`libffmedia`, `libgdcbin`, `libmultimedia`, …) + media dev headers |

For a **C++-only** consumer, install just the library and build against it with
`find_package(bcdl)` (it pulls `libbcdl` + `hobot-dnn` + `hobot-media`):

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge libbcdl
```

> The packaged hobot SDK still relies on the board's **device platform libraries**
> (`libbpu`, `libhbmem`, `libalog`, `libvdsp`, `libhbipcfhal`) under
> `/usr/hobot/lib` — they ship with the RDK system image and resolve via
> `ldconfig`, and are intentionally not redistributed.

## Quickstart

**Python** — streaming detection with preproc ‖ infer overlap:

```python
import bcdl

engine = bcdl.Engine("models/yolo26s_det_nashm_640x640_nv12.hbm")
cfg = bcdl.PipelineConfig(); cfg.detect.num_classes = 80
pipe = bcdl.AsyncDetectionPipeline(engine, cfg, depth=3)

for i, frame in enumerate(frames):          # frame: HxWx3 uint8 BGR
    pipe.submit(frame)                       # blocks if full (backpressure)
    if i >= 3:
        for d in pipe.next():                # results in submission order
            print(d.class_id, d.score, d.x1, d.y1, d.x2, d.y2)
pipe.finish()
while (dets := pipe.next()) is not None:
    ...                                      # drain the last in-flight frames
```

The pure-function decoders are also exposed for a NumPy-only path
(`bcdl.decode`, `bcdl.decode_pose`, `bcdl.decode_obb`, `bcdl.decode_ctc`, …).

**C++** — the [`examples/`](examples/) directory has standalone programs:

```bash
./build/det_demo    models/yolo26s_det_nashm_640x640_nv12.hbm data/images/bus.jpg
./build/ocr_demo    data/images/ocr.jpg          # PP-OCRv5 det -> cls -> rec
./build/track_demo  models/yolo26s_det_nashm_640x640_nv12.hbm  # detect + ByteTrack
./build/video_det_demo  stream.h264 model.hbm    # VPU decode -> BPU detect
```

For the full surface — every class, config, and decoder, with a usage snippet per
task — see the [API reference](#documentation) below.

## Build from source

Build on the board in your Python/conda env (needs the hobot SDK — from the
`hobot-dnn` / `hobot-media` conda packages above, or the system image at
`/usr/include/hobot` + `/usr/hobot/lib`):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds the `bcdl` library, the C++ examples, and the `bcdl_py` Python module.

**Install & consume from CMake** — `find_package(bcdl)` is supported:

```bash
cmake --install build --prefix /your/prefix
```
```cmake
find_package(bcdl CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE bcdl::bcdl)   # pulls in headers + hobot deps
```

**Install the Python module** — a pip-installable wheel (scikit-build-core);
build it on the board (the C++ build needs the hobot SDK):

```bash
pip install .          # or: pip wheel . -w dist/
python -c "import bcdl; print(bcdl.__version__)"
```

### Models

Compiled BPU models (`.hbm`) are **not** committed (they are large) — place them
under [`models/`](models/), populated by
[`scripts/fetch_models.sh`](scripts/fetch_models.sh). Model conversion
(ONNX → `.hbm`, PTQ calibration) is done **offline** with the D-Robotics
**OpenExplorer** toolchain on an x86 host; the full conversion recipes — export,
calibration, `hb_compile` config and each model's measured accuracy/latency —
live in the companion [**bcdl-model-zoo**](https://github.com/ruisv/bcdl-model-zoo)
repository. Each `.hbm` is compiled for a specific BPU **march**, so use one built
for your board (the S100 / S100P share the Nash march; S600 is compiled for its
own) — the BCDL runtime itself is the same across the S-series.

**What each model is, where it came from, which build to take, and its licence:
[`docs/MODELS.md`](docs/MODELS.md).** The example/benchmark models come from
third parties, and **the weights and their compiled `.hbm` follow their own
upstream licences** — notably the Ultralytics YOLO weights are **AGPL-3.0**
(copyleft; commercial use needs their Enterprise license) and the insightface
face weights are **academic / non-commercial**. Check each before redistributing
or using commercially. BCDL's own code (this repo, Apache-2.0) is independent of
these — it's a generic runtime that can load any `.hbm`; those licences apply to
the model weights, not to BCDL.

## Documentation

| Document | What it covers |
|----------|----------------|
| [`docs/API.md`](docs/API.md) | **Python API reference** — every class, config, and `decode_*` function, with a usage snippet per task. |
| [`docs/CPP_API.md`](docs/CPP_API.md) | **C++ API reference** — the same surface in `namespace bcdl`, keyed to the headers. |
| [`docs/MODELS.md`](docs/MODELS.md) | **Model catalogue** — every `.hbm` the examples and benchmarks use: origin, which build to take, and licence. |
| [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) | Full on-board benchmark numbers + the annotated check images in the [Gallery](#gallery). |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to set up, build (on the board), test, and submit changes. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes (Keep a Changelog / SemVer). |

Public headers live in [`include/bcdl/`](include/bcdl/); runnable programs in
[`examples/`](examples/).

## Benchmarks

Measured on **RDK S100P** (sw 4.0.5). `infer` is BPU inference only; `decode`
adds BCDL's post-processing; `model` is the `.hbm` size on disk. Annotated
outputs for each row are in the [Gallery](#gallery).

| task | model | input | infer ms | infer FPS | decode ms | model MB |
|------|-------|-------|----------|-----------|-----------|----------|
| cls    | yolo26n              | 224²        | 0.45  | 2235 | 0.45  | 3.9  |
| det    | yolo26n (LTRB)       | 640²        | 1.16  | 860  | 1.99  | 7.8  |
| det_dfl| yolov8 (DFL head)    | 640²        | 1.54  | 647  | 9.60  | 3.7  |
| pose   | yolo26n              | 640²        | 1.31  | 761  | 1.52  | 7.7  |
| seg    | yolo26n              | 640²        | 1.64  | 612  | 11.4  | 9.9  |
| obb    | yolo26n              | 640²        | 1.15  | 872  | 1.65  | 5.8  |
| semseg | deeplabv3plus        | 2048×1024   | 49.6  | 20   | 58.0  | 39.1 |
| semseg_rt| PIDNet-S (1/8 output)| 2048×1024  | 4.52  | 221  | 4.70  | 19.0 |
| depth  | depth-anything-v2    | 686×518     | 113   | 9    | 117   | 121.8|
| stereo | las2-m (crop)        | 640×480     | 14.2  | 71   | 21.8  | 40.7 |
| ocr    | PP-OCRv5 (det→cls→rec)| 960² / 48×320 | 20.2 | 50  | 137   | 35.3 |
| embed  | SigLIP base/16       | 224²        | 19.3  | 52   | 20.2  | 129  |
| drive  | YOLOP (det+drivable+lane)| 640²   | 2.80  | 358  | 5.44  | 12.0 |

Streaming throughput on yolo26s @1280×720: synchronous **216 FPS**, async
overlap **334 FPS** (1.55×). Hardware **JPEG** decode (JPU) is **≈3.6–5.3×**
faster than `cv2`/libjpeg on the sample images and offloads the CPU (zero-copy
NV12→BPU).

Compressed video, end to end (`AsyncVideoDetectionPipeline`, yolo26n @1080p):
**441 FPS** on H.264 and **439–451 FPS** on H.265, bounded by the VPU decode. The
decoded NV12 is letterboxed straight into the model input on the GDC hardware
engine — no BGR round-trip (0.97 ms/frame, only ~0.3 ms of it CPU). Pure VPU
decode runs at 452 / 481 FPS. The input must be an **Annex-B elementary stream**;
demux an MP4 first with `ffmpeg -c:v copy -bsf:v h264_mp4toannexb` (container
only, pixels untouched).

Reproduce on the board:

```bash
PYTHONPATH=build:python python scripts/board_bench.py
```

## Tests

With **bcdl installed** (from the conda channel above, or a pip wheel), run the
suite straight against the installed module — no `PYTHONPATH` needed:

```bash
pip install pytest          # plus the repo checked out for the test files
pytest tests/               # full suite; on-board tests skip if a model is absent
```

From a **source build**, point pytest at the in-tree module instead:

```bash
# (1) Engine-free numpy decode tests — no board / no models required:
PYTHONPATH=build:python pytest tests/test_detection.py tests/test_pose.py \
    tests/test_obb.py tests/test_instance_seg.py tests/test_depth_seg.py \
    tests/test_depth_seg_py.py tests/test_classification.py tests/test_ocr.py

# (2) On-board, real-model end-to-end — needs models/ populated; each task skips
#     cleanly if its model/image is absent:
PYTHONPATH=build:python pytest tests/
```

Two layers: the post-processing math is pinned by **deterministic NumPy tests**
through the Engine-free `decode_*` bindings (run anywhere), then the **on-board
suite** validates the full BPU/codec path on real models. On RDK S100P,
end-to-end against the **conda packages** (`bcdl` + `libbcdl` + `hobot-dnn` +
`hobot-media`): **96 passed, 1 skipped** (97 collected; the skip needs an
explicit `--hbm`).

| group | files | tests | needs board? |
|-------|-------|-------|--------------|
| Decode math (det/cls/pose/obb/instance-seg/depth/seg/ocr/stereo) + memory-safety | `test_detection`, `test_classification`, `test_pose`, `test_obb`, `test_instance_seg`, `test_depth_seg`(+`_py`), `test_ocr`, `test_stereo_py`, `test_memory_safety` | 66 | no |
| Tasks on real models (cls·det·det_dfl·pose·seg·obb·semseg·depth·OCR 3-stage) | `test_board_models` | 11 | yes |
| Stereo depth (real LAS2 `.hbm`) | `test_stereo_board_py` | 3 | yes |
| Media codecs (VPImage · JPEG/JPU · H.264/H.265/VPU) | `test_codec_py`, `test_video_decode_py` | 11 | yes |
| Pipelines (ByteTrack, async detection) | `test_tracking_py`, `test_async_detection_py` | 5 | yes |

## Community

Join the **BCDL (BPU) tech chat group** to discuss RDK / BPU deployment and using
this project.

<img src="docs/assets/bcdl-group-qr.jpg" alt="BCDL chat group QR" width="240">

> WeChat group QR codes expire — if it no longer works, please open an
> [Issue](../../issues) and we'll refresh it. You're also welcome to reach us via
> [Issues](../../issues).

## Contributing

Contributions are welcome — issues and pull requests alike. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide; the essentials:

- **Develop anywhere, build & run on the board.** The hobot SDK exists only on
  RDK hardware, so the C++ library and the on-board test suite must be built and
  exercised on an S100 / S100P / S600 board. The Engine-free NumPy `decode_*` tests run
  on any host (see [Tests](#tests)).
- **Conventions** — headers `.h`, implementations `.cc`; namespace `bcdl`;
  errors via `BCDL_CHECK(...)` → `bcdl::Error`. Match the style and structure of
  the surrounding code.
- **Tests** — add or update tests for any behavior change. Pin post-processing
  math with a deterministic NumPy test through the `decode_*` bindings (runs
  anywhere) and, where a model is involved, an on-board end-to-end test that
  skips cleanly when its model is absent.
- **Commits** — keep them focused; follow the existing
  [Conventional Commits](https://www.conventionalcommits.org/) style in the log.
- **Changelog** — note user-visible changes under `[Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md).

## Acknowledgments

- **D-Robotics** — the RDK S100 / S100P / S600 platform and the hobot SDK (`hbDNN`,
  `hbUCP`, `hb_vp`, media/codec) that BCDL is built on, and the `rdk_model_zoo`
  reference models.
- **[nanobind](https://github.com/wjakob/nanobind)** — the Python binding layer.
- **[OpenCV](https://opencv.org/)** — image ops on the preprocessing path.
- The upstream model authors whose post-processing BCDL re-implements —
  **PaddleOCR / PaddlePaddle** (PP-OCRv5, PP-LCNet), **Ultralytics** (YOLO
  family), **Depth-Anything-V2**, and **DeepLabV3+**. See [Models](#models) for
  their licenses; BCDL bundles none of these weights.

## License

BCDL is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).

This covers BCDL's own source code only. The third-party model weights under
`models/` and the `.hbm` compiled from them are governed by their own upstream
licenses (see [Models](#models)) — most notably the Ultralytics YOLO weights are
AGPL-3.0, so follow their upstream terms before using or redistributing.

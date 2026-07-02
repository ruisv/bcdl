# Contributing to BCDL

Thanks for your interest in BCDL — contributions are welcome, whether that's a
bug report, a fix, a new task decoder, a benchmark, or a docs improvement. This
guide covers how to set up, build, test, and submit changes.

By contributing you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).

## Table of contents

- [The one hard constraint: build on the board](#the-one-hard-constraint-build-on-the-board)
- [Project layout](#project-layout)
- [Getting set up](#getting-set-up)
- [Build](#build)
- [Tests](#tests)
- [Coding conventions](#coding-conventions)
- [Commits and changelog](#commits-and-changelog)
- [Pull requests](#pull-requests)
- [Reporting bugs](#reporting-bugs)
- [Scope: what belongs here](#scope-what-belongs-here)

## The one hard constraint: build on the board

BCDL links the D-Robotics **hobot SDK** (`hbDNN`, `hbUCP`, `hb_vp`, media/codec),
which exists **only on RDK S100 / S100P / S600 hardware** (`/usr/include/hobot`,
`/usr/hobot/lib`, or the `hobot-dnn` / `hobot-media` conda packages). Therefore:

- The **C++ library, examples, and the `bcdl_py` module must be built and run on
  an S100 / S100P / S600 board** (aarch64, Ubuntu 22.04, GCC 11, CMake ≥ 3.22).
- **Do not try to compile on a non-board host.** Editing, code review, and the
  Engine-free NumPy decode tests work anywhere; everything that touches the SDK
  does not.

A typical loop is "edit on a workstation, sync to the board, build & test there."
The board's Python environment is conda-managed (the `bcdl` env).

## Project layout

```
include/bcdl/, src/   library: core / backend / preproc / media / tasks / tracks / pipeline
  core/        SysMem · Task · Status · MemPool        (libhbucp -> hbUCP*)
  backend/     Engine, output readers                  (libdnn  -> hbDNN*)
  preproc/     CPU letterbox + BGR->NV12; VP units     (hb_vp)
  media/       JpegCodec (JPU) · VideoCodec H.264/H.265 (VPU)
  tasks/       det · cls · pose · seg · obb · semseg · depth · ocr
  tracks/      ByteTrack multi-object tracker
  pipeline/    sync / async detection · tracking · stereo
python/        nanobind bindings (NumPy <-> tensors)
examples/      standalone C++ programs
tests/         pytest suite (NumPy decode + on-board end-to-end)
benchmarks/    board benchmark driver + results
scripts/       sync / build / bootstrap / fetch-models / bench
docs/          API.md, CPP_API.md (Python + C++ API reference)
```

## Getting set up

One-time setup on the board creates the conda env from
[`env/environment.yml`](env/environment.yml):

```bash
scripts/bootstrap_board.sh
```

Then populate models if you'll run the on-board tests/benchmarks (the `.hbm`
files are large and are **not** committed; BCDL redistributes no weights):

```bash
scripts/fetch_models.sh        # writes into models/ (gitignored)
```

Helper scripts default to an ssh host alias `rdk` and a remote path
`projects/bcdl`; override with environment variables to point at your own board:

```bash
BOARD=myboard DEST=path/to/bcdl scripts/sync.sh
```

## Build

Iterate from a workstation, build on the board:

```bash
scripts/sync.sh                # rsync working tree -> board (fast, uncommitted work)
scripts/board_build.sh         # cmake + ninja in build/ on the board
scripts/board_build.sh --run   # build, then run det_infer ($HBM)
scripts/board_build.sh --clean # wipe build/ first
```

Or directly on the board, inside the conda env:

```bash
conda activate bcdl
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds the `bcdl` library, the C++ examples, and the `bcdl_py` Python
module. To build the pip-installable wheel (scikit-build-core): `pip install .`.

## Tests

Add or update tests for **every behavior change**. The suite has two layers:

1. **Decode math — Engine-free NumPy tests.** These pin the post-processing
   (NMS, mask assembly, keypoints, CTC, rotated-IoU, argmax, …) through the
   `decode_*` bindings and need **no board and no models**. Run them anywhere:

   ```bash
   PYTHONPATH=build:python pytest tests/test_detection.py tests/test_pose.py \
       tests/test_obb.py tests/test_instance_seg.py tests/test_depth_seg.py \
       tests/test_depth_seg_py.py tests/test_classification.py tests/test_ocr.py
   ```

2. **On-board, real-model end-to-end.** These validate the full BPU / codec /
   pipeline path on real `.hbm` models; each test **skips cleanly** if its model
   or image is absent. Run on the board with `models/` populated:

   ```bash
   PYTHONPATH=build:python pytest tests/
   ```

With BCDL installed (conda channel or a pip wheel) you can drop `PYTHONPATH` and
run `pytest tests/` against the installed module.

**Guidelines**

- New post-processing logic gets a deterministic NumPy test through a `decode_*`
  binding so it is reproducible off-board.
- New model-backed behavior gets an on-board test that skips when its model is
  missing (follow the existing patterns in `tests/test_board_models.py`).
- Keep tests deterministic — no network, fixed seeds, tolerances over exact
  float equality where appropriate.

## Coding conventions

- **Language** — C++17. Headers are `.h`, implementations `.cc`. Everything in
  namespace `bcdl`.
- **Errors** — use `BCDL_CHECK(...)` which raises `bcdl::Error`; don't introduce
  ad-hoc error paths.
- **Memory / cache discipline** — buffers are `hbUCPMallocCached`; the CPU must
  `cleanCache()` after writing an input and `invalidateCache()` before reading a
  device-written output. `Engine` handles this on the infer path — route through
  it rather than bypassing the cache flushes.
- **Zero-copy is the default** — BPU tensors, JPU/VPU images, and VP buffers all
  share `hbUCPSysMem`; prefer passing buffers through the fabric over copying.
- **OpenCV is optional** — image ops are guarded by `#ifdef BCDL_HAVE_OPENCV`
  with a hand-written fallback; keep both paths working when you touch them.
- **Style** — match the surrounding code (naming, comment density, structure).
  Post-processing ported from the CUDA stack (ccdl) becomes CPU/NEON here.

## Commits and changelog

- Keep commits focused and the message in the imperative mood.
- Follow the existing [Conventional Commits](https://www.conventionalcommits.org/)
  style in the log (`feat:`, `fix:`, `build:`, `docs:`, `test:`, …); a scope is
  encouraged, e.g. `fix(tasks): ...`.
- Record user-visible changes under `[Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md). The project follows
  [Keep a Changelog](https://keepachangelog.com/) and
  [Semantic Versioning](https://semver.org/). The version source of truth is the
  `project(... VERSION ...)` in `CMakeLists.txt`, kept in sync with
  `pyproject.toml`.

## Pull requests

Before opening a PR:

1. The relevant tests pass — at minimum the off-board decode tests; the on-board
   suite too if your change touches the SDK / model / pipeline paths.
2. New or changed behavior is covered by a test.
3. `CHANGELOG.md` notes any user-visible change.
4. The diff is focused; unrelated cleanups go in their own PR.

In the PR description, say **what** changed and **why**, note whether it was
validated on a board (and which: S100 / S100P / S600, software version), and include
before/after numbers for anything performance-related.

## Reporting bugs

Open an issue with:

- Board model and software version (e.g. S100P, sw 4.0.5), and how BCDL was
  obtained (conda package versions or a source build commit).
- The model/task involved and input shape, a minimal repro (command or short
  script), and the full error / unexpected output.
- For correctness issues, what you expected versus what you got.

## Scope: what belongs here

BCDL is an **on-board runtime** library: load a compiled `.hbm`, run inference,
pre/post-process, hardware codec, and pipeline — all on the board.

**Out of scope:** model conversion. ONNX → `.hbm`, PTQ calibration, and cosine
verification are done offline with the D-Robotics OpenExplorer toolchain by
separate projects; BCDL only consumes the finished `.hbm`. Changes that add a new
task decoder, media path, pipeline, or binding are in scope; a conversion /
calibration toolchain is not.

# BCDL Python API reference

Complete reference for the `bcdl` Python module (the nanobind bindings over the
C++ core). For the C++ API see [`CPP_API.md`](CPP_API.md) (the public headers
under [`include/bcdl/`](../include/bcdl/) are the source of truth); the names map
1:1 (Python snake_case ⇄ C++ camelCase).

- [Install](#install) · [Two ways to use BCDL](#two-ways-to-use-bcdl) ·
  [Conventions](#conventions)
- Engine: [Engine](#engine)
- Preprocessing: [Letterbox & geometry](#letterbox--geometry) ·
  [NV12 helpers](#nv12-helpers)
- Tasks: [Detection](#detection) · [Classification](#classification) ·
  [Pose](#pose) · [Instance segmentation](#instance-segmentation) ·
  [Oriented boxes (OBB)](#oriented-boxes-obb) ·
  [Semantic segmentation](#semantic-segmentation) · [Depth](#depth) ·
  [Stereo](#stereo) · [OCR](#ocr)
- Streaming & tracking: [Tracking (ByteTrack)](#tracking-bytetrack) ·
  [TrackingPipeline](#trackingpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline)
- Media: [Images & codecs](#images--codecs)

> Naming: snake_case functions and `lower_case` config fields are the Python
> spelling of the C++ camelCase members. A few decoders also have a camelCase
> alias (`decodeDepth`, `decodeSeg`) for parity with the C++ names.

## Install

See the [README](../README.md#install-conda) for the full table. The fast path:

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bcdl
python -c "import bcdl; print(bcdl.__version__)"
```

`import bcdl` pulls the C++ library and the packaged hobot SDK transitively, so
nothing else is needed on an RDK S100 / S100P / S600 board.

## Two ways to use BCDL

Every task is available at two levels — pick per how much you want BCDL to own:

1. **High-level task classes** (`Detector`, `Classifier`, `PoseEstimator`,
   `Segmenter`, `DepthEstimator`, …). You give them an `Engine` and a config;
   they run inference and post-process in one call. BCDL owns the device
   buffers, cache flushes, and dequantization.

2. **Pure `decode_*` functions** (`decode`, `decode_pose`, `decode_obb`,
   `decode_seg`, `decode_ctc`, …). These take **float32 NumPy arrays** (a model
   output you already have) and return the same result objects — **no Engine, no
   board, no model needed.** This is the path the deterministic tests use, and
   the way to plug BCDL's post-processing onto an output you produced elsewhere.

   > **Quantization caveat:** `decode_*` assume already-dequantized **float32**.
   > For an `F16` output, cast first (`np.asarray(out, np.float32)`). For a
   > **quantized int8/int16** output they would read raw integers and produce
   > wrong results — use the high-level class instead, which dequantizes via the
   > tensor's scale/zero-point.

## Conventions

- **Images are `HxWx3` uint8 BGR** (OpenCV order). Pipelines reject non-uint8
  input loudly rather than silently detecting nothing.
- **Coordinates come back in original-image pixels.** Tasks that need to undo
  the letterbox take a `LetterboxInfo` in `postprocess(...)`.
- **NV12 two-input models:** the standard RDK YOLO export takes two inputs
  `[Y, UV]`; the `*Detector.detect([...], lb)` wrappers set inputs in order.
- **Preprocessing ownership:** the high-level *task classes* (`Detector`, etc.)
  leave preprocessing to you (model layouts vary). The *pipelines*
  (`TrackingPipeline`, `AsyncDetectionPipeline`, `StereoPipeline`) own
  preprocessing in C++ and take a raw BGR frame.
- **`__version__`** is `bcdl.__version__`.

---

## Engine

Loads a compiled `.hbm` and runs BPU inference, handling the cache discipline
(clean inputs before infer, invalidate outputs before read) for you.

```python
import numpy as np, bcdl

engine = bcdl.Engine("model.hbm")            # optional model_name=""
print(engine.num_inputs, engine.num_outputs)
print(engine.input_shape(0), engine.input_dtype(0))
print(engine.output_shape(0), engine.output_dtype(0))

# Generic infer: list of input arrays in -> list of output arrays out.
outs = engine.infer([x])                     # outs[i] reshaped to output_shape(i)
```

| member | description |
|--------|-------------|
| `Engine(hbm_path, model_name="")` | Load a model. `model_name` selects one packaged model if the `.hbm` holds several. |
| `model_name -> str` | The active model's name. |
| `num_inputs / num_outputs -> int` | Tensor counts. |
| `input_shape(i) / output_shape(i) -> list[int]` | Tensor shape. |
| `input_bytes(i) -> int` | Allocated device-buffer size for input `i` (after BPU stride alignment). |
| `input_dtype(i) / output_dtype(i) -> str` | NumPy dtype string (e.g. `"float32"`, `"float16"`, `"int8"`). |
| `infer(inputs, timeout_ms=0) -> list[np.ndarray]` | Copy inputs into device buffers, run, return one array per output. `timeout_ms=0` blocks. |

> The high-level task classes wrap an `Engine` and call into the native
> post-processor directly (no per-call NumPy round-trip), so prefer them for a
> single task; use `Engine.infer()` when you want the raw output tensors.

---

## Letterbox & geometry

Aspect-preserving fit and coordinate mapping between original and model space.

```python
lb = bcdl.compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True)
mx, my = lb.fwd_x(x), lb.fwd_y(y)   # original -> model pixel
ox, oy = lb.inv_x(x), lb.inv_y(y)   # model -> original pixel

# One-shot CPU letterbox of a uint8 image (needs OpenCV):
canvas, lb = bcdl.letterbox_numpy(img, dst_w, dst_h, pad=114)
```

- **`compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True) -> LetterboxInfo`**
- **`LetterboxInfo`** — fields `scale`, `pad_x`, `pad_y`, `src_w/h`, `dst_w/h`
  (all read/write); methods `fwd_x/fwd_y` (orig→model) and `inv_x/inv_y`
  (model→orig). Pass it to every detection-family `postprocess(...)` so boxes
  come back in original pixels.
- **`letterbox_numpy(img, dst_w, dst_h, pad=114) -> (canvas, LetterboxInfo)`** —
  geometry matches the C++ VP letterbox; requires OpenCV.

## NV12 helpers

```python
nv12 = bcdl.bgr_to_nv12(bgr)         # (H*3//2, W) uint8; even dims; needs OpenCV
```

- **`bgr_to_nv12(bgr) -> np.ndarray`** — packed NV12, feeds `vp_image_from_nv12`
  and the JPU encoder. Also see [Images & codecs](#images--codecs).

## GDC hardware ops (board-only)

Fixed geometric transforms on the VPS GDC engine — NV12 in/out, CPU idle
during the op. `None` off-board / without GDC support. Full semantics + the
reverse-engineered CUSTOM-grid notes: [docs/GDC.md](GDC.md).

```python
# hardware letterbox (warp LUT generated at construction)
g = bcdl.GdcLetterbox(in_w, in_h, out_w, out_h, pad=114)
dst = g.run(src_vpimage)                    # NV12 VpImage -> NV12 VpImage

# hardware dense remap (cv2.remap semantics; LUT generated at runtime)
g = bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)   # (out_h,out_w) f32 maps
dst = g.run(src_vpimage)
```

- **`GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)`** — arbitrary FIXED
  warp, `out(x,y) = in(map_x[y,x], map_y[y,x])`; built for stereo
  rectification. `grid_step` must divide both output dims. 2448×2048: 6.3 ms
  wall / ~1 ms CPU vs 14.7 ms all-cores `cv2.remap`; matches cv2 to p99 ≤ 2
  grey-levels. `BCDL_GDC_TIMING=1` prints copy/op breakdown.
- **`GdcLetterbox(in_w, in_h, out_w, out_h, pad=114)`** — letterbox on GDC; the
  warp LUT is generated at construction (no offline `.bin`). `.info` returns the
  `LetterboxInfo`. 1920×1080 → 640×640: 0.97 ms wall, ~0.3 ms of it CPU. Geometry
  matches the CPU letterbox exactly; the resamplers alias differently on
  high-frequency detail — see [docs/GDC.md](GDC.md).

---

## Detection

Two decoder families:

- **Single-tensor** (`DecodeLayout.YoloV8` / `YoloV5`) — one float output tensor.
  Use `DetectConfig` + `decode(...)` / `Detector`.
- **Anchor-free LTRB multi-scale** (YOLO26 / standard RDK NV12 export) — a
  `(cls, box)` output pair per stride. Use `YoloLtrbConfig` + `YoloLtrbDetector`.

```python
# Single-tensor, numpy path:
cfg = bcdl.DetectConfig(); cfg.num_classes = 80; cfg.input_w = cfg.input_h = 640
out = engine.infer([x])[0].astype(np.float32)      # (N, 4+nc) or (4+nc, N)
dets = bcdl.decode(out, cfg, lb)
for d in dets:
    print(d.class_id, d.score, d.x1, d.y1, d.x2, d.y2)

# Anchor-free LTRB, high-level (NV12 two-input):
det = bcdl.YoloLtrbDetector(engine, bcdl.YoloLtrbConfig())
dets = det.detect([y_plane, uv_plane], lb)
```

**`DetectConfig`** — `input_w`, `input_h`, `num_classes`, `conf_thresh`,
`iou_thresh`, `max_dets`, `layout` (`DecodeLayout`), `channels_first`,
`apply_sigmoid`.

**`DecodeLayout`** enum — `YoloV8`, `YoloV5`.

**`Detection`** — `x1, y1, x2, y2, score, class_id` (all read/write); `__repr__`.

**`YoloLtrbConfig`** — `num_classes`, `conf_thresh`, `iou_thresh`, `max_dets`,
`strides` (e.g. `[8,16,32]`), `reg_max` (DFL bins; `0`/`1` = direct LTRB).

Functions / classes:

| symbol | signature |
|--------|-----------|
| `decode` | `decode(output: f32 ndarray, config: DetectConfig, letterbox: LetterboxInfo) -> list[Detection]` |
| `nms` | `nms(dets, iou_thresh, max_dets=300) -> list[int]` (indices to keep) |
| `iou` | `iou(a: Detection, b: Detection) -> float` |
| `Detector` | `Detector(engine, config, output_index=0)`; `.detect(model_input, lb, timeout_ms=0)`, `.config` |
| `YoloLtrbDetector` | `YoloLtrbDetector(engine, config=None, output_base=0)`; `.detect([inputs], lb, timeout_ms=0)`, `.postprocess(lb)`, `.config` |

---

## Classification

```python
cfg = bcdl.ClsConfig(); cfg.top_k = 5; cfg.apply_softmax = True
clf = bcdl.Classifier(engine, cfg)
for r in clf.classify(x):                 # x: single array or [Y, UV]
    print(r.class_id, r.score)

# numpy path:
top = bcdl.decode_classification(logits.astype(np.float32), cfg)
```

- **`ClsConfig`** — `top_k`, `apply_softmax`.
- **`ClsResult`** — `class_id`, `score`; `__repr__`.
- **`decode_classification(logits, config) -> list[ClsResult]`** — flat logit
  vector top-k.
- **`Classifier(engine, config=None, output_index=0)`** — `.classify(inputs, timeout_ms=0)`,
  `.postprocess()`, `.config`.

## Pose

LTRB multi-scale head: a person box plus K keypoints. NV12 two-input.

```python
cfg = bcdl.PoseConfig(); cfg.num_keypoints = 17
est = bcdl.PoseEstimator(engine, cfg)
for p in est.detect([y_plane, uv_plane], lb):
    print(p.score, [(k.x, k.y, k.score) for k in p.keypoints])
```

- **`PoseConfig`** — `num_keypoints`, `conf_thresh`, `iou_thresh`, `max_dets`,
  `strides`.
- **`PoseDetection`** — `x1, y1, x2, y2, score, class_id`, `keypoints: list[Keypoint]`.
- **`Keypoint`** — `x, y, score`.
- **`PoseEstimator(engine, config=None, output_base=0)`** — `.detect([inputs], lb, timeout_ms=0)`,
  `.postprocess(lb)`, `.config`.
- **`decode_pose(cls, box, kpt, config, letterbox) -> list[PoseDetection]`** — numpy
  path; `cls/box/kpt` are lists of per-stride `[H,W,1]` / `[H,W,4]` / `[H,W,K*3]`
  float arrays.

## Instance segmentation

LTRB boxes plus a per-instance binary mask assembled from a prototype tensor.

```python
cfg = bcdl.InstanceSegConfig(); cfg.compute_masks = True
seg = bcdl.InstanceSegmenter(engine, cfg)
for m in seg.detect([y_plane, uv_plane], lb, orig_w, orig_h):
    print(m.class_id, m.score, m.mask.shape)     # m.mask: (H, W) uint8 0/1
```

- **`InstanceSegConfig`** — `conf_thresh`, `iou_thresh`, `max_dets`, `strides`,
  `proto_index`, `compute_masks` (set `False` to skip mask assembly for speed).
- **`InstanceMask`** — `x1, y1, x2, y2, score, class_id`, `mask_w, mask_h`,
  `mask` (`(H,W)` uint8 0/1; empty when `compute_masks=False`).
- **`InstanceSegmenter(engine, config=None, output_base=0)`** —
  `.detect([inputs], lb, orig_w, orig_h, timeout_ms=0)`, `.postprocess(lb, orig_w, orig_h)`,
  `.config`.
- **`decode_instance_seg(cls, box, mc, proto, config, letterbox, orig_w, orig_h) -> list[InstanceMask]`**
  — numpy path; `cls/box/mc` per-stride `[H,W,nc]`/`[H,W,4]`/`[H,W,np]`, `proto`
  `[mH,mW,np]`.

## Oriented boxes (OBB)

LTRB plus an angle; rotated-IoU NMS.

```python
cfg = bcdl.ObbConfig(); cfg.num_classes = 15
obb = bcdl.ObbDetector(engine, cfg)
for o in obb.detect([y_plane, uv_plane], lb):
    r = o.rrect
    print(o.class_id, o.score, r.cx, r.cy, r.w, r.h, r.angle)  # angle in rad
```

- **`ObbConfig`** — `num_classes`, `conf_thresh`, `iou_thresh`, `max_dets`,
  `strides`, `regularize`, `angle_offset_rad`, `angle_sign`.
- **`RotatedBox`** — `cx, cy, w, h, angle` (radians).
- **`ObbDetection`** — `rrect: RotatedBox`, `score`, `class_id`.
- **`ObbDetector(engine, config=None, output_base=0)`** — `.detect([inputs], lb, timeout_ms=0)`,
  `.postprocess(lb)`, `.config`.
- **`decode_obb(cls, box, angle, config, letterbox) -> list[ObbDetection]`** — numpy
  path; per-stride `[H,W,nc]`/`[H,W,4]`/`[H,W,1]`.
- **`rotated_iou(a_cx,a_cy,a_w,a_h,a_angle, b_cx,b_cy,b_w,b_h,b_angle) -> float`**.

## Semantic segmentation

Argmax over a logit tensor (or pass through a pre-argmaxed id tensor) to a
per-pixel label map.

```python
cfg = bcdl.SegConfig(); cfg.num_classes = 19
seg = bcdl.Segmenter(engine, cfg)
mask = seg.segment(x)                 # SegMask
labels = mask.labels                  # (H, W) int32 class ids
bgr = bcdl.seg_colorize(mask)         # (H, W, 3) uint8 palette image

# numpy path:
mask = bcdl.decode_seg(logits.astype(np.float32), cfg)
```

- **`SegConfig`** — `num_classes`, `channels_first` (NCHW vs NHWC),
  `argmaxed` (set `True` if the model already emits ids).
- **`SegMask`** — `width`, `height`, `num_classes`, `labels` (`(H,W)` int32).
- **`decode_seg(output, config) -> SegMask`** (alias `decodeSeg`).
- **`seg_colorize(mask) -> np.ndarray`** — `(H,W,3)` uint8 fixed palette.
- **`Segmenter(engine, config=None, output_index=0)`** — `.segment(model_input, timeout_ms=0)`,
  `.postprocess()`, `.config`.

## Depth

Single-channel depth / disparity to a float map, with colorization helpers.

```python
cfg = bcdl.DepthConfig(); cfg.normalize = True
est = bcdl.DepthEstimator(engine, cfg)
dm = est.estimate(x)                  # DepthMap
arr = dm.data                         # (H, W) float32
bgr = bcdl.depth_colorize(dm)         # turbo colormap (H, W, 3) uint8
g8 = bcdl.depth_to_gray8(dm)          # (H, W) uint8

# numpy path:
dm = bcdl.decode_depth(out.astype(np.float32), cfg)
```

- **`DepthConfig`** — `width`, `height`, `normalize` (scale to [0,1]),
  `clip_lo`, `clip_hi`.
- **`DepthMap`** — `width`, `height`, `vmin`, `vmax`, `data` (`(H,W)` float32).
- **`decode_depth(output, config) -> DepthMap`** (alias `decodeDepth`).
- **`depth_colorize(dm) -> np.ndarray`** · **`depth_to_gray8(dm) -> np.ndarray`**.
- **`DepthEstimator(engine, config=None, output_index=0)`** — `.estimate(model_input, timeout_ms=0)`,
  `.postprocess()`, `.config`.

## Stereo

Two rectified images to a disparity map, optionally metric depth and a validity
mask. Pixel normalization is fused into the `.hbm`; the C++ core does the fit +
BGR→RGB + F32 NCHW pack. **The `fit` mode must match how the model was
calibrated.**

```python
cfg = bcdl.StereoConfig()
cfg.fit = bcdl.StereoFit.Crop      # or Resize — MUST match calibration
cfg.fx, cfg.baseline = 700.0, 0.12 # enable metric depth (z = fx*baseline/disp)
cfg.valid_mask = True
pipe = bcdl.StereoPipeline(engine, cfg)

res = pipe.process(left_bgr, right_bgr)
disp = res.disparity.data          # (H, W) float32 disparity (a DepthMap)
depth = res.depth                  # (H, W) float32 metric depth, or shape (0,) if off
valid = res.valid                  # (H, W) uint8 mask, or shape (0,) if off
```

- **`StereoFit`** enum — `Resize`, `Crop`.
- **`StereoConfig`** — `input_w`, `input_h`, `fit`, `to_rgb`, `left_index`,
  `right_index`, `output_index`, `fx`, `baseline`, `valid_mask`, `disp_min`,
  `max_disp`, `left_margin`, `lr_check`, `lr_thresh`.
- **`StereoResult`** — `disparity: DepthMap`, `depth` (numpy, empty when no
  `fx`/`baseline`), `valid` (numpy uint8, empty when `valid_mask=False`).
- **`StereoPipeline(engine, config=None)`** — `.process(left, right) -> StereoResult`,
  `.input_w`, `.input_h`.
- numpy helpers: **`pack_stereo_input(bgr, out_h, out_w, fit=StereoFit.Resize, to_rgb=True)`**,
  **`disparity_to_depth(disp, fx, baseline)`**,
  **`stereo_valid_mask(disp, disp_min=0.0, max_disp=192.0, left_margin=..., lr_check=False, lr_thresh=...)`**.

## OCR

Full PP-OCRv5 three-stage pipeline, each stage usable on its own. Pure decoders
(`decode_dbnet`, `decode_cls_dir`, `decode_ctc`, `load_char_dict`) need no model.

```python
chars = bcdl.load_char_dict("ppocrv5_dict.txt")

det = bcdl.DbTextDetector(engine_det)            # DBNet detect
boxes = det.postprocess(lb)                      # list[TextBox], 4-point rotated
cls = bcdl.TextAngleClassifier(engine_cls)       # 0/180 direction
rec = bcdl.TextRecognizer(engine_rec, "ppocrv5_dict.txt")  # CRNN/CTC
# (crop each box, run cls then rec on the crop's engine; see examples/ocr_demo)

# numpy decoders:
boxes = bcdl.decode_dbnet(prob.astype(np.float32), bcdl.DbConfig(), lb)
dir_  = bcdl.decode_cls_dir(logits.astype(np.float32), thresh=0.9)
text  = bcdl.decode_ctc(logits.astype(np.float32), chars)
```

- **`DbConfig`** — `bin_thresh`, `box_thresh`, `unclip_ratio`, `min_size`,
  `connectivity`.
- **`TextBox`** — `x1, y1, x2, y2`, `score`, `points` (`(4,2)` float, clockwise,
  original pixels).
- **`RecResult`** — `text`, `score`.
- **`ClsDirResult`** — `label`, `score`, `flip180`.
- **`load_char_dict(path) -> list[str]`** — one token per line.
- **`decode_dbnet(prob, config, letterbox) -> list[TextBox]`** — connected-component +
  unclip on a `(H,W)` probability map.
- **`decode_cls_dir(logits, thresh=0.9) -> ClsDirResult`**.
- **`decode_ctc(logits, dict) -> RecResult`** — CTC greedy decode of a `(T,C)`
  array.
- Engine-bound: **`DbTextDetector(engine, config=None, output_index=0)`**,
  **`TextAngleClassifier(engine, thresh=0.9, output_index=0)`**,
  **`TextRecognizer(engine, dict_path, output_index=0)`** — each has
  `.postprocess(...)`.

See [`examples/ocr_demo.cc`](../examples/ocr_demo.cc) for the full det→cls→rec
wiring (crop ordering and the dict gotchas are handled there).

---

## Tracking (ByteTrack)

Model-free: feed each frame's detections, get stable track ids back.

```python
tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
for frame in stream:
    dets = detector.detect(...)              # list[Detection]
    for t in tracker.update(dets):
        print(t.track_id, t.class_id, t.x1, t.y1, t.x2, t.y2)
```

- **`ByteTrackConfig`** — `track_thresh`, `high_thresh`, `match_thresh`,
  `track_buffer`, `frame_rate`.
- **`Track`** — `track_id`, `x1, y1, x2, y2`, `score`, `class_id`; `__repr__`.
- **`ByteTracker(config=None)`** — `.update(detections) -> list[Track]`,
  `.reset()`, `.config`.

## TrackingPipeline

Detect-and-track in one call; all preprocessing in C++. Needs an NV12-input
YOLO `.hbm`.

```python
pipe = bcdl.TrackingPipeline(engine)          # det_config, track_config optional
for frame in stream:                          # frame: HxWx3 uint8 BGR
    for t in pipe.process(frame):
        print(t.track_id, t.class_id, t.score)
print(pipe.last_detections)                   # pre-association dets of last frame
```

- **`PipelineConfig`** — `input_w`, `input_h`, `detect` (`DetectConfig`),
  `output_index`, `pad_value`, `head` (`DetectHead`), `ltrb_strides`.
- **`DetectHead`** enum — `Auto`, `SingleTensor`, `YoloLtrb`.
- **`TrackingPipeline(engine, det_config=None, track_config=None)`** —
  `.process(bgr) -> list[Track]`, `.last_detections`, `.reset()`.

## AsyncDetectionPipeline

Streaming detection with CPU preprocessing of later frames overlapped against BPU
infer+decode of earlier ones. `submit()` / `next()` block (backpressure / wait)
but **release the GIL**. Results return in submission order. Needs an NV12-input
YOLO `.hbm`.

```python
cfg = bcdl.PipelineConfig(); cfg.detect.num_classes = 80
pipe = bcdl.AsyncDetectionPipeline(engine, cfg, depth=3)

for i, frame in enumerate(stream):            # frame: HxWx3 uint8 BGR
    pipe.submit(frame)                        # blocks if full
    if i >= 3:
        for d in pipe.next():                 # in submission order
            ...
pipe.finish()
while (dets := pipe.next()) is not None:
    ...                                       # drain in-flight frames
```

- **`AsyncDetectionPipeline(engine, config=None, depth=3)`**:
  - `submit(bgr) -> bool` — enqueue (bytes copied; array reusable immediately).
    Blocks while full; returns `False` after `finish()`.
  - `next() -> list[Detection] | None` — pop next result; `None` once finished
    **and** drained.
  - `finish()` — signal end of stream (idempotent; also runs on GC).
  - `head -> DetectHead` — resolved decoder family.
  - `profile() -> StageProfile` — per-stage *service* time
    (`preproc_ms`/`infer_ms`/`postproc_ms` totals + `*_per_frame()`); the slowest
    stage bounds throughput. Read after `finish()` + drain.

## AsyncVideoDetectionPipeline

The whole compressed-video → detections path in C++ (`decode ‖ nv12→bgr ‖ preproc
‖ infer+NMS`, four overlapped stages). **Python only pumps bytes** — feed Annex-B
chunks (e.g. an `ffmpeg -c copy` RTSP/mp4 stream) and read detections; all
decode/convert/detect threads run in C++ with the GIL released. This is what lets
a thin Python driver hit the C++ decode-bound ceiling (**~441 FPS @1080p H.264**)
instead of the ~81 FPS a Python-orchestrated decode loop is capped at.

```python
cfg = bcdl.PipelineConfig(); cfg.detect.num_classes = 80
pipe = bcdl.AsyncVideoDetectionPipeline(engine, cfg, bcdl.VideoType.H264, depth=4)

while chunk := ffmpeg.stdout.read(65536):     # ffmpeg -c copy (no software decode)
    pipe.submit(chunk)                        # AUs segmented + VPU-decoded in C++
    while (dets := pipe.next_nowait()) is not None:
        ...                                   # drain what's ready
pipe.finish()
while (dets := pipe.next()) is not None:      # blocking drain of last frames
    ...
```

- **`AsyncVideoDetectionPipeline(engine, config=None, codec=VideoType.H264, depth=4)`**:
  - `submit(bytes) -> bool` — feed **Annex-B** bytes; blocks on backpressure.
    An MP4 is *not* Annex-B (AVCC length prefixes, SPS/PPS in the `avcC` box), so
    feeding one yields zero frames. Demux it first — container only, pixels
    untouched, VPU still the only decoder:
    `ffmpeg -i in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 -`. See
    `load_annexb()` in [`examples/video_det_demo.py`](../examples/video_det_demo.py).
  - `next() -> list[Detection] | None` — blocking pop in decode order.
  - `next_nowait() -> list[Detection] | None` — non-blocking pop.
  - `finish()`, `profile() -> StageProfile` (incl. `decode_ms`, `cvt_ms`).
- Video decode handles **H.264 and H.265** (a hierarchical-GOP HEVC stream
  decodes its base temporal layer only — see the VideoDecoder note above).
- See [`examples/rtsp_det_demo.py`](../examples/rtsp_det_demo.py) — a thin RTSP
  driver built on this (ffmpeg pipe → `submit` → `next_nowait`).

### Video demo (mp4/h264 in → detect → mp4 out)

[`examples/video_det_demo.py`](../examples/video_det_demo.py) is the end-to-end
Python demo: **VPU** decode → `AsyncDetectionPipeline` (BPU) → draw → **VPU**
encode → `.mp4`. It reads raw `.h264/.h265` **or** `.mp4/.mov` (MP4 is demuxed to
Annex-B with `ffmpeg -c copy`; the VPU still does the actual decode), and muxes
the VPU's elementary stream into `.mp4` with `ffmpeg -c copy` (container only, no
re-encode). Prints the per-stage `profile()` distribution.

```bash
python examples/video_det_demo.py det.hbm in.mp4 out.mp4          # mp4 -> mp4
python examples/video_det_demo.py det.hbm in.h264 out.mp4 300 4   # [max_frames] [depth]
```

Note: decode and encode share the single VPU core, so the full round-trip runs
slower (~104 FPS @1080p yolo26n, encode-bound at 3.56 ms/frame) than the
decode-only detect path (~441 FPS).

---

## Images & codecs

The unified shared-memory image plus the JPU (JPEG) and VPU (H.264/H.265)
hardware codecs. These allocate cached device buffers and drive the media units,
so they only run on the board.

```python
# JPEG (JPU) — one-shot helpers:
jpg = bcdl.jpeg_encode(bgr, quality=80)       # bytes
img = bcdl.jpeg_decode(jpg)                    # VpImage (NV12)
planes = img.to_numpy()                        # see VpImage.to_numpy below

# Reuse a decoder for a stream (re-creating one per call adds ~5 ms JPU setup):
dec = bcdl.JpegDecoder()
for blob in blobs:
    vp = dec.decode(blob)

# H.264 / H.265 (VPU):
ec = bcdl.VideoEncConfig(); ec.type = bcdl.VideoType.H264
ec.width, ec.height, ec.bitrate_kbps, ec.framerate = 1280, 720, 4000, 30
enc = bcdl.VideoEncoder(ec)
chunk = enc.encode(vp_image)                   # bytes (may be empty if buffered)

dc = bcdl.VideoDecConfig(); dc.type = bcdl.VideoType.H265   # or .H264
dec = bcdl.VideoDecoder(dc)
frame = dec.decode(nal_bytes)                  # VpImage or None while buffering
# Decoupled feed/drain (handles H.265 reorder): feed arbitrary bytes, drain in
# display order, flush the reorder tail at end-of-stream.
dec.feed(chunk_bytes)                           # queue bytes (no AU split needed)
while (f := dec.receive(0)) is not None: ...    # non-blocking drain (0 = no wait)
while (f := dec.flush()) is not None: ...        # after last feed: reorder tail
```

> **H.265 note:** the decoder uses the reorder-correct `media_codec` API, so H.265
> works (an older per-AU model timed out on it). A **hierarchical-GOP** HEVC stream
> (Hikvision SVC / "H.265+"/smart-codec) decodes **base temporal layer only**
> (~1/4 of frames) — an SoC-SDK limitation; disable the camera's SVC mode for full
> rate. H.264 and non-hierarchical HEVC decode fully.

- **`ImageFormat`** enum — `Y`, `NV12`, `RGB`, `BGR`.
- **`VpImage(width, height, format)`** — `width`, `height`, `format`, `valid`;
  `.to_numpy()` copies out honoring the device row stride: `BGR/RGB -> (H,W,3)`,
  `Y -> (H,W)`, `NV12 -> flat (W*H*3//2)` (Y plane then interleaved UV).
- **`vp_image_from_bgr(bgr) -> VpImage`** · **`vp_image_from_nv12(nv12, width, height) -> VpImage`**.
- **`JpegEncoder(width, height, quality=50, format=NV12)`** — `.encode(VpImage) -> bytes`
  (width must align to 16, height to 8).
- **`JpegDecoder(out_format=NV12)`** — `.decode(bytes) -> VpImage`. **Reuse one
  instance** across a stream; constructing per call costs ~5 ms.
- **`jpeg_encode(bgr, quality=50) -> bytes`** · **`jpeg_decode(data) -> VpImage`**
  — one-shot convenience wrappers (need OpenCV + even dims).
- **`VideoType`** enum — `H264`, `H265`.
- **`VideoEncConfig`** — `type`, `width`, `height`, `bitrate_kbps`, `framerate`,
  `intra_period`, `format`.
- **`VideoEncoder(config)`** — `.encode(frame: VpImage) -> bytes` (empty when the
  encoder buffered the frame); `.type/.width/.height/.format`.
- **`VideoDecConfig`** — `type`, `format`, `in_buf_size`.
- **`VideoDecoder(config)`** — `.decode(data) -> VpImage | None` (`None` while
  buffering reference frames); `.type/.format`.

---

## C++ API

The C++ surface mirrors the above; configs and result structs share field names
(camelCase). Start from [`include/bcdl/bcdl.h`](../include/bcdl/bcdl.h) and the
per-area headers:

| area | header |
|------|--------|
| core (SysMem, Task, Status, MemPool) | [`core/`](../include/bcdl/core/) |
| backend (Engine, output readers) | [`backend/engine.h`](../include/bcdl/backend/engine.h) |
| preprocessing (letterbox, VpImage) | [`preproc/`](../include/bcdl/preproc/) |
| media (JPEG/video codecs) | [`media/`](../include/bcdl/media/) |
| tasks (det/cls/pose/seg/obb/depth/ocr) | [`tasks/`](../include/bcdl/tasks/) |
| tracking | [`tracks/byte_tracker.h`](../include/bcdl/tracks/byte_tracker.h) |
| pipelines | [`pipeline/`](../include/bcdl/pipeline/) |

Errors are reported via `BCDL_CHECK(...)` raising `bcdl::Error`. See
[`examples/`](../examples/) for runnable programs.

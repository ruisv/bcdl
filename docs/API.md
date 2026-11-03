# BCDL Python API reference

**English** | [简体中文](API.zh.md)

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
- Speech: [Voice pipeline (start here)](#voice-pipeline) ·
  [Wake words](#keyword-spotting-wake-word) ·
  [VAD](#voice-activity-detection-silero-vad) ·
  [Recognition](#speech-recognition-paraformer) ·
  [Commands](#voice-commands-closed-set) ·
  [Synthesis](#speech-synthesis-matcha-tts--vocos)
- Streaming & tracking: [Tracking (ByteTrack)](#tracking-bytetrack) ·
  [TrackingPipeline](#trackingpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline)
- Media: [Images & codecs](#images--codecs)

> Naming: snake_case functions and `lower_case` config fields are the Python
> spelling of the C++ camelCase members — without exception. Nine decoders used
> to carry a camelCase alias as well; those names are deprecated, resolve with a
> `DeprecationWarning`, and are no longer part of this reference.

## Install

See the [README](../README.en.md#quickstart) for the full table. The fast path:

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
| `input_name(i) / output_name(i) -> str` | Tensor name as compiled (the ONNX name). |
| `input_shape(i) / output_shape(i) -> list[int]` | Tensor shape. |
| `input_bytes(i) -> int` | Allocated device-buffer size for input `i` (after BPU stride alignment). |
| `input_packed_bytes(i) -> int` | Size of input `i` as a contiguous row-major array. Smaller than `input_bytes(i)` when the model pads a dimension. |
| `input_stride(i) -> list[int]` | Resolved byte strides of input `i`, innermost last. |
| `output_stride(i) -> list[int]` | Resolved byte strides of output `i`. Outputs pad too — reshaping the raw buffer flat shears the tensor. |
| `output_packed_bytes(i) -> int` | Size of output `i` as a contiguous row-major array. Smaller than the device buffer when the model pads a dimension. |
| `input_dtype(i) / output_dtype(i) -> str` | NumPy dtype string (e.g. `"float32"`, `"float16"`, `"int8"`). |
| `infer(inputs, timeout_ms=0) -> list[np.ndarray]` | Copy inputs into device buffers, run, return one array per output. `timeout_ms=0` blocks. |

> **Padded outputs are gathered for you.** `infer()` returns packed row-major
> arrays even when the model pads a dimension (ViTPose's `[1,133,64,48]` heatmap
> has a 256-byte row stride against 192 valid bytes). Reading the device buffer
> flat instead displaces row `r` by `r * pad` and yields a tensor that still
> looks like data — it cost a day of debugging a model that was in fact correct.

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

## Whole-body pose (133 keypoints)

TOP-DOWN, unlike `PoseEstimator`: one inference per PERSON on a crop, so it needs
a detector in front of it and its cost scales with the head count. In exchange it
resolves feet, a 68-point face and both hands.

```python
est = bcdl.WholeBodyEstimator(engine)
for box in person_boxes:                     # from any detector above
    kpts = est.estimate(bgr, box)            # 133 keypoints, original-image px
    print(sum(k.score > 0.2 for k in kpts), "visible")
```

Keypoint layout (COCO-WholeBody): `0-16` body, `17-22` feet, `23-90` face,
`91-111` left hand, `112-132` right hand.

- **`WholeBodyConfig`** — `kpt_thresh`, `blur_kernel` (DARK modulation, odd),
  `box_pad`, `mean`, `std` (RGB, applied after scaling to `[0,1]`).
- **`WholeBodyCrop`** — `x1, y1, pad_left, pad_top, padded_w, padded_h`: how a
  person box became the model canvas, and what inverts it.
- **`WholeBodyEstimator(engine, config=None, output_index=0)`** —
  `.estimate(bgr, box, timeout_ms=0)`, `.postprocess(crop)`, `.config`.
- **`wholebody_preprocess(bgr, x1, y1, x2, y2, in_w=192, in_h=256, config=None)`**
  → `(input, crop)`. Takes **BGR** like the rest of this API and flips to the RGB
  the model wants. The box is widened by `box_pad`, zero-padded to the model's
  3:4 aspect and resized — not the mmpose center/scale affine.
- **`decode_wholebody(heatmaps, crop, config=None) -> list[Keypoint]`** — numpy
  path; `heatmaps` is a channel-first `[K,H,W]` (or `[1,K,H,W]`) float array.

## Super-resolution (tiled)

The model upscales one fixed tile; an arbitrary image is cut into overlapping
tiles and cross-faded back together.

```python
sr = bcdl.SuperResolver(engine)          # scale and tile come from the model
big = sr.upscale(img)                    # BGR in, BGR out, sr.scale x larger
print(sr.last_tile_count, "tiles")
```

- **`SuperResConfig`** — `overlap` (input pixels; also the cross-fade width).
- **`SuperResolver(engine, config=None, input_index=0, output_index=0)`** —
  `.upscale(bgr, timeout_ms=0)`, `.scale`, `.tile`, `.last_tile_count`.
- **`plan_tiles(width, height, tile_w, tile_h, overlap)`** → tile origins. The
  last tile on each axis is flush with the far edge, so the real overlap there
  can exceed the requested one.
- **`tile_weight(i, len, ramp)`** — the cross-fade weight, always > 0 so the
  normalized blend needs no border special-casing.

> Tile size is a deployment decision, not just a memory one: the compiled
> `.hbm` scales with tile AREA (256x256 → 148 MB, 128x128 → 37 MB for the same
> network) at identical per-pixel throughput.

## Sparse local features (XFeat)

Repeatable keypoints with L2-normalized 64-d descriptors, plus mutual-NN
matching. Only the convolutional trunk is on the BPU (~1.0 ms); the input
normalization and all of the softmax / NMS / top-k / sampling run on the CPU.

```python
ext = bcdl.FeatureExtractor(engine)
fa, fb = ext.extract(img_a), ext.extract(img_b)
for m in bcdl.match_features(fa, fb, min_cossim=0.82):
    print(fa.xy[m.a], "->", fb.xy[m.b], m.score)
```

- **`XfeatConfig`** — `detection_thresh`, `nms_kernel` (odd), `top_k`.
- **`FeatureSet`** — `keypoints: list[Feature]`, `descriptors` as an `(N, 64)`
  array, `xy` as an `(N, 2)` array, `dim`, `len()`.
- **`Feature`** — `x, y, score`. **`FeatureMatch`** — `a, b, score`.
- **`FeatureExtractor(engine, config=None, output_base=0)`** —
  `.extract(bgr, timeout_ms=0)`, `.postprocess(scale_x, scale_y)`, `.config`.
- **`xfeat_preprocess(bgr, in_w=640, in_h=480)`** → `(input, scale_x, scale_y)`.
  Grayscale is the plain CHANNEL MEAN, not a luma weighting.
- **`decode_xfeat(feats, keypoints, heatmap, config=None, scale_x=1, scale_y=1)`**
  — numpy path; the three maps are `[C,H,W]` (or `[1,C,H,W]`) float arrays with
  64 / 65 / 1 channels.
- **`match_features(a, b, min_cossim=0.82)`** — mutual nearest neighbours.
  **Cost is `O(|a|*|b|*64)`**: about 130 ms per pair at the default `top_k` of
  4096, 8 ms at 1024. Lower `top_k` before anything else if matching dominates.

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
- **`decode_seg(output, config) -> SegMask`**.
- **`seg_colorize(mask) -> np.ndarray`** — `(H,W,3)` uint8 fixed palette.
- **`Segmenter(engine, config=None, output_index=0)`** — `.segment(model_input, timeout_ms=0)`,
  `.postprocess()`, `.config`.

## Lidar 3-D detection

Point cloud in, oriented 3-D boxes out (centre, extent, yaw in the ego frame,
metres). A pillar detector is a **pipeline**, not one model, and the pieces stay
separate because their costs are wildly different:

```python
cfg = bcdl.VoxelConfig()                               # KITTI geometry by default
vox, npts, coords, stats = bcdl.voxelize_pillars(points_xyzi, cfg)
x = bcdl.pillar_point_features(vox, npts, coords, stats.num_pillars, cfg)
feats = vfe_engine.infer([x])[0]                       # BPU: (1, C, P, 1)
canvas = bcdl.scatter_pillars(feats.reshape(feats.shape[1], -1).T[:stats.num_pillars],
                              coords[:stats.num_pillars], stats.num_pillars, cfg)
head_engine.infer([canvas])                            # BPU (backbone fused in)
boxes = bcdl.LidarDetector3d(head_engine).postprocess()
```

**Points must already be in the ego frame** — nothing downstream can tell that a
cloud is still in sensor coordinates. And **the caps are real**: `stats` reports
points dropped out of range, points dropped because their pillar was full, and
occupied columns that had no slot left, so a sweep that overflows the model says
so instead of quietly going blind.

**Two VFE forms exist and the difference is 179×.** A VFE that takes the three
raw pillar tensors expands them into per-point features inside its own graph, and
pays 290 ms for a 95 MFLOP MLP — `(pillars, points, 4)` is a huge pillar axis
against a 4-wide channel axis, which is the worst shape a convolution engine can
be given. The same MLP re-exported as a 1×1 convolution over `(1, 10, P, 32)`
runs in 1.62 ms and takes `pillar_point_features()` instead. Feed whichever the
model asks for: a one-input VFE is the convolution form. Likewise a head whose
input is the scatter canvas has the backbone fused into it and needs no separate
backbone call. See `docs/MODELS.md` for both chains and how to get them.

- **`VoxelConfig`** — `min_x`/`max_x`/`min_y`/`max_y`/`min_z`/`max_z`,
  `pillar_x`/`pillar_y`, `max_points_per_pillar`, `max_pillars`, `num_features`;
  read-only `grid_x`/`grid_y`.
- **`voxelize_pillars(points, config) -> (voxels, num_points, coords, Pillars)`** —
  `points` is `(N, >=3)` float32 x, y, z[, intensity…].
- **`pillar_point_features(voxels, num_points, coords, num_pillars, config) -> np.ndarray`**
  — `(1, num_features+6, max_pillars, max_points)`: the raw point, its offset
  from the pillar's centroid, its offset from the pillar's centre. Unfilled point
  slots stay zero — they go through the MLP too, and the max over points would
  otherwise pick up whatever the bias maps them to.
- **`scatter_pillars(features, coords, num_pillars, config) -> np.ndarray`** —
  `(1, C, grid_y, grid_x)`.
- **`Box3D`** — `x`, `y`, `z`, `dx`, `dy`, `dz`, `heading`, `label`, `score`.
- **`Anchor3dConfig`** — grid + per-class `sizes`/`centers` + `rotations`.
  **`generate_anchors_3d(config)`** returns the `(N, 7)` grid; it reproduces the
  published model's own anchor constants to 8e-06.
- **`Det3dConfig`** — `score_thresh`, `nms_iou`, `max_boxes`, `apply_sigmoid`,
  `anchors`.
- **`decode_det3d(cls, box, dir=None, anchors=..., config=...) -> list[Box3D]`**
  Passing `dir=None` leaves every heading correct only
  modulo pi.
- **`bev_iou(a, b)`** · **`nms3d(boxes, iou_thresh, max_boxes)`** — the metric
  this family's NMS is defined on is the BEV footprint, not the volume.
- **`LidarDetector3d(engine, config=None, cls_index=0, box_index=1, dir_index=2)`** —
  `.postprocess()`, `.anchors`.

## Anomaly detection (one-class)

"Is this part like the good ones." There is no class list and no training set of
defects — the model is shown only **normal** examples of one product and reports
how unlike them each patch looks.

Two consequences the API cannot hide: the model is **per-product**, so the
shipped `.hbm` is an example rather than a general detector (a new part means
retraining); and the threshold is a **per-line decision** calibrated against that
product's normal variation, not a library constant.

```python
det = bcdl.AnomalyDetector(engine)
m = det.detect(bgr)                  # one (H, W, 3) uint8 BGR image
m.score                              # image-level verdict = max of the map
heat = m.data                        # (H, W) float32, at the model's input size
mask = bcdl.anomaly_mask(m, 0.5)     # (H, W) uint8 0/255

# numpy path:
m = bcdl.decode_anomaly(grid.astype(np.float32), cfg)
```

The image score is the map's **maximum**, not its mean: a part is as anomalous as
its worst patch, and a mean would let one large clean region hide a small defect
— the case the method exists to catch.

- **`AnomalyConfig`** — `out_width`/`out_height` (0 ⇒ keep the raw grid; the
  detector defaults them to the model's input size), `blur_kernel` (odd, 0
  disables), `blur_sigma`, `threshold`. The upsample **and** the Gaussian smoothing
  are part of the reference post-processing, not decoration.
- **`AnomalyMap`** — `width`, `height`, `score`, `data` (`(H,W)` float32).
- **`decode_anomaly(output, config=AnomalyConfig()) -> AnomalyMap`**.
- **`anomaly_mask(map, threshold) -> np.ndarray`** — `(H,W)` uint8 0/255.
- **`anomaly_preprocess(bgr, in_w, in_h) -> np.ndarray`** — resize + BGR→RGB +
  `/255` + ImageNet mean/std into `(3,H,W)`.
- **`AnomalyDetector(engine, config=None, output_index=0, input_index=0)`** —
  `.detect(bgr, timeout_ms=0)`, `.postprocess()`, `.input_size`.

## Optical flow

Two frames in, a per-pixel displacement field out. Vectors are in **source
pixels**, +u right, +v down (the OpenCV convention), so the field drops straight
into `cv2.remap` once split into x/y maps.

```python
est = bcdl.OpticalFlowEstimator(engine)
flow = est.estimate(frame0, frame1)     # two (H, W, 3) uint8 BGR frames
uv = flow.data                          # (H, W, 2) float32, source pixels
bgr = bcdl.flow_colorize(flow)          # Middlebury colour wheel

# numpy path:
flow = bcdl.decode_flow(out.astype(np.float32), cfg)
epe = bcdl.flow_endpoint_error(flow_a, flow_b)
```

Preprocessing is inside the estimator on purpose: the reference implementation
feeds **BGR in 0-255** and divides by 255 *inside the graph*, so handing it RGB
or pre-normalised pixels returns a plausible field rather than an error.

- **`FlowConfig`** — `scale_x`, `scale_y` (model pixels → source pixels;
  `estimate()` sets them from the sizes it was given), `channels_first`
  (`[1,2,H,W]` vs `[1,H,W,2]`).
- **`FlowField`** — `width`, `height`, `data` (`(H,W,2)` float32 interleaved).
- **`decode_flow(output, config=FlowConfig()) -> FlowField`**.
- **`flow_colorize(flow, max_magnitude=0.0) -> np.ndarray`** — `(H,W,3)` uint8
  BGR; `0` normalises by the field's own 99th-percentile magnitude so one fast
  object cannot flatten the rest of the frame.
- **`flow_endpoint_error(a, b) -> float`** — mean endpoint error in pixels, the
  metric to use when scoring a build against its float reference.
- **`flow_preprocess(bgr, in_w, in_h) -> np.ndarray`** — one frame to `(3,H,W)`
  float, BGR and 0-255 preserved.
- **`OpticalFlowEstimator(engine, config=None, output_index=0, input0_index=0, input1_index=1)`** —
  `.estimate(frame0, frame1, timeout_ms=0)`, `.postprocess()`, `.input_size`.

## End-to-end driving (camera + lidar → trajectory)

The only head here that returns a decision rather than an observation: three
forward cameras plus a lidar sweep in, one planned ego trajectory out, along
with the perception heads it was trained jointly with. The multi-modal diffusion
decoder lives inside the compiled graph — clustered trajectory anchors go in, one
already-selected trajectory comes out.

All poses are ego frame, metres, x forward / y left, heading in radians CCW, and
are the FUTURE at a fixed interval (8 poses over 4 s for the stock model).

```python
anchors = np.load("kmeans_navsim_traj_20.npy").astype(np.float32)   # (20, 8, 2)
planner = bcdl.DrivePlanner(engine, anchors)

camera = bcdl.stitch_drive_cameras(cam_left, cam_front, cam_right)  # (3, H, W) RGB [0,1]
lidar = bcdl.lidar_bev_histogram(points_xyz)                        # (C, H, W)

status = bcdl.DriveStatus()
status.driving_command = 0          # route hint, one-hot slot
status.velocity_x = 5.0             # m/s

r = planner.plan(camera, lidar, status)
[(p.x, p.y, p.heading) for p in r.trajectory]   # the plan
[(a.x, a.y, a.score) for a in r.agents]         # detected vehicles
bgr = bcdl.seg_colorize(r.bev)                  # BEV semantic map (SegMask)
```

The two sensor builders are geometric contracts, not conveniences: a transposed
BEV grid or a status vector in the wrong order still infers and still returns a
smooth-looking plan.

- **`lidar_bev_histogram(points, config=LidarBevConfig()) -> np.ndarray`** —
  `(N,>=3)` ego-frame xyz → `(C,H,W)` occupancy in `[0,1]`. Grid axis 0 is **x
  (forward)**, axis 1 is y. **`LidarBevConfig`** — `min_x`/`max_x`/`min_y`/`max_y`,
  `pixels_per_meter`, `max_height`, `split_height`, `hist_max_per_pixel`,
  `use_ground_plane` (false ⇒ one channel, points above the split only).
- **`stitch_drive_cameras(left, front, right, config=DriveCameraConfig()) -> np.ndarray`** —
  three `(H,W,3)` BGR images → `(3,out_h,out_w)` RGB float in `[0,1]`.
  **`DriveCameraConfig`** — `crop_top`, `crop_bottom`, `crop_side` (left/right
  cameras only), `out_width`, `out_height`. The resize happens after
  concatenation, so pixels interpolate across the seams.
- **`DriveStatus`** — `driving_command` (one-hot slot), `velocity_x`, `velocity_y`,
  `accel_x`, `accel_y`. **`drive_status_vector(status, num_commands=4)`** packs it.
- **`DriveAnchorConfig`** — normalization (`x_offset`/`x_span`/`y_offset`/`y_span`),
  DDIM schedule (`truncated_timestep`, `num_train_timesteps`, `beta_start`,
  `beta_end`), `noise_scale`, `seed`. **`drive_anchor_noise(anchors, config)`**
  applies it; `noise_scale=0` zeroes the random term (reproducible comparisons)
  while keeping the schedule's scaling.
- **`DriveResult`** — `trajectory` (list of `DrivePose`), `agents` (list of
  `DriveAgent`: `x`, `y`, `heading`, `length`, `width`, `score`), `bev` (a
  `SegMask`, so `seg_colorize` works on it).
  **The BEV map's axes are not screen axes**: row index increases with FORWARD
  distance (row 0 is the ego end) and column index increases to the LEFT, so
  drawing it north-up needs `bev.labels[::-1, ::-1]`. Upstream documents
  neither, and a straight road is nearly symmetric under a 180° rotation — the
  wrong orientation looks entirely plausible.
- **`DriveConfig`** — `agent_score_thresh`, `bev` (a `SegConfig`).
- **`DriveIoConfig`** — which Engine tensor is which; defaults match the stock
  export. Set an output index to `-1` to skip that head.
- **`decode_drive(trajectory=None, agent_states=None, agent_labels=None, bev=None, config=DriveConfig()) -> DriveResult`**
  — the numpy path; any head may be `None`.
- **`DrivePlanner(engine, anchors, config=None, anchor_config=None, io=None)`** —
  `.plan(camera, lidar, status, timeout_ms=0)`, `.postprocess()`,
  `.diffusion_input`.

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
- **`decode_depth(output, config) -> DepthMap`**.
- **`depth_colorize(dm) -> np.ndarray`** · **`depth_to_gray8(dm) -> np.ndarray`**.
- **`DepthEstimator(engine, config=None, output_index=0)`** — `.estimate(model_input, timeout_ms=0)`,
  `.postprocess()`, `.config`.

## Depth refinement (RGB-D)

Takes a depth map you already have — a stereo or ToF reading, noisy and full of
holes — plus the aligned RGB frame, and returns cleaned metric depth with a
per-pixel trust mask. This is the LingBot-Depth head; it does not estimate depth
from scratch the way `DepthEstimator` (monocular) or `StereoPipeline` do.

```python
refiner = bcdl.DepthRefiner(engine)          # inputs: image, depth_log
r = refiner.run(bgr, depth_m)                # BGR (H,W,3) uint8 + (H,W) float32 metres
r.depth                                      # (H, W) float32 metres, 0 where rejected
r.mask                                       # (H, W) uint8, 1 = trusted
r.vmin, r.vmax                               # range over trusted pixels

k = bcdl.Intrinsics(fx, fy, cx, cy)          # in the SENSOR image's pixels
k = bcdl.scale_intrinsics(k, bgr.shape[1], bgr.shape[0], r.width, r.height)
pts = bcdl.depth_to_pointcloud(r, k)         # (H, W, 3) float32 metres, camera space

# numpy path — feed the graph yourself:
img = bcdl.preprocess_refine_image(bgr)      # (1, 3, eh, ew) float32
dep = bcdl.preprocess_refine_depth(depth_m)  # (1, 1, eh, ew) float32
r = bcdl.decode_refined_depth(depth_out, mask_logit_out)
```

- **`DepthRefineConfig`** — `encoder_height`, `encoder_width` (taken from the
  model at construction), `min_valid_depth`, `mask_threshold`, `apply_mask`.
- **`RefinedDepth`** — `width`, `height`, `vmin`, `vmax`, `depth`, `mask`.
- **`Intrinsics(fx, fy, cx, cy)`** · **`scale_intrinsics(k, from_w, from_h, to_w, to_h)`**.
- **`preprocess_refine_image(bgr, config=None)`** · **`preprocess_refine_depth(depth_m, config=None)`**
  — the exact host preprocessing the model was calibrated with. The RGB
  downscale is area-averaged; a plain bilinear resize is measurably worse.
- **`decode_refined_depth(depth, mask_logit=None, config=None) -> RefinedDepth`**.
- **`depth_to_pointcloud(refined, intrinsics) -> np.ndarray`**.
- **`DepthRefiner(engine, config=None, image_input=0, depth_input=1, depth_output=0, mask_output=1)`**
  — `.run(bgr, depth_m)`, `.postprocess()`, `.config`.

> Depth goes into the graph as **log depth with 0 meaning "no reading"**, which
> is upstream's own convention — a genuine 1 m reading and a missing pixel land
> on the same value, and the model leans on the RGB tokens to separate them.
> The deployed graph also keeps every depth token, where upstream drops tokens
> whose patch has no valid reading at all (a data-dependent sequence length
> cannot be compiled). On the reference scenes that costs 0.06% absolute
> relative error; those scenes are 87–100% valid, so very sparse input depth is
> outside what was measured.

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

Full PP-OCR three-stage pipeline (PP-OCRv6 by default, v5 kept as a fallback),
each stage usable on its own. Pure decoders (`decode_dbnet`, `decode_cls_dir`,
`decode_ctc`, `load_char_dict`) need no model.

```python
chars = bcdl.load_char_dict("ppocr_dict.txt")

det = bcdl.DbTextDetector(engine_det)            # DBNet detect
boxes = det.postprocess(lb)                      # list[TextBox], 4-point rotated
cls = bcdl.TextAngleClassifier(engine_cls)       # 0/180 direction
rec = bcdl.TextRecognizer(engine_rec, "ppocr_dict.txt")  # CRNN/CTC
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

## Voice pipeline

Five pieces make a voice-controlled robot, and all of their networks run on
the BPU. BCDL takes and returns 16 kHz PCM; capturing from a microphone and
playing to a speaker stay in your code.

```
 mic ──> KeywordSpotter ──wake──> VadSegmenter ──utterance──> ParaformerAsr ──text──> CommandMatcher
         (always on)               (until silence)             (one pass)              (pinyin match)
                                                                                            │
 speaker <──────────────────────────── TextToSpeech <────────── reply text <────────────────┘
```

| piece | job | cost on the board |
|---|---|---|
| `KeywordSpotter` | listens for the wake word, all the time | 2.0 ms of BPU per 160 ms of audio |
| `VadSegmenter` (`SileroVad`) | after the wake word, cuts out one whole utterance | 0.39 ms per 32 ms |
| `ParaformerAsr` | utterance → Chinese text | ~35 ms per utterance |
| `CommandMatcher` | text → one of your commands, or "not a command" | a few µs |
| `TextToSpeech` | reply text → speech | ~15–25 ms per sentence |

**Try it first.** [`examples/voice_assistant.py`](../examples/voice_assistant.py)
(and its C++ twin [`voice_assistant.cc`](../examples/voice_assistant.cc)) runs
the whole loop. With no argument it synthesises a short session in the robot's
own voice — speech without the wake word, two commands, one request that is not
a command — so it runs on a board with no microphone and no recordings:

```bash
scripts/fetch_models.sh                       # once
python examples/voice_assistant.py            # or: ./build/voice_assistant
python examples/voice_assistant.py my.wav     # your own 16 kHz mono recording
```

```
    time  event
    4.36 s  wake word '你好小车'
    4.60 s  robot: 我在。  (14 ms to synthesise 0.7 s -> voice_out/reply_01.wav)
    5.28 s  heard '向左转' (0.8 s, 34 ms to recognise) -> command: 向左转
    6.30 s  robot: 好的，向左转。  (16 ms to synthesise 1.4 s -> voice_out/reply_02.wav)
    ...
   13.71 s  heard '给我唱首歌吧' (1.1 s, 33 ms to recognise) -> not a command
```

**Getting the models.** `scripts/fetch_models.sh` downloads the recogniser
(D-Robotics' prebuilt Paraformer). The other four — wake word, VAD, Matcha and
Vocos — are converted from their upstream checkpoints, and each has a recipe in
[bcdl-model-zoo](https://github.com/ruisv/bcdl-model-zoo) (`models/kws_zipformer`,
`silero_vad`, `matcha_tts`, `vocos`): export, calibration and `hb_compile` on an
x86 host with the OpenExplorer toolchain. Point the fetch script at what you
built and it copies the files into `models/` under the names BCDL expects:

```bash
BCDL_SRC_KWS=/path/to/kws_build BCDL_SRC_VAD=/path/to/vad_build \
BCDL_SRC_MATCHA=/path/to/matcha_build BCDL_SRC_VOCOS=/path/to/vocos_build \
  scripts/fetch_models.sh
```

Anything it cannot find is reported as `MISSING` with the recipe to build it,
and the tests that need it skip rather than fail.

The loop is short; the parts that matter:

```python
awake = False
for block in mic_blocks():                      # any length; float32 or int16, 16 kHz mono
    if not awake:
        if kws.feed(block):                     # the wake word
            awake = True; vad.reset()
        continue
    for seg in vad.feed(block):                 # one whole utterance
        m = matcher.match(asr.transcribe(seg.pcm).text)
        play(tts.synthesize(f"好的，{m.command}。" if m else "没听懂，请再说一遍。").pcm)
        awake = False; kws.reset()              # back to sleep on a fresh stream
        break
```

**Audio in.** 16 kHz, mono, float in [-1, 1] or int16 (int16 is rescaled, not
cast). Resample other rates before BCDL (`ffmpeg -i in.wav -ar 16000 -ac 1
out.wav`, or your audio driver). Any block size works. `KeywordSpotter`,
`SileroVad` and `VadSegmenter` keep state between calls, so give each stream its
own objects and feed them in order. `reset()` starts a new stream, and
`flush()` ends one: without it, a wake word or utterance in the last fraction
of a second of a recording is never reported.

**Your wake word.** A wake word is a line of the model's tokens — pinyin
initials and toned finals — so changing it needs no training. Write the
phrases one per line as `你好小车 @你好小车`, then let sherpa-onnx's tool turn
them into tokens:

```bash
pip install sherpa-onnx
sherpa-onnx-cli text2token --tokens models/kws_zipformer_zh_en_3m_tokens.txt \
    --tokens-type ppinyin keywords_raw.txt keywords.txt
# keywords.txt:  n ǐ h ǎo x iǎo ch ē @你好小车
```

Choose three or four syllables that do not occur in everyday speech or in your
commands; a short or common phrase wakes the robot by mistake. English wake
words need the CMU-style lexicon (`en.phone`) from the upstream model package
and `--tokens-type phone+ppinyin`.

**Tuning.**

| if… | change |
|---|---|
| the wake word is missed | lower `KwsConfig.keywords_threshold` (0.25), or add `#0.15` to that line only |
| it wakes on similar phrases | raise the threshold, or `:0.5` to lower that line's boost |
| a command is cut in two at a pause | raise `VadConfig.min_silence_ms` (300) |
| the robot answers too slowly | lower `min_silence_ms` — it is most of the delay: the end of speech is declared about that long after the speaker stops |
| commands are rejected | add the phrasings people actually use as more entries; `CommandMatcher(..., max_distance=0.5)` |
| other speech is taken as a command | lower `max_distance` |
| replies are too fast or slow | `MatchaConfig.length_scale` (> 1 slower) |
| pauses in replies sound long or clipped | `TextToSpeechConfig.silence_scale` (0.2; 1 keeps the model's) |

**Things to know.**

- **The robot hears itself.** Stop feeding the microphone to the VAD and the
  spotter while a reply plays, or use echo cancellation in your audio stack.
- **Chinese.** Recognition and synthesis are Chinese; synthesis drops Latin
  words (write 无线网, not WiFi) and reads digits out (35% → 百分之三十五).
- **Limits.** One utterance is at most 24 s (`VadConfig.max_speech_s` cuts at
  20 s). One reply sentence is at most ~8 s; longer text is split into
  sentences automatically.
- **Accuracy figures** in the sections below come from synthetic and public
  test sets. Before relying on them, record a hundred real commands on your
  robot, with its motors running, and run them through the example.

## Speech recognition (Paraformer)

One 16 kHz mono utterance in, text out, in a single forward pass — Paraformer is
non-autoregressive, so there is no decoding loop. The build is three compiled
stages with continuous integrate-and-fire (CIF) on the CPU between the last two:
fbank → LFR → CMVN → **encoder** → **predictor** → CIF → **decoder** → argmax.

```python
asr = bcdl.ParaformerAsr.from_files(
    "models/paraformer_encoder_int16.hbm",
    "models/paraformer_predictor_int16_v2.hbm",
    "models/paraformer_decoder_int16_v2.hbm",
    "models/paraformer_tokens_8404.json",
    "models/paraformer_am.mvn")
r = asr.transcribe(pcm)          # float32 in [-1, 1] (or int16), 16 kHz mono
print(r.text, r.truncated)

# the front end and CIF on their own (no model needed):
feats = bcdl.Fbank().compute(pcm)                  # (frames, 80) Kaldi log-mel
lfr = bcdl.lfr_stack(feats, 7, 6)                  # (ceil(frames/6), 560)
emb, n = bcdl.cif_integrate(alphas, hidden, valid_frames)
```

**One utterance at a time, at most `asr.max_seconds` (24 s).** Longer input is
truncated and `r.truncated` is set — split long audio with a VAD first. There is
no resampler: resample to 16 kHz before calling.

The front end reproduces `torchaudio.compliance.kaldi.fbank` (dither 0) — to
~1e-5 of a float64 reference, ~2e-4 of torchaudio's own float32 — and FunASR's
LFR bit for bit, because a model trained on those features transcribes worse,
silently, on features that are merely similar.

- **`FbankConfig`** — `sample_rate`, `num_mel_bins`, `frame_length_ms`,
  `frame_shift_ms`, `preemphasis`, `remove_dc_offset`, `low_freq`, `high_freq`,
  `input_scale` (float PCM is multiplied by this, 32768 by default, because
  Kaldi works on int16-scale samples; icefall / k2 models want 1), `window`
  (`"hamming"`, torchaudio's and FunASR's, or `"povey"`, Kaldi's default).
- **`Fbank(config=FbankConfig())`** — `.compute(pcm) -> (frames, num_mel_bins)`,
  `.num_frames(samples)`, `.frame_length`, `.frame_shift`, `.fft_size`.
- **`lfr_stack(feats, lfr_m=7, lfr_n=6) -> np.ndarray`** — low-frame-rate
  stacking, bit-exact with FunASR.
- **`Cmvn`** — `dim`, `shift`, `scale`, `.apply(feats)`;
  **`load_kaldi_cmvn(path)`** / **`parse_kaldi_cmvn(text)`** read an `am.mvn`.
- **`CifConfig`** — `threshold` (1.0), `tail_threshold` (0.45), `max_tokens`.
- **`cif_integrate(alphas, hidden, valid_frames, config=CifConfig())`** →
  `((max_tokens, D) embeddings, token count)`. The tail weight is placed at frame
  `valid_frames`, the true end of speech. A fixed-shape predictor graph can only
  put it at the end of its padding, where masking the padding deletes it — and
  then any utterance whose last character needed the tail loses it.
- **`load_token_list(path) -> list[str]`** — a JSON array (`tokens.json`) or one
  token per line; **`argmax_tokens(logits)`**; **`tokens_to_text(ids, vocab)`**
  (drops `<specials>`, glues `@@` pieces, spaces Latin words).
- **`AsrResult`** — `text`, `token_ids`, `feature_frames`, `truncated`.
- **`ParaformerConfig`** — `fbank`, `lfr_m`, `lfr_n`, `cif`.
- **`ParaformerAsr(encoder, predictor, decoder, vocab, cmvn, config=None)`** —
  three `Engine`s; also `ParaformerAsr.from_files(encoder, predictor, decoder,
  tokens, cmvn, config=None)`. `.transcribe(pcm, timeout_ms=0)`,
  `.transcribe_features(feats)`, `.features(pcm)`, `.max_seconds`,
  `.max_frames`, `.feature_dim`, `.max_tokens`, `.vocab_size`. The stages are
  matched by tensor shape, and a token list whose size disagrees with the
  decoder is refused.

### Voice activity detection (Silero VAD)

Finds where speech starts and stops in a 16 kHz stream and hands each utterance
to the recogniser whole. Silero VAD v6 runs on the BPU, one call per 32 ms
chunk; `VadSegmenter` turns its probabilities into utterances.

```python
vad = bcdl.SileroVad("models/silero_vad_16k_int16_nashm.hbm")
seg = bcdl.VadSegmenter(vad)                 # VadConfig(): 0.5, 250 ms on, 300 ms off
for block in mic_blocks():                   # float32 or int16, any length
    for s in seg.feed(block):                # utterances that closed in this block
        print(s.start / 16000, asr.transcribe(s.pcm).text)
for s in seg.flush(): ...                    # end of stream

segs = seg.segments(recording)               # a whole recording at once
p = vad.probabilities(recording)             # one probability per 512 samples
```

Measured on 100 noisy sessions (30 s each: AISHELL-1 utterances 0.3–3 s apart,
MUSAN noise at 10 and 0 dB SNR) and 24 minutes of noise alone, with the
default `VadConfig`:

| | frame F1, 10 / 0 dB | utterances missed | end of speech declared after (median / p90, 10 dB) | false speech on noise |
|---|---|---|---|---|
| float Silero (ONNX) | 0.929 / 0.870 | 2.3% / 11.5% | 0.36 / 0.44 s | 0 |
| **int16, BPU** | **0.924 / 0.863** | **3.4% / 13.2%** | **0.36 / 0.43 s** | **0** |

The two agree on speech vs. non-speech for 99.0% of 253k chunks (mean
probability difference 0.012). Each chunk costs 0.39 ms on the board, STFT and
Python included — 1.2% of real time. `threshold = 0.4` brings the BPU build's
F1 level with float's (0.929 / 0.872) but declares the end of speech later
(p90 0.61 s at 0 dB against 0.52 s).

- **`SileroVad(engine_or_path)`** — `.probability(chunk)` (512 samples →
  float), `.probabilities(pcm)` (whole recording from a reset), `.reset()`,
  `.chunk` (512), `.sample_rate` (16000). Stateful: the LSTM state and the
  previous chunk's last 64 samples carry over, so one object reads one stream,
  in order.
- **`VadConfig`** — `threshold` (0.5), `neg_threshold` (−1 = same as
  threshold; lower it for hysteresis), `min_speech_ms` (250), `min_silence_ms`
  (300), `speech_pad_ms` (30), `max_speech_s` (20 — the recogniser's window is
  24 s).
- **`VadSegmenter(vad, config=None)`** — `.feed(pcm)`, `.flush()`,
  `.segments(pcm)`, `.reset()`, `.in_speech`, `.last_probability`. A segment
  opens after `min_speech_ms` at or above `threshold` and closes after
  `min_silence_ms` below `neg_threshold`; the end of speech is therefore
  declared about `min_silence_ms` after the speaker stops.
- **`VadSegment`** — `start`, `end` (samples since the last reset, padding
  included), `pcm`.

The model's STFT runs on the CPU and the BPU takes the magnitude spectrum.
Compiled whole, the graph squares the real and imaginary parts on the BPU,
where the power spans ~5000: int16 then loses every quiet frequency bin and the
probability sagged to 0.2–0.5 in the middle of a sentence.

### Keyword spotting (wake word)

Listens continuously for a wake word, then the VAD and recogniser take over.
The model is sherpa-onnx's open-vocabulary zipformer transducer (zh-en, 3M
parameters): a wake word is a line of tokens, not a trained model. The encoder
runs on the BPU, one 160 ms chunk per call; the decoder, joiner and keyword
search run on the CPU and follow sherpa-onnx's KeywordSpotter step for step.

```python
kws = bcdl.KeywordSpotter("models/kws_zipformer_zh_en_3m_c16_int16_nashm.hbm",
                          "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                          "models/kws_zipformer_zh_en_3m_tokens.txt",
                          "n ǐ h ǎo x iǎo ch ē @你好小车\n"
                          "x iǎo ch ē x iǎo ch ē @小车小车\n")
for block in mic_blocks():                   # float32 or int16, any length
    for d in kws.feed(block):
        print(d.keyword, d.start, d.end)     # seconds since reset()
```

A keyword line is the model's tokens — pinyin initials and toned finals for
Chinese, ARPAbet phones for English — then optional `:boost` and
`#threshold` for that line, then `@phrase`. sherpa-onnx's `text2token` tool
(`pip install sherpa-onnx`) writes these lines from plain text.

Measured with 你好小车 and 小车小车 (202 clips each clean and with MUSAN noise,
9 synthetic voices), against AISHELL-1 speech, spoken commands and 2.3 h of
noisy sessions, and 350 confusable phrases (你好小菜, 校车校车, ...), at the
default threshold:

| | wake words found, clean / noisy | false wake-ups | confusable phrases that fire, clean / noisy |
|---|---|---|---|
| float (sherpa-onnx) | 96.5% / 87.1% | 0 in 4.1 h | 5.4% / 4.6% |
| **int16, BPU** | **97.5% / 89.1%** | **0 in 4.4 h** | **5.7% / 4.0%** |

The encoder takes 2.0 ms of BPU time per 160 ms chunk; with the CPU decoding
the whole spotter runs at a real-time factor of 0.023.

- **`KeywordSpotter(encoder, decoder_joiner, tokens, keywords=None,
  config=None)`** — `keywords` is the text of the lines or a path to a file.
  `.feed(pcm)`, `.flush()` (end of stream: a wake word in the last ~0.3 s
  is only seen after it), `.set_keywords(text)`, `.load_keywords(path)`, `.keywords`,
  `.reset()`, `.encode(pcm)` (encoder output of a whole recording, for tests),
  `.token_id(token)`. Streaming: one object reads one stream, in order.
- **`KwsConfig`** — `keywords_score` (1.0, the boost per matched token),
  `keywords_threshold` (0.25, the mean per-token probability a detection
  needs), `num_trailing_blanks` (1), `max_active_paths` (4): sherpa-onnx's
  defaults. Raise the threshold for fewer false wake-ups, lower it for more
  recall.
- **`KwsDetection`** — `keyword`, `index` (its line), `start`, `end`,
  `tokens`.

### Voice commands (closed set)

A robot taking spoken commands needs to know *which* command was said, not a
perfect transcript. `CommandMatcher` compares recogniser text with a fixed list
in toneless pinyin, which absorbs homophones (打开灯 → 大开灯, 出发 → 处罚) and
a stray trailing particle (前进 → 前进了), and rejects speech that is not a
command. Pure CPU, a few µs per call.

```python
m = bcdl.CommandMatcher(["前进", "后退", "左转", "右转", "打开灯", "关灯"])
r = m.match(asr.transcribe(pcm).text)    # e.g. "大开灯"
if r:                                    # r.index >= 0
    print(r.index, r.command, r.distance)   # 4 打开灯 0.0
```

On 40 commands × 9 synthetic voices, Paraformer's exact-sentence accuracy was
84.7% clean / 82.8% at 5 dB SNR; matched this way it is 96.9% / 95.6%, with
0.6% / 1.1% mapped to the wrong command and none of 600 news sentences accepted.

- **`to_pinyin(text) -> list[str]`** — one toneless syllable per CJK character
  (most common reading, ü as `v`); an ASCII letter run is one lower-cased token;
  a digit run is read as a Chinese number (`"90"` → `jiu shi`); full-width
  letters/digits fold to ASCII; everything else is dropped.
- **`CommandMatcher(commands, max_distance=0.5)`** — distance is the
  syllable-level edit distance divided by the command's syllable count; ties go
  to the closer length, then the earlier command. Latin words are only ever
  compared with Latin words ("you" is not 右). `.match(text) -> CommandMatch`,
  `.distances(text) -> list[float]`, `.commands`, `.max_distance`, `len()`.
- **`CommandMatch`** — `index` (-1 = rejected), `nearest` (closest command even
  when rejected; -1 for text with no pinyin), `distance`, `command`; truthy when
  accepted. Several phrasings of one intent are several entries.


### Speech synthesis back end (Vocos)

Mel spectrogram in, waveform out: Vocos on the BPU, the inverse STFT on the CPU.
It is the second half of a non-autoregressive TTS — pair it with an acoustic
model that emits 80-bin log-mels at 16 kHz / hop 256, such as Matcha-TTS zh-en.

```python
voc = bcdl.VocosVocoder("models/vocos_16k_t256_int16_nashm.hbm")
pcm = voc.synthesize(mel)        # (80, frames) float32 log-mel -> float32 PCM, 16 kHz

# the inverse STFT on its own (no model needed):
ist = bcdl.Istft()               # n_fft 1024, hop 256, centred, periodic Hann
wav = ist.compute(real, imag)    # (frames, 513) each
wav = ist.compute_polar(mag, cos, sin)   # (513, frames) each — Vocos's layout
```

The build takes 256 frames (4.1 s); longer mels are cut into chunks that
overlap by 32 frames and the seams are exact. Per 3 s of audio: 1.3 ms BPU and
6.6 ms CPU. Against the float model it is 0.54 dB in log-spectral distance over
the audible cells, and Paraformer transcribes its output exactly.

- **`IstftConfig`** — `n_fft` (power of two), `hop_length`, `center`.
- **`Istft(config=IstftConfig())`** — `.compute(real, imag)`,
  `.compute_polar(mag, cos, sin)`, `.num_samples(frames)` (`(frames - 1) * hop`
  centred), `.bins`. Matches `torch.istft` with a periodic Hann window; DC and
  Nyquist imaginary parts are ignored.
- **`VocosConfig`** — `stft`, `context` (overlap per side, ≥ 27 for exact
  seams), `pad_value` (log-mel for frames past either end, −16),
  `max_magnitude` (100), `sample_rate`.
- **`VocosVocoder(engine_or_path, config=None)`** — `.synthesize(mel,
  timeout_ms=0)` takes `(num_mels, frames)` or `(1, num_mels, frames)`;
  `.num_mels`, `.chunk_frames`, `.body_frames`, `.sample_rate`. The model must
  output `log_mag` and `phase`, in that order: `exp`/`cos`/`sin` are done on the
  CPU because the compiler's quantised `cos`/`sin` ruin the phase.


### Speech synthesis (Matcha-TTS + Vocos)

Chinese text in, 16 kHz speech out, every network on the BPU. Non-autoregressive:
each sentence is one encoder pass, one decoder pass (three flow-matching steps
compiled into it) and one or two vocoder chunks.

```python
tts = bcdl.TextToSpeech.from_files(
    "models/matcha_zh_en_enc_l128_int16_nashm.hbm",
    "models/matcha_zh_en_dec_t512_int16_nashm.hbm",
    "models/matcha_zh_en_embedding.npy",
    "models/vocos_16k_t256_int16_nashm.hbm",
    "models/matcha_zh_en_tokens.txt",
    "models/matcha_zh_en_lexicon.txt")
r = tts.synthesize("好的，马上出发。")      # r.pcm float32, r.sample_rate 16000

# or stage by stage:
fe = bcdl.MatchaFrontend("models/matcha_zh_en_tokens.txt", "models/matcha_zh_en_lexicon.txt")
for ids in fe.encode(text):                  # one id list per sentence
    m = acoustic.synthesize(ids)             # MatchaAcoustic -> m.mel (80, frames)
    pcm = vocoder.synthesize(m.mel)
```

On 60 robot-reply sentences the BPU chain is heard back by FireRedASR2 at 1.73%
CER, against 1.61% for the float model on the CPU (sherpa-onnx), at a
real-time factor of 0.009 — 26 ms per sentence on average.

- **`MatchaFrontend(tokens, lexicon, config=MatchaFrontendConfig())`** —
  `.encode(text) -> list[list[int]]`, `.sentences(text)`,
  `MatchaFrontend.read_numbers(text)`, `.vocab_size`, `.lexicon_size`.
  Reproduces sherpa-onnx's lexicon front end for this model id for id
  (punctuation folding, sentence split/merge, longest-match lexicon lookup, the
  closing punctuation token). Digits are read out first (`90` → 九十, `12.6` →
  十二点六, `35%` → 百分之三十五, `2026年` → 二零二六年). Latin words are dropped —
  the lexicon has no English.
- **`MatchaFrontendConfig`** — `min_words` (5), `max_words` (20).
- **`MatchaConfig`** — `noise_scale` (1.0; 0 makes the output deterministic),
  `length_scale` (> 1 is slower), `seed`.
- **`MatchaAcoustic(encoder, decoder, embedding, config=None)`** —
  `.synthesize(ids) -> MatchaResult` (`mel` `(80, frames)`, `frames`,
  `durations`, `truncated`), `.seed(n)`, `.max_tokens` (128), `.max_frames`
  (512 = 8.2 s), `.num_mels`, `.vocab_size`. Longer input is truncated and
  flagged. The token embedding is looked up on the CPU from the `.npy`.
- **`TextToSpeech(frontend, acoustic, vocoder, tts_config=None)`** /
  **`.from_files(..., config=None, tts_config=None)`** — `.synthesize(text) ->
  TtsResult` (`pcm`, `sample_rate`, `sentences`, `truncated`). Sentences are
  synthesised separately and concatenated, then every pause of 0.2 s or more
  keeps `TextToSpeechConfig.silence_scale` of its length (0.2, sherpa-onnx's
  default; 1 keeps the model's pauses, which run long at punctuation). On the
  60 test sentences that takes the audio from 225 s to 172 s (sherpa-onnx:
  176 s) with the round-trip CER unchanged.
- **`scale_silence(pcm, sample_rate, scale, min_pause_s=0.2, level=0.01)`** —
  that pause shortening on its own (`|x| <= level` counts as silence); matches
  sherpa-onnx sample for sample.

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
  `track_buffer`, `frame_rate`; appearance: `proximity_thresh`,
  `appearance_thresh`, `ema_alpha`; `boost` (`BoostConfig`).
- **`Track`** — `track_id`, `x1, y1, x2, y2`, `score`, `class_id`; `__repr__`.
- **`ByteTracker(config=None)`** — `.update(detections) -> list[Track]`,
  `.update(detections, embeddings) -> list[Track]`,
  `.apply_camera_motion(affine)`, `.reset()`, `.config`.

### Appearance (ReID)

Passing one embedding per detection switches on appearance association: the cost
becomes `min(IoU distance, gated cosine distance)`, so appearance can only
**rescue** a match geometry was about to miss — it can never break one geometry
already had. An **empty** entry means "no appearance for this detection", which
is how you skip the ReID model on cheap crops.

```python
reid = bcdl.ReIDExtractor(bcdl.Engine("osnet.hbm"))
tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
for frame in stream:
    dets = detector.detect(...)
    embs = reid.embed_detections(frame, dets, min_score=0.5)   # parallel to dets
    for t in tracker.update(dets, embs):
        print(t.track_id)
```

- **`ReIDExtractor(engine, config=None, output_index=0)`** — `.dim`,
  `.embed(model_input)`, `.embed_crop(crop_bgr)`,
  `.embed_detections(frame_bgr, detections, min_score=0.0)`.
- **`reid_preprocess(crop_bgr, width=128, height=256, config=None)`** and
  **`reid_crop_preprocess(bgr, x1, y1, x2, y2, in_w, in_h, config)`** — a
  **squashing** resize (not a letterbox: these models are trained on squashed
  crops), BGR→RGB, ImageNet normalization, NCHW float32.
- **`ReidConfig`** — `mean`, `std`.

The model runs **once per crop**, so on a crowded frame it, not the detector,
sets the frame time. Note that **int8 PTQ is not adequate for OSNet** — see the
0.4.0 entry in `CHANGELOG.md`.

### Camera motion

`apply_camera_motion(affine)` warps every tracklet by a 2x3 transform mapping
the previous frame onto the current one, before the next `update()`. Position
and size are warped; velocity is not, because velocity describes the target in
the world and a one-frame camera jolt is not the target accelerating. Skip the
call entirely for a static camera rather than passing an identity.

```python
M, _ = cv2.estimateAffinePartial2D(prev_pts, cur_pts, method=cv2.RANSAC)
tracker.apply_camera_motion(np.ascontiguousarray(M, np.float32))
```

### BoostConfig

BoostTrack++ additions, **all off by default** — each is independently
switchable so its contribution can be measured rather than assumed.

- `rich_similarity` — add Mahalanobis and shape-agreement terms to the cost;
  `lambda_iou`, `lambda_mhd`, `lambda_shape`, `min_iou`.
- `soft_biou` — grow both boxes by `1 - tracklet confidence`, so a tracklet that
  has been coasting searches a wider area.
- `boost_detections` — raise the scores of detections the tracklets vouch for,
  before the high/low split; `dlo_alpha`, `vt_start`, `vt_end`, `vt_steps`,
  `duo`, `duo_iou`. **Situational**: it recovers real detections when the
  detector is limited by misses, and manufactures false tracks when it is not.

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
- **`TrackingPipeline(engine, det_config=None, track_config=None,
  reid_engine=None, reid_config=None)`** — `.process(bgr) -> list[Track]`,
  `.last_detections`, `.has_reid`, `.last_embed_count`, `.reset()`. Passing
  `reid_engine` adds appearance to the native path; the crop size is read from
  that model.
- **`TrackingReidConfig`** — `min_score`, `max_crops`, `crop` (`ReidConfig`).
  Both are cost knobs: the ReID model runs once per qualifying crop, and
  `last_embed_count` reports how many that was.

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

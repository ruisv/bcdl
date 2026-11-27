# BCDL C++ API reference

**English** | [简体中文](CPP_API.zh.md)

Complete reference for the C++ library (namespace `bcdl`). The public headers
under [`include/bcdl/`](../include/bcdl/) are the source of truth — this document
summarizes them with usage. For the Python bindings see
[`API.md`](API.md); the names map 1:1 (Python snake_case ⇄ C++ camelCase).

- [Linking](#linking) · [Conventions](#conventions) · [Worked example](#worked-example)
- Core: [Error handling](#error-handling) · [SysMem](#sysmem) · [Task](#task) ·
  [MemPool](#mempool) · [Runtime info](#runtime-info)
- Backend: [Engine](#engine) · [Output reader & input writer](#output-reader--input-writer)
- Preprocessing: [Geometry & letterbox](#geometry--letterbox) ·
  [GDC hardware warp](#gdc-hardware-warp) · [VpImage](#vpimage)
- Tasks: [Detection](#detection) · [Panoptic driving (YOLOP)](#panoptic-driving-perception-yolop) ·
  [Classification](#classification) · [Image embeddings](#image-embeddings) ·
  [Face detection & alignment](#face-detection--alignment) · [Pose](#pose) ·
  [Whole-body pose](#whole-body-pose-133-keypoints) ·
  [Super-resolution](#super-resolution-tiled) · [XFeat](#sparse-local-features-xfeat) ·
  [Instance segmentation](#instance-segmentation) ·
  [Open-vocabulary detection (YOLOE)](#open-vocabulary-detection-yoloe) ·
  [Promptable segmentation (EdgeSAM)](#promptable-segmentation-edgesam) ·
  [Oriented boxes (OBB)](#oriented-boxes-obb) ·
  [Semantic segmentation](#semantic-segmentation) ·
  [Monocular 3-D detection (SMOKE)](#monocular-3-d-detection-smoke) ·
  [Lidar 3-D detection](#lidar-3-d-detection) ·
  [Anomaly detection](#anomaly-detection-one-class) · [Optical flow](#optical-flow) ·
  [End-to-end driving](#end-to-end-driving-camera--lidar--trajectory) ·
  [Depth](#depth) · [Depth refinement](#depth-refinement-rgb-d) · [OCR](#ocr)
- Speech: [Voice pipeline](#voice-pipeline) ·
  [Speech recognition](#speech-recognition-paraformer)
- Media: [JPEG (JPU)](#jpeg-jpu) · [Video (VPU)](#video-vpu)
- Tracking & pipelines: [ByteTracker](#bytetracker) · [DetectionPipeline](#detectionpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline) ·
  [AsyncVideoDetectionPipeline](#asyncvideodetectionpipeline) ·
  [TrackingPipeline](#trackingpipeline) · [StereoPipeline](#stereopipeline)

## Linking

The umbrella header pulls in everything:

```cpp
#include "bcdl/bcdl.h"     // or include just the sub-headers you use
```

With CMake, `find_package(bcdl)` exposes the `bcdl::bcdl` target (it transitively
brings the headers and the hobot SDK link libs — the board's own, under
`/usr/include/hobot` and `/usr/hobot/lib`):

```cmake
find_package(bcdl CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE bcdl::bcdl)
```

See the [README](../README.en.md#build-from-source) for the conda package
(`libbcdl`) and the on-board build. Everything here
runs **only on an RDK S100 / S100P / S600 board** — the hobot SDK and the BPU/JPU/VPU
units are board-only.

## Conventions

- **Namespace** `bcdl`. Headers are `.h`; the umbrella is `bcdl/bcdl.h`.
- **Errors** — every hobot SDK call is wrapped in `BCDL_CHECK(...)`, which throws
  `bcdl::Error` (carrying the SDK return `code()`) on non-zero. Library functions
  also throw `bcdl::Error` on misuse. Wrap calls in `try/catch`.
- **Images** are interleaved `HxWx3` uint8 BGR (OpenCV order), row stride
  `width*3` unless a `VpImage` says otherwise.
- **Result coordinates are original-image pixels** — tasks that undo the
  letterbox take a `LetterboxInfo`.
- **Cache discipline** (the most common correctness bug) is owned by `Engine`
  and the preproc/codec helpers; you only touch `cleanCache()` /
  `invalidateCache()` when you hand-roll a buffer. See
  [SysMem](#sysmem).
- **Ownership** — task/pipeline classes hold an `Engine&` by reference and do
  **not** own it; keep the `Engine` alive for their lifetime. RAII wrappers
  (`SysMem`, `Task`, `VpImage`, `Engine`, codecs) are move-only.

---

## Error handling

`bcdl/core/status.h`

```cpp
class Error : public std::runtime_error {
  int code() const noexcept;   // hobot SDK return code (0 == success)
};

BCDL_CHECK(expr);   // throws bcdl::Error(code, "expr @ file:line") if expr != 0
```

## SysMem

`bcdl/core/sys_mem.h` — RAII over `hbUCPSysMem`, the one shared-memory buffer used
by BPU tensors, JPU/VPU images, and VP buffers. Move-only.

```cpp
bcdl::SysMem mem(nbytes, /*cached=*/true, /*device_id=*/0);
void*    p   = mem.data();      // virtual address
uint64_t pa  = mem.phyAddr();   // physical address
mem.cleanCache();               // after CPU writes, before device reads
mem.invalidateCache();          // after device writes, before CPU reads
hbUCPSysMem& raw = mem.raw();   // e.g. assign into hbDNNTensor.sysMem
```

Cached buffers (the default) need explicit `cleanCache()` / `invalidateCache()`
around CPU⇄device hand-offs; pass `cached=false` for a coherent (slower) buffer.

## Task

`bcdl/core/task.h` — RAII over `hbUCPTaskHandle_t`, the one task handle returned by
`hbDNNInferV2` and every hbVP op. Move-only.

```cpp
bcdl::Task t;
hbDNNInferV2(t.addr(), outputs, inputs, dnn);  // producer fills the handle
t.submit(/*priority=*/HB_UCP_PRIORITY_LOWEST);
t.wait(/*timeout_ms=*/0);                       // 0 blocks forever
// t.release() runs in the destructor
```

## MemPool

`bcdl/core/mem_pool.h` — a pool of reusable cached `SysMem` blocks for
zero-per-frame-allocation pipelines. Hand back blocks via an RAII `Lease`.

```cpp
bcdl::MemPool pool;                 // cached=true, device_id=0
pool.reserve(/*size=*/4 << 20, /*count=*/4);   // warm the pool at startup
{
  auto lease = pool.acquire(nbytes);   // smallest free block >= nbytes, else allocs
  void* p = lease.data();
  lease.cleanCache();                  // forwards to the block
}                                      // Lease dtor returns the block to the pool
std::size_t free = pool.freeCount(), total = pool.blockCount();
uint64_t pooled = pool.bytesPooled();
pool.clear();                          // free all non-leased blocks
```

The pool must outlive every `Lease`. Thread-safe (acquire/release are mutexed);
recycled blocks are not zeroed.

---

## Runtime info

`bcdl/core/runtime.h` — which hobot runtime the process actually loaded. BCDL
links against the SDK that ships with the board's OS image rather than bundling
one; when results differ between boards, this is the first thing to compare.

```cpp
bcdl::RuntimeInfo r = bcdl::runtimeInfo();   // safe before any Engine exists
// r.dnn_version  "3.13.6_(4.7.5 HBRT)"   hbDNNGetVersion()
// r.ucp_version  "3.13.6"                hbUCPGetVersion()
// r.soc_name     "D-Robotics RDK S100P V1P0"
// r.dnn_library  "/usr/hobot/lib/libdnn.so" — anywhere else means another copy
//                of the SDK is shadowing the board's
```

| Member | Meaning |
|---|---|
| `runtimeInfo()` | `RuntimeInfo` with the four fields above (`dnn_library` is `""` if the linker cannot say) |

---

## Engine

`bcdl/backend/engine.h` — loads a compiled `.hbm`, runs BPU inference via
`hbDNN` + `hbUCP`. Tensor buffers are allocated once at construction and reused;
cache coherency is internal (`setInput` cleans, `infer` invalidates outputs).

```cpp
bcdl::Engine engine("model.hbm", /*model_name=*/"");   // "" => first model
int ni = engine.numInputs(), no = engine.numOutputs();
std::vector<int> ish = engine.inputShape(0);
int itype = engine.inputType(0);          // HB_DNN_TENSOR_TYPE_*
std::size_t ibytes = engine.inputBytes(0);

engine.setInput(0, host_ptr, nbytes);     // copy host -> device buffer (+ clean)
engine.infer(/*timeout_ms=*/0);           // submit + wait + invalidate outputs
const void* out = engine.outputData(0);   // honor outputProperties(0).stride
std::size_t obytes = engine.outputBytes(0);
```

| member | description |
|--------|-------------|
| `Engine(hbm_path, model_name="")` | Load a model (pick by name if the package holds several). |
| `modelName()` | Active model name. |
| `static modelNames(hbm_path)` | Every model name packed into the `.hbm`, without building an Engine — what to pass as `model_name`. A `.hbm` is a package; the SigLIP encoders, for one, hold two models. |
| `numInputs()` / `numOutputs()` | Tensor counts. |
| `inputName(i)` / `outputName(i)` | Tensor name as compiled (the ONNX name). |
| `inputShape(i)` / `outputShape(i)` | `std::vector<int>` shape. |
| `inputType(i)` / `outputType(i)` | `hbDNN` tensor type enum. |
| `inputProperties(i)` / `outputProperties(i)` | Full `hbDNNTensorProperties` (dtype, quant, stride). |
| `inputBytes(i)` / `outputBytes(i)` | Allocated device-buffer sizes. |
| `inputPackedBytes(i)` / `outputPackedBytes(i)` | Size of the tensor as a contiguous row-major array — smaller than the device buffer when the model pads a dimension. |
| `copyOutput(i, dst, bytes)` | Gather output `i` out of its padded device layout into a packed row-major array of `outputPackedBytes(i)` bytes; dtype unchanged (`outputAsFloat()` also dequantizes). |
| `inputStride(i)` | Resolved byte strides of input `i`, innermost last. |
| `outputStride(i)` | Resolved byte strides of output `i`. Outputs pad too; `outputAsFloat()` handles it, direct readers must not reshape flat. |
| `setInput(i, data, bytes)` | Copy host bytes into input `i` and flush. `bytes` must be `inputPackedBytes(i)` (scattered into the device layout) or `inputBytes(i)` (taken as-is); anything else throws. |
| `inputData(i)` | Read-only view of input `i`'s device buffer, in the device layout. |
| `infer(timeout_ms=0)` | Run one inference (blocks; invalidates output caches). |
| `outputData(i)` | Pointer to output `i` after `infer()` (honor the stride). |
| `static elemSize(tensor_type)` | Byte size of an `hbDNN` tensor element type. |

> Prefer a **task class** (`Detector`, `Segmenter`, …) over reading
> `outputData()` yourself — it dequantizes (F16/int8/int16) and de-strides for
> you. Drop to the raw output only for a custom decoder, and even then read it
> through `outputAsFloat()` ([below](#output-reader--input-writer)).

## Output reader & input writer

`bcdl/backend/output_reader.h`, `bcdl/backend/input_writer.h` — the float view of
an Engine's tensors that every task class is built on. Use them when you decode
(or feed) a model yourself, rather than reading `outputData()` or writing raw
bytes.

```cpp
std::vector<float> scratch;            // reuse across frames
std::vector<int> shape;
const float* out = bcdl::outputAsFloat(engine, /*out_idx=*/0, scratch, shape);
// logical row-major floats; valid until `scratch` is destroyed or the next infer()

std::vector<std::uint8_t> stage;       // one per input, reused across frames
std::size_t n = bcdl::inputElementCount(engine, /*in_idx=*/0);
bcdl::setInputFromFloat(engine, /*in_idx=*/0, values, n, stage);   // quantizes if needed
```

| member | description |
|---|---|
| `outputAsFloat(engine, out_idx, scratch, shape)` | Logical row-major float view of output `out_idx`; fills `shape`. A packed F32 tensor comes back zero-copy (a pointer into the already-invalidated device buffer); F16, quantized-integer and stride-padded tensors are gathered and dequantized into `scratch`. |
| `setInputFromFloat(engine, in_idx, data, count, scratch)` | Write `count` floats into input `in_idx` in the dtype the model declared: F32 as-is, an integer type (S8 … U32) as `round(value / scale) + zero_point` clamped to its range; `setInput()` then scatters the packed buffer into the device layout. |
| `inputElementCount(engine, in_idx)` | Element count of the input's valid shape — what `count` must be. |
| `sigmoid(x)` | The logistic sigmoid every YOLO-family decoder shares (header-only). |

The zero-copy path is not a micro-optimization: detection, pose, segmentation
and OBB heads emit 10^5–10^6 elements per frame, and skipping the gather is the
difference between ~10 ms and ~1 ms of post-processing.

`setInputFromFloat` exists for models compiled with **quantized inputs**. Their
input buffer holds integers with a per-tensor (or per-axis) scale, and handed
raw float bytes such a model runs on noise without an error anywhere. It throws
`Error(-1)` when `count` does not match the valid shape, and for an F16 input —
no model in this library has one, and an untested float→half path would be
worse than a clear refusal.

---

## Geometry & letterbox

`bcdl/preproc/geometry.h` — aspect-preserving fit + coordinate mapping (header-only).

```cpp
bcdl::LetterboxInfo lb = bcdl::computeLetterbox(srcW, srcH, dstW, dstH,
                                                /*centerPad=*/true);
float mx = lb.fwdX(x), my = lb.fwdY(y);   // original -> model pixel
float ox = lb.invX(x), oy = lb.invY(y);   // model -> original pixel (un-letterbox)
```

`LetterboxInfo` fields: `scale`, `padX`, `padY`, `srcW/H`, `dstW/H`; methods
`fwdX/fwdY`, `invX/invY`, `clampX/clampY`. Every detection-family
`postprocess(lb)` takes one so results land in original pixels.

**CPU preprocessing** (`bcdl/preproc/letterbox_cpu.h`) — the deployed board's
hbVP geometric ops are vDSP-backed and offline/root-only, so these CPU paths are
the ones the pipelines use. Geometry is identical to the hbVP path.

```cpp
bcdl::VpImage src(w, h, HB_VP_IMAGE_FORMAT_BGR);
bcdl::VpImage nv12(inW, inH, HB_VP_IMAGE_FORMAT_NV12);
bcdl::LetterboxInfo lb = bcdl::letterboxToNv12Cpu(nv12, src, /*padValue=*/114);
// also: letterboxCpu(dstBgrOrY, src, pad) ; bgrToNv12Cpu(dstNv12, srcBgr)
//       letterboxNv12Cpu(dstNv12, srcNv12, pad, range)   // NV12 -> NV12, no BGR
```

`bgrToNv12Cpu` uses BT.601 **full-range** (matches cv2's `COLOR_BGR2YUV_I420`);
verify against your model's calibrated range. `letterboxNv12Cpu` resamples each
plane directly: luma is exact, chroma differs slightly from the BGR path (the UV
plane is resampled at half resolution), and `range` (`YuvRange`, default
`kStudioToFull`) says whether to expand a video decoder's studio-swing levels.

**hbVP path** (`bcdl/preproc/letterbox.h`, DSP required): `letterbox(dst, src,
padValue, interp)`, `cvtColor(dst, src)`, `resizeExact(dst, src, interp)`.

> These three are **unverified — prefer the CPU or GDC path above.** They run on
> the vDSP, and the vDSP is not online on the S100P 4.0.5 image: `/dev/vdsp0`
> exists but the core reports OFFLINE even as root, and no DSP firmware ships in
> `/lib/firmware`, so the calls throw `bcdl::Error` instead of executing. In
> particular `letterbox()`'s affine convention (forward src→dst matrix,
> `isInverse=0`) follows the SDK header but has never been observed producing a
> frame. On a board where the DSP does come up, check the first output for
> mirroring or a wrong scale before trusting it.

**GDC hardware path** — `GdcLetterbox` and `GdcRemap` run fixed NV12
transforms on the VPS GDC engine with the CPU idle; see
[GDC hardware warp](#gdc-hardware-warp).

## GDC hardware warp

`bcdl/preproc/gdc_letterbox.h`, `bcdl/preproc/gdc_remap.h` (only when built with
`BCDL_HAVE_GDC`) — fixed transforms on the VPS GDC engine, NV12 in / NV12 out,
CPU idle during the op. Both classes generate their warp LUT at construction (no
offline `.bin`) and keep a persistent vnode and device input buffer, so `run()`
costs two NV12 copies around the hardware op; `BCDL_GDC_TIMING=1` prints the
per-run breakdown. Semantics, the measured comparison with the CPU path, and the
reverse-engineered CUSTOM-grid notes: [GDC.md](GDC.md).

```cpp
// letterbox: same geometry as computeLetterbox()
bcdl::GdcLetterbox lbg(inW, inH, outW, outH, /*pad=*/114, bcdl::YuvRange::kAsIs);
bcdl::LetterboxInfo lb = lbg.run(srcNv12, dstNv12);

// arbitrary fixed dense remap (cv2.remap semantics), LUT generated at runtime
bcdl::GdcRemap remap(mapX, mapY, inW, inH, outW, outH, /*gridStep=*/16);
remap.run(srcNv12, dstNv12);   // 2448x2048: ~6.3 ms wall, ~1 ms CPU
```

| member | description |
|---|---|
| `GdcLetterbox(in_w, in_h, out_w, out_h, pad=114, range=kAsIs)` | Open the GDC for one fixed letterbox. `range` must match what the model was calibrated on (see `YuvRange`); `pad` is a full-range luma value written after the conversion, so 114 stays 114. Throws on any hobot/GDC failure. |
| `GdcLetterbox::run(src, dst)` | NV12 `in_w×in_h` → NV12 `out_w×out_h`, `dst` (re)allocated if needed; returns the `LetterboxInfo` for un-letterboxing. |
| `info()` | The instance's fixed `LetterboxInfo`. |
| `GdcRemap(map_x, map_y, in_w, in_h, out_w, out_h, grid_step=16)` | Dense row-major `(out_h, out_w)` float32 maps, cv2 convention: `out(x, y) = in(map_x[y,x], map_y[y,x])`. Sampled every `grid_step` px into a CUSTOM grid, so `out_w` and `out_h` must both be multiples of it (16 divides 2448 and 2048). Coordinates outside the input are clamped. |
| `GdcRemap::run(src, dst)` | NV12 `in_w×in_h` → NV12 `out_w×out_h` on the GDC, `dst` (re)allocated if needed. |
| `inputWidth()` · `inputHeight()` · `outputWidth()` · `outputHeight()` | On both classes. |

`GdcRemap` is built for stereo rectification: the maps from
`cv2.initUndistortRectifyMap` are smooth, so sampling them on a grid and letting
the GDC interpolate between the nodes is enough. On real rectify maps it matches
`cv2.remap` to mean 0.25 / p99 2 grey levels, at 6.3 ms against 14.7 ms for
`cv2.remap` on all six cores. `GdcLetterbox` (1920×1080 → 640×640: 0.97 ms,
~0.3 ms of it CPU) has the CPU letterbox's geometry exactly but is not
bit-identical to it — the two resamplers alias differently on high-frequency
detail, a difference measured below the model's noise floor end to end.

## VpImage

`bcdl/preproc/vp_image.h` — RAII image backed by a cached `SysMem`; the unified
buffer for VP ops and the JPU/VPU codecs. Move-only.

```cpp
bcdl::VpImage img(width, height, HB_VP_IMAGE_FORMAT_NV12);  // BGR/RGB/Y/NV12
void* y = img.data();          // primary (Y) plane
int   st = img.raw().stride;   // row stride (16-byte aligned)
img.cleanCache(); img.invalidateCache();
```

Formats and layout: `BGR`/`RGB` interleaved C3 (`stride = align16(w*3)`); `Y`
grayscale C1 (`align16(w)`); `NV12` Y plane then interleaved UV (even dims).

---

## Detection

`bcdl/tasks/detection.h` — two decoder families.

```cpp
struct Detection { float x1, y1, x2, y2, score; int class_id; };  // original pixels
enum class DecodeLayout { kYoloV8, kYoloV5 };
```

**Single fused tensor** (`[1,4+nc,N]` / `[1,N,4+nc]`):

```cpp
bcdl::DetectConfig cfg;          // input_w/h, num_classes, conf_thresh,
cfg.num_classes = 80;            // iou_thresh, max_dets, layout, channels_first,
                                 // apply_sigmoid
// pure decode of a float tensor:
auto dets = bcdl::decode(data, shape /*={1,4+nc,N}*/, cfg, lb);
// or Engine-bound (dequantizes + de-strides + decodes):
bcdl::Detector det(engine, cfg, /*output_index=*/0);
auto dets2 = det.postprocess(lb);     // after engine.infer()
float i = bcdl::iou(a, b);
std::vector<int> keep = bcdl::nms(dets, cfg.iou_thresh, cfg.max_dets);
```

**Anchor-free LTRB multi-scale** (YOLO26 / standard RDK NV12 export) — paired
`(cls, box)` outputs per stride:

```cpp
bcdl::YoloLtrbConfig cfg;        // num_classes, conf/iou_thresh, max_dets,
cfg.strides = {8, 16, 32};       // strides, reg_max (DFL bins; 0 = plain LTRB)
bcdl::YoloLtrbDetector det(engine, cfg, /*output_base=*/0);
auto dets = det.postprocess(lb);      // reads 2*strides.size() outputs
// pure form:
auto d = bcdl::decodeYoloLtrb(cls, box, grid_hw, cfg, lb);
```

`reg_max > 0` selects the DFL head (`4*reg_max` box channels, softmax-reduced);
`YoloLtrbDetector` auto-detects it from the box channel count.

## Panoptic driving perception (YOLOP)

`bcdl/tasks/panoptic_drive.h` — the library's one ANCHOR-BASED detection decode
(the YOLOv5 formulation; everything else here is anchor-free LTRB/DFL). YOLOP
feeds it from the same inference that yields its drivable-area and lane-line
masks, which go through the ordinary `Segmenter`.

```cpp
struct Anchor { float w, h; };                               // model-input pixels
struct AnchorDetectConfig { int num_classes=1; float conf_thresh=0.35f, iou_thresh=0.45f;
                            int max_dets=300; std::vector<int> strides={8,16,32};
                            std::vector<std::vector<Anchor>> anchors; };  // per scale, 3 each

bcdl::Engine engine("models/yolop_cut_nashm_640x640_nv12.hbm");   // 5 outputs
// centred letterbox to 640x640 NV12, feedNv12Input(engine, nv12), engine.infer()
bcdl::AnchorDetector det(engine, bcdl::AnchorDetectConfig{}, /*output_base=*/0);
std::vector<bcdl::Detection> cars = det.postprocess(lb);    // outputs 0-2

bcdl::SegConfig sc; sc.num_classes = 2; sc.channels_first = true;
bcdl::SegMask drivable = bcdl::Segmenter(engine, sc, /*output_index=*/3).postprocess();
bcdl::SegMask lanes    = bcdl::Segmenter(engine, sc, /*output_index=*/4).postprocess();
// pure: decodeYoloV5Anchor(raw, grid_hw, cfg, lb)
```

| member | description |
|---|---|
| `AnchorDetector(engine, cfg={}, output_base=0)` | Bind `strides.size()` consecutive raw head outputs; throws if they run past the model's outputs. |
| `postprocess(lb)` | Dequantize and de-stride each head, take each scale's `(H, W)` from its own tensor, check its channel count is `na * (5 + num_classes)` — a wrong class count fails loudly instead of misreading columns — then decode and run one per-class NMS. |

**Take the `_cut` build.** The published export decodes in-graph and emits a
ready-made `[1, 25200, 6]` tensor, but that decode is assembled from `ScatterND`
and does not survive BPU compilation: the compile succeeds, and the compiled
tensor's objectness and class columns are simply never written, so a detector
finds zero objects at any threshold. The shipped build is cut before the decode
and emits the three raw head convolutions (`[1, na*(5+nc), H, W]`, unactivated),
and the arithmetic runs here on the CPU. That is also 10× faster: the in-graph
decode cost 28.33 ms on the device, against 2.80 ms (358 FPS) for the cut model's
whole inference, all three heads. On real frames the CPU decode matches the
export's own to 1.8e-4 over all 25200 candidates.

The decode is the reference's: sigmoid on every attribute (xy and wh included),
centre `(2·s − 0.5 + cell) · stride`, size `(2·s)² · anchor`, score = objectness ×
best class. Anchor sizes are in pixels, not stride units, and every scale must
carry the same number of them. The two masks come back on the model's
640×640 letterboxed grid; `Segmenter` takes no `LetterboxInfo`, so crop the
padding and resize them onto the frame yourself.

## Classification

`bcdl/tasks/classification.h`

```cpp
struct ClsConfig { int top_k = 5; bool apply_softmax = true; };
struct ClsResult { int class_id; float score; };

bcdl::Classifier clf(engine, cfg, /*output_index=*/0);
std::vector<bcdl::ClsResult> top = clf.postprocess();   // after infer()
// pure: decodeClassification(logits, num_classes, cfg)
```

## Image embeddings

`bcdl/tasks/embedding.h` — an image to a fixed-length unit vector, and a table to
search such vectors in. One operation does two jobs: **zero-shot
classification** dots the image vector against a table of class-name vectors,
**retrieval** against a table built from gallery images.

```cpp
struct EmbedConfig { bool l2_normalize = true; };
struct EmbedMatch { int index; float score; std::string label; };   // cosine, best first

// a .hbm is a package: bind the POOLED submodel by name, not the package default
std::vector<std::string> names = bcdl::Engine::modelNames(path);
bcdl::Engine engine(path, pooled_name);
bcdl::ImageEmbedder emb(engine);              // check emb.dim() once at startup

// squashing resize to the input size (NOT a letterbox), RGB, x / 127.5 - 1, NCHW
engine.setInput(0, x.data(), x.size() * sizeof(float));
engine.infer();
std::vector<float> v = emb.postprocess();     // L2-normalized

bcdl::EmbeddingBank bank;                     // the first add() fixes dim()
bank.add(gallery_vec, "img_0001.jpg");        // or a class name's text vector
std::vector<bcdl::EmbedMatch> top = bank.search(v, /*k=*/5);
// pure: decodeEmbedding(data, dim, cfg)
```

| member | description |
|---|---|
| `ImageEmbedder(engine, cfg={}, output_index=0)` | Bind one output; throws if the index is out of range. |
| `ImageEmbedder::postprocess()` | Flatten the output to `dim()` floats (dequantized if needed) and normalize per `cfg`. |
| `ImageEmbedder::dim()` | Product of the bound output's dims. |
| `EmbeddingBank::add(vec, label="")` | Append one vector, normalized into the bank (the caller's copy is untouched). A length different from the first entry's throws `Error(-1)`. |
| `EmbeddingBank::search(query, k=5)` | The `k` best entries by cosine, descending; `query` may be unnormalized. `k <= 0` or beyond `size()` returns every entry; an empty bank returns none; a wrong length throws. |
| `size()` · `dim()` · `label(i)` | |

**Bind the pooled submodel.** An embedding `.hbm` usually packs a pooled
whole-image vector *and* a per-patch feature grid, and the constructor's default
is whichever comes first. A pooled head is `[1, D]` and gives the tower's width;
a patch head is `[1, N, D]` and flattens to `N*D`, which `ImageEmbedder` cannot
tell from a very wide embedding — compare `dim()` with the tower's documented
width. [`examples/embed_demo.cc`](../examples/embed_demo.cc) lists the names with
`Engine::modelNames()` and binds the pooled one.

**Squash, don't letterbox.** These towers were trained on a direct resize to a
square; letterbox bars are pixels they never saw, and they move the vector
enough to cost retrieval accuracy. An embedding carries no coordinates, so there
is nothing to invert afterwards. `x / 127.5 - 1` on RGB is what the demo feeds
SigLIP; check the model card before assuming it for another tower.

Only the image tower runs on the board. A dual encoder's text tower runs once,
offline, on a host, and its vectors for a fixed prompt set ship as a table to
`add()`. `search()` is a linear scan, the right choice at board-sized tables
(thousands of entries). SigLIP base/16 takes 19.3 ms per image; against the float
model the per-image cosine is 0.9977–0.9995, and recall@1/@5 on a 200-image
gallery is identical to float (0.9250 / 1.0000).

## Face detection & alignment

`bcdl/tasks/face.h` — SCRFD detection with five landmarks, and the alignment
step recognition depends on. Recognition itself is an ordinary
[image embedding](#image-embeddings): detect → align to 112×112 → embed → compare.

```cpp
struct FaceDetection { float x1, y1, x2, y2, score;
                       std::vector<std::pair<float, float>> landmarks; };  // 5, original px
struct FaceDetectConfig { float conf_thresh=0.5f, iou_thresh=0.4f; int max_faces=100;
                          std::vector<int> strides={8,16,32}; int num_anchors=2; };

bcdl::Engine det_engine("models/scrfd_10g_nashm_640x640_nv12.hbm");   // 9 outputs
// SCRFD pads BOTTOM-RIGHT: the image sits at the top-left, so the letterbox has no pad
bcdl::LetterboxInfo lb = bcdl::computeLetterbox(w, h, 640, 640, /*centerPad=*/false);
// resize by lb.scale into the top-left of a zeroed 640x640 canvas, bgrToNv12Cpu,
// feedNv12Input(det_engine, nv12), det_engine.infer()
bcdl::FaceDetector det(det_engine, bcdl::FaceDetectConfig{}, /*output_base=*/0);
std::vector<bcdl::FaceDetection> faces = det.postprocess(lb);

std::vector<uint8_t> crop = bcdl::alignFace(bgr, w, h, /*src_stride=*/0,
                                            faces[0].landmarks, /*size=*/112);
bcdl::Engine rec_engine("models/arcface_r50_aligned_nashm_112x112.hbm");
// crop -> RGB, (x - 127.5) / 127.5, NCHW float32 -> rec_engine.setInput(), infer()
bcdl::ImageEmbedder rec(rec_engine);          // dim() == 512
bcdl::EmbeddingBank people;
people.add(rec.postprocess(), "alice");
// pure: decodeScrfd(score, bbox, kps, counts, cfg, lb) · similarityTransform(src, dst) · arcFaceTemplate()
```

| member | description |
|---|---|
| `FaceDetector(engine, cfg={}, output_base=0)` | Bind `3 * strides.size()` outputs as three groups — every score, then every bbox, then every kps, the order the reference export emits them in; throws if the model has fewer. |
| `FaceDetector::postprocess(lb)` | Decode every scale, threshold, run one NMS; boxes and landmarks in original-image pixels, boxes clamped to the image. Throws if a bbox or kps head is smaller than its score head implies. |
| `alignFace(bgr, src_w, src_h, src_stride, landmarks, size=112)` | Warp the face onto `arcFaceTemplate()` scaled to `size` with the similarity transform; bilinear, edge-clamped; returns a packed `size*size*3` BGR crop. `src_stride` 0 means tightly packed; anything but five landmarks throws. |

Landmarks come in the template's order: left eye, right eye, nose, left mouth
corner, right mouth corner — "left" being the viewer's left.

**This detector pads bottom-right.** Its reference preprocessing scales the
image to fit and leaves it at the top-left corner, so its `LetterboxInfo` has
`padX = padY = 0`; a centred letterbox shifts every box and landmark by the pad.
The head is anchor-free distance regression with `num_anchors` predictions per
cell, ordered (row, column, anchor); the cell centre sits ON the grid point, with
no half-cell offset, and distances are in stride units. `num_anchors` must
divide each tensor evenly, else `Error(-1)`.

**Alignment is a correctness requirement, not a refinement.** The recognition
model only ever saw faces warped so that eyes, nose and mouth corners land on
the template; a plain crop of the box degrades its embeddings badly while
erroring nowhere. `similarityTransform()` is the closed-form Umeyama solution
(rotation + uniform scale + translation), deterministic, and unable to shear or
stretch the face.

Against the float reference on the same frames: landmarks 0.35 px mean / 0.89 px
max, box corners 0.52 px mean; 60.2 ms detection (SCRFD-10G), 0.29 ms alignment.
Recognition is 0.975 mean cosine against float (0.966 min). ArcFace's input rows
are padded on the device (448 → 512 bytes); `setInput()` scatters a packed array
into that layout, and before it did, the padding alone left the embeddings at
0.015 cosine.

## Pose

`bcdl/tasks/pose.h` — LTRB multi-scale; person box + K keypoints. Three outputs
per scale `(cls, box, kpt)`.

```cpp
struct Keypoint { float x, y, score; };
struct PoseDetection { float x1,y1,x2,y2,score; int class_id;
                       std::vector<Keypoint> keypoints; };
struct PoseConfig { int num_keypoints=17; float conf_thresh, iou_thresh;
                    int max_dets; std::vector<int> strides; };

bcdl::PoseEstimator est(engine, cfg, /*output_base=*/0);
auto poses = est.postprocess(lb);   // reads 3*strides.size() outputs
// pure: decodePose(cls, box, kpt, grid_hw, cfg, lb)
```

`num_keypoints` is taken from the kpt tensor (last dim / 3), not trusted from cfg.

## Whole-body pose (133 keypoints)

`bcdl/tasks/wholebody.h` — TOP-DOWN: one inference per PERSON on a crop, so a
detector runs in front of it and cost scales with the head count. One output,
`[1,K,H,W]` channel-first heatmaps.

```cpp
struct WholeBodyCrop { int x1, y1, pad_left, pad_top, padded_w, padded_h; };
struct WholeBodyConfig { float kpt_thresh; int blur_kernel, box_pad;
                         float mean[3], std[3]; };

std::vector<float> in;
auto crop = bcdl::wholeBodyPreprocess(bgr, w, h, stride, x1, y1, x2, y2,
                                      192, 256, cfg, in);   // BGR in, RGB out
engine.setInput(0, in.data(), in.size() * sizeof(float));
engine.infer();
bcdl::WholeBodyEstimator est(engine, cfg);
auto kpts = est.postprocess(crop);   // 133 Keypoints, original-image pixels
// pure: decodeWholeBody(heatmaps, num_kpts, hm_h, hm_w, crop, cfg)
```

Layout: `0-16` body, `17-22` feet, `23-90` face, `91-111` left hand,
`112-132` right hand. K/H/W come from the tensor, not from cfg. The crop is the
reference's widen-pad-resize, NOT an mmpose affine, and sub-pixel refinement is
DARK-UDP over a window around each peak.

## Super-resolution (tiled)

`bcdl/tasks/superres.h` — a fixed-tile upscaler applied over an arbitrary image
by overlapping tiles and cross-fading.

```cpp
struct SuperResConfig { int overlap; };          // input pixels
struct SrImage { int width, height; std::vector<uint8_t> data; };  // BGR
struct TilePlacement { int x, y; };              // one tile's origin, input pixels

bcdl::SuperResolver sr(engine, cfg);
auto big = sr.upscale(bgr, w, h, stride);        // sr.scale() times larger
int tiles = sr.lastTileCount();                  // tiles the last call ran; cost is linear in it
// pure: planTiles(w, h, tile_w, tile_h, overlap) -> std::vector<TilePlacement>, tileWeight(i, len, ramp)
```

`scale()` and `tile()` are read from the model's shapes, never configured. The
blend accumulates `w * pixel` and `w` and divides, so image borders need no
special case. Note that the compiled `.hbm` scales with tile AREA — the same
network is 148 MB at a 256 tile and 37 MB at 128, for identical per-pixel
throughput.

`planTiles` pushes the last tile on each axis flush against the far edge
instead of letting it hang over, so the final overlap can be larger than asked
but never smaller, and no tile reads outside the image. A side shorter than one
tile gets a single origin at 0: `upscale()` edge-pads such an image up to the
tile itself, a direct `planTiles` caller has to.

## Sparse local features (XFeat)

`bcdl/tasks/features.h` — keypoints + L2-normalized 64-d descriptors, and
mutual-NN matching. Three outputs (`feats` 64ch, `keypoints` 65ch, `reliability`
1ch) at 1/8 scale. The input's InstanceNorm and every data-dependent step
(softmax / NMS / top-k / sampling) are on the CPU by design.

```cpp
struct Feature { float x, y, score; };
struct FeatureSet { std::vector<Feature> keypoints;
                    std::vector<float> descriptors; int dim; };
struct XfeatConfig { float detection_thresh; int nms_kernel, top_k; };

bcdl::FeatureExtractor ext(engine, cfg);
auto a = ext.extract(bgr_a, w, h, stride);
auto b = ext.extract(bgr_b, w, h, stride);
auto m = bcdl::matchFeatures(a, b, /*min_cossim=*/0.82f);
// pure: xfeatPreprocess(...) / decodeXfeat(feats, kpts, rel, fh, fw, ...)
```

Descriptors are sampled BICUBICALLY (the reference sampler's default) while the
reliability map is bilinear. `matchFeatures` is `O(|a|*|b|*dim)` and OpenMP
parallel — `XfeatConfig::top_k` is the knob that decides whether a pair costs
130 ms or 8 ms.

## Instance segmentation

`bcdl/tasks/instance_seg.h` — LTRB boxes + per-instance binary mask from a
prototype tensor. Three outputs per scale `(cls, box, mc)` plus one `proto`.

```cpp
struct InstanceMask { float x1,y1,x2,y2,score; int class_id;
                      int mask_w, mask_h; std::vector<uint8_t> mask; }; // 0/1
struct InstanceSegConfig { float conf_thresh, iou_thresh; int max_dets;
                           std::vector<int> strides; int proto_index=9;
                           bool compute_masks=true;
                           int reg_max=0; };   // 0 = plain LTRB; set by the segmenter from the box head

bcdl::InstanceSegmenter seg(engine, cfg, /*output_base=*/0);
auto masks = seg.postprocess(lb, orig_w, orig_h);
// pure: decodeInstanceSeg(cls, box, mc, grid_hw, num_classes, num_coef,
//                         proto, proto_h, proto_w, proto_c, cfg, lb, orig_w, orig_h)
```

Set `compute_masks=false` to get boxes/scores only and skip mask assembly.

## Open-vocabulary detection (YOLOE)

`bcdl/tasks/open_vocab.h` — a prompt-free YOLOE export has its vocabulary FIXED
at export time: the CLIP text embeddings are folded into the classification
convolution, so at runtime it is a plain anchor-free head and decodes through
the existing classes unchanged — `YoloLtrbDetector` for a detection build,
`InstanceSegmenter` for a segmentation build (both detect the DFL box head from
its channel count). The only new state is the class-id → prompt-name table.

```cpp
struct LabelMap { std::vector<std::string> names; };   // + the members below

bcdl::LabelMap labels = bcdl::LabelMap::fromFile("models/yoloe26m_indoor99_labels.txt");
bcdl::InstanceSegConfig cfg; cfg.compute_masks = false;   // boxes only
bcdl::InstanceSegmenter seg(engine, cfg);
for (const auto& m : seg.postprocess(lb, orig_w, orig_h))
  std::printf("%s %.2f\n", labels.name(m.class_id).c_str(), m.score);

// a detection build: YoloLtrbConfig::num_classes must equal labels.size()
```

| member | description |
|---|---|
| `LabelMap::fromFile(path)` | One label per line, in class-id order; whitespace trimmed, blank lines dropped. Throws `std::runtime_error` if the file cannot be opened. |
| `LabelMap::fromList(names)` | From an in-memory list. |
| `size()` · `name(id)` | `name()` returns `"?"` for an id out of range instead of throwing. |

Keep the `.hbm` and its labels file side by side: the file is the only record of
what each class id means, and **a new vocabulary means a new export** —
`fromList()` names the classes of a model, it does not change what the model
detects. The shipped builds are YOLOE-11s with the COCO-80 classes (detection and
segmentation) and YOLOE-26m with a fixed 99-class indoor vocabulary
(segmentation; names in `yoloe26m_indoor99_labels.txt`).

The indoor build is zero-shot — nothing was trained. On held-out Open Images
indoor images it scores 0.484 mAP50 over the 25 classes with at least 20 boxes
(float 0.478): on par with a supervised detector overall, not class by class.
Monitors are a blind spot, and the navigation classes (stairs, elevators,
escalators) score low enough to need a threshold below 0.25. Keep
`compute_masks` off when boxes are enough — assembling the masks costs far more
CPU time than the network costs on the BPU. Details:
[MODELS.md](MODELS.md#the-indoor-yoloe-build-is-zero-shot).

## Promptable segmentation (EdgeSAM)

`bcdl/tasks/promptable_seg.h` — the tail of a SAM-family mask decoder (EdgeSAM,
MobileSAM, SAM): pick the best candidate mask and binarize it. The session
around it is two models — an image encoder run once per image, and a light
prompt decoder run per click against the cached embedding.

```cpp
struct SamConfig { float mask_threshold = 0.0f; bool multimask = true; };
struct SamMask { int index; float iou; int mask_w, mask_h;
                 std::vector<uint8_t> mask; };           // Hm*Wm, 0/1, low resolution

bcdl::Engine enc("models/edge_sam_encoder_nashm.hbm");
bcdl::Engine dec("models/edge_sam_decoder_sp1_nashm.hbm");   // one point; _bp2 = a box
// enc: longest side to 1024, pad bottom-right, RGB, raw 0-255 -> infer once per image
// dec: the cached embedding, prompt coords in that 1024 canvas, SAM labels -> infer
bcdl::SamMaskDecoder tail(dec, bcdl::SamConfig{}, /*masks_index=*/0, /*iou_index=*/1);
bcdl::SamMask m = tail.postprocess();
// pure: decodeSamMasks(low_res_logits, num_masks, mask_h, mask_w, iou_pred, cfg)
```

| member | description |
|---|---|
| `SamMaskDecoder(engine, cfg={}, masks_index=0, iou_index=1)` | Bind the decoder's mask-logit output (`[1,N,Hm,Wm]` or `[N,Hm,Wm]`) and its IoU output (`[1,N]` or `[N]`). Pass your build's indices; the masks output is the 4-D one. |
| `SamMaskDecoder::postprocess()` | `decodeSamMasks()` on the current outputs; throws on an index out of range or a mask tensor of another rank. |

`decodeSamMasks` takes the argmax-IoU candidate when `multimask` is set (else
candidate 0, as also when `iou_pred` is null) and keeps `logit > mask_threshold`
— SAM's 0.0 on raw logits is sigmoid > 0.5. The mask stays at the decoder's low
resolution (256×256 for SAM); undoing the pad and resizing to the original image
is the caller's job.

**Only this tail is C++.** The two-stage session is the Python `SamSession`, and
it is the reference for feeding the models: the encoder takes the image with its
longest side resized to 1024 and padded bottom-right, BGR→RGB, as raw pixels
(the ImageNet normalization is compiled into the encoder); the decoder takes the
embedding, the prompt coordinates scaled into that canvas, and SAM labels (1 for
a foreground point, 2 / 3 for a box's two corners), as inputs 0, 1 and 2. Each
decoder build has a fixed prompt size: `_sp1` one point, `_bp2` a two-corner box.

**The shipped decoders' IoU head is int8-degraded**, so `SamSession` picks the
largest-area candidate rather than trust it. `SamMaskDecoder` has no area rule —
with `multimask = true` it trusts the IoU head, with `multimask = false` it takes
candidate 0 — so to select by area, read the logits with `outputAsFloat()` and
pick yourself.

## Oriented boxes (OBB)

`bcdl/tasks/obb.h` — LTRB + angle, rotated-IoU NMS. Three outputs per scale
`(cls, box, angle)`.

```cpp
struct RotatedBox { float cx, cy, w, h, angle; };      // angle in radians
struct ObbDetection { RotatedBox rrect; float score; int class_id; };
struct ObbConfig { int num_classes=15; float conf_thresh, iou_thresh; int max_dets;
                   std::vector<int> strides; bool regularize=true;
                   float angle_offset_rad=0; int angle_sign=1; };

bcdl::ObbDetector obb(engine, cfg, /*output_base=*/0);
auto dets = obb.postprocess(lb);
float i = bcdl::rotatedIoU(a, b);
std::vector<int> keep = bcdl::rotatedNms(dets, cfg.iou_thresh, cfg.max_dets);
// pure: decodeObb(cls, box, angle, grid_hw, cfg, lb)
```

## Semantic segmentation

`bcdl/tasks/segmentation.h` — argmax a logit tensor (or pass through ids) to a
per-pixel label map.

```cpp
struct SegConfig { int num_classes=0; bool channels_first=true; bool argmaxed=false; };
struct SegMask { int width, height, num_classes; std::vector<int32_t> labels; };

bcdl::Segmenter seg(engine, cfg, /*output_index=*/0);
bcdl::SegMask m = seg.postprocess();
std::vector<uint8_t> bgr = bcdl::segColorize(m);   // (H*W*3) BGR palette
// pure: decodeSeg(data, shape, cfg)
```

`num_classes=0` infers C from the tensor; `argmaxed=true` if the model already
emits ids.

## Monocular 3-D detection (SMOKE)

`bcdl/tasks/mono3d.h` — one camera image to KITTI-style 3-D boxes (Car, Cyclist,
Pedestrian). CenterNet-style: a class heatmap locates each object's projected
3-D centre, and an 8-channel regression map at that pixel, with the camera
intrinsics, lifts it to a full box.

```cpp
struct CameraIntrinsics { float fx, fy, cx, cy; };   // the ORIGINAL image's, zero skew
struct Mono3dBox { int class_id; float score; float x, y, z, h, w, l, yaw, alpha;
                   std::array<float, 4> box2d; };    // camera frame, metres; box2d original px
struct Mono3dConfig { int num_classes=3; float conf_thresh=0.25f; int max_dets=50;
                      int nms_kernel=3; bool pred_2d=true;
                      std::array<float, 2> depth_ref;               // {28.01, 16.32}
                      std::vector<std::array<float, 3>> dim_ref; }; // per-class size priors

// input: [1,3,384,1280] F32 RGB, /255 + ImageNet mean/std (not fused into the .hbm),
// warped with the same scale-to-width map the decode uses
bcdl::Mono3dDetector det(engine, bcdl::Mono3dConfig{}, /*output_base=*/0);  // cls, then reg
bcdl::CameraIntrinsics K{fx, fy, cx, cy};            // from your calibration
std::vector<bcdl::Mono3dBox> boxes = det.postprocess(orig_w, orig_h, K);
// pure: computeMono3dFeatureXform(origW, origH, featW, featH) · decodeMono3d(cls, reg, H, W, cfg, featXform, K)
```

Boxes are in KITTI camera coordinates — +x right, +y down, +z forward, metres —
and `(x, y, z)` is the box's BOTTOM centre; `yaw` is rotation_y about the camera
Y axis, `alpha` the observation angle, and `box2d` the 3-D box projected with `K`
and clamped to the image (`pred_2d`). The `Mono3dConfig` defaults are the
reference SMOKE configuration.

**The geometry is not a letterbox.** SMOKE scales to the image WIDTH and
centres the height, cropping or padding vertically: `scale = featW / origW`,
`padX = 0`, `padY = featH/2 − scale·origH/2`, which may be negative.
`computeMono3dFeatureXform()` builds that map at feature resolution, and the
384×1280 input warp is the same transform at 4× the scale. Preprocessing and
decode must agree, and `K` stays the original image's intrinsics — the resize
lives entirely in the transform, never folded into `K`.

Both outputs are **raw logits, channel-first** (`[1, nc, 96, 320]` and
`[1, 8, 96, 320]`, stride 4): the predictor's activations were stripped from the
compiled graph, and the decode re-applies them — a clamped sigmoid on the
heatmap, `sigmoid − 0.5` on the three dimension channels, L2 normalization of
`(sin, cos)` — before the CenterNet max-pool NMS and top-`max_dets`, matching the
SMOKE post-processor. `num_classes` comes from the cls tensor, never from `cfg`;
`dim_ref` must cover it, and a reg head with fewer than 8 channels or on another
grid throws `Error(-1)` instead of decoding garbage.

## Lidar 3-D detection

`bcdl/preproc/lidar_voxelize.h` + `bcdl/tasks/lidar_det3d.h` — a point cloud to
oriented 3-D boxes, as a pipeline of compiled stages with CPU work between.

```cpp
struct Box3D { float x,y,z, dx,dy,dz, heading; int label; float score; };

bcdl::VoxelConfig vcfg;                                   // KITTI geometry
bcdl::Pillars p = bcdl::voxelizePillars(points, n, /*stride=*/4, vcfg);
std::vector<float> x = bcdl::pillarPointFeatures(p.voxels.data(), p.num_points.data(),
                                                 p.coords.data(), p.num_pillars, vcfg);
// vfe engine -> pillar features
std::vector<float> canvas = bcdl::scatterPillars(feats, p.coords.data(),
                                                 p.num_pillars, 64, vcfg);
// backbone+head engine
bcdl::LidarDetector3d det(head_engine);
std::vector<bcdl::Box3D> boxes = det.postprocess();
// pure: generateAnchors3d(cfg) · decodeDet3d(...) · bevIou(a,b) · nms3d(...)
```

Points must already be in the EGO frame, and the caps in `VoxelConfig` are real
— `Pillars` reports every point and column it had to drop rather than truncating
silently.

`pillarPointFeatures()` is needed only by a VFE re-exported as a 1×1 convolution
over `(1, C, pillars, points)`; a VFE that takes the three raw pillar tensors does
the same expansion in its own graph. It is worth 179× to move it out: that shape
is a huge pillar axis against a 4-wide channel axis, and the in-graph version
spends 290 ms on a 95 MFLOP MLP where the convolution form spends 1.62 ms.

The anchor decode runs on the CPU on purpose: the published head bakes it into
the graph with `ScatterND`, which compiles without an error into a model whose
outputs are never written. `generateAnchors3d` reproduces that model's own anchor
constants to 8e-06, which is how the ordering (class-major, then rotation, on a
CLOSED linspace over the range) was established rather than guessed.

## Anomaly detection (one-class)

`bcdl/tasks/anomaly.h` — trained on one product's NORMAL images only; reports how
unlike them each patch looks. The model is per-product and the threshold is a
per-line decision, so the shipped model is an example, not a general detector.

```cpp
struct AnomalyConfig { int out_width=0, out_height=0; int blur_kernel=33;
                       float blur_sigma=4.0f, threshold=0.5f; };
struct AnomalyMap { int width, height; std::vector<float> data; float score; };

bcdl::AnomalyDetector det(engine);
bcdl::AnomalyMap m = det.detect(bgr, width, height, stride);
std::vector<uint8_t> mask = bcdl::anomalyMask(m, 0.5f);
// pure: decodeAnomaly(data, shape, cfg) · anomalyPreprocess(bgr, ..., out)
```

`score` is the map's **maximum** — a part is as anomalous as its worst patch, and
a mean lets a large clean region hide a small defect. The upsample to input
resolution and the Gaussian smoothing are the reference post-processing; setting
`blur_kernel = 0` skips the smoothing (worth ~6 ms at 288×288) but changes the
numbers.

`anomalyPreprocess` applies BGR→RGB, `/255` and ImageNet mean/std — that
normalisation is part of the trained contract, and the defaults are those
statistics.

## Optical flow

`bcdl/tasks/optical_flow.h` — two frames to a dense displacement field. Vectors
in source pixels, +u right / +v down (OpenCV convention).

```cpp
struct FlowConfig { float scale_x=1, scale_y=1; bool channels_first=true; };
struct FlowField { int width, height; std::vector<float> data; /* H*W*2, (u,v) */ };

bcdl::OpticalFlowEstimator est(engine);
bcdl::FlowField f = est.estimate(bgr0, bgr1, width, height, stride);
std::vector<uint8_t> vis = bcdl::flowColorize(f);            // BGR H*W*3
float epe = bcdl::flowEndpointError(f, reference);
// pure: decodeFlow(data, shape, cfg) · flowPreprocess(bgr, ..., out)
```

`estimate()` reads the model's input size from the Engine, resizes both frames,
and rescales the returned vectors from model pixels to source pixels — a field
left in model pixels warps everything by the wrong factor without looking wrong.

**Preprocessing keeps BGR and 0-255**: the reference implementation divides by
255 inside the graph. `flowPreprocess` is bound to the same C++ the Python path
uses, so both feed the model identical pixels.

## End-to-end driving (camera + lidar → trajectory)

`bcdl/tasks/e2e_drive.h` + `bcdl/preproc/drive_inputs.h` — three forward cameras
and a lidar sweep in, one planned ego trajectory out (plus detected vehicles and
a BEV semantic map). Ego frame, metres, x forward / y left; the poses are the
future at a fixed interval.

```cpp
struct DrivePose  { float x, y, heading; };
struct DriveAgent { float x, y, heading, length, width, score; };
struct DriveResult { std::vector<DrivePose> trajectory;
                     std::vector<DriveAgent> agents; bcdl::SegMask bev; };

bcdl::DrivePlanner planner(engine, anchors, num_modes, num_poses);

std::vector<float> camera = bcdl::stitchDriveCameras(l, lw, lh, ls, f, fw, fh, fs,
                                                     r, rw, rh, rs);   // (3,H,W) RGB [0,1]
bcdl::LidarBev bev = bcdl::lidarBevHistogram(points, n, /*stride_floats=*/3);

bcdl::DriveStatus st; st.driving_command = 0; st.velocity_x = 5.0f;
bcdl::DriveResult res = planner.plan(camera, bev.data, st);
// pure: decodeDrive(traj, traj_shape, agents, agent_shape, labels, label_shape,
//                   bev, bev_shape, cfg)
```

The BEV grid's first axis is **x (forward)**, the second y — the same order as
the upstream histogram. `LidarBevConfig::use_ground_plane=false` (the default)
emits one channel holding only the points above `split_height`.

`DriveResult::bev`'s axes are **not** screen axes: row index increases with
FORWARD distance (row 0 is the ego end) and column index increases to the LEFT,
so drawing it north-up means reversing both. Upstream documents neither, and a
straight road is nearly symmetric under a 180-degree rotation — the wrong
orientation is not visibly wrong. The map is returned as the model emits it.

`DriveAnchorConfig::noise_scale = 0` zeroes the sampler's random term while
keeping the schedule's scaling, which is the configuration to use when comparing
against another implementation. `DriveIoConfig` maps tensors to indices; an
output index of `-1` skips that head.

Inputs are written through `setInputFromFloat` (`bcdl/backend/input_writer.h`),
which quantizes to whatever dtype the model declared — this model's inputs are
int16 with a per-tensor scale, and handing it raw float bytes would run happily
on noise.

## Depth

`bcdl/tasks/depth.h` — single-channel depth/disparity to a float map.

```cpp
struct DepthConfig { int width=0, height=0; bool normalize=true;
                     float clip_lo=0, clip_hi=0; };
struct DepthMap { int width, height; std::vector<float> data; float vmin, vmax; };

bcdl::DepthEstimator est(engine, cfg, /*output_index=*/0);
bcdl::DepthMap dm = est.postprocess();
std::vector<uint8_t> g8  = bcdl::depthToGray8(dm);   // (H*W)
std::vector<uint8_t> bgr = bcdl::depthColorize(dm);  // (H*W*3) Turbo BGR
// pure: decodeDepth(data, shape, cfg)
```

## Depth refinement (RGB-D)

`bcdl/tasks/depth_refine.h` — an existing noisy/holey depth map plus the aligned
RGB frame in, cleaned metric depth + a trust mask out (LingBot-Depth). Unlike the
monocular and stereo heads this one refines rather than estimates.

```cpp
struct DepthRefineConfig { int encoder_height=420, encoder_width=560;
                           float min_valid_depth=0.01f, mask_threshold=0.5f;
                           bool apply_mask=true; };
struct RefinedDepth { int width, height; std::vector<float> depth;
                      std::vector<uint8_t> mask; float vmin, vmax; };
struct Intrinsics { float fx, fy, cx, cy; };

bcdl::DepthRefiner refiner(engine);            // adopts the model's encoder grid
bcdl::RefinedDepth r = refiner.run(bgr, w, h, stride, depth_m, depth_stride);

bcdl::Intrinsics k = bcdl::scaleIntrinsics(sensor_k, w, h, r.width, r.height);
std::vector<float> pts = bcdl::depthToPointCloud(r, k);   // (H*W*3) metres

// pure pieces, for feeding the engine yourself:
//   preprocessRefineImage(bgr, w, h, stride, cfg, &image_tensor)   // [3,eh,ew]
//   preprocessRefineDepth(depth_m, w, h, stride, cfg, &log_tensor) // [1,eh,ew]
//   depthU16ToMetres(raw16, w, h, stride, /*scale=*/1000.f)
//   decodeRefinedDepth(depth, mask_logit, h, w, cfg)
```

The engine's two inputs must be F32 featuremaps (the constructor throws
otherwise). Depth enters as log depth with 0 for "no reading", and the compiled
graph keeps every depth token where upstream drops empty ones — see
[`MODELS.md`](MODELS.md) and the header for what that costs.

## OCR

`bcdl/tasks/ocr.h` — three independent stages (det / cls / rec); the application
composes them (crop each detected box, optional 180° flip, recognize). Each stage
has a pure decoder + an Engine-bound wrapper. PP-OCRv6 is the default; v5 is kept
as a fallback.

```cpp
// Recognition (CRNN + CTC):
std::vector<std::string> dict = bcdl::loadCharDict("ppocr_dict.txt"); // blank @ 0
struct RecResult { std::string text; float score; };
bcdl::TextRecognizer rec(engine, "ppocr_dict.txt", /*out_idx=*/0);
bcdl::RecResult r = rec.postprocess();
// pure: decodeCtc(logits, num_steps, num_classes, dict)

// Direction (0°/180°):
struct ClsDirResult { int label; float score; bool flip180; };
bcdl::TextAngleClassifier cls(engine, /*thresh=*/0.9f, /*out_idx=*/0);
// pure: decodeClsDir(logits, n, thresh)

// Detection (DBNet, pure-C++ CCL + unclip):
struct DbConfig { float bin_thresh=0.3, box_thresh=0.6, unclip_ratio=1.5;
                  int min_size=3, connectivity=8; };
struct TextBox { float pts[8]; float x1,y1,x2,y2; float score; }; // 4-point + bbox
bcdl::DbTextDetector det(engine, DbConfig{}, /*out_idx=*/0);
std::vector<bcdl::TextBox> boxes = det.postprocess(lb);
// pure: decodeDbnet(prob, H, W, cfg, lb)
```

See [`examples/ocr_demo.cc`](../examples/ocr_demo.cc) for the full det→cls→rec
wiring (crop ordering + dict conventions handled there).

---

## Voice pipeline

Wake word → VAD → recognition → command → spoken reply, every network on the
BPU: [`examples/voice_assistant.cc`](../examples/voice_assistant.cc) is the
whole loop in C++ (WAV in, WAV replies out; with no input it synthesises a
demo session). Audio format, wake-word tokens and tuning are in the
[voice pipeline overview](API.md#voice-pipeline); the classes are below.

## Speech recognition (Paraformer)

`bcdl/preproc/fbank.h` — Kaldi-compatible front end; `bcdl/tasks/asr.h` — CIF,
token decoding and the three-stage `ParaformerAsr`. One 16 kHz mono utterance
(at most `maxSeconds()`, 24 s for the shipped build) in, text out.

```cpp
bcdl::Engine enc("models/paraformer_encoder_int16.hbm");
bcdl::Engine pred("models/paraformer_predictor_int16_v2.hbm");
bcdl::Engine dec("models/paraformer_decoder_int16_v2.hbm");
bcdl::ParaformerAsr asr(enc, pred, dec,
                        bcdl::loadTokenList("models/paraformer_tokens_8404.json"),
                        bcdl::loadKaldiCmvn("models/paraformer_am.mvn"));
bcdl::AsrResult r = asr.transcribe(pcm, num_samples);   // float in [-1, 1]
// r.text, r.token_ids, r.feature_frames, r.truncated

bcdl::Fbank fbank;                                         // FbankConfig{} = 16 kHz, 80 mel
std::vector<float> feats;
int frames = fbank.compute(pcm, num_samples, feats);       // [frames, 80]
// pure: lfrStack(feats, frames, dim, lfr_m, lfr_n, out) · parseKaldiCmvn(text)
// pure: cifIntegrate(alphas, hidden, frames, dim, valid_frames, cfg, embeds)
// pure: argmaxTokens(logits, rows, vocab) · tokensToText(ids, vocab)
```

| member | notes |
|---|---|
| `transcribe(pcm, samples, timeout_ms)` | front end + 3 BPU stages + CIF. Longer than the window → truncated, `truncated` set |
| `transcribeFeatures(feats, frames, timeout_ms)` | start from `[frames, featureDim()]` LFR+CMVN features |
| `features(pcm, samples, out)` | front end only |
| `maxFrames()` · `maxSeconds()` · `featureDim()` · `maxTokens()` · `vocabSize()` | geometry read from the Engines |

`ParaformerAsr` matches the stages by tensor shape rather than index, and refuses
a token list whose size differs from the decoder's class count. `cifIntegrate`
puts `CifConfig::tail_threshold` at frame `valid_frames` — the true end of
speech. A fixed-shape predictor graph can only add it at the end of its padding,
where masking the padding removes it, and any utterance whose last token
needed the tail then loses it.


### Voice activity detection — `bcdl/tasks/vad.h`

```cpp
bcdl::Engine eng("models/silero_vad_16k_int16_nashm.hbm");
bcdl::SileroVad vad(eng);                     // stateful: one stream per object
float p = vad.probability(chunk);             // 512 samples (32 ms) -> speech probability
bcdl::VadSegmenter seg(vad);                  // VadConfig{}: 0.5, 250 ms on, 300 ms off
for (auto& s : seg.feed(pcm, n))              // any length; returns the utterances that closed
  asr.transcribe(s.pcm.data(), s.pcm.size()); // s.start / s.end: samples since reset()
for (auto& s : seg.flush()) { /* end of stream */ }
```

| member | notes |
|---|---|
| `SileroVad::probability(chunk, timeout_ms)` · `reset()` | `kChunk` = 512 samples; carries LSTM state and the last 64 samples |
| `VadSegmenter::feed(pcm, n)` · `flush()` · `segments(pcm, n)` | `VadSegment{start, end, pcm}`; `segments` = reset + feed + flush |
| `inSpeech()` · `lastProbability()` · `config()` | |
| `VadConfig` | `threshold`, `neg_threshold` (< 0: = threshold), `min_speech_ms`, `min_silence_ms`, `speech_pad_ms`, `max_speech_s` |

The model's STFT (4 Hann frames of 256, hop 128) runs on the CPU and the BPU
takes the magnitude spectrum: squared on the BPU, the power spans ~5000 and
int16 loses every quiet bin. A segment opens after `min_speech_ms` at or above
`threshold` and closes after `min_silence_ms` below `neg_threshold`; a segment
reaching `max_speech_s` is cut there and a new one starts.


### Keyword spotting (wake word) — `bcdl/tasks/kws.h`

```cpp
bcdl::Engine enc("models/kws_zipformer_zh_en_3m_c16_int16_nashm.hbm");
bcdl::KeywordSpotter kws(enc, "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                         "models/kws_zipformer_zh_en_3m_tokens.txt");   // KwsConfig{}
kws.setKeywords("n ǐ h ǎo x iǎo ch ē @你好小车\n"
                "x iǎo ch ē x iǎo ch ē :1.5 #0.3 @小车小车\n");    // or loadKeywords(path)
for (const bcdl::KwsDetection& d : kws.feed(pcm, n))     // any length, float in [-1, 1]
  printf("%s at %.2f s\n", d.keyword.c_str(), d.end);
```

| member | notes |
|---|---|
| `setKeywords(text)` · `loadKeywords(path)` · `keywords()` | one line per keyword: model tokens, optional `:boost` and `#threshold`, `@phrase`; resets the stream |
| `feed(pcm, n, timeout_ms)` | `KwsDetection{keyword, index, start, end, tokens}`, seconds since `reset()` |
| `flush(timeout_ms)` | end of stream: pushes the last partial chunk through with silence, returns what fires, resets |
| `reset()` · `encode(pcm, n, frames)` · `tokenId(tok)` · `vocabSize()` | `encode` returns the encoder output of a whole recording (tests) |
| `KwsConfig` | `keywords_score` (1.0), `keywords_threshold` (0.25), `num_trailing_blanks` (1), `max_active_paths` (4) — sherpa-onnx's defaults |

The encoder runs on the BPU one 160 ms chunk at a time and carries 38 cache
tensors between calls; the decoder, joiner and keyword search (a beam search
over an Aho-Corasick graph of the keywords) run on the CPU and follow
sherpa-onnx's KeywordSpotter step for step — a float transcription of the same
search gave sherpa-onnx's detections on all 87 clips it was checked on. Features are Kaldi fbank
with the Povey window on samples in [-1, 1] (`FbankConfig::window`,
`input_scale`).


### Voice commands — `bcdl/tasks/command.h`

```cpp
bcdl::CommandMatcher m({"前进", "后退", "左转", "右转", "打开灯", "关灯"});  // max_distance 0.5
bcdl::CommandMatch c = m.match(r.text);   // "大开灯" -> index 4, distance 0
if (c.index >= 0) { /* c.command */ }
// pure: toPinyin(text) -> std::vector<std::string>, one toneless syllable per CJK char
```

| member | notes |
|---|---|
| `match(text)` | `CommandMatch{index (-1 = rejected), nearest, distance, command}` |
| `distances(text)` | normalised syllable edit distance to every command |
| `commands()` · `size()` · `maxDistance()` | |

Distance is the syllable edit distance over the command's syllable count; ties
go to the closer length, then the earlier command. Latin words compare only
with Latin words. On a 40-command set it lifts Paraformer from 84.7% / 82.8%
exact (clean / 5 dB) to 96.9% / 95.6% matched, with no false accepts on 600
news sentences. The pinyin table (U+3400..U+9FFF) is compiled in and generated
by `scripts/gen_pinyin_table.py`.


### Speech synthesis back end — `bcdl/preproc/istft.h`, `bcdl/tasks/vocoder.h`

```cpp
bcdl::Engine eng("models/vocos_16k_t256_int16_nashm.hbm");
bcdl::VocosVocoder voc(eng);                       // VocosConfig{}: context 32, pad -16
std::vector<float> pcm;
int n = voc.synthesize(mel, frames, pcm);          // mel [80, frames] row-major -> 16 kHz PCM
// pure: Istft(IstftConfig{}).compute(real, imag, frames, out)        // [frames, 513] each
// pure: Istft::computePolar(mag, cos, sin, frames, out)               // [513, frames] each
```

| member | notes |
|---|---|
| `synthesize(mel, frames, pcm, timeout_ms)` | chunks of `chunkFrames()` overlapping by `context`; returns `(frames - 1) * hop` |
| `numMels()` · `chunkFrames()` · `bodyFrames()` · `sampleRate()` | geometry read from the Engine |

The Engine must output `log_mag` and `phase` (`[1, 513, T]`, in that order);
`exp`, the magnitude clip, `cos`, `sin` and the iSTFT run on the CPU. Chunk
seams are exact for `context >= 27`; only the utterance's two ends see
`pad_value` instead of the network's zero padding.


### Speech synthesis — `bcdl/tasks/tts.h`

```cpp
bcdl::Engine enc("models/matcha_zh_en_enc_l128_int16_nashm.hbm");
bcdl::Engine dec("models/matcha_zh_en_dec_t512_int16_nashm.hbm");
bcdl::Engine voc_eng("models/vocos_16k_t256_int16_nashm.hbm");
bcdl::MatchaAcoustic am(enc, dec, "models/matcha_zh_en_embedding.npy");  // MatchaConfig{}
bcdl::VocosVocoder voc(voc_eng);
bcdl::TextToSpeech tts(bcdl::MatchaFrontend("models/matcha_zh_en_tokens.txt",
                                            "models/matcha_zh_en_lexicon.txt"),
                       am, voc);
bcdl::TtsResult r = tts.synthesize("好的，马上出发。");   // r.pcm, 16 kHz
```

| member | notes |
|---|---|
| `MatchaFrontend::encode(text)` | one id vector per sentence; matches sherpa-onnx's front end |
| `MatchaFrontend::readNumbers(text)` | digits read out as characters (static) |
| `MatchaAcoustic::synthesize(ids, n, timeout_ms)` | `MatchaResult{mel [80, frames], frames, durations, truncated}` |
| `MatchaAcoustic::seed(s)` · `maxTokens()` · `maxFrames()` · `numMels()` · `vocabSize()` | |
| `TextToSpeech::synthesize(text, timeout_ms)` | `TtsResult{pcm, sample_rate, sentences, truncated}`; pauses shortened per `TextToSpeechConfig::silence_scale` (0.2) |
| `scaleSilence(pcm, sample_rate, scale, min_pause_s, level)` | pure; pauses ≥ `min_pause_s` keep `scale` of their length (sherpa-onnx's rule) |

The encoder takes embeddings, not ids: the lookup runs on the CPU from the
`.npy` table (`[vocab, D]`, pre-scaled). Durations are `ceil(exp(logw)) *
length_scale`, regulated on the CPU exactly as Matcha's `generate_path`; the
decoder noise comes from a seeded `std::mt19937`.

---

## JPEG (JPU)

`bcdl/media/jpeg_codec.h` — hardware JPEG on the JPU. Move-only; reuse across a
stream.

```cpp
bcdl::JpegEncoder enc(width, height, /*quality=*/50,
                      HB_VP_IMAGE_FORMAT_NV12);   // width%16==0, height%8==0
std::vector<uint8_t> jpg = enc.encode(src_vpimage);

bcdl::JpegDecoder dec(/*outFormat=*/HB_VP_IMAGE_FORMAT_NV12);
bcdl::VpImage img = dec.decode(jpg.data(), jpg.size());  // owned NV12 VpImage
```

`encode()`/`decode()` copy out of the codec-internal buffer, so results are safe
to keep. **Reuse a `JpegDecoder`** — constructing one per call costs ~5 ms.

## Video (VPU)

`bcdl/media/video_codec.h` — hardware H.264 / H.265 on the VPU.

```cpp
bcdl::VideoEncConfig ec;                 // type, width(%32), height(%8),
ec.type = HB_VP_VIDEO_TYPE_H264;         // bitrate_kbps, framerate, intra_period,
ec.width = 1280; ec.height = 720;        // format
bcdl::VideoEncoder enc(ec);
std::vector<uint8_t> chunk = enc.encode(frame_vpimage);  // may be empty (buffered)

bcdl::VideoDecConfig dc; dc.type = HB_VP_VIDEO_TYPE_H264;  // or _H265; + format, in_buf_size
bcdl::VideoDecoder dec(dc);
bcdl::VpImage out;
if (dec.decode(nal_data, nal_size, out)) { /* frame ready (NV12) */ }
// else the decoder is still buffering (reorder / reference frames)
```

The **decoder** is built on the `media_codec` (`hb_mm_mc_*`) streaming API, which
**decouples input feeding from output draining and honors reorder** — required
for H.265 (a per-AU "decode → immediately get its frame" model times out on
reorder streams). Two ways to drive it:

- `decode(data, size, out) -> bool` — convenience: feed one access unit, wait
  briefly for a frame. Good for low-latency H.264; trailing reorder frames need
  `flush()`.
- **Decoupled** (feed thread + receive thread, as `AsyncVideoDetectionPipeline`
  uses): `feed(data, size)` queues one AU; `receive(out, timeout_ms)` drains a
  frame in display order (`timeout_ms=0` non-blocking); `feedEndOfStream()` then
  `flush(out)` drain the reorder tail.

The **encoder** has the same two forms: `encode(frame)` feeds one NV12 frame and
waits briefly for a packet (the bytes produced by *this* call — possibly none);
for a stream, `feed(frame, pts_us = 0) -> bool` queues a frame without waiting,
`receive(out, timeout_ms = 0) -> bool` drains one packet, and after the last
frame `flush(out)` — call until it returns `false` — signals end of stream and
drains the rest (`feedEndOfStream()` queues the marker alone, idempotent).
Drain after every `feed()`, and when it returns `false` because the input queue
is full, drain and retry rather than treat it as fatal: generating a 1080p frame
is far cheaper than encoding one, so a producer outruns the encoder easily.

> **Feed one access unit, not arbitrary bytes.** The decoder runs in
> `MC_FEEDING_MODE_FRAME_SIZE`: every `feed()`/`decode()` call must carry exactly
> one picture's worth of start-code-prefixed NALs. (`AsyncVideoDetectionPipeline`
> does this reassembly for you — its `submit()` takes arbitrary Annex-B bytes.)
> The old `STREAM_SIZE` mode accepted arbitrary chunks but corrupts the heap from
> inside the codec.

> **Annex-B elementary stream only — an MP4 is not one.** MP4/MOV store NALs in
> AVCC form (4-byte length prefixes, no start codes) with SPS/PPS hidden in the
> `avcC` box, so the AU splitter finds nothing and the decoder silently produces
> zero frames. Demux first, without touching the pixels:
> `ffmpeg -i in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 -`
> (`hevc_mp4toannexb` / `-f hevc` for H.265). `examples/video_det_demo.py`'s
> `load_annexb()` does exactly this; the VPU stays the only decoder.

> **Destroying a decoder mid-stream is safe, but only because the destructor
> drains it.** `hb_mm_mc_stop()` blocks until the codec's `vdec_render` component
> has emptied its output port, and `hb_mm_mc_flush()` blocks the same way — only
> the application can empty it, so a naive `stop()` on a decoder still holding
> decoded frames hangs forever. `~VideoDecoder` hands every pending output buffer
> back first. Anything that drives `hb_mm_mc_*` directly must do the same.

> **H.265 caveat (hierarchical GOP):** a stream with temporal sub-layers (e.g. a
> Hikvision cam with SVC / "H.265+"/smart-codec on) decodes **base temporal layer
> only** (~1/4 of frames) — the `target_dec_temporal_id_plus1` control has no
> effect on the current SoC SDK. Non-hierarchical HEVC and H.264 decode fully.

---

## ByteTracker

`bcdl/tracks/byte_tracker.h` — model-free multi-object tracker (Kalman + two-stage
IoU association). Move-only.

```cpp
struct Track { int track_id; float x1,y1,x2,y2,score; int class_id; };
struct ByteTrackConfig { float track_thresh=0.5, high_thresh=0.6, match_thresh=0.8;
                         int track_buffer=30, frame_rate=30;
                         float proximity_thresh=0.5, appearance_thresh=0.25,
                               ema_alpha=0.95;
                         bcdl::BoostConfig boost; };

bcdl::ByteTracker tracker(cfg);
for (auto& frame : stream) {
  std::vector<bcdl::Detection> dets = /* detector */;
  std::vector<bcdl::Track> tracks = tracker.update(dets);  // per frame
}
tracker.reset();   // on a stream cut
```

Detections must already be in original-image pixels (the detectors do this).

### Appearance (ReID)

`bcdl/tracks/reid.h`. Pass one embedding per detection and the first
association's cost becomes `min(IoU distance, gated cosine distance)` — so
appearance can only **rescue** a match geometry was about to miss, never break
one it already had. An **empty** entry means "no appearance for this detection",
which is how you skip the ReID model on the cheap crops.

```cpp
std::vector<std::vector<float>> embs(dets.size());
for (size_t i = 0; i < dets.size(); ++i) {
  if (dets[i].score < 0.5f) continue;                 // leave it empty
  bcdl::reidPreprocess(bgr, w, h, w * 3, dets[i].x1, dets[i].y1,
                       dets[i].x2, dets[i].y2, 128, 256, reid_cfg, crop);
  reid_engine.setInput(0, crop.data(), crop.size() * sizeof(float));
  reid_engine.infer();
  embs[i] = embedder.postprocess();                   // bcdl::ImageEmbedder
}
std::vector<bcdl::Track> tracks = tracker.update(dets, embs);
```

`reidPreprocess()` does a **squashing** resize to the model's crop size (not a
letterbox — these models are trained on squashed crops), BGR→RGB, ImageNet
normalization, NCHW float32. Read-out reuses `ImageEmbedder`;
`l2Normalize()` / `cosineSimilarity()` are the header-only primitives.

Throws `Error(-1)` if `embeddings.size()` differs from `detections.size()` or if
the embedding width changes mid-stream.

### Camera motion

```cpp
float affine[6] = {1,0,dx, 0,1,dy};   // maps the PREVIOUS frame onto this one
tracker.applyCameraMotion(affine);    // before the next update()
```

Position and size are warped; velocity is not, because velocity describes the
target in the world and a one-frame camera jolt is not the target accelerating.
The caller supplies the transform because this class only ever sees boxes.
For a static camera, skip the call rather than passing an identity.

### BoostConfig

BoostTrack++ additions, **all off by default**, each independently switchable so
its contribution can be measured rather than assumed:
`rich_similarity` (Mahalanobis + shape terms, with `min_iou` guarding the
result), `soft_biou` (grow both boxes by `1 - tracklet confidence`), and
`boost_detections` (DLO/DUO score boosting before the high/low split).
The last is **situational** — it recovers real detections when the detector is
limited by misses and manufactures false tracks when it is not.

## DetectionPipeline

`bcdl/pipeline/detection_pipeline.h` — synchronous, allocation-free-per-frame
BGR→NV12→infer→decode. Needs an NV12-input YOLO `.hbm`.

```cpp
struct PipelineConfig {
  int input_w=0, input_h=0;          // 0 => derive from Engine input[0]
  bcdl::DetectConfig detect;         // thresholds shared by both heads
  int output_index=0;                // kSingleTensor output
  uint8_t pad_value=114;
  bcdl::DetectHead head = bcdl::DetectHead::kAuto;  // kAuto/kSingleTensor/kYoloLtrb
  std::vector<int> ltrb_strides = {8,16,32};
};

bcdl::DetectionPipeline pipe(engine, cfg);
std::vector<bcdl::Detection> dets = pipe.process(bgr, width, height);
const bcdl::LetterboxInfo& lb = pipe.lastLetterbox();
bcdl::DetectHead resolved = pipe.head();

// Per-stage timing accumulated across every process() call (for profiling).
const bcdl::StageProfile& sp = pipe.profile();
// sp.preproc_ms / infer_ms / postproc_ms (summed), sp.frames, sp.totalMs(),
// sp.preprocPerFrame() / inferPerFrame() / postprocPerFrame().
pipe.resetProfile();
```

`DetectHead::kAuto` resolves from the Engine's output signature at construction.
Shared helpers (`resolveDetectionConfig`, `requireNv12InputModel`,
`feedNv12Input`, `preprocBgrToNv12`, `HeadDecoder`) are exposed for custom
pipelines.

`StageProfile` (shared with `AsyncDetectionPipeline`) breaks the per-frame cost
into `preproc` (letterbox BGR→NV12, CPU), `infer` (feed + BPU submit/wait), and
`postproc` (decode + NMS, CPU); the video pipelines also fill `decode_ms` and
`cvt_ms`. In the **sync** pipeline the three run
back-to-back so their sum is the `process()` cost; in the **async** pipeline they
are per-stage *service* times measured on separate threads, so their sum exceeds
wall time and the **slowest** stage bounds throughput.

## AsyncDetectionPipeline

`bcdl/pipeline/async_detection_pipeline.h` — two worker threads overlap CPU
preproc of later frames with BPU infer+decode of earlier ones. Results come back
from `next()` in submission order.

```cpp
bcdl::AsyncDetectionPipeline p(engine, cfg, /*depth=*/3);
std::vector<bcdl::Detection> dets;
int i = 0;
for (auto& f : stream) {
  p.submit(f.bgr, f.w, f.h);          // bytes copied; blocks if full (backpressure)
  if (i++ >= 3) p.next(dets);         // keep pipeline full, drain in order
}
p.finish();                           // signal end of stream (also in dtor)
while (p.next(dets)) { /* last in-flight results */ }
```

- `submit(bgr, w, h) -> bool` — `false` after `finish()` (frame not accepted).
- `next(out) -> bool` — `false` once finished **and** drained.
- `depth` ≥ 2 to overlap; larger tolerates more jitter at the cost of latency.
- `profile() -> StageProfile` — per-stage *service* times (each stage on its own
  thread; the slowest bounds throughput). Read after `finish()` + full drain.

Feeding decoded video frames through a decode thread → `AsyncDetectionPipeline`
gives a full `decode ‖ preproc ‖ infer+NMS` overlap; see
`examples/video_det_demo_async.cc` (yolo26n 1080p: serial 119 → overlapped 234
FPS, decode-bound). For a compressed-bytes-in interface, use
`AsyncVideoDetectionPipeline` (below) — it owns that overlap so callers only pump
bytes.

## AsyncVideoDetectionPipeline

`bcdl/pipeline/async_video_detection_pipeline.h` — the whole compressed-video →
detections path in one C++ object: it segments Annex-B bytes into access units,
VPU-decodes, converts NV12→BGR, and detects, with **four overlapped stages**
(`decode ‖ nv12→bgr ‖ preproc ‖ infer+NMS`). Callers only feed bytes, so a thin
driver (e.g. Python pumping an `ffmpeg -c copy` stream) reaches the C++
decode-bound ceiling instead of being throttled by its own orchestration.

```cpp
bcdl::AsyncVideoDetectionPipeline p(engine, cfg, HB_VP_VIDEO_TYPE_H264, /*depth=*/4);
std::vector<bcdl::Detection> dets;
while (int n = read(ffmpeg_stdout, buf, sizeof buf)) {
  p.submit(buf, n);                    // Annex-B bytes; blocks on backpressure
  while (p.tryNext(dets)) { /* draw / count */ }   // non-blocking drain
}
p.finish();
while (p.next(dets)) { /* drain the last in-flight frames */ }
```

- `submit(data, n) -> bool` — feed a chunk of Annex-B bytes (any size; AUs
  segmented internally). Blocks while full; `false` after `finish()`.
- `next(out) -> bool` — blocking pop in decode order; `false` once finished+drained.
- `tryNext(out) -> bool` — non-blocking pop; drain while feeding.
- `profile() -> StageProfile` — `decode_ms` (VPU decode) and `cvt_ms` (NV12→BGR)
  on top of preproc/infer/postproc. All five are per-thread *service* times, so
  the **slowest** bounds throughput.

Measured: **yolo26n 1080p H.264 → ~441 FPS**, decode-bound (`decode 0.29 |
preproc 1.32 | infer 1.47 | postproc 0.65` ms/f). The pipeline letterboxes the
decoded NV12 straight into the model input on the GDC hardware engine — no BGR
round-trip; `cvt_ms` stays 0. H.265 runs 300/300 frames at 439–451 FPS. Video
decode handles **H.264 and
H.265** (a hierarchical-GOP HEVC stream decodes base temporal layer only).
`nv12ToBgrCpu()` (preproc/letterbox_cpu.h) is still there for callers that want
BGR out of a decoded frame. It takes a `YuvRange`: `kStudioToFull` (default —
what a video decoder produces, and what `cv::cvtColor(COLOR_YUV2BGR_NV12)` does)
or `kAsIs` (full-range in, the bit-exact inverse of `bgrToNv12Cpu()`). Both the
OpenCV SIMD path and the hand-written fallback implement each range identically.

## TrackingPipeline

`bcdl/pipeline/tracking_pipeline.h` — `DetectionPipeline` feeding a `ByteTracker`,
one frame in, tracks out.

```cpp
bcdl::TrackingPipeline pipe(engine, det_cfg /*=PipelineConfig*/,
                                    track_cfg /*=ByteTrackConfig*/);
std::vector<bcdl::Track> tracks = pipe.process(bgr, width, height);
const auto& dets = pipe.lastDetections();   // pre-association, for overlay
pipe.reset();

// With appearance: pass a ReID Engine (which must outlive the pipeline). The
// crop size comes from that model, so switching models is a path change.
bcdl::TrackingPipeline pipe2(engine, reid_engine, det_cfg, track_cfg,
                             bcdl::TrackingReidConfig{});
pipe2.hasReid();          // true
pipe2.lastEmbedCount();   // crops embedded on the last frame
```

`TrackingReidConfig{min_score, max_crops, crop}` are cost knobs: the ReID model
runs **once per qualifying crop**, so on a crowded frame it, not the detector,
sets the frame time.

## StereoPipeline

`bcdl/pipeline/stereo_pipeline.h` — two rectified images → disparity (+ optional
metric depth / validity mask). Pixel normalization is fused into the `.hbm`.

```cpp
enum class StereoFit { kResize, kCrop };   // MUST match offline calibration
struct StereoConfig {
  int input_w=0, input_h=0; bcdl::StereoFit fit = bcdl::StereoFit::kResize;
  bool to_rgb=true; int left_index=0, right_index=1, output_index=0;
  float fx=0, baseline=0;                 // both > 0 => fills StereoResult.depth
  bool valid_mask=false; float disp_min=0, max_disp=192; int left_margin=0;
  bool lr_check=false; float lr_thresh=1.5;
};
struct StereoResult { bcdl::DepthMap disparity; std::vector<float> depth;
                      std::vector<uint8_t> valid; };

bcdl::StereoPipeline pipe(engine, cfg);
bcdl::StereoResult res = pipe.process(left_bgr, right_bgr, width, height);
// res.disparity.data (always), res.depth / res.valid (when enabled)
```

Pure helpers: `packStereoInputCHW(bgr, w, h, out_h, out_w, fit, to_rgb, dst)`,
`disparityToDepth(disp, fx, baseline)`,
`stereoValidMask(disp, disp_min, max_disp, left_margin, disp_right, lr_thresh)`.

---

## Worked example

End-to-end LTRB detection on one BGR image (mirrors `examples/`):

```cpp
#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  try {
    bcdl::Engine engine(argv[1]);            // an NV12-input YOLO26 .hbm

    // Easiest: let the pipeline own preproc + head selection.
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;
    bcdl::DetectionPipeline pipe(engine, cfg);

    // bgr: interleaved HxWx3 uint8 (e.g. from cv::imread, then .data)
    std::vector<bcdl::Detection> dets = pipe.process(bgr, width, height);
    for (const auto& d : dets)
      std::printf("cls=%d score=%.3f [%.1f,%.1f,%.1f,%.1f]\n",
                  d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
  } catch (const bcdl::Error& e) {
    std::fprintf(stderr, "bcdl error %d: %s\n", e.code(), e.what());
    return 1;
  }
}
```

For more, the [`examples/`](../examples/) directory has runnable programs:
`det_demo`, `ocr_demo`, `track_demo`, `video_det_demo` (serial, prints a
per-stage time distribution), `video_det_demo_async` (3-stage overlapped
decode‖preproc‖infer, serial-vs-async compare), `stereo_demo`, `jpeg_roundtrip`,
`video_roundtrip`, `mempool_demo`, and the `*_bench` drivers.
```

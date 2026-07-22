# BCDL C++ API reference

Complete reference for the C++ library (namespace `bcdl`). The public headers
under [`include/bcdl/`](../include/bcdl/) are the source of truth — this document
summarizes them with usage. For the Python bindings see
[`API.md`](API.md); the names map 1:1 (Python snake_case ⇄ C++ camelCase).

- [Linking](#linking) · [Conventions](#conventions) · [Worked example](#worked-example)
- Core: [Error handling](#error-handling) · [SysMem](#sysmem) · [Task](#task) ·
  [MemPool](#mempool)
- Backend: [Engine](#engine)
- Preprocessing: [Geometry & letterbox](#geometry--letterbox) · [VpImage](#vpimage)
- Tasks: [Detection](#detection) · [Classification](#classification) ·
  [Pose](#pose) · [Instance segmentation](#instance-segmentation) ·
  [Oriented boxes (OBB)](#oriented-boxes-obb) ·
  [Semantic segmentation](#semantic-segmentation) · [Depth](#depth) · [OCR](#ocr)
- Media: [JPEG (JPU)](#jpeg-jpu) · [Video (VPU)](#video-vpu)
- Tracking & pipelines: [ByteTracker](#bytetracker) · [DetectionPipeline](#detectionpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline) ·
  [TrackingPipeline](#trackingpipeline) · [StereoPipeline](#stereopipeline)

## Linking

The umbrella header pulls in everything:

```cpp
#include "bcdl/bcdl.h"     // or include just the sub-headers you use
```

With CMake, `find_package(bcdl)` exposes the `bcdl::bcdl` target (it transitively
brings the headers and the hobot SDK link libs):

```cmake
find_package(bcdl CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE bcdl::bcdl)
```

See the [README](../README.md#build-from-source) for the conda packages
(`libbcdl` + `hobot-dnn` + `hobot-media`) and the on-board build. Everything here
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
| `numInputs()` / `numOutputs()` | Tensor counts. |
| `inputShape(i)` / `outputShape(i)` | `std::vector<int>` shape. |
| `inputType(i)` / `outputType(i)` | `hbDNN` tensor type enum. |
| `inputProperties(i)` / `outputProperties(i)` | Full `hbDNNTensorProperties` (dtype, quant, stride). |
| `inputBytes(i)` / `outputBytes(i)` | Allocated device-buffer sizes. |
| `inputPackedBytes(i)` | Size of input `i` as a contiguous row-major array — smaller than `inputBytes(i)` when the model pads a dimension. |
| `inputStride(i)` | Resolved byte strides of input `i`, innermost last. |
| `outputStride(i)` | Resolved byte strides of output `i`. Outputs pad too; `outputAsFloat()` handles it, direct readers must not reshape flat. |
| `setInput(i, data, bytes)` | Copy host bytes into input `i` and flush. `bytes` must be `inputPackedBytes(i)` (scattered into the device layout) or `inputBytes(i)` (taken as-is); anything else throws. |
| `inputData(i)` | Read-only view of input `i`'s device buffer, in the device layout. |
| `infer(timeout_ms=0)` | Run one inference (blocks; invalidates output caches). |
| `outputData(i)` | Pointer to output `i` after `infer()` (honor the stride). |
| `static elemSize(tensor_type)` | Byte size of an `hbDNN` tensor element type. |

> Prefer a **task class** (`Detector`, `Segmenter`, …) over reading
> `outputData()` yourself — it dequantizes (F16/int8/int16) and de-strides for
> you. Drop to the raw output only for a custom decoder.

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
```

`bgrToNv12Cpu` uses BT.601 **full-range** (matches cv2's `COLOR_BGR2YUV_I420`);
verify against your model's calibrated range.

**hbVP path** (`bcdl/preproc/letterbox.h`, DSP required): `letterbox(dst, src,
padValue, interp)`, `cvtColor(dst, src)`, `resizeExact(dst, src, interp)`.

**GDC hardware path** (`bcdl/preproc/gdc_letterbox.h`, `gdc_remap.h`; only
when built with `BCDL_HAVE_GDC`) — fixed transforms on the VPS GDC engine,
NV12 in/out, CPU idle during the op. Semantics + reverse-engineered
CUSTOM-grid notes: [docs/GDC.md](GDC.md).

```cpp
// letterbox from an offline-generated AFFINE warp bin
bcdl::GdcLetterbox lbg(binPath, inW, inH, outW, outH, /*pad=*/114);
lbg.run(srcNv12, dstNv12);

// arbitrary fixed dense remap (cv2.remap semantics), LUT generated at runtime
bcdl::GdcRemap remap(mapX, mapY, inW, inH, outW, outH, /*gridStep=*/16);
remap.run(srcNv12, dstNv12);   // 2448x2048: ~6.3 ms wall, ~1 ms CPU
```

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

## Classification

`bcdl/tasks/classification.h`

```cpp
struct ClsConfig { int top_k = 5; bool apply_softmax = true; };
struct ClsResult { int class_id; float score; };

bcdl::Classifier clf(engine, cfg, /*output_index=*/0);
std::vector<bcdl::ClsResult> top = clf.postprocess();   // after infer()
// pure: decodeClassification(logits, num_classes, cfg)
```

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

bcdl::SuperResolver sr(engine, cfg);
auto big = sr.upscale(bgr, w, h, stride);        // sr.scale() times larger
// pure: planTiles(w, h, tile_w, tile_h, overlap), tileWeight(i, len, ramp)
```

`scale()` and `tile()` are read from the model's shapes, never configured. The
blend accumulates `w * pixel` and `w` and divides, so image borders need no
special case. Note that the compiled `.hbm` scales with tile AREA — the same
network is 148 MB at a 256 tile and 37 MB at 128, for identical per-pixel
throughput.

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
                           bool compute_masks=true; };

bcdl::InstanceSegmenter seg(engine, cfg, /*output_base=*/0);
auto masks = seg.postprocess(lb, orig_w, orig_h);
// pure: decodeInstanceSeg(cls, box, mc, grid_hw, num_classes, num_coef,
//                         proto, proto_h, proto_w, proto_c, cfg, lb, orig_w, orig_h)
```

Set `compute_masks=false` to get boxes/scores only and skip mask assembly.

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

## OCR

`bcdl/tasks/ocr.h` — three independent stages (det / cls / rec); the application
composes them (crop each detected box, optional 180° flip, recognize). Each stage
has a pure decoder + an Engine-bound wrapper.

```cpp
// Recognition (CRNN + CTC):
std::vector<std::string> dict = bcdl::loadCharDict("ppocrv5_dict.txt"); // blank @ 0
struct RecResult { std::string text; float score; };
bcdl::TextRecognizer rec(engine, "ppocrv5_dict.txt", /*out_idx=*/0);
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

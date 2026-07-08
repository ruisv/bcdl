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
| `setInput(i, data, bytes)` | Copy host bytes into input `i` and flush. |
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

bcdl::VideoDecConfig dc; dc.type = HB_VP_VIDEO_TYPE_H264;  // + format, in_buf_size
bcdl::VideoDecoder dec(dc);
bcdl::VpImage out;
if (dec.decode(nal_data, nal_size, out)) { /* frame ready (NV12) */ }
// else the decoder is still buffering reference frames
```

---

## ByteTracker

`bcdl/tracks/byte_tracker.h` — model-free multi-object tracker (Kalman + two-stage
IoU association). Move-only.

```cpp
struct Track { int track_id; float x1,y1,x2,y2,score; int class_id; };
struct ByteTrackConfig { float track_thresh=0.5, high_thresh=0.6, match_thresh=0.8;
                         int track_buffer=30, frame_rate=30; };

bcdl::ByteTracker tracker(cfg);
for (auto& frame : stream) {
  std::vector<bcdl::Detection> dets = /* detector */;
  std::vector<bcdl::Track> tracks = tracker.update(dets);  // per frame
}
tracker.reset();   // on a stream cut
```

Detections must already be in original-image pixels (the detectors do this).

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
```

`DetectHead::kAuto` resolves from the Engine's output signature at construction.
Shared helpers (`resolveDetectionConfig`, `requireNv12InputModel`,
`feedNv12Input`, `preprocBgrToNv12`, `HeadDecoder`) are exposed for custom
pipelines.

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

## TrackingPipeline

`bcdl/pipeline/tracking_pipeline.h` — `DetectionPipeline` feeding a `ByteTracker`,
one frame in, tracks out.

```cpp
bcdl::TrackingPipeline pipe(engine, det_cfg /*=PipelineConfig*/,
                                    track_cfg /*=ByteTrackConfig*/);
std::vector<bcdl::Track> tracks = pipe.process(bgr, width, height);
const auto& dets = pipe.lastDetections();   // pre-association, for overlay
pipe.reset();
```

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
`det_demo`, `ocr_demo`, `track_demo`, `video_det_demo`, `stereo_demo`,
`jpeg_roundtrip`, `video_roundtrip`, `mempool_demo`, and the `*_bench` drivers.
```

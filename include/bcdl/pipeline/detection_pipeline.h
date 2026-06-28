#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bcdl/preproc/vp_image.h"
#include "bcdl/tasks/detection.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Which detection-head decoder the pipeline drives.
enum class DetectHead {
  /// Pick automatically from the Engine's output signature at construction:
  /// `2 * ltrb_strides.size()` outputs with 4-channel box tensors => kYoloLtrb,
  /// otherwise kSingleTensor.
  kAuto,
  /// One fused tensor [1,4+nc,N] / [1,N,4+nc] (YOLOv5/v8/v11). Uses `detect`.
  kSingleTensor,
  /// Anchor-free LTRB multi-scale head: paired (cls,box) outputs per stride
  /// (YOLO26 / standard RDK YOLO export). Uses `ltrb_strides` + the thresholds
  /// in `detect` (num_classes / conf_thresh / iou_thresh / max_dets).
  kYoloLtrb,
};

/// Tunables for a streaming DetectionPipeline.
///
/// `input_w`/`input_h` are the model's letterbox canvas (e.g. 640x640). Leave
/// them <= 0 to have the pipeline derive them from the Engine's input[0] shape
/// (NHWC NV12 Y-plane or NCHW) at construction. `detect` carries the post-
/// processing config; its `input_w`/`input_h` are synced to the resolved canvas.
struct PipelineConfig {
  int input_w = 0;        ///< model input width  (0 => derive from Engine)
  int input_h = 0;        ///< model input height (0 => derive from Engine)
  DetectConfig detect;    ///< decode / NMS config (thresholds shared by both heads)
  int output_index = 0;   ///< kSingleTensor: which output holds the head
  uint8_t pad_value = 114;///< letterbox border color (YOLO default)
  DetectHead head = DetectHead::kAuto;     ///< decoder family (see DetectHead)
  std::vector<int> ltrb_strides = {8, 16, 32};  ///< kYoloLtrb: stride per scale
};

// ---------------------------------------------------------------------------
// Shared pipeline core — reused by the synchronous DetectionPipeline and the
// threaded AsyncDetectionPipeline so the preproc / feed / decode logic lives in
// exactly one place.
// ---------------------------------------------------------------------------

/// Fill in cfg.input_w/input_h from the Engine's input[0] when unset, sync the
/// detect canvas to them, and resolve DetectHead::kAuto to a concrete head from
/// the Engine's output signature. Pure; returns the resolved config.
PipelineConfig resolveDetectionConfig(Engine& engine, PipelineConfig cfg);

/// Throw bcdl::Error if the model's input 0 is float (F32/F16): the NV12
/// pipelines feed uint8 and such a model would silently infer on garbage.
void requireNv12InputModel(Engine& engine);

/// Feed a CPU-written NV12 VpImage to the Engine's input(s): the whole Y+UV
/// block for a 1-input fused-NV12 model, or Y->input0 / UV->input1 for the
/// standard 2-input RDK split layout. (Engine::setInput cleans the cache.)
void feedNv12Input(Engine& engine, const VpImage& nv12);

/// Letterbox a BGR frame into `nv12Dst`, reusing the caller-owned `src` and
/// `lbBgr` scratch buffers (re-allocating `src` only when (width,height) change).
/// Returns the letterbox geometry. No allocation in steady state.
LetterboxInfo preprocBgrToNv12(VpImage& src, VpImage& lbBgr, VpImage& nv12Dst,
                               const uint8_t* bgr, int width, int height,
                               uint8_t pad_value);

/// Owns the resolved detector (single-tensor or LTRB) for an already-resolved
/// PipelineConfig and dispatches postprocess to it. Holds an Engine&.
class HeadDecoder {
 public:
  HeadDecoder(Engine& engine, const PipelineConfig& resolved_cfg);
  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

 private:
  std::unique_ptr<Detector> single_;
  std::unique_ptr<YoloLtrbDetector> ltrb_;
};

/// Synchronous, allocation-free-per-frame streaming object detector.
///
/// Construct once with an Engine, then call process() per frame. ALL scratch
/// buffers are allocated up front and reused, so steady-state process() does no
/// per-frame heap allocation for image buffers — this is the M4 value (see
/// docs/PLAN.md: "camera stream ~90% of single-frame FPS"). Held buffers:
///   - `src_`    : a BGR wrapper sized to the *incoming* frame. Re-allocated
///                 only when the caller's (width,height) changes; otherwise the
///                 caller's rows are memcpy'd into the existing buffer.
///   - `lb_bgr_` : the letterboxed BGR canvas at (input_w,input_h). Fixed,
///                 allocated once. (We keep this ourselves instead of calling
///                 letterboxToNv12Cpu(), which would allocate a temporary BGR
///                 canvas on every frame.)
///   - `nv12_`   : the NV12 letterbox target at (input_w,input_h). Fixed,
///                 allocated once; fed to the Engine input(s).
///
/// Input feeding adapts to the model: a single combined-NV12 input gets the
/// whole Y+UV block; a two-input model (the standard RDK YOLO Y-plane +
/// half-res interleaved-UV layout) gets the Y plane in input 0 and the UV plane
/// in input 1 — both views of the same persistent nv12_ buffer, no extra copy.
///
/// Per-frame data flow (all synchronous; infer() blocks via WaitTaskDone):
///   memcpy BGR rows -> src_                    (CPU)
///   letterboxCpu(lb_bgr_, src_)                (CPU, cleanCache on lb_bgr_)
///   bgrToNv12Cpu(nv12_, lb_bgr_)              (CPU, cleanCache on nv12_)
///   engine.setInput(Y[,UV])                    (memcpy + cleanCache, in Engine)
///   engine.infer()                            (submit + wait + invalidate)
///   detector.postprocess(last_lb_)            (dequant + decode + NMS)
///
/// Cache discipline is fully owned by the layers we call (letterbox_cpu cleans
/// the NV12 dst; Engine cleans inputs and invalidates outputs) — this class does
/// not touch caches directly.
///
/// FUTURE WORK (not this round): an async variant chaining hbUCPSetTaskDoneCb so
/// frame N+1's preproc overlaps frame N's BPU infer. M4 here stays synchronous.
class DetectionPipeline {
 public:
  /// Build the pipeline. `cfg.input_w`/`input_h` <= 0 are resolved from the
  /// Engine's input[0] shape. The NV12 / letterbox scratch buffers and the
  /// detector are allocated here, once.
  DetectionPipeline(Engine& engine, PipelineConfig cfg);

  DetectionPipeline(const DetectionPipeline&) = delete;
  DetectionPipeline& operator=(const DetectionPipeline&) = delete;

  /// Run one frame end-to-end and return detections in ORIGINAL-image pixels.
  ///
  /// `bgr` points to an interleaved HxWx3 uint8 BGR image, row-contiguous
  /// (row stride == width*3). `width`/`height` are that image's dimensions.
  /// No image buffers are heap-allocated unless (width,height) differs from the
  /// previous call (which re-sizes the source wrapper only).
  std::vector<Detection> process(const uint8_t* bgr, int width, int height);

  const PipelineConfig& config() const noexcept { return cfg_; }
  /// Letterbox geometry of the most recent process() call.
  const LetterboxInfo& lastLetterbox() const noexcept { return last_lb_; }
  /// Resolved decoder family (kAuto is replaced by the concrete choice).
  DetectHead head() const noexcept { return cfg_.head; }

 private:
  Engine& engine_;
  PipelineConfig cfg_;
  VpImage src_;      ///< reused BGR source wrapper (last incoming frame size)
  VpImage lb_bgr_;   ///< reused letterboxed BGR canvas (input_w x input_h)
  VpImage nv12_;     ///< reused NV12 letterbox target (input_w x input_h)
  HeadDecoder decoder_;
  LetterboxInfo last_lb_;
};

}  // namespace bcdl

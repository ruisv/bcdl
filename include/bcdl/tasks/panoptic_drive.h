#pragma once

#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"
#include "bcdl/tasks/detection.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Anchor-based YOLOv5 detection head (panoptic driving models)
// ===========================================================================
//
// The rest of this library decodes ANCHOR-FREE heads (YOLO26/v8 LTRB, DFL). A
// panoptic driving model built on the YOLOv5 backbone uses the older
// ANCHOR-BASED formulation instead, which needs its own decode: each grid cell
// carries `na` predictions, one per prior box, and the box is expressed as an
// offset from the cell plus a scale factor on that prior.
//
// WHY THIS IS A CPU DECODE. The published ONNX export bakes the decode into the
// graph and emits a ready-made [1, N, 5+nc] tensor — tempting, but that decode
// chain is assembled out of ScatterND writes, and it does NOT survive
// compilation for this BPU: the compiled model emits a tensor whose objectness
// and class columns are simply never written (only two of every eight floats
// carry data, all of them coordinates). Nothing errors — the numbers are just
// missing, so a detector reading it finds zero objects at any threshold. The
// graph therefore has to be cut before the decode, exporting the three raw head
// convolutions, and the arithmetic below has to run on the CPU. It is cheap:
// three small tensors, no dependency on the number of detections.
//
// DECODE (per scale, matching the reference implementation exactly):
//   raw is [1, na*(5+nc), H, W] -> viewed as [na, H, W, 5+nc]
//   y   = sigmoid(raw)                        (ALL attributes, including xy/wh)
//   x   = (y.x * 2 - 0.5 + gx) * stride       gx,gy = cell indices
//   y   = (y.y * 2 - 0.5 + gy) * stride
//   w   = (y.w * 2)^2 * anchor_w              anchor sizes are in PIXELS
//   h   = (y.h * 2)^2 * anchor_h
//   score = y.obj * max(y.class)              class_id = argmax
// The *2-0.5 and (*2)^2 forms are the reference's; they widen the reachable
// offset/scale range versus the original exponential formulation. Boxes come out
// as (cx,cy,w,h) in model-input pixels, are converted to corners, mapped back
// through the letterbox, and run through the shared per-class NMS.
//
// Verified against the published export's own in-graph decode on real frames:
// max absolute difference 1.8e-4 over all 25200 candidates, identical detection
// count at threshold.

/// One anchor prior, in MODEL-INPUT pixels (not normalized, not stride units).
struct Anchor {
  float w;
  float h;
};

/// Post-processing parameters for an anchor-based multi-scale head.
struct AnchorDetectConfig {
  int num_classes = 1;
  float conf_thresh = 0.35f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the raw outputs (typically 8/16/32).
  std::vector<int> strides = {8, 16, 32};
  /// Priors per scale; every scale must carry the same count (`na`). Sizes are
  /// in model-input pixels.
  std::vector<std::vector<Anchor>> anchors = {
      {{3, 9}, {5, 11}, {4, 20}},
      {{7, 18}, {6, 39}, {12, 31}},
      {{19, 50}, {38, 81}, {68, 157}},
  };
};

/// Decode parallel per-scale raw head tensors into final Detections.
///
/// `raw[i]`     : row-major [na*(5+nc), H_i, W_i] floats for scale i — the head
///                convolution's output, still channels-first and UNACTIVATED.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `cfg`        : strides + anchors (one entry per scale) + thresholds.
/// `lb`         : letterbox geometry; boxes are mapped back to original-image
///                pixels via lb.invX/invY and clamped to the source extent.
///
/// Candidates from every scale are pooled, then thresholded, then run through a
/// single per-class NMS truncated to `max_dets`. `raw`, `grid_hw`,
/// `cfg.strides` and `cfg.anchors` must all share the same length, and every
/// scale must declare the same number of anchors, else Error(-1) is thrown.
std::vector<Detection> decodeYoloV5Anchor(
    const std::vector<const float*>& raw,
    const std::vector<std::pair<int, int>>& grid_hw,
    const AnchorDetectConfig& cfg, const LetterboxInfo& lb);

/// Engine-bound anchor-based detector. Reads `strides.size()` consecutive raw
/// head outputs starting at `output_base`, dequantizing + de-striding each into
/// a contiguous float buffer, then runs decodeYoloV5Anchor().
///
/// Each scale's (H,W) is taken from its own tensor shape, and `na` from
/// cfg.anchors — the channel count is checked against na*(5+num_classes) so a
/// mismatched class count fails loudly instead of silently misreading columns.
class AnchorDetector {
 public:
  AnchorDetector(Engine& engine, AnchorDetectConfig cfg = {}, int output_base = 0);

  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

  const AnchorDetectConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  AnchorDetectConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

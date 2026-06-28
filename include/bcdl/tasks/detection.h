#pragma once

#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One detected object, expressed in ORIGINAL-image pixel coordinates
/// (i.e. already un-letterboxed back from the model-input canvas).
struct Detection {
  float x1;        ///< top-left x
  float y1;        ///< top-left y
  float x2;        ///< bottom-right x
  float y2;        ///< bottom-right y
  float score;     ///< confidence in [0,1]
  int class_id;    ///< argmax class index
};

/// Raw-tensor memory layout / decode family.
enum class DecodeLayout {
  /// Anchor-free YOLOv8/v11 head. Output is [1, 4+nc, N] (channels_first) or
  /// [1, N, 4+nc] (channels_last). Each candidate is (cx,cy,w,h) in model-input
  /// pixels followed by `nc` class scores (already activated, or logits when
  /// `apply_sigmoid` is set). class_id = argmax, score = max.
  kYoloV8,
  /// YOLOv5-style head with objectness. Output is [1, N, 5+nc] (channels_last)
  /// or [1, 5+nc, N] (channels_first): (cx,cy,w,h, obj, nc class scores). Final
  /// score = obj * max(class). When `apply_sigmoid` is set, obj and class are
  /// sigmoid-activated first.
  kYoloV5,
};

/// Post-processing parameters for a single detection head.
struct DetectConfig {
  int input_w = 640;        ///< model input width  (letterbox canvas)
  int input_h = 640;        ///< model input height (letterbox canvas)
  int num_classes = 80;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  DecodeLayout layout = DecodeLayout::kYoloV8;
  /// kYoloV8/kYoloV5: true => [1, attrs, N], false => [1, N, attrs].
  bool channels_first = true;
  /// Apply sigmoid to class (and objectness) scores before thresholding.
  /// Set when the exported head emits logits rather than probabilities.
  bool apply_sigmoid = false;
};

/// Intersection-over-union of two axis-aligned boxes.
float iou(const Detection& a, const Detection& b);

/// Classic greedy per-class NMS. Sorts `dets` by score descending, suppresses
/// any lower-scoring box of the SAME class whose IoU with a kept box exceeds
/// `iou_thresh`, and returns the kept indices (into `dets`) truncated to
/// `max_dets`.
std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh, int max_dets);

/// Decode a contiguous (row-major) float output tensor into final Detections.
///
/// `data`  : pointer to product(shape) floats in logical row-major order.
/// `shape` : tensor shape, e.g. {1, 4+nc, N} or {1, N, 4+nc}.
/// `cfg`   : decode/threshold/NMS parameters.
/// `lb`    : letterbox geometry used at preprocess time; boxes are mapped back
///           to original-image pixels via lb.invX/invY and clamped to the source
///           extent. Runs threshold -> per-class NMS -> max_dets truncation.
std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb);

/// Convenience wrapper binding an Engine output to a DetectConfig.
///
/// The caller is responsible for preprocessing + setInput + infer on the Engine
/// (the preproc layer / Python does that). `postprocess()` reads the selected
/// output, converts it to float (F32 directly; F16 via half->float; SCALE-
/// quantized S8/S16/S32 via the per-tensor/per-channel dequant in the tensor
/// properties), honors the device byte strides, and runs decode().
class Detector {
 public:
  Detector(Engine& engine, DetectConfig cfg, int output_index = 0);

  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

  const DetectConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DetectConfig cfg_;
  int out_idx_;
};

// ===========================================================================
// Anchor-free LTRB multi-scale head (YOLO26 / standard RDK YOLO export)
// ===========================================================================
//
// This is the deployed RDK YOLO layout: the head emits a SEPARATE pair of
// outputs per feature scale rather than one fused [1,4+nc,N] tensor. For each
// stride s (default {8,16,32}, grids {80,40,20} at 640) there are two outputs:
//   - cls : [1, H, W, nc]  per-cell class logits (sigmoid-activated here)
//   - box : [1, H, W, 4]   per-cell LTRB distances (left,top,right,bottom)
// Boxes are anchor-free, decoded relative to the cell-center grid:
//   cx = (gx + 0.5),  cy = (gy + 0.5)            (grid units)
//   x1 = (cx - left)  * s,   y1 = (cy - top)    * s
//   x2 = (cx + right) * s,   y2 = (cy + bottom) * s   (model-input pixels)
// score = sigmoid(max class logit); class_id = argmax; per-class NMS.
//
// Matches rdk_model_zoo/.../yolo26_det.py (LTRB, no DFL, no objectness).

/// Post-processing parameters for the LTRB multi-scale head.
struct YoloLtrbConfig {
  int num_classes = 80;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the (cls,box) output pairs. The
  /// default {8,16,32} is the standard P3/P4/P5 head; grid sizes are taken from
  /// each output's own shape, so only the stride is needed here.
  std::vector<int> strides = {8, 16, 32};
  /// DFL (Distribution Focal Loss) box head: 0 => plain LTRB (box tensor has 4
  /// channels, distances read directly). >0 => the box tensor has `4*reg_max`
  /// channels (e.g. 64 for reg_max=16, the YOLOv8/v10/v11 head); each side's
  /// `reg_max` raw logits are softmaxed and reduced by Σ b·softmax(b) into one
  /// LTRB distance. YoloLtrbDetector auto-detects this from the box channel count.
  int reg_max = 0;
};

/// Decode parallel per-scale float cls/box buffers into final Detections.
///
/// `cls[i]`     : row-major [H_i, W_i, nc] class logits for scale i.
/// `box[i]`     : row-major [H_i, W_i, 4] LTRB distances (cfg.reg_max==0), or
///                [H_i, W_i, 4*reg_max] DFL logits (cfg.reg_max>0).
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `cfg`        : strides (one per scale) + thresholds + NMS + reg_max (DFL).
/// `lb`         : letterbox geometry; boxes are mapped back to original-image
///                pixels (lb.invX/invY) and clamped. Runs threshold ->
///                per-class NMS -> max_dets truncation across ALL scales.
/// The three input vectors plus cfg.strides must share the same length.
std::vector<Detection> decodeYoloLtrb(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<std::pair<int, int>>& grid_hw, const YoloLtrbConfig& cfg,
    const LetterboxInfo& lb);

/// Engine-bound LTRB multi-scale detector. Reads `2 * strides.size()` outputs as
/// consecutive (cls, box) pairs starting at `output_base` (i.e. cls = output
/// 2*i + output_base, box = output 2*i + 1 + output_base), dequantizing + de-
/// striding each into a contiguous float buffer, then runs decodeYoloLtrb().
class YoloLtrbDetector {
 public:
  YoloLtrbDetector(Engine& engine, YoloLtrbConfig cfg, int output_base = 0);

  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

  const YoloLtrbConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  YoloLtrbConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

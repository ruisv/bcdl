#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One instance-segmentation result. Box coordinates are in ORIGINAL-image
/// pixels (already un-letterboxed back from the model-input canvas, same as
/// bcdl::Detection). The mask is a full-frame binary map at original-image
/// resolution (row-major, `mask_h` rows of `mask_w`, each element 0 or 1).
struct InstanceMask {
  float x1;       ///< top-left x  (original-image pixels)
  float y1;       ///< top-left y
  float x2;       ///< bottom-right x
  float y2;       ///< bottom-right y
  float score;    ///< confidence in [0,1]
  int class_id;   ///< argmax class index
  int mask_w;     ///< == orig_w (set even when masks are not computed)
  int mask_h;     ///< == orig_h
  std::vector<uint8_t> mask;  ///< row-major mask_h*mask_w, 0/1. Empty when
                              ///< InstanceSegConfig.compute_masks == false.
};

/// Post-processing parameters for the YOLO26-style instance-seg head.
///
/// The head emits, per feature scale s (strides {8,16,32} => grids {80,40,20}
/// at 640), three outputs in the order (cls, box, mc):
///   - cls : [1, H, W, nc]  per-cell class logits (sigmoid-activated here)
///   - box : [1, H, W, 4]   per-cell LTRB distances (left,top,right,bottom)
///   - mc  : [1, H, W, np]  per-cell mask coefficients (np == proto channels)
/// plus a single prototype tensor proto : [1, mH, mW, np] (float, NHWC). The box
/// decode is IDENTICAL to the LTRB detector (see decodeYoloLtrb). Each instance
/// mask = sigmoid(mc[np] · proto) -> mH×mW, resized + cropped + un-letterboxed
/// to the original image (process_mask flow).
struct InstanceSegConfig {
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 100;
  /// One stride per scale, ordered to match the (cls,box,mc) output triplets.
  /// Grid sizes (H,W) and class/coef counts are taken from each output's own
  /// shape, so only the stride is needed here.
  std::vector<int> strides = {8, 16, 32};
  /// Prototype output index, RELATIVE to output_base (so the absolute index is
  /// output_base + proto_index). Default 9 == last of the 10-tensor seg head.
  int proto_index = 9;
  /// When false, only boxes/scores/classes are produced and the (relatively
  /// expensive) per-instance mask assembly is skipped (InstanceMask.mask empty).
  bool compute_masks = true;
};

/// Decode parallel per-scale float cls/box/mc buffers + a prototype tensor into
/// instance masks (Engine-free; the numpy entry point that mirrors
/// InstanceSegmenter::postprocess after dequant/de-stride).
///
/// `cls[i]`     : row-major [H_i, W_i, num_classes] class logits.
/// `box[i]`     : row-major [H_i, W_i, 4]           LTRB distances.
/// `mc[i]`      : row-major [H_i, W_i, num_coef]    mask coefficients.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `num_classes`/`num_coef` : the trailing dims of cls / mc.
/// `proto`      : row-major [proto_h, proto_w, proto_c] NHWC prototype;
///                proto_c must equal num_coef when cfg.compute_masks is true.
/// `cfg`/`lb`/`orig_w`/`orig_h` : as in InstanceSegmenter::postprocess.
/// Runs threshold -> per-class NMS (on model-input boxes) -> per-instance mask
/// assembly (process_mask), returning InstanceMasks in original-image pixels.
/// cls/box/mc/grid_hw/cfg.strides must share the same length, else Error(-1).
std::vector<InstanceMask> decodeInstanceSeg(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& mc,
    const std::vector<std::pair<int, int>>& grid_hw, int num_classes, int num_coef,
    const float* proto, int proto_h, int proto_w, int proto_c,
    const InstanceSegConfig& cfg, const LetterboxInfo& lb, int orig_w, int orig_h);

/// Engine-bound instance segmenter. Reads `3 * strides.size()` outputs as
/// consecutive (cls, box, mc) triplets starting at `output_base`
/// (cls = output_base + 3*s, box = +1, mc = +2) plus the prototype tensor at
/// output_base + proto_index, dequantizing + de-striding each into contiguous
/// float buffers, then runs the seg decode + NMS + mask assembly.
class InstanceSegmenter {
 public:
  InstanceSegmenter(Engine& engine, InstanceSegConfig cfg = {}, int output_base = 0);

  /// Decode -> per-class NMS (on model-input boxes) -> per-instance mask
  /// assembly. `lb` is the letterbox geometry used at preprocess time and
  /// (orig_w, orig_h) the original image size; both are needed to map masks and
  /// boxes back to original-image pixels.
  std::vector<InstanceMask> postprocess(const LetterboxInfo& lb, int orig_w, int orig_h) const;

  const InstanceSegConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  InstanceSegConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

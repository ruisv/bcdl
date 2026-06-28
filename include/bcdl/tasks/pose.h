#pragma once

#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One keypoint, in ORIGINAL-image pixel coordinates (already un-letterboxed).
struct Keypoint {
  float x;      ///< x in original-image pixels
  float y;      ///< y in original-image pixels
  float score;  ///< visibility/confidence in [0,1] (sigmoid of the raw logit)
};

/// One detected pose, expressed in ORIGINAL-image pixel coordinates (i.e.
/// already un-letterboxed back from the model-input canvas). The bounding box
/// is the person box; `keypoints` holds `num_keypoints` skeleton joints.
struct PoseDetection {
  float x1;                         ///< top-left x
  float y1;                         ///< top-left y
  float x2;                         ///< bottom-right x
  float y2;                         ///< bottom-right y
  float score;                      ///< person confidence in [0,1]
  int class_id;                     ///< always 0 (single "person" class)
  std::vector<Keypoint> keypoints;  ///< num_keypoints joints (typically 17)
};

// ===========================================================================
// Anchor-free LTRB multi-scale pose head (YOLO26-pose / RDK pose export)
// ===========================================================================
//
// The head emits THREE outputs per feature scale (9 total for 3 scales). For
// each stride s (default {8,16,32}, grids {80,40,20} at 640) there are:
//   - cls : [1, H, W, 1]              per-cell person logit (single class)
//   - box : [1, H, W, 4]              per-cell LTRB distances (left,top,right,bottom)
//   - kpt : [1, H, W, num_keypoints*3] per-cell keypoints, each (x_raw,y_raw,score)
//
// Box decode is IDENTICAL to the detection LTRB head, anchor-free about the
// cell-center grid (cx = gx+0.5, cy = gy+0.5, in grid units):
//   x1 = (cx - left)  * s,   y1 = (cy - top)    * s
//   x2 = (cx + right) * s,   y2 = (cy + bottom) * s   (model-input pixels)
// score = sigmoid(cls logit); class_id = 0.
//
// Keypoint decode differs from the box: it ADDS the (already centered) grid
// rather than subtracting a distance —
//   kx = (kpt_raw_x + gx + 0.5) * s,   ky = (kpt_raw_y + gy + 0.5) * s
//   kscore = sigmoid(kpt_raw_score)
// Boxes and keypoint xy are then projected to original-image pixels via
// lb.invX/invY and clamped to the source extent. Candidates from all scales are
// pooled and a single (box) NMS is run.
//
// Matches rdk_model_zoo/.../postprocess.py decode_pose_layer +
// scale_keypoints_to_original_image.

/// Post-processing parameters for the LTRB multi-scale pose head.
struct PoseConfig {
  /// Number of skeleton joints per pose (17 for COCO). The authoritative value
  /// is derived from the kpt tensor's last dim / 3 by PoseEstimator; this is the
  /// fallback used by the raw decodePose() entry point.
  int num_keypoints = 17;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.65f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the (cls,box,kpt) output triples.
  /// Grid sizes are taken from each output's own shape, so only stride is needed.
  std::vector<int> strides = {8, 16, 32};
};

/// Decode parallel per-scale float cls/box/kpt buffers into final PoseDetections.
///
/// `cls[i]`     : row-major [H_i, W_i, 1]                 person logits for scale i.
/// `box[i]`     : row-major [H_i, W_i, 4]                 LTRB distances for scale i.
/// `kpt[i]`     : row-major [H_i, W_i, num_keypoints*3]   keypoints for scale i.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `cfg`        : strides (one per scale) + num_keypoints + thresholds + NMS.
/// `lb`         : letterbox geometry; boxes and keypoint xy are mapped back to
///                original-image pixels (lb.invX/invY) and clamped. Runs
///                threshold -> box NMS -> max_dets truncation across ALL scales.
/// The three input vectors plus cfg.strides must share the same length, else
/// Error(-1) is thrown.
std::vector<PoseDetection> decodePose(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& kpt,
    const std::vector<std::pair<int, int>>& grid_hw, const PoseConfig& cfg,
    const LetterboxInfo& lb);

/// Engine-bound LTRB multi-scale pose estimator. Reads `3 * strides.size()`
/// outputs as consecutive (cls, box, kpt) triples starting at `output_base`
/// (cls = output_base + 3*i, box = +1, kpt = +2), dequantizing + de-striding each
/// into a contiguous float buffer, then runs decodePose().
///
/// num_keypoints is taken from the kpt tensor's last dim / 3 (never trusted from
/// config), and each scale's (H,W) from its cls tensor shape ([1,H,W,1]).
class PoseEstimator {
 public:
  PoseEstimator(Engine& engine, PoseConfig cfg = {}, int output_base = 0);

  std::vector<PoseDetection> postprocess(const LetterboxInfo& lb) const;

  const PoseConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  PoseConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

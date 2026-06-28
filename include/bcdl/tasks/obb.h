#pragma once

#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Oriented bounding box (OBB) head — YOLO26 OBB / RDK S100 NV12 export
// ===========================================================================
//
// The head emits a SEPARATE triple of outputs per feature scale (no fused
// tensor, no DFL, no objectness). For each stride s (default {8,16,32}, grids
// {80,40,20} at 640) there are three outputs:
//   - cls   : [1, H, W, nc]  per-cell class logits (sigmoid-activated here)
//   - box   : [1, H, W, 4]   per-cell LTRB-like distances (left,top,right,bot)
//   - angle : [1, H, W, 1]   per-cell raw angle logit
//
// Decode (anchor-free, cell-center grid; grid_x=gx+0.5, grid_y=gy+0.5):
//   v_box = abs(box)                                 (l,t,r,b >= 0)
//   a     = (sigmoid(angle) - 0.5) * pi * angle_sign + angle_offset_rad
//   xf = (r - l) / 2,  yf = (b - t) / 2,  c = cos(a),  sn = sin(a)
//   cx = (grid_x + xf*c - yf*sn) * stride
//   cy = (grid_y + xf*sn + yf*c) * stride
//   w  = (l + r) * stride,   h = (t + b) * stride
//   regularize: if (w < h) swap(w,h), a += pi/2      (canonicalize w >= h)
// Then rotated per-class NMS, then un-letterbox cx,cy (lb.invX/invY) and
// w,h (/lb.scale); angle is invariant to the affine letterbox map.
//
// Ported from the reference YOLO26OBB.post_process (ref_yolo26_obb.py). The
// reference NMS is cv2.dnn.NMSBoxesRotated (rotated-rect IoU); because the
// board has no OpenCV we implement rotated-rect IoU directly (Sutherland-
// Hodgman polygon clip + shoelace area) and run greedy per-class NMS to match
// the rest of this library's detection postproc (see detection.h::nms).

/// A rotated rectangle. (cx,cy) center, (w,h) size, `angle` in RADIANS. After
/// decode these are in MODEL-INPUT pixels; after ObbDetector::postprocess /
/// decodeObb they are un-letterboxed to ORIGINAL-image pixels.
struct RotatedBox {
  float cx;
  float cy;
  float w;
  float h;
  float angle;  ///< radians
};

/// One oriented detection in original-image pixel coordinates.
struct ObbDetection {
  RotatedBox rrect;
  float score;     ///< sigmoid(max class logit), in [0,1]
  int class_id;    ///< argmax class index
};

/// Post-processing parameters for the OBB head. Defaults mirror the reference
/// YOLO26OBBConfig (__init__ of ref_yolo26_obb.py):
///   score_thres=0.25, nms_thres=0.20, angle_sign=1.0, angle_offset=0.0deg,
///   regularize=True, strides={8,16,32}.
struct ObbConfig {
  int num_classes = 15;            ///< DOTA. Overridden from the cls tensor shape.
  float conf_thresh = 0.25f;       ///< reference score_thres
  float iou_thresh = 0.20f;        ///< rotated IoU; reference nms_thres = 0.2
  int max_dets = 300;
  std::vector<int> strides = {8, 16, 32};
  bool regularize = true;          ///< if w<h: swap(w,h) and angle += pi/2
  float angle_offset_rad = 0.0f;   ///< reference angle_offset(0deg) * pi/180 = 0
  int angle_sign = 1;              ///< reference angle_sign = 1.0
};

/// Rotated-rectangle IoU. Builds the 4 corners of each box (cos/sin rotation),
/// clips A against B with the Sutherland-Hodgman algorithm to obtain the
/// (convex) intersection polygon, takes its shoelace area, and divides by the
/// union. Returns 0 for any degenerate case (non-positive area / empty clip).
float rotatedIoU(const RotatedBox& a, const RotatedBox& b);

/// Greedy per-class rotated NMS. Sorts `dets` by score descending, suppresses
/// any lower-scoring box of the SAME class whose rotatedIoU with a kept box
/// exceeds `iou_thresh`, returns kept indices (into `dets`) truncated to
/// `max_dets`.
std::vector<int> rotatedNms(const std::vector<ObbDetection>& dets, float iou_thresh,
                            int max_dets);

/// Decode parallel per-scale float cls/box/angle buffers into final detections.
///
/// `cls[i]`     : row-major [H_i, W_i, nc] class logits for scale i.
/// `box[i]`     : row-major [H_i, W_i, 4]  LTRB-like distances for scale i.
/// `angle[i]`   : row-major [H_i, W_i, 1]  raw angle logits for scale i.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `cfg`        : nc + strides (one per scale) + thresholds + angle/regularize.
/// `lb`         : letterbox geometry; centers are mapped back to original-image
///                pixels (lb.invX/invY), sizes divided by lb.scale; angle kept.
/// The four input vectors plus cfg.strides must share the same length.
std::vector<ObbDetection> decodeObb(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& angle,
    const std::vector<std::pair<int, int>>& grid_hw, const ObbConfig& cfg,
    const LetterboxInfo& lb);

/// Engine-bound OBB detector. Reads `3 * strides.size()` outputs as consecutive
/// (cls, box, angle) triples starting at `output_base`:
///   cls   = output_base + 3*i
///   box   = output_base + 3*i + 1
///   angle = output_base + 3*i + 2
/// dequantizing + de-striding each into a contiguous float buffer, taking nc and
/// the grid (H,W) from the cls tensor's own shape, then running decodeObb().
class ObbDetector {
 public:
  ObbDetector(Engine& engine, ObbConfig cfg = {}, int output_base = 0);

  std::vector<ObbDetection> postprocess(const LetterboxInfo& lb) const;

  const ObbConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  ObbConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

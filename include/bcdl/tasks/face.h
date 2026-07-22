#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Face detection with 5 landmarks + alignment for recognition
// ===========================================================================
//
// A face pipeline is two models with a geometric step wedged between them, and
// that middle step is the part that actually decides whether recognition works:
//
//   detect (SCRFD)  ->  align to a canonical 112x112  ->  embed  ->  compare
//
// The embedding model was trained exclusively on faces warped so that the eyes,
// nose and mouth corners land on fixed template positions. Feed it a plain crop
// of the detected box and the embeddings degrade badly even though nothing
// errors — which is why the landmarks are not a bonus output here, they are the
// input to alignFace(). Recognition itself needs nothing new: an aligned crop
// goes through ImageEmbedder, and matching is EmbeddingBank, both from the
// embedding task.
//
// DETECTION HEAD. SCRFD is an anchor-free distance-regression head (FCOS/GFL
// family) with `na` predictions per cell, emitting three tensors per scale:
//
//   score : [N, 1]   already sigmoid-activated
//   bbox  : [N, 4]   distances (left, top, right, bottom) from the cell centre,
//                    in STRIDE UNITS
//   kps   : [N, 10]  five (dx, dy) landmark offsets from the cell centre, also
//                    in stride units
//
// with N = H * W * na, ordered (row, column, anchor) — anchors vary fastest. The
// cell centre is (gx * stride, gy * stride); note this head puts the centre on
// the grid POINT, not at the cell's middle, so there is no +0.5 anywhere:
//
//   x1 = cx - l*stride    x2 = cx + r*stride
//   y1 = cy - t*stride    y2 = cy + b*stride
//   kx = cx + dx*stride   ky = cy + dy*stride
//
// PADDING CONVENTION. This detector's reference preprocessing scales the image
// to fit and pads BOTTOM-RIGHT, so the content sits at the top-left corner —
// unlike the centred letterbox the YOLO paths use. Build the LetterboxInfo with
// padX = padY = 0 and scale = min(dstW/srcW, dstH/srcH) to match; passing a
// centred letterbox shifts every box and landmark by the pad.

/// One detected face in ORIGINAL-image pixels, with its five landmarks in the
/// canonical order: left eye, right eye, nose, left mouth corner, right mouth
/// corner. "Left" is the viewer's left, i.e. the subject's right.
struct FaceDetection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  /// 5 landmarks as (x, y) pairs, original-image pixels.
  std::vector<std::pair<float, float>> landmarks;
};

/// Post-processing parameters for the SCRFD detection head.
struct FaceDetectConfig {
  float conf_thresh = 0.5f;
  float iou_thresh = 0.4f;
  int max_faces = 100;
  /// One stride per scale, ordered to match the (score, bbox, kps) triples.
  std::vector<int> strides = {8, 16, 32};
  /// Predictions per cell. Two for the standard export; taken on faith only if
  /// it divides the tensor length evenly, else Error(-1).
  int num_anchors = 2;
};

/// Decode parallel per-scale score/bbox/kps buffers into faces.
///
/// `score[i]`  : row-major [N_i, 1]  sigmoid-activated confidences for scale i.
/// `bbox[i]`   : row-major [N_i, 4]  l,t,r,b distances in stride units.
/// `kps[i]`    : row-major [N_i, 10] five (dx,dy) offsets in stride units.
/// `counts[i]` : N_i for scale i (= H_i * W_i * num_anchors).
/// `cfg`       : strides + anchors per cell + thresholds.
/// `lb`        : letterbox geometry (see the padding note above); boxes and
///               landmarks are mapped back to original-image pixels and boxes
///               are clamped to the source extent.
///
/// The grid is assumed square (N / num_anchors must be a perfect square), which
/// holds for the square input this head is exported at. Candidates are pooled
/// across scales, thresholded, then run through a single NMS.
std::vector<FaceDetection> decodeScrfd(
    const std::vector<const float*>& score, const std::vector<const float*>& bbox,
    const std::vector<const float*>& kps, const std::vector<int>& counts,
    const FaceDetectConfig& cfg, const LetterboxInfo& lb);

/// The canonical 112x112 landmark template the recognition models were trained
/// against, in the same order as FaceDetection::landmarks.
const std::vector<std::pair<float, float>>& arcFaceTemplate();

/// Solve the similarity transform (rotation + uniform scale + translation) that
/// best maps `src` onto `dst` in the least-squares sense, returned row-major as
/// {a, b, tx, c, d, ty} for
///     x' = a*x + b*y + tx
///     y' = c*x + d*y + ty
///
/// This is the closed-form Umeyama solution, not an iterative or RANSAC fit: with
/// five exact correspondences there is nothing to be robust against, and a
/// deterministic answer is what makes the same face align identically every
/// time. Uniform scale (rather than a general affine) is deliberate — it cannot
/// shear or stretch the face, so it preserves the proportions the embedding
/// model keys on. Throws Error(-1) if the sizes differ or fewer than two points
/// are given.
std::vector<float> similarityTransform(
    const std::vector<std::pair<float, float>>& src,
    const std::vector<std::pair<float, float>>& dst);

/// Warp a face out of `bgr` into a `size` x `size` aligned crop, using the
/// similarity transform that takes `landmarks` onto arcFaceTemplate() (scaled to
/// `size`). Bilinear sampling; out-of-bounds reads clamp to the edge.
///
/// `bgr`     : source image, `src_h` rows of `src_w` BGR pixels, `src_stride`
///             bytes per row (0 => tightly packed).
/// Returns a tightly packed `size*size*3` BGR buffer — feed it through the same
/// normalization the embedding model expects.
std::vector<uint8_t> alignFace(const uint8_t* bgr, int src_w, int src_h,
                               int src_stride,
                               const std::vector<std::pair<float, float>>& landmarks,
                               int size = 112);

/// Engine-bound SCRFD detector. Reads `3 * strides.size()` outputs as three
/// consecutive GROUPS starting at `output_base` — all scores, then all bboxes,
/// then all kps (score = base + i, bbox = base + ns + i, kps = base + 2*ns + i),
/// which is the order the reference export emits them in.
class FaceDetector {
 public:
  FaceDetector(Engine& engine, FaceDetectConfig cfg = {}, int output_base = 0);

  std::vector<FaceDetection> postprocess(const LetterboxInfo& lb) const;

  const FaceDetectConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  FaceDetectConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// XFeat sparse local features (detector + descriptor) and matching
// ===========================================================================
//
// The geometric-vision counterpart to the stereo and GDC paths: repeatable
// points plus 64-d descriptors, for calibration, stitching, tracking and SLAM
// front-ends.
//
// The model is a small all-convolutional net emitting THREE maps at 1/8 scale:
//   feats       : [1, 64, H/8, W/8]  dense descriptors
//   keypoints   : [1, 65, H/8, W/8]  per-cell logits over an 8x8 block + a
//                                    "no keypoint here" bin
//   reliability : [1,  1, H/8, W/8]  how trustworthy this neighbourhood is
//
// TWO PIECES DELIBERATELY DO NOT RUN ON THE BPU:
//
//   * The input normalization. The reference applies InstanceNorm to the
//     grayscale image — a per-image, data-dependent statistic, i.e. the kind of
//     thing int8 quantizes badly. xfeatPreprocess() does it on the CPU instead,
//     which is both exact and puts the model's input in a fixed domain.
//   * Everything after the three maps. Softmax, non-maximum suppression, top-k
//     and sparse sampling are all data-dependent control flow, which is what
//     makes a graph un-compilable; keeping them here leaves the compiled part a
//     straight-line convolutional stack with no dynamic shapes at all.
//
// The 65th logit is the reject bin: after softmax over all 65, only the first 64
// are kept and scattered back to full resolution, so a cell that "votes for no
// keypoint" simply ends up with 64 small probabilities.

/// One detected feature, in ORIGINAL-image pixels.
struct Feature {
  float x;
  float y;
  float score;  ///< detector probability times the reliability of its neighbourhood
};

/// A frame's features plus their descriptors. `descriptors` is row-major
/// `size() x dim`, each row L2-normalized, so a dot product IS the cosine
/// similarity — which is what matchFeatures() relies on.
struct FeatureSet {
  std::vector<Feature> keypoints;
  std::vector<float> descriptors;
  int dim = 0;

  std::size_t size() const { return keypoints.size(); }
  const float* descriptor(std::size_t i) const { return descriptors.data() + i * dim; }
};

struct XfeatConfig {
  /// Minimum detector probability. The reference default; lower it for
  /// texture-poor scenes, raise it to thin out matches.
  float detection_thresh = 0.05f;
  /// Side of the square non-maximum suppression window, in full-resolution
  /// pixels. Must be odd.
  int nms_kernel = 5;
  /// Keep at most this many features, highest score first. Matching is
  /// O(N*M*64), so this is the knob that decides whether matching costs
  /// milliseconds or seconds — see matchFeatures().
  int top_k = 4096;
};

/// Grayscale + resize + InstanceNorm into the model's `[1,1,in_h,in_w]` input.
///
/// TAKES BGR like every other image entry point here. Grayscale is the plain
/// CHANNEL MEAN, not a luma weighting — the reference uses `x.mean(dim=1)`, and
/// a Rec.601 gray would shift every descriptor.
///
/// Writes `in_w*in_h` floats to `out` and returns the scale factors that map
/// model-space coordinates back to the source image, to hand to decodeXfeat().
void xfeatPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                     int in_h, std::vector<float>& out, float* scale_x, float* scale_y);

/// Decode the three maps into sparse features.
///
/// `feats`       : row-major [64, fh, fw]
/// `kpt_logits`  : row-major [65, fh, fw]
/// `reliability` : row-major [1, fh, fw]
/// `in_h/in_w`   : the model's input size; must be exactly 8*fh by 8*fw.
/// `scale_x/y`   : from xfeatPreprocess(), to return ORIGINAL-image pixels.
///
/// Descriptors are sampled from the dense map at each keypoint — BICUBICALLY,
/// which is the reference's sparse-interpolator default and not the bilinear one
/// would assume; reading it as bilinear is invisible in every shape and count
/// and still produces plausible descriptors, just measurably different ones. The
/// reliability map, by contrast, IS sampled bilinearly. The dense map is
/// L2-normalized BEFORE sampling and each descriptor again after; both steps are
/// load-bearing, since a mixture of unit vectors is not itself unit length.
FeatureSet decodeXfeat(const float* feats, const float* kpt_logits, const float* reliability,
                       int fh, int fw, int in_h, int in_w, const XfeatConfig& cfg,
                       float scale_x = 1.0f, float scale_y = 1.0f);

/// Engine-bound extractor. Reads three outputs starting at `output_base`, in the
/// model's order (feats, keypoints, reliability), de-striding each.
class FeatureExtractor {
 public:
  FeatureExtractor(Engine& engine, XfeatConfig cfg = {}, int output_base = 0);

  /// Preprocess + infer + decode one BGR frame.
  FeatureSet extract(const uint8_t* bgr, int width, int height, int stride,
                     int timeout_ms = 0);

  /// Decode the Engine's current outputs (run infer() yourself first).
  FeatureSet postprocess(float scale_x = 1.0f, float scale_y = 1.0f) const;

  const XfeatConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  XfeatConfig cfg_;
  int out_base_;
  std::vector<float> input_;
};

/// One accepted correspondence: index into set A, index into set B, cosine.
struct FeatureMatch {
  int a;
  int b;
  float score;
};

/// Mutual nearest-neighbour matching with a cosine floor.
///
/// A pair survives only if each is the other's best match, which is what makes
/// this usable without a ratio test: a repeated texture produces a high cosine
/// for many candidates, but only one of them can be mutual.
///
/// COST is O(|a| * |b| * dim) — with the default top_k of 4096 on both sides
/// that is about 10^9 multiply-adds, which is why this is OpenMP-parallel and
/// why lowering XfeatConfig::top_k is the first thing to try if matching
/// dominates a frame budget.
///
/// `min_cossim <= 0` disables the floor and keeps every mutual pair.
std::vector<FeatureMatch> matchFeatures(const FeatureSet& a, const FeatureSet& b,
                                        float min_cossim = 0.82f);

}  // namespace bcdl

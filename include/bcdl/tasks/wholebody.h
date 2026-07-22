#pragma once

#include <cstdint>
#include <vector>

#include "bcdl/tasks/pose.h"  // Keypoint

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// ViTPose whole-body 133-keypoint head (top-down, plain heatmaps)
// ===========================================================================
//
// TOP-DOWN, unlike every other pose path in this library. The bottom-up LTRB
// head in pose.h reads a whole frame once and emits every person; this one runs
// ONCE PER PERSON on a crop, so it needs a person detector in front of it (any
// of the YOLO detectors here) and its cost scales with the head count.
//
// The model takes a 192x256 (WxH) RGB crop and emits ONE output:
//   heatmap : [1, 133, 64, 48]  F32, channel-first, one map per keypoint
// at 1/4 the input resolution. 133 = 17 body + 6 feet + 68 face + 42 hands
// (the COCO-WholeBody layout).
//
// GEOMETRY. The reference implementation does NOT use the usual mmpose affine
// (center/scale about a 200-px unit with a 1.25 expansion). It is much simpler,
// and this is reproduced exactly because the model was trained and published
// against it:
//   1. widen the person box by `box_pad` px per side, clip to the image;
//   2. ZERO-pad that crop, centred, to the model's 3:4 aspect;
//   3. resize the padded crop to 192x256, /255, ImageNet z-score, NCHW.
// The mapping back is therefore a pure scale-and-translate with no matrix —
// see wholeBodyPreprocess() / WholeBodyCrop.
//
// SUB-PIXEL DECODE is DARK-UDP: Gaussian-blur the heatmap, take the log, and
// take one Newton step (inverse Hessian times gradient) from the argmax cell.
// This is what the published model's own decoder does; plain argmax alone
// quantizes keypoints to the 4-px heatmap grid, which is visible on faces and
// hands where the whole point is precision.

/// Geometry of one top-down crop: how a person box became the model's canvas.
/// Produced by wholeBodyPreprocess(), consumed by decodeWholeBody() to map
/// heatmap coordinates back to ORIGINAL-image pixels.
struct WholeBodyCrop {
  int x1 = 0;         ///< left of the widened+clipped person box, original pixels
  int y1 = 0;         ///< top of the widened+clipped person box, original pixels
  int pad_left = 0;   ///< zero columns prepended to reach the 3:4 aspect
  int pad_top = 0;    ///< zero rows prepended to reach the 3:4 aspect
  int padded_w = 0;   ///< crop width after padding, before the resize
  int padded_h = 0;   ///< crop height after padding, before the resize
};

/// Preprocessing + decode parameters. The defaults reproduce the reference
/// implementation; they are fields rather than constants so a differently
/// normalized re-export can be driven without a rebuild.
struct WholeBodyConfig {
  /// Keypoints scoring below this are still returned (callers usually want the
  /// full 133-length array) but are the ones to skip when drawing.
  float kpt_thresh = 0.2f;
  /// Gaussian kernel for the DARK modulation. Must be odd. sigma is derived the
  /// way OpenCV derives it from an explicit ksize: 0.3*((k-1)*0.5-1)+0.8.
  int blur_kernel = 11;
  /// Pixels added to each side of the person box before cropping.
  int box_pad = 10;
  /// Per-channel normalization in RGB order, applied after scaling to [0,1].
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3] = {0.229f, 0.224f, 0.225f};
};

/// Crop `box` out of a BGR image and write the model's NCHW F32 input to `out`
/// (resized to `in_w` x `in_h`, `3*in_w*in_h` floats).
///
/// TAKES BGR, like every other image entry point in this library (alignFace(),
/// the letterbox helpers, the JPU decoders), and flips to RGB internally —
/// the model itself is RGB, and having one BGR-in convention is worth more than
/// mirroring the reference API's RGB-in.
///
/// The zero padding is never materialized: samples that fall in the pad region
/// read as 0, which is what padding then resizing produces. Sampling uses the
/// OpenCV pixel-center convention, matching cv::resize(INTER_LINEAR).
///
/// Throws Error(-1) if the box is empty after clipping.
WholeBodyCrop wholeBodyPreprocess(const uint8_t* bgr, int width, int height, int stride,
                                  float bx1, float by1, float bx2, float by2, int in_w,
                                  int in_h, const WholeBodyConfig& cfg,
                                  std::vector<float>& out);

/// Decode one person's heatmaps into `num_kpts` keypoints in ORIGINAL-image
/// pixels.
///
/// `heatmaps` : row-major [num_kpts, hm_h, hm_w] (channel-first, as the model
///              emits it — NOT the NHWC layout the LTRB heads use).
/// `crop`     : the geometry wholeBodyPreprocess() returned for this person.
///
/// Each keypoint's score is the RAW heatmap maximum (the blur is only used to
/// refine position). A keypoint whose maximum is <= 0 is returned with score 0
/// and is not refined — the reference's index arithmetic is degenerate there.
std::vector<Keypoint> decodeWholeBody(const float* heatmaps, int num_kpts, int hm_h, int hm_w,
                                      const WholeBodyCrop& crop, const WholeBodyConfig& cfg);

/// Engine-bound whole-body estimator. Reads output[`output_index`] as
/// [1, K, H, W] (dequantizing + de-striding through outputAsFloat) and runs
/// decodeWholeBody(); K/H/W come from the tensor's own shape, never from config.
///
/// The caller preprocesses + setInput + infer on the Engine, then calls
/// postprocess() with the crop geometry wholeBodyPreprocess() returned.
class WholeBodyEstimator {
 public:
  WholeBodyEstimator(Engine& engine, WholeBodyConfig cfg = {}, int output_index = 0);

  std::vector<Keypoint> postprocess(const WholeBodyCrop& crop) const;

  const WholeBodyConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  WholeBodyConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl

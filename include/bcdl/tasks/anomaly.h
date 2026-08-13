#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Industrial anomaly detection (one-class: "is this part like the good ones")
// ===========================================================================
//
// Structurally unlike every other task here. There is no class list and no
// training set of defects — the model is shown only NORMAL examples of one
// product and reports how unlike them each patch looks. That has two
// consequences a runtime has to respect:
//
//   * the model is per-product, so the shipped .hbm is an example, not a
//     general-purpose detector: deploying to a new part means retraining;
//   * the score is calibrated against that product's normal variation, so a
//     threshold is a per-line decision, not a library default. The default here
//     exists to make the API usable, not because 0.5 means anything universal.
//
// The head emits a coarse per-patch score grid (36x36 for a 288x288 input),
// which is upsampled and smoothed to the input resolution — the smoothing is
// part of the reference post-processing, not a cosmetic choice, and it moves
// pixel-level agreement measurably.

/// Post-processing parameters.
struct AnomalyConfig {
  int out_width = 0;      ///< 0 => keep the raw grid size (no upsample)
  int out_height = 0;
  int blur_kernel = 33;   ///< odd; 0 or 1 disables the Gaussian smoothing
  float blur_sigma = 4.0f;
  float threshold = 0.5f; ///< for mask(); see the note above about defaults
};

/// A per-pixel anomaly map plus the image-level verdict.
///
/// `score` is the map's maximum: a part is as anomalous as its worst patch,
/// which is the convention the anomaly-detection benchmarks score against. A
/// mean would let one large clean region hide a small defect — exactly the case
/// the whole method exists to catch.
struct AnomalyMap {
  int width = 0;
  int height = 0;
  std::vector<float> data;
  float score = 0.0f;
};

/// Decode a score grid: optional bilinear upsample, optional Gaussian blur, then
/// the image-level max. `shape` is `{1,H,W}` / `{H,W}` / `{1,1,H,W}`.
/// Pure function: no Engine, numpy-testable.
AnomalyMap decodeAnomaly(const float* data, const std::vector<int>& shape,
                         const AnomalyConfig& cfg = {});

/// Threshold a map into a 0/255 mask (H*W uint8), for overlaying or blob counting.
std::vector<uint8_t> anomalyMask(const AnomalyMap& map, float threshold);

/// Format one frame for the model: resize, BGR->RGB, /255, then per-channel
/// mean/std. Defaults are the ImageNet statistics the reference implementation
/// uses; they are part of the trained contract, not a preference.
void anomalyPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                       int in_h, std::vector<float>& out,
                       const float mean[3] = nullptr, const float stddev[3] = nullptr);

/// Engine-bound detector: one image in, an AnomalyMap out.
///
/// The output map defaults to the model's INPUT resolution rather than the raw
/// grid, because a 36x36 grid is not something a caller can overlay on a part
/// photo without redoing the upsample the reference already specifies.
class AnomalyDetector {
 public:
  AnomalyDetector(Engine& engine, AnomalyConfig cfg = {}, int output_index = 0,
                  int input_index = 0);

  AnomalyMap detect(const uint8_t* bgr, int width, int height, int stride = 0,
                    int timeout_ms = 0);

  /// Decode the Engine's current output (for callers that drove infer()
  /// themselves).
  AnomalyMap postprocess() const;

  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  const AnomalyConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  AnomalyConfig cfg_;
  int out_idx_;
  int in_idx_;
  int in_w_ = 0;
  int in_h_ = 0;
  std::vector<float> buf_;
  std::vector<std::uint8_t> stage_;
};

}  // namespace bcdl

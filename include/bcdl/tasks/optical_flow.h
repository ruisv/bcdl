#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Dense optical flow (two frames -> per-pixel displacement)
// ===========================================================================
//
// The first task here that takes TWO images and answers a question about the
// pair rather than about either one: where did each pixel go. Everything else
// in this library reduces a frame to a description; this one keeps the frame's
// shape and fills it with motion.
//
// Vectors are in PIXELS, +u right, +v down — the OpenCV convention, so a decoded
// field drops straight into cv::remap once split into x/y maps.

/// Decode parameters.
///
/// `scale_x`/`scale_y` convert model pixels to source pixels. A model run on a
/// downscaled pair reports displacement in ITS pixels; feeding those numbers to
/// a full-resolution warp silently under-moves everything, and nothing about
/// the field looks wrong while it happens. `OpticalFlowEstimator` sets them from
/// the sizes it was given.
struct FlowConfig {
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  bool channels_first = true;  ///< [1,2,H,W] (true) vs [1,H,W,2]
};

/// A dense flow field. `data` is H*W*2 INTERLEAVED (u, v) — the CV_32FC2 layout,
/// not the model's planar one, because everything downstream (remap, magnitude,
/// visualisation) wants the pair together.
struct FlowField {
  int width = 0;
  int height = 0;
  std::vector<float> data;

  float u(int x, int y) const { return data[(static_cast<std::size_t>(y) * width + x) * 2]; }
  float v(int x, int y) const { return data[(static_cast<std::size_t>(y) * width + x) * 2 + 1]; }
};

/// Read a flow tensor into a FlowField, de-planarizing and applying the scale.
///
/// `shape` is `{1,2,H,W}` / `{2,H,W}` (channels_first) or `{1,H,W,2}` / `{H,W,2}`.
/// Pure function: no Engine, numpy-testable.
FlowField decodeFlow(const float* data, const std::vector<int>& shape,
                     const FlowConfig& cfg = {});

/// Middlebury colour-wheel visualisation -> BGR H*W*3 uint8.
///
/// Hue is direction, saturation is magnitude normalised by `max_magnitude`;
/// pass 0 to normalise by the field's own 99th-percentile magnitude, which keeps
/// one fast object from washing out the rest of the frame.
std::vector<uint8_t> flowColorize(const FlowField& flow, float max_magnitude = 0.0f);

/// Endpoint error against a reference field: mean over pixels of the vector
/// difference length. The standard flow metric, and the one to use when scoring
/// a quantised model against its float reference.
float flowEndpointError(const FlowField& a, const FlowField& b);

/// Format one frame for a two-frame flow model: resize to the model's input size
/// and emit CHW float.
///
/// **Channel order is left as BGR and the range as 0-255** — the reference
/// implementation feeds `cv::imread` output straight in and divides by 255
/// inside the graph. Handing it RGB, or pre-dividing, produces a plausible but
/// wrong field rather than an error.
void flowPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                    int in_h, std::vector<float>& out);

/// Engine-bound flow head: two frames in, a FlowField out.
///
/// The model's input size comes from the Engine, so `estimate()` takes source
/// images at any size, resizes them, and returns a field whose vectors are
/// already expressed in SOURCE pixels.
class OpticalFlowEstimator {
 public:
  OpticalFlowEstimator(Engine& engine, FlowConfig cfg = {}, int output_index = 0,
                       int input0_index = 0, int input1_index = 1);

  FlowField estimate(const uint8_t* bgr0, const uint8_t* bgr1, int width, int height,
                     int stride = 0, int timeout_ms = 0);

  /// Decode the Engine's current output (for callers that drove infer()
  /// themselves). Uses the configured scale as-is.
  FlowField postprocess() const;

  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  const FlowConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  FlowConfig cfg_;
  int out_idx_;
  int in0_idx_;
  int in1_idx_;
  int in_w_ = 0;
  int in_h_ = 0;
  std::vector<float> buf0_, buf1_;
  std::vector<std::uint8_t> stage0_, stage1_;
};

}  // namespace bcdl

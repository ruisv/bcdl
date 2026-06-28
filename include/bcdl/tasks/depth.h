#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Post-processing parameters for a single-channel dense depth / disparity head.
struct DepthConfig {
  int width = 0;             ///< output map width  (0 => infer from output shape)
  int height = 0;           ///< output map height (0 => infer from output shape)
  bool normalize = true;     ///< min-max normalize values to [0,1]
  /// Optional pre-normalize clip. When clip_hi > clip_lo, raw values are clamped
  /// to [clip_lo, clip_hi] BEFORE the min-max normalize (and before vmin/vmax are
  /// reported). Leave both 0 to disable.
  float clip_lo = 0.0f;
  float clip_hi = 0.0f;
};

/// Decoded depth map. `data` is row-major HxW; raw values when cfg.normalize is
/// false, else normalized to [0,1]. `vmin`/`vmax` are the raw min/max actually
/// observed (after clip, before normalize) so callers can re-colorize externally.
struct DepthMap {
  int width = 0;
  int height = 0;
  std::vector<float> data;
  float vmin = 0.0f;
  float vmax = 0.0f;
};

/// Decode a single-channel float depth/disparity tensor into a DepthMap.
///
/// `data`  : pointer to product(shape) floats in logical row-major order.
/// `shape` : one of {1,H,W}, {1,1,H,W}, or {H,W}.
/// `cfg`   : width/height override (0 => infer), clip + normalize options.
/// Computes vmin/vmax over all pixels. When cfg.normalize, emits (clip then)
/// (v-vmin)/(vmax-vmin) in [0,1]; else copies raw. Guards vmax==vmin.
DepthMap decodeDepth(const float* data, const std::vector<int>& shape,
                     const DepthConfig& cfg);

/// Min-max normalize a raw depth map to uint8 [0,255] (row-major HxW). When the
/// map is already normalized ([0,1]) this just scales; otherwise it renormalizes
/// from vmin/vmax. Useful as a grayscale visualization or colormap source.
std::vector<uint8_t> depthToGray8(const DepthMap& m);

/// Apply an (approximate) Turbo colormap to a depth map -> BGR HxWx3 uint8.
/// Output is OpenCV-convention BGR. The Turbo curve is a compact polynomial
/// approximation of Google's Turbo palette (good enough for visualization, not
/// colorimetrically exact).
std::vector<uint8_t> depthColorize(const DepthMap& m);

/// Convenience wrapper binding an Engine output to a DepthConfig.
///
/// The caller preprocesses + setInput + infer on the Engine. `postprocess()`
/// reads the selected output, converts it to float (F32 directly; F16 via
/// half->float; SCALE-quantized S8/S16/S32 via per-tensor/per-channel dequant),
/// honors the device byte strides, and runs decodeDepth().
class DepthEstimator {
 public:
  DepthEstimator(Engine& engine, DepthConfig cfg = {}, int output_index = 0);

  DepthMap postprocess() const;

  const DepthConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DepthConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl

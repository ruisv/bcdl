#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Post-processing parameters for a dense semantic-segmentation head.
struct SegConfig {
  int num_classes = 0;        ///< channel count C (0 => infer from a [1,C,H,W]/[1,H,W,C] tensor)
  bool channels_first = true;  ///< logits layout: [1,C,H,W] (true) vs [1,H,W,C] (false)
  /// Set when the model already emits per-pixel class ids ([1,H,W] integer or
  /// float). decodeSeg then rounds float->int instead of argmaxing channels.
  bool argmaxed = false;
};

/// Decoded segmentation mask. `labels` is row-major HxW class ids.
struct SegMask {
  int width = 0;
  int height = 0;
  int num_classes = 0;
  std::vector<int32_t> labels;
};

/// Argmax a logit tensor or pass through an id tensor into a SegMask.
///
/// `data`  : pointer to product(shape) floats in logical row-major order.
/// `shape` : {1,C,H,W} / {1,H,W,C} logits, or {1,H,W} / {H,W} class ids.
/// `cfg`   : layout + num_classes (0 => infer); argmaxed selects pass-through.
/// For logits, per pixel argmax over C. For argmaxed input, round float->int.
SegMask decodeSeg(const float* data, const std::vector<int>& shape,
                  const SegConfig& cfg);

/// Color a label mask with a fixed deterministic palette -> BGR HxWx3 uint8.
/// The first 21 ids use the classic PASCAL VOC palette; higher ids fall back to
/// a golden-ratio hue generator. Output is OpenCV-convention BGR.
std::vector<uint8_t> segColorize(const SegMask& m);

/// Convenience wrapper binding an Engine output to a SegConfig.
///
/// The caller preprocesses + setInput + infer on the Engine. `postprocess()`
/// reads the selected output, converts it to float (F32 directly; F16 via
/// half->float; SCALE-quantized S8/S16/S32 via per-tensor/per-channel dequant),
/// honors the device byte strides, and runs decodeSeg().
class Segmenter {
 public:
  Segmenter(Engine& engine, SegConfig cfg = {}, int output_index = 0);

  SegMask postprocess() const;

  const SegConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  SegConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl

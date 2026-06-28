#pragma once

#include <cmath>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Numerically-stable logistic sigmoid. Shared by every YOLO-family decoder
/// (class scores, keypoint / angle activations are all sigmoid logits).
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Return a logical row-major float view of Engine output[out_idx], filling
/// `shape` with its logical shape.
///
/// FAST PATH (the common RDK case): an F32 tensor with packed (contiguous)
/// strides is returned ZERO-COPY as a direct pointer into the (already
/// cache-invalidated) device buffer — no per-element dequant/stride walk, no
/// allocation. Detection/pose/seg/obb heads emit ~10^5–10^6 elements per frame,
/// so skipping the gather is the difference between ~10ms and ~1ms of postproc.
///
/// SLOW PATH: any other dtype/layout (F16, quantized int, or stride-padded) is
/// gathered + dequantized into `scratch` and that pointer is returned. The
/// returned pointer is valid until `scratch` is destroyed or the next infer().
const float* outputAsFloat(Engine& engine, int out_idx, std::vector<float>& scratch,
                           std::vector<int>& shape);

}  // namespace bcdl

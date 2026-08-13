#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Write `count` floats into Engine input[in_idx], converting to whatever the
/// model declared that input to be.
///
/// The mirror image of outputAsFloat(). Most RDK exports take F32 (or NV12
/// uint8) inputs, so callers could memcpy through Engine::setInput() directly —
/// but a model compiled with QUANTIZED INPUTS declares S8/S16 with a per-tensor
/// (or per-axis) scale, and its input buffer then holds integers, not floats.
/// Handing such a model raw float bytes is not a small error: the first four
/// bytes of one float land as two int16 samples and every value after them is
/// shifted, so the model runs happily on noise. This does the
/// `round(value / scale) + zero_point` the tensor asks for, clamped to the
/// integer range, and hands Engine::setInput() a packed buffer in the declared
/// dtype (setInput scatters it into the padded device layout).
///
/// `scratch` is the staging buffer; hold one per input across frames to keep the
/// conversion allocation-free.
///
/// Throws Error(-1) if `count` does not match the input's valid shape, or if the
/// input declares a dtype this cannot produce (F16 — no model in this library
/// has one, and an untested float->half path is worse than a clear refusal).
void setInputFromFloat(Engine& engine, int in_idx, const float* data, std::size_t count,
                       std::vector<std::uint8_t>& scratch);

/// Element count of input[in_idx]'s valid shape — what `count` above must be.
std::size_t inputElementCount(const Engine& engine, int in_idx);

}  // namespace bcdl

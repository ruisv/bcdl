#include "bcdl/tasks/depth.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "hobot/dnn/hb_dnn.h"

namespace bcdl {

namespace {

/// IEEE-754 binary16 -> binary32. hbDNN F16 outputs are little-endian uint16.
inline float half2float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -1;
      uint32_t m = mant;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      m &= 0x3FFu;
      const uint32_t fexp = static_cast<uint32_t>(127 - 15 - e);
      bits = sign | (fexp << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    const uint32_t fexp = exp - 15 + 127;
    bits = sign | (fexp << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/// Read element `linear` (logical row-major flat index over validShape) from a
/// raw device buffer, honoring byte strides and dequantizing per the tensor's
/// quant params. Mirrors the detection task's helper.
float readElement(const uint8_t* base, int tensor_type, const hbDNNQuantiScale& q,
                  hbDNNQuantiType quanti, int quant_axis, int num_dims,
                  const int32_t* dims, const int64_t* strides, int64_t linear) {
  int64_t byte_off = 0;
  int64_t rem = linear;
  int axis_coord = 0;
  for (int d = num_dims - 1; d >= 0; --d) {
    const int64_t coord = dims[d] > 0 ? (rem % dims[d]) : 0;
    rem = dims[d] > 0 ? (rem / dims[d]) : rem;
    byte_off += coord * strides[d];
    if (d == quant_axis) axis_coord = static_cast<int>(coord);
  }
  const uint8_t* p = base + byte_off;

  double raw;
  switch (tensor_type) {
    case HB_DNN_TENSOR_TYPE_F32: {
      float v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    case HB_DNN_TENSOR_TYPE_F16: {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      return half2float(v);
    }
    case HB_DNN_TENSOR_TYPE_S8: {
      int8_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U8: {
      uint8_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_S16: {
      int16_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U16: {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_S32: {
      int32_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    case HB_DNN_TENSOR_TYPE_U32: {
      uint32_t v;
      std::memcpy(&v, p, sizeof(v));
      raw = v;
      break;
    }
    default:
      throw Error(-1, "BCDL depth: unsupported output tensor type " +
                          std::to_string(tensor_type) +
                          "; export an F32 (or F16) depth head");
  }

  if (quanti != SCALE || q.scaleLen <= 0 || q.scaleData == nullptr) {
    return static_cast<float>(raw);
  }
  const int si = q.scaleLen == 1 ? 0 : axis_coord;
  double scaled = raw;
  if (q.zeroPointLen > 0 && q.zeroPointData != nullptr) {
    const int zi = q.zeroPointLen == 1 ? 0 : axis_coord;
    scaled -= q.zeroPointData[zi];
  }
  scaled *= q.scaleData[si];
  return static_cast<float>(scaled);
}

/// Resolve (H,W) from a single-channel dense tensor shape. Accepts {H,W},
/// {1,H,W}, {1,1,H,W} (and tolerates a trailing channel of 1, e.g. {1,H,W,1}).
void resolveHW(const std::vector<int>& shape, int& H, int& W) {
  // Collect the non-unit dims in order; the last two meaningful dims are H,W.
  std::vector<int> dims;
  for (int d : shape) {
    if (d > 1) dims.push_back(d);
  }
  if (dims.size() >= 2) {
    H = dims[dims.size() - 2];
    W = dims[dims.size() - 1];
    return;
  }
  // Degenerate fallback: trust raw shape (e.g. {1,1} or {H,1}).
  const int n = static_cast<int>(shape.size());
  H = n >= 2 ? shape[n - 2] : (n >= 1 ? shape[n - 1] : 0);
  W = n >= 1 ? shape[n - 1] : 0;
}

/// Turbo colormap approximation (Google Turbo). Compact polynomial fit; returns
/// RGB in [0,255]. Input t clamped to [0,1]. NOT colorimetrically exact.
void turbo(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
  t = std::min(1.0f, std::max(0.0f, t));
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float t4 = t3 * t;
  const float t5 = t4 * t;
  // Polynomial coefficients (degree-5 fit of the Turbo colormap).
  float fr = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 +
             132.13108234f * t3 - 152.94239396f * t4 + 59.28637943f * t5;
  float fg = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 -
             14.18503333f * t3 + 4.27729857f * t4 + 2.82956604f * t5;
  float fb = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 +
             110.36276771f * t3 - 89.90310912f * t4 + 27.34824973f * t5;
  fr = std::min(1.0f, std::max(0.0f, fr));
  fg = std::min(1.0f, std::max(0.0f, fg));
  fb = std::min(1.0f, std::max(0.0f, fb));
  r = static_cast<uint8_t>(std::lround(fr * 255.0f));
  g = static_cast<uint8_t>(std::lround(fg * 255.0f));
  b = static_cast<uint8_t>(std::lround(fb * 255.0f));
}

}  // namespace

DepthMap decodeDepth(const float* data, const std::vector<int>& shape,
                     const DepthConfig& cfg) {
  int H = 0, W = 0;
  resolveHW(shape, H, W);
  if (cfg.height > 0) H = cfg.height;
  if (cfg.width > 0) W = cfg.width;

  DepthMap m;
  m.width = W;
  m.height = H;
  const int64_t n = static_cast<int64_t>(H) * static_cast<int64_t>(W);
  if (n <= 0) {
    m.vmin = 0.0f;
    m.vmax = 0.0f;
    return m;
  }

  const bool do_clip = cfg.clip_hi > cfg.clip_lo;

  // Memory-safety bound: `data` holds product(shape) elements. If a cfg
  // height/width override (or a shape mismatch) makes H*W exceed that, only read
  // what the buffer holds — the tail of m.data stays zero — instead of reading
  // past the end (the override was previously unbounded -> out-of-bounds read).
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? static_cast<int64_t>(d) : 0);
  const int64_t readable = std::min(n, total);

  // First pass: gather clipped values + observe raw range.
  m.data.resize(static_cast<size_t>(n));
  float vmin = std::numeric_limits<float>::infinity();
  float vmax = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < readable; ++i) {
    float v = data[i];
    if (do_clip) v = std::min(cfg.clip_hi, std::max(cfg.clip_lo, v));
    m.data[static_cast<size_t>(i)] = v;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  m.vmin = (readable > 0) ? vmin : 0.0f;
  m.vmax = (readable > 0) ? vmax : 0.0f;

  if (cfg.normalize) {
    const float range = vmax - vmin;
    if (range > 0.0f) {
      const float inv = 1.0f / range;
      for (int64_t i = 0; i < n; ++i) {
        m.data[static_cast<size_t>(i)] =
            (m.data[static_cast<size_t>(i)] - vmin) * inv;
      }
    } else {
      // Flat map — avoid divide-by-zero; emit all zeros.
      std::fill(m.data.begin(), m.data.end(), 0.0f);
    }
  }
  return m;
}

std::vector<uint8_t> depthToGray8(const DepthMap& m) {
  const int64_t n = static_cast<int64_t>(m.width) * static_cast<int64_t>(m.height);
  std::vector<uint8_t> out(static_cast<size_t>(std::max<int64_t>(0, n)));
  if (n <= 0) return out;

  // Scale by the map's OWN observed range (decided once over the whole map, not
  // per pixel) so it renders correctly whether `data` is raw depth or already
  // normalized to [0,1].
  float lo = m.data[0], hi = m.data[0];
  for (int64_t i = 1; i < n; ++i) {
    const float v = m.data[static_cast<size_t>(i)];
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  const float range = hi - lo;
  if (range <= 0.0f) return out;  // flat map -> all zeros
  const float inv = 1.0f / range;
  for (int64_t i = 0; i < n; ++i) {
    const float t = std::min(1.0f, std::max(0.0f, (m.data[static_cast<size_t>(i)] - lo) * inv));
    out[static_cast<size_t>(i)] = static_cast<uint8_t>(std::lround(t * 255.0f));
  }
  return out;
}

std::vector<uint8_t> depthColorize(const DepthMap& m) {
  const std::vector<uint8_t> gray = depthToGray8(m);
  std::vector<uint8_t> out(gray.size() * 3);
  for (size_t i = 0; i < gray.size(); ++i) {
    uint8_t r, g, b;
    turbo(gray[i] / 255.0f, r, g, b);
    // BGR order (OpenCV convention).
    out[i * 3 + 0] = b;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = r;
  }
  return out;
}

DepthEstimator::DepthEstimator(Engine& engine, DepthConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {}

DepthMap DepthEstimator::postprocess() const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL depth: output index out of range");
  }
  const std::vector<int> shape = engine_.outputShape(out_idx_);
  const hbDNNTensorProperties& props = engine_.outputProperties(out_idx_);
  const auto* base = static_cast<const uint8_t*>(engine_.outputData(out_idx_));

  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);

  std::vector<float> buf(static_cast<size_t>(total));
  const int num_dims = props.validShape.numDimensions;
  for (int64_t i = 0; i < total; ++i) {
    buf[static_cast<size_t>(i)] =
        readElement(base, props.tensorType, props.scale, props.quantiType,
                    props.quantizeAxis, num_dims, props.validShape.dimensionSize,
                    props.stride, i);
  }

  return decodeDepth(buf.data(), shape, cfg_);
}

}  // namespace bcdl

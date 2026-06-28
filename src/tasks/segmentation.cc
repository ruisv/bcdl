#include "bcdl/tasks/segmentation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
      throw Error(-1, "BCDL segmentation: unsupported output tensor type " +
                          std::to_string(tensor_type) +
                          "; export an F32 (or F16) segmentation head");
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

/// Resolve (H,W) of a class-id tensor by dropping unit dims (e.g. {1,H,W}).
void resolveHW(const std::vector<int>& shape, int& H, int& W) {
  std::vector<int> dims;
  for (int d : shape) {
    if (d > 1) dims.push_back(d);
  }
  if (dims.size() >= 2) {
    H = dims[dims.size() - 2];
    W = dims[dims.size() - 1];
    return;
  }
  const int n = static_cast<int>(shape.size());
  H = n >= 2 ? shape[n - 2] : (n >= 1 ? shape[n - 1] : 0);
  W = n >= 1 ? shape[n - 1] : 0;
}

/// HSV (h,s,v all in [0,1]) -> RGB in [0,255].
void hsv2rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
  const float hh = h * 6.0f;
  const int i = static_cast<int>(std::floor(hh)) % 6;
  const float f = hh - std::floor(hh);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));
  float fr, fg, fb;
  switch (i) {
    case 0: fr = v; fg = t; fb = p; break;
    case 1: fr = q; fg = v; fb = p; break;
    case 2: fr = p; fg = v; fb = t; break;
    case 3: fr = p; fg = q; fb = v; break;
    case 4: fr = t; fg = p; fb = v; break;
    default: fr = v; fg = p; fb = q; break;
  }
  r = static_cast<uint8_t>(std::lround(fr * 255.0f));
  g = static_cast<uint8_t>(std::lround(fg * 255.0f));
  b = static_cast<uint8_t>(std::lround(fb * 255.0f));
}

/// Deterministic palette: classic PASCAL VOC bit-interleave for id < 21, then a
/// golden-ratio hue generator for higher ids (distinct, bright). RGB out.
void paletteColor(int32_t id, uint8_t& r, uint8_t& g, uint8_t& b) {
  if (id < 0) {
    r = g = b = 0;
    return;
  }
  if (id < 21) {
    int c = id;
    int rr = 0, gg = 0, bb = 0;
    for (int i = 0; i < 8; ++i) {
      rr |= ((c >> 0) & 1) << (7 - i);
      gg |= ((c >> 1) & 1) << (7 - i);
      bb |= ((c >> 2) & 1) << (7 - i);
      c >>= 3;
    }
    r = static_cast<uint8_t>(rr);
    g = static_cast<uint8_t>(gg);
    b = static_cast<uint8_t>(bb);
    return;
  }
  const float golden = 0.6180339887498949f;
  const float h = std::fmod(static_cast<float>(id) * golden, 1.0f);
  hsv2rgb(h, 0.7f, 0.95f, r, g, b);
}

/// Fast per-pixel argmax over the channel axis directly on an F32 device output
/// buffer — no full materialization, no per-element index decomposition. Honors
/// device byte strides; OpenMP over rows. This is the official
/// np.argmax(axis=-1) pattern and replaces the materialize-then-argmax path for
/// large F32 seg maps (e.g. deeplabv3plus [1,1024,2048,19] ~662ms -> tens of ms).
/// `num_classes_override` (>0) clamps C (never grows it — stays in-bounds).
SegMask argmaxF32(const uint8_t* base, const hbDNNTensorProperties& props,
                  bool channels_first, int num_classes_override) {
  SegMask m;
  const int nd = props.validShape.numDimensions;
  if (nd < 2) return m;
  const int32_t* dim = props.validShape.dimensionSize;
  const int64_t* st = props.stride;  // BYTE strides, per dim
  // Locate the channel / height / width dims (tolerating a missing batch dim).
  const int ci = channels_first ? nd - 3 : nd - 1;
  const int hi = channels_first ? nd - 2 : nd - 3;
  const int wi = channels_first ? nd - 1 : nd - 2;
  if (ci < 0 || hi < 0 || wi < 0) return m;

  int C = dim[ci];
  const int H = dim[hi];
  const int W = dim[wi];
  if (num_classes_override > 0 && num_classes_override < C) C = num_classes_override;
  m.width = W;
  m.height = H;
  m.num_classes = C;
  const int64_t npix = static_cast<int64_t>(H) * static_cast<int64_t>(W);
  m.labels.resize(static_cast<size_t>(std::max<int64_t>(0, npix)));
  if (npix <= 0 || C <= 0) return m;

  const int64_t sH = st[hi], sW = st[wi], sC = st[ci];
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    const uint8_t* yrow = base + static_cast<int64_t>(y) * sH;
    int32_t* out = m.labels.data() + static_cast<int64_t>(y) * W;
    for (int x = 0; x < W; ++x) {
      const uint8_t* pix = yrow + static_cast<int64_t>(x) * sW;
      float best;
      std::memcpy(&best, pix, sizeof(float));  // c = 0
      int best_c = 0;
      for (int c = 1; c < C; ++c) {
        float v;
        std::memcpy(&v, pix + static_cast<int64_t>(c) * sC, sizeof(float));
        if (v > best) {  // ties -> lowest channel, matching decodeSeg
          best = v;
          best_c = c;
        }
      }
      out[x] = best_c;
    }
  }
  return m;
}

}  // namespace

SegMask decodeSeg(const float* data, const std::vector<int>& shape,
                  const SegConfig& cfg) {
  SegMask m;

  if (cfg.argmaxed) {
    // Pass-through class-id tensor ({1,H,W} / {H,W}); round float -> int.
    int H = 0, W = 0;
    resolveHW(shape, H, W);
    m.width = W;
    m.height = H;
    m.num_classes = cfg.num_classes;  // unknown unless caller specified
    const int64_t n = static_cast<int64_t>(H) * static_cast<int64_t>(W);
    m.labels.resize(static_cast<size_t>(std::max<int64_t>(0, n)));
    int32_t maxlabel = -1;
    for (int64_t i = 0; i < n; ++i) {
      const int32_t lbl = static_cast<int32_t>(std::lround(data[i]));
      m.labels[static_cast<size_t>(i)] = lbl;
      if (lbl > maxlabel) maxlabel = lbl;
    }
    if (m.num_classes <= 0) m.num_classes = maxlabel + 1;
    return m;
  }

  // Logit tensor: infer C,H,W from shape per layout (tolerating a missing batch
  // dim, i.e. {C,H,W} or {H,W,C} as well as the {1,...} forms).
  std::vector<int> dims;
  for (int d : shape) dims.push_back(d);
  int C = 0, H = 0, W = 0;
  if (cfg.channels_first) {
    // [1,C,H,W] or [C,H,W].
    if (dims.size() >= 4) {
      C = dims[dims.size() - 3];
      H = dims[dims.size() - 2];
      W = dims[dims.size() - 1];
    } else if (dims.size() == 3) {
      C = dims[0];
      H = dims[1];
      W = dims[2];
    }
  } else {
    // [1,H,W,C] or [H,W,C].
    if (dims.size() >= 4) {
      H = dims[dims.size() - 3];
      W = dims[dims.size() - 2];
      C = dims[dims.size() - 1];
    } else if (dims.size() == 3) {
      H = dims[0];
      W = dims[1];
      C = dims[2];
    }
  }
  if (cfg.num_classes > 0) C = cfg.num_classes;

  const int64_t npix = static_cast<int64_t>(H) * static_cast<int64_t>(W);
  // Memory-safety bound: the per-pixel argmax reads C channels from `data`,
  // which holds product(shape) elements. Clamp C so a too-large cfg.num_classes
  // or a shape/channel mismatch can never read past the buffer (CRITICAL: the
  // override was previously unbounded -> out-of-bounds read).
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? static_cast<int64_t>(d) : 0);
  if (npix > 0 && static_cast<int64_t>(C) * npix > total)
    C = static_cast<int>(total / npix);

  m.width = W;
  m.height = H;
  m.num_classes = C;
  m.labels.resize(static_cast<size_t>(std::max<int64_t>(0, npix)));
  if (npix <= 0 || C <= 0) return m;

  // Per-pixel argmax over C. Logical row-major addressing:
  //  channels_first: value(c, p) = data[c*npix + p]   (batch dim of 1)
  //  channels_last : value(c, p) = data[p*C + c]
  for (int64_t p = 0; p < npix; ++p) {
    int best_c = 0;
    float best_v;
    if (cfg.channels_first) {
      best_v = data[p];
      for (int c = 1; c < C; ++c) {
        const float v = data[static_cast<int64_t>(c) * npix + p];
        if (v > best_v) {
          best_v = v;
          best_c = c;
        }
      }
    } else {
      const int64_t base = p * C;
      best_v = data[base];
      for (int c = 1; c < C; ++c) {
        const float v = data[base + c];
        if (v > best_v) {
          best_v = v;
          best_c = c;
        }
      }
    }
    m.labels[static_cast<size_t>(p)] = best_c;
  }
  return m;
}

std::vector<uint8_t> segColorize(const SegMask& m) {
  const int64_t n = static_cast<int64_t>(m.width) * static_cast<int64_t>(m.height);
  std::vector<uint8_t> out(static_cast<size_t>(std::max<int64_t>(0, n)) * 3);
  for (int64_t i = 0; i < n; ++i) {
    uint8_t r, g, b;
    paletteColor(m.labels[static_cast<size_t>(i)], r, g, b);
    // BGR order (OpenCV convention).
    out[static_cast<size_t>(i) * 3 + 0] = b;
    out[static_cast<size_t>(i) * 3 + 1] = g;
    out[static_cast<size_t>(i) * 3 + 2] = r;
  }
  return out;
}

Segmenter::Segmenter(Engine& engine, SegConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {}

SegMask Segmenter::postprocess() const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL segmentation: output index out of range");
  }
  const hbDNNTensorProperties& props = engine_.outputProperties(out_idx_);
  const auto* base = static_cast<const uint8_t*>(engine_.outputData(out_idx_));

  // Fast path: F32 logit map -> direct channel-argmax (official
  // np.argmax(axis=-1)) on the device buffer — OpenMP, no 40M-float temp, no
  // per-element index decomposition. The board's deeplabv3plus output is F32
  // NHWC, so this is the hot path; the materialize+decodeSeg fallback below
  // still covers F16 / quantized / pre-argmaxed pass-through outputs.
  if (!cfg_.argmaxed && props.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    return argmaxF32(base, props, cfg_.channels_first, cfg_.num_classes);
  }

  const std::vector<int> shape = engine_.outputShape(out_idx_);
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

  return decodeSeg(buf.data(), shape, cfg_);
}

}  // namespace bcdl

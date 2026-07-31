#include "bcdl/tasks/depth_refine.h"

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

constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};

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
/// quant params. Mirrors the depth / detection task helpers.
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
      throw Error(-1, "BCDL depth_refine: unsupported output tensor type " +
                          std::to_string(tensor_type));
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

/// One output sample's contribution window along an axis: source range plus the
/// per-source-pixel weights, exactly cv::resize(INTER_AREA)'s table.
struct AreaTap {
  int start;
  int count;
  float weight[64];
};

/// Build the area-average table mapping `dst` samples onto `src` samples.
/// Weights are the overlap lengths of [d*s, (d+1)*s] with each integer cell,
/// normalized to sum to 1.
std::vector<AreaTap> buildAreaTable(int src, int dst) {
  std::vector<AreaTap> taps(static_cast<size_t>(dst));
  const double scale = static_cast<double>(src) / static_cast<double>(dst);
  for (int d = 0; d < dst; ++d) {
    const double f0 = d * scale;
    const double f1 = std::min(f0 + scale, static_cast<double>(src));
    int s0 = static_cast<int>(std::floor(f0));
    const int s1 = std::max(s0 + 1, static_cast<int>(std::ceil(f1)));
    AreaTap& t = taps[static_cast<size_t>(d)];
    t.start = s0;
    t.count = std::min(s1 - s0, 64);
    double total = 0.0;
    for (int i = 0; i < t.count; ++i) {
      const int s = s0 + i;
      const double lo = std::max(f0, static_cast<double>(s));
      const double hi = std::min(f1, static_cast<double>(s + 1));
      const double w = std::max(0.0, hi - lo);
      t.weight[i] = static_cast<float>(w);
      total += w;
    }
    if (total > 0.0) {
      for (int i = 0; i < t.count; ++i) t.weight[i] = static_cast<float>(t.weight[i] / total);
    } else {
      t.count = 1;
      t.weight[0] = 1.0f;
    }
  }
  return taps;
}

}  // namespace

Intrinsics scaleIntrinsics(const Intrinsics& k, int from_w, int from_h, int to_w, int to_h) {
  if (from_w <= 0 || from_h <= 0) return k;
  const float sx = static_cast<float>(to_w) / static_cast<float>(from_w);
  const float sy = static_cast<float>(to_h) / static_cast<float>(from_h);
  return Intrinsics{k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy};
}

void preprocessRefineImage(const uint8_t* bgr, int width, int height, int stride_bytes,
                           const DepthRefineConfig& cfg, std::vector<float>* out) {
  if (bgr == nullptr || out == nullptr) throw Error(-1, "BCDL depth_refine: null image buffer");
  const int eh = cfg.encoder_height, ew = cfg.encoder_width;
  if (width <= 0 || height <= 0 || eh <= 0 || ew <= 0) {
    throw Error(-1, "BCDL depth_refine: bad image or encoder size");
  }
  if (stride_bytes <= 0) stride_bytes = width * 3;

  const std::vector<AreaTap> xt = buildAreaTable(width, ew);
  const std::vector<AreaTap> yt = buildAreaTable(height, eh);

  // Horizontal pass into an [height, ew, 3] float scratch, then vertical.
  std::vector<float> rows(static_cast<size_t>(height) * ew * 3);
  for (int y = 0; y < height; ++y) {
    const uint8_t* src = bgr + static_cast<size_t>(y) * stride_bytes;
    float* dst = rows.data() + static_cast<size_t>(y) * ew * 3;
    for (int x = 0; x < ew; ++x) {
      const AreaTap& t = xt[static_cast<size_t>(x)];
      float b = 0.0f, g = 0.0f, r = 0.0f;
      for (int i = 0; i < t.count; ++i) {
        const int s = std::min(t.start + i, width - 1);
        const uint8_t* px = src + static_cast<size_t>(s) * 3;
        b += t.weight[i] * px[0];
        g += t.weight[i] * px[1];
        r += t.weight[i] * px[2];
      }
      dst[x * 3 + 0] = b;
      dst[x * 3 + 1] = g;
      dst[x * 3 + 2] = r;
    }
  }

  out->assign(static_cast<size_t>(3) * eh * ew, 0.0f);
  const size_t plane = static_cast<size_t>(eh) * ew;
  for (int y = 0; y < eh; ++y) {
    const AreaTap& t = yt[static_cast<size_t>(y)];
    for (int x = 0; x < ew; ++x) {
      float acc[3] = {0.0f, 0.0f, 0.0f};
      for (int i = 0; i < t.count; ++i) {
        const int s = std::min(t.start + i, height - 1);
        const float* px = rows.data() + (static_cast<size_t>(s) * ew + x) * 3;
        acc[0] += t.weight[i] * px[0];
        acc[1] += t.weight[i] * px[1];
        acc[2] += t.weight[i] * px[2];
      }
      // BGR scratch -> planar RGB, /255, ImageNet mean/std.
      const size_t o = static_cast<size_t>(y) * ew + x;
      (*out)[0 * plane + o] = (acc[2] / 255.0f - kImageMean[0]) / kImageStd[0];
      (*out)[1 * plane + o] = (acc[1] / 255.0f - kImageMean[1]) / kImageStd[1];
      (*out)[2 * plane + o] = (acc[0] / 255.0f - kImageMean[2]) / kImageStd[2];
    }
  }
}

void preprocessRefineDepth(const float* depth_m, int width, int height,
                           int depth_stride_elems, const DepthRefineConfig& cfg,
                           std::vector<float>* out) {
  if (depth_m == nullptr || out == nullptr) throw Error(-1, "BCDL depth_refine: null depth buffer");
  const int eh = cfg.encoder_height, ew = cfg.encoder_width;
  if (width <= 0 || height <= 0 || eh <= 0 || ew <= 0) {
    throw Error(-1, "BCDL depth_refine: bad depth or encoder size");
  }
  if (depth_stride_elems <= 0) depth_stride_elems = width;

  out->assign(static_cast<size_t>(eh) * ew, 0.0f);
  const double sy = static_cast<double>(height) / eh;
  const double sx = static_cast<double>(width) / ew;
  for (int y = 0; y < eh; ++y) {
    // Nearest-neighbour, floor convention — matches both torch's
    // F.interpolate(mode='nearest') and cv::resize(INTER_NEAREST).
    const int srow = std::min(height - 1, static_cast<int>(std::floor(y * sy)));
    const float* src = depth_m + static_cast<size_t>(srow) * depth_stride_elems;
    float* dst = out->data() + static_cast<size_t>(y) * ew;
    for (int x = 0; x < ew; ++x) {
      const int scol = std::min(width - 1, static_cast<int>(std::floor(x * sx)));
      const float v = src[scol];
      // Invalid readings (missing, non-finite) go in as 0, which is upstream's
      // own convention for "no depth here".
      dst[x] = (std::isfinite(v) && v > cfg.min_valid_depth) ? std::log(v) : 0.0f;
    }
  }
}

std::vector<float> depthU16ToMetres(const uint16_t* raw, int width, int height,
                                    int stride_elems, float scale) {
  if (raw == nullptr) throw Error(-1, "BCDL depth_refine: null raw depth buffer");
  if (stride_elems <= 0) stride_elems = width;
  if (scale <= 0.0f) scale = 1.0f;
  std::vector<float> out(static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    const uint16_t* src = raw + static_cast<size_t>(y) * stride_elems;
    float* dst = out.data() + static_cast<size_t>(y) * width;
    for (int x = 0; x < width; ++x) dst[x] = static_cast<float>(src[x]) / scale;
  }
  return out;
}

RefinedDepth decodeRefinedDepth(const float* depth, const float* mask_logit, int height,
                                int width, const DepthRefineConfig& cfg) {
  if (depth == nullptr) throw Error(-1, "BCDL depth_refine: null depth output");
  RefinedDepth r;
  r.width = width;
  r.height = height;
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (width <= 0 || height <= 0) return r;
  r.depth.assign(n, 0.0f);
  r.mask.assign(n, 1);

  // sigmoid(logit) > threshold, rearranged so no exponential is needed per pixel.
  const bool has_mask = mask_logit != nullptr;
  const float t = std::min(1.0f - 1e-6f, std::max(1e-6f, cfg.mask_threshold));
  const float logit_cut = std::log(t / (1.0f - t));

  float lo = 0.0f, hi = 0.0f;
  bool any = false;
  for (size_t i = 0; i < n; ++i) {
    const bool keep = !has_mask || mask_logit[i] > logit_cut;
    r.mask[i] = keep ? 1 : 0;
    const float v = depth[i];
    const bool usable = keep && std::isfinite(v);
    r.depth[i] = (usable || !cfg.apply_mask) ? v : 0.0f;
    if (usable) {
      lo = any ? std::min(lo, v) : v;
      hi = any ? std::max(hi, v) : v;
      any = true;
    }
  }
  r.vmin = any ? lo : 0.0f;
  r.vmax = any ? hi : 0.0f;
  return r;
}

std::vector<float> depthToPointCloud(const RefinedDepth& d, const Intrinsics& k) {
  const size_t n = static_cast<size_t>(d.width) * static_cast<size_t>(d.height);
  std::vector<float> pts(n * 3, 0.0f);
  if (n == 0 || d.depth.size() < n) return pts;
  if (k.fx == 0.0f || k.fy == 0.0f) {
    throw Error(-1, "BCDL depth_refine: intrinsics have zero focal length");
  }
  for (int y = 0; y < d.height; ++y) {
    for (int x = 0; x < d.width; ++x) {
      const size_t i = static_cast<size_t>(y) * d.width + x;
      if (!d.mask.empty() && d.mask[i] == 0) continue;
      const float z = d.depth[i];
      if (!std::isfinite(z) || z <= 0.0f) continue;
      pts[i * 3 + 0] = (static_cast<float>(x) - k.cx) * z / k.fx;
      pts[i * 3 + 1] = (static_cast<float>(y) - k.cy) * z / k.fy;
      pts[i * 3 + 2] = z;
    }
  }
  return pts;
}

DepthRefiner::DepthRefiner(Engine& engine, DepthRefineConfig cfg, int image_input,
                           int depth_input, int depth_output, int mask_output)
    : engine_(engine),
      cfg_(cfg),
      image_in_(image_input),
      depth_in_(depth_input),
      depth_out_(depth_output),
      mask_out_(mask_output) {
  if (image_in_ < 0 || image_in_ >= engine_.numInputs() || depth_in_ < 0 ||
      depth_in_ >= engine_.numInputs()) {
    throw Error(-1, "BCDL depth_refine: input index out of range");
  }
  if (depth_out_ < 0 || depth_out_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL depth_refine: depth output index out of range");
  }
  if (mask_out_ >= engine_.numOutputs()) mask_out_ = -1;  // model without a mask head

  // Both inputs are fed as F32 featuremaps. A model compiled with quantized
  // inputs would need the host to quantize too, and silently copying float bytes
  // into an int8 buffer would produce garbage rather than an error.
  for (int idx : {image_in_, depth_in_}) {
    const int t = engine_.inputType(idx);
    if (t != HB_DNN_TENSOR_TYPE_F32) {
      throw Error(-1, "BCDL depth_refine: input " + std::to_string(idx) +
                          " is not F32 — compile the model with "
                          "input_type_rt=featuremap and no_preprocess");
    }
  }

  // Adopt the model's own encoder grid so a re-exported resolution just works.
  const std::vector<int> shape = engine_.inputShape(image_in_);
  if (shape.size() >= 2) {
    const int h = shape[shape.size() - 2];
    const int w = shape[shape.size() - 1];
    if (h > 1 && w > 1) {
      cfg_.encoder_height = h;
      cfg_.encoder_width = w;
    }
  }
}

RefinedDepth DepthRefiner::run(const uint8_t* bgr, int width, int height, int stride_bytes,
                               const float* depth_m, int depth_stride_elems) {
  preprocessRefineImage(bgr, width, height, stride_bytes, cfg_, &image_buf_);
  preprocessRefineDepth(depth_m, width, height, depth_stride_elems, cfg_, &depth_buf_);
  engine_.setInput(image_in_, image_buf_.data(), image_buf_.size() * sizeof(float));
  engine_.setInput(depth_in_, depth_buf_.data(), depth_buf_.size() * sizeof(float));
  engine_.infer();
  return postprocess();
}

RefinedDepth DepthRefiner::postprocess() const {
  const std::vector<int> shape = engine_.outputShape(depth_out_);
  int H = 0, W = 0;
  {
    std::vector<int> dims;
    for (int d : shape) {
      if (d > 1) dims.push_back(d);
    }
    if (dims.size() >= 2) {
      H = dims[dims.size() - 2];
      W = dims[dims.size() - 1];
    }
  }
  if (H <= 0 || W <= 0) throw Error(-1, "BCDL depth_refine: cannot resolve output HxW");

  const int64_t n = static_cast<int64_t>(H) * W;
  auto read_output = [&](int idx, std::vector<float>& into) {
    const hbDNNTensorProperties& props = engine_.outputProperties(idx);
    const auto* base = static_cast<const uint8_t*>(engine_.outputData(idx));
    into.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      into[static_cast<size_t>(i)] =
          readElement(base, props.tensorType, props.scale, props.quantiType,
                      props.quantizeAxis, props.validShape.numDimensions,
                      props.validShape.dimensionSize, props.stride, i);
    }
  };

  std::vector<float> depth;
  read_output(depth_out_, depth);
  std::vector<float> mask;
  if (mask_out_ >= 0) read_output(mask_out_, mask);

  return decodeRefinedDepth(depth.data(), mask_out_ >= 0 ? mask.data() : nullptr, H, W, cfg_);
}

}  // namespace bcdl

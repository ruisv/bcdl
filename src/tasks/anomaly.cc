#include "bcdl/tasks/anomaly.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

const float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
const float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};

/// Bilinear resize of a single-channel float map, OpenCV pixel-center convention.
std::vector<float> resizeMap(const std::vector<float>& src, int sw, int sh, int dw,
                             int dh) {
  std::vector<float> dst(static_cast<std::size_t>(dw) * dh);
  const float sx = static_cast<float>(sw) / static_cast<float>(dw);
  const float sy = static_cast<float>(sh) / static_cast<float>(dh);
  for (int y = 0; y < dh; ++y) {
    float fy = (y + 0.5f) * sy - 0.5f;
    fy = std::clamp(fy, 0.0f, static_cast<float>(sh - 1));
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - static_cast<float>(y0);
    for (int x = 0; x < dw; ++x) {
      float fx = (x + 0.5f) * sx - 0.5f;
      fx = std::clamp(fx, 0.0f, static_cast<float>(sw - 1));
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - static_cast<float>(x0);
      const float v00 = src[static_cast<std::size_t>(y0) * sw + x0];
      const float v01 = src[static_cast<std::size_t>(y0) * sw + x1];
      const float v10 = src[static_cast<std::size_t>(y1) * sw + x0];
      const float v11 = src[static_cast<std::size_t>(y1) * sw + x1];
      dst[static_cast<std::size_t>(y) * dw + x] =
          (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11);
    }
  }
  return dst;
}

/// Separable Gaussian blur with edge replication — the border handling
/// cv::GaussianBlur defaults to, so a map smoothed here matches one smoothed
/// there rather than darkening at the frame edge.
void gaussianBlur(std::vector<float>& img, int w, int h, int ksize, float sigma) {
  if (ksize <= 1 || sigma <= 0.0f) return;
  if ((ksize & 1) == 0) ++ksize;
  const int r = ksize / 2;
  std::vector<float> k(ksize);
  float sum = 0.0f;
  for (int i = 0; i < ksize; ++i) {
    const float d = static_cast<float>(i - r);
    k[i] = std::exp(-(d * d) / (2.0f * sigma * sigma));
    sum += k[i];
  }
  for (float& v : k) v /= sum;

  std::vector<float> tmp(img.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int i = -r; i <= r; ++i) {
        const int xx = std::clamp(x + i, 0, w - 1);
        acc += k[i + r] * img[static_cast<std::size_t>(y) * w + xx];
      }
      tmp[static_cast<std::size_t>(y) * w + x] = acc;
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int i = -r; i <= r; ++i) {
        const int yy = std::clamp(y + i, 0, h - 1);
        acc += k[i + r] * tmp[static_cast<std::size_t>(yy) * w + x];
      }
      img[static_cast<std::size_t>(y) * w + x] = acc;
    }
  }
}

}  // namespace

AnomalyMap decodeAnomaly(const float* data, const std::vector<int>& shape,
                         const AnomalyConfig& cfg) {
  if (data == nullptr) {
    throw Error(-1, "BCDL decodeAnomaly: null tensor");
  }
  std::vector<int> s = shape;
  while (s.size() > 2 && s.front() == 1) s.erase(s.begin());
  if (s.size() != 2 || s[0] <= 0 || s[1] <= 0) {
    throw Error(-1, "BCDL decodeAnomaly: expected a single-channel {1,H,W} / {H,W} "
                    "score grid");
  }
  const int gh = s[0], gw = s[1];

  AnomalyMap m;
  m.width = cfg.out_width > 0 ? cfg.out_width : gw;
  m.height = cfg.out_height > 0 ? cfg.out_height : gh;
  std::vector<float> grid(data, data + static_cast<std::size_t>(gw) * gh);
  m.data = (m.width == gw && m.height == gh) ? std::move(grid)
                                             : resizeMap(grid, gw, gh, m.width, m.height);
  gaussianBlur(m.data, m.width, m.height, cfg.blur_kernel, cfg.blur_sigma);

  m.score = m.data.empty() ? 0.0f : *std::max_element(m.data.begin(), m.data.end());
  return m;
}

std::vector<uint8_t> anomalyMask(const AnomalyMap& map, float threshold) {
  std::vector<uint8_t> mask(map.data.size());
  for (std::size_t i = 0; i < map.data.size(); ++i) {
    mask[i] = map.data[i] >= threshold ? 255 : 0;
  }
  return mask;
}

void anomalyPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                       int in_h, std::vector<float>& out, const float mean[3],
                       const float stddev[3]) {
  if (!bgr || width <= 0 || height <= 0 || in_w <= 0 || in_h <= 0) {
    throw Error(-1, "BCDL anomalyPreprocess: bad image or model dimensions");
  }
  if (stride <= 0) stride = width * 3;
  const float* mu = mean ? mean : kImagenetMean;
  const float* sd = stddev ? stddev : kImagenetStd;
  for (int c = 0; c < 3; ++c) {
    if (sd[c] == 0.0f) {
      throw Error(-1, "BCDL anomalyPreprocess: zero standard deviation");
    }
  }

  out.assign(static_cast<std::size_t>(3) * in_w * in_h, 0.0f);
  const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;
  const float sx = static_cast<float>(width) / static_cast<float>(in_w);
  const float sy = static_cast<float>(height) / static_cast<float>(in_h);

  for (int oy = 0; oy < in_h; ++oy) {
    float fy = (oy + 0.5f) * sy - 0.5f;
    fy = std::clamp(fy, 0.0f, static_cast<float>(height - 1));
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, height - 1);
    const float wy = fy - static_cast<float>(y0);

    for (int ox = 0; ox < in_w; ++ox) {
      float fx = (ox + 0.5f) * sx - 0.5f;
      fx = std::clamp(fx, 0.0f, static_cast<float>(width - 1));
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, width - 1);
      const float wx = fx - static_cast<float>(x0);

      const uint8_t* r0 = bgr + static_cast<std::size_t>(y0) * stride;
      const uint8_t* r1 = bgr + static_cast<std::size_t>(y1) * stride;
      const uint8_t* p00 = r0 + static_cast<std::size_t>(x0) * 3;
      const uint8_t* p01 = r0 + static_cast<std::size_t>(x1) * 3;
      const uint8_t* p10 = r1 + static_cast<std::size_t>(x0) * 3;
      const uint8_t* p11 = r1 + static_cast<std::size_t>(x1) * 3;
      const float w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx;
      const float w10 = wy * (1 - wx), w11 = wy * wx;

      const std::size_t o = static_cast<std::size_t>(oy) * in_w + ox;
      for (int c = 0; c < 3; ++c) {
        const float v = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
        // BGR source -> RGB planes, /255, then the per-channel statistics.
        out[static_cast<std::size_t>(2 - c) * plane + o] =
            (v / 255.0f - mu[2 - c]) / sd[2 - c];
      }
    }
  }
}

AnomalyDetector::AnomalyDetector(Engine& engine, AnomalyConfig cfg, int output_index,
                                 int input_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index), in_idx_(input_index) {
  const std::vector<int> s = engine.inputShape(in_idx_);
  if (s.size() != 4) {
    throw Error(-1, "BCDL AnomalyDetector: input " + std::to_string(in_idx_) +
                        " is not a 4-D image tensor");
  }
  in_h_ = s[2];
  in_w_ = s[3];
  if (in_w_ <= 0 || in_h_ <= 0) {
    throw Error(-1, "BCDL AnomalyDetector: model input has an empty spatial size");
  }
  // Default the map to the input resolution: a raw 36x36 grid is not something a
  // caller can overlay on the part photo without redoing the reference upsample.
  if (cfg_.out_width <= 0) cfg_.out_width = in_w_;
  if (cfg_.out_height <= 0) cfg_.out_height = in_h_;
}

AnomalyMap AnomalyDetector::detect(const uint8_t* bgr, int width, int height, int stride,
                                   int timeout_ms) {
  anomalyPreprocess(bgr, width, height, stride, in_w_, in_h_, buf_);
  setInputFromFloat(engine_, in_idx_, buf_.data(), buf_.size(), stage_);
  engine_.infer(timeout_ms);
  return postprocess();
}

AnomalyMap AnomalyDetector::postprocess() const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* out = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decodeAnomaly(out, shape, cfg_);
}

}  // namespace bcdl

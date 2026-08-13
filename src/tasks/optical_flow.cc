#include "bcdl/tasks/optical_flow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// Middlebury colour wheel: 55 hues in six ramps. Built once, shared.
const std::vector<std::array<uint8_t, 3>>& colorWheel() {
  static const std::vector<std::array<uint8_t, 3>> wheel = [] {
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    std::vector<std::array<uint8_t, 3>> w;
    w.reserve(RY + YG + GC + CB + BM + MR);
    auto ramp = [](int i, int n) { return static_cast<uint8_t>(255 * i / n); };
    for (int i = 0; i < RY; ++i) w.push_back({255, ramp(i, RY), 0});
    for (int i = 0; i < YG; ++i) w.push_back({static_cast<uint8_t>(255 - ramp(i, YG)), 255, 0});
    for (int i = 0; i < GC; ++i) w.push_back({0, 255, ramp(i, GC)});
    for (int i = 0; i < CB; ++i) w.push_back({0, static_cast<uint8_t>(255 - ramp(i, CB)), 255});
    for (int i = 0; i < BM; ++i) w.push_back({ramp(i, BM), 0, 255});
    for (int i = 0; i < MR; ++i) w.push_back({255, 0, static_cast<uint8_t>(255 - ramp(i, MR))});
    return w;
  }();
  return wheel;
}

/// Strip a leading batch dim of 1 and validate the rest.
std::vector<int> squeezeBatch(const std::vector<int>& shape) {
  std::vector<int> s = shape;
  if (s.size() == 4 && s[0] == 1) s.erase(s.begin());
  return s;
}

}  // namespace

FlowField decodeFlow(const float* data, const std::vector<int>& shape,
                     const FlowConfig& cfg) {
  if (data == nullptr) {
    throw Error(-1, "BCDL decodeFlow: null tensor");
  }
  const std::vector<int> s = squeezeBatch(shape);
  if (s.size() != 3) {
    throw Error(-1, "BCDL decodeFlow: expected {1,2,H,W} or {1,H,W,2}, got a rank-" +
                        std::to_string(shape.size()) + " tensor");
  }
  int h = 0, w = 0;
  if (cfg.channels_first) {
    if (s[0] != 2) {
      throw Error(-1, "BCDL decodeFlow: channels_first expects 2 channels, got " +
                          std::to_string(s[0]));
    }
    h = s[1];
    w = s[2];
  } else {
    if (s[2] != 2) {
      throw Error(-1, "BCDL decodeFlow: channels_last expects 2 channels, got " +
                          std::to_string(s[2]));
    }
    h = s[0];
    w = s[1];
  }
  if (h <= 0 || w <= 0) {
    throw Error(-1, "BCDL decodeFlow: empty flow field");
  }

  FlowField f;
  f.width = w;
  f.height = h;
  f.data.resize(static_cast<std::size_t>(w) * h * 2);
  const std::size_t plane = static_cast<std::size_t>(w) * h;
  for (std::size_t i = 0; i < plane; ++i) {
    const float u = cfg.channels_first ? data[i] : data[i * 2];
    const float v = cfg.channels_first ? data[plane + i] : data[i * 2 + 1];
    f.data[i * 2] = u * cfg.scale_x;
    f.data[i * 2 + 1] = v * cfg.scale_y;
  }
  return f;
}

std::vector<uint8_t> flowColorize(const FlowField& flow, float max_magnitude) {
  if (flow.width <= 0 || flow.height <= 0) {
    throw Error(-1, "BCDL flowColorize: empty field");
  }
  const std::size_t plane = static_cast<std::size_t>(flow.width) * flow.height;

  float norm = max_magnitude;
  if (norm <= 0.0f) {
    // The 99th percentile rather than the max: one fast object (or one bad
    // pixel) otherwise sets the scale and flattens the rest of the frame to
    // near-grey.
    std::vector<float> mags(plane);
    for (std::size_t i = 0; i < plane; ++i) {
      mags[i] = std::hypot(flow.data[i * 2], flow.data[i * 2 + 1]);
    }
    const std::size_t k = static_cast<std::size_t>(0.99 * (plane - 1));
    std::nth_element(mags.begin(), mags.begin() + k, mags.end());
    norm = mags[k];
  }
  if (!(norm > 0.0f)) norm = 1.0f;

  const auto& wheel = colorWheel();
  const int ncols = static_cast<int>(wheel.size());
  std::vector<uint8_t> bgr(plane * 3);
  for (std::size_t i = 0; i < plane; ++i) {
    const float u = flow.data[i * 2] / norm;
    const float v = flow.data[i * 2 + 1] / norm;
    const float mag = std::min(std::hypot(u, v), 1.0f);
    const float ang = std::atan2(-v, -u) / static_cast<float>(M_PI);  // [-1,1]
    const float fk = (ang + 1.0f) / 2.0f * (ncols - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % ncols;
    const float f = fk - static_cast<float>(k0);
    for (int c = 0; c < 3; ++c) {
      const float c0 = wheel[k0][c] / 255.0f;
      const float c1 = wheel[k1][c] / 255.0f;
      float col = (1.0f - f) * c0 + f * c1;
      // Toward white at the centre of the wheel, as the reference viz does.
      col = 1.0f - mag * (1.0f - col);
      bgr[i * 3 + (2 - c)] = static_cast<uint8_t>(std::lround(255.0f * col));
    }
  }
  return bgr;
}

float flowEndpointError(const FlowField& a, const FlowField& b) {
  if (a.width != b.width || a.height != b.height) {
    throw Error(-1, "BCDL flowEndpointError: field sizes differ");
  }
  const std::size_t plane = static_cast<std::size_t>(a.width) * a.height;
  if (plane == 0) return 0.0f;
  double sum = 0.0;
  for (std::size_t i = 0; i < plane; ++i) {
    sum += std::hypot(a.data[i * 2] - b.data[i * 2], a.data[i * 2 + 1] - b.data[i * 2 + 1]);
  }
  return static_cast<float>(sum / static_cast<double>(plane));
}

void flowPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                    int in_h, std::vector<float>& out) {
  if (!bgr || width <= 0 || height <= 0 || in_w <= 0 || in_h <= 0) {
    throw Error(-1, "BCDL flowPreprocess: bad image or model dimensions");
  }
  if (stride <= 0) stride = width * 3;

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
      const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
      const float w10 = wy * (1.0f - wx), w11 = wy * wx;

      const std::size_t o = static_cast<std::size_t>(oy) * in_w + ox;
      for (int c = 0; c < 3; ++c) {
        // BGR stays BGR and 0-255 stays 0-255: the graph normalises internally.
        out[static_cast<std::size_t>(c) * plane + o] =
            w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
      }
    }
  }
}

OpticalFlowEstimator::OpticalFlowEstimator(Engine& engine, FlowConfig cfg,
                                           int output_index, int input0_index,
                                           int input1_index)
    : engine_(engine),
      cfg_(cfg),
      out_idx_(output_index),
      in0_idx_(input0_index),
      in1_idx_(input1_index) {
  const std::vector<int> s = engine.inputShape(in0_idx_);
  if (s.size() != 4) {
    throw Error(-1, "BCDL OpticalFlowEstimator: input " + std::to_string(in0_idx_) +
                        " is not a 4-D image tensor");
  }
  // NCHW is what the flow exports use; NHWC would put 3 last.
  in_h_ = s[2];
  in_w_ = s[3];
  if (in_w_ <= 0 || in_h_ <= 0) {
    throw Error(-1, "BCDL OpticalFlowEstimator: model input has an empty spatial size");
  }
}

FlowField OpticalFlowEstimator::estimate(const uint8_t* bgr0, const uint8_t* bgr1,
                                         int width, int height, int stride,
                                         int timeout_ms) {
  flowPreprocess(bgr0, width, height, stride, in_w_, in_h_, buf0_);
  flowPreprocess(bgr1, width, height, stride, in_w_, in_h_, buf1_);
  setInputFromFloat(engine_, in0_idx_, buf0_.data(), buf0_.size(), stage0_);
  setInputFromFloat(engine_, in1_idx_, buf1_.data(), buf1_.size(), stage1_);
  engine_.infer(timeout_ms);

  // Vectors come back in model pixels; the caller asked about source pixels.
  FlowConfig cfg = cfg_;
  cfg.scale_x = static_cast<float>(width) / static_cast<float>(in_w_);
  cfg.scale_y = static_cast<float>(height) / static_cast<float>(in_h_);

  std::vector<float> scratch;
  std::vector<int> shape;
  const float* out = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decodeFlow(out, shape, cfg);
}

FlowField OpticalFlowEstimator::postprocess() const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* out = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decodeFlow(out, shape, cfg_);
}

}  // namespace bcdl

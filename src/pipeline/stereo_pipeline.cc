#include "bcdl/pipeline/stereo_pipeline.h"

#ifdef BCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // cv::resize
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "hobot/dnn/hb_dnn.h"

namespace bcdl {
namespace {

/// Resolve (H,W) for an NCHW float input tensor shape: {1,3,H,W} (or {3,H,W} /
/// {H,W} as degenerate fallbacks). The stereo models we target are NCHW.
void resolveInputHW(const std::vector<int>& shape, int& H, int& W) {
  const int n = static_cast<int>(shape.size());
  if (n >= 2) {
    H = shape[n - 2];
    W = shape[n - 1];
  } else if (n == 1) {
    H = 1;
    W = shape[0];
  } else {
    H = W = 0;
  }
}

/// Horizontally flip an interleaved HxWx3 uint8 frame into `dst` (same size).
void flipH3(const uint8_t* src, int w, int h, std::vector<uint8_t>& dst) {
  dst.resize(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y) {
    const uint8_t* srow = src + static_cast<size_t>(y) * w * 3;
    uint8_t* drow = dst.data() + static_cast<size_t>(y) * w * 3;
    for (int x = 0; x < w; ++x) {
      const uint8_t* s = srow + static_cast<size_t>(x) * 3;
      uint8_t* d = drow + static_cast<size_t>(w - 1 - x) * 3;
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
  }
}

/// Horizontally flip a row-major HxW float disparity map (values unchanged — a
/// right-view disparity is already in right-image coordinates after the flip).
DepthMap flipHDisp(const DepthMap& m) {
  DepthMap out;
  out.width = m.width;
  out.height = m.height;
  out.vmin = m.vmin;
  out.vmax = m.vmax;
  out.data.resize(m.data.size());
  for (int y = 0; y < m.height; ++y) {
    const float* srow = m.data.data() + static_cast<size_t>(y) * m.width;
    float* drow = out.data.data() + static_cast<size_t>(y) * m.width;
    for (int x = 0; x < m.width; ++x) drow[m.width - 1 - x] = srow[x];
  }
  return out;
}

}  // namespace

void packStereoInputCHW(const uint8_t* bgr, int width, int height, int out_h,
                        int out_w, StereoFit fit, bool to_rgb, float* dst) {
  if (out_h <= 0 || out_w <= 0 || width <= 0 || height <= 0) return;
  const int64_t plane = static_cast<int64_t>(out_h) * out_w;
  // Plane order: to_rgb => dst plane 0=R,1=G,2=B; else B,G,R. The source pixel
  // is BGR (b=p[0],g=p[1],r=p[2]); src_ch[c] picks the source byte for dst plane c.
  const int src_ch[3] = {to_rgb ? 2 : 0, 1, to_rgb ? 0 : 2};

  // Pixels not covered by the source map to 0 (crop larger than the frame).
  std::fill(dst, dst + 3 * plane, 0.0f);

  if (fit == StereoFit::kCrop) {
    // Center crop: source pixel (sy,sx) = (top+y, left+x).
    const int top = (height - out_h) / 2;
    const int left = (width - out_w) / 2;
    for (int y = 0; y < out_h; ++y) {
      const int sy = top + y;
      if (sy < 0 || sy >= height) continue;
      const uint8_t* srow = bgr + static_cast<size_t>(sy) * width * 3;
      for (int x = 0; x < out_w; ++x) {
        const int sx = left + x;
        if (sx < 0 || sx >= width) continue;
        const uint8_t* s = srow + static_cast<size_t>(sx) * 3;
        const int64_t o = static_cast<int64_t>(y) * out_w + x;
        dst[0 * plane + o] = static_cast<float>(s[src_ch[0]]);
        dst[1 * plane + o] = static_cast<float>(s[src_ch[1]]);
        dst[2 * plane + o] = static_cast<float>(s[src_ch[2]]);
      }
    }
    return;
  }

  // Anamorphic resize to (out_w,out_h).
#ifdef BCDL_HAVE_OPENCV
  cv::Mat srcMat(height, width, CV_8UC3, const_cast<uint8_t*>(bgr),
                 static_cast<size_t>(width) * 3);
  cv::Mat dstMat;
  cv::resize(srcMat, dstMat, cv::Size(out_w, out_h), 0.0, 0.0, cv::INTER_LINEAR);
  for (int y = 0; y < out_h; ++y) {
    const uint8_t* drow = dstMat.ptr<uint8_t>(y);
    for (int x = 0; x < out_w; ++x) {
      const uint8_t* s = drow + static_cast<size_t>(x) * 3;
      const int64_t o = static_cast<int64_t>(y) * out_w + x;
      dst[0 * plane + o] = static_cast<float>(s[src_ch[0]]);
      dst[1 * plane + o] = static_cast<float>(s[src_ch[1]]);
      dst[2 * plane + o] = static_cast<float>(s[src_ch[2]]);
    }
  }
#else
  // Hand bilinear, OpenCV pixel-center convention (matches cv::resize / numpy ref).
  const float sx_ratio = static_cast<float>(width) / out_w;
  const float sy_ratio = static_cast<float>(height) / out_h;
  for (int y = 0; y < out_h; ++y) {
    float fy = (y + 0.5f) * sy_ratio - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > height - 1.0f) fy = static_cast<float>(height - 1);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, height - 1);
    const float wy = fy - y0;
    const uint8_t* r0 = bgr + static_cast<size_t>(y0) * width * 3;
    const uint8_t* r1 = bgr + static_cast<size_t>(y1) * width * 3;
    for (int x = 0; x < out_w; ++x) {
      float fx = (x + 0.5f) * sx_ratio - 0.5f;
      if (fx < 0.0f) fx = 0.0f;
      if (fx > width - 1.0f) fx = static_cast<float>(width - 1);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, width - 1);
      const float wx = fx - x0;
      const int64_t o = static_cast<int64_t>(y) * out_w + x;
      for (int c = 0; c < 3; ++c) {
        const int sc = src_ch[c];
        const float v00 = r0[x0 * 3 + sc], v01 = r0[x1 * 3 + sc];
        const float v10 = r1[x0 * 3 + sc], v11 = r1[x1 * 3 + sc];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        dst[c * plane + o] = top + (bot - top) * wy;
      }
    }
  }
#endif
}

std::vector<float> disparityToDepth(const DepthMap& disp, float fx,
                                    float baseline) {
  const int64_t n = static_cast<int64_t>(disp.width) * disp.height;
  std::vector<float> depth(static_cast<size_t>(std::max<int64_t>(0, n)), 0.0f);
  const float k = fx * baseline;
  for (int64_t i = 0; i < n; ++i) {
    const float d = disp.data[static_cast<size_t>(i)];
    depth[static_cast<size_t>(i)] = d > 1e-3f ? k / d : 0.0f;
  }
  return depth;
}

std::vector<uint8_t> stereoValidMask(const DepthMap& disp, float disp_min,
                                     float max_disp, int left_margin,
                                     const DepthMap* disp_right,
                                     float lr_thresh) {
  const int W = disp.width, H = disp.height;
  const int64_t n = static_cast<int64_t>(W) * H;
  std::vector<uint8_t> mask(static_cast<size_t>(std::max<int64_t>(0, n)), 0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int64_t o = static_cast<int64_t>(y) * W + x;
      const float d = disp.data[static_cast<size_t>(o)];
      bool ok = (d >= disp_min) && (d <= max_disp);     // disparity range
      if (ok) ok = (static_cast<float>(x) - d >= 0.0f);  // right column in-frame
      if (ok && x < left_margin) ok = false;             // left margin
      if (ok && disp_right != nullptr) {                 // left-right consistency
        int xr = static_cast<int>(std::lround(x - d));
        xr = std::min(std::max(xr, 0), W - 1);
        const float dR = disp_right->data[static_cast<size_t>(y) * W + xr];
        if (std::fabs(d - dR) > lr_thresh) ok = false;
      }
      mask[static_cast<size_t>(o)] = ok ? 1 : 0;
    }
  }
  return mask;
}

StereoPipeline::StereoPipeline(Engine& engine, StereoConfig cfg)
    : engine_(engine), cfg_(cfg) {
  const int ni = engine_.numInputs();
  if (cfg_.left_index < 0 || cfg_.left_index >= ni || cfg_.right_index < 0 ||
      cfg_.right_index >= ni || cfg_.left_index == cfg_.right_index) {
    throw Error(-1,
                "StereoPipeline: model needs two distinct float inputs "
                "(left_index/right_index); got " +
                    std::to_string(ni) + " input(s)");
  }
  for (int idx : {cfg_.left_index, cfg_.right_index}) {
    if (engine_.inputType(idx) != HB_DNN_TENSOR_TYPE_F32) {
      throw Error(-1,
                  "StereoPipeline: input " + std::to_string(idx) +
                      " is not F32 — this pipeline feeds F32 featuremap inputs "
                      "(norm fused into the .hbm)");
    }
  }
  if (cfg_.output_index < 0 || cfg_.output_index >= engine_.numOutputs()) {
    throw Error(-1, "StereoPipeline: output_index out of range");
  }

  if (cfg_.input_w <= 0 || cfg_.input_h <= 0) {
    int H = 0, W = 0;
    resolveInputHW(engine_.inputShape(cfg_.left_index), H, W);
    if (cfg_.input_w <= 0) cfg_.input_w = W;
    if (cfg_.input_h <= 0) cfg_.input_h = H;
  }
  if (cfg_.input_w <= 0 || cfg_.input_h <= 0) {
    throw Error(-1, "StereoPipeline: could not resolve model input H/W");
  }

  // The model expects F32 [1,3,H,W]; verify the input buffer is large enough.
  const std::size_t need =
      static_cast<std::size_t>(3) * cfg_.input_h * cfg_.input_w * sizeof(float);
  if (engine_.inputBytes(cfg_.left_index) < need ||
      engine_.inputBytes(cfg_.right_index) < need) {
    throw Error(-1,
                "StereoPipeline: input buffer smaller than 3*H*W floats — is "
                "this a 3-channel NCHW stereo model?");
  }

  left_buf_.assign(static_cast<size_t>(3) * cfg_.input_h * cfg_.input_w, 0.0f);
  right_buf_.assign(static_cast<size_t>(3) * cfg_.input_h * cfg_.input_w, 0.0f);
}

DepthMap StereoPipeline::inferDisparity(const uint8_t* left, const uint8_t* right,
                                        int width, int height) {
  packStereoInputCHW(left, width, height, cfg_.input_h, cfg_.input_w, cfg_.fit,
                     cfg_.to_rgb, left_buf_.data());
  packStereoInputCHW(right, width, height, cfg_.input_h, cfg_.input_w, cfg_.fit,
                     cfg_.to_rgb, right_buf_.data());
  engine_.setInput(cfg_.left_index, left_buf_.data(),
                   left_buf_.size() * sizeof(float));
  engine_.setInput(cfg_.right_index, right_buf_.data(),
                   right_buf_.size() * sizeof(float));
  engine_.infer();

  DepthConfig dc;
  dc.normalize = false;  // keep raw disparity values
  DepthEstimator est(engine_, dc, cfg_.output_index);
  return est.postprocess();
}

StereoResult StereoPipeline::process(const uint8_t* left, const uint8_t* right,
                                     int width, int height) {
  StereoResult out;
  out.disparity = inferDisparity(left, right, width, height);

  if (cfg_.fx > 0.0f && cfg_.baseline > 0.0f) {
    out.depth = disparityToDepth(out.disparity, cfg_.fx, cfg_.baseline);
  }

  if (cfg_.valid_mask) {
    DepthMap disp_right;
    const DepthMap* dr = nullptr;
    if (cfg_.lr_check) {
      // Right-view disparity: feed horizontally-flipped frames in swapped order
      // (flipped-right as left, flipped-left as right), then flip the result
      // back. flipH(fit(x)) == fit(flipH(x)) for resize and center-crop, so we
      // flip the raw source. Mirrors the LAS2 infer.py reference.
      std::vector<uint8_t> fl, fr;
      flipH3(left, width, height, fl);
      flipH3(right, width, height, fr);
      DepthMap flipped = inferDisparity(fr.data(), fl.data(), width, height);
      disp_right = flipHDisp(flipped);
      dr = &disp_right;
    }
    out.valid = stereoValidMask(out.disparity, cfg_.disp_min, cfg_.max_disp,
                                cfg_.left_margin, dr, cfg_.lr_thresh);
  }
  return out;
}

}  // namespace bcdl

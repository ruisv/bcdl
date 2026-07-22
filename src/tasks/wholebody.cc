#include "bcdl/tasks/wholebody.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// OpenCV's ksize -> sigma rule for GaussianBlur(sigma=0), so the DARK
/// modulation blurs with exactly the kernel the reference decoder used.
float sigmaForKernel(int k) { return 0.3f * ((k - 1) * 0.5f - 1.0f) + 0.8f; }

std::vector<float> gaussianKernel(int k) {
  const float sigma = sigmaForKernel(k);
  const int r = k / 2;
  std::vector<float> w(k);
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    const float d = static_cast<float>(i - r);
    w[i] = std::exp(-(d * d) / (2.0f * sigma * sigma));
    sum += w[i];
  }
  for (float& v : w) v /= sum;
  return w;
}

/// BORDER_REFLECT_101 index folding (cv::GaussianBlur's default): the edge pixel
/// is not repeated, so -1 -> 1 and n -> n-2.
int reflect101(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
  }
  return i;
}

/// The 3x3 log-of-blurred-heatmap neighbourhood around (cx, cy).
///
/// The reference blurs all 133 maps in full and then reads seven cells around
/// each argmax. Blurring only what is read is ~300x less arithmetic per person
/// and gives identical numbers: a separable 3x3-of-blur needs just a
/// (3+2r)x(3+2r) source window.
///
/// TWO BORDER CONVENTIONS meet here and they are not the same one:
///   - inside the blur, cv::GaussianBlur reflects (BORDER_REFLECT_101);
///   - reading the 3x3 around a peak ON the border, the reference has already
///     edge-padded the log map, i.e. the index is CLAMPED.
void logBlurNeighbourhood(const float* hm, int h, int w, const std::vector<float>& kern, int cx,
                          int cy, float out[9]) {
  const int r = static_cast<int>(kern.size()) / 2;
  // Horizontal pass over the rows the vertical pass will need.
  const int tw = 3, th = 3 + 2 * r;
  std::vector<float> tmp(static_cast<std::size_t>(tw) * th);
  for (int ty = 0; ty < th; ++ty) {
    const int sy = reflect101(cy - 1 - r + ty, h);
    for (int tx = 0; tx < tw; ++tx) {
      float acc = 0.0f;
      for (int i = 0; i < static_cast<int>(kern.size()); ++i) {
        const int sx = reflect101(cx - 1 + tx - r + i, w);
        acc += kern[i] * hm[static_cast<std::size_t>(sy) * w + sx];
      }
      tmp[static_cast<std::size_t>(ty) * tw + tx] = acc;
    }
  }
  for (int dy = 0; dy < 3; ++dy) {
    for (int dx = 0; dx < 3; ++dx) {
      // Clamp, not reflect: the reference reads an edge-padded log map here.
      const int px = std::clamp(cx - 1 + dx, 0, w - 1);
      const int py = std::clamp(cy - 1 + dy, 0, h - 1);
      float acc = 0.0f;
      for (int i = 0; i < static_cast<int>(kern.size()); ++i) {
        acc += kern[i] * tmp[static_cast<std::size_t>(py - cy + 1 + i) * tw + (px - cx + 1)];
      }
      out[dy * 3 + dx] = std::log(std::clamp(acc, 0.001f, 50.0f));
    }
  }
}

}  // namespace

WholeBodyCrop wholeBodyPreprocess(const uint8_t* bgr, int width, int height, int stride,
                                  float bx1, float by1, float bx2, float by2, int in_w,
                                  int in_h, const WholeBodyConfig& cfg,
                                  std::vector<float>& out) {
  if (!bgr || width <= 0 || height <= 0 || in_w <= 0 || in_h <= 0) {
    throw Error(-1, "BCDL: wholeBodyPreprocess: bad image or model dimensions");
  }

  WholeBodyCrop c;
  c.x1 = std::clamp(static_cast<int>(std::lround(bx1)) - cfg.box_pad, 0, width);
  c.y1 = std::clamp(static_cast<int>(std::lround(by1)) - cfg.box_pad, 0, height);
  const int x2 = std::clamp(static_cast<int>(std::lround(bx2)) + cfg.box_pad, 0, width);
  const int y2 = std::clamp(static_cast<int>(std::lround(by2)) + cfg.box_pad, 0, height);
  const int cw = x2 - c.x1, ch = y2 - c.y1;
  if (cw <= 0 || ch <= 0) {
    throw Error(-1, "BCDL: wholeBodyPreprocess: person box is empty after clipping");
  }

  // Zero-pad, centred, to the model's aspect ratio. Truncating to int here (not
  // rounding) is what the reference does, and the pad offsets feed the inverse
  // map, so a half-pixel disagreement would shift every keypoint.
  const float aspect = static_cast<float>(in_w) / static_cast<float>(in_h);
  if (static_cast<float>(cw) / static_cast<float>(ch) < aspect) {
    c.padded_w = static_cast<int>(aspect * ch);
    c.padded_h = ch;
    c.pad_left = (c.padded_w - cw) / 2;
  } else {
    c.padded_w = cw;
    c.padded_h = static_cast<int>(cw / aspect);
    c.pad_top = (c.padded_h - ch) / 2;
  }

  out.assign(static_cast<std::size_t>(3) * in_w * in_h, 0.0f);
  const float sx_ratio = static_cast<float>(c.padded_w) / static_cast<float>(in_w);
  const float sy_ratio = static_cast<float>(c.padded_h) / static_cast<float>(in_h);
  const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;

  for (int oy = 0; oy < in_h; ++oy) {
    float fy = (oy + 0.5f) * sy_ratio - 0.5f;
    fy = std::clamp(fy, 0.0f, static_cast<float>(c.padded_h - 1));
    const int py0 = static_cast<int>(fy);
    const int py1 = std::min(py0 + 1, c.padded_h - 1);
    const float wy = fy - py0;

    for (int ox = 0; ox < in_w; ++ox) {
      float fx = (ox + 0.5f) * sx_ratio - 0.5f;
      fx = std::clamp(fx, 0.0f, static_cast<float>(c.padded_w - 1));
      const int px0 = static_cast<int>(fx);
      const int px1 = std::min(px0 + 1, c.padded_w - 1);
      const float wx = fx - px0;

      // Sample the padded crop without materializing it: anything outside the
      // source rectangle is the zero padding.
      float acc[3] = {0.0f, 0.0f, 0.0f};
      const int pys[2] = {py0, py1};
      const int pxs[2] = {px0, px1};
      const float wys[2] = {1.0f - wy, wy};
      const float wxs[2] = {1.0f - wx, wx};
      for (int j = 0; j < 2; ++j) {
        const int sy = c.y1 + pys[j] - c.pad_top;
        if (sy < c.y1 || sy >= c.y1 + ch) continue;
        for (int i = 0; i < 2; ++i) {
          const int sx = c.x1 + pxs[i] - c.pad_left;
          if (sx < c.x1 || sx >= c.x1 + cw) continue;
          const uint8_t* p = bgr + static_cast<std::size_t>(sy) * stride + static_cast<std::size_t>(sx) * 3;
          const float wgt = wys[j] * wxs[i];
          acc[0] += wgt * p[2];  // R (source is BGR)
          acc[1] += wgt * p[1];  // G
          acc[2] += wgt * p[0];  // B
        }
      }

      const std::size_t o = static_cast<std::size_t>(oy) * in_w + ox;
      for (int ch_i = 0; ch_i < 3; ++ch_i) {
        out[ch_i * plane + o] = (acc[ch_i] / 255.0f - cfg.mean[ch_i]) / cfg.std[ch_i];
      }
    }
  }
  return c;
}

std::vector<Keypoint> decodeWholeBody(const float* heatmaps, int num_kpts, int hm_h, int hm_w,
                                      const WholeBodyCrop& crop, const WholeBodyConfig& cfg) {
  if (!heatmaps || num_kpts <= 0 || hm_h <= 0 || hm_w <= 0) {
    throw Error(-1, "BCDL: decodeWholeBody: bad heatmap dimensions");
  }
  if (hm_w < 2 || hm_h < 2) {
    throw Error(-1, "BCDL: decodeWholeBody: heatmap must be at least 2x2");
  }
  if (cfg.blur_kernel < 3 || cfg.blur_kernel % 2 == 0) {
    throw Error(-1, "BCDL: decodeWholeBody: blur_kernel must be odd and >= 3, got " +
                        std::to_string(cfg.blur_kernel));
  }

  const std::vector<float> kern = gaussianKernel(cfg.blur_kernel);
  const std::size_t plane = static_cast<std::size_t>(hm_h) * hm_w;

  // UDP maps the heatmap onto the padded crop by its EXTREME cells, not its cell
  // count: the last cell centre sits on the last pixel, hence (size - 1).
  const float scale_x = static_cast<float>(crop.padded_w) / static_cast<float>(hm_w - 1);
  const float scale_y = static_cast<float>(crop.padded_h) / static_cast<float>(hm_h - 1);
  // center - scale*0.5, with center from INTEGER division: zero for an even
  // extent and -0.5 for an odd one. Small, but it is what the reference does.
  const float off_x = static_cast<float>(crop.padded_w / 2) - crop.padded_w * 0.5f;
  const float off_y = static_cast<float>(crop.padded_h / 2) - crop.padded_h * 0.5f;

  std::vector<Keypoint> kpts(static_cast<std::size_t>(num_kpts));
  for (int k = 0; k < num_kpts; ++k) {
    const float* hm = heatmaps + k * plane;

    std::size_t best = 0;
    float maxval = hm[0];
    for (std::size_t i = 1; i < plane; ++i) {
      if (hm[i] > maxval) {
        maxval = hm[i];
        best = i;
      }
    }

    float x, y;
    if (maxval > 0.0f) {
      const int cx = static_cast<int>(best % hm_w);
      const int cy = static_cast<int>(best / hm_w);
      x = static_cast<float>(cx);
      y = static_cast<float>(cy);

      // DARK: one Newton step on the log of the blurred map.
      float n[9];
      logBlurNeighbourhood(hm, hm_h, hm_w, kern, cx, cy, n);
      const float c0 = n[4];                        // (cx, cy)
      const float dx = 0.5f * (n[5] - n[3]);        // (cx+1) - (cx-1)
      const float dy = 0.5f * (n[7] - n[1]);        // (cy+1) - (cy-1)
      const float dxx = n[5] - 2.0f * c0 + n[3];
      const float dyy = n[7] - 2.0f * c0 + n[1];
      const float dxy = 0.5f * (n[8] - n[5] - n[7] + c0 + c0 - n[3] - n[1] + n[0]);

      const float eps = 1.1920929e-7f;  // float32 epsilon, as upstream
      const float a = dxx + eps, b = dxy, c = dxy, d = dyy + eps;
      const float det = a * d - b * c;
      // A singular Hessian means the peak is flat: there is no sub-pixel
      // information to recover, so keep the argmax rather than divide by ~0.
      if (std::fabs(det) > 1e-12f) {
        x -= (d * dx - b * dy) / det;
        y -= (-c * dx + a * dy) / det;
      }
    } else {
      x = -1.0f;  // reference marks an all-non-positive map this way
      y = -1.0f;
    }

    Keypoint& kp = kpts[static_cast<std::size_t>(k)];
    kp.x = x * scale_x + off_x + static_cast<float>(crop.x1 - crop.pad_left);
    kp.y = y * scale_y + off_y + static_cast<float>(crop.y1 - crop.pad_top);
    kp.score = std::max(0.0f, maxval);
  }
  return kpts;
}

WholeBodyEstimator::WholeBodyEstimator(Engine& engine, WholeBodyConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  if (output_index < 0 || output_index >= engine.numOutputs()) {
    throw Error(-1, "BCDL: WholeBodyEstimator: output index out of range");
  }
}

std::vector<Keypoint> WholeBodyEstimator::postprocess(const WholeBodyCrop& crop) const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* hm = outputAsFloat(engine_, out_idx_, scratch, shape);
  // [1,K,H,W] channel-first; a 3-dim [K,H,W] export is accepted too.
  int k, h, w;
  if (shape.size() == 4) {
    k = shape[1];
    h = shape[2];
    w = shape[3];
  } else if (shape.size() == 3) {
    k = shape[0];
    h = shape[1];
    w = shape[2];
  } else {
    throw Error(-1, "BCDL: WholeBodyEstimator: expected a [1,K,H,W] heatmap output");
  }
  return decodeWholeBody(hm, k, h, w, crop, cfg_);
}

}  // namespace bcdl

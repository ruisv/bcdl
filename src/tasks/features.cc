#include "bcdl/tasks/features.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// Bilinear sample of a [C, h, w] plane stack at (gx, gy) in SOURCE-grid
/// coordinates, writing C values. Outside the grid reads as zero, matching the
/// reference sampler's zero padding — a keypoint near the border legitimately
/// gets a partially-zero descriptor rather than a clamped-edge one.
void sampleBilinear(const float* planes, int c, int h, int w, float gx, float gy, float* out) {
  const int x0 = static_cast<int>(std::floor(gx));
  const int y0 = static_cast<int>(std::floor(gy));
  const float fx = gx - x0, fy = gy - y0;
  const float wts[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
  const int xs[4] = {x0, x0 + 1, x0, x0 + 1};
  const int ys[4] = {y0, y0, y0 + 1, y0 + 1};

  for (int ch = 0; ch < c; ++ch) out[ch] = 0.0f;
  for (int k = 0; k < 4; ++k) {
    if (xs[k] < 0 || xs[k] >= w || ys[k] < 0 || ys[k] >= h || wts[k] == 0.0f) continue;
    const float* p = planes + static_cast<std::size_t>(ys[k]) * w + xs[k];
    const std::size_t plane = static_cast<std::size_t>(h) * w;
    for (int ch = 0; ch < c; ++ch) out[ch] += wts[k] * p[ch * plane];
  }
}

/// Cubic convolution kernel with A = -0.75, the coefficient torch's grid_sample
/// (and cv::INTER_CUBIC) use.
inline float cubic1(float x, float a) { return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f; }
inline float cubic2(float x, float a) { return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a; }

void cubicWeights(float t, float w[4]) {
  constexpr float kA = -0.75f;
  w[0] = cubic2(t + 1.0f, kA);
  w[1] = cubic1(t, kA);
  w[2] = cubic1(1.0f - t, kA);
  w[3] = cubic2(2.0f - t, kA);
}

/// Bicubic sample of a [C, h, w] plane stack, zero outside the grid.
///
/// The descriptors are sampled BICUBICALLY, not bilinearly — the reference's
/// sparse interpolator defaults to bicubic and XFeat takes that default. Reading
/// it as bilinear is invisible in every shape and count and still produces
/// plausible descriptors; it just costs about 0.4% of descriptor cosine against
/// the reference, which is enough to move marginal matches.
void sampleBicubic(const float* planes, int c, int h, int w, float gx, float gy, float* out) {
  const int x0 = static_cast<int>(std::floor(gx));
  const int y0 = static_cast<int>(std::floor(gy));
  float wx[4], wy[4];
  cubicWeights(gx - x0, wx);
  cubicWeights(gy - y0, wy);

  for (int ch = 0; ch < c; ++ch) out[ch] = 0.0f;
  const std::size_t plane = static_cast<std::size_t>(h) * w;
  for (int j = 0; j < 4; ++j) {
    const int sy = y0 - 1 + j;
    if (sy < 0 || sy >= h) continue;
    for (int i = 0; i < 4; ++i) {
      const int sx = x0 - 1 + i;
      if (sx < 0 || sx >= w) continue;
      const float wgt = wy[j] * wx[i];
      if (wgt == 0.0f) continue;
      const float* p = planes + static_cast<std::size_t>(sy) * w + sx;
      for (int ch = 0; ch < c; ++ch) out[ch] += wgt * p[ch * plane];
    }
  }
}

/// Map a full-resolution pixel to the coordinates of a `src`-wide grid, in the
/// reference's convention: normalize by (full - 1), then un-normalize with
/// align_corners=false. The two do not use the same convention, which looks like
/// a mistake but is what the published model's own sampler does — matching it
/// matters more than tidying it.
inline float toGrid(int p, int src, int full) {
  return static_cast<float>(p) * src / static_cast<float>(full - 1) - 0.5f;
}

/// In-place separable max filter with a square window, used for non-maximum
/// suppression. Separable because a square max IS the composition of a
/// horizontal and a vertical max — 2*k comparisons per pixel instead of k*k,
/// which on a 640x480 map is the difference between ~3M and ~7.7M.
void maxFilter(const std::vector<float>& src, int h, int w, int k, std::vector<float>& dst) {
  const int r = k / 2;
  std::vector<float> tmp(src.size());
  for (int y = 0; y < h; ++y) {
    const float* row = src.data() + static_cast<std::size_t>(y) * w;
    float* trow = tmp.data() + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      float m = row[x];
      const int lo = std::max(0, x - r), hi = std::min(w - 1, x + r);
      for (int i = lo; i <= hi; ++i) m = std::max(m, row[i]);
      trow[x] = m;
    }
  }
  dst.assign(src.size(), 0.0f);
  for (int y = 0; y < h; ++y) {
    const int lo = std::max(0, y - r), hi = std::min(h - 1, y + r);
    float* drow = dst.data() + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      float m = tmp[static_cast<std::size_t>(lo) * w + x];
      for (int j = lo + 1; j <= hi; ++j) m = std::max(m, tmp[static_cast<std::size_t>(j) * w + x]);
      drow[x] = m;
    }
  }
}

}  // namespace

void xfeatPreprocess(const uint8_t* bgr, int width, int height, int stride, int in_w,
                     int in_h, std::vector<float>& out, float* scale_x, float* scale_y) {
  if (!bgr || width <= 0 || height <= 0 || in_w <= 0 || in_h <= 0) {
    throw Error(-1, "BCDL: xfeatPreprocess: bad image or model dimensions");
  }
  out.assign(static_cast<std::size_t>(in_w) * in_h, 0.0f);

  // Grayscale by CHANNEL MEAN + bilinear resize, in one pass.
  const float rx = static_cast<float>(width) / in_w;
  const float ry = static_cast<float>(height) / in_h;
  for (int oy = 0; oy < in_h; ++oy) {
    float sy = std::clamp((oy + 0.5f) * ry - 0.5f, 0.0f, static_cast<float>(height - 1));
    const int y0 = static_cast<int>(sy);
    const int y1 = std::min(y0 + 1, height - 1);
    const float wy = sy - y0;
    for (int ox = 0; ox < in_w; ++ox) {
      float sx = std::clamp((ox + 0.5f) * rx - 0.5f, 0.0f, static_cast<float>(width - 1));
      const int x0 = static_cast<int>(sx);
      const int x1 = std::min(x0 + 1, width - 1);
      const float wx = sx - x0;
      float acc = 0.0f;
      const int ys[2] = {y0, y1};
      const int xs[2] = {x0, x1};
      const float wys[2] = {1 - wy, wy};
      const float wxs[2] = {1 - wx, wx};
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          const uint8_t* p = bgr + static_cast<std::size_t>(ys[j]) * stride +
                             static_cast<std::size_t>(xs[i]) * 3;
          acc += wys[j] * wxs[i] * (p[0] + p[1] + p[2]) / 3.0f;
        }
      }
      out[static_cast<std::size_t>(oy) * in_w + ox] = acc;
    }
  }

  // InstanceNorm over the single channel: a plain standardization of the frame.
  // The variance is the BIASED one and eps is 1e-5, both as in the reference —
  // an unbiased variance would shift every activation slightly.
  double sum = 0.0, sq = 0.0;
  for (float v : out) {
    sum += v;
    sq += static_cast<double>(v) * v;
  }
  const double n = static_cast<double>(out.size());
  const double mean = sum / n;
  const double var = sq / n - mean * mean;
  const float inv = static_cast<float>(1.0 / std::sqrt(var + 1e-5));
  for (float& v : out) v = (v - static_cast<float>(mean)) * inv;

  if (scale_x) *scale_x = static_cast<float>(width) / in_w;
  if (scale_y) *scale_y = static_cast<float>(height) / in_h;
}

FeatureSet decodeXfeat(const float* feats, const float* kpt_logits, const float* reliability,
                       int fh, int fw, int in_h, int in_w, const XfeatConfig& cfg,
                       float scale_x, float scale_y) {
  if (!feats || !kpt_logits || !reliability || fh <= 0 || fw <= 0) {
    throw Error(-1, "BCDL: decodeXfeat: bad map dimensions");
  }
  if (in_h != fh * 8 || in_w != fw * 8) {
    throw Error(-1, "BCDL: decodeXfeat: input must be exactly 8x the map (" +
                        std::to_string(in_w) + "x" + std::to_string(in_h) + " vs " +
                        std::to_string(fw) + "x" + std::to_string(fh) + ")");
  }
  if (cfg.nms_kernel < 1 || cfg.nms_kernel % 2 == 0) {
    throw Error(-1, "BCDL: decodeXfeat: nms_kernel must be odd, got " +
                        std::to_string(cfg.nms_kernel));
  }

  const std::size_t cell = static_cast<std::size_t>(fh) * fw;
  const std::size_t full = static_cast<std::size_t>(in_h) * in_w;

  // --- 1. logits -> softmax over 65 -> scatter the 64 kept bins to full res ---
  // Channel c of a cell owns the pixel at offset (c/8, c%8) inside its 8x8
  // block; the 65th is the reject bin and is only ever part of the denominator.
  std::vector<float> heat(full, 0.0f);
  for (int gy = 0; gy < fh; ++gy) {
    for (int gx = 0; gx < fw; ++gx) {
      const std::size_t o = static_cast<std::size_t>(gy) * fw + gx;
      float m = kpt_logits[o];
      for (int c = 1; c < 65; ++c) m = std::max(m, kpt_logits[c * cell + o]);
      float denom = 0.0f;
      for (int c = 0; c < 65; ++c) denom += std::exp(kpt_logits[c * cell + o] - m);
      for (int c = 0; c < 64; ++c) {
        const int py = gy * 8 + c / 8;
        const int px = gx * 8 + c % 8;
        heat[static_cast<std::size_t>(py) * in_w + px] =
            std::exp(kpt_logits[c * cell + o] - m) / denom;
      }
    }
  }

  // --- 2. non-maximum suppression ------------------------------------------
  std::vector<float> local;
  maxFilter(heat, in_h, in_w, cfg.nms_kernel, local);

  struct Cand {
    int px, py;
    float score;
  };
  std::vector<Cand> cands;
  std::vector<float> rel(1);
  for (int py = 0; py < in_h; ++py) {
    for (int px = 0; px < in_w; ++px) {
      const std::size_t o = static_cast<std::size_t>(py) * in_w + px;
      if (heat[o] <= cfg.detection_thresh || heat[o] != local[o]) continue;
      // Score = detector probability at the pixel times the reliability of its
      // neighbourhood, sampled off the 1/8 map.
      sampleBilinear(reliability, 1, fh, fw, toGrid(px, fw, in_w), toGrid(py, fh, in_h),
                     rel.data());
      const float s = heat[o] * rel[0];
      if (s > 0.0f) cands.push_back({px, py, s});
    }
  }

  // --- 3. top-k by score ----------------------------------------------------
  const std::size_t keep =
      cfg.top_k > 0 ? std::min(cands.size(), static_cast<std::size_t>(cfg.top_k)) : cands.size();
  std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                    [](const Cand& a, const Cand& b) { return a.score > b.score; });
  cands.resize(keep);

  // --- 4. descriptors -------------------------------------------------------
  // The dense map is L2-normalized PER PIXEL first and each sampled descriptor
  // again afterwards. Both matter: sampling mixes four unit vectors, and the
  // mixture is not itself unit length.
  constexpr int kDim = 64;
  std::vector<float> norm(static_cast<std::size_t>(kDim) * cell);
  for (std::size_t o = 0; o < cell; ++o) {
    float ss = 0.0f;
    for (int c = 0; c < kDim; ++c) ss += feats[c * cell + o] * feats[c * cell + o];
    const float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
    for (int c = 0; c < kDim; ++c) norm[c * cell + o] = feats[c * cell + o] * inv;
  }

  FeatureSet out;
  out.dim = kDim;
  out.keypoints.reserve(cands.size());
  out.descriptors.resize(cands.size() * kDim);
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const Cand& c = cands[i];
    float* d = out.descriptors.data() + i * kDim;
    sampleBicubic(norm.data(), kDim, fh, fw, toGrid(c.px, fw, in_w), toGrid(c.py, fh, in_h), d);
    float ss = 0.0f;
    for (int k = 0; k < kDim; ++k) ss += d[k] * d[k];
    const float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
    for (int k = 0; k < kDim; ++k) d[k] *= inv;
    out.keypoints.push_back({c.px * scale_x, c.py * scale_y, c.score});
  }
  return out;
}

FeatureExtractor::FeatureExtractor(Engine& engine, XfeatConfig cfg, int output_base)
    : engine_(engine), cfg_(cfg), out_base_(output_base) {
  if (output_base < 0 || output_base + 3 > engine.numOutputs()) {
    throw Error(-1, "BCDL: FeatureExtractor: needs 3 outputs from output_base");
  }
}

FeatureSet FeatureExtractor::extract(const uint8_t* bgr, int width, int height, int stride,
                                     int timeout_ms) {
  const std::vector<int> ishape = engine_.inputShape(0);
  if (ishape.size() != 4) {
    throw Error(-1, "BCDL: FeatureExtractor: expected a [1,1,H,W] input");
  }
  const int in_h = ishape[2], in_w = ishape[3];
  float sx = 1.0f, sy = 1.0f;
  xfeatPreprocess(bgr, width, height, stride, in_w, in_h, input_, &sx, &sy);
  engine_.setInput(0, input_.data(), input_.size() * sizeof(float));
  engine_.infer(timeout_ms);
  return postprocess(sx, sy);
}

FeatureSet FeatureExtractor::postprocess(float scale_x, float scale_y) const {
  std::vector<float> s0, s1, s2;
  std::vector<int> h0, h1, h2;
  const float* feats = outputAsFloat(engine_, out_base_ + 0, s0, h0);
  const float* kpts = outputAsFloat(engine_, out_base_ + 1, s1, h1);
  const float* rel = outputAsFloat(engine_, out_base_ + 2, s2, h2);
  if (h0.size() != 4 || h1.size() != 4 || h2.size() != 4) {
    throw Error(-1, "BCDL: FeatureExtractor: expected [1,C,H,W] outputs");
  }
  const int fh = h0[2], fw = h0[3];
  return decodeXfeat(feats, kpts, rel, fh, fw, fh * 8, fw * 8, cfg_, scale_x, scale_y);
}

std::vector<FeatureMatch> matchFeatures(const FeatureSet& a, const FeatureSet& b,
                                        float min_cossim) {
  std::vector<FeatureMatch> out;
  if (a.size() == 0 || b.size() == 0) return out;
  if (a.dim != b.dim) throw Error(-1, "BCDL: matchFeatures: descriptor dims differ");

  const int dim = a.dim;
  const int na = static_cast<int>(a.size()), nb = static_cast<int>(b.size());
  std::vector<int> best_b(na, -1);
  std::vector<float> best_b_val(na, -2.0f);
  std::vector<int> best_a(nb, -1);
  std::vector<float> best_a_val(nb, -2.0f);

  // One pass computes BOTH directions: while scanning row `i` we also keep each
  // column's running best. That avoids materializing the full na x nb similarity
  // matrix, which at the default top_k would be 67 MB.
#pragma omp parallel
  {
    std::vector<int> loc_a(nb, -1);
    std::vector<float> loc_a_val(nb, -2.0f);
#pragma omp for schedule(static)
    for (int i = 0; i < na; ++i) {
      const float* da = a.descriptor(static_cast<std::size_t>(i));
      float bv = -2.0f;
      int bi = -1;
      for (int j = 0; j < nb; ++j) {
        const float* db = b.descriptor(static_cast<std::size_t>(j));
        float s = 0.0f;
        for (int k = 0; k < dim; ++k) s += da[k] * db[k];
        if (s > bv) {
          bv = s;
          bi = j;
        }
        if (s > loc_a_val[j]) {
          loc_a_val[j] = s;
          loc_a[j] = i;
        }
      }
      best_b[i] = bi;
      best_b_val[i] = bv;
    }
#pragma omp critical
    {
      for (int j = 0; j < nb; ++j) {
        if (loc_a_val[j] > best_a_val[j]) {
          best_a_val[j] = loc_a_val[j];
          best_a[j] = loc_a[j];
        }
      }
    }
  }

  for (int i = 0; i < na; ++i) {
    const int j = best_b[i];
    if (j < 0 || best_a[j] != i) continue;              // not mutual
    if (min_cossim > 0.0f && best_b_val[i] <= min_cossim) continue;
    out.push_back({i, j, best_b_val[i]});
  }
  return out;
}

}  // namespace bcdl

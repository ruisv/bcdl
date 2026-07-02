#include "bcdl/tasks/mono3d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"  // sigmoid, outputAsFloat
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

// SMOKE hardcodes PI = 3.14159 (smoke_coder.py). Use the SAME truncated value
// for the alpha quadrant shift and the yaw wrap so the decode matches the host
// reference bit-for-bit, not M_PI.
constexpr float kSmokePi = 3.14159f;

float wrapPi(float a) {
  if (a > kSmokePi) a -= 2.0f * kSmokePi;
  if (a < -kSmokePi) a += 2.0f * kSmokePi;
  return a;
}

// Project the 8 corners of a KITTI 3D box (bottom-center location `loc`,
// dims = (l,h,w), rotation `ry`) with intrinsics K, return the axis-aligned
// [x1,y1,x2,y2] enclosing box clamped to the original image. Mirrors
// SMOKECoder.encode_label / encode_box2d (corner SET is order-independent, so
// the simpler encode_label formula yields the same min/max).
std::array<float, 4> project3dBox(float x, float y, float z, float l, float h, float w,
                                  float ry, const CameraIntrinsics& K, int origW,
                                  int origH) {
  // Local corner template (KITTI convention), centered as in encode_label.
  const float xc[8] = {0, l, l, l, l, 0, 0, 0};
  const float yc[8] = {0, 0, h, h, 0, 0, h, h};
  const float zc[8] = {0, 0, 0, w, w, w, w, 0};
  const float c = std::cos(ry);
  const float s = std::sin(ry);

  float x1 = 1e9f, y1 = 1e9f, x2 = -1e9f, y2 = -1e9f;
  for (int i = 0; i < 8; ++i) {
    const float lx = xc[i] - l * 0.5f;
    const float ly = yc[i] - h;
    const float lz = zc[i] - w * 0.5f;
    // rot_y: [[c,0,s],[0,1,0],[-s,0,c]]
    const float wx = c * lx + s * lz + x;
    const float wy = ly + y;
    const float wz = -s * lx + c * lz + z;
    const float u = (K.fx * wx + K.cx * wz) / wz;
    const float v = (K.fy * wy + K.cy * wz) / wz;
    x1 = std::min(x1, u);
    y1 = std::min(y1, v);
    x2 = std::max(x2, u);
    y2 = std::max(y2, v);
  }
  const float W = static_cast<float>(origW);
  const float H = static_cast<float>(origH);
  return {std::clamp(x1, 0.0f, W), std::clamp(y1, 0.0f, H), std::clamp(x2, 0.0f, W),
          std::clamp(y2, 0.0f, H)};
}

}  // namespace

LetterboxInfo computeMono3dFeatureXform(int origW, int origH, int featW, int featH) {
  LetterboxInfo lb;
  lb.srcW = origW;
  lb.srcH = origH;
  lb.dstW = featW;
  lb.dstH = featH;
  if (origW <= 0 || origH <= 0) return lb;
  // SMOKE scales TO WIDTH and centers the height (get_transfrom_matrix with
  // center=image center, scale=[origW,origH], output=[featW,featH]):
  //   s = featW/origW,  padX = 0,  padY = featH/2 - s*origH/2.
  lb.scale = static_cast<float>(featW) / static_cast<float>(origW);
  lb.padX = 0.0f;
  lb.padY = static_cast<float>(featH) * 0.5f - lb.scale * static_cast<float>(origH) * 0.5f;
  return lb;
}

std::vector<Mono3dBox> decodeMono3d(const float* cls, const float* reg, int H, int W,
                                    const Mono3dConfig& cfg, const LetterboxInfo& featXform,
                                    const CameraIntrinsics& K) {
  const int nc = cfg.num_classes;
  if (cls == nullptr || reg == nullptr || H <= 0 || W <= 0 || nc <= 0) {
    return {};
  }
  if (static_cast<int>(cfg.dim_ref.size()) < nc) {
    throw Error(-1, "BCDL decodeMono3d: dim_ref has fewer rows than num_classes");
  }
  const int64_t HW = static_cast<int64_t>(H) * W;

  // ---- CenterNet heatmap NMS (3x3 local max on sigmoid) -> peak candidates ----
  struct Peak {
    float score;
    int c, y, x;
  };
  std::vector<Peak> peaks;
  const int k = std::max(1, cfg.nms_kernel) / 2;  // half-window (1 for 3x3)
  for (int c = 0; c < nc; ++c) {
    const float* hm = cls + static_cast<int64_t>(c) * HW;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float v = sigmoid(hm[static_cast<int64_t>(y) * W + x]);
        // local maximum over the (2k+1)^2 window (>= keeps plateaus, as nms_hm)
        bool is_max = true;
        for (int dy = -k; dy <= k && is_max; ++dy) {
          const int ny = y + dy;
          if (ny < 0 || ny >= H) continue;
          for (int dx = -k; dx <= k; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= W) continue;
            if (sigmoid(hm[static_cast<int64_t>(ny) * W + nx]) > v) {
              is_max = false;
              break;
            }
          }
        }
        if (is_max) peaks.push_back({v, c, y, x});
      }
    }
  }

  // top-K across all classes, then threshold (matches select_topk K then >thr).
  const int topk = std::min<int>(cfg.max_dets, static_cast<int>(peaks.size()));
  std::partial_sort(peaks.begin(), peaks.begin() + topk, peaks.end(),
                    [](const Peak& a, const Peak& b) { return a.score > b.score; });
  peaks.resize(topk);

  std::vector<Mono3dBox> out;
  out.reserve(topk);
  for (const Peak& p : peaks) {
    if (p.score <= cfg.conf_thresh) continue;
    const int64_t cell = static_cast<int64_t>(p.y) * W + p.x;
    auto rc = [&](int ch) { return reg[static_cast<int64_t>(ch) * HW + cell]; };

    // depth: z = off * depth_ref[1] + depth_ref[0]
    const float z = rc(0) * cfg.depth_ref[1] + cfg.depth_ref[0];

    // location: feature peak + offset -> original px (featXform.inv) -> K^-1
    const float proj_x = static_cast<float>(p.x) + rc(1);
    const float proj_y = static_cast<float>(p.y) + rc(2);
    const float px = featXform.invX(proj_x);
    const float py = featXform.invY(proj_y);
    float x = (px - K.cx) / K.fx * z;
    float y = (py - K.cy) / K.fy * z;

    // dimensions: exp(sigmoid(off)-0.5) * dim_ref[c]; internal (l,h,w) = (d0,d1,d2)
    const std::array<float, 3>& dref = cfg.dim_ref[p.c];
    const float l = std::exp(sigmoid(rc(3)) - 0.5f) * dref[0];
    const float h = std::exp(sigmoid(rc(4)) - 0.5f) * dref[1];
    const float w = std::exp(sigmoid(rc(5)) - 0.5f) * dref[2];
    y += h * 0.5f;  // feature/3D center -> box bottom center

    // orientation: l2-normalize (sin,cos), alpha = atan(sin/cos) + quadrant fix
    float si = rc(6), co = rc(7);
    const float nrm = std::sqrt(si * si + co * co) + 1e-12f;
    si /= nrm;
    co /= nrm;
    float alpha = std::atan(si / (co + 1e-7f));
    alpha += (co >= 0.0f) ? -kSmokePi * 0.5f : kSmokePi * 0.5f;
    const float ray = std::atan(x / (z + 1e-7f));
    const float yaw = wrapPi(alpha + ray);

    Mono3dBox b;
    b.class_id = p.c;
    b.score = p.score;
    b.x = x;
    b.y = y;
    b.z = z;
    b.h = h;
    b.w = w;
    b.l = l;
    b.yaw = yaw;
    b.alpha = alpha;
    if (cfg.pred_2d) {
      b.box2d = project3dBox(x, y, z, l, h, w, yaw, K, featXform.srcW, featXform.srcH);
    }
    out.push_back(b);
  }
  return out;
}

Mono3dDetector::Mono3dDetector(Engine& engine, Mono3dConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {}

std::vector<Mono3dBox> Mono3dDetector::postprocess(int origW, int origH,
                                                   const CameraIntrinsics& K) const {
  if (out_base_ < 0 || out_base_ + 2 > engine_.numOutputs()) {
    throw Error(-1, "BCDL Mono3dDetector: output index range out of bounds");
  }

  std::vector<float> cls_buf, reg_buf;
  std::vector<int> cls_shape, reg_shape;
  const float* cls = outputAsFloat(engine_, out_base_, cls_buf, cls_shape);
  const float* reg = outputAsFloat(engine_, out_base_ + 1, reg_buf, reg_shape);

  // cls is CHANNEL-FIRST [1, nc, H, W] (or [nc, H, W]); reg is [1, 8, H, W].
  int nc = 0, H = 0, W = 0;
  if (cls_shape.size() == 4) {
    nc = cls_shape[1];
    H = cls_shape[2];
    W = cls_shape[3];
  } else if (cls_shape.size() == 3) {
    nc = cls_shape[0];
    H = cls_shape[1];
    W = cls_shape[2];
  } else {
    throw Error(-1, "BCDL Mono3dDetector: unexpected cls tensor rank");
  }
  if (nc <= 0 || H <= 0 || W <= 0) {
    throw Error(-1, "BCDL Mono3dDetector: degenerate cls shape");
  }

  Mono3dConfig eff = cfg_;
  eff.num_classes = nc;  // authoritative from the tensor, never trusted from cfg
  const LetterboxInfo featXform = computeMono3dFeatureXform(origW, origH, W, H);
  return decodeMono3d(cls, reg, H, W, eff, featXform, K);
}

}  // namespace bcdl

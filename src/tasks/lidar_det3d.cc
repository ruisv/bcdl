#include "bcdl/tasks/lidar_det3d.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "bcdl/tasks/obb.h"

namespace bcdl {

namespace {

constexpr float kPi = 3.14159265358979323846f;
/// The published head's direction classifier convention (OpenPCDet defaults):
/// the yaw is folded into a pi-period and the classifier says which half.
constexpr float kDirOffset = 0.78539816f;  // pi/4

float limitPeriod(float val, float offset, float period) {
  return val - std::floor(val / period + offset) * period;
}

}  // namespace

std::vector<float> generateAnchors3d(const Anchor3dConfig& cfg) {
  if (cfg.grid_x <= 1 || cfg.grid_y <= 1) {
    throw Error(-1, "BCDL generateAnchors3d: grid must be at least 2x2");
  }
  if (cfg.sizes.size() != cfg.centers.size() || cfg.sizes.empty()) {
    throw Error(-1, "BCDL generateAnchors3d: one centre height per class size");
  }
  for (const std::vector<float>& s : cfg.sizes) {
    if (s.size() != 3) {
      throw Error(-1, "BCDL generateAnchors3d: each size is (dx, dy, dz)");
    }
  }
  if (cfg.rotations.empty()) {
    throw Error(-1, "BCDL generateAnchors3d: at least one rotation");
  }

  // Cell centres are NOT used: the reference generator lays anchors on a closed
  // linspace over the range (its align_center=False path), so the first anchor
  // sits exactly on min and the last exactly on max. Half a pillar of offset is
  // enough to shift every box.
  const int nx = cfg.grid_x, ny = cfg.grid_y;
  const float sx = (cfg.max_x - cfg.min_x) / static_cast<float>(nx - 1);
  const float sy = (cfg.max_y - cfg.min_y) / static_cast<float>(ny - 1);
  const std::size_t nc = cfg.sizes.size(), nr = cfg.rotations.size();

  std::vector<float> a;
  a.reserve(static_cast<std::size_t>(ny) * nx * nc * nr * 7);
  // Row-major over (y, x) then class-major over (class, rotation): this is the
  // order the head's channels are laid out in, and a permutation of it decodes
  // into well-formed boxes with the wrong sizes.
  for (int iy = 0; iy < ny; ++iy) {
    const float y = cfg.min_y + sy * static_cast<float>(iy);
    for (int ix = 0; ix < nx; ++ix) {
      const float x = cfg.min_x + sx * static_cast<float>(ix);
      for (std::size_t c = 0; c < nc; ++c) {
        for (std::size_t r = 0; r < nr; ++r) {
          a.push_back(x);
          a.push_back(y);
          a.push_back(cfg.centers[c]);
          a.push_back(cfg.sizes[c][0]);
          a.push_back(cfg.sizes[c][1]);
          a.push_back(cfg.sizes[c][2]);
          a.push_back(cfg.rotations[r]);
        }
      }
    }
  }
  return a;
}

std::vector<Box3D> decodeDet3d(const float* cls, const float* box, const float* dir,
                               int num_classes, int height, int width,
                               const float* anchors, const Det3dConfig& cfg) {
  if (cls == nullptr || box == nullptr || anchors == nullptr) {
    throw Error(-1, "BCDL decodeDet3d: cls, box and anchors are required");
  }
  if (num_classes <= 0 || height <= 0 || width <= 0) {
    throw Error(-1, "BCDL decodeDet3d: bad head geometry");
  }
  const int num_rot = static_cast<int>(cfg.anchors.rotations.size());
  const int num_size = static_cast<int>(cfg.anchors.sizes.size());
  const int A = num_size * num_rot;                 // anchors per cell
  const std::size_t plane = static_cast<std::size_t>(height) * width;

  // Threshold in LOGIT space. sigmoid is monotone, so this is the same cut, and
  // it keeps the transcendental off the 99.9% of the 321k anchors that lose.
  const float logit_thresh =
      (cfg.apply_sigmoid && cfg.score_thresh > 0.0f && cfg.score_thresh < 1.0f)
          ? std::log(cfg.score_thresh / (1.0f - cfg.score_thresh))
          : -1e30f;

  // Two passes, and the split is about MEMORY not arithmetic. The tensors are
  // {1, A*C, H, W}, so scoring cell-by-cell reads one float per 214 KB stride —
  // a cache miss per class per anchor over 321k anchors. Scanning each channel
  // plane linearly instead turns the whole scoring pass sequential, and only the
  // handful of survivors pay the strided reads for their box residuals.
  std::vector<float> best_logit(plane * static_cast<std::size_t>(A), -1e30f);
  std::vector<int> best_cls(plane * static_cast<std::size_t>(A), 0);
  for (int a = 0; a < A; ++a) {
    for (int c = 0; c < num_classes; ++c) {
      const float* src = cls + static_cast<std::size_t>(a * num_classes + c) * plane;
      float* bl = best_logit.data() + static_cast<std::size_t>(a) * plane;
      int* bc = best_cls.data() + static_cast<std::size_t>(a) * plane;
      for (std::size_t i = 0; i < plane; ++i) {
        if (src[i] > bl[i]) {
          bl[i] = src[i];
          bc[i] = c;
        }
      }
    }
  }

  std::vector<Box3D> out;
  for (int a = 0; a < A; ++a) {
    const float* bl = best_logit.data() + static_cast<std::size_t>(a) * plane;
    const int* bc = best_cls.data() + static_cast<std::size_t>(a) * plane;
    for (std::size_t cell = 0; cell < plane; ++cell) {
      if (cfg.apply_sigmoid ? (bl[cell] < logit_thresh)
                            : (bl[cell] < cfg.score_thresh)) continue;
      const float score = cfg.apply_sigmoid ? sigmoid(bl[cell]) : bl[cell];

      const std::size_t ai = (cell * static_cast<std::size_t>(A) + a) * 7;
      const float xa = anchors[ai + 0], ya = anchors[ai + 1], za = anchors[ai + 2];
      const float dxa = anchors[ai + 3], dya = anchors[ai + 4], dza = anchors[ai + 5];
      const float ra = anchors[ai + 6];

      float t[7];
      for (int k = 0; k < 7; ++k) {
        t[k] = box[static_cast<std::size_t>(a * 7 + k) * plane + cell];
      }
      // ResidualCoder: centres are normalised by the anchor's BEV diagonal and
      // extents are log-ratios, so a raw residual is meaningless without the
      // anchor it belongs to.
      const float diag = std::sqrt(dxa * dxa + dya * dya);
      Box3D b;
      b.x = t[0] * diag + xa;
      b.y = t[1] * diag + ya;
      b.z = t[2] * dza + za;
      b.dx = std::exp(t[3]) * dxa;
      b.dy = std::exp(t[4]) * dya;
      b.dz = std::exp(t[5]) * dza;
      b.heading = t[6] + ra;
      b.label = bc[cell];
      b.score = score;

      if (dir != nullptr) {
        // Without this the yaw is right modulo pi — which looks fine in a BEV
        // picture and is a reversed vehicle in anything that consumes it.
        const float d0 = dir[static_cast<std::size_t>(a * 2 + 0) * plane + cell];
        const float d1 = dir[static_cast<std::size_t>(a * 2 + 1) * plane + cell];
        const float rot = limitPeriod(b.heading - kDirOffset, 0.0f, kPi);
        b.heading = rot + kDirOffset + (d1 > d0 ? kPi : 0.0f);
      }
      out.push_back(b);
    }
  }

  return nms3d(std::move(out), cfg.nms_iou, cfg.max_boxes);
}

float bevIou(const Box3D& a, const Box3D& b) {
  // The BEV footprint is exactly the oriented rectangle the OBB task already
  // intersects; 3-D NMS in this family is defined on that projection, not on
  // volume, so there is nothing extra to compute.
  const RotatedBox ra{a.x, a.y, a.dx, a.dy, a.heading};
  const RotatedBox rb{b.x, b.y, b.dx, b.dy, b.heading};
  return rotatedIoU(ra, rb);
}

std::vector<Box3D> nms3d(std::vector<Box3D> boxes, float iou_thresh, int max_boxes) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j) { return boxes[i].score > boxes[j].score; });

  std::vector<char> dead(boxes.size(), 0);
  std::vector<Box3D> kept;
  for (int idx : order) {
    if (dead[idx]) continue;
    kept.push_back(boxes[idx]);
    if (max_boxes > 0 && static_cast<int>(kept.size()) >= max_boxes) break;
    for (int other : order) {
      if (dead[other] || other == idx) continue;
      if (boxes[other].label != boxes[idx].label) continue;
      if (bevIou(boxes[idx], boxes[other]) > iou_thresh) dead[other] = 1;
    }
  }
  return kept;
}

LidarDetector3d::LidarDetector3d(Engine& engine, Det3dConfig cfg, int cls_index,
                                 int box_index, int dir_index)
    : engine_(engine),
      cfg_(cfg),
      cls_idx_(cls_index),
      box_idx_(box_index),
      dir_idx_(dir_index) {
  const std::vector<int> s = engine.outputShape(cls_idx_);
  if (s.size() == 4) {
    // Take the head geometry from the model rather than the config, so a
    // different grid is a model swap and not a code change.
    cfg_.anchors.grid_y = s[2];
    cfg_.anchors.grid_x = s[3];
  }
  anchors_ = generateAnchors3d(cfg_.anchors);
}

std::vector<Box3D> LidarDetector3d::postprocess() const {
  std::vector<float> cls_buf, box_buf, dir_buf;
  std::vector<int> cls_shape, box_shape, dir_shape;
  const float* cls = outputAsFloat(engine_, cls_idx_, cls_buf, cls_shape);
  const float* box = outputAsFloat(engine_, box_idx_, box_buf, box_shape);
  const float* dir = nullptr;
  if (dir_idx_ >= 0) dir = outputAsFloat(engine_, dir_idx_, dir_buf, dir_shape);

  if (cls_shape.size() != 4 || box_shape.size() != 4) {
    throw Error(-1, "BCDL LidarDetector3d: expected {1,C,H,W} head tensors");
  }
  const int h = cls_shape[2], w = cls_shape[3];
  const int num_anchors = static_cast<int>(cfg_.anchors.sizes.size() *
                                           cfg_.anchors.rotations.size());
  if (cls_shape[1] % num_anchors != 0) {
    throw Error(-1, "BCDL LidarDetector3d: class channels (" +
                        std::to_string(cls_shape[1]) + ") are not a multiple of the " +
                        std::to_string(num_anchors) + " anchors per cell");
  }
  return decodeDet3d(cls, box, dir, cls_shape[1] / num_anchors, h, w, anchors_.data(),
                     cfg_);
}

}  // namespace bcdl

#include "bcdl/tasks/obb.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

#ifdef BCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/geometry/2d.hpp>  // rotatedRectangleIntersection / contourArea
#endif

namespace bcdl {

namespace {

constexpr float kPi = 3.14159265358979323846f;

#ifndef BCDL_HAVE_OPENCV
// 以下多边形裁剪求交的 helper 仅在没有 OpenCV 的降级路径下使用;有 OpenCV 时
// 用 cv::rotatedRectangleIntersection。
/// A 2D point used by the polygon-clipping intersection routine.
struct Pt {
  float x;
  float y;
};

/// Expand a rotated rect into its 4 corners (CCW in image coords, given the
/// usual y-down convention the winding is consistent for the area sign we use).
///
///   local corners: (-dx,-dy) (dx,-dy) (dx,dy) (-dx,dy)   dx=w/2, dy=h/2
///   world:  x = cx + lx*cos - ly*sin,  y = cy + lx*sin + ly*cos
void boxCorners(const RotatedBox& r, Pt out[4]) {
  const float c = std::cos(r.angle);
  const float s = std::sin(r.angle);
  const float dx = r.w * 0.5f;
  const float dy = r.h * 0.5f;
  const float lx[4] = {-dx, dx, dx, -dx};
  const float ly[4] = {-dy, -dy, dy, dy};
  for (int i = 0; i < 4; ++i) {
    out[i].x = r.cx + lx[i] * c - ly[i] * s;
    out[i].y = r.cy + lx[i] * s + ly[i] * c;
  }
}

/// Shoelace area of a polygon (absolute value, so winding-independent).
float polygonArea(const std::vector<Pt>& p) {
  const size_t n = p.size();
  if (n < 3) return 0.0f;
  float acc = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const Pt& a = p[i];
    const Pt& b = p[(i + 1) % n];
    acc += a.x * b.y - b.x * a.y;
  }
  return std::fabs(acc) * 0.5f;
}

/// Signed side of point `p` relative to the directed edge a->b. Positive on the
/// left of the edge. Used to keep the "inside" half-plane during clipping.
inline float cross(const Pt& a, const Pt& b, const Pt& p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

/// Intersection point of segment p1->p2 with the infinite line a->b. Caller
/// guarantees the segment crosses the line (signs of cross differ), so the
/// denominator is non-degenerate up to float noise; we guard it anyway.
Pt lineIntersect(const Pt& p1, const Pt& p2, const Pt& a, const Pt& b) {
  const float a1 = b.y - a.y;
  const float b1 = a.x - b.x;
  const float c1 = a1 * a.x + b1 * a.y;
  const float a2 = p2.y - p1.y;
  const float b2 = p1.x - p2.x;
  const float c2 = a2 * p1.x + b2 * p1.y;
  const float det = a1 * b2 - a2 * b1;
  if (std::fabs(det) < 1e-12f) return p2;  // near-parallel: fall back to endpoint
  return Pt{(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}

/// Sutherland-Hodgman: clip the (convex) `subject` polygon against the convex
/// `clip` polygon, returning the intersection polygon. The clip polygon must be
/// convex (both inputs here are rectangles, so they are). We determine the
/// "inside" sign from the clip polygon's own winding so we don't assume CW/CCW.
std::vector<Pt> clipPolygon(const std::vector<Pt>& subject, const Pt clip[4]) {
  // Winding sign of the clip rect: inside points have cross() of this sign.
  // Use the third vertex against edge[0] (clip[0]->clip[1]) to read the sign.
  const float wind = cross(clip[0], clip[1], clip[2]);
  const float inside_sign = wind >= 0.0f ? 1.0f : -1.0f;

  std::vector<Pt> output = subject;
  for (int e = 0; e < 4 && !output.empty(); ++e) {
    const Pt& a = clip[e];
    const Pt& b = clip[(e + 1) % 4];
    const std::vector<Pt> input = output;
    output.clear();
    const size_t n = input.size();
    for (size_t i = 0; i < n; ++i) {
      const Pt& cur = input[i];
      const Pt& prv = input[(i + n - 1) % n];
      const float d_cur = cross(a, b, cur) * inside_sign;
      const float d_prv = cross(a, b, prv) * inside_sign;
      const bool cur_in = d_cur >= 0.0f;
      const bool prv_in = d_prv >= 0.0f;
      if (cur_in) {
        if (!prv_in) output.push_back(lineIntersect(prv, cur, a, b));
        output.push_back(cur);
      } else if (prv_in) {
        output.push_back(lineIntersect(prv, cur, a, b));
      }
    }
  }
  return output;
}
#endif  // !BCDL_HAVE_OPENCV

}  // namespace

float rotatedIoU(const RotatedBox& a, const RotatedBox& b) {
  const float area_a = a.w * a.h;
  const float area_b = b.w * b.h;
  if (area_a <= 0.0f || area_b <= 0.0f) return 0.0f;

#ifdef BCDL_HAVE_OPENCV
  // OpenCV 的 RotatedRect 角度以「度」表示。两个框用同一弧度->度换算,相对几何
  // 不变,因此交/并面积与 IoU 正确(IoU 与绝对角度约定无关)。
  const float to_deg = 180.0f / kPi;
  const cv::RotatedRect ra(cv::Point2f(a.cx, a.cy), cv::Size2f(a.w, a.h), a.angle * to_deg);
  const cv::RotatedRect rb(cv::Point2f(b.cx, b.cy), cv::Size2f(b.w, b.h), b.angle * to_deg);
  std::vector<cv::Point2f> inter;
  cv::rotatedRectangleIntersection(ra, rb, inter);
  if (inter.size() < 3) return 0.0f;  // 无交或退化
  const float inter_area = static_cast<float>(std::fabs(cv::contourArea(inter)));
  if (inter_area <= 0.0f) return 0.0f;
  const float uni = area_a + area_b - inter_area;
  return uni > 0.0f ? inter_area / uni : 0.0f;
#else
  Pt ca[4], cb[4];
  boxCorners(a, ca);
  boxCorners(b, cb);

  const std::vector<Pt> subject(ca, ca + 4);
  const std::vector<Pt> inter = clipPolygon(subject, cb);
  const float inter_area = polygonArea(inter);
  if (inter_area <= 0.0f) return 0.0f;

  const float uni = area_a + area_b - inter_area;
  return uni > 0.0f ? inter_area / uni : 0.0f;
#endif
}

std::vector<int> rotatedNms(const std::vector<ObbDetection>& dets, float iou_thresh,
                            int max_dets) {
  std::vector<int> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return dets[a].score > dets[b].score;
  });

  std::vector<int> keep;
  std::vector<char> suppressed(dets.size(), 0);
  for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
    const int i = order[oi];
    if (suppressed[i]) continue;
    keep.push_back(i);
    if (max_dets > 0 && static_cast<int>(keep.size()) >= max_dets) break;
    for (int oj = oi + 1; oj < static_cast<int>(order.size()); ++oj) {
      const int j = order[oj];
      if (suppressed[j]) continue;
      if (dets[j].class_id != dets[i].class_id) continue;  // per-class
      if (rotatedIoU(dets[i].rrect, dets[j].rrect) > iou_thresh) suppressed[j] = 1;
    }
  }
  return keep;
}

std::vector<ObbDetection> decodeObb(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& angle,
    const std::vector<std::pair<int, int>>& grid_hw, const ObbConfig& cfg,
    const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales || angle.size() != scales ||
      cfg.strides.size() != scales) {
    throw Error(-1, "BCDL decodeObb: cls/box/angle/grid/strides length mismatch");
  }
  if (nc <= 0) {
    throw Error(-1, "BCDL decodeObb: non-positive class count");
  }

  // Threshold on the RAW max class logit (matches the reference, which compares
  // max_logit >= conf_raw with conf_raw = -ln(1/score_thres - 1)). Equivalent to
  // sigmoid(max_logit) >= conf_thresh but avoids a sigmoid per cell pre-filter.
  const float t = std::min(std::max(cfg.conf_thresh, 1e-6f), 1.0f - 1e-6f);
  const float conf_raw = -std::log(1.0f / t - 1.0f);
  const float angle_sign = static_cast<float>(cfg.angle_sign);

  std::vector<ObbDetection> dets;
  for (size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    const float* ap = angle[s];
    if (cp == nullptr || bp == nullptr || ap == nullptr || H <= 0 || W <= 0) continue;

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const int64_t cell = static_cast<int64_t>(gy) * W + gx;
        const float* logits = cp + cell * nc;

        // argmax over classes (sigmoid is monotonic, so argmax is on raw logits).
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < nc; ++k) {
          if (logits[k] > best_raw) {
            best_raw = logits[k];
            best_k = k;
          }
        }
        if (best_raw < conf_raw) continue;
        const float score = sigmoid(best_raw);

        // box distances: take absolute value (reference v_box = abs(box)).
        const float* d = bp + cell * 4;
        const float l = std::fabs(d[0]);
        const float top = std::fabs(d[1]);
        const float r = std::fabs(d[2]);
        const float bot = std::fabs(d[3]);

        // angle: (sigmoid(raw) - 0.5) * pi * sign + offset.
        const float a_rad =
            (sigmoid(ap[cell]) - 0.5f) * kPi * angle_sign + cfg.angle_offset_rad;

        const float grid_x = static_cast<float>(gx) + 0.5f;
        const float grid_y = static_cast<float>(gy) + 0.5f;
        const float xf = (r - l) * 0.5f;
        const float yf = (bot - top) * 0.5f;
        const float ca = std::cos(a_rad);
        const float sa = std::sin(a_rad);

        float cx = (grid_x + xf * ca - yf * sa) * stride;
        float cy = (grid_y + xf * sa + yf * ca) * stride;
        float w = (l + r) * stride;
        float h = (top + bot) * stride;
        float a = a_rad;

        if (cfg.regularize && w < h) {
          std::swap(w, h);
          a += kPi * 0.5f;
        }

        ObbDetection det;
        det.rrect = RotatedBox{cx, cy, w, h, a};
        det.score = score;
        det.class_id = best_k;
        dets.push_back(det);
      }
    }
  }

  const std::vector<int> keep = rotatedNms(dets, cfg.iou_thresh, cfg.max_dets);
  std::vector<ObbDetection> out;
  out.reserve(keep.size());
  for (int idx : keep) {
    ObbDetection det = dets[idx];
    // Un-letterbox: centers map back affinely (lb.invX/invY), sizes divide by the
    // uniform letterbox scale; the angle is invariant under the affine map.
    det.rrect.cx = lb.invX(det.rrect.cx);
    det.rrect.cy = lb.invY(det.rrect.cy);
    const float inv_scale = lb.scale > 0.0f ? 1.0f / lb.scale : 1.0f;
    det.rrect.w *= inv_scale;
    det.rrect.h *= inv_scale;
    out.push_back(det);
  }
  return out;
}

ObbDetector::ObbDetector(Engine& engine, ObbConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {}

std::vector<ObbDetection> ObbDetector::postprocess(const LetterboxInfo& lb) const {
  const int scales = static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || out_base_ + 3 * scales > engine_.numOutputs()) {
    throw Error(-1, "BCDL ObbDetector: output index range out of bounds");
  }

  // View each (cls, box, angle) triple as row-major floats — zero-copy for
  // packed F32, dequant-into-scratch otherwise. Scratch buffers must outlive the
  // decodeObb() call below that consumes the pointers.
  std::vector<std::vector<float>> cls_buf(scales), box_buf(scales), ang_buf(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales), ang_ptr(scales);
  std::vector<std::pair<int, int>> grid_hw(scales);

  // nc and grid (H,W) come from the cls tensor's OWN shape ([1,H,W,nc]) so a
  // mis-configured num_classes can't drive decodeObb past the buffer.
  int nc = 0;
  for (int s = 0; s < scales; ++s) {
    std::vector<int> cls_shape, box_shape, ang_shape;
    cls_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s, cls_buf[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 1, box_buf[s], box_shape);
    ang_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 2, ang_buf[s], ang_shape);
    int H = 0, W = 0, this_nc = 0;
    if (cls_shape.size() == 4) {  // [1, H, W, nc]
      H = cls_shape[1];
      W = cls_shape[2];
      this_nc = cls_shape[3];
    } else if (cls_shape.size() == 3) {  // [H, W, nc]
      H = cls_shape[0];
      W = cls_shape[1];
      this_nc = cls_shape[2];
    }
    if (s == 0) {
      nc = this_nc;
    } else if (this_nc != nc) {
      throw Error(-1, "BCDL ObbDetector: inconsistent class count across scales");
    }
    grid_hw[s] = {H, W};
  }

  ObbConfig eff = cfg_;
  if (nc > 0) eff.num_classes = nc;
  return decodeObb(cls_ptr, box_ptr, ang_ptr, grid_hw, eff, lb);
}

}  // namespace bcdl

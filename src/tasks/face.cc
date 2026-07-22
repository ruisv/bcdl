#include "bcdl/tasks/face.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "bcdl/tasks/detection.h"

namespace bcdl {

namespace {

/// Integer square root, exact-only: returns -1 unless `n` is a perfect square.
/// Used to recover the grid side from a flattened prediction count, so a wrong
/// num_anchors (or a non-square export) is caught instead of silently
/// mis-indexing every cell.
int exactSqrt(int n) {
  if (n < 0) return -1;
  int r = static_cast<int>(std::lround(std::sqrt(static_cast<double>(n))));
  for (int c = std::max(0, r - 2); c <= r + 2; ++c) {
    if (c * c == n) return c;
  }
  return -1;
}

}  // namespace

std::vector<FaceDetection> decodeScrfd(
    const std::vector<const float*>& score, const std::vector<const float*>& bbox,
    const std::vector<const float*>& kps, const std::vector<int>& counts,
    const FaceDetectConfig& cfg, const LetterboxInfo& lb) {
  const size_t ns = score.size();
  if (ns == 0 || bbox.size() != ns || kps.size() != ns || counts.size() != ns ||
      cfg.strides.size() != ns) {
    throw Error(-1, "BCDL face detect: score/bbox/kps/counts/strides length mismatch");
  }
  if (cfg.num_anchors <= 0) {
    throw Error(-1, "BCDL face detect: num_anchors must be positive");
  }

  // Boxes are NMS'd with the shared per-class routine, so candidates are staged
  // as Detection first and the landmarks are carried alongside by index.
  std::vector<Detection> boxes;
  std::vector<std::vector<std::pair<float, float>>> marks;

  for (size_t s = 0; s < ns; ++s) {
    if (score[s] == nullptr || bbox[s] == nullptr || kps[s] == nullptr) continue;
    const int n = counts[s];
    if (n <= 0) continue;
    if (n % cfg.num_anchors != 0) {
      throw Error(-1, "BCDL face detect: scale " + std::to_string(s) + " has " +
                          std::to_string(n) + " predictions, not divisible by " +
                          std::to_string(cfg.num_anchors) + " anchors");
    }
    const int cells = n / cfg.num_anchors;
    const int g = exactSqrt(cells);
    if (g < 0) {
      throw Error(-1, "BCDL face detect: scale " + std::to_string(s) + " has " +
                          std::to_string(cells) +
                          " cells, which is not a square grid — check num_anchors");
    }
    const float stride = static_cast<float>(cfg.strides[s]);

    for (int i = 0; i < n; ++i) {
      const float conf = score[s][i];
      if (conf < cfg.conf_thresh) continue;

      // Ordering is (row, column, anchor): anchors vary fastest.
      const int cell = i / cfg.num_anchors;
      const int gy = cell / g;
      const int gx = cell % g;
      // Centre sits ON the grid point for this head — no half-cell offset.
      const float cx = static_cast<float>(gx) * stride;
      const float cy = static_cast<float>(gy) * stride;

      const float* b = bbox[s] + static_cast<size_t>(i) * 4;
      Detection d;
      d.x1 = lb.clampX(lb.invX(cx - b[0] * stride));
      d.y1 = lb.clampY(lb.invY(cy - b[1] * stride));
      d.x2 = lb.clampX(lb.invX(cx + b[2] * stride));
      d.y2 = lb.clampY(lb.invY(cy + b[3] * stride));
      d.score = conf;
      d.class_id = 0;

      const float* k = kps[s] + static_cast<size_t>(i) * 10;
      std::vector<std::pair<float, float>> pts(5);
      for (int p = 0; p < 5; ++p) {
        // Landmarks are NOT clamped: a partly out-of-frame face still yields a
        // usable alignment from the points that are inside, and clamping would
        // drag the off-frame ones onto the border and skew the warp.
        pts[static_cast<size_t>(p)] = {lb.invX(cx + k[2 * p] * stride),
                                       lb.invY(cy + k[2 * p + 1] * stride)};
      }
      boxes.push_back(d);
      marks.push_back(std::move(pts));
    }
  }

  const std::vector<int> keep = nms(boxes, cfg.iou_thresh, cfg.max_faces);
  std::vector<FaceDetection> out;
  out.reserve(keep.size());
  for (int i : keep) {
    const Detection& d = boxes[static_cast<size_t>(i)];
    FaceDetection f;
    f.x1 = d.x1; f.y1 = d.y1; f.x2 = d.x2; f.y2 = d.y2;
    f.score = d.score;
    f.landmarks = marks[static_cast<size_t>(i)];
    out.push_back(std::move(f));
  }
  return out;
}

const std::vector<std::pair<float, float>>& arcFaceTemplate() {
  // The de-facto standard 5-point template for 112x112 recognition inputs.
  static const std::vector<std::pair<float, float>> kTemplate = {
      {38.2946f, 51.6963f},   // left eye
      {73.5318f, 51.5014f},   // right eye
      {56.0252f, 71.7366f},   // nose
      {41.5493f, 92.3655f},   // left mouth corner
      {70.7299f, 92.2041f},   // right mouth corner
  };
  return kTemplate;
}

std::vector<float> similarityTransform(
    const std::vector<std::pair<float, float>>& src,
    const std::vector<std::pair<float, float>>& dst) {
  const size_t n = src.size();
  if (n < 2 || dst.size() != n) {
    throw Error(-1, "BCDL face align: need >= 2 matching point pairs");
  }

  // Umeyama, closed form. Centre both sets, build the 2x2 covariance, take its
  // SVD (done analytically here since it is only 2x2), and read off rotation and
  // scale. Reflections are excluded: a mirrored "alignment" would map a face
  // onto its own mirror image, which is a different face to the embedder.
  double mx = 0, my = 0, dx = 0, dy = 0;
  for (size_t i = 0; i < n; ++i) {
    mx += src[i].first;  my += src[i].second;
    dx += dst[i].first;  dy += dst[i].second;
  }
  const double inv = 1.0 / static_cast<double>(n);
  mx *= inv; my *= inv; dx *= inv; dy *= inv;

  double var_src = 0.0;
  double c00 = 0, c01 = 0, c10 = 0, c11 = 0;   // covariance dst^T * src
  for (size_t i = 0; i < n; ++i) {
    const double sx = src[i].first - mx, sy = src[i].second - my;
    const double tx = dst[i].first - dx, ty = dst[i].second - dy;
    var_src += sx * sx + sy * sy;
    c00 += tx * sx; c01 += tx * sy;
    c10 += ty * sx; c11 += ty * sy;
  }
  var_src *= inv;
  c00 *= inv; c01 *= inv; c10 *= inv; c11 *= inv;

  // For a 2x2 covariance the similarity rotation reduces to the angle of the
  // (trace, antitrace) pair — equivalent to the SVD route without needing one.
  const double a_ = c00 + c11;
  const double b_ = c10 - c01;
  const double norm = std::sqrt(a_ * a_ + b_ * b_);
  double cos_t = 1.0, sin_t = 0.0;
  if (norm > 1e-12) { cos_t = a_ / norm; sin_t = b_ / norm; }

  // Uniform scale = correlation along the rotation / source variance.
  const double scale = (var_src > 1e-12) ? (norm / var_src) : 1.0;

  const double a = scale * cos_t;
  const double b = -scale * sin_t;
  const double c = scale * sin_t;
  const double d = scale * cos_t;

  return {static_cast<float>(a), static_cast<float>(b),
          static_cast<float>(dx - (a * mx + b * my)),
          static_cast<float>(c), static_cast<float>(d),
          static_cast<float>(dy - (c * mx + d * my))};
}

std::vector<uint8_t> alignFace(const uint8_t* bgr, int src_w, int src_h,
                               int src_stride,
                               const std::vector<std::pair<float, float>>& landmarks,
                               int size) {
  if (bgr == nullptr || src_w <= 0 || src_h <= 0 || size <= 0) {
    throw Error(-1, "BCDL face align: bad source image or output size");
  }
  if (landmarks.size() != arcFaceTemplate().size()) {
    throw Error(-1, "BCDL face align: expected " +
                        std::to_string(arcFaceTemplate().size()) + " landmarks, got " +
                        std::to_string(landmarks.size()));
  }
  if (src_stride <= 0) src_stride = src_w * 3;

  // Template is defined at 112; scale it to whatever output size was asked for.
  const float k = static_cast<float>(size) / 112.0f;
  std::vector<std::pair<float, float>> dst;
  dst.reserve(arcFaceTemplate().size());
  for (const auto& p : arcFaceTemplate()) dst.emplace_back(p.first * k, p.second * k);

  // Solve source -> template, then INVERT it: the warp loop walks destination
  // pixels and samples the source, so it needs template -> source.
  const std::vector<float> m = similarityTransform(landmarks, dst);
  const float det = m[0] * m[4] - m[1] * m[3];
  if (std::fabs(det) < 1e-12f) {
    throw Error(-1, "BCDL face align: degenerate landmark configuration");
  }
  const float i00 = m[4] / det, i01 = -m[1] / det;
  const float i10 = -m[3] / det, i11 = m[0] / det;
  const float i02 = -(i00 * m[2] + i01 * m[5]);
  const float i12 = -(i10 * m[2] + i11 * m[5]);

  std::vector<uint8_t> out(static_cast<size_t>(size) * size * 3);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      float sx = i00 * fx + i01 * fy + i02;
      float sy = i10 * fx + i11 * fy + i12;
      // Clamp to the edge rather than filling black: a face near the frame
      // border then extends its own skin tones outward instead of gaining a
      // hard black band the embedder has never seen.
      sx = std::min(std::max(sx, 0.0f), static_cast<float>(src_w - 1));
      sy = std::min(std::max(sy, 0.0f), static_cast<float>(src_h - 1));

      const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
      const int x1 = std::min(x0 + 1, src_w - 1);
      const int y1 = std::min(y0 + 1, src_h - 1);
      const float ax = sx - static_cast<float>(x0);
      const float ay = sy - static_cast<float>(y0);

      const uint8_t* r0 = bgr + static_cast<size_t>(y0) * src_stride;
      const uint8_t* r1 = bgr + static_cast<size_t>(y1) * src_stride;
      uint8_t* o = out.data() + (static_cast<size_t>(y) * size + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const float v = (1 - ax) * (1 - ay) * r0[x0 * 3 + c] +
                        ax * (1 - ay) * r0[x1 * 3 + c] +
                        (1 - ax) * ay * r1[x0 * 3 + c] +
                        ax * ay * r1[x1 * 3 + c];
        o[c] = static_cast<uint8_t>(std::lround(std::min(std::max(v, 0.0f), 255.0f)));
      }
    }
  }
  return out;
}

FaceDetector::FaceDetector(Engine& engine, FaceDetectConfig cfg, int output_base)
    : engine_(engine), cfg_(cfg), out_base_(output_base) {
  const int need = out_base_ + 3 * static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || need > engine_.numOutputs()) {
    throw Error(-1, "BCDL face detect: needs " + std::to_string(need) +
                        " outputs, model has " + std::to_string(engine_.numOutputs()));
  }
}

std::vector<FaceDetection> FaceDetector::postprocess(const LetterboxInfo& lb) const {
  const size_t ns = cfg_.strides.size();
  // Scratch must outlive the pointers handed to the decoder.
  std::vector<std::vector<float>> scratch(3 * ns);
  std::vector<const float*> score(ns), bbox(ns), kps(ns);
  std::vector<int> counts(ns);

  for (size_t i = 0; i < ns; ++i) {
    std::vector<int> shape;
    // Outputs come in three groups: every score, then every bbox, then every kps.
    score[i] = outputAsFloat(engine_, out_base_ + static_cast<int>(i), scratch[i], shape);
    int64_t total = 1;
    for (int d : shape) total *= (d > 0 ? d : 1);
    counts[i] = static_cast<int>(total);

    bbox[i] = outputAsFloat(engine_, out_base_ + static_cast<int>(ns + i),
                            scratch[ns + i], shape);
    kps[i] = outputAsFloat(engine_, out_base_ + static_cast<int>(2 * ns + i),
                           scratch[2 * ns + i], shape);
  }
  return decodeScrfd(score, bbox, kps, counts, cfg_, lb);
}

}  // namespace bcdl

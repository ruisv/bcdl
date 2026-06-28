#include "bcdl/tasks/ocr.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

#ifdef BCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>  // minAreaRect / boundingRect (OpenCV 5)
#endif

namespace bcdl {

// ===========================================================================
// A. Recognition (CTC)
// ===========================================================================

std::vector<std::string> loadCharDict(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw Error(-1, "BCDL ocr: cannot open char dict: " + path);
  }
  // One token per line, kept verbatim (UTF-8 bytes). Trailing '\r' from
  // CRLF-terminated files is stripped so a Windows-edited dict still matches.
  std::vector<std::string> dict;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    dict.push_back(line);
  }
  return dict;
}

RecResult decodeCtc(const float* logits, int num_steps, int num_classes,
                    const std::vector<std::string>& dict) {
  RecResult out;
  out.score = 0.0f;
  if (logits == nullptr || num_steps <= 0 || num_classes <= 0) return out;

  // Greedy best-path decode (collapse repeats, drop blank at index 0). This is
  // a transcription of ccdl::post_process_crnn.
  int last_index = 0;       // argmax of the previous timestep (0 == blank seed)
  double score_sum = 0.0;   // sum of per-step max over EMITTED steps
  int emitted = 0;

  for (int t = 0; t < num_steps; ++t) {
    const float* row = logits + static_cast<int64_t>(t) * num_classes;
    int argmax_idx = 0;
    float max_value = row[0];
    for (int c = 1; c < num_classes; ++c) {
      if (row[c] > max_value) {
        max_value = row[c];
        argmax_idx = c;
      }
    }

    // Emit when non-blank AND not a repeat of the previous step's class.
    // (t>0 guard preserved from ccdl: the very first step can always emit.)
    if (argmax_idx > 0 && !(t > 0 && argmax_idx == last_index)) {
      // dict[idx] directly — blank is index 0, characters are 1..C-1.
      if (argmax_idx < static_cast<int>(dict.size())) {
        out.text += dict[argmax_idx];
        score_sum += max_value;
        ++emitted;
      }
    }
    last_index = argmax_idx;
  }

  if (emitted > 0) out.score = static_cast<float>(score_sum / emitted);
  return out;
}

TextRecognizer::TextRecognizer(Engine& engine, std::string dict_path, int out_idx)
    : engine_(engine), dict_(loadCharDict(dict_path)), out_idx_(out_idx) {}

RecResult TextRecognizer::postprocess() const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL ocr rec: output index out of range");
  }
  // scratch must outlive `data` (slow path returns a pointer into it).
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  if (shape.empty()) return {};

  // Layout [1, T, C] (PaddleOCR rec): last dim is the vocabulary, the rest is
  // the time axis. Robust to a missing batch dim ([T, C]).
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  const int num_classes = shape.back() > 0 ? shape.back() : 1;
  const int num_steps = static_cast<int>(total / num_classes);

  return decodeCtc(data, num_steps, num_classes, dict_);
}

// ===========================================================================
// B. Orientation classifier (0° / 180°)
// ===========================================================================

ClsDirResult decodeClsDir(const float* logits, int n, float thresh) {
  ClsDirResult res{0, 0.0f, false};
  if (logits == nullptr || n <= 0) return res;
  // [1,2] => argmax over the two direction logits/probs. Generalised to take the
  // argmax over all elements so a [1,N] head still yields a sane label.
  int best = 0;
  float best_v = logits[0];
  for (int i = 1; i < n; ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  res.label = best;
  res.score = best_v;
  res.flip180 = (best == 1 && best_v > thresh);
  return res;
}

TextAngleClassifier::TextAngleClassifier(Engine& engine, float thresh, int out_idx)
    : engine_(engine), thresh_(thresh), out_idx_(out_idx) {}

ClsDirResult TextAngleClassifier::postprocess() const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL ocr cls: output index out of range");
  }
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);

  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  if (data == nullptr || total <= 0) {
    throw Error(-1, "BCDL ocr cls: empty output");
  }
  return decodeClsDir(data, static_cast<int>(total), thresh_);
}

// ===========================================================================
// C. Detection (DBNet)
// ===========================================================================

namespace {

// Map a probability-map pixel (px,py) to original-image pixels via the letterbox
// geometry: prob -> model-input (scaleX/Y for a down-sampled DB head) -> original
// (lb.invX/invY + clamp). Identity when lb is default-constructed (dstW/H == 0).
struct ProbToImage {
  float sx, sy;
  bool has_lb;
  const LetterboxInfo& lb;
  ProbToImage(int H, int W, const LetterboxInfo& l)
      : sx(l.dstW > 0 ? static_cast<float>(l.dstW) / W : 1.0f),
        sy(l.dstH > 0 ? static_cast<float>(l.dstH) / H : 1.0f),
        has_lb(l.dstW > 0 && l.dstH > 0),
        lb(l) {}
  void operator()(float px, float py, float& ox, float& oy) const {
    if (has_lb) {
      ox = lb.clampX(lb.invX(px * sx));
      oy = lb.clampY(lb.invY(py * sy));
    } else {
      ox = px;
      oy = py;
    }
  }
};

// Fill TextBox.pts (clockwise) from four already-mapped corner points and set the
// axis-aligned bbox x1..y2 as their min/max.
TextBox makeBox(const float (&ox)[4], const float (&oy)[4], float score) {
  TextBox b;
  b.score = score;
  float minx = ox[0], miny = oy[0], maxx = ox[0], maxy = oy[0];
  for (int k = 0; k < 4; ++k) {
    b.pts[2 * k] = ox[k];
    b.pts[2 * k + 1] = oy[k];
    minx = std::min(minx, ox[k]);
    miny = std::min(miny, oy[k]);
    maxx = std::max(maxx, ox[k]);
    maxy = std::max(maxy, oy[k]);
  }
  b.x1 = minx;
  b.y1 = miny;
  b.x2 = maxx;
  b.y2 = maxy;
  return b;
}

#ifdef BCDL_HAVE_OPENCV
// box_score_fast: mean of the probability map inside the polygon (PaddleOCR's
// per-box confidence), masked to the contour within its axis-aligned ROI.
float boxScore(const cv::Mat& prob, const std::vector<cv::Point>& contour) {
  cv::Rect r = cv::boundingRect(contour) & cv::Rect(0, 0, prob.cols, prob.rows);
  if (r.width <= 0 || r.height <= 0) return 0.0f;
  cv::Mat mask = cv::Mat::zeros(r.height, r.width, CV_8U);
  std::vector<cv::Point> shifted;
  shifted.reserve(contour.size());
  for (const auto& p : contour) shifted.emplace_back(p.x - r.x, p.y - r.y);
  std::vector<std::vector<cv::Point>> polys{shifted};
  cv::fillPoly(mask, polys, cv::Scalar(1));
  return static_cast<float>(cv::mean(prob(r), mask)[0]);
}
#endif

}  // namespace

std::vector<TextBox> decodeDbnet(const float* prob, int H, int W,
                                 const DbConfig& cfg, const LetterboxInfo& lb) {
  std::vector<TextBox> boxes;
  if (prob == nullptr || H <= 0 || W <= 0) return boxes;
  const ProbToImage map(H, W, lb);

#ifdef BCDL_HAVE_OPENCV
  // 1. Binarise the probability map (PaddleOCR `thresh`).
  cv::Mat probm(H, W, CV_32F, const_cast<float*>(prob));  // view, no copy
  cv::Mat bin;
  cv::threshold(probm, bin, cfg.bin_thresh, 255.0, cv::THRESH_BINARY);
  bin.convertTo(bin, CV_8U);

  // 2. Outer contours of the foreground regions.
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  for (const auto& c : contours) {
    if (c.size() < 3) continue;
    // 3. Rotated min-area rect of the region.
    cv::RotatedRect rr = cv::minAreaRect(c);
    if (std::min(rr.size.width, rr.size.height) < static_cast<float>(cfg.min_size))
      continue;
    // 4. Region confidence (box_score_fast) + threshold.
    const float score = boxScore(probm, c);
    if (score < cfg.box_thresh) continue;
    // 5. Unclip: grow w,h by 2*d about the center, d = area*ratio/perimeter.
    const float w0 = rr.size.width, h0 = rr.size.height;
    const float perim = 2.0f * (w0 + h0);
    const float d = (perim > 0.0f) ? (w0 * h0) * cfg.unclip_ratio / perim : 0.0f;
    cv::RotatedRect grown(rr.center, cv::Size2f(w0 + 2.0f * d, h0 + 2.0f * d), rr.angle);
    cv::Point2f p[4];
    grown.points(p);
    // 6. Map the 4 corners back to original-image pixels.
    float ox[4], oy[4];
    for (int k = 0; k < 4; ++k) map(p[k].x, p[k].y, ox[k], oy[k]);
    boxes.push_back(makeBox(ox, oy, score));
  }
  return boxes;
#else
  // --- no-OpenCV fallback: hand-rolled BFS connected components -> axis-aligned
  // box (pts = its 4 corners). Same thresholds as the OpenCV path.
  const int64_t N = static_cast<int64_t>(H) * W;
  std::vector<uint8_t> fg(static_cast<size_t>(N), 0);
  for (int64_t i = 0; i < N; ++i) fg[i] = (prob[i] > cfg.bin_thresh) ? 1 : 0;
  std::vector<uint8_t> visited(static_cast<size_t>(N), 0);
  static const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int dys[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  const int nneigh = (cfg.connectivity == 4) ? 4 : 8;

  std::vector<int64_t> stack;
  for (int sy = 0; sy < H; ++sy) {
    for (int sx = 0; sx < W; ++sx) {
      const int64_t seed = static_cast<int64_t>(sy) * W + sx;
      if (!fg[seed] || visited[seed]) continue;
      int minx = sx, maxx = sx, miny = sy, maxy = sy;
      double prob_sum = 0.0;
      int64_t count = 0;
      stack.clear();
      stack.push_back(seed);
      visited[seed] = 1;
      while (!stack.empty()) {
        const int64_t cur = stack.back();
        stack.pop_back();
        const int cx = static_cast<int>(cur % W);
        const int cy = static_cast<int>(cur / W);
        prob_sum += prob[cur];
        ++count;
        if (cx < minx) minx = cx;
        if (cx > maxx) maxx = cx;
        if (cy < miny) miny = cy;
        if (cy > maxy) maxy = cy;
        for (int k = 0; k < nneigh; ++k) {
          const int nx = cx + dxs[k], ny = cy + dys[k];
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
          const int64_t nidx = static_cast<int64_t>(ny) * W + nx;
          if (fg[nidx] && !visited[nidx]) {
            visited[nidx] = 1;
            stack.push_back(nidx);
          }
        }
      }
      const float w = static_cast<float>(maxx - minx + 1);
      const float h = static_cast<float>(maxy - miny + 1);
      if (std::min(w, h) < static_cast<float>(cfg.min_size)) continue;
      const float score = (count > 0) ? static_cast<float>(prob_sum / count) : 0.0f;
      if (score < cfg.box_thresh) continue;
      const float area = w * h, perim = 2.0f * (w + h);
      const float dd = (perim > 0.0f) ? area * cfg.unclip_ratio / perim : 0.0f;
      const float rx1 = minx - dd, ry1 = miny - dd, rx2 = maxx + dd, ry2 = maxy + dd;
      float ox[4], oy[4];
      map(rx1, ry1, ox[0], oy[0]);  // CW: TL, TR, BR, BL
      map(rx2, ry1, ox[1], oy[1]);
      map(rx2, ry2, ox[2], oy[2]);
      map(rx1, ry2, ox[3], oy[3]);
      boxes.push_back(makeBox(ox, oy, score));
    }
  }
  return boxes;
#endif
}

DbTextDetector::DbTextDetector(Engine& engine, DbConfig cfg, int out_idx)
    : engine_(engine), cfg_(cfg), out_idx_(out_idx) {}

std::vector<TextBox> DbTextDetector::postprocess(const LetterboxInfo& lb) const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL ocr det: output index out of range");
  }
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  if (shape.empty()) return {};

  // Infer H/W as the two non-unit dims in order. Works for [1,1,H,W] (NCHW) and
  // [1,H,W,1] (NHWC); falls back to the last two dims if everything is >1.
  int H = 0, W = 0;
  for (int d : shape) {
    if (d <= 1) continue;
    if (H == 0) {
      H = d;
    } else if (W == 0) {
      W = d;
      break;
    }
  }
  if (H == 0 || W == 0) {
    const int n = static_cast<int>(shape.size());
    if (n >= 2) {
      H = shape[n - 2];
      W = shape[n - 1];
    } else {
      return {};
    }
  }

  return decodeDbnet(data, H, W, cfg_, lb);
}

}  // namespace bcdl

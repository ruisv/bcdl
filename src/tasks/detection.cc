#include "bcdl/tasks/detection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

float iou(const Detection& a, const Detection& b) {
  const float ix1 = std::max(a.x1, b.x1);
  const float iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2);
  const float iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh, int max_dets) {
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
      if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
    }
  }
  return keep;
}

std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const bool has_obj = cfg.layout == DecodeLayout::kYoloV5;
  const int attrs = (has_obj ? 5 : 4) + nc;

  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  if (attrs <= 0 || total < attrs) return {};
  const int64_t N = total / attrs;

  // at(a, j): attribute `a` of candidate `j` in the contiguous logical buffer.
  // channels_first  => [.., attrs, N]: offset = a*N + j
  // channels_last   => [.., N, attrs]: offset = j*attrs + a
  const auto at = [&](int a, int64_t j) -> float {
    const int64_t off = cfg.channels_first ? (static_cast<int64_t>(a) * N + j)
                                           : (j * attrs + a);
    return data[off];
  };

  const int cls_start = has_obj ? 5 : 4;
  std::vector<Detection> dets;
  for (int64_t j = 0; j < N; ++j) {
    // argmax over classes (sigmoid is monotonic, so argmax is invariant to it).
    int best_k = 0;
    float best_raw = at(cls_start, j);
    for (int k = 1; k < nc; ++k) {
      const float v = at(cls_start + k, j);
      if (v > best_raw) {
        best_raw = v;
        best_k = k;
      }
    }
    float cls_score = cfg.apply_sigmoid ? sigmoid(best_raw) : best_raw;

    float score;
    if (has_obj) {
      float obj = at(4, j);
      if (cfg.apply_sigmoid) obj = sigmoid(obj);
      score = obj * cls_score;
    } else {
      score = cls_score;
    }
    if (score < cfg.conf_thresh) continue;

    const float cx = at(0, j);
    const float cy = at(1, j);
    const float w = at(2, j);
    const float h = at(3, j);
    // model-input pixel box -> original-image pixels (un-letterbox) + clamp.
    const float mx1 = cx - w * 0.5f;
    const float my1 = cy - h * 0.5f;
    const float mx2 = cx + w * 0.5f;
    const float my2 = cy + h * 0.5f;

    Detection det;
    det.x1 = lb.clampX(lb.invX(mx1));
    det.y1 = lb.clampY(lb.invY(my1));
    det.x2 = lb.clampX(lb.invX(mx2));
    det.y2 = lb.clampY(lb.invY(my2));
    det.score = score;
    det.class_id = best_k;
    dets.push_back(det);
  }

  const std::vector<int> keep = nms(dets, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(dets[idx]);
  return out;
}

// ---------------------------------------------------------------------------
// Anchor-free LTRB multi-scale head (YOLO26 / standard RDK YOLO export)
// ---------------------------------------------------------------------------

std::vector<Detection> decodeYoloLtrb(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<std::pair<int, int>>& grid_hw, const YoloLtrbConfig& cfg,
    const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales ||
      cfg.strides.size() != scales) {
    throw Error(-1, "BCDL decodeYoloLtrb: cls/box/grid/strides length mismatch");
  }
  // Box head: plain LTRB (reg==0, 4 ch/cell) vs DFL (reg>0, 4*reg ch/cell).
  const int reg = cfg.reg_max;
  const int box_stride = (reg > 0) ? 4 * reg : 4;

  std::vector<Detection> dets;
  for (size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    if (cp == nullptr || bp == nullptr || H <= 0 || W <= 0) continue;

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const int64_t cell = static_cast<int64_t>(gy) * W + gx;
        const float* logits = cp + cell * nc;

        // argmax over classes (sigmoid is monotonic, so argmax is on raw logits)
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < nc; ++k) {
          if (logits[k] > best_raw) {
            best_raw = logits[k];
            best_k = k;
          }
        }
        const float score = sigmoid(best_raw);
        if (score < cfg.conf_thresh) continue;

        // LTRB distances about the cell center. Plain head reads 4 values; DFL
        // head reduces each side's `reg` raw logits via softmax-weighted sum
        // Σ b·softmax(b) (ultralytics DFL: 64 ch = 4 sides × 16 bins, side-major).
        float d[4];
        if (reg > 0) {
          const float* bb = bp + cell * box_stride;
          for (int side = 0; side < 4; ++side) {
            const float* sl = bb + static_cast<int64_t>(side) * reg;
            float maxv = sl[0];
            for (int b = 1; b < reg; ++b) maxv = std::max(maxv, sl[b]);
            float sum = 0.0f, acc = 0.0f;
            for (int b = 0; b < reg; ++b) {
              const float e = std::exp(sl[b] - maxv);
              sum += e;
              acc += e * static_cast<float>(b);
            }
            d[side] = (sum > 0.0f) ? acc / sum : 0.0f;
          }
        } else {
          const float* bb = bp + cell * 4;
          d[0] = bb[0]; d[1] = bb[1]; d[2] = bb[2]; d[3] = bb[3];
        }
        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;
        const float mx1 = (cx - d[0]) * stride;
        const float my1 = (cy - d[1]) * stride;
        const float mx2 = (cx + d[2]) * stride;
        const float my2 = (cy + d[3]) * stride;

        Detection det;
        det.x1 = lb.clampX(lb.invX(mx1));
        det.y1 = lb.clampY(lb.invY(my1));
        det.x2 = lb.clampX(lb.invX(mx2));
        det.y2 = lb.clampY(lb.invY(my2));
        det.score = score;
        det.class_id = best_k;
        dets.push_back(det);
      }
    }
  }

  const std::vector<int> keep = nms(dets, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(dets[idx]);
  return out;
}

YoloLtrbDetector::YoloLtrbDetector(Engine& engine, YoloLtrbConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {}

std::vector<Detection> YoloLtrbDetector::postprocess(const LetterboxInfo& lb) const {
  const int scales = static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || out_base_ + 2 * scales > engine_.numOutputs()) {
    throw Error(-1, "BCDL YoloLtrbDetector: output index range out of bounds");
  }

  // View each (cls, box) pair as row-major floats — zero-copy for packed F32
  // (the common RDK case), dequant-into-scratch otherwise. Scratch buffers live
  // until the decode below has consumed the pointers.
  std::vector<std::vector<float>> cls_buf(scales), box_buf(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales);
  std::vector<std::pair<int, int>> grid_hw(scales);

  // The class count, grid, and (implicitly) the cls buffer's per-cell stride all
  // come from the cls tensor's OWN shape ([1,H,W,nc]) — never from cfg_, so a
  // mis-configured num_classes can't make decodeYoloLtrb index past the buffer.
  int nc = 0, box_ch = 0;
  for (int s = 0; s < scales; ++s) {
    std::vector<int> cls_shape, box_shape;
    cls_ptr[s] = outputAsFloat(engine_, out_base_ + 2 * s, cls_buf[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, out_base_ + 2 * s + 1, box_buf[s], box_shape);
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
    // Box channel count drives plain-LTRB (4) vs DFL (4*reg_max) auto-detection.
    int this_box = 0;
    if (box_shape.size() == 4) this_box = box_shape[3];
    else if (box_shape.size() == 3) this_box = box_shape[2];
    if (s == 0) {
      nc = this_nc;
      box_ch = this_box;
    } else {
      if (this_nc != nc)
        throw Error(-1, "BCDL YoloLtrbDetector: inconsistent class count across scales");
      if (this_box != box_ch)
        throw Error(-1, "BCDL YoloLtrbDetector: inconsistent box channels across scales");
    }
    grid_hw[s] = {H, W};  // cls_ptr/box_ptr already set by outputAsFloat above
  }

  // Decode with the class count + box layout read from the model, not cfg.
  YoloLtrbConfig eff = cfg_;
  if (nc > 0) eff.num_classes = nc;
  if (box_ch == 4) {
    eff.reg_max = 0;                       // plain LTRB
  } else if (box_ch > 0 && box_ch % 4 == 0) {
    eff.reg_max = box_ch / 4;              // DFL (e.g. 64 -> reg_max 16)
  } else if (box_ch != 0) {
    throw Error(-1, "BCDL YoloLtrbDetector: box channel count is neither 4 nor 4*reg_max");
  }
  return decodeYoloLtrb(cls_ptr, box_ptr, grid_hw, eff, lb);
}

Detector::Detector(Engine& engine, DetectConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {}

std::vector<Detection> Detector::postprocess(const LetterboxInfo& lb) const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL detection: output index out of range");
  }
  // Zero-copy for packed F32, dequant-into-scratch otherwise, so decode() stays
  // layout/precision agnostic while the common case avoids the per-element walk.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decode(data, shape, cfg_, lb);
}

}  // namespace bcdl

#include "bcdl/tasks/panoptic_drive.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

std::vector<Detection> decodeYoloV5Anchor(
    const std::vector<const float*>& raw,
    const std::vector<std::pair<int, int>>& grid_hw,
    const AnchorDetectConfig& cfg, const LetterboxInfo& lb) {
  const size_t ns = raw.size();
  if (ns == 0 || grid_hw.size() != ns || cfg.strides.size() != ns ||
      cfg.anchors.size() != ns) {
    throw Error(-1, "BCDL anchor detect: raw/grid/strides/anchors length mismatch");
  }
  const int nc = cfg.num_classes;
  if (nc <= 0) throw Error(-1, "BCDL anchor detect: num_classes must be positive");

  const size_t na = cfg.anchors[0].size();
  if (na == 0) throw Error(-1, "BCDL anchor detect: no anchors given");
  for (const auto& a : cfg.anchors) {
    if (a.size() != na) {
      throw Error(-1, "BCDL anchor detect: every scale must declare the same anchor count");
    }
  }

  const int no = 5 + nc;  // cx,cy,w,h,obj + classes
  std::vector<Detection> cands;

  for (size_t s = 0; s < ns; ++s) {
    const float* p = raw[s];
    if (p == nullptr) continue;
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    if (H <= 0 || W <= 0) continue;
    const float stride = static_cast<float>(cfg.strides[s]);
    const size_t plane = static_cast<size_t>(H) * W;

    for (size_t a = 0; a < na; ++a) {
      // Channels-first: attribute `k` of anchor `a` is the plane at channel
      // (a*no + k), so one grid cell's attributes are `plane` elements apart.
      const size_t base = (a * static_cast<size_t>(no)) * plane;
      const float aw = cfg.anchors[s][a].w;
      const float ah = cfg.anchors[s][a].h;

      for (int gy = 0; gy < H; ++gy) {
        for (int gx = 0; gx < W; ++gx) {
          const size_t off = base + static_cast<size_t>(gy) * W + gx;

          // Objectness gates everything else: check it before paying for the
          // class scan, which is where most of the 25k candidates die.
          const float obj = sigmoid(p[off + 4 * plane]);
          if (obj < cfg.conf_thresh) continue;

          int best_c = 0;
          float best_s = -1.0f;
          for (int c = 0; c < nc; ++c) {
            const float v = sigmoid(p[off + static_cast<size_t>(5 + c) * plane]);
            if (v > best_s) { best_s = v; best_c = c; }
          }
          const float score = obj * best_s;
          if (score < cfg.conf_thresh) continue;

          const float tx = sigmoid(p[off + 0 * plane]);
          const float ty = sigmoid(p[off + 1 * plane]);
          const float tw = sigmoid(p[off + 2 * plane]);
          const float th = sigmoid(p[off + 3 * plane]);

          const float cx = (tx * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
          const float cy = (ty * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
          const float bw = (tw * 2.0f) * (tw * 2.0f) * aw;
          const float bh = (th * 2.0f) * (th * 2.0f) * ah;

          Detection d;
          d.x1 = lb.clampX(lb.invX(cx - bw * 0.5f));
          d.y1 = lb.clampY(lb.invY(cy - bh * 0.5f));
          d.x2 = lb.clampX(lb.invX(cx + bw * 0.5f));
          d.y2 = lb.clampY(lb.invY(cy + bh * 0.5f));
          d.score = score;
          d.class_id = best_c;
          cands.push_back(d);
        }
      }
    }
  }

  const std::vector<int> keep = nms(cands, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int i : keep) out.push_back(cands[static_cast<size_t>(i)]);
  return out;
}

AnchorDetector::AnchorDetector(Engine& engine, AnchorDetectConfig cfg, int output_base)
    : engine_(engine), cfg_(cfg), out_base_(output_base) {
  const int need = out_base_ + static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || need > engine_.numOutputs()) {
    throw Error(-1, "BCDL anchor detect: output range [" + std::to_string(out_base_) +
                        "," + std::to_string(need) + ") exceeds the model's " +
                        std::to_string(engine_.numOutputs()) + " outputs");
  }
}

std::vector<Detection> AnchorDetector::postprocess(const LetterboxInfo& lb) const {
  const size_t ns = cfg_.strides.size();
  const size_t na = cfg_.anchors.empty() ? 0 : cfg_.anchors[0].size();
  const int no = 5 + cfg_.num_classes;

  // Scratch buffers must outlive the pointers handed to the decoder.
  std::vector<std::vector<float>> scratch(ns);
  std::vector<const float*> raw(ns, nullptr);
  std::vector<std::pair<int, int>> grid(ns, {0, 0});

  for (size_t s = 0; s < ns; ++s) {
    std::vector<int> shape;
    raw[s] = outputAsFloat(engine_, out_base_ + static_cast<int>(s), scratch[s], shape);

    // Expect [1, na*(5+nc), H, W] (channels-first head convolution output).
    if (shape.size() < 3) {
      throw Error(-1, "BCDL anchor detect: output " + std::to_string(out_base_ + s) +
                          " has too few dims for an anchor head");
    }
    const size_t d = shape.size();
    const int C = shape[d - 3];
    grid[s] = {shape[d - 2], shape[d - 1]};

    const int expect = static_cast<int>(na) * no;
    if (C != expect) {
      throw Error(-1, "BCDL anchor detect: output " + std::to_string(out_base_ + s) +
                          " has " + std::to_string(C) + " channels, expected " +
                          std::to_string(expect) + " (= " + std::to_string(na) +
                          " anchors x (5 + " + std::to_string(cfg_.num_classes) +
                          " classes)) — check num_classes/anchors");
    }
  }
  return decodeYoloV5Anchor(raw, grid, cfg_, lb);
}

}  // namespace bcdl

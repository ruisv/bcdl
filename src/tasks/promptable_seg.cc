#include "bcdl/tasks/promptable_seg.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"  // outputAsFloat
#include "bcdl/core/status.h"

namespace bcdl {

SamMask decodeSamMasks(const float* low_res_logits, int num_masks, int mask_h,
                       int mask_w, const float* iou_pred, const SamConfig& cfg) {
  SamMask out;
  if (low_res_logits == nullptr || num_masks <= 0 || mask_h <= 0 || mask_w <= 0) {
    return out;
  }

  // Select the candidate: argmax predicted-IoU in multimask mode, else index 0.
  int best = 0;
  float best_iou = iou_pred != nullptr ? iou_pred[0] : 0.0f;
  if (cfg.multimask && iou_pred != nullptr) {
    for (int m = 1; m < num_masks; ++m) {
      if (iou_pred[m] > best_iou) {
        best_iou = iou_pred[m];
        best = m;
      }
    }
  }

  const int64_t plane = static_cast<int64_t>(mask_h) * mask_w;
  const float* mp = low_res_logits + static_cast<int64_t>(best) * plane;

  out.index = best;
  out.iou = best_iou;
  out.mask_w = mask_w;
  out.mask_h = mask_h;
  out.mask.resize(static_cast<size_t>(plane));
  for (int64_t i = 0; i < plane; ++i) {
    out.mask[static_cast<size_t>(i)] = mp[i] > cfg.mask_threshold ? 1u : 0u;
  }
  return out;
}

SamMaskDecoder::SamMaskDecoder(Engine& engine, SamConfig cfg, int masks_index,
                               int iou_index)
    : engine_(engine),
      cfg_(std::move(cfg)),
      masks_idx_(masks_index),
      iou_idx_(iou_index) {}

SamMask SamMaskDecoder::postprocess() const {
  if (masks_idx_ < 0 || masks_idx_ >= engine_.numOutputs() || iou_idx_ < 0 ||
      iou_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL SamMaskDecoder: output index out of bounds");
  }

  std::vector<float> mbuf, ibuf;
  std::vector<int> mshape, ishape;
  const float* mp = outputAsFloat(engine_, masks_idx_, mbuf, mshape);
  const float* ip = outputAsFloat(engine_, iou_idx_, ibuf, ishape);

  // masks: [1,Nmask,Hm,Wm] or [Nmask,Hm,Wm]; iou: [1,Nmask] or [Nmask].
  int nmask = 0, mh = 0, mw = 0;
  if (mshape.size() == 4) {
    nmask = mshape[1];
    mh = mshape[2];
    mw = mshape[3];
  } else if (mshape.size() == 3) {
    nmask = mshape[0];
    mh = mshape[1];
    mw = mshape[2];
  } else {
    throw Error(-1, "BCDL SamMaskDecoder: unexpected mask rank (want [1,N,H,W] or [N,H,W])");
  }

  return decodeSamMasks(mp, nmask, mh, mw, ip, cfg_);
}

}  // namespace bcdl

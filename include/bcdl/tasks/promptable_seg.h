#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Promptable segmentation mask decode — EdgeSAM / MobileSAM / SAM mask decoder
// ===========================================================================
//
// SAM's mask decoder emits, for a given prompt, `Nmask` candidate masks as
// LOW-RESOLUTION logits plus a predicted IoU per candidate:
//   - low_res_logits : [Nmask, Hm, Wm]   (Hm=Wm=256 for SAM)
//   - iou_pred       : [Nmask]
// In multimask mode the head returns 3 masks (whole / part / subpart) and the
// caller keeps the highest predicted-IoU one; single-mask mode returns 1.
//
// This is the PURE, board-independent tail of the pipeline: pick the best mask
// and binarize it at `mask_threshold` (SAM uses 0.0 on the raw logits, i.e.
// sigmoid > 0.5). Mapping the 256x256 mask back to original-image pixels
// (unpad + resize) is done by the caller with OpenCV on the board, and the full
// two-stage session (RepViT image encoder -> cached embedding -> prompt encoder
// + mask decoder) is layered on once the two `.hbm`s are converted (see the
// `SamSession` Python wrapper).

/// One decoded promptable-segmentation mask (low-resolution, binary).
struct SamMask {
  int index = -1;                ///< which candidate was selected (argmax IoU)
  float iou = 0.0f;              ///< the selected candidate's predicted IoU
  int mask_w = 0;                ///< Wm
  int mask_h = 0;                ///< Hm
  std::vector<uint8_t> mask;     ///< Hm*Wm row-major 0/1 (logit > mask_threshold)
};

/// Post-processing parameters for the SAM mask decoder tail.
struct SamConfig {
  float mask_threshold = 0.0f;   ///< binarize raw logits at this value (SAM: 0.0)
  bool multimask = true;         ///< pick argmax-IoU among candidates (else index 0)
};

/// Select and binarize the best mask from the decoder's raw outputs.
///
/// `low_res_logits` : Nmask*Hm*Wm floats, row-major [Nmask, Hm, Wm].
/// `iou_pred`       : Nmask floats (may be null -> candidate 0 is used).
/// `cfg`            : threshold + multimask selection.
/// Returns the chosen candidate as a low-resolution binary SamMask.
SamMask decodeSamMasks(const float* low_res_logits, int num_masks, int mask_h,
                       int mask_w, const float* iou_pred, const SamConfig& cfg);

/// Engine-bound mask-decoder tail. Reads the decoder model's two outputs — the
/// low-res mask logits ([1,Nmask,Hm,Wm] or [Nmask,Hm,Wm]) and the IoU
/// predictions ([1,Nmask] or [Nmask]) — and runs decodeSamMasks(). The image
/// encoder + prompt encoding are the caller's responsibility (they need the
/// second `.hbm` and prompt geometry); this only owns the pure mask tail.
class SamMaskDecoder {
 public:
  SamMaskDecoder(Engine& engine, SamConfig cfg = {}, int masks_index = 0,
                 int iou_index = 1);

  SamMask postprocess() const;

  const SamConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  SamConfig cfg_;
  int masks_idx_;
  int iou_idx_;
};

}  // namespace bcdl

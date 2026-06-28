#pragma once

#include <cstdint>

#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/vp_image.h"
#include "hobot/vp/hb_vp.h"

namespace bcdl {

/// Aspect-preserving "letterbox" resize of `src` into the pre-allocated `dst`
/// (sized to the model input). A single uniform scale is applied and the scaled
/// image is centered with `padValue` borders, in one hbVPWarpAffine op.
///
/// Cache discipline: the caller's CPU-written `src` is cleanCache()'d before the
/// op; `dst` is pre-filled with `padValue` (+ cleanCache) so the borders are the
/// pad color; `dst` is invalidateCache()'d after so the CPU/next op reads device
/// writes. Returns the geometry needed to un-letterbox detections.
LetterboxInfo letterbox(VpImage& dst, const VpImage& src, uint8_t padValue = 114,
                        hbVPInterpolationType interp = HB_VP_INTER_LINEAR);

/// Color-space conversion (e.g. NV12 -> BGR) via hbVPCvtColor. `dst` must be
/// pre-allocated with the target format. Same cache discipline as above.
void cvtColor(VpImage& dst, const VpImage& src);

/// Plain stretch-resize of `src` into `dst` via hbVPResize — NO padding; the
/// aspect ratio is NOT preserved (x and y scales differ when aspects differ).
/// The returned LetterboxInfo carries srcW/H, dstW/H and the X-axis scale with
/// zero padding; its uniform-scale inverse map is only exact when the aspect
/// ratio is preserved. Prefer letterbox() for detection pipelines.
LetterboxInfo resizeExact(VpImage& dst, const VpImage& src,
                          hbVPInterpolationType interp = HB_VP_INTER_LINEAR);

}  // namespace bcdl

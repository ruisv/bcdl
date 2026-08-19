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
/// UNVERIFIED — prefer letterboxToNv12Cpu() (preproc/letterbox_cpu.h) or
/// GdcLetterbox (preproc/gdc_letterbox.h). Every function in this header runs on
/// the vDSP, and the vDSP is not online on the S100P 4.0.5 image: /dev/vdsp0
/// exists but the core reports OFFLINE even as root, and no DSP firmware ships
/// in /lib/firmware, so these calls throw bcdl::Error rather than execute. That
/// also means the affine convention below has never been observed running: the
/// matrix is passed as the FORWARD src->dst map with isInverse=0, which is what
/// the SDK header documents but not something this project has been able to
/// confirm. On a board where the DSP does come up, check the first output frame
/// for mirroring or a wrong scale before trusting it — the alternatives above
/// are both measured and covered by tests.
///
/// Cache discipline: the caller's CPU-written `src` is cleanCache()'d before the
/// op; `dst` is pre-filled with `padValue` (+ cleanCache) so the borders are the
/// pad color; `dst` is invalidateCache()'d after so the CPU/next op reads device
/// writes. Returns the geometry needed to un-letterbox detections.
LetterboxInfo letterbox(VpImage& dst, const VpImage& src, uint8_t padValue = 114,
                        hbVPInterpolationType interp = HB_VP_INTER_LINEAR);

/// Color-space conversion (e.g. NV12 -> BGR) via hbVPCvtColor. `dst` must be
/// pre-allocated with the target format. Same cache discipline as above, and the
/// same vDSP availability caveat as letterbox().
void cvtColor(VpImage& dst, const VpImage& src);

/// Plain stretch-resize of `src` into `dst` via hbVPResize — NO padding; the
/// aspect ratio is NOT preserved (x and y scales differ when aspects differ).
/// The returned LetterboxInfo carries srcW/H, dstW/H and the X-axis scale with
/// zero padding; its uniform-scale inverse map is only exact when the aspect
/// ratio is preserved. Prefer letterbox() for detection pipelines — and see its
/// vDSP availability caveat, which applies here too.
LetterboxInfo resizeExact(VpImage& dst, const VpImage& src,
                          hbVPInterpolationType interp = HB_VP_INTER_LINEAR);

}  // namespace bcdl

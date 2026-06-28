#include "bcdl/preproc/letterbox.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bcdl/core/status.h"
#include "bcdl/core/task.h"
#include "hobot/vp/hb_vp_cvt_color.h"
#include "hobot/vp/hb_vp_resize.h"
#include "hobot/vp/hb_vp_warp_affine.h"

namespace bcdl {

LetterboxInfo letterbox(VpImage& dst, const VpImage& src, uint8_t padValue,
                        hbVPInterpolationType interp) {
  const LetterboxInfo lb =
      computeLetterbox(src.width(), src.height(), dst.width(), dst.height());

  // Pre-fill the whole dst buffer with the pad color so any pixel WarpAffine
  // does not touch (the letterbox borders) is already the pad color. hbVPWarpAffine
  // only accepts Y / NV12 sources (the VP units are NV12-native); for NV12 the
  // luma border is padValue but the chroma border must be 128 (neutral grey) or
  // the padding shows a color cast.
  if (dst.format() == HB_VP_IMAGE_FORMAT_NV12) {
    const std::size_t ySize =
        static_cast<std::size_t>(dst.raw().stride) * static_cast<std::size_t>(dst.height());
    std::memset(dst.data(), padValue, ySize);
    std::memset(static_cast<uint8_t*>(dst.data()) + ySize, 128, dst.mem().size() - ySize);
  } else {
    std::memset(dst.data(), padValue, dst.mem().size());
  }
  // CPU wrote dst -> flush before the device (VP) reads it.
  dst.cleanCache();
  // CPU wrote src (decoded/copied in) -> flush before the device reads it.
  src.cleanCache();

  // Forward map src(x,y) -> dst(scale*x+padX, scale*y+padY): a 2x3 affine
  //   [ scale  0     padX ]
  //   [ 0      scale  padY ]
  // isInverse=0 declares this as the forward (src->dst) matrix; the VP unit
  // inverts it internally to sample dst from src.
  // NOTE: verify the isInverse convention on-board — if output is mirrored /
  // wrongly scaled, the VP expects the inverse (dst->src) matrix here instead
  // (set isInverse=1, or supply [1/scale,0,-padX/scale; 0,1/scale,-padY/scale]).
  hbVPAffineParam param{};
  param.transformMatrix[0] = lb.scale;
  param.transformMatrix[1] = 0.0f;
  param.transformMatrix[2] = lb.padX;
  param.transformMatrix[3] = 0.0f;
  param.transformMatrix[4] = lb.scale;
  param.transformMatrix[5] = lb.padY;
  param.interpolation = static_cast<int8_t>(interp);
  param.borderType = static_cast<int8_t>(HB_VP_BORDER_CONSTANT);  // reserved field
  param.borderValue = padValue;                                   // reserved field
  param.isInverse = 0;

  Task task;
  BCDL_CHECK(hbVPWarpAffine(task.addr(), &dst.raw(), &src.raw(), &param));
  task.submit();
  task.wait();
  // Device wrote dst -> drop stale CPU cache lines before the CPU/next op reads.
  dst.invalidateCache();

  return lb;
}

void cvtColor(VpImage& dst, const VpImage& src) {
  // CPU wrote src -> flush before the device reads it.
  src.cleanCache();

  Task task;
  BCDL_CHECK(hbVPCvtColor(task.addr(), &dst.raw(), &src.raw()));
  task.submit();
  task.wait();
  // Device wrote dst -> invalidate before the CPU/next op reads.
  dst.invalidateCache();
}

LetterboxInfo resizeExact(VpImage& dst, const VpImage& src, hbVPInterpolationType interp) {
  // CPU wrote src -> flush before the device reads it.
  src.cleanCache();

  Task task;
  BCDL_CHECK(hbVPResize(task.addr(), &dst.raw(), &src.raw(), interp));
  task.submit();
  task.wait();
  // Device wrote dst -> invalidate before the CPU/next op reads.
  dst.invalidateCache();

  // Stretch fit: no padding; record the X-axis scale. The uniform inverse map
  // is only exact when src/dst aspect ratios match.
  LetterboxInfo lb;
  lb.srcW = src.width();
  lb.srcH = src.height();
  lb.dstW = dst.width();
  lb.dstH = dst.height();
  lb.scale = src.width() > 0 ? static_cast<float>(dst.width()) / src.width() : 1.0f;
  lb.padX = 0.0f;
  lb.padY = 0.0f;
  return lb;
}

}  // namespace bcdl

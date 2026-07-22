#include "bcdl/tracks/reid.h"

#include <algorithm>
#include <cstddef>

#include "bcdl/core/status.h"

// The rest of tracks/reid.h is header-only maths; this file exists for the one
// piece that touches pixels. It mirrors wholeBodyPreprocess() in tasks/wholebody
// — same bilinear sampler, same pixel-center convention — minus the aspect
// padding and the inverse-map bookkeeping, neither of which a ReID crop needs.

namespace bcdl {

void reidPreprocess(const uint8_t* bgr, int width, int height, int stride,
                    float bx1, float by1, float bx2, float by2, int in_w, int in_h,
                    const ReidConfig& cfg, std::vector<float>& out) {
  if (!bgr || width <= 0 || height <= 0 || in_w <= 0 || in_h <= 0) {
    throw Error(-1, "BCDL: reidPreprocess: bad image or model dimensions");
  }

  const int x1 = std::clamp(static_cast<int>(std::lround(bx1)), 0, width);
  const int y1 = std::clamp(static_cast<int>(std::lround(by1)), 0, height);
  const int x2 = std::clamp(static_cast<int>(std::lround(bx2)), 0, width);
  const int y2 = std::clamp(static_cast<int>(std::lround(by2)), 0, height);
  const int cw = x2 - x1, ch = y2 - y1;
  if (cw <= 0 || ch <= 0) {
    throw Error(-1, "BCDL: reidPreprocess: box is empty after clipping");
  }

  out.assign(static_cast<std::size_t>(3) * in_w * in_h, 0.0f);
  const float sx_ratio = static_cast<float>(cw) / static_cast<float>(in_w);
  const float sy_ratio = static_cast<float>(ch) / static_cast<float>(in_h);
  const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;

  for (int oy = 0; oy < in_h; ++oy) {
    float fy = (oy + 0.5f) * sy_ratio - 0.5f;
    fy = std::clamp(fy, 0.0f, static_cast<float>(ch - 1));
    const int py0 = static_cast<int>(fy);
    const int py1 = std::min(py0 + 1, ch - 1);
    const float wy = fy - py0;

    for (int ox = 0; ox < in_w; ++ox) {
      float fx = (ox + 0.5f) * sx_ratio - 0.5f;
      fx = std::clamp(fx, 0.0f, static_cast<float>(cw - 1));
      const int px0 = static_cast<int>(fx);
      const int px1 = std::min(px0 + 1, cw - 1);
      const float wx = fx - px0;

      const uint8_t* r0 = bgr + static_cast<std::size_t>(y1 + py0) * stride;
      const uint8_t* r1 = bgr + static_cast<std::size_t>(y1 + py1) * stride;
      const uint8_t* p00 = r0 + static_cast<std::size_t>(x1 + px0) * 3;
      const uint8_t* p01 = r0 + static_cast<std::size_t>(x1 + px1) * 3;
      const uint8_t* p10 = r1 + static_cast<std::size_t>(x1 + px0) * 3;
      const uint8_t* p11 = r1 + static_cast<std::size_t>(x1 + px1) * 3;
      const float w00 = (1.0f - wy) * (1.0f - wx);
      const float w01 = (1.0f - wy) * wx;
      const float w10 = wy * (1.0f - wx);
      const float w11 = wy * wx;

      const std::size_t o = static_cast<std::size_t>(oy) * in_w + ox;
      for (int c = 0; c < 3; ++c) {
        const int s = 2 - c;  // source is BGR, the model wants RGB
        const float v = w00 * p00[s] + w01 * p01[s] + w10 * p10[s] + w11 * p11[s];
        out[c * plane + o] = (v / 255.0f - cfg.mean[c]) / cfg.std[c];
      }
    }
  }
}

}  // namespace bcdl

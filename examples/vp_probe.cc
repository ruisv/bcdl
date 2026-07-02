// Runtime probe: does the hardware VP (Vision Processor) path actually run on
// THIS board? Exercises the three hbVP ops BCDL wraps — WarpAffine (letterbox),
// CvtColor (NV12->BGR) and Resize — on a real JPU-decoded frame, reporting
// OK / FAILED (+ error) per op. det_demo's header note claims "the hbVP path is
// offline on this board"; this settles it empirically.
//
//   ./vp_probe [image.jpg]

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  try {
    auto make_src = [&]() -> bcdl::VpImage {
      if (argc > 1) {
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> buf(n);
        size_t got = std::fread(buf.data(), 1, n, f); std::fclose(f);
        if (got != static_cast<size_t>(n)) throw std::runtime_error("short read");
        bcdl::JpegDecoder dec;
        bcdl::VpImage s = dec.decode(buf.data(), buf.size());
        std::printf("JPU decode: OK  -> NV12 %dx%d\n", s.width(), s.height());
        return s;
      }
      bcdl::VpImage s(1280, 720, HB_VP_IMAGE_FORMAT_NV12);
      std::memset(s.data(), 128, s.mem().size());
      s.cleanCache();
      std::printf("synthetic NV12 src %dx%d\n", s.width(), s.height());
      return s;
    };
    bcdl::VpImage src = make_src();

    // 1) VP letterbox (hbVPWarpAffine) -> NV12 640x640
    try {
      bcdl::VpImage dst(640, 640, HB_VP_IMAGE_FORMAT_NV12);
      const bcdl::LetterboxInfo lb = bcdl::letterbox(dst, src);
      const auto* y = static_cast<const uint8_t*>(dst.data());
      std::printf("VP letterbox  (hbVPWarpAffine): OK   scale=%.4f pad=(%.1f,%.1f) Y@center=%d\n",
                  lb.scale, lb.padX, lb.padY, y[320 * dst.raw().stride + 320]);
    } catch (const std::exception& e) {
      std::printf("VP letterbox  (hbVPWarpAffine): FAILED: %s\n", e.what());
    }

    // 2) VP cvtColor (hbVPCvtColor) NV12 -> BGR
    try {
      bcdl::VpImage bgr(src.width(), src.height(), HB_VP_IMAGE_FORMAT_BGR);
      bcdl::cvtColor(bgr, src);
      std::printf("VP cvtColor   (hbVPCvtColor):   OK\n");
    } catch (const std::exception& e) {
      std::printf("VP cvtColor   (hbVPCvtColor):   FAILED: %s\n", e.what());
    }

    // 3) VP resizeExact (hbVPResize) -> NV12 640x640
    try {
      bcdl::VpImage rz(640, 640, HB_VP_IMAGE_FORMAT_NV12);
      bcdl::resizeExact(rz, src);
      std::printf("VP resizeExact(hbVPResize):     OK\n");
    } catch (const std::exception& e) {
      std::printf("VP resizeExact(hbVPResize):     FAILED: %s\n", e.what());
    }
  } catch (const std::exception& e) {
    std::printf("setup failed: %s\n", e.what());
    return 2;
  }
  return 0;
}

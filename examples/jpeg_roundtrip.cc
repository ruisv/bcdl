// M2 demo: JPU JPEG encode/decode roundtrip + CPU letterbox (no vDSP needed).
//
// Builds a synthetic NV12 image, encodes it to JPEG on the JPU, decodes it back,
// and reports sizes + a luma fidelity metric. Also exercises the CPU letterbox
// path (BGR -> letterboxed NV12) since the hbVP/vDSP path is offline on the board.
//
//   ./jpeg_roundtrip [width height quality out.jpg]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  const int w = argc > 1 ? std::atoi(argv[1]) : 640;
  const int h = argc > 2 ? std::atoi(argv[2]) : 384;
  const int q = argc > 3 ? std::atoi(argv[3]) : 90;
  const char* out_path = argc > 4 ? argv[4] : nullptr;

  try {
    // --- synthetic NV12 source (gradient luma, neutral chroma) -------------
    bcdl::VpImage src(w, h, HB_VP_IMAGE_FORMAT_NV12);
    {
      auto* y = static_cast<unsigned char*>(src.data());
      const int ys = src.raw().stride;
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
          y[r * ys + c] = static_cast<unsigned char>((c * 255) / (w - 1));
      auto* uv = static_cast<unsigned char*>(src.raw().uvVirAddr);
      const int us = src.raw().uvStride;
      for (int r = 0; r < h / 2; ++r)
        for (int c = 0; c < w; ++c) uv[r * us + c] = 128;
      src.cleanCache();
    }

    // --- encode on JPU -----------------------------------------------------
    bcdl::JpegEncoder enc(w, h, q, HB_VP_IMAGE_FORMAT_NV12);
    std::vector<uint8_t> jpeg = enc.encode(src);
    std::printf("encode: %dx%d q=%d -> %zu bytes (%.2f bpp)\n", w, h, q, jpeg.size(),
                8.0 * jpeg.size() / (double)(w * h));
    if (jpeg.size() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
      std::printf("WARN: output does not start with JPEG SOI (FFD8)\n");
    }

    // --- decode on JPU -----------------------------------------------------
    bcdl::JpegDecoder dec(HB_VP_IMAGE_FORMAT_NV12);
    bcdl::VpImage back = dec.decode(jpeg);
    std::printf("decode: %zu bytes -> %dx%d NV12\n", jpeg.size(), back.width(),
                back.height());

    // --- luma fidelity (mean abs error on the Y plane) ---------------------
    if (back.width() == w && back.height() == h) {
      const auto* a = static_cast<const unsigned char*>(src.data());
      const auto* b = static_cast<const unsigned char*>(back.data());
      const int as = src.raw().stride, bs = back.raw().stride;
      double sad = 0;
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
          sad += std::abs(static_cast<int>(a[r * as + c]) - static_cast<int>(b[r * bs + c]));
      std::printf("Y-plane MAE after roundtrip: %.3f (lower is better)\n", sad / (w * h));
    }

    if (out_path) {
      if (FILE* f = std::fopen(out_path, "wb")) {
        std::fwrite(jpeg.data(), 1, jpeg.size(), f);
        std::fclose(f);
        std::printf("wrote %s\n", out_path);
      }
    }

    // --- CPU letterbox path (vDSP-free) ------------------------------------
    bcdl::VpImage bgr(1280, 720, HB_VP_IMAGE_FORMAT_BGR);
    {
      auto* p = static_cast<unsigned char*>(bgr.data());
      const int s = bgr.raw().stride;
      for (int r = 0; r < 720; ++r)
        for (int c = 0; c < 1280; ++c) {
          unsigned char* px = p + r * s + c * 3;
          px[0] = static_cast<unsigned char>(c & 0xFF);
          px[1] = static_cast<unsigned char>(r & 0xFF);
          px[2] = 128;
        }
    }
    bcdl::VpImage canvas_nv12(w, h, HB_VP_IMAGE_FORMAT_NV12);
    const bcdl::LetterboxInfo lb = bcdl::letterboxToNv12Cpu(canvas_nv12, bgr);
    std::printf("cpu letterbox: 1280x720 -> %dx%d NV12  scale=%.4f pad=(%.1f,%.1f)\n",
                w, h, lb.scale, lb.padX, lb.padY);

    std::printf("OK: JPU encode/decode + CPU letterbox paths ran.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

// Bench the hardware GDC letterbox (bcdl::GdcLetterbox) end-to-end:
// JPU-decode a JPEG -> NV12, then GDC-letterbox it to 640x640, looping to
// measure steady-state per-frame latency. Dumps the last output as raw NV12.
//
//   ./gdc_letterbox_bench image.jpg [out.nv12] [iters]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"
#include "bcdl/preproc/gdc_letterbox.h"

static std::vector<uint8_t> readFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n);
  if (std::fread(b.data(), 1, n, f) != (size_t)n) { std::fclose(f); throw std::runtime_error("short read"); }
  std::fclose(f);
  return b;
}

static void dumpNv12(const char* path, const bcdl::VpImage& im) {
  FILE* f = std::fopen(path, "wb");
  const auto& r = im.raw();
  const auto* y = static_cast<const uint8_t*>(im.data());
  const auto* uv = static_cast<const uint8_t*>(r.uvVirAddr);
  for (int i = 0; i < im.height(); ++i) std::fwrite(y + (size_t)i * r.stride, 1, im.width(), f);
  for (int i = 0; i < im.height() / 2; ++i) std::fwrite(uv + (size_t)i * r.uvStride, 1, im.width(), f);
  std::fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s image.jpg [out.nv12] [iters]\n", argv[0]); return 1; }
  const char* out_path = argc > 2 ? argv[2] : "gdc_lb_out.nv12";
  const int iters = argc > 3 ? std::atoi(argv[3]) : 200;
  try {
    bcdl::JpegDecoder jpu;
    bcdl::VpImage src = jpu.decode(readFile(argv[1]));
    std::printf("JPU decode -> NV12 %dx%d\n", src.width(), src.height());

    bcdl::GdcLetterbox gdc(src.width(), src.height(), 640, 640);
    const auto& lb = gdc.info();
    std::printf("letterbox: scale=%.4f pad=(%.1f,%.1f)\n", lb.scale, lb.padX, lb.padY);

    bcdl::VpImage dst(640, 640, HB_VP_IMAGE_FORMAT_NV12);
    for (int i = 0; i < 5; ++i) gdc.run(src, dst);  // warm-up

    double best = 1e9, sum = 0;
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      gdc.run(src, dst);
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0).count();
      sum += ms; if (ms < best) best = ms;
    }
    std::printf("GDC letterbox: mean %.3f ms  best %.3f ms  (%.1f img/s)  over %d iters\n",
                sum / iters, best, 1000.0 * iters / sum, iters);
    dumpNv12(out_path, dst);
    std::printf("wrote %s (640x640 NV12)\n", out_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
  return 0;
}

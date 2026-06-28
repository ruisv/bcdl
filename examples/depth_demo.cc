// M3 demo: monocular depth post-processing, chained with M1 (CPU BGR->NV12) and
// M2 (JPU JPEG) to write a colorized depth map — all on paths that work without
// the vDSP.
//
//   ./depth_demo depth.hbm [out.jpg]
//
// Feeds a zero input (path validation, not accuracy), runs DepthEstimator, then
// colorize -> BGR -> NV12 -> JPEG.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s depth.hbm [out.jpg]\n", argv[0]);
    return 1;
  }
  const char* out_path = argc > 2 ? argv[2] : nullptr;
  try {
    bcdl::Engine engine(argv[1], "");
    std::printf("model: %s  (inputs=%d outputs=%d)\n", engine.modelName().c_str(),
                engine.numInputs(), engine.numOutputs());

    for (int i = 0; i < engine.numInputs(); ++i) {
      std::vector<unsigned char> zeros(engine.inputBytes(i), 0);
      engine.setInput(i, zeros.data(), zeros.size());
    }
    engine.infer();

    bcdl::DepthConfig cfg;  // normalize to [0,1]
    bcdl::DepthEstimator est(engine, cfg, /*output_index=*/0);
    bcdl::DepthMap dm = est.postprocess();
    std::printf("depth: %dx%d  raw range=[%.4f, %.4f]\n", dm.width, dm.height, dm.vmin,
                dm.vmax);
    if (dm.width <= 0 || dm.height <= 0) {
      std::printf("no depth map produced; aborting.\n");
      return 0;
    }

    // colorize -> BGR HxWx3 (contiguous)
    std::vector<uint8_t> bgr = bcdl::depthColorize(dm);
    const int W = dm.width, H = dm.height;

    // JPU requires width aligned to 16 and height aligned to 8. Crop to fit.
    const int w2 = W & ~15, h2 = H & ~7;
    bcdl::VpImage bgrImg(w2, h2, HB_VP_IMAGE_FORMAT_BGR);
    {
      auto* dp = static_cast<uint8_t*>(bgrImg.data());
      const int ds = bgrImg.raw().stride;
      for (int r = 0; r < h2; ++r)
        std::memcpy(dp + r * ds, bgr.data() + static_cast<size_t>(r) * W * 3,
                    static_cast<size_t>(w2) * 3);
    }
    bcdl::VpImage nv12(w2, h2, HB_VP_IMAGE_FORMAT_NV12);
    bcdl::bgrToNv12Cpu(nv12, bgrImg);

    bcdl::JpegEncoder enc(w2, h2, 90, HB_VP_IMAGE_FORMAT_NV12);
    std::vector<uint8_t> jpeg = enc.encode(nv12);
    std::printf("colorized depth -> NV12 -> JPEG: %zu bytes\n", jpeg.size());

    if (out_path) {
      if (FILE* fp = std::fopen(out_path, "wb")) {
        std::fwrite(jpeg.data(), 1, jpeg.size(), fp);
        std::fclose(fp);
        std::printf("wrote %s\n", out_path);
      }
    }
    std::printf("OK: depth post-proc + colorize + NV12 + JPEG paths ran.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

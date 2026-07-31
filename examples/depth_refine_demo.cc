// RGB-D depth refinement: an RGB frame plus a raw sensor depth map go in, a
// cleaned metric depth map, a trust mask and (optionally) a point cloud come out.
//
//   ./depth_refine_demo model.hbm rgb.png raw_depth.png [out.png] [intrinsics.txt]
//
// `raw_depth.png` is the usual 16-bit depth encoding, millimetres by default
// (--depth-scale changes the divisor). The output image is a side-by-side of the
// input depth and the refined depth, both Turbo-colorized over a shared range so
// the two are actually comparable. When an intrinsics file is given (3x3 matrix,
// whitespace separated, in the RGB image's own pixels) the point cloud is written
// next to it as an ASCII .xyz.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bcdl/bcdl.h"

namespace {

/// Colorize a metric depth map over an explicit range so two maps drawn with the
/// same range stay comparable. Rejected pixels (0) render black.
cv::Mat colorize(const std::vector<float>& depth, int h, int w, float lo, float hi) {
  bcdl::DepthMap dm;
  dm.width = w;
  dm.height = h;
  dm.data = depth;
  dm.vmin = lo;
  dm.vmax = hi;
  const float range = hi > lo ? hi - lo : 1.0f;
  for (float& v : dm.data) v = v > 0.0f ? (v - lo) / range : 0.0f;
  std::vector<uint8_t> bgr = bcdl::depthColorize(dm);
  cv::Mat img(h, w, CV_8UC3);
  std::memcpy(img.data, bgr.data(), bgr.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (depth[static_cast<size_t>(y) * w + x] <= 0.0f)
        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
  return img;
}

bool loadIntrinsics(const char* path, bcdl::Intrinsics* k) {
  std::ifstream f(path);
  if (!f) return false;
  float m[9] = {0};
  for (float& v : m) {
    if (!(f >> v)) return false;
  }
  *k = bcdl::Intrinsics{m[0], m[4], m[2], m[5]};
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s model.hbm rgb.png raw_depth.png [out.png] [intrinsics.txt]\n",
                 argv[0]);
    return 1;
  }
  const char* out_path = argc > 4 ? argv[4] : "refined_depth.png";
  const char* intr_path = argc > 5 ? argv[5] : nullptr;
  const float depth_scale = 1000.0f;  // 16-bit millimetres

  try {
    cv::Mat bgr = cv::imread(argv[2], cv::IMREAD_COLOR);
    cv::Mat raw = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    if (bgr.empty() || raw.empty()) {
      std::fprintf(stderr, "failed to read %s or %s\n", argv[2], argv[3]);
      return 1;
    }
    if (raw.size() != bgr.size()) {
      std::fprintf(stderr, "rgb %dx%d and depth %dx%d must be the same size\n", bgr.cols,
                   bgr.rows, raw.cols, raw.rows);
      return 1;
    }

    std::vector<float> depth_m;
    if (raw.type() == CV_16UC1) {
      depth_m = bcdl::depthU16ToMetres(reinterpret_cast<const uint16_t*>(raw.data), raw.cols,
                                       raw.rows, static_cast<int>(raw.step / sizeof(uint16_t)),
                                       depth_scale);
    } else {
      cv::Mat f32;
      raw.convertTo(f32, CV_32F, raw.depth() == CV_8U ? 1.0 / 255.0 : 1.0 / depth_scale);
      depth_m.assign(reinterpret_cast<const float*>(f32.data),
                     reinterpret_cast<const float*>(f32.data) + f32.total());
    }

    bcdl::Engine engine(argv[1], "");
    std::printf("model: %s  (inputs=%d outputs=%d)\n", engine.modelName().c_str(),
                engine.numInputs(), engine.numOutputs());

    bcdl::DepthRefiner refiner(engine);
    std::printf("encoder input: %dx%d\n", refiner.config().encoder_height,
                refiner.config().encoder_width);

    bcdl::RefinedDepth r =
        refiner.run(bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), depth_m.data(),
                    bgr.cols);

    size_t kept = 0;
    for (uint8_t v : r.mask) kept += v;
    std::printf("refined: %dx%d  range=[%.3f, %.3f] m  trusted=%.1f%%\n", r.width, r.height,
                r.vmin, r.vmax, 100.0 * static_cast<double>(kept) / r.mask.size());

    // The input depth is at the sensor's resolution; draw it at the model's.
    cv::Mat in_small;
    cv::Mat in_mat(bgr.rows, bgr.cols, CV_32F, depth_m.data());
    cv::resize(in_mat, in_small, cv::Size(r.width, r.height), 0, 0, cv::INTER_NEAREST);
    std::vector<float> in_vec(in_small.ptr<float>(), in_small.ptr<float>() + in_small.total());

    cv::Mat side;
    cv::hconcat(colorize(in_vec, r.height, r.width, r.vmin, r.vmax),
                colorize(r.depth, r.height, r.width, r.vmin, r.vmax), side);
    cv::imwrite(out_path, side);
    std::printf("wrote %s (input | refined)\n", out_path);

    if (intr_path != nullptr) {
      bcdl::Intrinsics k;
      if (!loadIntrinsics(intr_path, &k)) {
        std::fprintf(stderr, "could not parse intrinsics from %s\n", intr_path);
        return 1;
      }
      k = bcdl::scaleIntrinsics(k, bgr.cols, bgr.rows, r.width, r.height);
      const std::vector<float> pts = bcdl::depthToPointCloud(r, k);
      const std::string xyz = std::string(out_path) + ".xyz";
      if (FILE* fp = std::fopen(xyz.c_str(), "w")) {
        size_t n = 0;
        for (size_t i = 0; i < pts.size(); i += 3) {
          if (pts[i + 2] <= 0.0f) continue;
          std::fprintf(fp, "%.4f %.4f %.4f\n", pts[i], pts[i + 1], pts[i + 2]);
          ++n;
        }
        std::fclose(fp);
        std::printf("wrote %s (%zu points)\n", xyz.c_str(), n);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

// Stereo depth demo: a rectified stereo pair -> StereoPipeline (LAS2 .hbm) ->
// disparity (+ optional metric depth / validity mask), written as a
// side-by-side visualization (left | disparity | masked disparity).
//
//   ./stereo_demo las2.hbm stereo_sbs.png [out.png] [--mode crop|resize]
//                 [--fx F --baseline B] [--lr-check] [--disp-min D]
//
// `stereo_sbs` is a single image with the left and right views concatenated
// horizontally (the common ZED / TRT-demo layout). Use --left/--right for
// separate files. --mode MUST match the mode the .hbm was calibrated with.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bcdl/backend/engine.h"
#include "bcdl/pipeline/stereo_pipeline.h"

namespace {

const char* argval(int argc, char** argv, const char* key) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return nullptr;
}
bool hasflag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}

// Colorize a raw disparity map (TURBO), graying out masked-invalid pixels.
cv::Mat colorizeDisp(const bcdl::DepthMap& d, const std::vector<uint8_t>& mask) {
  cv::Mat gray(d.height, d.width, CV_8UC1);
  const float vmax = d.vmax > 0.0f ? d.vmax : 1.0f;
  for (int i = 0; i < d.height * d.width; ++i) {
    float t = d.data[i] / vmax;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    gray.data[i] = static_cast<uint8_t>(t * 255.0f + 0.5f);
  }
  cv::Mat color;
  cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
  if (!mask.empty()) {
    for (int i = 0; i < d.height * d.width; ++i)
      if (!mask[i]) color.at<cv::Vec3b>(i / d.width, i % d.width) = {60, 60, 60};
  }
  return color;
}

constexpr int kBarH = 28;
const cv::Scalar kCropClr(0, 255, 0);  // crop-box / linkage color (BGR green)

// Prepend a black title strip with `text` (white) to a panel.
cv::Mat titled(const cv::Mat& panel, const std::string& text) {
  cv::Mat bar = cv::Mat::zeros(kBarH, panel.cols, CV_8UC3);
  cv::putText(bar, text, {6, kBarH - 9}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
              {255, 255, 255}, 1, cv::LINE_AA);
  cv::Mat out;
  cv::vconcat(std::vector<cv::Mat>{bar, panel}, out);
  return out;
}

// Vertically center-pad a panel to H rows (black) for hconcat alignment.
cv::Mat padH(const cv::Mat& m, int H) {
  if (m.rows >= H) return m;
  const int top = (H - m.rows) / 2;
  cv::Mat o;
  cv::copyMakeBorder(m, o, top, H - m.rows - top, 0, 0, cv::BORDER_CONSTANT, {0, 0, 0});
  return o;
}

// A scale legend: TURBO bar labelled near->far. Height `body_h`.
cv::Mat legend(int body_h) {
  cv::Mat panel(body_h, 84, CV_8UC3, cv::Scalar(30, 30, 30));
  const int gh = body_h - 30;
  cv::Mat grad(gh, 28, CV_8UC1), cb;
  for (int y = 0; y < gh; ++y)
    grad.row(y).setTo(static_cast<uint8_t>(255 - 255 * y / std::max(1, gh - 1)));
  cv::applyColorMap(grad, cb, cv::COLORMAP_TURBO);
  cb.copyTo(panel(cv::Rect(10, 10, 28, gh)));
  cv::rectangle(panel, {10, 10, 28, gh}, {200, 200, 200}, 1);
  cv::putText(panel, "near", {42, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {255, 255, 255}, 1, cv::LINE_AA);
  cv::putText(panel, "far", {42, 10 + gh}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {255, 255, 255}, 1, cv::LINE_AA);
  return titled(panel, "scale");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <las2.hbm> <stereo_sbs.png> [out.png] "
                 "[--mode crop|resize] [--fx F --baseline B] [--lr-check] "
                 "[--disp-min D]\n  or: %s <las2.hbm> --left L.png --right R.png ...\n",
                 argv[0], argv[0]);
    return 1;
  }
  try {
    const char* hbm = argv[1];
    const char* mode = argval(argc, argv, "--mode");
    const char* left_p = argval(argc, argv, "--left");
    const char* right_p = argval(argc, argv, "--right");
    const char* fx_s = argval(argc, argv, "--fx");
    const char* base_s = argval(argc, argv, "--baseline");
    const char* dmin_s = argval(argc, argv, "--disp-min");
    // Output: explicit --out wins; else the 2nd positional in the sbs form
    // (argv[2]=sbs, argv[3]=out). In the --left/--right form there is no
    // positional out (argv[3] is a flag value), so it defaults — never guess a
    // positional there, or we'd clobber an input path.
    const char* out_opt = argval(argc, argv, "--out");
    const bool sbs_form = !(left_p && right_p) && argv[2][0] != '-';
    const char* sbs_p = sbs_form ? argv[2] : nullptr;
    const char* out_p = out_opt ? out_opt
                       : (sbs_form && argc > 3 && argv[3][0] != '-') ? argv[3]
                                                                     : "stereo_out.png";

    // Load + split into left/right BGR.
    cv::Mat left, right;
    if (left_p && right_p) {
      left = cv::imread(left_p, cv::IMREAD_COLOR);
      right = cv::imread(right_p, cv::IMREAD_COLOR);
    } else if (sbs_p) {
      cv::Mat sbs = cv::imread(sbs_p, cv::IMREAD_COLOR);
      if (sbs.empty()) throw std::runtime_error("cannot read stereo image");
      const int w = sbs.cols - (sbs.cols % 2);
      left = sbs(cv::Rect(0, 0, w / 2, sbs.rows)).clone();
      right = sbs(cv::Rect(w / 2, 0, w / 2, sbs.rows)).clone();
    } else {
      throw std::runtime_error("provide a sbs image or --left/--right");
    }
    if (left.empty() || right.empty()) throw std::runtime_error("empty input image");

    bcdl::Engine engine(hbm, "");
    std::printf("model: %s  (inputs=%d outputs=%d)  input=%dx%d\n",
                engine.modelName().c_str(), engine.numInputs(), engine.numOutputs(),
                left.cols, left.rows);

    bcdl::StereoConfig cfg;
    cfg.fit = (mode && std::strcmp(mode, "crop") == 0) ? bcdl::StereoFit::kCrop
                                                       : bcdl::StereoFit::kResize;
    if (fx_s && base_s) {
      cfg.fx = static_cast<float>(std::atof(fx_s));
      cfg.baseline = static_cast<float>(std::atof(base_s));
    }
    cfg.valid_mask = (dmin_s != nullptr) || hasflag(argc, argv, "--lr-check");
    if (dmin_s) cfg.disp_min = static_cast<float>(std::atof(dmin_s));
    cfg.lr_check = hasflag(argc, argv, "--lr-check");

    bcdl::StereoPipeline pipe(engine, cfg);
    bcdl::StereoResult r = pipe.process(left.data, right.data, left.cols, left.rows);
    const bcdl::DepthMap& d = r.disparity;
    std::printf("disparity: %dx%d  range=[%.2f, %.2f]  mode=%s\n", d.width, d.height,
                d.vmin, d.vmax, cfg.fit == bcdl::StereoFit::kCrop ? "crop" : "resize");
    if (!r.valid.empty()) {
      long n = 0;
      for (uint8_t v : r.valid) n += v;
      std::printf("valid pixels: %.1f%%\n", 100.0 * n / r.valid.size());
    }
    if (!r.depth.empty()) {
      float lo = 1e30f, hi = 0.0f;
      for (size_t i = 0; i < r.depth.size(); ++i) {
        const float z = r.depth[i];
        if (z > 0.0f && (r.valid.empty() || r.valid[i])) {
          lo = std::min(lo, z);
          hi = std::max(hi, z);
        }
      }
      if (hi > 0.0f) std::printf("depth(valid): %.2f ~ %.2f m\n", lo, hi);
    }

    // Annotated viz: input | disparity | scale legend. In crop mode the input
    // panel is the FULL original with the center-crop box drawn, and the SAME
    // green box is drawn around the disparity — so it reads at a glance that the
    // disparity covers exactly the boxed crop (narrow FOV is by design). Resize
    // mode shows the full frame. (The validity mask is still computed/printed;
    // it is just not rendered as a gray panel.)
    cv::Mat disp = colorizeDisp(d, {});
    std::vector<cv::Mat> content;
    std::vector<std::string> titles;
    int body_h;
    if (cfg.fit == bcdl::StereoFit::kCrop) {
      const int t = (left.rows - d.height) / 2, l = (left.cols - d.width) / 2;
      cv::Mat left_vis = left.clone();
      cv::rectangle(left_vis, cv::Rect(std::max(0, l), std::max(0, t), d.width, d.height),
                    kCropClr, 2);
      cv::putText(left_vis, "crop (model input)", {std::max(0, l) + 6, std::max(0, t) + 24},
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, kCropClr, 2, cv::LINE_AA);
      cv::rectangle(disp, {0, 0, disp.cols, disp.rows}, kCropClr, 2);
      body_h = std::max(left.rows, d.height);
      content = {padH(left_vis, body_h), padH(disp, body_h)};
      titles = {"input  (green = crop fed to model)", "disparity  near->far"};
    } else {
      cv::Mat fl;
      cv::resize(left, fl, cv::Size(d.width, d.height));
      body_h = d.height;
      content = {fl, disp};
      titles = {"input  (full frame)", "disparity  near->far"};
    }
    std::vector<cv::Mat> panels;
    for (size_t i = 0; i < content.size(); ++i) panels.push_back(titled(content[i], titles[i]));
    panels.push_back(legend(body_h));
    cv::Mat vis;
    cv::hconcat(panels, vis);
    cv::imwrite(out_p, vis);
    std::printf("OK: wrote %s\n", out_p);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  return 0;
}

// Image-embedding retrieval demo: embed a folder of images into an
// EmbeddingBank, then rank the whole gallery against one query image.
//
//   ./embed_demo siglip.hbm gallery_dir [query.jpg]
//
// With no query, the first gallery image is used — the top hit should then be
// that image itself at ~1.0, which is the quickest way to tell a working tower
// from a broken binding.
//
// TWO THINGS THIS DEMO EXISTS TO SHOW:
//
// 1. Pick the pooled submodel. An embedding .hbm is a PACKAGE: these towers ship
//    a pooled whole-image vector AND a per-patch feature grid in one file. The
//    demo enumerates them with Engine::modelNames() and binds the pooled one by
//    name. Taking the package default gets you whichever came first — and a
//    patch grid flattens to num_patches * width, which is not an embedding.
//
// 2. Preprocess with a squashing resize, NOT a letterbox. These towers are
//    trained on a direct resize to a square; letterbox bars are pixels they have
//    never seen and they move the output vector. Nothing needs un-letterboxing
//    on the way out, because an embedding carries no coordinates.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bcdl/bcdl.h"

namespace {

/// Squashing resize to size x size, BGR->RGB, x/127.5 - 1, HWC->NCHW.
std::vector<float> preprocess(const cv::Mat& bgr, int size) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
  std::vector<float> out(static_cast<size_t>(3) * size * size);
  for (int y = 0; y < size; ++y) {
    const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < size; ++x) {
      const size_t px = static_cast<size_t>(y) * size + x;
      // OpenCV is BGR; the tower wants RGB, hence the reversed channel index.
      for (int c = 0; c < 3; ++c) {
        out[static_cast<size_t>(c) * size * size + px] =
            row[x][2 - c] / 127.5f - 1.0f;
      }
    }
  }
  return out;
}

std::vector<std::string> listImages(const std::string& dir) {
  std::vector<cv::String> found;
  for (const char* pat : {"/*.jpg", "/*.jpeg", "/*.png"}) {
    std::vector<cv::String> hit;
    cv::glob(dir + pat, hit, false);
    found.insert(found.end(), hit.begin(), hit.end());
  }
  std::vector<std::string> paths(found.begin(), found.end());
  std::sort(paths.begin(), paths.end());
  return paths;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.hbm gallery_dir [query.jpg]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string gallery_dir = argv[2];

  try {
    // Show what is packed in the file, then bind the pooled head by name.
    const std::vector<std::string> submodels = bcdl::Engine::modelNames(model_path);
    std::printf("submodels in package:");
    for (const std::string& n : submodels) std::printf(" %s", n.c_str());
    std::printf("\n");

    std::string pooled;
    for (const std::string& n : submodels) {
      if (n.find("pool") != std::string::npos) pooled = n;
    }
    if (pooled.empty() && !submodels.empty()) pooled = submodels.back();

    bcdl::Engine engine(model_path, pooled);
    bcdl::ImageEmbedder embedder(engine);
    const std::vector<int> in_shape = engine.inputShape(0);
    const int size = in_shape.size() >= 4 ? in_shape[2] : 224;
    std::printf("bound '%s': input %dx%d, embedding dim %d\n\n",
                engine.modelName().c_str(), size, size, embedder.dim());

    const std::vector<std::string> paths = listImages(gallery_dir);
    if (paths.empty()) {
      std::fprintf(stderr, "no images found in %s\n", gallery_dir.c_str());
      return 1;
    }

    bcdl::EmbeddingBank bank;
    for (const std::string& p : paths) {
      cv::Mat img = cv::imread(p);
      if (img.empty()) {
        std::fprintf(stderr, "skipping unreadable %s\n", p.c_str());
        continue;
      }
      const std::vector<float> input = preprocess(img, size);
      engine.setInput(0, input.data(), input.size() * sizeof(float));
      engine.infer();
      bank.add(embedder.postprocess(), p);
    }
    std::printf("gallery: %d images embedded\n", bank.size());
    if (bank.size() == 0) return 1;

    const std::string query_path = argc > 3 ? argv[3] : paths.front();
    cv::Mat query_img = cv::imread(query_path);
    if (query_img.empty()) {
      std::fprintf(stderr, "cannot read query %s\n", query_path.c_str());
      return 1;
    }
    const std::vector<float> qin = preprocess(query_img, size);
    engine.setInput(0, qin.data(), qin.size() * sizeof(float));
    engine.infer();
    const std::vector<float> qvec = embedder.postprocess();

    std::printf("\nquery: %s\n", query_path.c_str());
    const std::vector<bcdl::EmbedMatch> hits = bank.search(qvec, 5);
    for (size_t i = 0; i < hits.size(); ++i) {
      std::printf("  %zu. %.4f  %s\n", i + 1, hits[i].score, hits[i].label.c_str());
    }
    return 0;
  } catch (const bcdl::Error& e) {
    std::fprintf(stderr, "bcdl error %d: %s\n", e.code(), e.what());
    return 2;
  }
}

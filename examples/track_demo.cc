// Detect-and-track demo: bcdl::TrackingPipeline (DetectionPipeline + ByteTracker)
// on a real YOLO NV12 .hbm.
//
// A single still image is panned to synthesise a short moving sequence (all
// objects translate together), so a real detector produces real, moving boxes
// frame-to-frame and ByteTracker can assign stable track_ids. This validates the
// detect->track wiring end-to-end on the board with a real model. (The pure
// tracker algorithm is separately checked on synthetic boxes in
// examples/tracks_ocr_check.cc.)
//
//   ./track_demo <det.hbm> <image.jpg> [frames] [out_dir]
//
// Prints, per frame, the detections, the active track ids, and each track's box.
// If `out_dir` is given, writes annotated frame_NN.jpg (boxes + id labels).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bcdl/backend/engine.h"
#include "bcdl/pipeline/tracking_pipeline.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <det.hbm> <image.jpg> [frames] [out_dir]\n",
                 argv[0]);
    return 1;
  }
  const std::string det_path = argv[1];
  const std::string image_path = argv[2];
  const int frames = argc > 3 ? std::atoi(argv[3]) : 12;
  const std::string out_dir = argc > 4 ? argv[4] : "";

  try {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);  // BGR
    if (img.empty()) throw std::runtime_error("cannot read image: " + image_path);
    const int W = img.cols, H = img.rows;

    bcdl::Engine engine(det_path);
    bcdl::PipelineConfig det_cfg;
    det_cfg.detect.num_classes = 80;     // COCO
    det_cfg.detect.conf_thresh = 0.25f;
    bcdl::TrackingPipeline pipe(engine, det_cfg, bcdl::ByteTrackConfig{});
    std::printf("track: %s  inputs=%d outputs=%d  %dx%d  frames=%d\n",
                engine.modelName().c_str(), engine.numInputs(), engine.numOutputs(),
                W, H, frames);

    if (!out_dir.empty()) std::filesystem::create_directories(out_dir);

    // Pan the image by `step` px/frame to make objects move uniformly.
    const double step = 10.0;
    std::set<int> all_ids;
    for (int f = 0; f < frames; ++f) {
      const double dx = step * f;
      cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, 0);
      cv::Mat frame;
      cv::warpAffine(img, frame, M, img.size(), cv::INTER_LINEAR,
                     cv::BORDER_REPLICATE);
      if (!frame.isContinuous()) frame = frame.clone();

      const std::vector<bcdl::Track> tracks =
          pipe.process(frame.ptr<uint8_t>(0), frame.cols, frame.rows);

      std::printf("frame %2d: dets=%zu tracks=%zu  ids=[", f,
                  pipe.lastDetections().size(), tracks.size());
      for (const auto& t : tracks) {
        std::printf("%d ", t.track_id);
        all_ids.insert(t.track_id);
      }
      std::printf("]\n");

      if (!out_dir.empty()) {
        cv::Mat vis = frame.clone();
        for (const auto& t : tracks) {
          cv::rectangle(vis, cv::Point((int)t.x1, (int)t.y1),
                        cv::Point((int)t.x2, (int)t.y2), cv::Scalar(0, 255, 0), 2);
          cv::putText(vis, "id" + std::to_string(t.track_id) + " c" +
                               std::to_string(t.class_id),
                      cv::Point((int)t.x1, (int)t.y1 - 5), cv::FONT_HERSHEY_SIMPLEX,
                      0.6, cv::Scalar(0, 255, 0), 2);
        }
        char name[64];
        std::snprintf(name, sizeof(name), "/frame_%02d.jpg", f);
        cv::imwrite(out_dir + name, vis);
      }
    }
    std::printf("OK: %zu distinct track ids over %d frames\n", all_ids.size(), frames);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "track_demo error: %s\n", e.what());
    return 2;
  }
  return 0;
}

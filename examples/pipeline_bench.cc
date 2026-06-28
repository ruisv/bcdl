// M4 benchmark: steady-state throughput of the streaming DetectionPipeline.
//
//   ./pipeline_bench model.hbm [W H] [iters]
//
// Builds a DetectionPipeline once (which allocates all scratch buffers up front:
// the letterboxed BGR canvas + the NV12 model-input target), then drives a
// synthetic W x H BGR frame through pipeline.process() `iters` times after a
// short warm-up. Because every buffer is reused, steady-state process() does NO
// per-frame heap allocation for images — this run measures that hot path
// (CPU letterbox + BGR->NV12 -> BPU infer -> decode/NMS), reporting total ms,
// ms/frame and FPS. Zero/gradient input won't yield real objects; this measures
// throughput, not accuracy (accuracy is verified in tests/test_detection.py).
//
// The pipeline auto-detects the head: a single fused output uses the YOLOv8
// tensor decode; a 6-output (cls,box)x3 model (standard RDK YOLO26 NV12 export,
// two Y/UV inputs) uses the anchor-free LTRB multi-scale decode.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bcdl/bcdl.h"
#include "bcdl/pipeline/detection_pipeline.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.hbm [W H] [iters]\n", argv[0]);
    return 1;
  }
  const int W = argc > 2 ? std::atoi(argv[2]) : 1280;
  const int H = argc > 3 ? std::atoi(argv[3]) : 720;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 200;
  if (W <= 0 || H <= 0 || iters <= 0) {
    std::fprintf(stderr, "bad W/H/iters\n");
    return 1;
  }

  try {
    bcdl::Engine engine(argv[1], "");
    std::printf("model: %s  (inputs=%d outputs=%d)\n", engine.modelName().c_str(),
                engine.numInputs(), engine.numOutputs());

    // input_w/input_h left at 0 => derived from the Engine's input[0] shape.
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;
    cfg.detect.layout = bcdl::DecodeLayout::kYoloV8;
    cfg.detect.conf_thresh = 0.25f;
    cfg.output_index = 0;
    bcdl::DetectionPipeline pipeline(engine, cfg);
    const char* head_name =
        pipeline.head() == bcdl::DetectHead::kYoloLtrb ? "YOLO-LTRB (multi-scale)"
                                                       : "single-tensor (YOLOv8)";
    std::printf("pipeline canvas: %dx%d   input frame: %dx%d   head: %s\n",
                pipeline.config().input_w, pipeline.config().input_h, W, H, head_name);

    // Synthetic BGR frame (row-contiguous HxWx3); a simple gradient.
    std::vector<uint8_t> frame(static_cast<std::size_t>(W) * H * 3);
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        uint8_t* px = frame.data() + (static_cast<std::size_t>(r) * W + c) * 3;
        px[0] = static_cast<uint8_t>(c & 0xFF);          // B
        px[1] = static_cast<uint8_t>(r & 0xFF);          // G
        px[2] = static_cast<uint8_t>((c + r) & 0xFF);    // R
      }
    }

    // Warm-up (first call also allocates the source wrapper; excluded from timing).
    const int warmup = 5;
    std::size_t last_dets = 0;
    for (int i = 0; i < warmup; ++i) {
      last_dets = pipeline.process(frame.data(), W, H).size();
    }

    // Timed steady-state loop — no image buffers are allocated per frame.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      last_dets = pipeline.process(frame.data(), W, H).size();
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per_frame = total_ms / iters;
    const double fps = ms_per_frame > 0.0 ? 1000.0 / ms_per_frame : 0.0;

    std::printf("iters=%d  total=%.2f ms  per-frame=%.3f ms  FPS=%.1f\n", iters,
                total_ms, ms_per_frame, fps);
    std::printf("last-frame detections: %zu\n", last_dets);
    std::printf("OK: streaming pipeline ran (buffers reused, zero per-frame alloc).\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

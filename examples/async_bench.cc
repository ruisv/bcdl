// M4 async overlap benchmark: sync vs threaded streaming throughput.
//
//   ./async_bench model.hbm [W H] [iters] [depth]
//
// Runs the SAME frames through the synchronous DetectionPipeline and the
// threaded AsyncDetectionPipeline and prints both throughputs. The async
// pipeline overlaps CPU preprocessing of later frames with BPU infer + decode of
// earlier ones, so its steady-state FPS approaches 1 / max(preproc, infer+decode)
// instead of the sync sum. Synthetic frames measure throughput, not accuracy.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bcdl/bcdl.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/detection_pipeline.h"

namespace {

std::vector<uint8_t> makeFrame(int W, int H) {
  std::vector<uint8_t> frame(static_cast<std::size_t>(W) * H * 3);
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      uint8_t* px = frame.data() + (static_cast<std::size_t>(r) * W + c) * 3;
      px[0] = static_cast<uint8_t>(c & 0xFF);
      px[1] = static_cast<uint8_t>(r & 0xFF);
      px[2] = static_cast<uint8_t>((c + r) & 0xFF);
    }
  return frame;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.hbm [W H] [iters] [depth]\n", argv[0]);
    return 1;
  }
  const int W = argc > 2 ? std::atoi(argv[2]) : 1280;
  const int H = argc > 3 ? std::atoi(argv[3]) : 720;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 300;
  const int depth = argc > 5 ? std::atoi(argv[5]) : 3;
  if (W <= 0 || H <= 0 || iters <= 0) {
    std::fprintf(stderr, "bad W/H/iters\n");
    return 1;
  }

  try {
    bcdl::Engine engine(argv[1], "");
    std::printf("model: %s  (inputs=%d outputs=%d)\n", engine.modelName().c_str(),
                engine.numInputs(), engine.numOutputs());

    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;
    cfg.detect.conf_thresh = 0.25f;
    const std::vector<uint8_t> frame = makeFrame(W, H);

    // -------- synchronous baseline --------
    {
      bcdl::DetectionPipeline sync(engine, cfg);
      for (int i = 0; i < 5; ++i) sync.process(frame.data(), W, H);  // warm-up
      const auto t0 = std::chrono::steady_clock::now();
      std::size_t dets = 0;
      for (int i = 0; i < iters; ++i) dets = sync.process(frame.data(), W, H).size();
      const double total = ms_since(t0);
      std::printf("sync : %d frames  %.1f ms  %.3f ms/frame  %.1f FPS  (last dets=%zu)\n",
                  iters, total, total / iters, 1000.0 * iters / total, dets);
    }

    // -------- threaded async overlap --------
    {
      bcdl::AsyncDetectionPipeline async(engine, cfg, depth);
      std::vector<bcdl::Detection> dets;
      // PRIME the pipeline with `depth` frames and keep them in flight — do NOT
      // drain here. Then each loop iteration submits one and pulls one, holding
      // ~`depth` frames in flight so the preproc/infer stages actually overlap.
      // (Draining after the prime would leave only 1 frame in flight and
      // serialize the stages, hiding the overlap entirely.)
      for (int i = 0; i < depth; ++i) async.submit(frame.data(), W, H);
      for (int i = 0; i < 5; ++i) {  // warm-up, still `depth` in flight
        async.submit(frame.data(), W, H);
        async.next(dets);
      }

      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        async.submit(frame.data(), W, H);
        async.next(dets);
      }
      const double total = ms_since(t0);
      std::printf("async: %d frames  %.1f ms  %.3f ms/frame  %.1f FPS  (depth=%d, last dets=%zu)\n",
                  iters, total, total / iters, 1000.0 * iters / total, depth, dets.size());
      async.finish();
      while (async.next(dets)) { /* drain the last `depth` in flight */ }
    }

    std::printf("OK: sync vs async streaming compared.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

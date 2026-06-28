// M4 async CORRECTNESS check: the threaded AsyncDetectionPipeline must return,
// in submission order, exactly what the synchronous DetectionPipeline returns
// for the same frames — proving the concurrent slot/queue handoff has no race,
// aliasing, or reordering.
//
//   ./async_check model.hbm frame.bgr W H
//
// frame.bgr is a raw interleaved HxWx3 uint8 BGR dump (e.g. from
// cv2.imread(...).tofile()). We build a SECOND, distinct frame (zeros) so the
// two have different detection sets, then push an interleaved A/B/.. pattern
// through async and assert every result matches the right reference.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>

#include "bcdl/bcdl.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/detection_pipeline.h"

namespace {

// Same frame -> same code path -> bit-identical floats, so an exact-ish compare
// (tiny epsilon for safety) is the right correctness bar.
bool sameDets(const std::vector<bcdl::Detection>& a,
              const std::vector<bcdl::Detection>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].class_id != b[i].class_id) return false;
    if (std::fabs(a[i].score - b[i].score) > 1e-4f) return false;
    if (std::fabs(a[i].x1 - b[i].x1) > 1e-2f || std::fabs(a[i].y1 - b[i].y1) > 1e-2f ||
        std::fabs(a[i].x2 - b[i].x2) > 1e-2f || std::fabs(a[i].y2 - b[i].y2) > 1e-2f)
      return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s model.hbm frame.bgr W H\n", argv[0]);
    return 1;
  }
  const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
  const std::size_t n = static_cast<std::size_t>(W) * H * 3;

  try {
    std::vector<uint8_t> A(n);
    std::ifstream f(argv[2], std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(A.data()), static_cast<std::streamsize>(n))) {
      std::fprintf(stderr, "could not read %zu BGR bytes from %s\n", n, argv[2]);
      return 1;
    }
    std::vector<uint8_t> B(n, 0);  // distinct frame: solid black -> no detections

    bcdl::Engine engine(argv[1], "");
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;
    cfg.detect.conf_thresh = 0.25f;

    // Synchronous reference detections for each frame.
    bcdl::DetectionPipeline sync(engine, cfg);
    const std::vector<bcdl::Detection> refA = sync.process(A.data(), W, H);
    const std::vector<bcdl::Detection> refB = sync.process(B.data(), W, H);
    std::printf("reference: frameA dets=%zu  frameB dets=%zu\n", refA.size(), refB.size());
    if (refA.empty()) {
      std::fprintf(stderr, "frameA produced no detections; check the BGR dump.\n");
      return 2;
    }

    // Interleaved submission pattern (A,B,A,A,B,B,A,...) repeated; the result
    // sequence MUST match this pattern's references in order.
    const char pattern[] = {'A', 'B', 'A', 'A', 'B', 'B', 'A', 'B'};
    const int reps = 12;
    std::vector<char> seq;
    for (int r = 0; r < reps; ++r)
      for (char c : pattern) seq.push_back(c);

    bcdl::AsyncDetectionPipeline async(engine, cfg, /*depth=*/3);
    int mismatches = 0, checked = 0, produced = 0;
    std::vector<bcdl::Detection> got;
    std::size_t out_idx = 0;

    for (std::size_t i = 0; i < seq.size(); ++i) {
      const uint8_t* src = (seq[i] == 'A') ? A.data() : B.data();
      async.submit(src, W, H);
      // Keep ~depth in flight: start pulling once primed.
      if (i >= 3) {
        async.next(got);
        ++produced;
        const auto& ref = (seq[out_idx] == 'A') ? refA : refB;
        if (!sameDets(got, ref)) ++mismatches;
        ++checked;
        ++out_idx;
      }
    }
    async.finish();
    while (async.next(got)) {  // drain the last in-flight frames
      ++produced;
      const auto& ref = (seq[out_idx] == 'A') ? refA : refB;
      if (!sameDets(got, ref)) ++mismatches;
      ++checked;
      ++out_idx;
    }

    std::printf("submitted=%zu produced=%d checked=%d mismatches=%d\n", seq.size(),
                produced, checked, mismatches);
    if (produced != static_cast<int>(seq.size())) {
      std::printf("FAIL: produced != submitted (frames lost or duplicated)\n");
      return 3;
    }
    if (mismatches != 0) {
      std::printf("FAIL: %d results disagree with the sync reference\n", mismatches);
      return 3;
    }
    std::printf("PASS: async matches sync for all %d frames, in order.\n", checked);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

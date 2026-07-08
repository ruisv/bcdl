#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "bcdl/pipeline/detection_pipeline.h"  // PipelineConfig, DetectHead, core
#include "bcdl/tasks/detection.h"              // Detection

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Threaded streaming detector that OVERLAPS CPU preprocessing of later frames
/// with BPU inference + decode of earlier ones.
///
/// Profiling the synchronous DetectionPipeline shows the per-frame cost split
/// across three stages that use different hardware:
///   preproc (CPU letterbox + BGR->NV12)  |  infer (BPU)  |  decode+NMS (CPU)
/// Running them serially wastes the BPU while the CPU preprocesses and vice
/// versa. This pipeline runs two worker threads connected by bounded channels:
///   preproc thread : BGR -> letterbox -> NV12 into one of `depth` slots
///   infer thread   : setInput -> Engine::infer -> HeadDecoder::postprocess
///                    (the Engine has a single thread of use, as it must)
/// so steady-state throughput approaches 1 / max(preproc, infer+decode) instead
/// of 1 / (preproc + infer + decode). Backpressure: every channel is bounded, so
/// submit() blocks when the pipeline is full rather than growing unboundedly.
///
/// Results are returned by next() in SUBMISSION order (the single preproc thread
/// and single infer thread both preserve FIFO). Typical streaming use:
///
///   AsyncDetectionPipeline p(engine, cfg, /*depth=*/3);
///   std::vector<Detection> dets;
///   int i = 0;
///   for (const Frame& f : stream) {
///     p.submit(f.bgr, f.w, f.h);        // blocks if full (backpressure)
///     if (i++ >= 3) p.next(dets);       // keep the pipeline full, drain in order
///   }
///   p.finish();
///   while (p.next(dets)) { /* handle the last in-flight results */ }
///
/// The same head auto-detection, 1-/2-input NV12 feeding, and float-input guard
/// as DetectionPipeline apply (shared core in detection_pipeline.h).
class AsyncDetectionPipeline {
 public:
  /// `depth` = number of in-flight frames / NV12 slots (>= 2 to overlap). Larger
  /// depth tolerates more jitter at the cost of latency + memory.
  AsyncDetectionPipeline(Engine& engine, PipelineConfig cfg, int depth = 3);
  ~AsyncDetectionPipeline();

  AsyncDetectionPipeline(const AsyncDetectionPipeline&) = delete;
  AsyncDetectionPipeline& operator=(const AsyncDetectionPipeline&) = delete;

  /// Enqueue a BGR frame (interleaved HxWx3 uint8, row stride == width*3). The
  /// frame bytes are COPIED, so the caller's buffer can be reused immediately.
  /// Blocks while the pipeline is full (backpressure). Returns false if the
  /// pipeline has been finish()ed (the frame is not accepted).
  bool submit(const uint8_t* bgr, int width, int height);

  /// Pop the next result, in submission order. Blocks until one is ready.
  /// Returns false once the pipeline is finished AND fully drained.
  bool next(std::vector<Detection>& out);

  /// Non-blocking variant of next(): fills `out` and returns true if a result is
  /// already available, else returns false immediately (nothing ready yet, or
  /// finished+drained). Never blocks.
  bool tryNext(std::vector<Detection>& out);

  /// Signal that no more frames will be submitted. After the in-flight frames
  /// drain, next() returns false. Idempotent. Called by the destructor.
  void finish();

  const PipelineConfig& config() const noexcept { return cfg_; }
  /// Resolved decoder family (kAuto replaced by the concrete choice).
  DetectHead head() const noexcept { return cfg_.head; }

  /// Per-stage SERVICE timing accumulated across the pipeline's lifetime (each
  /// stage timed on its own worker thread, so the fields sum to more than wall
  /// time — the max of the per-frame stages is what bounds throughput). Read
  /// after finish() + full drain for a settled value.
  StageProfile profile() const;

 private:
  PipelineConfig cfg_;  // resolved at construction (mirrors impl_'s copy)
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcdl

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bcdl/media/video_codec.h"                 // VideoDecConfig / hbVPVideoType
#include "bcdl/pipeline/detection_pipeline.h"       // PipelineConfig, StageProfile
#include "bcdl/tasks/detection.h"                    // Detection

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Full compressed-video → detections pipeline, entirely in C++ so the Python
/// (or any thin) caller only pumps bytes. It fuses the whole path:
///
///   Annex-B bytes --split AUs--> VideoDecoder (VPU) --NV12->BGR (CPU)-->
///     AsyncDetectionPipeline (CPU preproc ‖ BPU infer + NMS)
///
/// Three C++ worker stages run concurrently across frames (VPU decode ‖ CPU
/// preproc ‖ BPU infer), so steady-state throughput approaches
/// 1 / max(stage) — the same overlap the standalone examples/video_det_demo_async
/// demonstrates, but owned by the library. This is what lets a Python driver that
/// merely feeds an ffmpeg `-c copy` byte stream reach the C++ decode-bound ceiling
/// instead of being throttled by Python-side orchestration.
///
/// Feed granularity: submit() takes RAW Annex-B byte chunks of any size — the
/// pipeline segments them into access units internally, so the caller never has
/// to parse NAL start codes. Results come back from next()/tryNext() in decode
/// (submission) order.
///
/// Typical use (H.264 over RTSP; ffmpeg only demuxes with `-c copy`):
///   AsyncVideoDetectionPipeline p(engine, cfg, HB_VP_VIDEO_TYPE_H264, /*depth=*/4);
///   std::vector<Detection> dets;
///   while (int n = read(ffmpeg_stdout, buf, sizeof buf)) {
///     p.submit(buf, n);                 // blocks on backpressure
///     while (p.tryNext(dets)) { /* draw / count */ }
///   }
///   p.finish();
///   while (p.next(dets)) { /* drain the last in-flight frames */ }
///
/// NOTE: video decode uses the media_codec-based VideoDecoder, which handles both
/// H.264 and H.265 (reorder-correct). A hierarchical-GOP HEVC stream decodes its
/// base temporal layer only — see VideoDecoder for that caveat.
class AsyncVideoDetectionPipeline {
 public:
  /// `codec` is the elementary-stream type (HB_VP_VIDEO_TYPE_H264 / _H265).
  /// `depth` >= 2 is the number of in-flight frames per stage (overlap depth).
  AsyncVideoDetectionPipeline(Engine& engine, PipelineConfig cfg,
                              hbVPVideoType codec, int depth = 4);
  ~AsyncVideoDetectionPipeline();

  AsyncVideoDetectionPipeline(const AsyncVideoDetectionPipeline&) = delete;
  AsyncVideoDetectionPipeline& operator=(const AsyncVideoDetectionPipeline&) = delete;

  /// Feed a chunk of Annex-B compressed bytes (any size; internally reassembled
  /// into access units and decoded on the VPU). Blocks while the pipeline is
  /// full (backpressure). Returns false once finish()ed (chunk not accepted).
  bool submit(const uint8_t* data, std::size_t size);

  /// Pop the next frame's detections in decode order. Blocks until one is ready.
  /// Returns false once finished AND fully drained.
  bool next(std::vector<Detection>& out);

  /// Non-blocking next(): true if a result was ready (out filled), else false
  /// immediately. Never blocks — use it to drain while feeding.
  bool tryNext(std::vector<Detection>& out);

  /// Signal end of stream. Flushes any buffered trailing access unit, then the
  /// in-flight frames drain and next() returns false. Idempotent; also on dtor.
  void finish();

  /// Per-stage timing: decode_ms (VPU decode + NV12->BGR) plus the detector's
  /// preproc/infer/postproc. Read after finish() + full drain for a settled value.
  StageProfile profile() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcdl

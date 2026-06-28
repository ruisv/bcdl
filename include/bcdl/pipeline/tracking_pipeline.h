#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/pipeline/detection_pipeline.h"
#include "bcdl/tracks/byte_tracker.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Streaming detect-and-track: a DetectionPipeline (NV12 preproc + BPU infer +
/// decode/NMS, all buffer-reuse) feeding a ByteTracker once per frame. Returns
/// the frame's active tracks, each with a stable `track_id`, in ORIGINAL-image
/// pixels.
///
/// This is the thin composition the two halves were built for: DetectionPipeline
/// already un-letterboxes its boxes to original-image pixels (the coordinate
/// convention bcdl::Track / ByteTracker also use), so process() is just
/// `tracker.update(det.process(...))`. Per-frame image buffers are reused by the
/// underlying DetectionPipeline; the tracker keeps only its Kalman/lost state.
class TrackingPipeline {
 public:
  /// Build the pipeline. `det_cfg.input_w/input_h <= 0` are resolved from the
  /// Engine's input[0] (see DetectionPipeline). The model must take uint8 NV12
  /// input (the DetectionPipeline contract).
  TrackingPipeline(Engine& engine, PipelineConfig det_cfg = {},
                   ByteTrackConfig track_cfg = {})
      : det_(engine, std::move(det_cfg)), tracker_(track_cfg) {}

  TrackingPipeline(const TrackingPipeline&) = delete;
  TrackingPipeline& operator=(const TrackingPipeline&) = delete;

  /// Run one frame end-to-end and return its active tracks. `bgr` is interleaved
  /// HxWx3 uint8 (row stride == width*3); `width`/`height` are its dimensions.
  std::vector<Track> process(const uint8_t* bgr, int width, int height) {
    last_dets_ = det_.process(bgr, width, height);
    return tracker_.update(last_dets_);
  }

  /// Detections from the most recent process() (pre-association), for overlay /
  /// debugging.
  const std::vector<Detection>& lastDetections() const noexcept { return last_dets_; }
  /// Letterbox geometry of the most recent frame.
  const LetterboxInfo& lastLetterbox() const noexcept { return det_.lastLetterbox(); }

  /// Drop all tracklets and reset the id/frame counters (e.g. on a stream cut).
  void reset() { tracker_.reset(); }

  const DetectionPipeline& detection() const noexcept { return det_; }
  ByteTracker& tracker() noexcept { return tracker_; }

 private:
  DetectionPipeline det_;
  ByteTracker tracker_;
  std::vector<Detection> last_dets_;
};

}  // namespace bcdl

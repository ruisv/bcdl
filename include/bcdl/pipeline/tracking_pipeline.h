#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "bcdl/pipeline/detection_pipeline.h"
#include "bcdl/tasks/embedding.h"
#include "bcdl/tracks/byte_tracker.h"
#include "bcdl/tracks/reid.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Appearance settings for TrackingPipeline's optional ReID stage.
struct TrackingReidConfig {
  /// Only detections at or above this score are embedded. Everything below gets
  /// no appearance vector and is associated on geometry alone. This is a COST
  /// knob with teeth: the ReID model runs once per crop, so on a crowded frame
  /// it, not the detector, sets the frame time.
  float min_score = 0.5f;
  /// Cap on crops embedded per frame, so one pathological frame cannot stall the
  /// stream. Detections are considered in the order the detector emitted them
  /// (score-descending after NMS), so the cap drops the least confident.
  /// <= 0 means no cap.
  int max_crops = 32;
  /// Crop normalization (see reidPreprocess).
  ReidConfig crop;
};

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

  /// Same, plus appearance: `reid` is an Engine holding a ReID model that maps
  /// one crop to one embedding. process() then embeds each qualifying detection
  /// and associates on geometry AND appearance, which is what keeps ids through
  /// occlusions and crossings that motion alone loses.
  ///
  /// The ReID model's input shape is read from `reid` (a person model is
  /// typically 1x3x256x128), so nothing here needs to be told the crop size.
  TrackingPipeline(Engine& engine, Engine& reid, PipelineConfig det_cfg = {},
                   ByteTrackConfig track_cfg = {}, TrackingReidConfig reid_cfg = {});

  TrackingPipeline(const TrackingPipeline&) = delete;
  TrackingPipeline& operator=(const TrackingPipeline&) = delete;

  /// Run one frame end-to-end and return its active tracks. `bgr` is interleaved
  /// HxWx3 uint8 (row stride == width*3); `width`/`height` are its dimensions.
  std::vector<Track> process(const uint8_t* bgr, int width, int height);

  /// Detections from the most recent process() (pre-association), for overlay /
  /// debugging.
  const std::vector<Detection>& lastDetections() const noexcept { return last_dets_; }
  /// Letterbox geometry of the most recent frame.
  const LetterboxInfo& lastLetterbox() const noexcept { return det_.lastLetterbox(); }

  /// Drop all tracklets and reset the id/frame counters (e.g. on a stream cut).
  void reset() { tracker_.reset(); }

  const DetectionPipeline& detection() const noexcept { return det_; }
  ByteTracker& tracker() noexcept { return tracker_; }
  /// True when this pipeline was built with a ReID Engine.
  bool hasReid() const noexcept { return reid_ != nullptr; }
  /// Number of crops embedded on the most recent frame (0 without ReID) — the
  /// term that makes frame time scale with crowd size.
  int lastEmbedCount() const noexcept { return last_embed_count_; }

 private:
  DetectionPipeline det_;
  ByteTracker tracker_;
  Engine* reid_ = nullptr;
  std::unique_ptr<ImageEmbedder> embedder_;
  TrackingReidConfig reid_cfg_;
  int reid_w_ = 0, reid_h_ = 0;
  int last_embed_count_ = 0;
  std::vector<float> crop_buf_;                  // reused per crop
  std::vector<std::vector<float>> embeddings_;   // reused per frame
  std::vector<Detection> last_dets_;
};

}  // namespace bcdl

#pragma once

#include <memory>
#include <vector>

#include "bcdl/tasks/detection.h"

namespace bcdl {

// ===========================================================================
// ByteTrack multi-object tracker (pure algorithm, no model)
// ===========================================================================
//
// Associates per-frame detection boxes into temporally stable tracklets, each
// carrying a persistent `track_id`. This is a faithful C++ port of the classic
// ByteTrack reference (Zhang et al., ECCV 2022): an 8-dimensional constant-
// velocity Kalman filter predicts each tracklet forward, and a TWO-STAGE
// IoU association recovers both confident and low-confidence detections:
//
//   1. High-score detections are matched (IoU, score-fused) against the union
//      of currently-tracked and recently-lost tracklets.
//   2. Remaining still-tracked tracklets get a SECOND chance to match the
//      leftover low-score detections (this is the "BYTE" idea — keep the weak
//      boxes instead of discarding them, to bridge occlusions).
//   3. Unconfirmed (single-frame) tracklets match the leftover high-score
//      detections; unmatched ones are dropped immediately.
//   4. Leftover high-score detections above `high_thresh` spawn new tracklets.
//   5. Lost tracklets older than `max_time_lost` frames are removed.
//
// Coordinates are ORIGINAL-image pixels throughout (the same convention as
// `bcdl::Detection`); the model->original un-letterbox is assumed done already
// by the detector. Association uses box geometry only; the detection's
// `class_id` is passed through to the matched track's output for convenience.

/// One tracked object for the current frame, in ORIGINAL-image pixels.
struct Track {
  int track_id;   ///< stable identity, unique within this tracker instance
  float x1, y1, x2, y2;  ///< tlbr box (Kalman-filtered current estimate)
  float score;    ///< score of the detection last associated to this track
  int class_id;   ///< class of the last associated detection (passed through)
};

/// Tunables for the ByteTracker. Defaults follow the reference implementation.
struct ByteTrackConfig {
  /// High/low score split. Detections with score > track_thresh take part in
  /// the first (strong) association; those in (0.1, track_thresh] are reserved
  /// for the second (weak) association.
  float track_thresh = 0.5f;
  /// Minimum score for a leftover high-score detection to START a new track
  /// (reference: det_thresh = track_thresh + 0.1).
  float high_thresh = 0.6f;
  /// First-association gate: a (track,det) pair is only accepted when its
  /// IoU distance (1 - IoU) does not exceed this value.
  float match_thresh = 0.8f;
  /// Lost tracklets are kept for `frame_rate/30 * track_buffer` frames before
  /// being permanently removed.
  int track_buffer = 30;
  int frame_rate = 30;
};

/// Stateful tracker. Construct once, call `update()` once per frame with that
/// frame's detections, and receive the currently-active tracks. All tracklet
/// bookkeeping (Kalman state, lost buffer, id counter) lives behind the pImpl.
class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackConfig cfg = {});
  ~ByteTracker();

  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;
  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;

  /// Advance one frame. `detections` are this frame's boxes (original-image
  /// tlbr, with score/class_id). Returns the active, confirmed tracks for this
  /// frame, each with a stable `track_id`.
  std::vector<Track> update(const std::vector<Detection>& detections);

  /// Drop all tracklets and reset the frame counter and id counter to 0. After
  /// reset the next created track gets id 1 again.
  void reset();

  const ByteTrackConfig& config() const { return cfg_; }

 private:
  ByteTrackConfig cfg_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcdl

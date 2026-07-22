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
// by the detector. The detection's `class_id` is passed through to the matched
// track's output for convenience.
//
// APPEARANCE (ReID). Geometry alone loses an identity whenever a target is
// occluded for longer than the motion model stays believable, or when two
// targets cross. Passing a per-detection appearance embedding to the two-argument
// `update()` switches on BoT-SORT association: each tracklet keeps an
// EMA-smoothed embedding template, and the first association's cost becomes
//
//     cost = min(IoU_dist, gated_cosine_dist)
//
// so appearance can only ever RESCUE a match that geometry was about to miss —
// it never vetoes one. Two gates keep it from inventing matches: a pair whose
// raw IoU distance exceeds `proximity_thresh` is too far apart to be the same
// object no matter how similar it looks, and a cosine distance above
// `appearance_thresh` is not similar enough to count. The low-score second
// association stays pure IoU (its crops are the blurry / occluded ones whose
// embeddings are least trustworthy). Embeddings come from a ReID model run over
// the detection crops — see `tracks/reid.h`.

/// One tracked object for the current frame, in ORIGINAL-image pixels.
struct Track {
  int track_id;   ///< stable identity, unique within this tracker instance
  float x1, y1, x2, y2;  ///< tlbr box (Kalman-filtered current estimate)
  float score;    ///< score of the detection last associated to this track
  int class_id;   ///< class of the last associated detection (passed through)
};

/// BoostTrack++ additions, layered on top of the BoT-SORT association above.
///
/// Each is switchable ON ITS OWN and every one is OFF by default, so a tracker
/// configured `{}` behaves exactly as it did before this existed and each
/// addition's contribution can be measured against that baseline rather than
/// assumed. Formulas and constants follow the reference implementation
/// (Stanojevic & Todorovic, BoostTrack / BoostTrack++), not the paper text —
/// the two disagree on at least one sign (the paper's prose reads as though a
/// CONFIDENT tracklet expands its box, while the code expands by
/// `1 - confidence`, i.e. an UNRELIABLE tracklet gets the wider search).
///
/// A tracklet's confidence here is not its detection score: it decays as the
/// tracklet goes unmatched and ramps up while it is young
/// (`0.9^(7-age)` before age 7, `0.9^(frames_since_update-1)` after), so it
/// measures how much the tracklet's predicted box should be trusted.
struct BoostConfig {
  /// Add BoostTrack's Mahalanobis and shape terms to the first association's
  /// cost, each weighted by the detection-tracklet confidence product. Alone,
  /// these can only pull a cost DOWN, so `min_iou` below is what stops them
  /// from admitting a geometrically impossible pair.
  bool rich_similarity = false;
  /// Replace the plain IoU in the first association with soft buffered IoU:
  /// both boxes are grown in proportion to `1 - tracklet_confidence`, so a
  /// tracklet that has been coasting searches a wider area. Growth factors are
  /// the reference's 0.25 for the detection and 0.5 for the tracklet.
  bool soft_biou = false;
  /// Raise the scores of detections the tracklets say are probably real, BEFORE
  /// the high/low split — so a boosted detection joins the strong association
  /// instead of the weak one. This is where BoostTrack gets its "detect more
  /// objects" result; it changes recall, not just matching.
  bool boost_detections = false;

  /// Weights on the cost bonuses (reference defaults).
  float lambda_iou = 0.5f;
  float lambda_mhd = 0.25f;
  float lambda_shape = 0.25f;
  /// Raw-IoU floor applied AFTER assignment when `rich_similarity` is on: a
  /// matched pair below this is rejected. The reference relies on the same
  /// guard, and without it the bonuses can drag an absurd pair under the gate.
  float min_iou = 0.3f;

  /// DLO soft boost: `score <- max(score, a*score + (1-a)*S^1.5)` where S is
  /// the detection's best combined similarity to any tracklet.
  float dlo_alpha = 0.65f;
  /// DLO varying threshold: a detection whose similarity to some tracklet
  /// exceeds a threshold that relaxes from `vt_start` to `vt_end` over
  /// `vt_steps` frames-since-update is promoted to just above `high_thresh`.
  /// Tracklets that have been unmatched for longer are given more benefit of
  /// the doubt, because their predicted box has drifted.
  float vt_start = 0.95f;
  float vt_end = 0.80f;
  int vt_steps = 20;
  /// DUO boost: a low-scoring detection that NO tracklet explains (Mahalanobis
  /// distance beyond the 4-dof chi-square 99% point to every one of them) is
  /// promoted, since it is more likely a new object than a false positive.
  /// Overlapping candidates are deduplicated at this IoU, keeping the highest
  /// score, so one blob does not spawn several tracks.
  bool duo = true;
  float duo_iou = 0.3f;
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

  // --- appearance association (only consulted when embeddings are passed) ---

  /// Appearance is only allowed to rescue pairs that are already plausible in
  /// space: a (track,det) pair whose RAW IoU distance (1 - IoU, before score
  /// fusion) exceeds this is excluded from the appearance term entirely.
  float proximity_thresh = 0.5f;
  /// Cosine-distance gate, on the reference's halved scale: the raw cosine
  /// distance (1 - cos, in [0,2]) is divided by 2 before comparison, so 0.25
  /// here means "reject below cos = 0.5". Pairs above it are excluded.
  float appearance_thresh = 0.25f;
  /// Base smoothing factor for a tracklet's embedding template
  /// (`t = a*t + (1-a)*new`). This is the value used for a FULLY confident
  /// detection; the effective factor is raised toward 1 as the detection's
  /// score falls toward `track_thresh`, so a half-occluded or motion-blurred
  /// crop barely moves the template instead of poisoning it (Deep OC-SORT's
  /// dynamic appearance). Must be in [0, 1].
  float ema_alpha = 0.95f;

  /// BoostTrack++ additions (all off by default — see BoostConfig).
  BoostConfig boost;
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

  /// Advance one frame with appearance. `embeddings` runs PARALLEL to
  /// `detections` — entry i is the ReID vector for detection i — and is
  /// L2-normalized internally, so raw model output is fine.
  ///
  /// An EMPTY entry means "no appearance for this detection", and that pair
  /// falls back to pure IoU. That is the intended way to skip the ReID model on
  /// the cheap detections: embed only the high-score crops (the ones that reach
  /// the first association) and pass empty vectors for the rest. Passing an
  /// all-empty list is exactly equivalent to the one-argument `update()`.
  ///
  /// Throws Error(-1) if `embeddings.size()` differs from `detections.size()`,
  /// or if two non-empty entries disagree in length — a silently misaligned or
  /// ragged embedding list would otherwise degrade matching invisibly.
  std::vector<Track> update(const std::vector<Detection>& detections,
                            const std::vector<std::vector<float>>& embeddings);

  /// Warp every tracklet's predicted position by a camera-motion transform,
  /// before the next update(). `affine` is a row-major 2x3 matrix mapping a
  /// point in the PREVIOUS frame to the current one — exactly what
  /// cv::estimateAffinePartial2D / cv::findTransformECC return.
  ///
  /// WHY THE CALLER SUPPLIES IT. A tracklet's constant-velocity model describes
  /// how the target moves in the world; when the CAMERA moves, every predicted
  /// box is wrong by the camera's motion and the tracker blames the targets.
  /// Estimating that motion needs the frames, which this class never sees — it
  /// takes boxes. So the estimate comes from wherever the pixels are: the C++
  /// TrackingPipeline does it internally, and the Python path can pass whatever
  /// OpenCV gives it.
  ///
  /// Only the box position and size are warped, not the velocities: the
  /// velocity is in world terms and a one-frame camera jolt should not be
  /// interpreted as the target accelerating.
  ///
  /// For a static camera this is unnecessary — skip the call rather than
  /// passing an identity, and nothing is spent.
  void applyCameraMotion(const float affine[6]);

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

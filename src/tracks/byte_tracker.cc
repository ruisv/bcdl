#include "bcdl/tracks/byte_tracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bcdl/core/status.h"
#include "bcdl/tracks/reid.h"

// All numeric tracking machinery (the 8-D Kalman filter, the Hungarian
// assignment, the STrack state machine) lives here in the .cc behind the
// pImpl so the public header stays a tiny data interface. Heavy math uses
// `double`; only the final tlbr output is cast back to float.

namespace bcdl {
namespace {

// ===========================================================================
// Fixed-size linear algebra (no Eigen / no third party)
// ===========================================================================
//
// Vec8 is the Kalman state mean [cx, cy, a, h, vcx, vcy, va, vh] (a = w/h).
// Mat8 is a row-major 8x8 matrix: element (i,j) at index i*8 + j.

using Vec8 = std::array<double, 8>;
using Mat8 = std::array<double, 64>;  // row-major

inline double& at(Mat8& m, int i, int j) { return m[i * 8 + j]; }
inline double at(const Mat8& m, int i, int j) { return m[i * 8 + j]; }

// C = A * B  (8x8 by 8x8).
Mat8 matmul(const Mat8& A, const Mat8& B) {
  Mat8 C{};
  for (int i = 0; i < 8; ++i) {
    for (int k = 0; k < 8; ++k) {
      const double a = at(A, i, k);
      if (a == 0.0) continue;  // F is sparse; skip cheaply
      for (int j = 0; j < 8; ++j) C[i * 8 + j] += a * at(B, k, j);
    }
  }
  return C;
}

Mat8 transpose8(const Mat8& A) {
  Mat8 T{};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) at(T, i, j) = at(A, j, i);
  return T;
}

// Cholesky factorization of a 4x4 symmetric positive-definite matrix S into a
// lower-triangular L with S = L*L^T. Diagonal is clamped to a tiny positive
// floor to stay robust against round-off making a near-zero pivot negative.
void cholesky4(const double S[4][4], double L[4][4]) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) L[i][j] = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = S[i][j];
      for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
      if (i == j) {
        if (sum < 1e-12) sum = 1e-12;
        L[i][i] = std::sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
}

// Solve S x = b given the Cholesky factor L of S (S = L*L^T) via forward then
// back substitution. b and x are length 4 and may NOT alias.
void cholSolve4(const double L[4][4], const double b[4], double x[4]) {
  double y[4];
  for (int i = 0; i < 4; ++i) {  // L y = b
    double sum = b[i];
    for (int k = 0; k < i; ++k) sum -= L[i][k] * y[k];
    y[i] = sum / L[i][i];
  }
  for (int i = 3; i >= 0; --i) {  // L^T x = y
    double sum = y[i];
    for (int k = i + 1; k < 4; ++k) sum -= L[k][i] * x[k];
    x[i] = sum / L[i][i];
  }
}

// ===========================================================================
// Kalman filter (8-D box tracker, identical model to the ByteTrack reference)
// ===========================================================================
//
// State [cx, cy, a, h, vcx, vcy, va, vh] with a constant-velocity transition.
// The measurement is xyah = (cx, cy, a, h). Process / observation noise scale
// with box height h (the reference's "a bit hacky" relative-uncertainty trick).
class KalmanFilter {
 public:
  static constexpr double kStdPos = 1.0 / 20.0;   // _std_weight_position
  static constexpr double kStdVel = 1.0 / 160.0;  // _std_weight_velocity

  KalmanFilter() {
    // Motion matrix F: 8x8 identity with dt(=1) coupling pos<-vel.
    motion_.fill(0.0);
    for (int i = 0; i < 8; ++i) at(motion_, i, i) = 1.0;
    for (int i = 0; i < 4; ++i) at(motion_, i, i + 4) = 1.0;
    motion_t_ = transpose8(motion_);
  }

  // Create a track from an unassociated measurement (xyah). Velocities start
  // at 0; covariance is diagonal, scaled by h.
  void initiate(const std::array<double, 4>& meas, Vec8& mean, Mat8& cov) const {
    mean = {meas[0], meas[1], meas[2], meas[3], 0.0, 0.0, 0.0, 0.0};
    const double h = meas[3];
    const std::array<double, 8> std = {
        2 * kStdPos * h, 2 * kStdPos * h, 1e-2,           2 * kStdPos * h,
        10 * kStdVel * h, 10 * kStdVel * h, 1e-5,          10 * kStdVel * h};
    cov.fill(0.0);
    for (int i = 0; i < 8; ++i) at(cov, i, i) = std[i] * std[i];
  }

  // Predict one step: mean = F mean; cov = F cov F^T + Q.
  void predict(Vec8& mean, Mat8& cov) const {
    const double h = mean[3];
    const std::array<double, 8> std = {
        kStdPos * h, kStdPos * h, 1e-2,  kStdPos * h,
        kStdVel * h, kStdVel * h, 1e-5,  kStdVel * h};

    Vec8 nm{};
    for (int i = 0; i < 8; ++i) {
      double s = 0.0;
      for (int k = 0; k < 8; ++k) s += at(motion_, i, k) * mean[k];
      nm[i] = s;
    }
    mean = nm;

    cov = matmul(matmul(motion_, cov), motion_t_);
    for (int i = 0; i < 8; ++i) at(cov, i, i) += std[i] * std[i];
  }

  // Project state to measurement space: pmean = H mean (first 4 dims),
  // pcov = H cov H^T + R (top-left 4x4 block plus innovation covariance).
  void project(const Vec8& mean, const Mat8& cov, std::array<double, 4>& pmean,
               double pcov[4][4]) const {
    const double h = mean[3];
    const std::array<double, 4> std = {kStdPos * h, kStdPos * h, 1e-1, kStdPos * h};
    for (int i = 0; i < 4; ++i) pmean[i] = mean[i];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) pcov[i][j] = at(cov, i, j);
    for (int i = 0; i < 4; ++i) pcov[i][i] += std[i] * std[i];
  }

  // Correction step with measurement xyah. Kalman gain K solves
  //   S K^T = (cov H^T)^T   where S = pcov (4x4 SPD).
  // Equivalently, row i of K (length 4) = S^{-1} * cov[i][0:4]; we Cholesky-
  // factor S once and back-solve the 8 right-hand sides (this is the explicit
  // form of scipy's cho_solve in the reference).
  void update(Vec8& mean, Mat8& cov, const std::array<double, 4>& meas) const {
    std::array<double, 4> pmean;
    double pcov[4][4];
    project(mean, cov, pmean, pcov);

    double L[4][4];
    cholesky4(pcov, L);

    // gain is 8x4: gain[i] = S^{-1} * (cov[i][0..3]).
    double gain[8][4];
    for (int i = 0; i < 8; ++i) {
      double b[4] = {at(cov, i, 0), at(cov, i, 1), at(cov, i, 2), at(cov, i, 3)};
      cholSolve4(L, b, gain[i]);
    }

    // innovation = measurement - projected_mean (length 4).
    double innov[4];
    for (int k = 0; k < 4; ++k) innov[k] = meas[k] - pmean[k];

    // new_mean = mean + gain * innovation.
    for (int i = 0; i < 8; ++i) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += gain[i][k] * innov[k];
      mean[i] += s;
    }

    // new_cov = cov - gain * S * gain^T.
    // M = gain * S (8x4); new_cov[i][j] -= sum_k M[i][k] * gain[j][k].
    double M[8][4];
    for (int i = 0; i < 8; ++i) {
      for (int c = 0; c < 4; ++c) {
        double s = 0.0;
        for (int k = 0; k < 4; ++k) s += gain[i][k] * pcov[k][c];
        M[i][c] = s;
      }
    }
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        double s = 0.0;
        for (int k = 0; k < 4; ++k) s += M[i][k] * gain[j][k];
        at(cov, i, j) -= s;
      }
    }
  }

 private:
  Mat8 motion_{};
  Mat8 motion_t_{};
};

// ===========================================================================
// Track state machine (STrack)
// ===========================================================================

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

struct Box {  // tlbr helper, reuses bcdl::iou via a Detection view
  float x1, y1, x2, y2;
};

inline float boxIoU(const Box& a, const Box& b) {
  Detection da{a.x1, a.y1, a.x2, a.y2, 0.0f, 0};
  Detection db{b.x1, b.y1, b.x2, b.y2, 0.0f, 0};
  return iou(da, db);
}

class STrack {
 public:
  // A detection turned into a (not-yet-activated) tracklet candidate.
  STrack(float tlwh_x, float tlwh_y, float tlwh_w, float tlwh_h, float score,
         int class_id, std::vector<float> feat = {})
      : tlwh_init_{tlwh_x, tlwh_y, tlwh_w, tlwh_h},
        score_(score),
        class_id_(class_id),
        curr_feat_(std::move(feat)) {
    l2Normalize(curr_feat_);
  }

  // --- geometry --------------------------------------------------------
  // Current (top-left x, y, w, h). Before activation: the raw detection box;
  // after activation: derived from the Kalman mean.
  std::array<double, 4> tlwh() const {
    if (!mean_set_) {
      return {tlwh_init_[0], tlwh_init_[1], tlwh_init_[2], tlwh_init_[3]};
    }
    double w = mean_[2] * mean_[3];  // a * h
    double h = mean_[3];
    return {mean_[0] - w / 2.0, mean_[1] - h / 2.0, w, h};
  }

  Box tlbr() const {
    auto t = tlwh();
    return Box{static_cast<float>(t[0]), static_cast<float>(t[1]),
               static_cast<float>(t[0] + t[2]), static_cast<float>(t[1] + t[3])};
  }

  // tlwh -> xyah (center x, center y, aspect = w/h, height).
  static std::array<double, 4> tlwhToXyah(const std::array<double, 4>& t) {
    return {t[0] + t[2] / 2.0, t[1] + t[3] / 2.0, t[2] / t[3], t[3]};
  }
  std::array<double, 4> toXyah() const { return tlwhToXyah(tlwh()); }

  // --- lifecycle -------------------------------------------------------
  // Zero the height-velocity for non-Tracked tracks before predicting (the
  // reference's multi_predict trick), then advance the Kalman state.
  void predict(const KalmanFilter& kf) {
    if (state_ != TrackState::Tracked) mean_[7] = 0.0;
    kf.predict(mean_, cov_);
  }

  void activate(const KalmanFilter& kf, int frame_id, int new_id) {
    track_id_ = new_id;
    auto xyah = tlwhToXyah(
        {tlwh_init_[0], tlwh_init_[1], tlwh_init_[2], tlwh_init_[3]});
    kf.initiate(xyah, mean_, cov_);
    mean_set_ = true;
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = (frame_id == 1);  // only frame-1 starts confirmed
    frame_id_ = frame_id;
    start_frame_ = frame_id;
    smooth_feat_ = curr_feat_;  // this STrack IS the spawning detection
  }

  // Re-acquire a lost/leftover track from a new detection (keeps the id).
  void reActivate(const KalmanFilter& kf, const STrack& det, int frame_id,
                  double alpha) {
    auto t = det.tlwh();
    kf.update(mean_, cov_, tlwhToXyah(t));
    updateFeatures(det.currFeat(), alpha);
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    score_ = det.score_;
    class_id_ = det.class_id_;
  }

  // Update a matched, still-tracked track from a new detection.
  void update(const KalmanFilter& kf, const STrack& det, int frame_id,
              double alpha) {
    frame_id_ = frame_id;
    ++tracklet_len_;
    auto t = det.tlwh();
    kf.update(mean_, cov_, tlwhToXyah(t));
    updateFeatures(det.currFeat(), alpha);
    state_ = TrackState::Tracked;
    is_activated_ = true;
    score_ = det.score_;
    class_id_ = det.class_id_;
  }

  // --- appearance ------------------------------------------------------
  // Fold a detection's (already L2-normalized) embedding into this tracklet's
  // template. `alpha` is the caller-computed, confidence-adjusted smoothing
  // factor; the result is re-normalized so the template stays a unit vector and
  // cosine against it stays a plain dot product.
  void updateFeatures(const std::vector<float>& feat, double alpha) {
    if (feat.empty()) return;
    curr_feat_ = feat;
    if (smooth_feat_.size() != feat.size()) {  // first sighting
      smooth_feat_ = feat;
      return;
    }
    for (size_t i = 0; i < smooth_feat_.size(); ++i) {
      smooth_feat_[i] =
          static_cast<float>(alpha * smooth_feat_[i] + (1.0 - alpha) * feat[i]);
    }
    l2Normalize(smooth_feat_);
  }

  const std::vector<float>& smoothFeat() const { return smooth_feat_; }
  const std::vector<float>& currFeat() const { return curr_feat_; }

  // --- BoostTrack++ support ---------------------------------------------
  // How much this tracklet's PREDICTED box deserves to be trusted, in [0,1].
  // Distinct from the detection score: it ramps up while the tracklet is young
  // and decays once it stops being matched, which is exactly when its
  // constant-velocity prediction starts drifting.
  double confidence(int frame_id) const {
    constexpr double kCoef = 0.9;
    constexpr int kYoung = 7;
    const int age = frame_id - start_frame_;
    if (age < kYoung) return std::pow(kCoef, kYoung - age);
    const int since = std::max(1, frame_id - frame_id_);
    return std::pow(kCoef, since - 1);
  }

  // Frames since this tracklet was last matched (>= 1 after a predict()).
  int sinceUpdate(int frame_id) const { return std::max(1, frame_id - frame_id_); }

  // Squared Mahalanobis distance from this tracklet's state to a measurement,
  // using the DIAGONAL of the state covariance — the reference's approximation,
  // kept because the off-diagonal terms of an 8-D constant-velocity filter are
  // small and the full solve would cost a Cholesky per pair.
  double mahalanobisSq(const std::array<double, 4>& meas) const {
    if (!mean_set_) return std::numeric_limits<double>::max();
    double acc = 0.0;
    for (int i = 0; i < 4; ++i) {
      const double var = at(cov_, i, i);
      if (var <= 1e-12) return std::numeric_limits<double>::max();
      const double d = meas[i] - mean_[i];
      acc += d * d / var;
    }
    return acc;
  }

  // Warp the tracklet's box by a camera-motion affine, leaving velocity alone
  // (see ByteTracker::applyCameraMotion).
  void cameraUpdate(const float m[6]) {
    if (!mean_set_) return;
    const Box b = tlbr();
    const double x1 = m[0] * b.x1 + m[1] * b.y1 + m[2];
    const double y1 = m[3] * b.x1 + m[4] * b.y1 + m[5];
    const double x2 = m[0] * b.x2 + m[1] * b.y2 + m[2];
    const double y2 = m[3] * b.x2 + m[4] * b.y2 + m[5];
    const double w = x2 - x1, h = y2 - y1;
    if (w <= 0.0 || h <= 0.0) return;  // degenerate warp: leave the track alone
    mean_[0] = x1 + w / 2.0;
    mean_[1] = y1 + h / 2.0;
    mean_[2] = w / h;
    mean_[3] = h;
  }

  void markLost() { state_ = TrackState::Lost; }
  void markRemoved() { state_ = TrackState::Removed; }

  // --- accessors -------------------------------------------------------
  int trackId() const { return track_id_; }
  int classId() const { return class_id_; }
  float score() const { return score_; }
  TrackState state() const { return state_; }
  bool isActivated() const { return is_activated_; }
  int frameId() const { return frame_id_; }
  int startFrame() const { return start_frame_; }

 private:
  std::array<double, 4> tlwh_init_;  // raw detection box (pre-activation)
  Vec8 mean_{};
  Mat8 cov_{};
  bool mean_set_ = false;

  float score_;
  int class_id_;
  std::vector<float> curr_feat_;    ///< this frame's embedding (unit norm)
  std::vector<float> smooth_feat_;  ///< EMA template (unit norm), empty if unused
  int track_id_ = 0;
  int tracklet_len_ = 0;
  int frame_id_ = 0;
  int start_frame_ = 0;
  bool is_activated_ = false;
  TrackState state_ = TrackState::New;
};

using STrackPtr = std::shared_ptr<STrack>;

// ===========================================================================
// Association: IoU cost, score fusion, Hungarian assignment
// ===========================================================================

// cost[i][j] = 1 - IoU(track_i, det_j). Stored row-major, R*C.
std::vector<double> iouDistance(const std::vector<STrackPtr>& tracks,
                                const std::vector<STrackPtr>& dets) {
  const int R = static_cast<int>(tracks.size());
  const int C = static_cast<int>(dets.size());
  std::vector<double> cost(static_cast<size_t>(R) * C, 0.0);
  std::vector<Box> tb(R), db(C);
  for (int i = 0; i < R; ++i) tb[i] = tracks[i]->tlbr();
  for (int j = 0; j < C; ++j) db[j] = dets[j]->tlbr();
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      cost[i * C + j] = 1.0 - static_cast<double>(boxIoU(tb[i], db[j]));
  return cost;
}

// Fuse detection confidence into the IoU cost:
//   dist = 1 - (1 - dist) * det.score   (reference fuse_score).
void fuseScore(std::vector<double>& cost, int R, int C,
               const std::vector<STrackPtr>& dets) {
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) {
      double iou_sim = 1.0 - cost[i * C + j];
      cost[i * C + j] = 1.0 - iou_sim * static_cast<double>(dets[j]->score());
    }
}

// BoT-SORT appearance fusion, applied to a first-association cost matrix.
//
//   cost = min(iou_cost_after_score_fusion, gated_cosine_distance)
//
// `raw_iou` is the IoU distance BEFORE score fusion, which is what the
// proximity gate is defined against (score fusion would otherwise let a
// confident detection widen its own spatial gate). A pair is excluded from the
// appearance term — leaving it at its geometric cost — when any of these hold:
//   · either side has no embedding (caller skipped ReID for that crop)
//   · the two sides carry different class ids (no cross-class appearance
//     matching; there is no shared appearance space between classes)
//   · they are too far apart (raw IoU distance > proximity_thresh)
//   · they do not look alike enough (cosine distance > appearance_thresh)
// Taking the MINIMUM is what makes this safe: appearance can only lower a cost,
// so a bad embedding can never break a match that geometry already had.
void fuseAppearance(std::vector<double>& cost, const std::vector<double>& raw_iou,
                    int R, int C, const std::vector<STrackPtr>& tracks,
                    const std::vector<STrackPtr>& dets,
                    const ByteTrackConfig& cfg) {
  for (int i = 0; i < R; ++i) {
    const auto& tf = tracks[i]->smoothFeat();
    if (tf.empty()) continue;
    for (int j = 0; j < C; ++j) {
      const auto& df = dets[j]->currFeat();
      if (df.empty() || df.size() != tf.size()) continue;
      if (tracks[i]->classId() != dets[j]->classId()) continue;
      if (raw_iou[i * C + j] > cfg.proximity_thresh) continue;
      // Both vectors are unit norm, so cosine is a dot product. Halve the
      // distance to the reference's scale, on which appearance_thresh is set.
      double dot = 0.0;
      for (size_t k = 0; k < tf.size(); ++k) dot += static_cast<double>(tf[k]) * df[k];
      const double emb_dist = (1.0 - dot) / 2.0;
      if (emb_dist > cfg.appearance_thresh) continue;
      cost[i * C + j] = std::min(cost[i * C + j], emb_dist);
    }
  }
}

// ===========================================================================
// BoostTrack++ similarity terms
// ===========================================================================
//
// Constants are the reference implementation's. Note where the code and the
// paper's prose disagree: the expansion below scales with (1 - confidence), so
// it is the UNRELIABLE tracklet that gets the wider search window, which is the
// behaviour that makes sense and the one the published numbers came from.

/// Soft buffered IoU: grow both boxes in proportion to how little the tracklet
/// is trusted, then take the IoU. `track_conf` in [0,1].
double softBIoU(const Box& det, const Box& trk, double track_conf) {
  constexpr double kDet = 0.25;   // detection growth factor
  constexpr double kTrk = 0.5;    // tracklet growth factor
  const double slack = 1.0 - track_conf;
  const double dx = (det.x2 - det.x1) * slack * kDet;
  const double dy = (det.y2 - det.y1) * slack * kDet;
  const double tx = (trk.x2 - trk.x1) * slack * kTrk;
  const double ty = (trk.y2 - trk.y1) * slack * kTrk;

  const double ax1 = det.x1 - dx, ay1 = det.y1 - dy;
  const double ax2 = det.x2 + dx, ay2 = det.y2 + dy;
  const double bx1 = trk.x1 - tx, by1 = trk.y1 - ty;
  const double bx2 = trk.x2 + tx, by2 = trk.y2 + ty;

  const double w = std::max(0.0, std::min(ax2, bx2) - std::max(ax1, bx1));
  const double h = std::max(0.0, std::min(ay2, by2) - std::max(ay1, by1));
  const double inter = w * h;
  const double uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return uni > 0.0 ? inter / uni : 0.0;
}

/// Shape agreement in [0,1]: exp of the negative relative width and height
/// mismatch. This is the reference's CORRECTED form (`s_sim_corr`); its default
/// path divides the height term by the width scale, which reads as a typo.
double shapeSimilarity(const Box& det, const Box& trk) {
  const double dw = det.x2 - det.x1, dh = det.y2 - det.y1;
  const double tw = trk.x2 - trk.x1, th = trk.y2 - trk.y1;
  const double mw = std::max(dw, tw), mh = std::max(dh, th);
  if (mw <= 0.0 || mh <= 0.0) return 0.0;
  return std::exp(-(std::abs(dw - tw) / mw + std::abs(dh - th) / mh));
}

/// 4-dof chi-square 99% point: beyond this a detection is not plausibly the
/// same object as the tracklet, however close the boxes happen to look.
constexpr double kMahalanobisLimit = 13.2767;

/// Turn a matrix of squared Mahalanobis distances into similarities, following
/// the reference: clamp at the chi-square limit, invert, softmax DOWN EACH
/// TRACK's column, and zero out everything that was clamped. The softmax makes
/// this a competition between detections for one tracklet rather than an
/// absolute score, which is why it is not simply exp(-d).
std::vector<double> mahalanobisSimilarity(const std::vector<double>& dist, int R, int C) {
  std::vector<double> sim(static_cast<size_t>(R) * C, 0.0);
  if (R == 0 || C == 0) return sim;
  for (int j = 0; j < C; ++j) {
    double sum = 0.0;
    for (int i = 0; i < R; ++i) {
      const double d = dist[i * C + j];
      if (d > kMahalanobisLimit) continue;
      sum += std::exp(kMahalanobisLimit - d);
    }
    if (sum <= 0.0) continue;
    for (int i = 0; i < R; ++i) {
      const double d = dist[i * C + j];
      if (d > kMahalanobisLimit) continue;
      sim[i * C + j] = std::exp(kMahalanobisLimit - d) / sum;
    }
  }
  return sim;
}

struct Assignment {
  std::vector<std::pair<int, int>> matches;  // (row=track, col=det)
  std::vector<int> u_rows;                    // unmatched tracks
  std::vector<int> u_cols;                    // unmatched dets
};

// Minimum-cost assignment with a gating threshold, mirroring the reference's
// lap.lapjv(extend_cost=True, cost_limit=thresh): find the optimal full
// assignment on a square-padded matrix via the O(n^3) Hungarian (Kuhn-Munkres)
// algorithm, then keep only real (row,col) pairs whose cost <= thresh.
//
// `cost` is R*C row-major. Detection counts are small (tens), so O(n^3) is fine.
Assignment linearAssignment(const std::vector<double>& cost, int R, int C,
                            double thresh) {
  Assignment out;
  if (R == 0 || C == 0) {
    for (int i = 0; i < R; ++i) out.u_rows.push_back(i);
    for (int j = 0; j < C; ++j) out.u_cols.push_back(j);
    return out;
  }

  const int n = std::max(R, C);  // pad to square
  const double BIG = 1e9;
  // a[1..n][1..n], 1-indexed for the classic potentials-based algorithm.
  std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, 0.0));
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j)
      a[i][j] = (i <= R && j <= C) ? cost[(i - 1) * C + (j - 1)] : BIG;

  const double INF = std::numeric_limits<double>::max();
  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);  // p[j] = row matched to col j
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, INF);
    std::vector<char> used(n + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = -1;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // p[j] = row assigned to column j. Keep real pairs under the gate.
  std::vector<char> row_used(R, 0), col_used(C, 0);
  for (int j = 1; j <= n; ++j) {
    const int i = p[j];  // 1-indexed row
    if (i >= 1 && i <= R && j <= C && cost[(i - 1) * C + (j - 1)] <= thresh) {
      out.matches.emplace_back(i - 1, j - 1);
      row_used[i - 1] = 1;
      col_used[j - 1] = 1;
    }
  }
  for (int i = 0; i < R; ++i)
    if (!row_used[i]) out.u_rows.push_back(i);
  for (int j = 0; j < C; ++j)
    if (!col_used[j]) out.u_cols.push_back(j);
  return out;
}

// ===========================================================================
// Track-list set operations (joint / sub / remove-duplicate)
// ===========================================================================

std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a,
                                    const std::vector<STrackPtr>& b) {
  std::vector<STrackPtr> res;
  std::vector<int> seen;
  for (const auto& t : a) {
    seen.push_back(t->trackId());
    res.push_back(t);
  }
  for (const auto& t : b) {
    const int id = t->trackId();
    if (std::find(seen.begin(), seen.end(), id) == seen.end()) {
      seen.push_back(id);
      res.push_back(t);
    }
  }
  return res;
}

// a minus b, by track_id.
std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a,
                                  const std::vector<STrackPtr>& b) {
  std::vector<STrackPtr> res;
  for (const auto& t : a) {
    bool in_b = false;
    for (const auto& s : b)
      if (s->trackId() == t->trackId()) {
        in_b = true;
        break;
      }
    if (!in_b) res.push_back(t);
  }
  return res;
}

// Drop near-duplicate tracks across two lists (IoU distance < 0.15): keep the
// longer-lived one, remove the shorter-lived duplicate.
void removeDuplicateStracks(std::vector<STrackPtr>& a,
                            std::vector<STrackPtr>& b) {
  const int R = static_cast<int>(a.size());
  const int C = static_cast<int>(b.size());
  if (R == 0 || C == 0) return;
  auto cost = iouDistance(a, b);
  std::vector<char> dupa(R, 0), dupb(C, 0);
  for (int p = 0; p < R; ++p)
    for (int q = 0; q < C; ++q) {
      if (cost[p * C + q] >= 0.15) continue;
      const int timep = a[p]->frameId() - a[p]->startFrame();
      const int timeq = b[q]->frameId() - b[q]->startFrame();
      if (timep > timeq)
        dupb[q] = 1;
      else
        dupa[p] = 1;
    }
  std::vector<STrackPtr> ra, rb;
  for (int i = 0; i < R; ++i)
    if (!dupa[i]) ra.push_back(a[i]);
  for (int j = 0; j < C; ++j)
    if (!dupb[j]) rb.push_back(b[j]);
  a.swap(ra);
  b.swap(rb);
}

}  // namespace

// ===========================================================================
// ByteTracker::Impl — the per-frame update pipeline
// ===========================================================================

struct ByteTracker::Impl {
  ByteTrackConfig cfg;
  KalmanFilter kf;

  std::vector<STrackPtr> tracked;  // currently tracked
  std::vector<STrackPtr> lost;     // recently lost (in the recovery buffer)

  int frame_id = 0;
  int id_count = 0;  // instance-level track_id counter
  int max_time_lost = 30;
  int embed_dim = 0;  // ReID width, latched on the first embedding ever seen

  explicit Impl(ByteTrackConfig c) : cfg(c) {
    max_time_lost = static_cast<int>(
        std::lround(static_cast<double>(cfg.frame_rate) / 30.0 * cfg.track_buffer));
  }

  int nextId() { return ++id_count; }

  void reset() {
    tracked.clear();
    lost.clear();
    frame_id = 0;
    id_count = 0;
    embed_dim = 0;
  }

  // Smoothing factor for folding a detection of this score into a template.
  // Scales from `ema_alpha` at full confidence up to 1.0 (= no update at all)
  // as the score approaches the high/low split, so the crops least likely to
  // show the whole target contribute least to what the target looks like.
  double emaAlpha(float score) const {
    const double span = 1.0 - static_cast<double>(cfg.track_thresh);
    double trust = 1.0;
    if (span > 1e-6) trust = (static_cast<double>(score) - cfg.track_thresh) / span;
    trust = std::clamp(trust, 0.0, 1.0);
    const double a = static_cast<double>(cfg.ema_alpha);
    return a + (1.0 - a) * (1.0 - trust);
  }

  std::vector<Track> update(const std::vector<Detection>& dets,
                            const std::vector<std::vector<float>>& embeddings);

  // BoostTrack++ pieces (no-ops unless the matching BoostConfig flag is set).
  void boostDetections(const std::vector<Detection>& dets,
                       const std::vector<STrackPtr>& pool,
                       std::vector<float>& scores) const;
  void addBoostBonuses(std::vector<double>& cost, const std::vector<double>& raw_iou,
                       int R, int C, const std::vector<STrackPtr>& tracks,
                       const std::vector<STrackPtr>& dets) const;
};

// Combined detection-to-tracklet similarity, S in [0,1]: the mean of soft
// buffered IoU, Mahalanobis similarity and shape agreement. BoostTrack++ uses
// the CONSENSUS of three different notions of "same object" rather than any one
// of them, so a detection has to look right in position, motion and size before
// its score is raised.
std::vector<double> combinedSimilarity(const std::vector<Box>& det_boxes,
                                       const std::vector<std::array<double, 4>>& det_xyah,
                                       const std::vector<STrackPtr>& pool, int frame_id) {
  const int R = static_cast<int>(det_boxes.size());
  const int C = static_cast<int>(pool.size());
  std::vector<double> mh(static_cast<size_t>(R) * C, 0.0);
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) mh[i * C + j] = pool[j]->mahalanobisSq(det_xyah[i]);
  const std::vector<double> mh_sim = mahalanobisSimilarity(mh, R, C);

  std::vector<double> s(static_cast<size_t>(R) * C, 0.0);
  for (int j = 0; j < C; ++j) {
    const double conf = pool[j]->confidence(frame_id);
    const Box tb = pool[j]->tlbr();
    for (int i = 0; i < R; ++i) {
      s[i * C + j] = (mh_sim[i * C + j] + shapeSimilarity(det_boxes[i], tb) +
                      softBIoU(det_boxes[i], tb, conf)) / 3.0;
    }
  }
  return s;
}

void ByteTracker::Impl::boostDetections(const std::vector<Detection>& dets,
                                        const std::vector<STrackPtr>& pool,
                                        std::vector<float>& scores) const {
  const int R = static_cast<int>(dets.size());
  const int C = static_cast<int>(pool.size());
  if (R == 0 || C == 0) return;

  std::vector<Box> det_boxes(R);
  std::vector<std::array<double, 4>> det_xyah(R);
  for (int i = 0; i < R; ++i) {
    det_boxes[i] = Box{dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2};
    det_xyah[i] = STrack::tlwhToXyah({dets[i].x1, dets[i].y1,
                                      static_cast<double>(dets[i].x2 - dets[i].x1),
                                      static_cast<double>(dets[i].y2 - dets[i].y1)});
  }

  const std::vector<double> S = combinedSimilarity(det_boxes, det_xyah, pool, frame_id);
  const auto& b = cfg.boost;

  for (int i = 0; i < R; ++i) {
    double best = 0.0;
    bool vouched = false;
    for (int j = 0; j < C; ++j) {
      const double sij = S[i * C + j];
      best = std::max(best, sij);
      // Varying threshold: the longer a tracklet has gone unmatched, the more
      // its predicted box has drifted, so demand less similarity from it.
      const double step = (b.vt_start - b.vt_end) / std::max(1, b.vt_steps);
      const double thr = std::max(b.vt_start - (pool[j]->sinceUpdate(frame_id) - 1) * step,
                                  static_cast<double>(b.vt_end));
      if (sij > thr) vouched = true;
    }
    // Soft boost: blend the detection's own score with its best similarity,
    // never lowering it.
    const double soft = b.dlo_alpha * scores[i] + (1.0 - b.dlo_alpha) * std::pow(best, 1.5);
    scores[i] = static_cast<float>(std::max(static_cast<double>(scores[i]), soft));
    if (vouched) scores[i] = std::max(scores[i], cfg.high_thresh + 1e-5f);
  }

  if (!b.duo) return;

  // DUO: a weak detection that NO tracklet can explain is more likely a new
  // object than a false positive, so promote it. "Cannot explain" is the
  // Mahalanobis limit, not IoU — a tracklet may overlap a detection and still
  // be a poor motion explanation for it.
  std::vector<int> candidates;
  for (int i = 0; i < R; ++i) {
    if (scores[i] >= cfg.high_thresh) continue;
    double min_mh = std::numeric_limits<double>::max();
    for (int j = 0; j < C; ++j)
      min_mh = std::min(min_mh, pool[j]->mahalanobisSq(det_xyah[i]));
    if (min_mh > kMahalanobisLimit) candidates.push_back(i);
  }
  // Deduplicate overlapping candidates, keeping the strongest: one blob of
  // unexplained boxes should promote one detection, not a cluster of them.
  for (int a : candidates) {
    bool best_of_cluster = true;
    for (int c : candidates) {
      if (c == a) continue;
      if (boxIoU(det_boxes[a], det_boxes[c]) > b.duo_iou &&
          (scores[c] > scores[a] || (scores[c] == scores[a] && c < a))) {
        best_of_cluster = false;
        break;
      }
    }
    if (best_of_cluster) scores[a] = std::max(scores[a], cfg.high_thresh + 1e-4f);
  }
}

void ByteTracker::Impl::addBoostBonuses(std::vector<double>& cost,
                                        const std::vector<double>& raw_iou, int R, int C,
                                        const std::vector<STrackPtr>& tracks,
                                        const std::vector<STrackPtr>& dets) const {
  const auto& b = cfg.boost;
  if (R == 0 || C == 0) return;

  std::vector<Box> det_boxes(C);
  std::vector<std::array<double, 4>> det_xyah(C);
  for (int j = 0; j < C; ++j) {
    det_boxes[j] = dets[j]->tlbr();
    det_xyah[j] = dets[j]->toXyah();
  }

  // Mahalanobis similarity is normalized DOWN each tracklet's column, so it has
  // to be built over the whole matrix rather than per pair.
  std::vector<double> mh(static_cast<size_t>(C) * R, 0.0);
  for (int j = 0; j < C; ++j)
    for (int i = 0; i < R; ++i) mh[j * R + i] = tracks[i]->mahalanobisSq(det_xyah[j]);
  const std::vector<double> mh_sim = mahalanobisSimilarity(mh, C, R);

  for (int i = 0; i < R; ++i) {
    const double tconf = tracks[i]->confidence(frame_id);
    const Box tb = tracks[i]->tlbr();
    for (int j = 0; j < C; ++j) {
      const double iou = 1.0 - raw_iou[i * C + j];
      // The confidence product gates the IoU and shape bonuses the way the
      // reference does: a pair that is not already plausible in space gets no
      // bonus at all, which is what keeps the bonuses from inventing matches.
      const double conf = (iou < b.min_iou) ? 0.0 : tconf * dets[j]->score();
      double bonus = b.lambda_mhd * mh_sim[j * R + i];
      bonus += b.lambda_iou * conf * iou;
      bonus += b.lambda_shape * conf * shapeSimilarity(det_boxes[j], tb);
      cost[i * C + j] -= bonus;
    }
  }
}

std::vector<Track> ByteTracker::Impl::update(
    const std::vector<Detection>& dets,
    const std::vector<std::vector<float>>& embeddings) {
  ++frame_id;

  // --- validate the embedding list before it can quietly skew matching -----
  const bool has_embeddings = !embeddings.empty();
  if (has_embeddings) {
    if (embeddings.size() != dets.size()) {
      throw Error(-1,
                  "ByteTracker::update: embeddings must run parallel to "
                  "detections (one entry each, empty entries allowed)");
    }
    for (const auto& e : embeddings) {
      if (e.empty()) continue;  // "no appearance for this detection"
      const int d = static_cast<int>(e.size());
      if (embed_dim == 0) embed_dim = d;
      if (d != embed_dim) {
        throw Error(-1,
                    "ByteTracker::update: embedding width changed mid-stream "
                    "(all ReID vectors must come from one model)");
      }
    }
  }

  std::vector<STrackPtr> activated;  // matched & confirmed this frame
  std::vector<STrackPtr> refind;     // re-acquired lost tracks
  std::vector<STrackPtr> lost_now;   // newly lost this frame
  std::vector<STrackPtr> removed_now;

  // --- partition existing tracked into confirmed vs unconfirmed ----------
  std::vector<STrackPtr> unconfirmed;
  std::vector<STrackPtr> tracked_stracks;
  for (const auto& t : tracked) {
    if (t->isActivated())
      tracked_stracks.push_back(t);
    else
      unconfirmed.push_back(t);
  }

  // Kalman predict happens BEFORE the high/low split, not after: detection
  // boosting compares detections against the tracklets' predicted boxes, and
  // its whole purpose is to change which side of the split a detection lands
  // on. With boosting off this is the same work in a different order.
  std::vector<STrackPtr> strack_pool = jointStracks(tracked_stracks, lost);
  for (const auto& t : strack_pool) t->predict(kf);  // Kalman predict (multi)

  // --- BoostTrack++: raise scores the tracklets vouch for ----------------
  std::vector<float> scores(dets.size());
  for (size_t i = 0; i < dets.size(); ++i) scores[i] = dets[i].score;
  if (cfg.boost.boost_detections) boostDetections(dets, strack_pool, scores);

  // --- split detections by score into high / low pools -------------------
  auto makeStrack = [&](size_t i) {
    const Detection& d = dets[i];
    const float w = d.x2 - d.x1;
    const float h = d.y2 - d.y1;
    std::vector<float> feat;
    if (has_embeddings) feat = embeddings[i];
    return std::make_shared<STrack>(d.x1, d.y1, w, h, scores[i], d.class_id,
                                    std::move(feat));
  };

  std::vector<STrackPtr> detections;        // high score (> track_thresh)
  std::vector<STrackPtr> detections_second;  // low score (0.1, track_thresh]
  for (size_t i = 0; i < dets.size(); ++i) {
    if (scores[i] > cfg.track_thresh) {
      detections.push_back(makeStrack(i));
    } else if (scores[i] > 0.1f) {
      detections_second.push_back(makeStrack(i));
    }
  }

  // --- Step 2: first association (high score) ----------------------------

  {
    const int R = static_cast<int>(strack_pool.size());
    const int C = static_cast<int>(detections.size());
    auto raw_iou = iouDistance(strack_pool, detections);
    auto cost = raw_iou;
    if (cfg.boost.soft_biou) {
      // Widen the overlap test for tracklets whose prediction is stale. The
      // gates below still use RAW IoU, so this loosens matching without
      // loosening what counts as geometrically possible.
      for (int i = 0; i < R; ++i) {
        const double conf = strack_pool[i]->confidence(frame_id);
        const Box tb = strack_pool[i]->tlbr();
        for (int j = 0; j < C; ++j)
          cost[i * C + j] = 1.0 - softBIoU(detections[j]->tlbr(), tb, conf);
      }
    }
    fuseScore(cost, R, C, detections);
    if (has_embeddings) {
      fuseAppearance(cost, raw_iou, R, C, strack_pool, detections, cfg);
    }
    if (cfg.boost.rich_similarity) {
      addBoostBonuses(cost, raw_iou, R, C, strack_pool, detections);
    }
    auto as = linearAssignment(cost, R, C, cfg.match_thresh);
    if (cfg.boost.rich_similarity) {
      // The bonuses only ever lower a cost, so on their own they could drag a
      // geometrically impossible pair under the gate. Re-check the RAW IoU of
      // everything that matched and hand back anything that fails.
      Assignment kept;
      std::vector<char> row_used(R, 0), col_used(C, 0);
      for (const auto& m : as.matches) {
        if (1.0 - raw_iou[m.first * C + m.second] >= cfg.boost.min_iou) {
          kept.matches.push_back(m);
          row_used[m.first] = 1;
          col_used[m.second] = 1;
        }
      }
      for (int i = 0; i < R; ++i)
        if (!row_used[i]) kept.u_rows.push_back(i);
      for (int j = 0; j < C; ++j)
        if (!col_used[j]) kept.u_cols.push_back(j);
      as = std::move(kept);
    }
    for (const auto& m : as.matches) {
      auto& track = strack_pool[m.first];
      const auto& det = detections[m.second];
      const double alpha = emaAlpha(det->score());
      if (track->state() == TrackState::Tracked) {
        track->update(kf, *det, frame_id, alpha);
        activated.push_back(track);
      } else {
        track->reActivate(kf, *det, frame_id, alpha);
        refind.push_back(track);
      }
    }

    // --- Step 3: second association (low score), only still-tracked rows --
    std::vector<STrackPtr> r_tracked;
    for (int i : as.u_rows)
      if (strack_pool[i]->state() == TrackState::Tracked)
        r_tracked.push_back(strack_pool[i]);

    const int R2 = static_cast<int>(r_tracked.size());
    const int C2 = static_cast<int>(detections_second.size());
    auto cost2 = iouDistance(r_tracked, detections_second);
    auto as2 = linearAssignment(cost2, R2, C2, 0.5);
    // Pure IoU by design — these are the weak boxes, whose crops are exactly
    // the occluded / blurred ones. If the caller did embed them anyway, the
    // confidence-adaptive alpha keeps them from moving the template much.
    for (const auto& m : as2.matches) {
      auto& track = r_tracked[m.first];
      const auto& det = detections_second[m.second];
      const double alpha = emaAlpha(det->score());
      if (track->state() == TrackState::Tracked) {
        track->update(kf, *det, frame_id, alpha);
        activated.push_back(track);
      } else {
        track->reActivate(kf, *det, frame_id, alpha);
        refind.push_back(track);
      }
    }
    for (int it : as2.u_rows) {
      auto& track = r_tracked[it];
      if (track->state() != TrackState::Lost) {
        track->markLost();
        lost_now.push_back(track);
      }
    }

    // --- unconfirmed tracks vs leftover high-score detections ------------
    std::vector<STrackPtr> det_left;
    for (int j : as.u_cols) det_left.push_back(detections[j]);

    const int R3 = static_cast<int>(unconfirmed.size());
    const int C3 = static_cast<int>(det_left.size());
    auto raw_iou3 = iouDistance(unconfirmed, det_left);
    auto cost3 = raw_iou3;
    fuseScore(cost3, R3, C3, det_left);
    if (has_embeddings) {
      fuseAppearance(cost3, raw_iou3, R3, C3, unconfirmed, det_left, cfg);
    }
    auto as3 = linearAssignment(cost3, R3, C3, 0.7);
    for (const auto& m : as3.matches) {
      const auto& det = det_left[m.second];
      unconfirmed[m.first]->update(kf, *det, frame_id, emaAlpha(det->score()));
      activated.push_back(unconfirmed[m.first]);
    }
    for (int it : as3.u_rows) {
      auto& track = unconfirmed[it];
      track->markRemoved();
      removed_now.push_back(track);
    }

    // --- Step 4: init new tracks from leftover high-score detections -----
    for (int inew : as3.u_cols) {
      auto& track = det_left[inew];
      if (track->score() < cfg.high_thresh) continue;
      track->activate(kf, frame_id, nextId());
      activated.push_back(track);
    }
  }

  // --- Step 5: expire lost tracks older than the recovery buffer ---------
  for (const auto& track : lost) {
    if (frame_id - track->frameId() > max_time_lost) {
      track->markRemoved();
      removed_now.push_back(track);
    }
  }

  // --- merge / dedup the persistent lists --------------------------------
  std::vector<STrackPtr> new_tracked;
  for (const auto& t : tracked)
    if (t->state() == TrackState::Tracked) new_tracked.push_back(t);
  new_tracked = jointStracks(new_tracked, activated);
  new_tracked = jointStracks(new_tracked, refind);

  lost = subStracks(lost, new_tracked);
  for (const auto& t : lost_now) lost.push_back(t);
  lost = subStracks(lost, removed_now);

  removeDuplicateStracks(new_tracked, lost);
  tracked = std::move(new_tracked);

  // --- emit confirmed active tracks --------------------------------------
  std::vector<Track> out;
  for (const auto& t : tracked) {
    if (!t->isActivated()) continue;
    Box b = t->tlbr();
    out.push_back(Track{t->trackId(), b.x1, b.y1, b.x2, b.y2, t->score(),
                        t->classId()});
  }
  return out;
}

// ===========================================================================
// ByteTracker public surface
// ===========================================================================

ByteTracker::ByteTracker(ByteTrackConfig cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>(cfg)) {
  if (cfg_.frame_rate <= 0) throw Error(-1, "ByteTracker: frame_rate must be > 0");
  if (cfg_.ema_alpha < 0.0f || cfg_.ema_alpha > 1.0f) {
    throw Error(-1, "ByteTracker: ema_alpha must be in [0, 1]");
  }
}

ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections) {
  return impl_->update(detections, {});
}

std::vector<Track> ByteTracker::update(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& embeddings) {
  return impl_->update(detections, embeddings);
}

void ByteTracker::applyCameraMotion(const float affine[6]) {
  if (affine == nullptr) return;
  for (const auto& t : impl_->tracked) t->cameraUpdate(affine);
  for (const auto& t : impl_->lost) t->cameraUpdate(affine);
}

void ByteTracker::reset() { impl_->reset(); }

}  // namespace bcdl

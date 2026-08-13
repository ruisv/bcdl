#pragma once

#include <cstdint>
#include <vector>

#include "bcdl/tasks/segmentation.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// End-to-end driving: camera + lidar -> planned trajectory
// ===========================================================================
//
// Unlike every other task in this library, this model does not report what it
// sees and leave the decision to you — it emits the ego trajectory to drive.
// The multi-modal machinery (a diffusion decoder denoising a set of clustered
// trajectory anchors) lives inside the compiled graph: anchors go in, ONE
// already-selected trajectory comes out, along with the perception heads the
// planner was trained jointly with (nearby vehicles + a BEV semantic map).
//
// Frame convention throughout: ego frame, metres, x forward, y left, heading in
// radians CCW. Poses are the FUTURE at a fixed interval (the stock model: 8
// poses over 4 s), not a spatial path.

/// One future ego pose.
struct DrivePose {
  float x = 0.0f;
  float y = 0.0f;
  float heading = 0.0f;
};

/// A detected vehicle in the ego frame. `score` is the sigmoid of the head's
/// logit; the stock model's detection head is vehicles-only.
struct DriveAgent {
  float x = 0.0f;
  float y = 0.0f;
  float heading = 0.0f;
  float length = 0.0f;
  float width = 0.0f;
  float score = 0.0f;
};

/// Everything one forward pass produces.
///
/// `bev` reuses SegMask so segColorize() works on it unchanged. Its classes are
/// the upstream TransFuser set: 0 background, 1 road, 2 walkway, 3 centerline,
/// 4 static object, 5 vehicle, 6 pedestrian, at 0.25 m per pixel.
///
/// **Its axes are not screen axes, and getting them wrong still looks right.**
/// Row index increases with FORWARD distance (row 0 is the ego end, so the map
/// covers x in [0, rows*0.25] ahead) and column index increases to the LEFT
/// (+y). Drawing it as an image with north-up therefore needs BOTH axes
/// reversed. Upstream documents neither convention; this one was established by
/// correlating the map's obstacle classes against lidar occupancy across frames
/// and confirmed on turning frames, because a straight road is nearly symmetric
/// under a 180-degree rotation and the wrong orientation is not visibly wrong.
/// The map is returned exactly as the model emits it — reorienting it here would
/// be a silent geometric opinion baked into the decode.
struct DriveResult {
  std::vector<DrivePose> trajectory;
  std::vector<DriveAgent> agents;
  SegMask bev;
};

/// Decode parameters.
struct DriveConfig {
  float agent_score_thresh = 0.5f;  ///< keep agents with sigmoid(logit) above this
  SegConfig bev{};                  ///< layout of the BEV semantic head ([1,C,H,W] logits)
};

/// Decode the four heads into a DriveResult.
///
/// Every head is optional — pass a null pointer (with an empty shape) to skip
/// one. Shapes: trajectory `{1,P,3}` or `{P,3}`; agent states `{1,N,5}` or
/// `{N,5}` as (x, y, heading, length, width); agent labels `{1,N}` or `{N}` as
/// logits; BEV as SegConfig describes.
///
/// Pure function: no Engine, no allocation beyond the result, numpy-testable.
DriveResult decodeDrive(const float* trajectory, const std::vector<int>& trajectory_shape,
                        const float* agent_states, const std::vector<int>& agent_states_shape,
                        const float* agent_labels, const std::vector<int>& agent_labels_shape,
                        const float* bev, const std::vector<int>& bev_shape,
                        const DriveConfig& cfg = {});

/// Ego state the planner is conditioned on.
///
/// `driving_command` is the route hint (which way the ego is asked to go at the
/// next junction), encoded one-hot into the first `num_commands` slots of the
/// status vector; the velocity and acceleration pairs follow. The order is the
/// upstream one and is NOT self-evident from the tensor — a status vector
/// assembled in any other order still runs and still yields a smooth-looking
/// trajectory, it just ignores your speed.
struct DriveStatus {
  int driving_command = 0;
  float velocity_x = 0.0f;
  float velocity_y = 0.0f;
  float accel_x = 0.0f;
  float accel_y = 0.0f;
};

/// Pack a DriveStatus into the model's status vector
/// (`num_commands` one-hot slots + vx, vy, ax, ay).
std::vector<float> driveStatusVector(const DriveStatus& status, int num_commands = 4);

/// Parameters for turning trajectory anchors into the diffusion input.
///
/// The planner is not seeded with noise but with CLUSTERED ANCHORS — k-means
/// centroids of human driving — normalized into the decoder's box and nudged by
/// a truncated forward diffusion step. The normalization constants are part of
/// the trained contract, not a preference: they map metres onto the [-1,1] the
/// decoder was trained in.
struct DriveAnchorConfig {
  float x_offset = 1.2f;    ///< x' = 2*(x + x_offset)/x_span - 1
  float x_span = 56.9f;
  float y_offset = 20.0f;   ///< y' = 2*(y + y_offset)/y_span - 1
  float y_span = 46.0f;
  int truncated_timestep = 8;      ///< where the truncated schedule starts
  int num_train_timesteps = 1000;  ///< DDIM training schedule length
  float beta_start = 1e-4f;        ///< diffusers "scaled_linear" defaults
  float beta_end = 0.02f;
  float noise_scale = 1.0f;  ///< 0 => the sampler with its noise term zeroed
  std::uint64_t seed = 0;
};

/// Normalize `num_modes x num_poses x 2` (x, y) anchors and add the truncated
/// diffusion noise, producing the planner's diffusion input in the same layout.
///
/// Noise is Box-Muller over a seeded mt19937_64 so the result is reproducible
/// across runs and platforms. `noise_scale = 0` zeroes the random term while
/// keeping the schedule's scaling — i.e. it is the sampler with its noise turned
/// off, not a different starting point, which is what makes it comparable
/// against an upstream implementation.
std::vector<float> driveAnchorNoise(const float* anchors, int num_modes, int num_poses,
                                    const DriveAnchorConfig& cfg = {});

/// Which Engine tensor is which. Defaults match the stock export
/// (camera, lidar, status, noise in; trajectory, agent states, agent labels,
/// BEV out). Set an output index to -1 to skip reading that head.
struct DriveIoConfig {
  int camera_input = 0;
  int lidar_input = 1;
  int status_input = 2;
  int noise_input = 3;
  int trajectory_output = 0;
  int agent_states_output = 1;
  int agent_labels_output = 2;
  int bev_output = 3;
};

/// Engine-bound planner: writes the four input tensors (quantizing to whatever
/// dtype the model declared), runs one inference, decodes the four heads.
///
/// The anchor set is fixed for a given model, so the diffusion input is built
/// once in the constructor and reused; call setAnchors() to change it. Staging
/// buffers are held across calls, so steady-state planning allocates nothing but
/// the result.
class DrivePlanner {
 public:
  /// `anchors` is `num_modes * num_poses * 2` floats in metres (the model's
  /// clustered anchor file). Throws if it does not match the noise input.
  DrivePlanner(Engine& engine, const float* anchors, int num_modes, int num_poses,
               DriveConfig cfg = {}, DriveAnchorConfig anchor_cfg = {},
               DriveIoConfig io = {});

  void setAnchors(const float* anchors, int num_modes, int num_poses);

  /// Full pass. `camera` is the stitched [1,3,H,W] RGB canvas in [0,1] and
  /// `lidar` the [1,C,H,W] BEV histogram — both sized by the model.
  DriveResult plan(const std::vector<float>& camera, const std::vector<float>& lidar,
                   const DriveStatus& status, int timeout_ms = 0);

  /// Decode the Engine's current outputs (for callers that drove infer()
  /// themselves).
  DriveResult postprocess() const;

  const DriveConfig& config() const { return cfg_; }
  const std::vector<float>& diffusionInput() const { return noise_; }

 private:
  Engine& engine_;
  DriveConfig cfg_;
  DriveAnchorConfig anchor_cfg_;
  DriveIoConfig io_;
  std::vector<float> noise_;
  std::vector<std::uint8_t> stage_[4];
};

}  // namespace bcdl

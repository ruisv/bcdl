#include "bcdl/tasks/e2e_drive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// Product of a shape, ignoring a leading batch of 1. Returns 0 for an empty
/// shape so callers can treat "no head" uniformly.
std::size_t elementCount(const std::vector<int>& shape) {
  if (shape.empty()) return 0;
  std::size_t n = 1;
  for (int d : shape) n *= static_cast<std::size_t>(d > 0 ? d : 1);
  return n;
}

/// Rows x cols of a {1,R,C} / {R,C} tensor. Throws if it is neither.
void rowsCols(const std::vector<int>& shape, const char* what, int expect_cols, int* rows,
              int* cols) {
  std::vector<int> s = shape;
  if (s.size() == 3 && s[0] == 1) s.erase(s.begin());
  if (s.size() != 2 || s[0] <= 0 || s[1] <= 0) {
    throw Error(-1, std::string("BCDL decodeDrive: ") + what + " wants {1,N," +
                        std::to_string(expect_cols) + "} or {N," +
                        std::to_string(expect_cols) + "}");
  }
  if (expect_cols > 0 && s[1] != expect_cols) {
    throw Error(-1, std::string("BCDL decodeDrive: ") + what + " has " +
                        std::to_string(s[1]) + " columns, expected " +
                        std::to_string(expect_cols));
  }
  *rows = s[0];
  *cols = s[1];
}

}  // namespace

DriveResult decodeDrive(const float* trajectory, const std::vector<int>& trajectory_shape,
                        const float* agent_states, const std::vector<int>& agent_states_shape,
                        const float* agent_labels, const std::vector<int>& agent_labels_shape,
                        const float* bev, const std::vector<int>& bev_shape,
                        const DriveConfig& cfg) {
  DriveResult out;

  if (trajectory != nullptr && !trajectory_shape.empty()) {
    int poses = 0, cols = 0;
    rowsCols(trajectory_shape, "trajectory", 3, &poses, &cols);
    out.trajectory.resize(static_cast<std::size_t>(poses));
    for (int i = 0; i < poses; ++i) {
      const float* p = trajectory + static_cast<std::size_t>(i) * 3;
      out.trajectory[static_cast<std::size_t>(i)] = {p[0], p[1], p[2]};
    }
  }

  if (agent_states != nullptr && !agent_states_shape.empty()) {
    int agents = 0, cols = 0;
    rowsCols(agent_states_shape, "agent_states", 5, &agents, &cols);
    // Labels are optional: without them every slot is reported, since the head
    // always emits a fixed number of boxes and the logit is the only thing that
    // says which are real.
    int label_count = 0;
    if (agent_labels != nullptr && !agent_labels_shape.empty()) {
      label_count = static_cast<int>(elementCount(agent_labels_shape));
      if (label_count != agents) {
        throw Error(-1, "BCDL decodeDrive: agent_labels has " +
                            std::to_string(label_count) + " entries but agent_states has " +
                            std::to_string(agents));
      }
    }
    for (int i = 0; i < agents; ++i) {
      const float score =
          label_count > 0 ? sigmoid(agent_labels[static_cast<std::size_t>(i)]) : 1.0f;
      if (score < cfg.agent_score_thresh) continue;
      const float* a = agent_states + static_cast<std::size_t>(i) * 5;
      out.agents.push_back({a[0], a[1], a[2], a[3], a[4], score});
    }
  }

  if (bev != nullptr && !bev_shape.empty()) {
    out.bev = decodeSeg(bev, bev_shape, cfg.bev);
  }
  return out;
}

std::vector<float> driveStatusVector(const DriveStatus& status, int num_commands) {
  if (num_commands < 0) {
    throw Error(-1, "BCDL driveStatusVector: num_commands must be >= 0");
  }
  std::vector<float> v(static_cast<std::size_t>(num_commands) + 4, 0.0f);
  if (num_commands > 0) {
    if (status.driving_command < 0 || status.driving_command >= num_commands) {
      throw Error(-1, "BCDL driveStatusVector: driving_command " +
                          std::to_string(status.driving_command) + " out of range [0," +
                          std::to_string(num_commands) + ")");
    }
    v[static_cast<std::size_t>(status.driving_command)] = 1.0f;
  }
  const std::size_t k = static_cast<std::size_t>(num_commands);
  v[k + 0] = status.velocity_x;
  v[k + 1] = status.velocity_y;
  v[k + 2] = status.accel_x;
  v[k + 3] = status.accel_y;
  return v;
}

std::vector<float> driveAnchorNoise(const float* anchors, int num_modes, int num_poses,
                                    const DriveAnchorConfig& cfg) {
  if (anchors == nullptr || num_modes <= 0 || num_poses <= 0) {
    throw Error(-1, "BCDL driveAnchorNoise: anchors must be a non-empty MxPx2 array");
  }
  if (cfg.x_span == 0.0f || cfg.y_span == 0.0f) {
    throw Error(-1, "BCDL driveAnchorNoise: normalization span must be non-zero");
  }
  if (cfg.num_train_timesteps <= 0 || cfg.truncated_timestep < 0 ||
      cfg.truncated_timestep >= cfg.num_train_timesteps) {
    throw Error(-1, "BCDL driveAnchorNoise: truncated_timestep must be in "
                    "[0, num_train_timesteps)");
  }

  const std::size_t n = static_cast<std::size_t>(num_modes) * num_poses * 2;
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; i += 2) {
    out[i] = 2.0f * (anchors[i] + cfg.x_offset) / cfg.x_span - 1.0f;
    out[i + 1] = 2.0f * (anchors[i + 1] + cfg.y_offset) / cfg.y_span - 1.0f;
  }
  // "scaled_linear" betas: linear in sqrt-space, squared back. Only the
  // cumulative product up to the truncation point is needed, so this is a
  // 9-term loop for the stock schedule rather than a 1000-entry table.
  const double lo = std::sqrt(static_cast<double>(cfg.beta_start));
  const double hi = std::sqrt(static_cast<double>(cfg.beta_end));
  const double step = (cfg.num_train_timesteps > 1)
                          ? (hi - lo) / static_cast<double>(cfg.num_train_timesteps - 1)
                          : 0.0;
  double alpha_bar = 1.0;
  for (int t = 0; t <= cfg.truncated_timestep; ++t) {
    const double beta_sqrt = lo + step * static_cast<double>(t);
    alpha_bar *= 1.0 - beta_sqrt * beta_sqrt;
  }
  const float keep = static_cast<float>(std::sqrt(alpha_bar));
  const float add = static_cast<float>(std::sqrt(1.0 - alpha_bar)) * cfg.noise_scale;

  // noise_scale == 0 keeps the schedule's scaling and drops only the random
  // draw, so it is exactly the sampler with its noise term zeroed rather than a
  // different starting point — which is what makes it usable as the reference
  // configuration when comparing against an upstream implementation.
  if (cfg.noise_scale == 0.0f) {
    for (float& v : out) v *= keep;
    return out;
  }

  // Box-Muller on a seeded engine rather than std::normal_distribution: the
  // distribution's output is implementation-defined, and this value has to be
  // reproducible for the C++ and Python paths to agree on the same seed.
  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (std::size_t i = 0; i < n; i += 2) {
    double u1 = uni(rng);
    if (u1 <= 0.0) u1 = std::numeric_limits<double>::min();
    const double u2 = uni(rng);
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double z0 = r * std::cos(2.0 * M_PI * u2);
    const double z1 = r * std::sin(2.0 * M_PI * u2);
    out[i] = keep * out[i] + add * static_cast<float>(z0);
    if (i + 1 < n) out[i + 1] = keep * out[i + 1] + add * static_cast<float>(z1);
  }
  return out;
}

// --- Engine-bound planner ---------------------------------------------------

DrivePlanner::DrivePlanner(Engine& engine, const float* anchors, int num_modes,
                           int num_poses, DriveConfig cfg, DriveAnchorConfig anchor_cfg,
                           DriveIoConfig io)
    : engine_(engine), cfg_(cfg), anchor_cfg_(anchor_cfg), io_(io) {
  setAnchors(anchors, num_modes, num_poses);
}

void DrivePlanner::setAnchors(const float* anchors, int num_modes, int num_poses) {
  noise_ = driveAnchorNoise(anchors, num_modes, num_poses, anchor_cfg_);
  const std::size_t want = inputElementCount(engine_, io_.noise_input);
  if (noise_.size() != want) {
    throw Error(-1, "BCDL DrivePlanner: anchors give " + std::to_string(noise_.size()) +
                        " values but the model's diffusion input wants " +
                        std::to_string(want));
  }
}

DriveResult DrivePlanner::plan(const std::vector<float>& camera,
                               const std::vector<float>& lidar, const DriveStatus& status,
                               int timeout_ms) {
  // The status vector's width is read from the model rather than assumed: the
  // command one-hot is whatever is left after the four kinematic slots.
  const std::size_t status_len = inputElementCount(engine_, io_.status_input);
  if (status_len < 4) {
    throw Error(-1, "BCDL DrivePlanner: status input is too small to hold vx, vy, ax, ay");
  }
  const std::vector<float> status_vec =
      driveStatusVector(status, static_cast<int>(status_len) - 4);

  setInputFromFloat(engine_, io_.camera_input, camera.data(), camera.size(), stage_[0]);
  setInputFromFloat(engine_, io_.lidar_input, lidar.data(), lidar.size(), stage_[1]);
  setInputFromFloat(engine_, io_.status_input, status_vec.data(), status_vec.size(),
                    stage_[2]);
  setInputFromFloat(engine_, io_.noise_input, noise_.data(), noise_.size(), stage_[3]);

  engine_.infer(timeout_ms);
  return postprocess();
}

DriveResult DrivePlanner::postprocess() const {
  std::vector<float> traj_buf, agent_buf, label_buf, bev_buf;
  std::vector<int> traj_shape, agent_shape, label_shape, bev_shape;
  const float *traj = nullptr, *agents = nullptr, *labels = nullptr, *bev = nullptr;

  if (io_.trajectory_output >= 0) {
    traj = outputAsFloat(engine_, io_.trajectory_output, traj_buf, traj_shape);
  }
  if (io_.agent_states_output >= 0) {
    agents = outputAsFloat(engine_, io_.agent_states_output, agent_buf, agent_shape);
  }
  if (io_.agent_labels_output >= 0) {
    labels = outputAsFloat(engine_, io_.agent_labels_output, label_buf, label_shape);
  }
  if (io_.bev_output >= 0) {
    bev = outputAsFloat(engine_, io_.bev_output, bev_buf, bev_shape);
  }
  return decodeDrive(traj, traj_shape, agents, agent_shape, labels, label_shape, bev,
                     bev_shape, cfg_);
}

}  // namespace bcdl

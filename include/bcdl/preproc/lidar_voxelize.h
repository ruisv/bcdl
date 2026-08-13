#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bcdl {

// ===========================================================================
// Point cloud -> pillars (the input side of a lidar detector)
// ===========================================================================
//
// A lidar sweep is an unordered list of points; a BPU wants a fixed-shape
// tensor. Pillar encoding is the bridge: quantise the points into a 2-D grid of
// vertical columns, keep at most N points per column and at most M non-empty
// columns, and emit three tensors of constant shape.
//
// Two consequences worth stating, because both are invisible until they bite:
//
//   * the caps are REAL. A dense sweep produces more occupied columns than the
//     compiled model has slots, and the ones past the cap are dropped — so a
//     detector can go blind in exactly the busy scenes it matters in. `Pillars`
//     reports what it dropped rather than silently truncating.
//   * the grid is metric and ego-frame. Points must already be in the ego frame
//     (sensor extrinsics applied); nothing downstream can tell that they are
//     not, because a shifted point cloud is still a perfectly valid one.

/// Pillar-encoding geometry. Defaults are the KITTI configuration the published
/// PointPillars models use: a 69.12 x 79.36 m box at 0.16 m, 32 points per
/// pillar, 11546 pillars.
struct VoxelConfig {
  float min_x = 0.0f;
  float max_x = 69.12f;
  float min_y = -39.68f;
  float max_y = 39.68f;
  float min_z = -3.0f;
  float max_z = 1.0f;
  float pillar_x = 0.16f;   ///< pillar footprint in metres
  float pillar_y = 0.16f;
  int max_points_per_pillar = 32;
  int max_pillars = 11546;
  int num_features = 4;     ///< per point: x, y, z, then intensity (or whatever the model was trained on)

  int gridX() const { return static_cast<int>((max_x - min_x) / pillar_x + 0.5f); }
  int gridY() const { return static_cast<int>((max_y - min_y) / pillar_y + 0.5f); }
};

/// The three tensors a pillar model takes, in the layout the compiled models
/// declare.
///
/// `coords` is (max_pillars, 4) as (batch, z, y, x) — z is always 0 for pillars
/// and exists because the same slot holds a voxel index in 3-D encoders.
struct Pillars {
  std::vector<float> voxels;      ///< max_pillars * max_points * num_features
  std::vector<float> num_points;  ///< max_pillars (float: that is how the models declare it)
  std::vector<float> coords;      ///< max_pillars * 4
  int num_pillars = 0;            ///< how many slots are actually filled
  std::size_t points_used = 0;
  std::size_t points_out_of_range = 0;
  std::size_t points_dropped_full = 0;   ///< pillar already had max_points
  std::size_t pillars_dropped = 0;       ///< occupied columns past max_pillars
};

/// Encode a point cloud into pillars.
///
/// `points` is `num_points` records of `stride_floats` floats; the first three
/// are x, y, z in the EGO frame and the next `cfg.num_features - 3` are copied
/// through (intensity, and elongation/time if the model was trained with them).
Pillars voxelizePillars(const float* points, std::size_t num_points,
                        std::size_t stride_floats, const VoxelConfig& cfg = {});

/// Expand the raw points of each pillar into the per-point features a PillarVFE
/// MLP consumes, laid out as (1, C, max_pillars, max_points) — the "image" a
/// 1x1-convolution VFE reads, channels on the channel axis.
///
/// C is `cfg.num_features + 6`: the raw point, then its offset from the pillar's
/// centroid, then its offset from the pillar's geometric centre. For the KITTI
/// configuration that is the familiar 10.
///
/// The published `vfe.hbm` does this inside its own graph, which is why it takes
/// (pillars, points, 4) directly — and why it is slow: that shape is a huge
/// pillar axis against a 4-wide channel axis, the worst case for a convolution
/// engine, and it spends 290 ms on a 95 MFLOP MLP. Expressing the same MLP as a
/// 1x1 convolution over this layout, with the max over points as a pooling
/// window, runs in 1.6 ms. Doing the expansion here is what buys that.
///
/// Slots past a pillar's point count are zeroed rather than left at whatever the
/// centroid arithmetic produced: they go through the MLP too, and the max over
/// points would otherwise pick up whatever the bias maps them to.
std::vector<float> pillarPointFeatures(const float* voxels, const float* num_points,
                                       const float* coords, int num_pillars,
                                       const VoxelConfig& cfg = {});

/// Scatter per-pillar features back onto the dense BEV canvas the 2-D backbone
/// consumes: `features` is (num_pillars, C) and the result is (1, C, gridY, gridX).
///
/// Trivial as arithmetic, and the reason it is here rather than in the graph:
/// the published ONNX expresses it as a 55 MB ScatterND, which is both large and
/// the exact construct that compiles cleanly into a broken model.
std::vector<float> scatterPillars(const float* features, const float* coords,
                                  int num_pillars, int channels, const VoxelConfig& cfg);

}  // namespace bcdl

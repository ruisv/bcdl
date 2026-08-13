#include "bcdl/preproc/lidar_voxelize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>

#include "bcdl/core/status.h"

namespace bcdl {

Pillars voxelizePillars(const float* points, std::size_t num_points,
                        std::size_t stride_floats, const VoxelConfig& cfg) {
  if (points == nullptr && num_points > 0) {
    throw Error(-1, "BCDL voxelizePillars: null point cloud");
  }
  if (stride_floats < 3) {
    throw Error(-1, "BCDL voxelizePillars: stride_floats must be >= 3 (x, y, z)");
  }
  if (cfg.num_features < 3 || static_cast<std::size_t>(cfg.num_features) > stride_floats) {
    throw Error(-1, "BCDL voxelizePillars: num_features must be in [3, stride_floats]");
  }
  if (cfg.pillar_x <= 0.0f || cfg.pillar_y <= 0.0f || cfg.max_pillars <= 0 ||
      cfg.max_points_per_pillar <= 0) {
    throw Error(-1, "BCDL voxelizePillars: non-positive grid or capacity");
  }
  const int gx = cfg.gridX(), gy = cfg.gridY();
  if (gx <= 0 || gy <= 0) {
    throw Error(-1, "BCDL voxelizePillars: range and pillar size give an empty grid");
  }

  Pillars out;
  const std::size_t f = static_cast<std::size_t>(cfg.num_features);
  const std::size_t pts = static_cast<std::size_t>(cfg.max_points_per_pillar);
  out.voxels.assign(static_cast<std::size_t>(cfg.max_pillars) * pts * f, 0.0f);
  out.num_points.assign(static_cast<std::size_t>(cfg.max_pillars), 0.0f);
  out.coords.assign(static_cast<std::size_t>(cfg.max_pillars) * 4, 0.0f);

  // Cell -> slot. A hash map rather than a gx*gy dense array: the grid is
  // 496x432 here but a 3-D or long-range config makes the dense table larger
  // than the sweep, and it has to be cleared every frame either way.
  std::unordered_map<int, int> slot_of;
  slot_of.reserve(static_cast<std::size_t>(cfg.max_pillars) * 2);

  for (std::size_t i = 0; i < num_points; ++i) {
    const float* p = points + i * stride_floats;
    const float x = p[0], y = p[1], z = p[2];
    if (!(x >= cfg.min_x && x < cfg.max_x) || !(y >= cfg.min_y && y < cfg.max_y) ||
        !(z >= cfg.min_z && z < cfg.max_z)) {
      ++out.points_out_of_range;   // the !(>=) form also rejects NaN
      continue;
    }
    const int cx = static_cast<int>((x - cfg.min_x) / cfg.pillar_x);
    const int cy = static_cast<int>((y - cfg.min_y) / cfg.pillar_y);
    if (cx < 0 || cx >= gx || cy < 0 || cy >= gy) {
      ++out.points_out_of_range;
      continue;
    }
    const int cell = cy * gx + cx;

    auto it = slot_of.find(cell);
    int slot;
    if (it == slot_of.end()) {
      if (out.num_pillars >= cfg.max_pillars) {
        ++out.pillars_dropped;     // busy scene, no slots left
        continue;
      }
      slot = out.num_pillars++;
      slot_of.emplace(cell, slot);
      float* c = out.coords.data() + static_cast<std::size_t>(slot) * 4;
      c[0] = 0.0f;                          // batch
      c[1] = 0.0f;                          // z index (pillars are 2-D)
      c[2] = static_cast<float>(cy);
      c[3] = static_cast<float>(cx);
    } else {
      slot = it->second;
    }

    const int n = static_cast<int>(out.num_points[static_cast<std::size_t>(slot)]);
    if (n >= cfg.max_points_per_pillar) {
      ++out.points_dropped_full;
      continue;
    }
    float* dst = out.voxels.data() +
                 (static_cast<std::size_t>(slot) * pts + static_cast<std::size_t>(n)) * f;
    std::memcpy(dst, p, f * sizeof(float));
    out.num_points[static_cast<std::size_t>(slot)] = static_cast<float>(n + 1);
    ++out.points_used;
  }
  return out;
}

std::vector<float> pillarPointFeatures(const float* voxels, const float* num_points,
                                       const float* coords, int num_pillars,
                                       const VoxelConfig& cfg) {
  if (voxels == nullptr || num_points == nullptr || coords == nullptr) {
    throw Error(-1, "BCDL pillarPointFeatures: null pillar tensors");
  }
  if (cfg.num_features < 3 || cfg.max_pillars <= 0 || cfg.max_points_per_pillar <= 0) {
    throw Error(-1, "BCDL pillarPointFeatures: bad config");
  }
  if (num_pillars < 0 || num_pillars > cfg.max_pillars) {
    throw Error(-1, "BCDL pillarPointFeatures: num_pillars out of range");
  }
  const std::size_t f = static_cast<std::size_t>(cfg.num_features);
  const std::size_t pts = static_cast<std::size_t>(cfg.max_points_per_pillar);
  const std::size_t pil = static_cast<std::size_t>(cfg.max_pillars);
  const std::size_t chan = f + 6;

  // Zeroed once: everything past num_pillars, and every unfilled point slot,
  // has to stay zero.
  std::vector<float> out(chan * pil * pts, 0.0f);

  const float half_x = cfg.pillar_x * 0.5f + cfg.min_x;
  const float half_y = cfg.pillar_y * 0.5f + cfg.min_y;
  const float pillar_z = cfg.max_z - cfg.min_z;          // pillars span the full z range
  const float half_z = pillar_z * 0.5f + cfg.min_z;

#pragma omp parallel for schedule(static)
  for (int p = 0; p < num_pillars; ++p) {
    const std::size_t up = static_cast<std::size_t>(p);
    const int n = static_cast<int>(num_points[up]);
    if (n <= 0) continue;
    const int take = n < cfg.max_points_per_pillar ? n : cfg.max_points_per_pillar;
    const float* v = voxels + up * pts * f;
    const float* c = coords + up * 4;

    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (int k = 0; k < take; ++k) {
      mx += v[static_cast<std::size_t>(k) * f + 0];
      my += v[static_cast<std::size_t>(k) * f + 1];
      mz += v[static_cast<std::size_t>(k) * f + 2];
    }
    const float inv = 1.0f / static_cast<float>(take);
    mx *= inv; my *= inv; mz *= inv;

    // The pillar's geometric centre, from its grid index. coords is (batch, z, y, x).
    const float cx = c[3] * cfg.pillar_x + half_x;
    const float cy = c[2] * cfg.pillar_y + half_y;
    const float cz = c[1] * pillar_z + half_z;

    for (int k = 0; k < take; ++k) {
      const float* pt = v + static_cast<std::size_t>(k) * f;
      const std::size_t at = up * pts + static_cast<std::size_t>(k);
      for (std::size_t ch = 0; ch < f; ++ch) {
        out[(ch * pil * pts) + at] = pt[ch];
      }
      out[((f + 0) * pil * pts) + at] = pt[0] - mx;
      out[((f + 1) * pil * pts) + at] = pt[1] - my;
      out[((f + 2) * pil * pts) + at] = pt[2] - mz;
      out[((f + 3) * pil * pts) + at] = pt[0] - cx;
      out[((f + 4) * pil * pts) + at] = pt[1] - cy;
      out[((f + 5) * pil * pts) + at] = pt[2] - cz;
    }
  }
  return out;
}

std::vector<float> scatterPillars(const float* features, const float* coords,
                                  int num_pillars, int channels,
                                  const VoxelConfig& cfg) {
  if (features == nullptr || coords == nullptr) {
    throw Error(-1, "BCDL scatterPillars: null features or coords");
  }
  if (channels <= 0 || num_pillars < 0) {
    throw Error(-1, "BCDL scatterPillars: bad channel or pillar count");
  }
  const int gx = cfg.gridX(), gy = cfg.gridY();
  const std::size_t plane = static_cast<std::size_t>(gx) * static_cast<std::size_t>(gy);
  std::vector<float> canvas(plane * static_cast<std::size_t>(channels), 0.0f);

  for (int i = 0; i < num_pillars; ++i) {
    const int cy = static_cast<int>(coords[static_cast<std::size_t>(i) * 4 + 2]);
    const int cx = static_cast<int>(coords[static_cast<std::size_t>(i) * 4 + 3]);
    if (cx < 0 || cx >= gx || cy < 0 || cy >= gy) continue;
    const std::size_t cell = static_cast<std::size_t>(cy) * gx + cx;
    for (int c = 0; c < channels; ++c) {
      canvas[static_cast<std::size_t>(c) * plane + cell] =
          features[static_cast<std::size_t>(i) * channels + c];
    }
  }
  return canvas;
}

}  // namespace bcdl

#include "bcdl/preproc/drive_inputs.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// Bin index for `v` in a uniform histogram over [lo, hi) with `bins` cells, or
/// -1 when the value falls outside. The right edge is inclusive in the last bin,
/// matching numpy's histogramdd — a point sitting exactly on the far boundary is
/// counted, not silently dropped.
inline int binIndex(float v, float lo, float hi, int bins) {
  if (!(v >= lo) || v > hi) return -1;  // the !(>=) form also rejects NaN
  if (v == hi) return bins - 1;
  const int i = static_cast<int>((v - lo) / (hi - lo) * static_cast<float>(bins));
  return std::clamp(i, 0, bins - 1);
}

/// One cropped camera inside the concatenated strip.
struct StripSegment {
  const uint8_t* base = nullptr;
  int stride = 0;
  int x0 = 0;     ///< first source column of the crop
  int y0 = 0;     ///< first source row of the crop
  int width = 0;  ///< cropped width
};

}  // namespace

LidarBev lidarBevHistogram(const float* points, std::size_t num_points,
                           std::size_t stride_floats, const LidarBevConfig& cfg) {
  if (stride_floats < 3) {
    throw Error(-1, "BCDL lidarBevHistogram: stride_floats must be >= 3 (x, y, z)");
  }
  if (!(cfg.max_x > cfg.min_x) || !(cfg.max_y > cfg.min_y) || cfg.pixels_per_meter <= 0.0f) {
    throw Error(-1, "BCDL lidarBevHistogram: empty box or non-positive pixels_per_meter");
  }
  if (cfg.hist_max_per_pixel <= 0.0f) {
    throw Error(-1, "BCDL lidarBevHistogram: hist_max_per_pixel must be > 0");
  }

  const int nx = static_cast<int>(std::lround((cfg.max_x - cfg.min_x) * cfg.pixels_per_meter));
  const int ny = static_cast<int>(std::lround((cfg.max_y - cfg.min_y) * cfg.pixels_per_meter));
  if (nx <= 0 || ny <= 0) {
    throw Error(-1, "BCDL lidarBevHistogram: box and resolution give an empty grid");
  }

  LidarBev bev;
  bev.channels = cfg.use_ground_plane ? 2 : 1;
  bev.height = nx;  // first axis is x (forward), as upstream
  bev.width = ny;
  const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  bev.data.assign(plane * static_cast<std::size_t>(bev.channels), 0.0f);

  // Accumulate counts in place, then clamp+scale once at the end — one pass over
  // the sweep (which is ~10^5 points) instead of a count buffer plus a copy.
  for (std::size_t i = 0; i < num_points; ++i) {
    const float* p = points + i * stride_floats;
    const float z = p[2];
    if (!(z < cfg.max_height)) continue;  // also drops NaN
    const int ix = binIndex(p[0], cfg.min_x, cfg.max_x, nx);
    if (ix < 0) continue;
    const int iy = binIndex(p[1], cfg.min_y, cfg.max_y, ny);
    if (iy < 0) continue;

    const bool above = z > cfg.split_height;
    if (!above && !cfg.use_ground_plane) continue;
    const std::size_t channel = cfg.use_ground_plane ? (above ? 1u : 0u) : 0u;
    bev.data[channel * plane + static_cast<std::size_t>(ix) * ny + iy] += 1.0f;
  }

  for (float& v : bev.data) {
    v = std::min(v, cfg.hist_max_per_pixel) / cfg.hist_max_per_pixel;
  }
  return bev;
}

std::vector<float> stitchDriveCameras(const uint8_t* left, int left_w, int left_h,
                                      int left_stride, const uint8_t* front, int front_w,
                                      int front_h, int front_stride, const uint8_t* right,
                                      int right_w, int right_h, int right_stride,
                                      const DriveCameraConfig& cfg) {
  if (!left || !front || !right) {
    throw Error(-1, "BCDL stitchDriveCameras: all three camera buffers are required");
  }
  if (cfg.out_width <= 0 || cfg.out_height <= 0) {
    throw Error(-1, "BCDL stitchDriveCameras: output canvas is empty");
  }

  const int strip_h = front_h - cfg.crop_top - cfg.crop_bottom;
  if (strip_h <= 0) {
    throw Error(-1, "BCDL stitchDriveCameras: vertical crop leaves nothing");
  }
  if (left_h - cfg.crop_top - cfg.crop_bottom != strip_h ||
      right_h - cfg.crop_top - cfg.crop_bottom != strip_h) {
    throw Error(-1, "BCDL stitchDriveCameras: the three cameras differ in cropped height");
  }

  StripSegment seg[3];
  seg[0] = {left, left_stride > 0 ? left_stride : left_w * 3, cfg.crop_side, cfg.crop_top,
            left_w - 2 * cfg.crop_side};
  seg[1] = {front, front_stride > 0 ? front_stride : front_w * 3, 0, cfg.crop_top, front_w};
  seg[2] = {right, right_stride > 0 ? right_stride : right_w * 3, cfg.crop_side, cfg.crop_top,
            right_w - 2 * cfg.crop_side};
  for (const StripSegment& s : seg) {
    if (s.width <= 0) {
      throw Error(-1, "BCDL stitchDriveCameras: horizontal crop leaves nothing");
    }
  }
  const int strip_w = seg[0].width + seg[1].width + seg[2].width;

  // Sample the concatenated strip without materializing it: 3 cropped 1080p
  // frames would be a ~12 MB copy every frame, and the only thing the copy buys
  // is contiguity. Bilinear taps that straddle a seam simply resolve to
  // different segments, which is exactly what resizing the concatenation does.
  const auto pixelAt = [&](int sx, int sy) -> const uint8_t* {
    int idx = 0;
    int x = sx;
    if (x >= seg[0].width) {
      x -= seg[0].width;
      idx = 1;
      if (x >= seg[1].width) {
        x -= seg[1].width;
        idx = 2;
      }
    }
    const StripSegment& s = seg[idx];
    return s.base + static_cast<std::size_t>(s.y0 + sy) * s.stride +
           static_cast<std::size_t>(s.x0 + x) * 3;
  };

  const int ow = cfg.out_width, oh = cfg.out_height;
  std::vector<float> out(static_cast<std::size_t>(3) * ow * oh, 0.0f);
  const std::size_t plane = static_cast<std::size_t>(ow) * oh;
  const float sx_ratio = static_cast<float>(strip_w) / static_cast<float>(ow);
  const float sy_ratio = static_cast<float>(strip_h) / static_cast<float>(oh);

  for (int oy = 0; oy < oh; ++oy) {
    float fy = (oy + 0.5f) * sy_ratio - 0.5f;
    fy = std::clamp(fy, 0.0f, static_cast<float>(strip_h - 1));
    const int py0 = static_cast<int>(fy);
    const int py1 = std::min(py0 + 1, strip_h - 1);
    const float wy = fy - static_cast<float>(py0);

    for (int ox = 0; ox < ow; ++ox) {
      float fx = (ox + 0.5f) * sx_ratio - 0.5f;
      fx = std::clamp(fx, 0.0f, static_cast<float>(strip_w - 1));
      const int px0 = static_cast<int>(fx);
      const int px1 = std::min(px0 + 1, strip_w - 1);
      const float wx = fx - static_cast<float>(px0);

      const uint8_t* p00 = pixelAt(px0, py0);
      const uint8_t* p01 = pixelAt(px1, py0);
      const uint8_t* p10 = pixelAt(px0, py1);
      const uint8_t* p11 = pixelAt(px1, py1);
      const float w00 = (1.0f - wy) * (1.0f - wx);
      const float w01 = (1.0f - wy) * wx;
      const float w10 = wy * (1.0f - wx);
      const float w11 = wy * wx;

      const std::size_t o = static_cast<std::size_t>(oy) * ow + ox;
      for (int c = 0; c < 3; ++c) {
        const float v = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
        // BGR source -> RGB planes, scaled to [0,1].
        out[static_cast<std::size_t>(2 - c) * plane + o] = v / 255.0f;
      }
    }
  }
  return out;
}

}  // namespace bcdl

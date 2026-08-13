#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bcdl {

// ===========================================================================
// Sensor tensors for camera+lidar driving models (TransFuser-style)
// ===========================================================================
//
// End-to-end driving models do not eat an image — they eat a fixed rig: a wide
// forward canvas stitched from several cameras, plus the lidar sweep rasterized
// into a bird's-eye occupancy grid in the ego frame. Both are geometric
// contracts, not preferences: the network learned one specific crop, one
// specific metre-per-pixel, one specific axis order, and a plausible-looking
// tensor built any other way produces a plausible-looking trajectory that is
// simply wrong. These build them to spec.

/// Bird's-eye lidar histogram parameters.
///
/// Defaults are the TransFuser/NAVSIM contract: a 64 m x 64 m box around the ego
/// at 4 px/m => a 256x256 grid, points counted per cell, clamped at 5 and scaled
/// to [0,1]. `split_height` divides ground from everything above it; with
/// `use_ground_plane` false only the ABOVE band is emitted (one channel), which
/// is what the deployed single-channel models expect.
struct LidarBevConfig {
  float min_x = -32.0f;
  float max_x = 32.0f;
  float min_y = -32.0f;
  float max_y = 32.0f;
  float pixels_per_meter = 4.0f;
  float max_height = 100.0f;       ///< drop points above this (roof / ego returns)
  float split_height = 0.2f;       ///< z <= split => ground band, z > split => above
  float hist_max_per_pixel = 5.0f; ///< count clamp, then divide by it
  bool use_ground_plane = false;   ///< true => 2 channels [below, above]
};

/// A [C,H,W] float grid in [0,1]; C is 1 (above only) or 2 ([below, above]).
struct LidarBev {
  int channels = 0;
  int height = 0;
  int width = 0;
  std::vector<float> data;
};

/// Rasterize a lidar sweep (ego frame, metres) into the BEV histogram.
///
/// `points` is `num_points` records of `stride_floats` floats each, the first
/// three being x, y, z (a 4th intensity channel is ignored). Points outside the
/// box, or above `max_height`, are dropped.
///
/// Axis order follows the upstream `np.histogramdd(points[:, :2])`: the FIRST
/// grid axis is x and the second is y — i.e. row index counts forward distance,
/// column index counts left/right. Transposing this is the easiest way to get a
/// model that runs and steers into the scenery.
LidarBev lidarBevHistogram(const float* points, std::size_t num_points,
                           std::size_t stride_floats, const LidarBevConfig& cfg = {});

/// Crop margins + output canvas for the 3-camera forward stitch.
///
/// Upstream crops `[top:-bottom]` from all three and additionally `[side:-side]`
/// from the two angled cameras, concatenates left|front|right horizontally, then
/// resizes the whole strip to `out_width` x `out_height`. The resize happens
/// AFTER concatenation, so pixels interpolate across the seams — matching that
/// is why this builds the strip instead of resizing each camera separately.
struct DriveCameraConfig {
  int crop_top = 28;
  int crop_bottom = 28;
  int crop_side = 416;  ///< applied to the left/right cameras only
  int out_width = 1024;
  int out_height = 256;
};

/// Build the model's [1,3,out_height,out_width] RGB float canvas from three BGR
/// images, scaled to [0,1].
///
/// Each camera is passed as a packed-or-strided BGR8 buffer (`stride` bytes per
/// row; 0 => width*3). Output is CHW RGB — the channel swap is here because
/// every board-side capture path in this library hands out BGR, and doing it at
/// the last moment keeps one conversion rather than three.
///
/// Throws Error(-1) if a crop leaves an empty image or the cameras disagree in
/// height after cropping.
std::vector<float> stitchDriveCameras(const uint8_t* left, int left_w, int left_h,
                                      int left_stride, const uint8_t* front, int front_w,
                                      int front_h, int front_stride, const uint8_t* right,
                                      int right_w, int right_h, int right_stride,
                                      const DriveCameraConfig& cfg = {});

}  // namespace bcdl

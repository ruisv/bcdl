#pragma once

#include <cstdint>

#include "bcdl/preproc/vp_image.h"

namespace bcdl {

/// Hardware dense remap on the VPS **GDC** engine (`/dev/gdc`): an arbitrary
/// FIXED warp with cv2.remap semantics — `out(x, y) = in(map_x[y,x], map_y[y,x])`
/// — compiled at construction into a GDC CUSTOM-grid warp LUT via
/// `hbn_gen_gdc_cfg` (runtime; no offline bin file, unlike GdcLetterbox).
///
/// Built for stereo rectification: the undistort/rectify maps from
/// `cv2.initUndistortRectifyMap` are smooth, so they are sampled every
/// `grid_step` px into the CUSTOM transformation grid (the GDC interpolates
/// between grid nodes). `out_w % grid_step == out_h % grid_step == 0` required
/// (16 divides both 2448 and 2048).
///
/// Same vnode/buffer plumbing as GdcLetterbox: persistent FEEDBACK vnode,
/// persistent device input buffer, NV12 in -> NV12 out. Cost model: two NV12
/// copies (into the device buffer / out of the vnode frame) around the
/// hardware op. `BCDL_GDC_TIMING=1` prints the per-run breakdown.
class GdcRemap {
 public:
  /// map_x/map_y: dense row-major (out_h, out_w) float32 maps, cv2 convention.
  /// Sampled input coords may fall outside [0,in_w)x[0,in_h); they are clamped.
  GdcRemap(const float* map_x, const float* map_y, int in_w, int in_h,
           int out_w, int out_h, int grid_step = 16);
  ~GdcRemap();

  GdcRemap(const GdcRemap&) = delete;
  GdcRemap& operator=(const GdcRemap&) = delete;

  /// Warp `src` (NV12, in_w x in_h) into `dst` (NV12, out_w x out_h) on the GDC
  /// hardware. `dst` is (re)allocated to the output size if needed.
  void run(const VpImage& src, VpImage& dst);

  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  int outputWidth() const { return out_w_; }
  int outputHeight() const { return out_h_; }

 private:
  struct Impl;
  Impl* p_;
  int in_w_, in_h_, out_w_, out_h_;
};

}  // namespace bcdl

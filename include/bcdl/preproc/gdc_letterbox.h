#pragma once

#include <cstdint>
#include <string>

#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/vp_image.h"

namespace bcdl {

/// Hardware letterbox on the VPS **GDC** engine (`/dev/gdc`), bypassing the
/// vDSP that `hb_vp_warp_affine` (bcdl::letterbox) uses — the vDSP is root-locked
/// / firmware-offline on the S100P, so this is the working hardware path.
///
/// A persistent GDC vnode (FEEDBACK mode) is opened once, driven by a warp-LUT
/// "bin" generated at construction for the fixed in->out geometry. run() maps an
/// in_w x in_h NV12 into an out_w x out_h NV12 letterbox (aspect-preserving,
/// centered, flat `pad` borders) entirely on GDC hardware.
///
/// Cost model: sendframe/getframe are the GDC hardware op; the src NV12 is copied
/// into a persistent device input buffer first (GDC consumes hb_mem graphic
/// buffers, not bcdl SysMem), and the result is copied into `dst`.
class GdcLetterbox {
 public:
  /// Open the GDC vnode with a pre-generated warp-LUT `bin_path` for the fixed
  /// (in_w,in_h)->(out_w,out_h) letterbox. Generate the bin offline for the SAME
  /// geometry via the SDK's generate_bin tool (aspect-preserving Affine placing
  /// the full ROI at the centered window computeLetterbox() yields). Throws
  /// bcdl::Error / std::runtime_error on any hobot/GDC failure. (v2 TODO: generate
  /// the bin at runtime so the path arg goes away.)
  GdcLetterbox(const std::string& bin_path, int in_w, int in_h, int out_w, int out_h,
               uint8_t pad = 114);
  ~GdcLetterbox();

  GdcLetterbox(const GdcLetterbox&) = delete;
  GdcLetterbox& operator=(const GdcLetterbox&) = delete;

  /// Transform `src` (NV12, in_w x in_h) into `dst` (NV12, out_w x out_h) on the
  /// GDC hardware. `dst` is (re)allocated to the output size if needed. Returns
  /// the letterbox geometry (scale/pad) for un-letterboxing detections.
  LetterboxInfo run(const VpImage& src, VpImage& dst);

  /// The fixed letterbox geometry for this instance.
  const LetterboxInfo& info() const { return lb_; }

  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  int outputWidth() const { return out_w_; }
  int outputHeight() const { return out_h_; }

 private:
  struct Impl;
  Impl* p_;
  LetterboxInfo lb_;
  int in_w_, in_h_, out_w_, out_h_;
};

}  // namespace bcdl

#pragma once

#include <array>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Monocular 3D detection — SMOKE head (KITTI 3-class, RDK S100 nash-m export)
// ===========================================================================
//
// SMOKE (Single-stage Monocular 3D Object Detection via Keypoint Estimation)
// is anchor-free / CenterNet-style: a class heatmap locates each object's
// projected 3D center, and an 8-channel regression map at that pixel carries
// everything needed to lift it to a full 3D box given the camera intrinsics.
//
// MODEL CONTRACT (mono3d_m3_384x1280_nashm.hbm, verified on board)
//   input  "image": [1,3,384,1280] F32 RGB CHW. Preproc = warp original image
//          to 384x1280 (see geometry note below) + /255 + ImageNet normalize
//          (mean .485/.456/.406, std .229/.224/.225). Normalization is NOT
//          fused into the .hbm.
//   output "cls": [1, nc, H, W]  raw class logits  (nc=3, H=96, W=320, stride 4)
//   output "reg": [1, 8,  H, W]  raw regression     (channel-FIRST, NCHW)
//   Both outputs are RAW logits — the predictor activations were stripped from
//   the BPU graph and MUST be re-applied here (see decode below).
//
//   NOTE: cls/reg are CHANNEL-FIRST [C,H,W] (unlike the NHWC det/pose/obb
//   heads). Index a channel c at (y,x) as buf[c*H*W + y*W + x].
//
// RE-APPLIED ACTIVATIONS (stripped from the .hbm)
//   cls       = clamp(sigmoid(cls), 1e-4, 1-1e-4)
//   reg[3:6]  = sigmoid(reg[3:6]) - 0.5        // dim channels
//   reg[6:8]  = l2_normalize(reg[6:8])         // ori channels (sin,cos)
//   reg[0], reg[1:3] are used raw.
//
// REG 8-CHANNEL SEMANTICS
//   ch 0   : depth offset  -> z = off*depth_ref[1] + depth_ref[0]
//   ch 1,2 : projected-center offset (dx,dy) on the feature map
//   ch 3-5 : dimension offset -> dims = exp(sigmoid(raw)-0.5) * dim_ref[cls]
//   ch 6,7 : orientation (sin,cos) -> alpha = atan(sin/cos) + quadrant fix
//
// DECODE (per kept peak), matching the SMOKE PostProcessor exactly:
//   1. heat = cls * (maxpool(cls, k=nms_kernel) == cls)      // CenterNet NMS
//   2. take top-`max_dets` peaks across all classes; keep score > conf_thresh.
//      For peak at feature pixel (xf, yf), class c, score s:
//   3. z   = reg0 * depth_ref[1] + depth_ref[0]
//      proj = (xf + reg1, yf + reg2)                         // feature pixels
//      (px, py) = featXform.inv(proj)                        // -> original px
//      x = (px - K.cx)/K.fx * z,  y = (py - K.cy)/K.fy * z   // K^-1 (no skew)
//      d = exp(sigmoid(reg[3:6])-0.5) * dim_ref[c]           // (d0,d1,d2)
//      h,w,l = d1,d2,d0   ;   y += h/2     // feature center -> box bottom
//      alpha = atan(reg6/reg7) (-pi/2 if cos>=0 else +pi/2)
//      ray   = atan(x / z) ;  yaw = wrap_pi(alpha + ray)
//   4. (pred_2d) project the 8 corners of the 3D box with K, take the
//      axis-aligned min/max, clamp to the original image extent -> box2d.
//
// GEOMETRY NOTE — featXform is the original<->feature(WxH) affine, NOT a
// standard letterbox. SMOKE scales TO WIDTH (s = W/origW) and centers the
// height (the aspect ratio is cropped/padded, not fit-both), so it is a
// LetterboxInfo with scale=W/origW, padX=0, padY=H/2 - s*origH/2 (may be
// negative = vertical crop). Build it with computeMono3dFeatureXform(). The
// 384x1280 INPUT affine fed to the model is the SAME transform at 4x scale
// (s_in = 1280/origW); preproc and featXform MUST stay consistent or accuracy
// drops. The camera K stays the ORIGINAL-image intrinsics — the resize is
// carried entirely by featXform, never folded into K.

/// Pinhole camera intrinsics for the ORIGINAL (un-resized) image, KITTI-style
/// with zero skew. K = [[fx,0,cx],[0,fy,cy],[0,0,1]]. Back-projection uses the
/// closed-form inverse (x=(u-cx)/fx*z, y=(v-cy)/fy*z), so no matrix inverse.
struct CameraIntrinsics {
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

/// One monocular-3D detection. 3D quantities are in camera coordinates (meters,
/// KITTI convention: +x right, +y down, +z forward; (x,y,z) is the box BOTTOM
/// center). `box2d` is the projected 3D box in ORIGINAL-image pixels.
struct Mono3dBox {
  int class_id = 0;        ///< 0=Car, 1=Cyclist, 2=Pedestrian (per dim_ref rows)
  float score = 0.0f;      ///< heatmap peak score in [0,1]
  float x = 0.0f;          ///< bottom-center X (camera coords, meters)
  float y = 0.0f;          ///< bottom-center Y
  float z = 0.0f;          ///< depth (forward)
  float h = 0.0f;          ///< height (meters)
  float w = 0.0f;          ///< width
  float l = 0.0f;          ///< length
  float yaw = 0.0f;        ///< rotation_y about camera Y axis (radians)
  float alpha = 0.0f;      ///< observation angle (radians)
  std::array<float, 4> box2d = {0, 0, 0, 0};  ///< x1,y1,x2,y2 original px (pred_2d)
};

/// Post-processing parameters for the SMOKE head. Defaults mirror the reference
/// SMOKE cfg (DETECTIONS_THRESHOLD=0.25, DETECTIONS_PER_IMG=50, PRED_2D=True,
/// DEPTH_REFERENCE=(28.01,16.32), DIMENSION_REFERENCE = KITTI Car/Cyclist/Ped).
struct Mono3dConfig {
  int num_classes = 3;             ///< overridden from the cls tensor's C dim
  float conf_thresh = 0.25f;       ///< keep peaks with score > this
  int max_dets = 50;               ///< top-K peaks taken before thresholding
  int nms_kernel = 3;              ///< CenterNet max-pool heatmap-NMS window
  bool pred_2d = true;             ///< project the 3D box to a 2D box2d
  std::array<float, 2> depth_ref = {28.01f, 16.32f};  ///< z = off*ref[1] + ref[0]
  /// Per-class dimension priors (one row per class, internal order d0,d1,d2 ==
  /// l,h,w as consumed by the decode). Must have >= num_classes rows.
  std::vector<std::array<float, 3>> dim_ref = {
      {3.88f, 1.63f, 1.53f},  // Car
      {1.78f, 1.70f, 0.58f},  // Cyclist
      {0.88f, 1.73f, 0.67f},  // Pedestrian
  };
};

/// Build the original<->feature(featW x featH) affine used by the SMOKE decode.
/// This is the "scale-to-width, center-height" map (NOT computeLetterbox's
/// fit-both): scale = featW/origW, padX = 0, padY = featH/2 - scale*origH/2.
/// Its inverse (LetterboxInfo::invX/invY) lifts a feature-map pixel back to the
/// original image, which is exactly what decodeMono3d needs.
LetterboxInfo computeMono3dFeatureXform(int origW, int origH, int featW, int featH);

/// Decode raw SMOKE cls/reg buffers into 3D detections.
///
/// `cls`        : CHANNEL-FIRST [num_classes, H, W] raw class logits.
/// `reg`        : CHANNEL-FIRST [8, H, W] raw regression.
/// `H`, `W`     : feature-map grid (96, 320 for the 384x1280 export).
/// `cfg`        : thresholds + depth/dim priors (dim_ref must cover num_classes).
/// `featXform`  : original<->feature affine (computeMono3dFeatureXform()).
/// `K`          : ORIGINAL-image intrinsics for back-projection.
/// Re-applies the stripped activations, runs CenterNet heatmap-NMS + top-K, and
/// lifts each kept peak to a Mono3dBox (sorted by score, descending).
std::vector<Mono3dBox> decodeMono3d(const float* cls, const float* reg, int H, int W,
                                    const Mono3dConfig& cfg,
                                    const LetterboxInfo& featXform,
                                    const CameraIntrinsics& K);

/// Engine-bound SMOKE monocular-3D detector. Reads two outputs starting at
/// `output_base`: cls = output_base (shape [1,nc,H,W]) and reg = output_base+1
/// (shape [1,8,H,W]). num_classes and (H,W) are taken from the cls tensor; the
/// feature affine is built from (origW,origH) at postprocess time.
class Mono3dDetector {
 public:
  Mono3dDetector(Engine& engine, Mono3dConfig cfg = {}, int output_base = 0);

  /// Decode the current outputs for an original image of size (origW,origH)
  /// captured with intrinsics `K`.
  std::vector<Mono3dBox> postprocess(int origW, int origH,
                                     const CameraIntrinsics& K) const;

  const Mono3dConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  Mono3dConfig cfg_;
  int out_base_;
};

}  // namespace bcdl

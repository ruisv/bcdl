#pragma once

#include <cstdint>
#include <vector>

#include "bcdl/tasks/depth.h"  // DepthMap, decodeDepth, depthColorize

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// How a source frame is fit to the model's (input_h, input_w) before infer.
///
/// MUST match the offline calibration mode (gen_calib.py --mode). A model
/// calibrated on resized inputs fed center-crops (or vice versa) shifts the
/// disparity scale outside the quantization range and the output collapses.
enum class StereoFit {
  /// Anamorphic full-frame resize to (input_w,input_h). Preserves field of
  /// view but compresses disparity (far/sub-pixel detail is lost).
  kResize,
  /// Center crop to (input_w,input_h). Preserves the native disparity scale
  /// (best metric depth) at the cost of a narrower field of view.
  kCrop,
};

/// Tunables for a two-image stereo disparity / depth pipeline (e.g. LAS2).
///
/// The model takes two F32 NCHW images (left -> input[left_index], right ->
/// input[right_index]) and emits a single-channel disparity map. Pixel
/// normalization (e.g. /255, mean/std) is expected to be FUSED into the `.hbm`
/// — the pipeline feeds raw float pixel values, channel-swapped to RGB when
/// `to_rgb` is set (LAS2 trains on RGB; OpenCV frames arrive BGR).
struct StereoConfig {
  int input_w = 0;            ///< model input width  (0 => derive from Engine input[left_index])
  int input_h = 0;            ///< model input height (0 => derive from Engine input[left_index])
  StereoFit fit = StereoFit::kResize;  ///< resize vs center-crop (match calibration!)
  bool to_rgb = true;         ///< swap incoming BGR -> RGB before feeding (LAS2 = RGB)
  int left_index = 0;         ///< Engine input index fed the left image
  int right_index = 1;        ///< Engine input index fed the right image
  int output_index = 0;       ///< Engine output index holding the disparity map

  // ---- optional metric depth: z = fx * baseline / disparity ----
  /// Focal length in pixels AT THE MODEL INPUT RESOLUTION (after fit). When both
  /// fx and baseline are > 0, process() fills StereoResult.depth.
  float fx = 0.0f;
  float baseline = 0.0f;      ///< stereo baseline in metres

  // ---- optional validity mask (geometry-based; see stereoValidMask) ----
  bool valid_mask = false;    ///< compute StereoResult.valid
  float disp_min = 0.0f;      ///< disparities below this are invalid (drops far/sub-pixel noise)
  float max_disp = 192.0f;    ///< disparities above this are invalid (too near)
  int left_margin = 0;        ///< also drop the leftmost N columns
  /// Left-right consistency check: run a second (flipped) inference to estimate
  /// the right-view disparity and reject pixels that disagree (catches
  /// occlusions). Doubles inference cost. Needs valid_mask = true.
  bool lr_check = false;
  float lr_thresh = 1.5f;     ///< max |dL - dR| (px) to keep a pixel under lr_check
};

/// Result of one stereo frame. `disparity` is always populated (raw, in input-
/// resolution pixels). `depth`/`valid` are populated only when requested via the
/// config; both are row-major HxW matching `disparity.width`/`height`.
struct StereoResult {
  DepthMap disparity;             ///< raw disparity map (cfg.normalize=false semantics)
  std::vector<float> depth;       ///< metric depth (m); empty unless fx & baseline set
  std::vector<uint8_t> valid;     ///< 1=trustworthy, 0=masked; empty unless valid_mask
};

// ---------------------------------------------------------------------------
// Pure helpers (Engine-free; reused by the pipeline, examples, and bindings).
// ---------------------------------------------------------------------------

/// Fit an interleaved HxWx3 uint8 BGR frame to (out_h,out_w) and pack it as a
/// planar F32 CHW tensor into `dst` (which must hold 3*out_h*out_w floats).
///
/// `fit` selects anamorphic resize (bilinear, OpenCV when available else a
/// hand bilinear) or center crop. When `to_rgb`, the channel order is swapped
/// so plane 0=R, 1=G, 2=B; otherwise it stays B,G,R. Pixel values are copied as
/// raw floats (no /255, no mean/std) — those belong in the fused `.hbm`. Pixels
/// outside the source (crop larger than the frame) are written as 0.
void packStereoInputCHW(const uint8_t* bgr, int width, int height, int out_h,
                        int out_w, StereoFit fit, bool to_rgb, float* dst);

/// Convert a disparity map to metric depth: z = fx * baseline / max(disp, eps),
/// row-major HxW. Pixels with disparity <= eps map to 0 (invalid / infinite).
std::vector<float> disparityToDepth(const DepthMap& disp, float fx, float baseline);

/// Geometry-based validity mask for a disparity map (1=keep, 0=mask), mirroring
/// the LAS2 reference: disp in [disp_min, max_disp], the right-view column
/// x-disp must be in-frame (x-disp >= 0), and the leftmost `left_margin`
/// columns are dropped. When `disp_right` is non-null (a right-view disparity
/// of the same size), a left-right consistency check rejects |dL - dR| >
/// lr_thresh (occlusions). Returns a row-major HxW uint8 buffer.
std::vector<uint8_t> stereoValidMask(const DepthMap& disp, float disp_min,
                                     float max_disp, int left_margin,
                                     const DepthMap* disp_right, float lr_thresh);

/// Streaming two-image stereo pipeline. Construct once with an Engine, then call
/// process() per frame pair. Input scratch tensors are allocated once and reused
/// (no per-frame heap allocation for the model inputs). Synchronous: infer()
/// blocks via WaitTaskDone.
///
/// Per-frame flow:
///   packStereoInputCHW(left)  -> left_buf_  -> Engine.setInput(left_index)
///   packStereoInputCHW(right) -> right_buf_ -> Engine.setInput(right_index)
///   Engine.infer()
///   decodeDepth(output, normalize=false)            -> disparity
///   [disparityToDepth] [stereoValidMask (+lr_check)] -> depth / valid
class StereoPipeline {
 public:
  /// Build the pipeline. cfg.input_w/input_h <= 0 are resolved from the Engine's
  /// input[left_index] shape (NCHW). Requires both inputs to be float (F32/F16).
  StereoPipeline(Engine& engine, StereoConfig cfg);

  StereoPipeline(const StereoPipeline&) = delete;
  StereoPipeline& operator=(const StereoPipeline&) = delete;

  /// Run one stereo pair. `left`/`right` point to interleaved HxWx3 uint8 BGR
  /// frames of identical (width,height). Returns disparity (+ optional depth /
  /// valid) at the model input resolution.
  StereoResult process(const uint8_t* left, const uint8_t* right, int width,
                       int height);

  const StereoConfig& config() const noexcept { return cfg_; }
  int inputWidth() const noexcept { return cfg_.input_w; }
  int inputHeight() const noexcept { return cfg_.input_h; }

 private:
  /// pack `left`->input[left_index] and `right`->input[right_index] (channel-
  /// swap + fit) into the scratch buffers, run one inference, and decode the raw
  /// disparity map. The lr_check pass calls this with horizontally-flipped,
  /// order-swapped frames (see process()).
  DepthMap inferDisparity(const uint8_t* left, const uint8_t* right, int width,
                          int height);

  Engine& engine_;
  StereoConfig cfg_;
  std::vector<float> left_buf_;   ///< reused [3*input_h*input_w] planar input
  std::vector<float> right_buf_;
};

}  // namespace bcdl

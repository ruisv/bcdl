#pragma once

#include <cstdint>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// RGB-D depth refinement — LingBot-Depth (MDM v2, DINOv2 ViT-L/14 RGBD encoder)
// ===========================================================================
//
// Unlike the monocular (depth.h) and stereo heads, this model does not estimate
// depth from scratch: it takes an EXISTING depth map — a stereo or ToF sensor
// reading, noisy and full of holes — together with the aligned RGB frame, and
// returns a cleaned, hole-filled metric depth map plus a per-pixel trust mask.
//
// The compiled graph is deliberately thin. Everything that is not a static
// tensor op was pushed out to the host so the BPU sees a clean static graph:
//
//   host  ->  resize RGB to the encoder grid (area average) + mean/std normalize
//             resize depth (nearest) + zero invalid readings + natural log
//   BPU   ->  patch embed (image tokens ++ depth tokens) -> 24 transformer
//             blocks -> conv neck -> depth head + mask head
//   host  ->  threshold the mask logits, optionally unproject to a point cloud
//
// Two consequences worth knowing:
//
//   * The graph's depth input is LOG depth with zero meaning "no reading". That
//     is upstream's own convention, and it means a genuine 1 m reading and an
//     invalid pixel land on the same value; the model leans on the RGB tokens to
//     tell them apart.
//   * Upstream drops depth tokens whose patch has no valid reading at all, which
//     makes the sequence length depend on the data and cannot be compiled into a
//     static graph. The deployed graph keeps every token instead. Measured on the
//     reference scenes that costs 0.06% absolute relative depth error and 0.9999
//     mask IoU — but those scenes are 87-100% valid, and very sparse input depth
//     (a lidar sweep, say) was not part of that measurement.

/// Preprocessing + decode parameters. The defaults match the shipped 1200-token
/// 480x640 model: a 30x40 token grid, hence a 420x560 encoder input.
struct DepthRefineConfig {
  int encoder_height = 420;      ///< token_rows * 14
  int encoder_width = 560;       ///< token_cols * 14
  float min_valid_depth = 0.01f; ///< metres; at or below this the reading is "missing"
  float mask_threshold = 0.5f;   ///< keep a pixel when sigmoid(mask_logit) exceeds this
  bool apply_mask = true;        ///< zero out depth the mask rejects
};

/// A refined depth map at the model's output resolution.
struct RefinedDepth {
  int width = 0;
  int height = 0;
  std::vector<float> depth;   ///< row-major HxW metric depth; 0 where rejected
  std::vector<uint8_t> mask;  ///< row-major HxW, 1 = trusted
  float vmin = 0.0f;          ///< min/max over trusted pixels only, for colorizing
  float vmax = 0.0f;
};

/// Pinhole intrinsics in pixels, expressed for a specific image size.
struct Intrinsics {
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

/// Rescale intrinsics from one image size to another (pure ratio scaling — valid
/// for a plain resize, not for a crop).
Intrinsics scaleIntrinsics(const Intrinsics& k, int from_w, int from_h, int to_w, int to_h);

/// Build the model's `image` input: area-average downscale to the encoder grid,
/// BGR->RGB, /255, then ImageNet mean/std. Output is planar [3, eh, ew] float.
///
/// The downscale is area-averaged rather than plain bilinear on purpose. The
/// reference implementation resizes with an antialiased bilinear filter, which
/// neither this nor OpenCV reproduces exactly; measured against it across the
/// reference scenes, area averaging costs 6.1e-4 mean absolute relative depth
/// error and plain bilinear costs 9.1e-4. Both are dominated by quantization,
/// but area is the cheaper half of that choice and needs no extra dependency.
void preprocessRefineImage(const uint8_t* bgr, int width, int height, int stride_bytes,
                           const DepthRefineConfig& cfg, std::vector<float>* out);

/// Build the model's `depth_log` input: nearest-neighbour downscale to the
/// encoder grid, then log(depth) with 0 for readings at or below
/// `min_valid_depth` (and for non-finite ones). Output is [1, eh, ew] float.
///
/// `depth_m` is row-major metric depth with `depth_stride_elems` floats per row
/// (pass `width` for a packed buffer).
void preprocessRefineDepth(const float* depth_m, int width, int height,
                           int depth_stride_elems, const DepthRefineConfig& cfg,
                           std::vector<float>* out);

/// Convert a raw uint16 depth image (the usual sensor encoding) to metres.
std::vector<float> depthU16ToMetres(const uint16_t* raw, int width, int height,
                                    int stride_elems, float scale = 1000.0f);

/// Decode the model's two outputs into a RefinedDepth.
///
/// `depth` is HxW metric depth straight from the depth head (this checkpoint
/// uses a linear output remap, so no exponential is needed). `mask_logit` is the
/// pre-sigmoid mask head output, or null to trust every pixel.
RefinedDepth decodeRefinedDepth(const float* depth, const float* mask_logit, int height,
                                int width, const DepthRefineConfig& cfg);

/// Unproject to a camera-space point cloud: 3 floats (x, y, z) per pixel,
/// row-major, metres. `k` must be expressed for the RefinedDepth's own size —
/// use scaleIntrinsics() when the sensor image was a different resolution.
/// Rejected pixels come back as (0, 0, 0).
std::vector<float> depthToPointCloud(const RefinedDepth& d, const Intrinsics& k);

/// Engine-bound wrapper: preprocess -> infer -> decode.
///
/// Input tensor order follows the exported graph (`image`, then `depth_log`);
/// pass explicit indices if a re-exported model orders them differently.
class DepthRefiner {
 public:
  explicit DepthRefiner(Engine& engine, DepthRefineConfig cfg = {}, int image_input = 0,
                        int depth_input = 1, int depth_output = 0, int mask_output = 1);

  /// Full frame: BGR image + metric depth, both at the sensor's own resolution
  /// (they are resized internally and need not match the model's).
  RefinedDepth run(const uint8_t* bgr, int width, int height, int stride_bytes,
                   const float* depth_m, int depth_stride_elems);

  /// Decode whatever is currently in the engine's output buffers. For callers
  /// that drove setInput()/infer() themselves.
  RefinedDepth postprocess() const;

  const DepthRefineConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DepthRefineConfig cfg_;
  int image_in_;
  int depth_in_;
  int depth_out_;
  int mask_out_;
  std::vector<float> image_buf_;
  std::vector<float> depth_buf_;
};

}  // namespace bcdl

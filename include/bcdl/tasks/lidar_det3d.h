#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Lidar 3-D object detection (anchor-based, PointPillars family)
// ===========================================================================
//
// The first task here whose output is a box in SPACE rather than in an image:
// centre, extent and yaw in the ego frame, in metres. The head predicts a
// residual against a fixed grid of anchors, so decoding needs the same anchor
// set the model was trained with — get it wrong and the boxes are plausible,
// well-formed and in the wrong places.
//
// The decode runs on the CPU deliberately. The published head bakes it into the
// graph with ScatterND, which is the construct that compiles without a single
// error into a model whose outputs are never written (the same trap this library
// already hit once with a panoptic driving head). Cutting the graph after the
// three convolutions and decoding here is both correct and faster.

/// One detected object, ego frame, metres and radians.
struct Box3D {
  float x = 0.0f;        ///< centre
  float y = 0.0f;
  float z = 0.0f;
  float dx = 0.0f;       ///< extent along the box's own axes (length, width, height)
  float dy = 0.0f;
  float dz = 0.0f;
  float heading = 0.0f;  ///< yaw about +z, radians, CCW from +x
  int label = 0;
  float score = 0.0f;
};

/// Anchor grid definition. Defaults are the KITTI configuration published with
/// the PointPillars models: three classes at two yaws each over a 248x216 map.
///
/// `sizes`/`bottoms` are per class; every class gets every entry of `rotations`.
/// The anchor order must match the head's channel order — class-major, then
/// rotation — which is what `generateAnchors3d` produces and what the shipped
/// model expects.
struct Anchor3dConfig {
  float min_x = 0.0f;
  float max_x = 69.12f;
  float min_y = -39.68f;
  float max_y = 39.68f;
  int grid_x = 216;                       ///< feature-map width
  int grid_y = 248;                        ///< feature-map height
  /// (dx, dy, dz) per class, then the anchor's CENTRE height per class.
  std::vector<std::vector<float>> sizes{{3.9f, 1.6f, 1.56f},
                                        {0.8f, 0.6f, 1.73f},
                                        {1.76f, 0.6f, 1.73f}};
  std::vector<float> centers{-1.0f, 0.265f, 0.265f};
  std::vector<float> rotations{0.0f, 1.57f};
};

/// Decode parameters.
struct Det3dConfig {
  float score_thresh = 0.3f;
  float nms_iou = 0.01f;       ///< BEV IoU; PointPillars ships a very low threshold
  int max_boxes = 100;
  bool apply_sigmoid = true;   ///< the head emits logits
  Anchor3dConfig anchors{};
};

/// Build the anchor grid: `grid_y * grid_x * num_classes * num_rotations` rows of
/// (x, y, z, dx, dy, dz, yaw), class-major.
std::vector<float> generateAnchors3d(const Anchor3dConfig& cfg = {});

/// Decode the three head tensors into boxes.
///
/// `cls`  : `{1, A*C, H, W}` logits, `box` : `{1, A*7, H, W}` residuals,
/// `dir`  : `{1, A*2, H, W}` direction logits (optional — pass nullptr to skip
/// the yaw disambiguation, which costs you a heading flipped by pi on roughly
/// half the boxes).
///
/// `anchors` must hold the rows `generateAnchors3d` produces. Pure function.
std::vector<Box3D> decodeDet3d(const float* cls, const float* box, const float* dir,
                               int num_classes, int height, int width,
                               const float* anchors, const Det3dConfig& cfg = {});

/// Rotated-rectangle IoU in the BEV plane (the metric 3-D NMS actually uses).
float bevIou(const Box3D& a, const Box3D& b);

/// Greedy NMS on BEV IoU, per class.
std::vector<Box3D> nms3d(std::vector<Box3D> boxes, float iou_thresh, int max_boxes);

/// Engine-bound head: three output tensors in, boxes out.
///
/// This owns only the LAST stage. A pillar detector is a pipeline of three
/// compiled models with CPU stages between them, so the pieces stay separate:
/// `voxelizePillars` -> VFE engine -> `scatterPillars` -> backbone engine ->
/// this. Fusing them would hide which stage costs what, and on this model that
/// answer is the whole story (see docs/MODELS.md).
class LidarDetector3d {
 public:
  LidarDetector3d(Engine& engine, Det3dConfig cfg = {}, int cls_index = 0,
                  int box_index = 1, int dir_index = 2);

  /// Decode the Engine's current outputs (run infer() first).
  std::vector<Box3D> postprocess() const;

  const Det3dConfig& config() const { return cfg_; }
  const std::vector<float>& anchors() const { return anchors_; }

 private:
  Engine& engine_;
  Det3dConfig cfg_;
  int cls_idx_;
  int box_idx_;
  int dir_idx_;
  std::vector<float> anchors_;
};

}  // namespace bcdl

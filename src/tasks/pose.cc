#include "bcdl/tasks/pose.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "bcdl/tasks/detection.h"  // bcdl::Detection, bcdl::nms (box NMS reuse)

namespace bcdl {

std::vector<PoseDetection> decodePose(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& kpt,
    const std::vector<std::pair<int, int>>& grid_hw, const PoseConfig& cfg,
    const LetterboxInfo& lb) {
  const size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales || kpt.size() != scales ||
      cfg.strides.size() != scales) {
    throw Error(-1, "BCDL decodePose: cls/box/kpt/grid/strides length mismatch");
  }
  const int nkpt = cfg.num_keypoints;
  if (nkpt < 0) {
    throw Error(-1, "BCDL decodePose: negative num_keypoints");
  }
  const int64_t kstep = static_cast<int64_t>(nkpt) * 3;  // floats per cell in kpt

  std::vector<PoseDetection> dets;
  for (size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    const float* kp = kpt[s];
    if (cp == nullptr || bp == nullptr || kp == nullptr || H <= 0 || W <= 0) {
      continue;
    }

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const int64_t cell = static_cast<int64_t>(gy) * W + gx;

        // Single "person" class: per-cell cls stride is 1, class_id = 0.
        const float score = sigmoid(cp[cell]);
        if (score < cfg.conf_thresh) continue;

        // LTRB distances -> model-input pixel corners about the cell center.
        const float* d = bp + cell * 4;
        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;
        const float mx1 = (cx - d[0]) * stride;
        const float my1 = (cy - d[1]) * stride;
        const float mx2 = (cx + d[2]) * stride;
        const float my2 = (cy + d[3]) * stride;

        PoseDetection det;
        det.x1 = lb.clampX(lb.invX(mx1));
        det.y1 = lb.clampY(lb.invY(my1));
        det.x2 = lb.clampX(lb.invX(mx2));
        det.y2 = lb.clampY(lb.invY(my2));
        det.score = score;
        det.class_id = 0;

        // Keypoints: ADD the centered grid (note: ADD, unlike the box subtract).
        //   kx = (raw_x + gx + 0.5) * stride,  ky = (raw_y + gy + 0.5) * stride
        //   kscore = sigmoid(raw_score)
        // kpt layout per cell: [k0_x,k0_y,k0_s, k1_x,k1_y,k1_s, ...].
        const float* kpc = kp + cell * kstep;
        det.keypoints.reserve(nkpt);
        for (int k = 0; k < nkpt; ++k) {
          const float raw_x = kpc[k * 3 + 0];
          const float raw_y = kpc[k * 3 + 1];
          const float raw_s = kpc[k * 3 + 2];
          const float mkx = (raw_x + cx) * stride;  // cx = gx + 0.5
          const float mky = (raw_y + cy) * stride;  // cy = gy + 0.5
          Keypoint kpnt;
          kpnt.x = lb.clampX(lb.invX(mkx));
          kpnt.y = lb.clampY(lb.invY(mky));
          kpnt.score = sigmoid(raw_s);
          det.keypoints.push_back(kpnt);
        }

        dets.push_back(std::move(det));
      }
    }
  }

  // Box NMS over the pooled candidates. All boxes share class_id 0, so the
  // per-class nms() degenerates to a single global pass. Project each pose box
  // into a Detection, suppress, then gather the surviving PoseDetections.
  std::vector<Detection> boxes;
  boxes.reserve(dets.size());
  for (const PoseDetection& p : dets) {
    Detection b;
    b.x1 = p.x1;
    b.y1 = p.y1;
    b.x2 = p.x2;
    b.y2 = p.y2;
    b.score = p.score;
    b.class_id = p.class_id;  // 0 for all
    boxes.push_back(b);
  }

  const std::vector<int> keep = nms(boxes, cfg.iou_thresh, cfg.max_dets);
  std::vector<PoseDetection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(std::move(dets[idx]));
  return out;
}

PoseEstimator::PoseEstimator(Engine& engine, PoseConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {}

std::vector<PoseDetection> PoseEstimator::postprocess(const LetterboxInfo& lb) const {
  const int scales = static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || out_base_ + 3 * scales > engine_.numOutputs()) {
    throw Error(-1, "BCDL PoseEstimator: output index range out of bounds");
  }

  // View each (cls, box, kpt) triple as row-major floats — zero-copy for packed
  // F32 (the common RDK case), dequant-into-scratch otherwise. Scratch buffers
  // live until decodePose() below has consumed the pointers.
  std::vector<std::vector<float>> cls_buf(scales), box_buf(scales), kpt_buf(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales), kpt_ptr(scales);
  std::vector<std::pair<int, int>> grid_hw(scales);

  // num_keypoints comes from the kpt tensor's OWN last dim ([1,H,W,K*3] => K),
  // and the grid (H,W) from the cls tensor's OWN shape ([1,H,W,1]) — never from
  // cfg_, so a mis-configured value can't make decodePose index past a buffer.
  int nkpt = 0;
  for (int s = 0; s < scales; ++s) {
    std::vector<int> cls_shape, box_shape, kpt_shape;
    cls_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s, cls_buf[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 1, box_buf[s], box_shape);
    kpt_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 2, kpt_buf[s], kpt_shape);

    int H = 0, W = 0;
    if (cls_shape.size() == 4) {  // [1, H, W, 1]
      H = cls_shape[1];
      W = cls_shape[2];
    } else if (cls_shape.size() == 3) {  // [H, W, 1]
      H = cls_shape[0];
      W = cls_shape[1];
    } else {
      throw Error(-1, "BCDL PoseEstimator: unexpected cls tensor rank");
    }

    int kdim = 0;  // last dim of the kpt tensor
    if (kpt_shape.size() == 4) {  // [1, H, W, K*3]
      kdim = kpt_shape[3];
    } else if (kpt_shape.size() == 3) {  // [H, W, K*3]
      kdim = kpt_shape[2];
    } else {
      throw Error(-1, "BCDL PoseEstimator: unexpected kpt tensor rank");
    }
    if (kdim <= 0 || kdim % 3 != 0) {
      throw Error(-1, "BCDL PoseEstimator: kpt last dim not a positive multiple of 3");
    }
    const int this_nkpt = kdim / 3;
    if (s == 0) {
      nkpt = this_nkpt;
    } else if (this_nkpt != nkpt) {
      throw Error(-1, "BCDL PoseEstimator: inconsistent keypoint count across scales");
    }

    grid_hw[s] = {H, W};
  }

  // Decode with num_keypoints read from the model, not the configured one.
  PoseConfig eff = cfg_;
  if (nkpt > 0) eff.num_keypoints = nkpt;
  return decodePose(cls_ptr, box_ptr, kpt_ptr, grid_hw, eff, lb);
}

}  // namespace bcdl

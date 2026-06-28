#include "bcdl/tasks/instance_seg.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "bcdl/tasks/detection.h"  // Detection, nms() (model-input-coord NMS)

#ifdef BCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // cv::resize
#endif

namespace bcdl {

namespace {

/// Bilinear resize of a single-channel float image, OpenCV-compatible.
///
/// Uses the OpenCV pixel-center mapping srcX = (dstX + 0.5)*sw/dw - 0.5 (and the
/// same on y), clamped to the source extent, so it matches cv2.resize(...,
/// INTER_LINEAR) used by the reference process_mask. `src` is sh*sw row-major,
/// `dst` is dh*dw row-major (caller-allocated). Sizes must all be > 0.
void bilinearResize(const float* src, int sh, int sw, float* dst, int dh, int dw) {
  if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) return;
#ifdef BCDL_HAVE_OPENCV
  // cv::resize writes into the caller's dst buffer (matching size/type => create
  // does not reallocate). INTER_LINEAR matches the hand-written pixel-center map.
  const cv::Mat s(sh, sw, CV_32F, const_cast<float*>(src));
  cv::Mat d(dh, dw, CV_32F, dst);
  cv::resize(s, d, cv::Size(dw, dh), 0.0, 0.0, cv::INTER_LINEAR);
#else
  const float ratio_y = static_cast<float>(sh) / static_cast<float>(dh);
  const float ratio_x = static_cast<float>(sw) / static_cast<float>(dw);
  for (int dy = 0; dy < dh; ++dy) {
    // Pixel-center source coordinate, clamped into [0, sh-1].
    float fy = (static_cast<float>(dy) + 0.5f) * ratio_y - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > static_cast<float>(sh - 1)) fy = static_cast<float>(sh - 1);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - static_cast<float>(y0);
    const float* row0 = src + static_cast<int64_t>(y0) * sw;
    const float* row1 = src + static_cast<int64_t>(y1) * sw;
    float* drow = dst + static_cast<int64_t>(dy) * dw;
    for (int dx = 0; dx < dw; ++dx) {
      float fx = (static_cast<float>(dx) + 0.5f) * ratio_x - 0.5f;
      if (fx < 0.0f) fx = 0.0f;
      if (fx > static_cast<float>(sw - 1)) fx = static_cast<float>(sw - 1);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - static_cast<float>(x0);
      const float top = row0[x0] * (1.0f - wx) + row0[x1] * wx;
      const float bot = row1[x0] * (1.0f - wx) + row1[x1] * wx;
      drow[dx] = top * (1.0f - wy) + bot * wy;
    }
  }
#endif
}

/// Assemble one instance's full-frame binary mask (process_mask, ref. lines
/// 853-903). Returns a row-major orig_h*orig_w buffer of 0/1.
///
///  1. mask160[y][x] = sigmoid( Σ_c coef[c] * proto[y][x][c] )  (proto is NHWC
///     [mH,mW,np], so element = proto[(y*mW + x)*np + c]).
///  2. bilinear resize mask160 -> model-input canvas (lb.dstH × lb.dstW).
///  3. crop_mask: zero everything outside the model-input box (mx1,my1,mx2,my2).
///  4. remove letterbox padding: scale=lb.scale, nw=int(orig_w*scale),
///     nh=int(orig_h*scale), pad_w=(dstW-nw)/2, pad_h=(dstH-nh)/2; take the
///     [pad_h:pad_h+nh, pad_w:pad_w+nw] sub-image (integer floor pad, exactly as
///     the reference; lb.scale == min(dstW/orig_w, dstH/orig_h)).
///  5. bilinear resize that sub-image -> (orig_h × orig_w).
///  6. threshold > 0.5 -> binary.
std::vector<uint8_t> computeInstanceMask(const float* coef, int np,
                                         const float* proto, int mH, int mW, int pC,
                                         float mx1, float my1, float mx2, float my2,
                                         const LetterboxInfo& lb,
                                         int orig_w, int orig_h) {
  const int dstW = lb.dstW;
  const int dstH = lb.dstH;
  const int chans = std::min(np, pC);  // defensive; np == pC is expected

  // 1. prototype linear combination + sigmoid -> mH×mW.
  std::vector<float> m160(static_cast<size_t>(mH) * mW);
  for (int y = 0; y < mH; ++y) {
    for (int x = 0; x < mW; ++x) {
      const float* pv = proto + (static_cast<int64_t>(y) * mW + x) * pC;
      float acc = 0.0f;
      for (int c = 0; c < chans; ++c) acc += coef[c] * pv[c];
      m160[static_cast<size_t>(y) * mW + x] = sigmoid(acc);
    }
  }

  // 2. resize mH×mW -> dstH×dstW (model-input canvas).
  std::vector<float> minput(static_cast<size_t>(dstH) * dstW);
  bilinearResize(m160.data(), mH, mW, minput.data(), dstH, dstW);

  // 3. crop_mask: keep only pixels with x in [mx1,mx2) and y in [my1,my2).
  for (int y = 0; y < dstH; ++y) {
    const float fy = static_cast<float>(y);
    const bool y_in = (fy >= my1) && (fy < my2);
    float* row = minput.data() + static_cast<int64_t>(y) * dstW;
    for (int x = 0; x < dstW; ++x) {
      const float fx = static_cast<float>(x);
      if (!(y_in && fx >= mx1 && fx < mx2)) row[x] = 0.0f;
    }
  }

  // 4. de-pad: extract the scaled-image region (integer floor, as reference).
  const float scale = lb.scale;
  int nw = static_cast<int>(static_cast<float>(orig_w) * scale);
  int nh = static_cast<int>(static_cast<float>(orig_h) * scale);
  if (nw < 1) nw = 1;
  if (nh < 1) nh = 1;
  if (nw > dstW) nw = dstW;
  if (nh > dstH) nh = dstH;
  const int pad_w = (dstW - nw) / 2;
  const int pad_h = (dstH - nh) / 2;

  std::vector<float> cropped(static_cast<size_t>(nh) * nw);
  for (int ry = 0; ry < nh; ++ry) {
    const float* srow = minput.data() + static_cast<int64_t>(pad_h + ry) * dstW + pad_w;
    float* crow = cropped.data() + static_cast<int64_t>(ry) * nw;
    for (int rx = 0; rx < nw; ++rx) crow[rx] = srow[rx];
  }

  // 5. resize nh×nw -> orig_h×orig_w.
  std::vector<float> mfull(static_cast<size_t>(orig_h) * orig_w);
  bilinearResize(cropped.data(), nh, nw, mfull.data(), orig_h, orig_w);

  // 6. threshold.
  std::vector<uint8_t> out(static_cast<size_t>(orig_h) * orig_w);
  for (size_t i = 0; i < out.size(); ++i) out[i] = mfull[i] > 0.5f ? 1 : 0;
  return out;
}

}  // namespace

std::vector<InstanceMask> decodeInstanceSeg(
    const std::vector<const float*>& cls, const std::vector<const float*>& box,
    const std::vector<const float*>& mc,
    const std::vector<std::pair<int, int>>& grid_hw, int num_classes, int num_coef,
    const float* proto, int proto_h, int proto_w, int proto_c,
    const InstanceSegConfig& cfg, const LetterboxInfo& lb, int orig_w, int orig_h) {
  const size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales || mc.size() != scales ||
      cfg.strides.size() != scales) {
    throw Error(-1, "BCDL decodeInstanceSeg: cls/box/mc/grid/strides length mismatch");
  }
  if (num_classes <= 0 || num_coef <= 0) {
    throw Error(-1, "BCDL decodeInstanceSeg: non-positive class/coef count");
  }
  if (orig_w <= 0 || orig_h <= 0) {
    throw Error(-1, "BCDL decodeInstanceSeg: invalid original image size");
  }
  if (cfg.compute_masks &&
      (proto == nullptr || proto_h <= 0 || proto_w <= 0 || proto_c <= 0)) {
    throw Error(-1, "BCDL decodeInstanceSeg: invalid prototype for mask assembly");
  }
  if (cfg.compute_masks && num_coef != proto_c) {
    throw Error(-1, "BCDL decodeInstanceSeg: mask-coef count != prototype channels");
  }

  // Per-candidate state: model-input-coord box (for NMS + mask crop), score,
  // class, and the num_coef mask coefficients (kept parallel to `cands`).
  std::vector<Detection> cands;       // boxes in MODEL-INPUT pixels
  std::vector<std::vector<float>> coefs;

  for (size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float* cp = cls[s];
    const float* bp = box[s];
    const float* mp = mc[s];
    if (cp == nullptr || bp == nullptr || mp == nullptr || H <= 0 || W <= 0) continue;

    const float stride = static_cast<float>(cfg.strides[s]);
    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const int64_t cell = static_cast<int64_t>(gy) * W + gx;
        const float* logits = cp + cell * num_classes;

        // argmax over classes (sigmoid monotonic => argmax on raw logits).
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < num_classes; ++k) {
          if (logits[k] > best_raw) {
            best_raw = logits[k];
            best_k = k;
          }
        }
        const float score = sigmoid(best_raw);
        if (score < cfg.conf_thresh) continue;

        // LTRB -> model-input pixel box about the cell center (no un-letterbox).
        const float* d = bp + cell * 4;
        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;
        Detection det;
        det.x1 = (cx - d[0]) * stride;
        det.y1 = (cy - d[1]) * stride;
        det.x2 = (cx + d[2]) * stride;
        det.y2 = (cy + d[3]) * stride;
        det.score = score;
        det.class_id = best_k;
        cands.push_back(det);

        const float* mcp = mp + cell * num_coef;
        coefs.emplace_back(mcp, mcp + num_coef);
      }
    }
  }

  // Per-class NMS on the model-input boxes (IoU is invariant to the uniform
  // letterbox scale + translation, so this matches NMS on original-image boxes).
  const std::vector<int> keep = nms(cands, cfg.iou_thresh, cfg.max_dets);

  std::vector<InstanceMask> out;
  out.reserve(keep.size());
  for (int idx : keep) {
    const Detection& d = cands[idx];
    InstanceMask im;
    // Final box: un-letterbox model-input corners -> original-image pixels.
    im.x1 = lb.clampX(lb.invX(d.x1));
    im.y1 = lb.clampY(lb.invY(d.y1));
    im.x2 = lb.clampX(lb.invX(d.x2));
    im.y2 = lb.clampY(lb.invY(d.y2));
    im.score = d.score;
    im.class_id = d.class_id;
    im.mask_w = orig_w;
    im.mask_h = orig_h;
    if (cfg.compute_masks) {
      im.mask = computeInstanceMask(coefs[idx].data(), num_coef, proto, proto_h,
                                    proto_w, proto_c, d.x1, d.y1, d.x2, d.y2, lb,
                                    orig_w, orig_h);
    }
    out.push_back(std::move(im));
  }
  return out;
}

InstanceSegmenter::InstanceSegmenter(Engine& engine, InstanceSegConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {}

std::vector<InstanceMask> InstanceSegmenter::postprocess(const LetterboxInfo& lb,
                                                         int orig_w, int orig_h) const {
  const int scales = static_cast<int>(cfg_.strides.size());
  const int proto_idx = out_base_ + cfg_.proto_index;
  const int num_out = engine_.numOutputs();
  if (out_base_ < 0 || out_base_ + 3 * scales > num_out ||
      proto_idx < 0 || proto_idx >= num_out) {
    throw Error(-1, "BCDL InstanceSegmenter: output index range out of bounds");
  }

  // Read the prototype once; proto shape is [1,mH,mW,np] (NHWC) or [mH,mW,np].
  std::vector<float> proto_buf;
  std::vector<int> proto_shape;
  const float* proto = outputAsFloat(engine_, proto_idx, proto_buf, proto_shape);
  int mH = 0, mW = 0, pC = 0;
  if (proto_shape.size() == 4) {
    mH = proto_shape[1];
    mW = proto_shape[2];
    pC = proto_shape[3];
  } else if (proto_shape.size() == 3) {
    mH = proto_shape[0];
    mW = proto_shape[1];
    pC = proto_shape[2];
  }

  // Gather per-scale dequantized buffers (kept alive until decode runs) plus the
  // grid (H,W) and the trailing class / mask-coef counts.
  std::vector<std::vector<float>> cls_bufs(scales), box_bufs(scales), mc_bufs(scales);
  std::vector<const float*> cls_ptr(scales, nullptr), box_ptr(scales, nullptr),
      mc_ptr(scales, nullptr);
  std::vector<std::pair<int, int>> grid(scales, {0, 0});
  int nc = 0, np = 0;

  for (int s = 0; s < scales; ++s) {
    std::vector<int> cls_shape, box_shape, mc_shape;
    cls_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 0, cls_bufs[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 1, box_bufs[s], box_shape);
    mc_ptr[s] = outputAsFloat(engine_, out_base_ + 3 * s + 2, mc_bufs[s], mc_shape);

    // Grid + counts from each tensor's OWN shape ([1,H,W,*] or [H,W,*]).
    int H = 0, W = 0, this_nc = 0, this_np = 0;
    if (cls_shape.size() == 4) {
      H = cls_shape[1];
      W = cls_shape[2];
      this_nc = cls_shape[3];
    } else if (cls_shape.size() == 3) {
      H = cls_shape[0];
      W = cls_shape[1];
      this_nc = cls_shape[2];
    }
    if (mc_shape.size() == 4) {
      this_np = mc_shape[3];
    } else if (mc_shape.size() == 3) {
      this_np = mc_shape[2];
    }
    grid[s] = {H, W};
    if (this_nc > 0) {
      if (nc == 0) nc = this_nc;
      else if (this_nc != nc)
        throw Error(-1, "BCDL InstanceSegmenter: inconsistent class count across scales");
    }
    if (this_np > 0) {
      if (np == 0) np = this_np;
      else if (this_np != np)
        throw Error(-1, "BCDL InstanceSegmenter: inconsistent mask-coef count across scales");
    }
  }

  return decodeInstanceSeg(cls_ptr, box_ptr, mc_ptr, grid, nc, np, proto, mH, mW, pC,
                           cfg_, lb, orig_w, orig_h);
}

}  // namespace bcdl

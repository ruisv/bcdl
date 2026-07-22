#include "bcdl/pipeline/tracking_pipeline.h"

#include <algorithm>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"

// The motion-only path is a one-liner (detect, then associate) and lives in the
// header. This file exists for the appearance path, whose cost model is worth
// being explicit about: the ReID model runs ONCE PER CROP, so on a crowded frame
// it — not the detector — dominates the frame time. That is why the crops are
// filtered by score and capped, and why an unembedded detection is passed
// through as an empty vector rather than being dropped.

namespace bcdl {

TrackingPipeline::TrackingPipeline(Engine& engine, Engine& reid, PipelineConfig det_cfg,
                                   ByteTrackConfig track_cfg, TrackingReidConfig reid_cfg)
    : det_(engine, std::move(det_cfg)),
      tracker_(track_cfg),
      reid_(&reid),
      reid_cfg_(reid_cfg) {
  // Crop size comes from the model rather than from the caller, so a different
  // ReID model is a path change and nothing else.
  const std::vector<int> shape = reid.inputShape(0);
  if (shape.size() != 4 || shape[1] != 3) {
    throw Error(-1,
                "TrackingPipeline: the ReID model must take one NCHW 3-channel "
                "crop input (e.g. 1x3x256x128)");
  }
  reid_h_ = shape[2];
  reid_w_ = shape[3];
  if (reid_w_ <= 0 || reid_h_ <= 0) {
    throw Error(-1, "TrackingPipeline: ReID model has a degenerate input size");
  }
  embedder_ = std::make_unique<ImageEmbedder>(reid);
}

std::vector<Track> TrackingPipeline::process(const uint8_t* bgr, int width, int height) {
  last_dets_ = det_.process(bgr, width, height);
  if (reid_ == nullptr) {
    last_embed_count_ = 0;
    return tracker_.update(last_dets_);
  }

  embeddings_.assign(last_dets_.size(), {});
  int embedded = 0;
  for (std::size_t i = 0; i < last_dets_.size(); ++i) {
    if (last_dets_[i].score < reid_cfg_.min_score) continue;
    if (reid_cfg_.max_crops > 0 && embedded >= reid_cfg_.max_crops) break;
    const Detection& d = last_dets_[i];
    // A box can hang off the frame or collapse to nothing after clipping;
    // reidPreprocess throws there. That detection simply goes without an
    // appearance vector instead of taking the frame down with it.
    try {
      reidPreprocess(bgr, width, height, width * 3, d.x1, d.y1, d.x2, d.y2, reid_w_,
                     reid_h_, reid_cfg_.crop, crop_buf_);
    } catch (const Error&) {
      continue;
    }
    reid_->setInput(0, crop_buf_.data(), crop_buf_.size() * sizeof(float));
    reid_->infer();
    embeddings_[i] = embedder_->postprocess();
    ++embedded;
  }
  last_embed_count_ = embedded;
  return tracker_.update(last_dets_, embeddings_);
}

}  // namespace bcdl

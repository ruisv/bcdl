#include "bcdl/pipeline/detection_pipeline.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "bcdl/preproc/letterbox_cpu.h"

namespace bcdl {

// ---------------------------------------------------------------------------
// Shared pipeline core
// ---------------------------------------------------------------------------

PipelineConfig resolveDetectionConfig(Engine& engine, PipelineConfig cfg) {
  if (cfg.input_w <= 0 || cfg.input_h <= 0) {
    // Derive the spatial input size from input[0]. For the RDK NV12 layout the
    // Y plane is NHWC [1,H,W,1]; for a fused NCHW input it is [1,C,H,W].
    const std::vector<int> ishape = engine.inputShape(0);
    int in_h = 640, in_w = 640;
    if (ishape.size() == 4) {
      // NCHW [N,C,H,W] only when the channel dim is small and the trailing dim
      // is not; every other case (incl. the NV12 Y-plane [1,H,W,1]) is NHWC.
      const bool nchw = ishape[1] <= 4 && ishape[3] > 4;
      in_h = nchw ? ishape[2] : ishape[1];
      in_w = nchw ? ishape[3] : ishape[2];
    }
    if (cfg.input_w <= 0) cfg.input_w = in_w;
    if (cfg.input_h <= 0) cfg.input_h = in_h;
  }
  // Keep the detect canvas consistent with the resolved letterbox canvas.
  cfg.detect.input_w = cfg.input_w;
  cfg.detect.input_h = cfg.input_h;

  // Resolve kAuto: the LTRB head exposes one (cls,box) output pair per stride;
  // a box output's last dim is 4. Treat that exact shape as kYoloLtrb.
  if (cfg.head == DetectHead::kAuto) {
    const int want = 2 * static_cast<int>(cfg.ltrb_strides.size());
    bool ltrb = !cfg.ltrb_strides.empty() && engine.numOutputs() == want;
    if (ltrb) {
      for (int s = 0; s < static_cast<int>(cfg.ltrb_strides.size()); ++s) {
        const std::vector<int> bshape = engine.outputShape(2 * s + 1);
        if (bshape.empty() || bshape.back() != 4) {
          ltrb = false;
          break;
        }
      }
    }
    cfg.head = ltrb ? DetectHead::kYoloLtrb : DetectHead::kSingleTensor;
  }
  return cfg;
}

void requireNv12InputModel(Engine& engine) {
  // The NV12 pipelines produce uint8 NV12 and feed it straight to the model. A
  // float-input model (e.g. NCHW RGB) would silently receive NV12 bytes and
  // infer on garbage, so reject it loudly — such models must be preprocessed
  // externally and driven via Engine + Detector directly.
  const int it = engine.inputType(0);
  if (it == HB_DNN_TENSOR_TYPE_F32 || it == HB_DNN_TENSOR_TYPE_F16) {
    throw Error(-1,
                "DetectionPipeline feeds uint8 NV12, but model input 0 is float; "
                "preprocess externally and use Engine + Detector directly");
  }
}

void feedNv12Input(Engine& engine, const VpImage& nv12) {
  // NV12 is one contiguous block: the Y plane (stride * h) is followed
  // immediately by the interleaved half-res UV plane (uvStride * h/2), with
  // nv12.data() at Y.
  const auto* base = static_cast<const uint8_t*>(nv12.data());
  const std::size_t y_bytes =
      static_cast<std::size_t>(nv12.raw().stride) * static_cast<std::size_t>(nv12.height());
  const std::size_t uv_bytes =
      static_cast<std::size_t>(nv12.raw().uvStride) *
      static_cast<std::size_t>(nv12.height() / 2);

  if (engine.numInputs() >= 2) {
    engine.setInput(0, base, y_bytes);            // Y plane
    engine.setInput(1, base + y_bytes, uv_bytes);  // interleaved UV plane
  } else {
    engine.setInput(0, base, y_bytes + uv_bytes);  // fused Y+UV
  }
}

LetterboxInfo preprocBgrToNv12(VpImage& src, VpImage& lbBgr, VpImage& nv12Dst,
                               const uint8_t* bgr, int width, int height,
                               uint8_t pad_value) {
  // (Re)allocate the source wrapper only when the incoming dimensions change.
  if (!src.valid() || src.width() != width || src.height() != height) {
    src = VpImage(width, height, HB_VP_IMAGE_FORMAT_BGR);
  }
  // Copy the caller's contiguous BGR rows (stride == width*3) into src,
  // honoring src's (possibly larger, 16-byte-aligned) row stride.
  auto* dst = static_cast<uint8_t*>(src.data());
  const int dst_stride = src.raw().stride;
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
  for (int r = 0; r < height; ++r) {
    std::memcpy(dst + static_cast<std::size_t>(r) * dst_stride,
                bgr + static_cast<std::size_t>(r) * row_bytes, row_bytes);
  }
  // Letterbox in BGR into the persistent canvas, then convert to NV12 into the
  // persistent target (both cleanCache() their dst, so the NV12 pixels are
  // flushed to DRAM before the BPU reads them).
  const LetterboxInfo lb = letterboxCpu(lbBgr, src, pad_value);
  bgrToNv12Cpu(nv12Dst, lbBgr);
  return lb;
}

HeadDecoder::HeadDecoder(Engine& engine, const PipelineConfig& cfg) {
  if (cfg.head == DetectHead::kYoloLtrb) {
    YoloLtrbConfig lc;
    lc.num_classes = cfg.detect.num_classes;
    lc.conf_thresh = cfg.detect.conf_thresh;
    lc.iou_thresh = cfg.detect.iou_thresh;
    lc.max_dets = cfg.detect.max_dets;
    lc.strides = cfg.ltrb_strides;
    ltrb_ = std::make_unique<YoloLtrbDetector>(engine, std::move(lc), cfg.output_index);
  } else {
    single_ = std::make_unique<Detector>(engine, cfg.detect, cfg.output_index);
  }
}

std::vector<Detection> HeadDecoder::postprocess(const LetterboxInfo& lb) const {
  return ltrb_ ? ltrb_->postprocess(lb) : single_->postprocess(lb);
}

// ---------------------------------------------------------------------------
// DetectionPipeline (synchronous)
// ---------------------------------------------------------------------------

DetectionPipeline::DetectionPipeline(Engine& engine, PipelineConfig cfg)
    : engine_(engine),
      cfg_(resolveDetectionConfig(engine, cfg)),  // cfg_ before the buffers
      src_(),                                      // lazily sized to first frame
      lb_bgr_(cfg_.input_w, cfg_.input_h, HB_VP_IMAGE_FORMAT_BGR),
      nv12_(cfg_.input_w, cfg_.input_h, HB_VP_IMAGE_FORMAT_NV12),
      decoder_(engine_, cfg_),
      last_lb_() {
  requireNv12InputModel(engine_);
}

std::vector<Detection> DetectionPipeline::process(const uint8_t* bgr, int width, int height) {
  if (bgr == nullptr || width <= 0 || height <= 0) {
    throw Error(-1, "DetectionPipeline::process: invalid BGR frame");
  }

  last_lb_ = preprocBgrToNv12(src_, lb_bgr_, nv12_, bgr, width, height, cfg_.pad_value);
  feedNv12Input(engine_, nv12_);

  // Synchronous infer (submit + wait + invalidate output caches inside Engine).
  engine_.infer();

  // Decode + per-class NMS; boxes mapped back to original-image pixels via the
  // letterbox geometry of this frame.
  return decoder_.postprocess(last_lb_);
}

}  // namespace bcdl

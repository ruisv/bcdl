#include "bcdl/pipeline/async_detection_pipeline.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "bcdl/preproc/letterbox_cpu.h"  // letterboxNv12Cpu (fallback)
#include "bcdl/preproc/vp_image.h"
#ifdef BCDL_HAVE_GDC
#include "bcdl/preproc/gdc_letterbox.h"
#endif

namespace bcdl {

namespace {

/// A bounded, closable multi-producer/consumer queue. push() blocks while full;
/// pop() blocks while empty. Once close()d: push() returns false; pop() drains
/// the remaining items and then returns false. This gives backpressure and a
/// clean shutdown handshake for the pipeline stages.
template <class T>
class BoundedChannel {
 public:
  explicit BoundedChannel(std::size_t cap) : cap_(cap) {}

  bool push(T v) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push(std::move(v));
    not_empty_.notify_one();
    return true;
  }

  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;  // closed and drained
    out = std::move(q_.front());
    q_.pop();
    not_full_.notify_one();
    return true;
  }

  /// Non-blocking pop: fills `out` and returns true if an item was available,
  /// else returns false immediately (whether or not the channel is closed).
  bool tryPop(T& out) {
    std::lock_guard<std::mutex> lk(m_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    not_full_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closed_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable not_full_, not_empty_;
  std::queue<T> q_;
  std::size_t cap_;
  bool closed_ = false;
};

struct Frame {
  std::vector<uint8_t> bgr;  // owned copy of the caller's frame
  int w = 0;
  int h = 0;
};

struct Ready {
  int slot = -1;
  LetterboxInfo lb;
};

}  // namespace

struct AsyncDetectionPipeline::Impl {
  Engine& engine;
  PipelineConfig cfg;
  HeadDecoder decoder;
  int depth;

  // NV12-native letterboxer for submitNv12(), built for the first frame's size.
  // GDC when the hardware is there (CPU idle during the warp), else CPU.
  // BCDL_NO_GDC=1 forces the CPU path — useful for A/B.
#ifdef BCDL_HAVE_GDC
  std::unique_ptr<GdcLetterbox> gdc;
  int gdc_w = 0, gdc_h = 0;
  YuvRange gdc_range = YuvRange::kStudioToFull;
  bool gdc_off = std::getenv("BCDL_NO_GDC") != nullptr;
#endif

  std::vector<VpImage> slots;  // `depth` NV12 buffers, one per in-flight frame

  BoundedChannel<Frame> raw;     // submitted BGR frames awaiting preproc
  BoundedChannel<int> freeq;     // indices of NV12 slots available to write
  BoundedChannel<Ready> ready;   // filled slots awaiting infer
  // Finished detections (FIFO). UNBOUNDED on purpose: `results` is the caller's
  // queue, and only the caller drains it. Bounding it lets a full results queue
  // block the infer thread, which stops slots from recycling, which stops every
  // upstream stage — and the caller cannot drain because it is itself blocked in
  // submit(). That ring deadlocks. (The vendor's sunrise_camera avoids the same
  // trap by handing BPU results to a callback queue rather than making its decode
  // path wait on the consumer.) Occupancy is bounded by how far behind the caller
  // is; the frames in flight are still capped by `slots`.
  BoundedChannel<std::vector<Detection>> results;

  std::thread preproc_thr;
  std::thread infer_thr;
  // finish() may race: AsyncVideoDetectionPipeline calls it both from its cvt
  // thread (normal end of stream) and from its destructor (to unblock that same
  // thread if it is stuck in submit()).
  std::atomic<bool> finished{false};

  // Per-stage service timing. preproc_ms is written only by preproc_thr; the
  // infer/postproc/frames fields only by infer_thr — so each is single-writer
  // and safe to read once both threads have joined (after finish()/destruction).
  double preproc_ms = 0;
  double infer_ms = 0;
  double postproc_ms = 0;
  uint64_t prof_frames = 0;

  Impl(Engine& e, PipelineConfig resolved, int d)
      : engine(e),
        cfg(std::move(resolved)),
        decoder(e, cfg),
        depth(d),
        raw(d),
        freeq(d),
        ready(d),
        results(std::numeric_limits<std::size_t>::max()) {
    requireNv12InputModel(engine);
    for (int i = 0; i < depth; ++i) {
      slots.emplace_back(cfg.input_w, cfg.input_h, HB_VP_IMAGE_FORMAT_NV12);
      freeq.push(i);  // all slots start free
    }
    preproc_thr = std::thread([this] { preprocLoop(); });
    infer_thr = std::thread([this] { inferLoop(); });
  }

  ~Impl() {
    // Unblock both workers from any wait and let them exit, then join.
    raw.close();
    freeq.close();
    ready.close();
    results.close();
    if (preproc_thr.joinable()) preproc_thr.join();
    if (infer_thr.joinable()) infer_thr.join();
  }

  // Letterbox an NV12 source straight into a model-input slot. Runs on whatever
  // thread called submitNv12().
  LetterboxInfo letterboxInto(VpImage& dst, const VpImage& src, YuvRange range) {
#ifdef BCDL_HAVE_GDC
    if (!gdc_off) {
      if (!gdc || gdc_w != src.width() || gdc_h != src.height() || gdc_range != range) {
        try {
          gdc = std::make_unique<GdcLetterbox>(src.width(), src.height(), cfg.input_w,
                                               cfg.input_h, cfg.pad_value, range);
          gdc_w = src.width();
          gdc_h = src.height();
          gdc_range = range;
        } catch (const std::exception&) {
          gdc.reset();     // no /dev/gdc, or it is busy: fall back for good
          gdc_off = true;
        }
      }
      if (gdc) return gdc->run(src, dst);
    }
#endif
    return letterboxNv12Cpu(dst, src, cfg.pad_value, range);
  }

  // Preproc stage: one thread, owns its BGR/letterbox scratch (so reuse across
  // frames is allocation-free in steady state).
  void preprocLoop() {
    VpImage src, lb_bgr(cfg.input_w, cfg.input_h, HB_VP_IMAGE_FORMAT_BGR);
    Frame f;
    while (raw.pop(f)) {
      int slot;
      if (!freeq.pop(slot)) break;  // shutting down
      auto a = std::chrono::steady_clock::now();
      const LetterboxInfo lb = preprocBgrToNv12(src, lb_bgr, slots[slot],
                                                f.bgr.data(), f.w, f.h, cfg.pad_value);
      preproc_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - a).count();
      if (!ready.push(Ready{slot, lb})) break;  // shutting down
    }
    ready.close();  // no more preprocessed frames will arrive
  }

  // Infer stage: one thread = the Engine's single thread of use.
  void inferLoop() {
    Ready item;
    while (ready.pop(item)) {
      auto a = std::chrono::steady_clock::now();
      feedNv12Input(engine, slots[item.slot]);
      engine.infer();
      auto b = std::chrono::steady_clock::now();
      std::vector<Detection> dets = decoder.postprocess(item.lb);
      auto c = std::chrono::steady_clock::now();
      infer_ms += std::chrono::duration<double, std::milli>(b - a).count();
      postproc_ms += std::chrono::duration<double, std::milli>(c - b).count();
      ++prof_frames;
      freeq.push(item.slot);  // slot reusable now that infer consumed it
      if (!results.push(std::move(dets))) break;  // shutting down
    }
    results.close();  // no more results will arrive
  }
};

AsyncDetectionPipeline::AsyncDetectionPipeline(Engine& engine, PipelineConfig cfg, int depth)
    : cfg_(resolveDetectionConfig(engine, cfg)) {
  if (depth < 2) depth = 2;  // need >= 2 slots to overlap
  impl_ = std::make_unique<Impl>(engine, cfg_, depth);
}

AsyncDetectionPipeline::~AsyncDetectionPipeline() = default;

bool AsyncDetectionPipeline::submit(const uint8_t* bgr, int width, int height) {
  if (bgr == nullptr || width <= 0 || height <= 0) {
    throw Error(-1, "AsyncDetectionPipeline::submit: invalid BGR frame");
  }
  Frame f;
  f.w = width;
  f.h = height;
  f.bgr.assign(bgr, bgr + static_cast<std::size_t>(width) * height * 3);
  return impl_->raw.push(std::move(f));
}

int AsyncDetectionPipeline::acquireSlot() {
  int slot;
  if (!impl_->freeq.pop(slot)) return -1;  // finished / shutting down
  return slot;
}

LetterboxInfo AsyncDetectionPipeline::letterboxIntoSlot(int slot, const VpImage& nv12,
                                                        YuvRange range) {
  if (!nv12.valid() || nv12.format() != HB_VP_IMAGE_FORMAT_NV12) {
    throw Error(-1, "AsyncDetectionPipeline: NV12 submit needs a valid NV12 VpImage");
  }
  const auto t0 = std::chrono::steady_clock::now();
  const LetterboxInfo lb = impl_->letterboxInto(impl_->slots[slot], nv12, range);
  // Single-writer still holds: a pipeline is driven either by submit() (preproc
  // thread writes this) or by the NV12 path (one caller thread does), never both.
  impl_->preproc_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return lb;
}

bool AsyncDetectionPipeline::commitSlot(int slot, const LetterboxInfo& lb) {
  return impl_->ready.push(Ready{slot, lb});
}

void AsyncDetectionPipeline::releaseSlot(int slot) { impl_->freeq.push(slot); }

bool AsyncDetectionPipeline::submitNv12(const VpImage& nv12, YuvRange range) {
  const int slot = acquireSlot();
  if (slot < 0) return false;
  return commitSlot(slot, letterboxIntoSlot(slot, nv12, range));
}

bool AsyncDetectionPipeline::next(std::vector<Detection>& out) {
  return impl_->results.pop(out);
}

bool AsyncDetectionPipeline::tryNext(std::vector<Detection>& out) {
  return impl_->results.tryPop(out);
}

void AsyncDetectionPipeline::finish() {
  if (impl_->finished.exchange(true)) return;  // idempotent under concurrent calls
  impl_->raw.close();  // preproc drains raw -> closes ready -> infer closes results
}

StageProfile AsyncDetectionPipeline::profile() const {
  // Single-writer fields; call after finish() + full drain for a settled read.
  StageProfile p;
  p.preproc_ms = impl_->preproc_ms;
  p.infer_ms = impl_->infer_ms;
  p.postproc_ms = impl_->postproc_ms;
  p.frames = impl_->prof_frames;
  return p;
}

}  // namespace bcdl

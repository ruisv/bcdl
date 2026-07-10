#include "bcdl/pipeline/async_video_detection_pipeline.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include "bcdl/backend/engine.h"
#include "bcdl/core/status.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/preproc/vp_image.h"

namespace bcdl {

namespace {

constexpr int kRetryDrainMs = 5;  // brief wait when the codec's input queue is full

int nalType(uint8_t b, bool h265) { return h265 ? (b >> 1) & 0x3F : b & 0x1F; }
bool isVcl(int t, bool h265) { return h265 ? (t >= 0 && t <= 31) : (t >= 1 && t <= 5); }

// A bounded, closable move-only channel connecting one pipeline stage to the
// next: push() blocks while full (backpressure), pop() blocks while empty,
// close() drains then ends. Used for AUs (submit→decode) and NV12 frames
// (decode→letterbox).
template <class T>
class Channel {
 public:
  explicit Channel(std::size_t cap) : cap_(cap) {}
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
  std::size_t cap_;
  std::mutex m_;
  std::condition_variable not_full_, not_empty_;
  std::queue<T> q_;
  bool closed_ = false;
};

}  // namespace

struct AsyncVideoDetectionPipeline::Impl {
  bool h265;
  VideoDecoder dec;               // created here (main); fed AND drained on dec_thr
                                  // — one thread owns the codec context
  AsyncDetectionPipeline detect;  // internal preproc ‖ infer overlap

  Channel<std::vector<uint8_t>> au_q;   // submit -> feed
  Channel<VpImage> nv12_q;              // decode -> letterbox (own stage so the
                                        // letterbox hides behind decode)
  Channel<VpImage> free_q;              // letterbox -> decode: recycled buffers
  std::thread dec_thr, lb_thr;
  bool finished = false;

  // Annex-B reassembly state — touched only by submit() (single caller thread).
  std::vector<uint8_t> buf;

  // Per-stage service time; each field written by exactly one thread, so it is
  // safe to read once the threads have joined (after finish() + full drain).
  double decode_ms = 0;  // VPU decode wait/frame (decode thread)

  Impl(Engine& engine, PipelineConfig cfg, hbVPVideoType codec, int depth)
      : h265(codec == HB_VP_VIDEO_TYPE_H265),
        dec(VideoDecConfig{codec, HB_VP_IMAGE_FORMAT_NV12, 0}),
        detect(engine, std::move(cfg), depth),
        au_q(static_cast<std::size_t>(depth) + 2),
        nv12_q(static_cast<std::size_t>(depth) + 2),
        free_q(static_cast<std::size_t>(depth) + 3) {
    // Seed the recycling pool. Enough buffers to fill nv12_q AND leave one in
    // flight on each of the decode and letterbox stages, so neither ever waits on the
    // pool when the other is the bottleneck.
    for (int i = 0; i < depth + 3; ++i) free_q.push(VpImage{});
    dec_thr = std::thread([this] { decodeLoop(); });
    lb_thr = std::thread([this] { lbLoop(); });
  }

  ~Impl() {
    au_q.close();    // unblock the decode thread
    free_q.close();  // ...including when it waits for a recycled buffer
    nv12_q.close();  // unblock the letterbox thread
    // lbLoop can also be blocked INSIDE detect.submitNv12(), on the detector's own
    // backpressure — a caller that destroys us without draining results leaves
    // that queue full. Closing our channels cannot reach it, so finish() the
    // detector too; submitNv12() then returns false instead of waiting forever.
    detect.finish();
    if (dec_thr.joinable()) dec_thr.join();
    if (lb_thr.joinable()) lb_thr.join();
  }

  // Dequeue one ready frame into a recycled buffer and forward it.
  // Returns 1 = forwarded, 0 = nothing ready right now, -1 = shutting down.
  //
  // Decode into a RECYCLED buffer from free_q, not a fresh VpImage: receive()
  // reuses a buffer whose size already matches, so after warm-up this stage does
  // no hbUCPMallocCached/hbUCPFree at all. (Moving the frame into nv12_q leaves
  // the local empty, which is why a plain reused local would still allocate every
  // frame.) An empty free_q is backpressure — the letterbox stage owes us a buffer.
  int drainOne(int timeout_ms) {
    VpImage nv12;
    if (!free_q.pop(nv12)) return -1;  // shutting down
    const auto t0 = std::chrono::steady_clock::now();
    if (!dec.receive(nv12, timeout_ms)) {
      free_q.push(std::move(nv12));  // nothing decoded: put the buffer back
      return 0;
    }
    decode_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return nv12_q.push(std::move(nv12)) ? 1 : -1;
  }

  // Decode stage: feed one access unit, then drain every frame the decoder has
  // ready, then feed the next. One thread owns the codec context.
  //
  // Feeding and draining MUST NOT run concurrently on the same context: a
  // blocking dequeue_output_buffer holds the codec's internal lock, so a feed
  // thread racing a recv thread stalls whenever the decoder has no frame to give.
  // H.264 hides this (a frame is nearly always ready), but a reorder codec has
  // gaps: driving HEVC that way collapsed to 16 FPS and dropped 17 of 300 frames.
  // The drain cadence is also what keeps the decoder's frame buffers from filling
  // up, which is what makes the VPU stall with INTERRUPT TIMEOUT.
  void decodeLoop() {
    std::vector<uint8_t> au;
    bool stop = false;
    while (!stop && au_q.pop(au)) {
      while (!dec.feed(au.data(), au.size())) {  // input queue full: drain, retry
        const int r = drainOne(kRetryDrainMs);
        if (r < 0) { stop = true; break; }
        if (r == 0) break;  // still no room and nothing to drain: drop this AU
      }
      if (stop) break;
      int r;
      while ((r = drainOne(0)) == 1) {}  // drain everything ready right now
      if (r < 0) break;
    }
    // End of stream: flush the reorder tail. Without this the last picture (and,
    // for HEVC, several) never come out.
    while (!stop) {
      VpImage nv12;
      if (!free_q.pop(nv12)) break;
      if (!dec.flush(nv12)) {
        free_q.push(std::move(nv12));
        break;
      }
      if (!nv12_q.push(std::move(nv12))) break;
    }
    nv12_q.close();  // no more decoded frames
  }

  // Letterbox stage: feed the decoded NV12 straight to the detector, which
  // letterboxes it into a model-input slot (GDC hardware when present, else CPU).
  // The model input IS NV12, so there is no colour conversion here at all — this
  // path used to go NV12 -> BGR -> letterbox -> NV12, two conversions and four
  // full-frame copies to end up where it started. Own thread so the letterbox
  // overlaps VPU decode instead of serializing behind it.
  // Wait for detector capacity BEFORE taking a frame. A recycled NV12 buffer held
  // while blocked in the detector is a buffer the decode thread cannot have, and
  // the decode thread is what drains au_q — so submit() would block forever on a
  // full au_q while the caller, stuck in submit(), never drains the results the
  // infer stage is waiting to push. Acquiring the slot first holds nothing.
  void lbLoop() {
    while (true) {
      const int slot = detect.acquireSlot();  // holds no frame while it waits
      if (slot < 0) break;                    // finished / shutting down
      VpImage nv12;
      if (!nv12_q.pop(nv12)) {                // decoder drained: give the slot back
        detect.releaseSlot(slot);
        break;
      }
      const LetterboxInfo lb = detect.letterboxIntoSlot(slot, nv12);
      free_q.push(std::move(nv12));           // frame consumed: recycle it at once
      if (!detect.commitSlot(slot, lb)) break;
    }
    detect.finish();  // no more frames will be submitted to the detector
  }

  // Split complete access units out of `buf` and push them to the decode thread.
  // ONE AU PER PUSH is mandatory: the decoder feeds in FRAME_SIZE mode, where an
  // input buffer holding two concatenated pictures decodes only the first.
  //
  // While streaming (`final == false`) the last start code is left alone — its
  // NAL may still be incomplete — so the trailing AU stays buffered. At EOS
  // (`final == true`) no more bytes are coming: every start code is a real
  // boundary, and whatever follows the last one is the final AU.
  // Returns false if the pipeline is shutting down.
  bool drainAus(bool final) {
    // Start-code offsets (3-byte 00 00 01) across the current buffer.
    std::vector<std::size_t> sc;
    for (std::size_t i = 0; i + 3 <= buf.size(); ++i)
      if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1) sc.push_back(i);
    if (sc.empty()) {
      if (final) buf.clear();  // no NALs at all: nothing decodable
      return true;
    }
    if (!final && sc.size() < 2) return true;  // need more bytes to bound an AU

    const std::size_t last = final ? sc.size() : sc.size() - 1;
    std::size_t cur = sc[0];
    bool has_vcl = false;
    for (std::size_t k = 0; k < last; ++k) {
      const std::size_t p = sc[k];
      const bool vcl = isVcl(nalType(buf[p + 3], h265), h265);
      if (vcl && has_vcl) {
        if (!au_q.push(std::vector<uint8_t>(buf.begin() + cur, buf.begin() + p)))
          return false;
        cur = p;
        has_vcl = false;
      }
      if (vcl) has_vcl = true;
    }
    if (final) {
      if (cur < buf.size())  // the bytes after the last boundary are the last AU
        if (!au_q.push(std::vector<uint8_t>(buf.begin() + cur, buf.end()))) return false;
      buf.clear();
    } else {
      buf.erase(buf.begin(), buf.begin() + cur);  // keep the unfinished tail
    }
    return true;
  }
};

AsyncVideoDetectionPipeline::AsyncVideoDetectionPipeline(Engine& engine, PipelineConfig cfg,
                                                         hbVPVideoType codec, int depth) {
  if (depth < 2) depth = 2;
  impl_ = std::make_unique<Impl>(engine, std::move(cfg), codec, depth);
}

AsyncVideoDetectionPipeline::~AsyncVideoDetectionPipeline() = default;

bool AsyncVideoDetectionPipeline::submit(const uint8_t* data, std::size_t size) {
  if (impl_->finished) return false;
  if (data == nullptr || size == 0) return true;
  impl_->buf.insert(impl_->buf.end(), data, data + size);
  return impl_->drainAus(/*final=*/false);
}

bool AsyncVideoDetectionPipeline::next(std::vector<Detection>& out) {
  return impl_->detect.next(out);
}

bool AsyncVideoDetectionPipeline::tryNext(std::vector<Detection>& out) {
  return impl_->detect.tryNext(out);
}

void AsyncVideoDetectionPipeline::finish() {
  if (impl_->finished) return;
  impl_->finished = true;
  // Split the trailing bytes into access units and push them individually. They
  // can hold MORE than one AU: streaming never bounds the final start code, so
  // the last coded picture (and sometimes the one before it) is still in `buf`.
  impl_->drainAus(/*final=*/true);
  impl_->au_q.close();  // decode thread drains remaining AUs, then finishes detect
}

StageProfile AsyncVideoDetectionPipeline::profile() const {
  StageProfile p = impl_->detect.profile();  // preproc / infer / postproc / frames
  // Single-writer (recv thread); settled once it has joined. cvt_ms stays 0: the
  // NV12-native path has no colour conversion, and the letterbox is reported as
  // preproc_ms by submitNv12().
  p.decode_ms = impl_->decode_ms;
  return p;
}

}  // namespace bcdl

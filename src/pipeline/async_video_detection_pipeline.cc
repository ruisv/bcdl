#include "bcdl/pipeline/async_video_detection_pipeline.h"

#include <atomic>
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
#include "bcdl/preproc/letterbox_cpu.h"  // nv12ToBgrCpu
#include "bcdl/preproc/vp_image.h"

namespace bcdl {

namespace {

constexpr int kRecvMs = 200;  // recv-thread blocking wait; returns instantly when a
                              // frame is ready, so this only bounds end-of-stream /
                              // starvation latency (not steady-state throughput)

int nalType(uint8_t b, bool h265) { return h265 ? (b >> 1) & 0x3F : b & 0x1F; }
bool isVcl(int t, bool h265) { return h265 ? (t >= 0 && t <= 31) : (t >= 1 && t <= 5); }

// A bounded, closable move-only channel connecting one pipeline stage to the
// next: push() blocks while full (backpressure), pop() blocks while empty,
// close() drains then ends. Used for AUs (submit→decode) and NV12 frames
// (decode→cvt).
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
  VideoDecoder dec;               // created here (main); fed on feed_thr, drained
                                  // on recv_thr — the sample_codec threading model
  AsyncDetectionPipeline detect;  // internal preproc ‖ infer overlap

  Channel<std::vector<uint8_t>> au_q;   // submit -> feed
  Channel<VpImage> nv12_q;              // decode -> cvt (own stage so cvt hides
                                        // behind decode, keeping decode the floor)
  std::thread feed_thr, recv_thr, cvt_thr;
  std::atomic<bool> fed_all{false};     // feed thread finished (incl. EOS marker)
  bool finished = false;

  // Annex-B reassembly state — touched only by submit() (single caller thread).
  std::vector<uint8_t> buf;

  // Per-stage service time; each field written by exactly one thread, so it is
  // safe to read once the threads have joined (after finish() + full drain).
  double decode_ms = 0;  // VPU decode wait/frame (recv thread)
  double cvt_ms = 0;     // NV12->BGR (cvt thread)

  Impl(Engine& engine, PipelineConfig cfg, hbVPVideoType codec, int depth)
      : h265(codec == HB_VP_VIDEO_TYPE_H265),
        dec(VideoDecConfig{codec, HB_VP_IMAGE_FORMAT_NV12, 0}),
        detect(engine, std::move(cfg), depth),
        au_q(static_cast<std::size_t>(depth) + 2),
        nv12_q(static_cast<std::size_t>(depth) + 2) {
    feed_thr = std::thread([this] { feedLoop(); });
    recv_thr = std::thread([this] { recvLoop(); });
    cvt_thr = std::thread([this] { cvtLoop(); });
  }

  ~Impl() {
    au_q.close();    // unblock feed thread
    nv12_q.close();  // unblock cvt thread
    if (feed_thr.joinable()) feed_thr.join();
    if (recv_thr.joinable()) recv_thr.join();
    if (cvt_thr.joinable()) cvt_thr.join();
  }

  // Feed stage: pump access units into the decoder as fast as it accepts them
  // (blocks only when the codec's input queue is full). Decoupling feed from the
  // output drain is what makes reorder codecs (HEVC) run at the VPU's decode rate
  // instead of being throttled by a per-AU output poll.
  void feedLoop() {
    std::vector<uint8_t> au;
    while (au_q.pop(au)) dec.feed(au.data(), au.size());
    dec.feedEndOfStream();
    fed_all.store(true);
  }

  // Receive stage: drain decoded frames in display order as the VPU produces
  // them, and hand them to the cvt stage. A blocking receive() waits on real
  // frames (no busy poll); a timeout after the feed thread is done means the
  // decoder is fully drained.
  void recvLoop() {
    VpImage nv12;
    while (true) {
      auto t0 = std::chrono::steady_clock::now();
      bool got = dec.receive(nv12, kRecvMs);
      if (got) {
        decode_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (!nv12_q.push(std::move(nv12))) break;  // shutting down
      } else if (fed_all.load()) {
        break;  // feed done + no more frames within the timeout => drained
      }
    }
    nv12_q.close();  // no more decoded frames
  }

  // Colour-convert stage: NV12 -> BGR (CPU), then feed the detector. Separate
  // thread so this overlaps VPU decode instead of serializing behind it.
  void cvtLoop() {
    VpImage nv12, bgr;
    std::vector<uint8_t> contig;  // fallback for non-contiguous BGR strides
    while (nv12_q.pop(nv12)) {
      const int w = nv12.width(), h = nv12.height();
      auto t0 = std::chrono::steady_clock::now();
      if (bgr.width() != w || bgr.height() != h)
        bgr = VpImage(w, h, HB_VP_IMAGE_FORMAT_BGR);
      nv12ToBgrCpu(bgr, nv12);
      cvt_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();

      const int stride = bgr.raw().stride;
      const auto* data = static_cast<const uint8_t*>(bgr.data());
      if (stride == w * 3) {
        detect.submit(data, w, h);       // contiguous (16-aligned widths)
      } else {
        contig.resize(static_cast<std::size_t>(w) * h * 3);
        for (int r = 0; r < h; ++r)
          std::memcpy(contig.data() + static_cast<std::size_t>(r) * w * 3,
                      data + static_cast<std::size_t>(r) * stride,
                      static_cast<std::size_t>(w) * 3);
        detect.submit(contig.data(), w, h);
      }
    }
    detect.finish();  // no more frames will be submitted to the detector
  }

  // Extract complete access units from `buf` and push them to the decode thread.
  // The trailing (still-growing) AU stays buffered until a later start code — or
  // finish() — proves it complete. Returns false if the pipeline is finished.
  bool drainAus() {
    // Start-code offsets (3-byte 00 00 01) across the current buffer.
    std::vector<std::size_t> sc;
    for (std::size_t i = 0; i + 3 <= buf.size(); ++i)
      if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1) sc.push_back(i);
    if (sc.size() < 2) return true;  // need more bytes to bound an AU

    std::size_t cur = sc[0];
    bool has_vcl = false;
    for (std::size_t k = 0; k + 1 < sc.size(); ++k) {  // last NAL may be incomplete
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
    buf.erase(buf.begin(), buf.begin() + cur);  // keep the unfinished tail
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
  return impl_->drainAus();
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
  // Flush the trailing access unit (the last coded picture has no following
  // start code to bound it during streaming).
  if (!impl_->buf.empty()) {
    impl_->au_q.push(std::move(impl_->buf));
    impl_->buf.clear();
  }
  impl_->au_q.close();  // decode thread drains remaining AUs, then finishes detect
}

StageProfile AsyncVideoDetectionPipeline::profile() const {
  StageProfile p = impl_->detect.profile();  // preproc / infer / postproc / frames
  p.decode_ms = impl_->decode_ms;            // single-writer; settled after drain
  return p;
}

}  // namespace bcdl

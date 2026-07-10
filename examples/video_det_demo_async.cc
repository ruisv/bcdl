// 3-stage overlapped video detection — the throughput counterpart to the serial
// video_det_demo. The serial demo runs decode -> nv12->bgr -> detect one frame
// fully before the next, so its wall time is the SUM of the three stages, each
// on a different unit (VPU / CPU / BPU) with the other two idle.
//
// This demo overlaps them across frames with a producer thread + the library's
// AsyncDetectionPipeline (which itself overlaps CPU preproc with BPU infer+NMS):
//
//   decode thread : AU --VPU--> NV12 --CPU cvt--> BGR --push--> bounded queue
//   AsyncDetectionPipeline : preproc (BGR->NV12 letterbox) || infer + NMS
//
// So the three hardware stages run concurrently and steady-state throughput
// approaches 1 / max(decode+cvt, preproc, infer+nms) instead of the serial sum.
// The same binary runs a serial baseline first, then the async path, over the
// SAME stream, and prints both end-to-end FPS for a direct comparison.
//
//   ./video_det_demo_async <det.hbm> <in.h264|in.h265> [max_frames] [depth]

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bcdl/backend/engine.h"
#include "bcdl/media/video_codec.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/detection_pipeline.h"
#include "bcdl/preproc/vp_image.h"

namespace {

std::vector<uint8_t> readFile(const char* path) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) throw std::runtime_error(std::string("cannot open ") + path);
  std::fseek(fp, 0, SEEK_END);
  long n = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> buf(n > 0 ? static_cast<size_t>(n) : 0);
  if (n > 0 && std::fread(buf.data(), 1, buf.size(), fp) != buf.size()) {
    std::fclose(fp);
    throw std::runtime_error("short read");
  }
  std::fclose(fp);
  return buf;
}

int nalType(uint8_t b, bool h265) { return h265 ? (b >> 1) & 0x3F : b & 0x1F; }
bool isVcl(int t, bool h265) { return h265 ? (t >= 0 && t <= 31) : (t >= 1 && t <= 5); }

std::vector<std::pair<size_t, size_t>> accessUnits(const std::vector<uint8_t>& s, bool h265) {
  std::vector<std::pair<size_t, size_t>> aus;
  std::vector<size_t> sc;
  for (size_t i = 0; i + 3 <= s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) sc.push_back(i);
  if (sc.empty()) return aus;
  size_t au_begin = sc[0];
  bool au_has_vcl = false;
  for (size_t pos : sc) {
    size_t hdr = pos + 3;
    if (hdr >= s.size()) break;
    bool vcl = isVcl(nalType(s[hdr], h265), h265);
    if (vcl && au_has_vcl) {
      aus.emplace_back(au_begin, pos);
      au_begin = pos;
      au_has_vcl = false;
    }
    if (vcl) au_has_vcl = true;
  }
  aus.emplace_back(au_begin, s.size());
  return aus;
}

cv::Mat nv12ToBgr(const bcdl::VpImage& f) {
  const int w = f.width(), h = f.height();
  cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
  const auto* y = static_cast<const uint8_t*>(f.data());
  const int ys = f.raw().stride;
  for (int r = 0; r < h; ++r) std::memcpy(nv12.ptr(r), y + (size_t)r * ys, w);
  const auto* uv = static_cast<const uint8_t*>(f.raw().uvVirAddr);
  const int us = f.raw().uvStride;
  for (int r = 0; r < h / 2; ++r) std::memcpy(nv12.ptr(h + r), uv + (size_t)r * us, w);
  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

// A minimal bounded, closable channel between the decode thread and the main
// bridge loop. push() blocks while full (backpressure so the decoder can't
// outrun the detector); pop() blocks while empty until closed+drained. Carries
// move-only NV12 VpImages so the CPU nv12->bgr conversion can run OFF the decode
// thread (in the consumer), leaving the decode thread doing pure VPU decode.
class FrameChannel {
 public:
  explicit FrameChannel(size_t cap) : cap_(cap) {}
  bool push(bcdl::VpImage v) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push(std::move(v));
    not_empty_.notify_one();
    return true;
  }
  bool pop(bcdl::VpImage& out) {
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
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  size_t cap_;
  std::mutex m_;
  std::condition_variable not_full_, not_empty_;
  std::queue<bcdl::VpImage> q_;
  bool closed_ = false;
};

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <det.hbm> <in.h264|in.h265> [max_frames] [depth]\n", argv[0]);
    return 1;
  }
  const char* hbm = argv[1];
  const char* in_path = argv[2];
  const int max_frames = argc > 3 ? std::atoi(argv[3]) : 0;
  const int depth = argc > 4 ? std::atoi(argv[4]) : 4;

  std::string ext;
  if (const char* dot = std::strrchr(in_path, '.')) ext = dot;
  const bool h265 = (ext == ".h265" || ext == ".hevc" || ext == ".265");
  const auto vtype = h265 ? HB_VP_VIDEO_TYPE_H265 : HB_VP_VIDEO_TYPE_H264;

  try {
    auto stream = readFile(in_path);
    auto aus = accessUnits(stream, h265);
    if (aus.empty()) {
      std::fprintf(stderr, "no NAL start codes — not an Annex-B stream?\n");
      return 2;
    }

    bcdl::Engine engine(hbm);
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;  // COCO
    cfg.detect.conf_thresh = 0.25f;
    cfg.detect.iou_thresh = 0.45f;

    std::printf("model=%s  | video=%s (%s, %zu AUs) | depth=%d\n",
                engine.modelName().c_str(), in_path, h265 ? "H.265" : "H.264",
                aus.size(), depth);

    // ---- Pass 1: serial baseline (decode -> cvt -> sync detect), timed. ----
    long serial_dets = 0;
    int serial_frames = 0;
    double serial_ms = 0;
    {
      bcdl::VideoDecoder dec({vtype});
      bcdl::DetectionPipeline pipe(engine, cfg);
      bcdl::VpImage nv12;
      auto t0 = Clock::now();
      // Same drain cadence as Pass 2 (feed, drain all ready frames, flush the
      // reorder tail) so both passes see the SAME frames — otherwise the serial
      // baseline silently drops the tail and the speedup comparison is void.
      auto handleFrame = [&](bcdl::VpImage& f) {
        cv::Mat bgr = nv12ToBgr(f);
        if (!bgr.isContinuous()) bgr = bgr.clone();
        auto dets = pipe.process(bgr.ptr<uint8_t>(0), bgr.cols, bgr.rows);
        serial_dets += static_cast<long>(dets.size());
        ++serial_frames;
      };
      for (const auto& [b, e] : aus) {
        if (max_frames && serial_frames >= max_frames) break;
        while (!dec.feed(stream.data() + b, e - b)) {
          if (!dec.receive(nv12, 5)) break;
          handleFrame(nv12);
        }
        while (dec.receive(nv12, 0)) {
          handleFrame(nv12);
          if (max_frames && serial_frames >= max_frames) break;
        }
      }
      while (!max_frames || serial_frames < max_frames) {
        if (!dec.flush(nv12)) break;
        handleFrame(nv12);
      }
      serial_ms = msSince(t0);
    }

    // ---- Pass 2: 3-stage async (decode thread || preproc || infer+nms). ----
    long async_dets = 0;
    int async_recv = 0, async_sub = 0;
    double async_ms = 0, a_dec_ms = 0, a_cvt_ms = 0;
    bcdl::StageProfile a_sp;
    {
      bcdl::VideoDecoder dec({vtype});
      bcdl::AsyncDetectionPipeline pipe(engine, cfg, depth);
      // A recycling buffer pool: free_chan holds empty NV12 buffers, filled_chan
      // holds decoded ones. The producer pulls a free buffer, decodes into it
      // (VideoDecoder::receive REUSES the buffer, so after warm-up there is NO
      // per-frame hbUCP alloc/free on the decode thread), and forwards it; the
      // consumer uses it and returns it to free_chan. The same fixed set of
      // buffers cycles for the whole run, so steady state does no per-frame
      // hbUCP alloc/free on the decode thread.
      const int poolN = depth + 4;
      FrameChannel free_chan(poolN + 1), filled_chan(poolN + 1);
      for (int i = 0; i < poolN; ++i) free_chan.push(bcdl::VpImage{});

      auto t0 = Clock::now();
      std::thread producer([&] {
        int made = 0;
        bool stop = false;
        // Decode one ready frame into a recycled buffer and forward it.
        // Returns: 1 = forwarded, 0 = no frame ready now, -1 = stop.
        auto drainOne = [&]() -> int {
          if (max_frames && made >= max_frames) return -1;
          bcdl::VpImage buf;
          if (!free_chan.pop(buf)) return -1;                 // shutting down
          if (!dec.receive(buf, 0)) { free_chan.push(std::move(buf)); return 0; }
          if (!filled_chan.push(std::move(buf))) return -1;
          ++made;
          return 1;
        };
        for (const auto& [b, e] : aus) {
          if (max_frames && made >= max_frames) break;
          auto d0 = Clock::now();
          while (!dec.feed(stream.data() + b, e - b)) {       // input full: drain + retry
            int r = drainOne();
            if (r < 0) { stop = true; break; }
            if (r == 0) break;
          }
          if (!stop) { int r; while ((r = drainOne()) == 1) {} if (r < 0) stop = true; }
          a_dec_ms += msSince(d0);
          if (stop) break;
        }
        // Flush the reorder tail at EOS, into recycled buffers.
        while (!stop) {
          if (max_frames && made >= max_frames) break;
          bcdl::VpImage buf;
          if (!free_chan.pop(buf)) break;
          if (!dec.flush(buf)) { free_chan.push(std::move(buf)); break; }
          if (!filled_chan.push(std::move(buf))) break;
          ++made;
        }
        filled_chan.close();
      });

      // Main: pop a decoded NV12 buffer, convert to BGR (CPU, overlaps the decode
      // thread), feed the pipeline, recycle the buffer, drain results in order.
      std::vector<bcdl::Detection> dets;
      bcdl::VpImage buf;
      while (filled_chan.pop(buf)) {
        auto c0 = Clock::now();
        cv::Mat bgr = nv12ToBgr(buf);
        if (!bgr.isContinuous()) bgr = bgr.clone();
        a_cvt_ms += msSince(c0);
        pipe.submit(bgr.ptr<uint8_t>(0), bgr.cols, bgr.rows);
        ++async_sub;
        free_chan.push(std::move(buf));  // recycle the NV12 buffer for reuse
        if (async_sub > depth && pipe.next(dets)) {
          async_dets += static_cast<long>(dets.size());
          ++async_recv;
        }
      }
      pipe.finish();
      while (pipe.next(dets)) {
        async_dets += static_cast<long>(dets.size());
        ++async_recv;
      }
      async_ms = msSince(t0);
      free_chan.close();  // unblock the producer if it is waiting for a buffer
      producer.join();
      // Full drain above means both workers have exited their loops and their
      // single-writer timing fields are published through the channel mutexes.
      a_sp = pipe.profile();
    }

    std::printf(
        "serial : %d frames  %.1f ms  %.2f ms/f  %.1f FPS  (dets=%ld)\n",
        serial_frames, serial_ms, serial_frames ? serial_ms / serial_frames : 0.0,
        serial_ms > 0 ? serial_frames * 1000.0 / serial_ms : 0.0, serial_dets);
    std::printf(
        "async  : %d frames  %.1f ms  %.2f ms/f  %.1f FPS  (dets=%ld, depth=%d)\n",
        async_recv, async_ms, async_recv ? async_ms / async_recv : 0.0,
        async_ms > 0 ? async_recv * 1000.0 / async_ms : 0.0, async_dets, depth);
    if (serial_ms > 0 && async_ms > 0 && serial_frames == async_recv)
      std::printf("speedup: %.2fx  (same %d frames, same stream)\n",
                  serial_ms / async_ms, serial_frames);

    // Per-stage SERVICE time (each stage on its own thread/unit). Because the
    // stages overlap, they sum to MORE than the wall time; throughput is bounded
    // by the SLOWEST stage (marked <== bottleneck), not their sum.
    const int af = async_recv ? async_recv : 1;
    struct Row { const char* name; double per_f; const char* unit; };
    Row rows[] = {
        {"decode",   a_dec_ms / af,        "VPU  hardware decode"},
        {"nv12->bgr", a_cvt_ms / af,       "CPU  cvtColor (main)"},
        {"preproc",  a_sp.preprocPerFrame(), "CPU  letterbox BGR->NV12"},
        {"infer",    a_sp.inferPerFrame(),   "BPU  feed + submit/wait"},
        {"postproc", a_sp.postprocPerFrame(),"CPU  decode + NMS"},
    };
    double slowest = 0, sum = 0;
    for (const auto& r : rows) { slowest = r.per_f > slowest ? r.per_f : slowest; sum += r.per_f; }
    std::printf("\n=== async per-stage service time (%d frames, stages OVERLAP) ===\n", async_recv);
    for (const auto& r : rows)
      std::printf("  %-10s %7.2f ms/f   (%s)%s\n", r.name, r.per_f, r.unit,
                  r.per_f == slowest ? "  <== bottleneck" : "");
    std::printf("  serial sum = %.2f ms/f  ->  overlapped wall = %.2f ms/f  "
                "(overlap efficiency %.2fx, ceiling ~%.0f FPS = 1/bottleneck)\n",
                sum, async_recv ? async_ms / async_recv : 0.0,
                (async_ms > 0 && async_recv) ? sum / (async_ms / async_recv) : 0.0,
                slowest > 0 ? 1000.0 / slowest : 0.0);
    std::printf("OK: 3-stage overlapped video detection ran.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

// End-to-end demo: video file -> VPU hardware decode -> DetectionPipeline.
//
// This is the decode half of the future camera path (a video file standing in
// for live VIN/MIPI capture): each frame is hardware-decoded on the VPU, handed
// to the BPU object detector via DetectionPipeline, and (optionally) written out
// as an annotated JPEG.
//
//   ./video_det_demo <det.hbm> <in.h264|in.h265> [out_dir] [max_frames]
//
// Pipeline per frame:
//   Annex-B access unit --VPU--> NV12 VpImage --cv::cvtColor--> BGR
//     --DetectionPipeline.process--> boxes (original-image px) --draw--> JPEG
//
// The NV12->BGR->(letterbox NV12) hop is a small redundancy we accept to reuse
// the validated BGR-input DetectionPipeline unchanged; a future NV12-direct
// pipeline entry would skip it.

#include <sys/stat.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bcdl/backend/engine.h"
#include "bcdl/media/video_codec.h"
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

// Split an Annex-B byte stream into access units (one coded picture each).
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

// Copy a decoded NV12 VpImage into a tight [h*3/2, w] CV_8UC1 Mat (honouring the
// device strides), then convert to BGR.
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <det.hbm> <in.h264|in.h265> [out_dir] [max_frames]\n", argv[0]);
    return 1;
  }
  const char* hbm = argv[1];
  const char* in_path = argv[2];
  const char* out_dir = (argc > 3 && argv[3][0]) ? argv[3] : nullptr;  // "" => no output
  const int max_frames = argc > 4 ? std::atoi(argv[4]) : 0;

  std::string ext;
  if (const char* dot = std::strrchr(in_path, '.')) ext = dot;
  const bool h265 = (ext == ".h265" || ext == ".hevc" || ext == ".265");

  try {
    auto stream = readFile(in_path);
    auto aus = accessUnits(stream, h265);
    if (aus.empty()) {
      std::fprintf(stderr, "no NAL start codes — not an Annex-B stream?\n");
      return 2;
    }
    if (out_dir) ::mkdir(out_dir, 0755);

    bcdl::Engine engine(hbm);
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;       // COCO
    cfg.detect.conf_thresh = 0.25f;
    cfg.detect.iou_thresh = 0.45f;
    bcdl::DetectionPipeline pipe(engine, cfg);

    bcdl::VideoDecConfig dcfg;
    dcfg.type = h265 ? HB_VP_VIDEO_TYPE_H265 : HB_VP_VIDEO_TYPE_H264;
    bcdl::VideoDecoder dec(dcfg);

    std::printf("model=%s  inputs=%d outputs=%d  | video=%s (%s, %zu AUs)\n",
                engine.modelName().c_str(), engine.numInputs(), engine.numOutputs(),
                in_path, h265 ? "H.265" : "H.264", aus.size());

    int frames = 0;
    long total_dets = 0;
    double dec_ms = 0, det_ms = 0;
    bcdl::VpImage nv12;
    for (const auto& [b, e] : aus) {
      if (max_frames && frames >= max_frames) break;

      auto t0 = std::chrono::steady_clock::now();
      bool got = dec.decode(stream.data() + b, e - b, nv12);
      auto t1 = std::chrono::steady_clock::now();
      if (!got) continue;
      dec_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

      cv::Mat bgr = nv12ToBgr(nv12);
      if (!bgr.isContinuous()) bgr = bgr.clone();

      auto t2 = std::chrono::steady_clock::now();
      auto dets = pipe.process(bgr.ptr<uint8_t>(0), bgr.cols, bgr.rows);
      auto t3 = std::chrono::steady_clock::now();
      det_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

      ++frames;
      total_dets += static_cast<long>(dets.size());
      if (frames <= 3 || frames % 30 == 0)
        std::printf("frame %3d: %dx%d  dets=%zu\n", frames, bgr.cols, bgr.rows, dets.size());

      if (out_dir) {
        for (const auto& d : dets) {
          cv::rectangle(bgr, cv::Point((int)d.x1, (int)d.y1),
                        cv::Point((int)d.x2, (int)d.y2), cv::Scalar(0, 255, 0), 2);
          cv::putText(bgr, "c" + std::to_string(d.class_id),
                      cv::Point((int)d.x1, (int)d.y1 - 4), cv::FONT_HERSHEY_SIMPLEX,
                      0.5, cv::Scalar(0, 255, 0), 1);
        }
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/det_%04d.jpg", out_dir, frames);
        cv::imwrite(path, bgr);
      }
    }

    std::printf(
        "decoded+detected %d frames | decode %.2f ms/f, detect %.2f ms/f "
        "(%.1f FPS end-to-end) | total dets %ld\n",
        frames, frames ? dec_ms / frames : 0.0, frames ? det_ms / frames : 0.0,
        (dec_ms + det_ms) > 0 ? frames * 1000.0 / (dec_ms + det_ms) : 0.0, total_dets);
    if (frames == 0) {
      std::fprintf(stderr, "FAIL: no frames processed\n");
      return 3;
    }
    std::printf("OK: video -> VPU decode -> DetectionPipeline ran.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

// Video hardware-decode demo: VPU H.264/H.265 decode of a REAL elementary
// stream (Annex-B .h264/.h265 file), as the first step of the "live camera ->
// pipeline" path (a video file standing in for live VIN/MIPI capture).
//
// It reads the file, splits the Annex-B byte stream into access units (one
// coded picture each), feeds each AU to the VPU VideoDecoder, counts the
// frames that come back, reports decode FPS, and dumps a couple of decoded
// frames to JPEG (via the JPU) so the pixels can be eyeballed.
//
//   ./video_decode <in.h264|in.h265> [out_dir] [max_frames]
//
// Type is inferred from the extension (.h265/.hevc -> H.265, else H.264).
// Examples on the board:
//   ./video_decode /app/res/assets/1080P_test.h264 /tmp/vdec
//   ./video_decode /app/multimedia_samples/sample_codec/640x480_30fps.h264 /tmp/vdec

#include <sys/stat.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

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

// Offsets of every Annex-B start code (00 00 01 / 00 00 00 01) in the stream.
std::vector<size_t> startCodes(const std::vector<uint8_t>& s) {
  std::vector<size_t> out;
  for (size_t i = 0; i + 3 <= s.size(); ++i) {
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) out.push_back(i);
  }
  return out;
}

// NAL unit type. H.264: byte & 0x1F. H.265: (byte >> 1) & 0x3F.
int nalType(uint8_t b, bool h265) { return h265 ? (b >> 1) & 0x3F : b & 0x1F; }

// A VCL (slice) NAL carries picture data. H.264: types 1..5. H.265: 0..31.
bool isVcl(int t, bool h265) { return h265 ? (t >= 0 && t <= 31) : (t >= 1 && t <= 5); }

// Split the stream into access units (one coded picture each). Heuristic: start
// a new AU when a VCL NAL arrives and the current AU already has one — leading
// parameter sets / SEI / AUD then group with the following slice. Good enough
// for single-slice-per-frame elementary streams (the common case).
std::vector<std::pair<size_t, size_t>> accessUnits(const std::vector<uint8_t>& s, bool h265) {
  std::vector<std::pair<size_t, size_t>> aus;  // [begin, end) into s, start codes included
  auto sc = startCodes(s);
  if (sc.empty()) return aus;
  size_t au_begin = sc[0];
  bool au_has_vcl = false;
  for (size_t k = 0; k < sc.size(); ++k) {
    size_t pos = sc[k];
    size_t hdr = pos + 3;                          // byte after 00 00 01
    if (hdr >= s.size()) break;
    int t = nalType(s[hdr], h265);
    bool vcl = isVcl(t, h265);
    if (vcl && au_has_vcl) {                        // boundary before this NAL
      aus.emplace_back(au_begin, pos);
      au_begin = pos;
      au_has_vcl = false;
    }
    if (vcl) au_has_vcl = true;
  }
  aus.emplace_back(au_begin, s.size());
  return aus;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <in.h264|in.h265> [out_dir] [max_frames]\n", argv[0]);
    return 1;
  }
  const char* in_path = argv[1];
  const char* out_dir = argc > 2 ? argv[2] : nullptr;
  const int max_frames = argc > 3 ? std::atoi(argv[3]) : 0;  // 0 => all

  std::string ext;
  if (const char* dot = std::strrchr(in_path, '.')) ext = dot;
  const bool h265 = (ext == ".h265" || ext == ".hevc" || ext == ".265");

  try {
    auto stream = readFile(in_path);
    auto aus = accessUnits(stream, h265);
    std::printf("input: %s (%zu bytes, %s) -> %zu access units\n", in_path,
                stream.size(), h265 ? "H.265" : "H.264", aus.size());
    if (aus.empty()) {
      std::fprintf(stderr, "no NAL start codes found — not an Annex-B stream?\n");
      return 2;
    }

    if (out_dir) ::mkdir(out_dir, 0755);  // create the dump dir; ignore "exists"

    bcdl::VideoDecConfig dcfg;
    dcfg.type = h265 ? HB_VP_VIDEO_TYPE_H265 : HB_VP_VIDEO_TYPE_H264;
    bcdl::VideoDecoder dec(dcfg);

    int decoded = 0, saved = 0, w0 = 0, h0 = 0;
    bcdl::VpImage frame;

    // Handle one decoded frame (count + optionally dump a few JPEGs).
    auto handleFrame = [&](bcdl::VpImage& f) {
      ++decoded;
      if (decoded == 1) {
        w0 = f.width();
        h0 = f.height();
        std::printf("first decoded frame: %dx%d NV12\n", w0, h0);
      }
      if (out_dir && saved < 3 && (f.width() % 16) == 0 && (f.height() % 8) == 0) {
        bcdl::JpegEncoder jenc(f.width(), f.height(), 90, HB_VP_IMAGE_FORMAT_NV12);
        auto jpg = jenc.encode(f);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/frame_%03d.jpg", out_dir, decoded);
        if (FILE* fp = std::fopen(path, "wb")) {
          std::fwrite(jpg.data(), 1, jpg.size(), fp);
          std::fclose(fp);
          std::printf("  wrote %s (%zu bytes)\n", path, jpg.size());
          ++saved;
        }
      }
    };

    auto t0 = std::chrono::steady_clock::now();
    // Feed each AU, then drain EVERY frame the decoder currently has ready. The
    // VPU decodes on its own threads; if we feed faster than we dequeue, its
    // frame buffers fill and the next hardware decode stalls (INTERRUPT TIMEOUT).
    // So the correct streaming cadence (matching the vendor sample_codec, which
    // runs a dedicated continuous output-drain thread) is: never leave a ready
    // frame sitting in the decoder — drain to empty after every feed.
    for (const auto& [b, e] : aus) {
      if (max_frames && decoded >= max_frames) break;
      // If the input queue is momentarily full, drain ready frames and retry.
      while (!dec.feed(stream.data() + b, e - b)) {
        if (!dec.receive(frame, 5)) break;
        handleFrame(frame);
      }
      while (dec.receive(frame, 0)) handleFrame(frame);  // drain all ready now
    }
    while (dec.flush(frame)) handleFrame(frame);  // drain the reorder tail at EOS
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("decoded %d / %zu frames in %.1f ms (%.1f FPS)\n", decoded, aus.size(),
                ms, decoded > 0 ? decoded * 1000.0 / ms : 0.0);
    std::printf(
        "note: the VPU may hold the last frame(s) in its reorder buffer; there is "
        "no flush API, so a small tail can be missing.\n");
    if (decoded == 0) {
      std::fprintf(stderr, "FAIL: no frames decoded\n");
      return 3;
    }
    std::printf("OK: VPU hardware decode of %s ran.\n", h265 ? "H.265" : "H.264");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

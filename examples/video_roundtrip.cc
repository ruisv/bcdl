// M2b demo: VPU H.264/H.265 encode -> decode roundtrip (no vDSP needed;
// /dev/vpu is accessible).
//
// Encodes a synthetic NV12 sequence, writes the elementary stream, then feeds it
// back through the decoder and checks that the recovered pixels match what went
// in. Both codecs are driven with the DECOUPLED feed -> drain-to-empty -> flush
// cadence: one packet does not necessarily come out per frame fed in (a rate
// controller may buffer), so counting on a 1:1 correspondence is what the old
// synchronous API got wrong.
//
//   ./video_roundtrip [frames width height out.264 h264|h265]
//
// width must be a multiple of 32, height a multiple of 8.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

namespace {

// The source pattern: a diagonal ramp that moves with the frame index, so a
// decoder that repeats or drops a frame shows up as a large error, not a small one.
unsigned char lumaAt(int r, int c, int f) {
  return static_cast<unsigned char>((c + r + f * 8) & 0xFF);
}

// Mean |delta| of a decoded Y plane against the pattern it was generated from.
double lumaMae(const bcdl::VpImage& img, int f, int w, int h) {
  const auto* y = static_cast<const unsigned char*>(img.raw().dataVirAddr);
  const int ys = img.raw().stride;
  double acc = 0.0;
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c)
      acc += std::abs(static_cast<int>(y[r * ys + c]) - static_cast<int>(lumaAt(r, c, f)));
  return acc / (static_cast<double>(w) * h);
}

}  // namespace

int main(int argc, char** argv) {
  const int frames = argc > 1 ? std::atoi(argv[1]) : 12;
  const int w = argc > 2 ? std::atoi(argv[2]) : 640;
  const int h = argc > 3 ? std::atoi(argv[3]) : 384;
  const char* out_path = argc > 4 ? argv[4] : nullptr;
  const bool h265 = argc > 5 && std::strcmp(argv[5], "h265") == 0;
  const hbVPVideoType codec = h265 ? HB_VP_VIDEO_TYPE_H265 : HB_VP_VIDEO_TYPE_H264;
  const char* codec_name = h265 ? "H.265" : "H.264";

  try {
    // Each element is one access unit (one picture's worth of NALs) — which is
    // exactly what the decoder's FRAME_SIZE feeding mode wants back.
    std::vector<std::vector<uint8_t>> packets;
    std::vector<uint8_t> stream;

    // The encoder lives in its own scope and is destroyed before the decoder is
    // built. Two 1080p codec contexts open at once do NOT fit: the encoder holds
    // 5 frame buffers and the decoder wants 8, and the second one to start never
    // gets buffers — it produces no frame, its input queue fills, and feed()
    // starts failing. At 640x384 both fit and the failure hides. Encode-then-
    // decode in one process is a demo shape anyway; a real transcoder should size
    // frame_buf_count deliberately rather than assume both defaults fit.
    {
    bcdl::VideoEncConfig ecfg;
    ecfg.type = codec;
    ecfg.width = w;
    ecfg.height = h;
    ecfg.framerate = 30;
    ecfg.bitrate_kbps = 4000;
    bcdl::VideoEncoder enc(ecfg);

    bcdl::VpImage frame(w, h, HB_VP_IMAGE_FORMAT_NV12);
    auto* y = static_cast<unsigned char*>(frame.data());
    const int ys = frame.raw().stride;
    auto* uv = static_cast<unsigned char*>(frame.raw().uvVirAddr);
    const int us = frame.raw().uvStride;

    auto take = [&](std::vector<uint8_t>&& pkt) {
      stream.insert(stream.end(), pkt.begin(), pkt.end());
      packets.push_back(std::move(pkt));
    };

    for (int f = 0; f < frames; ++f) {
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) y[r * ys + c] = lumaAt(r, c, f);
      for (int r = 0; r < h / 2; ++r)
        for (int c = 0; c < w; ++c) uv[r * us + c] = 128;

      // Same cadence rule as the decoder (see video_decode.cc): never leave a
      // ready packet sitting in the codec, and if its input queue is momentarily
      // full, drain and RETRY rather than give up. Feeding 60 frames as fast as
      // the CPU can generate them outruns the encoder at 1080p, and the codec
      // then has no input buffer to hand back.
      std::vector<uint8_t> pkt;
      bool fed = false;
      for (int retry = 0; retry < 100 && !fed; ++retry) {
        fed = enc.feed(frame, static_cast<uint64_t>(f) * 33333);
        if (!fed && enc.receive(pkt, 5)) take(std::move(pkt));
      }
      if (!fed) {
        std::fprintf(stderr, "encoder rejected frame %d\n", f);
        return 2;
      }
      while (enc.receive(pkt, 0)) take(std::move(pkt));  // drain to empty
    }
    // Tail: the encoder may still hold packets for frames already fed.
    {
      std::vector<uint8_t> pkt;
      while (enc.flush(pkt)) take(std::move(pkt));
    }
    }  // encoder destroyed here, before the decoder is built

    std::printf("encode(%s): %d frames %dx%d -> %zu packets, %zu bytes\n", codec_name,
                frames, w, h, packets.size(), stream.size());
    std::printf("  packet sizes:");
    for (size_t i = 0; i < packets.size() && i < 12; ++i)
      std::printf(" %zu", packets[i].size());
    std::printf("\n");
    if (stream.size() >= 5)
      std::printf("  first NAL: %02X %02X %02X %02X %02X (expect 00 00 00 01 ..)\n",
                  stream[0], stream[1], stream[2], stream[3], stream[4]);

    if (out_path) {
      if (FILE* fp = std::fopen(out_path, "wb")) {
        std::fwrite(stream.data(), 1, stream.size(), fp);
        std::fclose(fp);
        std::printf("wrote %s\n", out_path);
      }
    }

    // --- decode: feed each AU, drain to empty, flush the tail ---------------
    bcdl::VideoDecConfig dcfg;
    dcfg.type = codec;
    bcdl::VideoDecoder dec(dcfg);

    int decoded = 0;
    double worst_mae = 0.0;
    bcdl::VpImage out;
    auto check = [&]() {
      if (out.width() >= w && out.height() >= h && decoded < frames)
        worst_mae = std::max(worst_mae, lumaMae(out, decoded, w, h));
      if (decoded == 0)
        std::printf("  first decoded frame: %dx%d NV12\n", out.width(), out.height());
      ++decoded;
    };

    for (const auto& pkt : packets) {
      if (pkt.empty()) continue;
      bool fed = false;
      for (int retry = 0; retry < 100 && !fed; ++retry) {
        fed = dec.feed(pkt.data(), pkt.size());
        if (!fed && dec.receive(out, 5)) check();
      }
      if (!fed) {
        std::fprintf(stderr, "decoder rejected a packet\n");
        return 2;
      }
      while (dec.receive(out, 0)) check();  // drain to empty
    }
    while (dec.flush(out)) check();         // reorder / latency tail

    std::printf("decode: recovered %d of %d frame(s); worst Y MAE vs source %.2f\n",
                decoded, frames, worst_mae);

    // Lossy codec at 4 Mbps on a smooth ramp: a few grey levels is normal, a
    // wrong/repeated frame is not (the ramp moves 8 levels per frame).
    const bool ok = decoded == frames && worst_mae < 4.0;
    std::printf("%s: VPU %s encode/decode roundtrip.\n", ok ? "OK" : "FAIL", codec_name);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
}

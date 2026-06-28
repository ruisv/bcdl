// M2b demo: VPU H.264 encode/decode (no vDSP needed; /dev/vpu is accessible).
//
// Encodes a short synthetic NV12 sequence to H.264, writes the elementary stream,
// then feeds it back through the decoder and counts recovered frames.
//
//   ./video_roundtrip [frames width height out.h264]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  const int frames = argc > 1 ? std::atoi(argv[1]) : 12;
  const int w = argc > 2 ? std::atoi(argv[2]) : 640;
  const int h = argc > 3 ? std::atoi(argv[3]) : 384;
  const char* out_path = argc > 4 ? argv[4] : nullptr;

  try {
    bcdl::VideoEncConfig ecfg;
    ecfg.type = HB_VP_VIDEO_TYPE_H264;
    ecfg.width = w;
    ecfg.height = h;
    ecfg.framerate = 30;
    ecfg.bitrate_kbps = 4000;
    bcdl::VideoEncoder enc(ecfg);

    std::vector<uint8_t> stream;
    std::vector<std::size_t> frame_sizes;
    bcdl::VpImage frame(w, h, HB_VP_IMAGE_FORMAT_NV12);
    auto* y = static_cast<unsigned char*>(frame.data());
    const int ys = frame.raw().stride;
    auto* uv = static_cast<unsigned char*>(frame.raw().uvVirAddr);
    const int us = frame.raw().uvStride;

    for (int f = 0; f < frames; ++f) {
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
          y[r * ys + c] = static_cast<unsigned char>((c + r + f * 8) & 0xFF);  // moving diagonal
      for (int r = 0; r < h / 2; ++r)
        for (int c = 0; c < w; ++c) uv[r * us + c] = 128;
      frame.cleanCache();

      std::vector<uint8_t> pkt = enc.encode(frame);
      frame_sizes.push_back(pkt.size());
      stream.insert(stream.end(), pkt.begin(), pkt.end());
    }
    std::printf("encode: %d frames %dx%d -> %zu bytes total\n", frames, w, h, stream.size());
    std::printf("  frame sizes:");
    for (size_t i = 0; i < frame_sizes.size() && i < 12; ++i)
      std::printf(" %zu", frame_sizes[i]);
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

    // --- decode: feed per-frame packets, count recovered frames ------------
    bcdl::VideoDecConfig dcfg;
    dcfg.type = HB_VP_VIDEO_TYPE_H264;
    bcdl::VideoDecoder dec(dcfg);

    int decoded = 0;
    std::size_t off = 0;
    bcdl::VpImage out;
    for (size_t i = 0; i < frame_sizes.size(); ++i) {
      if (frame_sizes[i] == 0) continue;
      if (dec.decode(stream.data() + off, frame_sizes[i], out)) {
        ++decoded;
        if (decoded == 1)
          std::printf("  first decoded frame: %dx%d NV12\n", out.width(), out.height());
      }
      off += frame_sizes[i];
    }
    std::printf("decode: recovered %d frame(s) (decoder latency/buffering may hold "
                "the tail without an explicit flush API)\n", decoded);
    std::printf("OK: VPU H.264 encode/decode paths ran.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  return 0;
}

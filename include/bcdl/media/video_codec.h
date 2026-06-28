#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bcdl/core/sys_mem.h"
#include "bcdl/preproc/vp_image.h"
#include "hobot/vp/hb_vp_video_codec.h"

namespace bcdl {

/// Configuration for VideoEncoder. Mirrors the CBR fields the VPU H264/H265
/// encoder needs; everything else is taken from hbVPGetDefaultVideoEncParam.
struct VideoEncConfig {
  hbVPVideoType type = HB_VP_VIDEO_TYPE_H264;
  int width = 0;                                  ///< must be a multiple of 32, [256, 8192]
  int height = 0;                                 ///< must be a multiple of 8,  [128, 4096]
  int bitrate_kbps = 4000;                        ///< CBR target bitrate, [1, 700000]
  int framerate = 30;                             ///< target fps, [1, 240]
  int intra_period = 28;                          ///< I-frame interval, [0, 2047]
  hbVPImageFormat format = HB_VP_IMAGE_FORMAT_NV12;
};

/// Hardware H.264/H.265 encoder running on the VPU (HB_UCP_VPU_CORE_0).
///
/// One context is created up-front for a fixed (type, width, height, format,
/// rate-control) and reused for every frame; create a new encoder if those
/// change.
///
/// Lifetime note on output buffers: the codec allocates the compressed byte
/// buffer internally and frees it when the task is released. encode() therefore
/// copies the bytes into a caller-owned std::vector before releasing its task,
/// so the returned vector is always safe to keep. The first IDR frame's bytes
/// typically carry SPS/PPS (H264) or VPS/SPS/PPS (H265); the full elementary
/// stream is the concatenation of successive encode() results.
///
/// Cache discipline: the source VpImage is CPU-written, so encode() flushes it
/// (cleanCache) before the VPU reads it.
class VideoEncoder {
 public:
  explicit VideoEncoder(const VideoEncConfig& cfg);
  ~VideoEncoder();

  VideoEncoder(const VideoEncoder&) = delete;
  VideoEncoder& operator=(const VideoEncoder&) = delete;

  /// Encode one NV12/YUV420 frame to compressed bytes (copied out of the
  /// codec-internal buffer; the encode task is released before returning).
  /// Returns the bytes produced for THIS frame, which may be empty if the
  /// encoder buffered the frame and emitted nothing yet.
  std::vector<uint8_t> encode(const VpImage& frame);

  hbVPVideoType type() const noexcept { return cfg_.type; }
  int width() const noexcept { return cfg_.width; }
  int height() const noexcept { return cfg_.height; }
  hbVPImageFormat format() const noexcept { return cfg_.format; }

 private:
  hbVPVideoContext ctx_ = nullptr;
  VideoEncConfig cfg_;
};

/// Configuration for VideoDecoder.
struct VideoDecConfig {
  hbVPVideoType type = HB_VP_VIDEO_TYPE_H264;
  hbVPImageFormat format = HB_VP_IMAGE_FORMAT_NV12;
  /// Codec-internal bitstream buffer size. 0 => a sane default (10 MiB, aligned
  /// to 1024) per the header guidance; must be >= the largest fed chunk.
  uint32_t in_buf_size = 0;
};

/// Hardware H.264/H.265 decoder running on the VPU (HB_UCP_VPU_CORE_0).
///
/// Produces an owned NV12 VpImage (the only planar-YUV layout VpImage can back).
///
/// Lifetime note: the codec decodes into an internal hbVPImage that is freed
/// when the task is released. decode() copies the Y/UV planes (honouring the
/// codec's stride) into a fresh, caller-owned VpImage before releasing the task.
///
/// Input reuse: a single device buffer (in_buf_) holds the compressed bytes
/// handed to the VPU. It is grown only when an incoming chunk is larger than its
/// current capacity, so steady-state decoding does no per-call allocation.
///
/// Buffering: like all stream decoders, the VPU may buffer one or more access
/// units before emitting a frame (it needs reference frames). decode() returns
/// true and fills `out` only when a frame is ready; otherwise it returns false
/// and `out` is untouched. Feed successive chunks until a frame appears.
class VideoDecoder {
 public:
  explicit VideoDecoder(const VideoDecConfig& cfg);
  ~VideoDecoder();

  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  /// Feed compressed bytes (one access unit / chunk). Returns true and fills
  /// `out` with an owned NV12 VpImage when a frame is ready, else returns false.
  bool decode(const uint8_t* data, std::size_t size, VpImage& out);
  bool decode(const std::vector<uint8_t>& data, VpImage& out) {
    return decode(data.data(), data.size(), out);
  }

  hbVPVideoType type() const noexcept { return cfg_.type; }
  hbVPImageFormat format() const noexcept { return cfg_.format; }

 private:
  hbVPVideoContext ctx_ = nullptr;
  VideoDecConfig cfg_;
  SysMem in_buf_;  // reused device buffer for the input bytes (grows as needed)
};

}  // namespace bcdl

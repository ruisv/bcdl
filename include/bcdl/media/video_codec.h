#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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

/// Hardware H.264/H.265 decoder running on the VPU, built on the `media_codec`
/// (`hb_mm_mc_*`) streaming API — a DECOUPLED feed/drain model with reorder
/// support. This matters for HEVC (and any stream with B-frame reorder): the
/// display-order frame for an input access unit is generally NOT ready the
/// instant that AU is fed, so the older "submit one AU, immediately get its
/// frame" model timed out. Here inputs are queued and decoded frames are drained
/// independently, in display order, exactly as D-Robotics' sample_codec does.
///
/// Produces an owned NV12 VpImage (the only planar-YUV layout VpImage can back);
/// each returned frame is copied out (honoring the codec strides) into a fresh,
/// caller-owned VpImage and the codec buffer is returned immediately.
///
/// Two interfaces:
///  - decode(data,size,out): convenience — feed one AU then wait briefly for a
///    frame. Fine for low-latency H.264 (~1:1). For reorder streams the trailing
///    frames need flush().
///  - feed()/receive()/flush(): the decoupled path used by
///    AsyncVideoDetectionPipeline — feed AUs, drain ready frames non-blockingly,
///    then flush() the reorder tail at end-of-stream. This is the HEVC-correct
///    path.
class VideoDecoder {
 public:
  explicit VideoDecoder(const VideoDecConfig& cfg);
  ~VideoDecoder();

  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  /// Feed one access unit, then wait up to a short internal timeout for one
  /// decoded frame. Returns true + fills `out` (display order) when a frame is
  /// ready, else false. Trailing reorder-buffered frames need flush().
  bool decode(const uint8_t* data, std::size_t size, VpImage& out);
  bool decode(const std::vector<uint8_t>& data, VpImage& out) {
    return decode(data.data(), data.size(), out);
  }

  /// Queue one compressed access unit for decoding (does not wait for output).
  /// Blocks briefly if the codec's input queue is full. Returns false on error.
  ///
  /// `data` must be exactly ONE access unit (one picture's worth of NALs,
  /// start-code prefixed) — the decoder runs in FRAME_SIZE feeding mode. Feeding
  /// arbitrary byte chunks is not supported.
  bool feed(const uint8_t* data, std::size_t size);

  /// Drain one decoded frame in display order. `timeout_ms == 0` is non-blocking
  /// (returns false immediately if none ready). Returns true + fills `out` on a
  /// frame, false on timeout / no-frame / end-of-stream marker.
  bool receive(VpImage& out, int timeout_ms = 0);

  /// After the last feed(): signal end-of-stream once, then drain the remaining
  /// reorder-buffered frames — call repeatedly until it returns false.
  bool flush(VpImage& out);

  /// Queue an end-of-stream marker WITHOUT draining (for a decoupled feed thread
  /// + receive thread setup, mirroring D-Robotics' sample_codec): the feed side
  /// calls this once after the last feed(); the receive side keeps calling
  /// receive() until frames stop. Idempotent.
  void feedEndOfStream();

  hbVPVideoType type() const noexcept { return cfg_.type; }
  hbVPImageFormat format() const noexcept { return cfg_.format; }

 private:
  VideoDecConfig cfg_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcdl

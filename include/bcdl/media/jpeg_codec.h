#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bcdl/core/sys_mem.h"
#include "bcdl/preproc/vp_image.h"
#include "hobot/vp/hb_vp_jpeg_codec.h"

namespace bcdl {

/// Hardware JPEG encoder running on the JPU (HB_UCP_JPU_CORE_0).
///
/// One context is created up-front for a fixed (width, height, format, quality)
/// and reused for every frame; create a new encoder if those change.
///
/// Lifetime note on output buffers: the codec allocates the encoded byte buffer
/// internally and frees it when the task is released. encode() therefore copies
/// the bytes into a caller-owned std::vector before releasing its task, so the
/// returned vector is always safe to keep.
///
/// Cache discipline: the source VpImage is CPU-written, so encode() flushes it
/// (cleanCache) before the JPU reads it.
class JpegEncoder {
 public:
  /// quality in [1,100] (50 recommended). format must be one of NV12, YUV420,
  /// YUV444 or YUV444_P (NV12 by default). The JPU requires width aligned to 16
  /// and height aligned to 8; the ctor throws bcdl::Error otherwise.
  JpegEncoder(int width, int height, int quality = 50,
              hbVPImageFormat format = HB_VP_IMAGE_FORMAT_NV12);
  ~JpegEncoder();

  JpegEncoder(const JpegEncoder&) = delete;
  JpegEncoder& operator=(const JpegEncoder&) = delete;

  /// Encode one image to a JPEG byte stream. The bytes are copied out of the
  /// codec-internal buffer; the encode task is released before returning.
  std::vector<uint8_t> encode(const VpImage& src);

  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }
  hbVPImageFormat format() const noexcept { return format_; }

 private:
  hbVPJPEGContext ctx_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  hbVPImageFormat format_ = HB_VP_IMAGE_FORMAT_NV12;
};

/// Hardware JPEG decoder running on the JPU (HB_UCP_JPU_CORE_0).
///
/// Currently produces an owned NV12 VpImage (the only planar-YUV layout VpImage
/// can back); construct with a non-NV12 outFormat only once VpImage grows
/// support for it.
///
/// Lifetime note: the codec decodes into an internal hbVPImage that is freed
/// when the task is released. decode() copies the Y/UV planes (honouring the
/// codec's stride) into a fresh, caller-owned VpImage before releasing the task.
///
/// Input reuse: a single device buffer (in_buf_) holds the JPEG bytes handed to
/// the JPU. It is grown only when an incoming stream is larger than its current
/// capacity, so steady-state decoding does no per-call allocation.
class JpegDecoder {
 public:
  explicit JpegDecoder(hbVPImageFormat outFormat = HB_VP_IMAGE_FORMAT_NV12);
  ~JpegDecoder();

  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  /// Decode a JPEG byte stream into an owned VpImage (NV12 by default).
  VpImage decode(const uint8_t* data, std::size_t size);
  VpImage decode(const std::vector<uint8_t>& data) {
    return decode(data.data(), data.size());
  }

  hbVPImageFormat outFormat() const noexcept { return out_format_; }

 private:
  hbVPJPEGContext ctx_ = nullptr;
  hbVPImageFormat out_format_ = HB_VP_IMAGE_FORMAT_NV12;
  SysMem in_buf_;  // reused device buffer for the input JPEG bytes (grows as needed)
};

}  // namespace bcdl

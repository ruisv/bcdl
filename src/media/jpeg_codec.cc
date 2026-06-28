#include "bcdl/media/jpeg_codec.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "bcdl/core/status.h"
#include "bcdl/core/task.h"
#include "hobot/hb_ucp.h"

namespace bcdl {

namespace {

bool isSupportedJpegFormat(hbVPImageFormat f) {
  switch (f) {
    case HB_VP_IMAGE_FORMAT_NV12:
    case HB_VP_IMAGE_FORMAT_YUV420:
    case HB_VP_IMAGE_FORMAT_YUV444:
    case HB_VP_IMAGE_FORMAT_YUV444_P:
      return true;
    default:
      return false;
  }
}

/// Scan a JPEG for its first Start-Of-Frame marker and read the luma (first)
/// component's horizontal/vertical sampling factors. Returns false if no SOF is
/// found before the data ends (malformed / truncated header).
///
/// WHY: the JPU JPEG decoder targets 4:2:0 (luma sampling 2x2, matching its NV12
/// output). On non-4:2:0 streams (4:2:2 / 4:4:4) it reads/writes out of bounds in
/// firmware and HARD-CRASHES (segfault) at larger resolutions — uncatchable from
/// C++. We parse the subsampling up front and reject non-4:2:0 with a clean
/// bcdl::Error so callers can fall back to a software decoder.
bool jpegLumaSampling(const uint8_t* d, std::size_t n, int& hs, int& vs) {
  std::size_t i = 2;  // skip SOI (FFD8)
  while (i + 1 < n) {
    if (d[i] != 0xFF) { ++i; continue; }
    const uint8_t m = d[i + 1];
    // Standalone markers (no length): SOI/EOI, RSTn, TEM, and fill bytes.
    if (m == 0xD8 || m == 0xD9 || (m >= 0xD0 && m <= 0xD7) || m == 0x01 || m == 0xFF) {
      i += 2;
      continue;
    }
    if (i + 4 > n) break;
    const int len = (d[i + 2] << 8) | d[i + 3];
    // SOF0 (baseline), SOF1 (extended), SOF2 (progressive): the subsampling lives
    // in the frame header; SOF3..SOFF (arithmetic / lossless) are not expected.
    if (m == 0xC0 || m == 0xC1 || m == 0xC2) {
      const std::size_t p = i + 4 + 1 + 2 + 2;  // -> number-of-components byte
      if (p + 3 >= n) return false;
      const int nc = d[p];
      if (nc < 1) return false;
      const uint8_t samp = d[p + 1 + 1];        // first comp: id(1), sampling(1)
      hs = samp >> 4;
      vs = samp & 0x0F;
      return true;
    }
    i += 2 + static_cast<std::size_t>(len);
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// JpegEncoder
// ---------------------------------------------------------------------------

JpegEncoder::JpegEncoder(int width, int height, int quality, hbVPImageFormat format)
    : width_(width), height_(height), format_(format) {
  if (width < 32 || width > 8192 || height < 32 || height > 8192) {
    throw Error(-1, "BCDL: JpegEncoder width/height must be in [32, 8192]");
  }
  if ((width & 15) || (height & 7)) {
    throw Error(-1, "BCDL: JpegEncoder requires width aligned to 16 and height to 8 (got " +
                        std::to_string(width) + "x" + std::to_string(height) + ")");
  }
  if (quality < 1 || quality > 100) {
    throw Error(-1, "BCDL: JpegEncoder quality must be in [1, 100]");
  }
  if (!isSupportedJpegFormat(format)) {
    throw Error(-1, "BCDL: JpegEncoder format must be NV12/YUV420/YUV444/YUV444_P");
  }

  hbVPJPEGEncParam param;
  HB_VP_INITIALIZE_JPEG_ENC_PARAM(&param);
  param.imageFormat = static_cast<uint8_t>(format);
  param.width = width;
  param.height = height;
  param.qualityFactor = static_cast<uint32_t>(quality);
  param.backend = HB_UCP_JPU_CORE_0;

  BCDL_CHECK(hbVPCreateJPEGEncContext(&ctx_, &param));
}

JpegEncoder::~JpegEncoder() {
  if (ctx_) {
    // best-effort release; never throw from a destructor
    hbVPReleaseJPEGEncContext(ctx_);
    ctx_ = nullptr;
  }
}

std::vector<uint8_t> JpegEncoder::encode(const VpImage& src) {
  if (src.format() != format_) {
    throw Error(-1, "BCDL: JpegEncoder.encode source format != encoder format");
  }
  if (src.width() != width_ || src.height() != height_) {
    throw Error(-1, "BCDL: JpegEncoder.encode source size != encoder size");
  }

  // The CPU wrote the source pixels; flush so the JPU reads fresh data.
  src.cleanCache();

  // Task owns the codec output buffer; it is freed when the task is released, so
  // we must copy the encoded bytes out before this Task goes out of scope.
  Task task;
  BCDL_CHECK(hbVPJPEGEncode(task.addr(), &src.raw(), ctx_));
  task.submit();
  task.wait();

  hbVPArray out{};
  BCDL_CHECK(hbVPGetJPEGEncOutputBuffer(&out, task.get()));

  std::vector<uint8_t> jpeg;
  if (out.virAddr && out.size > 0) {
    const auto* bytes = static_cast<const uint8_t*>(out.virAddr);
    jpeg.assign(bytes, bytes + out.size);
  }
  return jpeg;
}

// ---------------------------------------------------------------------------
// JpegDecoder
// ---------------------------------------------------------------------------

JpegDecoder::JpegDecoder(hbVPImageFormat outFormat) : out_format_(outFormat) {
  if (!isSupportedJpegFormat(outFormat)) {
    throw Error(-1, "BCDL: JpegDecoder outFormat must be NV12/YUV420/YUV444/YUV444_P");
  }
  if (outFormat != HB_VP_IMAGE_FORMAT_NV12) {
    // VpImage can only back NV12 among the planar-YUV formats today.
    throw Error(-1, "BCDL: JpegDecoder currently only supports NV12 output");
  }

  hbVPJPEGDecParam param;
  HB_VP_INITIALIZE_JPEG_DEC_PARAM(&param);
  param.imageFormat = static_cast<uint8_t>(outFormat);
  param.backend = HB_UCP_JPU_CORE_0;

  BCDL_CHECK(hbVPCreateJPEGDecContext(&ctx_, &param));
}

JpegDecoder::~JpegDecoder() {
  if (ctx_) {
    // best-effort release; never throw from a destructor
    hbVPReleaseJPEGDecContext(ctx_);
    ctx_ = nullptr;
  }
}

VpImage JpegDecoder::decode(const uint8_t* data, std::size_t size) {
  if (data == nullptr || size == 0) {
    throw Error(-1, "BCDL: JpegDecoder.decode given empty input");
  }

  // Reject non-4:2:0 JPEGs BEFORE the JPU touches them: the hardware decoder
  // hard-crashes (segfault in firmware) on 4:2:2 / 4:4:4 streams at larger sizes.
  // A clean error lets the caller fall back to a software decoder.
  int hs = 0, vs = 0;
  if (jpegLumaSampling(data, size, hs, vs) && !(hs == 2 && vs == 2)) {
    throw Error(-1, "BCDL: JpegDecoder hardware path supports only 4:2:0 JPEGs "
                    "(luma sampling 2x2); got " + std::to_string(hs) + "x" +
                    std::to_string(vs) + " — decode this image in software (cv2) instead");
  }

  // Stage the JPEG bytes in a device-readable buffer. Grow it only when the
  // incoming stream exceeds the current capacity.
  if (!in_buf_.valid() || in_buf_.size() < size) {
    in_buf_ = SysMem(static_cast<uint64_t>(size), /*cached=*/true);
  }
  std::memcpy(in_buf_.data(), data, size);
  in_buf_.cleanCache();  // CPU wrote the JPEG bytes; flush so the JPU reads them.

  hbVPArray src{};
  src.phyAddr = in_buf_.phyAddr();
  src.virAddr = in_buf_.data();
  src.memSize = static_cast<uint32_t>(in_buf_.size());  // buffer capacity
  src.size = static_cast<uint32_t>(size);               // valid JPEG byte count

  // Task owns the decoded image buffer; copy the planes out before release.
  Task task;
  BCDL_CHECK(hbVPJPEGDecode(task.addr(), &src, ctx_));
  task.submit();
  task.wait();

  hbVPImage out{};
  BCDL_CHECK(hbVPGetJPEGDecOutputBuffer(&out, task.get()));

  if (out.width <= 0 || out.height <= 0 || out.dataVirAddr == nullptr) {
    throw Error(-1, "BCDL: JpegDecoder.decode produced an invalid image");
  }

  // Build an owned NV12 image and copy plane rows, honouring both strides.
  VpImage img(out.width, out.height, HB_VP_IMAGE_FORMAT_NV12);
  const hbVPImage& dst = img.raw();

  // Y plane: `height` rows of `width` visible bytes.
  {
    const auto* sp = static_cast<const uint8_t*>(out.dataVirAddr);
    auto* dp = static_cast<uint8_t*>(dst.dataVirAddr);
    const int copy = std::min({out.width, out.stride, dst.stride});
    for (int y = 0; y < out.height; ++y) {
      std::memcpy(dp + static_cast<std::size_t>(y) * dst.stride,
                  sp + static_cast<std::size_t>(y) * out.stride,
                  static_cast<std::size_t>(copy));
    }
  }

  // Interleaved UV plane: height/2 rows; a row spans width bytes (width/2 UV pairs).
  if (out.uvVirAddr != nullptr && dst.uvVirAddr != nullptr) {
    const auto* sp = static_cast<const uint8_t*>(out.uvVirAddr);
    auto* dp = static_cast<uint8_t*>(dst.uvVirAddr);
    const int copy = std::min({out.width, out.uvStride, dst.uvStride});
    const int rows = out.height / 2;
    for (int y = 0; y < rows; ++y) {
      std::memcpy(dp + static_cast<std::size_t>(y) * dst.uvStride,
                  sp + static_cast<std::size_t>(y) * out.uvStride,
                  static_cast<std::size_t>(copy));
    }
  }

  return img;
}

}  // namespace bcdl

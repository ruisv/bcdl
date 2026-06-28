#include "bcdl/media/video_codec.h"

#include <algorithm>
#include <cstring>

#include "bcdl/core/status.h"
#include "bcdl/core/task.h"
#include "hobot/hb_ucp.h"

namespace bcdl {

namespace {

// The VPU H264/H265 codec accepts NV12 and YUV420 only.
bool isSupportedVideoFormat(hbVPImageFormat f) {
  switch (f) {
    case HB_VP_IMAGE_FORMAT_NV12:
    case HB_VP_IMAGE_FORMAT_YUV420:
      return true;
    default:
      return false;
  }
}

// CBR rate-control mode for the given video type.
hbVPVideoRcMode cbrModeFor(hbVPVideoType type) {
  return type == HB_VP_VIDEO_TYPE_H265 ? HB_VP_VIDEO_RC_MODE_H265_CBR
                                       : HB_VP_VIDEO_RC_MODE_H264_CBR;
}

constexpr uint32_t kDecInBufAlign = 1024;
constexpr uint32_t kDecInBufDefault = 10u * 1024u * 1024u;  // 10 MiB, per header

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1u) / a * a; }

}  // namespace

// ---------------------------------------------------------------------------
// VideoEncoder
// ---------------------------------------------------------------------------

VideoEncoder::VideoEncoder(const VideoEncConfig& cfg) : cfg_(cfg) {
  if (!isSupportedVideoFormat(cfg.format)) {
    throw Error(-1, "BCDL: VideoEncoder format must be NV12 or YUV420");
  }
  // Dimensions are fixed for the lifetime of the BPU/VPU context, so we REQUIRE
  // pre-aligned dims rather than silently aligning up (which would change the
  // pixel geometry the caller intends).
  if (cfg.width < 256 || cfg.width > 8192 || (cfg.width % 32) != 0) {
    throw Error(-1, "BCDL: VideoEncoder width must be a multiple of 32 in [256, 8192]");
  }
  if (cfg.height < 128 || cfg.height > 4096 || (cfg.height % 8) != 0) {
    throw Error(-1, "BCDL: VideoEncoder height must be a multiple of 8 in [128, 4096]");
  }
  if (cfg.bitrate_kbps < 1 || cfg.bitrate_kbps > 700000) {
    throw Error(-1, "BCDL: VideoEncoder bitrate_kbps must be in [1, 700000]");
  }
  if (cfg.framerate < 1 || cfg.framerate > 240) {
    throw Error(-1, "BCDL: VideoEncoder framerate must be in [1, 240]");
  }
  if (cfg.intra_period > 2047) {
    throw Error(-1, "BCDL: VideoEncoder intra_period must be in [0, 2047]");
  }

  hbVPVideoEncParam param{};
  // The header requires videoType to be set BEFORE querying defaults.
  param.videoType = cfg.type;
  BCDL_CHECK(hbVPGetDefaultVideoEncParam(&param));

  param.pixelFormat = static_cast<uint8_t>(cfg.format);
  param.width = cfg.width;
  param.height = cfg.height;
  param.outBufCount = 5;  // header-recommended default
  param.backend = HB_UCP_VPU_CORE_0;

  // CBR rate control. hbVPVideoH265Cbr is a typedef of hbVPVideoH264Cbr, so the
  // h264Cbr union member has the right layout for both; selecting the H265_CBR
  // mode is what routes it to the H.265 path.
  param.rcParam.mode = cbrModeFor(cfg.type);
  param.rcParam.h264Cbr.intraPeriod = static_cast<uint32_t>(cfg.intra_period);
  param.rcParam.h264Cbr.bitRate = static_cast<uint32_t>(cfg.bitrate_kbps);
  param.rcParam.h264Cbr.frameRate = static_cast<uint32_t>(cfg.framerate);
  param.rcParam.h264Cbr.initialRcQp = 63;  // > 51 => let the VPU pick the initial QP

  // Standard streaming GOP: IDR refresh, consecutive-P preset.
  param.gopParam.decodingRefreshType = 2;
  param.gopParam.gopPresetIdx = 2;

  BCDL_CHECK(hbVPCreateVideoEncContext(&ctx_, &param));
}

VideoEncoder::~VideoEncoder() {
  if (ctx_) {
    // best-effort release; never throw from a destructor
    hbVPReleaseVideoEncContext(ctx_);
    ctx_ = nullptr;
  }
}

std::vector<uint8_t> VideoEncoder::encode(const VpImage& frame) {
  if (frame.format() != cfg_.format) {
    throw Error(-1, "BCDL: VideoEncoder.encode source format != encoder format");
  }
  if (frame.width() != cfg_.width || frame.height() != cfg_.height) {
    throw Error(-1, "BCDL: VideoEncoder.encode source size != encoder size");
  }

  // The CPU wrote the source pixels; flush so the VPU reads fresh data.
  frame.cleanCache();

  // Task owns the codec output buffer; it is freed when the task is released, so
  // we must copy the compressed bytes out before this Task goes out of scope.
  Task task;
  BCDL_CHECK(hbVPVideoEncode(task.addr(), &frame.raw(), ctx_));
  task.submit();
  task.wait();

  hbVPArray out{};
  BCDL_CHECK(hbVPGetVideoEncOutputBuffer(&out, task.get()));

  std::vector<uint8_t> bytes;
  if (out.virAddr && out.size > 0) {
    const auto* p = static_cast<const uint8_t*>(out.virAddr);
    bytes.assign(p, p + out.size);
  }
  return bytes;  // empty => frame buffered, nothing emitted this call
}

// ---------------------------------------------------------------------------
// VideoDecoder
// ---------------------------------------------------------------------------

VideoDecoder::VideoDecoder(const VideoDecConfig& cfg) : cfg_(cfg) {
  if (!isSupportedVideoFormat(cfg.format)) {
    throw Error(-1, "BCDL: VideoDecoder format must be NV12 or YUV420");
  }
  if (cfg.format != HB_VP_IMAGE_FORMAT_NV12) {
    // VpImage can only back NV12 among the planar-YUV formats today.
    throw Error(-1, "BCDL: VideoDecoder currently only supports NV12 output");
  }

  uint32_t in_buf_size = cfg.in_buf_size != 0
                             ? alignUp(cfg.in_buf_size, kDecInBufAlign)
                             : kDecInBufDefault;

  hbVPVideoDecParam param{};
  // The header requires videoType to be set BEFORE querying defaults.
  param.videoType = cfg.type;
  BCDL_CHECK(hbVPGetDefaultVideoDecParam(&param));

  param.pixelFormat = static_cast<uint8_t>(cfg.format);
  param.inBufSize = in_buf_size;
  param.outBufCount = 5;  // header-recommended default
  param.backend = HB_UCP_VPU_CORE_0;

  cfg_.in_buf_size = in_buf_size;

  BCDL_CHECK(hbVPCreateVideoDecContext(&ctx_, &param));
}

VideoDecoder::~VideoDecoder() {
  if (ctx_) {
    // best-effort release; never throw from a destructor
    hbVPReleaseVideoDecContext(ctx_);
    ctx_ = nullptr;
  }
}

bool VideoDecoder::decode(const uint8_t* data, std::size_t size, VpImage& out) {
  if (data == nullptr || size == 0) {
    throw Error(-1, "BCDL: VideoDecoder.decode given empty input");
  }

  // Stage the compressed bytes in a device-readable buffer. Grow it only when
  // the incoming chunk exceeds the current capacity.
  if (!in_buf_.valid() || in_buf_.size() < size) {
    in_buf_ = SysMem(static_cast<uint64_t>(size), /*cached=*/true);
  }
  std::memcpy(in_buf_.data(), data, size);
  in_buf_.cleanCache();  // CPU wrote the bytes; flush so the VPU reads them.

  hbVPArray src{};
  src.phyAddr = in_buf_.phyAddr();
  src.virAddr = in_buf_.data();
  src.memSize = static_cast<uint32_t>(in_buf_.size());  // buffer capacity
  src.size = static_cast<uint32_t>(size);               // valid byte count

  // Task owns the decoded image buffer; copy the planes out before release.
  Task task;
  BCDL_CHECK(hbVPVideoDecode(task.addr(), &src, ctx_));
  task.submit();
  task.wait();

  // The decoder may have buffered this access unit without producing a frame
  // (it needs reference frames before the first output). We therefore do NOT
  // treat a non-zero return or an empty image from the get-output call as a
  // fatal error: both are interpreted as "no frame ready yet" and reported as
  // a benign `false`. NOTE: this behaviour must be confirmed on-board — if the
  // SDK instead signals "no frame" through a distinct, documented code we
  // should special-case only that code here.
  hbVPImage img{};
  int32_t ret = hbVPGetVideoDecOutputBuffer(&img, task.get());
  if (ret != 0 || img.width <= 0 || img.height <= 0 || img.dataVirAddr == nullptr) {
    return false;
  }

  // Build an owned NV12 image and copy plane rows, honouring both strides.
  VpImage frame(img.width, img.height, HB_VP_IMAGE_FORMAT_NV12);
  const hbVPImage& dst = frame.raw();

  // Y plane: `height` rows of `width` visible bytes.
  {
    const auto* sp = static_cast<const uint8_t*>(img.dataVirAddr);
    auto* dp = static_cast<uint8_t*>(dst.dataVirAddr);
    const int copy = std::min({img.width, img.stride, dst.stride});
    for (int y = 0; y < img.height; ++y) {
      std::memcpy(dp + static_cast<std::size_t>(y) * dst.stride,
                  sp + static_cast<std::size_t>(y) * img.stride,
                  static_cast<std::size_t>(copy));
    }
  }

  // Interleaved UV plane: height/2 rows; a row spans width bytes (width/2 pairs).
  if (img.uvVirAddr != nullptr && dst.uvVirAddr != nullptr) {
    const auto* sp = static_cast<const uint8_t*>(img.uvVirAddr);
    auto* dp = static_cast<uint8_t*>(dst.uvVirAddr);
    const int copy = std::min({img.width, img.uvStride, dst.uvStride});
    const int rows = img.height / 2;
    for (int y = 0; y < rows; ++y) {
      std::memcpy(dp + static_cast<std::size_t>(y) * dst.uvStride,
                  sp + static_cast<std::size_t>(y) * img.uvStride,
                  static_cast<std::size_t>(copy));
    }
  }

  out = std::move(frame);
  return true;
}

}  // namespace bcdl

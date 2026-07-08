#include "bcdl/media/video_codec.h"

#include <algorithm>
#include <cstring>

#include "bcdl/core/status.h"
#include "bcdl/core/task.h"
#include "hb_media_codec.h"  // media_codec (hb_mm_mc_*) streaming decode API
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

namespace {
constexpr int kDecInTimeoutMs = 1000;  // dequeue/queue an input buffer
constexpr int kDecodeWaitMs = 40;      // decode(): wait for one ready frame
constexpr int kFlushWaitMs = 300;      // flush(): drain reorder tail
}  // namespace

// media_codec-backed decoupled decoder. hb_mm_mc_* separates input queueing from
// output dequeueing and honors reorder, so display-order frames are drained
// independently of which AU produced them — the HEVC-correct model.
struct VideoDecoder::Impl {
  media_codec_context_t ctx{};
  bool started = false;
  bool eos_fed = false;

  explicit Impl(const VideoDecConfig& cfg) {
    const media_codec_id_t id =
        (cfg.type == HB_VP_VIDEO_TYPE_H265) ? MEDIA_CODEC_ID_H265 : MEDIA_CODEC_ID_H264;
    // Populate a valid default context first (sets internal/version fields), THEN
    // override — matching the hbVPGetDefault* pattern. Without this the params we
    // set are ignored and the decoder runs on its own defaults (e.g. base-layer-
    // only temporal decoding, which drops 3/4 of a hierarchical HEVC stream).
    std::memset(&ctx, 0, sizeof(ctx));
    BCDL_CHECK(hb_mm_mc_get_default_context(id, /*encoder=*/false, &ctx));
    ctx.encoder = false;
    ctx.codec_id = id;
    mc_video_codec_dec_params_t* p = &ctx.video_dec_params;
    // STREAM_SIZE: feed arbitrary compressed bytes and let the decoder find frame
    // boundaries itself. More robust than FRAME_SIZE for streams whose pictures
    // span multiple slice NALs (some HEVC), and it means callers need not split
    // access units at all.
    p->feed_mode = MC_FEEDING_MODE_STREAM_SIZE;
    p->pix_fmt = MC_PIXEL_FORMAT_NV12;
    const uint32_t bs = cfg.in_buf_size != 0 ? cfg.in_buf_size : kDecInBufDefault;
    p->bitstream_buf_size = alignUp(bs, kDecInBufAlign);
    p->bitstream_buf_count = 5;
    p->frame_buf_count = 8;   // reorder + in-flight headroom
    // bandwidth_Opt lets the VPU skip WRITING non-reference / non-display
    // pictures to the frame buffer — which drops frames we want. Keep it OFF so
    // every decoded picture is emitted.
    if (ctx.codec_id == MEDIA_CODEC_ID_H265) {
      p->h265_dec_config.bandwidth_Opt = false;
      p->h265_dec_config.reorder_enable = true;   // HEVC display-order reorder
      p->h265_dec_config.skip_mode = 0;           // normal decode (no frame skip)
      p->h265_dec_config.cra_as_bla = false;
      // Temporal-layer decode: absolute mode, plus1=0 = "constraint off" per the
      // header (decode all sub-layers). NOTE (measured on S100P / libmultimedia
      // v1.2.3): this control has NO effect — a hierarchical-GOP HEVC stream
      // (e.g. a Hikvision cam with SVC / "H.265+"/smart-codec on) still yields
      // only the base temporal layer (~1/4 of frames). Non-hierarchical HEVC and
      // all H.264 decode fully. Disable the camera's hierarchical/SVC mode for
      // full-rate HEVC. Left at the documented value in case a later SDK honors it.
      p->h265_dec_config.dec_temporal_id_mode = 0;
      p->h265_dec_config.target_dec_temporal_id_plus1 = 0;
    } else {
      p->h264_dec_config.bandwidth_Opt = false;
      p->h264_dec_config.reorder_enable = true;
      p->h264_dec_config.skip_mode = 0;
    }
    BCDL_CHECK(hb_mm_mc_initialize(&ctx));
    BCDL_CHECK(hb_mm_mc_configure(&ctx));
    mc_av_codec_startup_params_t su{};
    BCDL_CHECK(hb_mm_mc_start(&ctx, &su));
    started = true;
  }

  ~Impl() {
    if (started) {
      hb_mm_mc_stop(&ctx);      // best-effort; never throw from a destructor
      hb_mm_mc_release(&ctx);
    }
  }

  // Queue one AU (eos=false) or an end-of-stream marker (eos=true).
  bool feedAu(const uint8_t* data, std::size_t size, bool eos) {
    media_codec_buffer_t in{};
    in.type = MC_VIDEO_STREAM_BUFFER;
    if (hb_mm_mc_dequeue_input_buffer(&ctx, &in, kDecInTimeoutMs) != 0) return false;
    if (eos) {
      in.vstream_buf.size = 0;
      in.vstream_buf.stream_end = 1;
    } else {
      if (in.vstream_buf.size < size) {  // dequeued buffer capacity too small
        in.vstream_buf.size = 0;
        hb_mm_mc_queue_input_buffer(&ctx, &in, kDecInTimeoutMs);
        throw Error(-1, "BCDL: VideoDecoder access unit exceeds bitstream buffer size");
      }
      std::memcpy(in.vstream_buf.vir_ptr, data, size);
      in.vstream_buf.size = static_cast<uint32_t>(size);
      in.vstream_buf.stream_end = 0;
    }
    return hb_mm_mc_queue_input_buffer(&ctx, &in, kDecInTimeoutMs) == 0;
  }

  // Dequeue one decoded frame (display order). Returns false on timeout / EOS
  // marker; copies a valid frame into `out` and returns the codec buffer.
  bool recv(VpImage& out, int timeout_ms) {
    media_codec_buffer_t ob{};
    media_codec_output_buffer_info_t info{};
    if (hb_mm_mc_dequeue_output_buffer(&ctx, &ob, &info, timeout_ms) != 0) return false;

    const mc_video_frame_buffer_info_t& vb = ob.vframe_buf;
    const bool valid = vb.size > 0 && vb.width > 0 && vb.height > 0 &&
                       vb.vir_ptr[0] != nullptr;
    if (!valid) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // return the empty/EOS buffer
      return false;
    }

    VpImage frame(vb.width, vb.height, HB_VP_IMAGE_FORMAT_NV12);
    const hbVPImage& dst = frame.raw();
    // Y plane: `height` rows of `width` bytes at the codec's luma stride.
    {
      const auto* sp = vb.vir_ptr[0];
      auto* dp = static_cast<uint8_t*>(dst.dataVirAddr);
      const int copy = std::min({vb.width, vb.stride, dst.stride});
      for (int y = 0; y < vb.height; ++y)
        std::memcpy(dp + static_cast<std::size_t>(y) * dst.stride,
                    sp + static_cast<std::size_t>(y) * vb.stride,
                    static_cast<std::size_t>(copy));
    }
    // Interleaved UV plane: height/2 rows at the same (luma) stride.
    if (vb.vir_ptr[1] != nullptr && dst.uvVirAddr != nullptr) {
      const auto* sp = vb.vir_ptr[1];
      auto* dp = static_cast<uint8_t*>(dst.uvVirAddr);
      const int copy = std::min({vb.width, vb.stride, dst.uvStride});
      for (int y = 0; y < vb.height / 2; ++y)
        std::memcpy(dp + static_cast<std::size_t>(y) * dst.uvStride,
                    sp + static_cast<std::size_t>(y) * vb.stride,
                    static_cast<std::size_t>(copy));
    }
    hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // return the buffer to the codec
    out = std::move(frame);
    return true;
  }
};

VideoDecoder::VideoDecoder(const VideoDecConfig& cfg) : cfg_(cfg) {
  if (!isSupportedVideoFormat(cfg.format)) {
    throw Error(-1, "BCDL: VideoDecoder format must be NV12 or YUV420");
  }
  if (cfg.format != HB_VP_IMAGE_FORMAT_NV12) {
    throw Error(-1, "BCDL: VideoDecoder currently only supports NV12 output");
  }
  impl_ = std::make_unique<Impl>(cfg);
}

VideoDecoder::~VideoDecoder() = default;

bool VideoDecoder::feed(const uint8_t* data, std::size_t size) {
  if (data == nullptr || size == 0) {
    throw Error(-1, "BCDL: VideoDecoder.feed given empty input");
  }
  return impl_->feedAu(data, size, /*eos=*/false);
}

bool VideoDecoder::receive(VpImage& out, int timeout_ms) {
  return impl_->recv(out, timeout_ms);
}

bool VideoDecoder::flush(VpImage& out) {
  feedEndOfStream();
  return impl_->recv(out, kFlushWaitMs);
}

void VideoDecoder::feedEndOfStream() {
  if (!impl_->eos_fed) {
    impl_->feedAu(nullptr, 0, /*eos=*/true);
    impl_->eos_fed = true;
  }
}

bool VideoDecoder::decode(const uint8_t* data, std::size_t size, VpImage& out) {
  if (!feed(data, size)) return false;
  return impl_->recv(out, kDecodeWaitMs);
}

}  // namespace bcdl

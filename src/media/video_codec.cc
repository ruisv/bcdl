#include "bcdl/media/video_codec.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "bcdl/core/status.h"
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

constexpr uint32_t kDecInBufAlign = 1024;
constexpr uint32_t kDecInBufDefault = 10u * 1024u * 1024u;  // 10 MiB, per header

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1u) / a * a; }

using Clock = std::chrono::steady_clock;

constexpr int kStatusPollUs = 200;  // how often to re-read the codec's status
constexpr int kReadyWaitMs = 3;     // dequeue wait once status says a buffer exists

// True once the codec reports at least one output buffer ready, waiting until
// `deadline`.
//
// Why not just block inside the dequeue: a dequeue that finds nothing prints an
// ERROR line — on STDOUT ("Component vdec_render isn't ready!"). A drain-to-empty
// cadence ends every input with exactly such a call, so a clean 1385-frame decode
// emitted 1385 error lines. Polling this status counter is silent. Shared by both
// codecs so the two can't drift apart on a lesson this expensive to relearn.
bool waitForOutput(media_codec_context_t& ctx, Clock::time_point deadline) {
  for (;;) {
    mc_inter_status_t st{};
    if (hb_mm_mc_get_status(&ctx, &st) != 0) return true;  // can't tell: let dequeue decide
    if (st.cur_output_buf_cnt > 0) return true;
    if (Clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(kStatusPollUs));
  }
}

// Give every output buffer the codec still holds back to it, then stop/release.
// Destroying mid-stream is legal, but both hb_mm_mc_stop() and hb_mm_mc_flush()
// block until the codec's internal components have emptied their output port —
// and only the application can empty it. Skip this and the destructor hangs
// forever. Bounded: the codec can only hold what its queued inputs produced.
void drainAndRelease(media_codec_context_t& ctx, int poll_ms) {
  media_codec_buffer_t ob{};
  media_codec_output_buffer_info_t info{};
  for (int empty_polls = 0; empty_polls < 3;) {
    if (waitForOutput(ctx, Clock::now() + std::chrono::milliseconds(poll_ms)) &&
        hb_mm_mc_dequeue_output_buffer(&ctx, &ob, &info, kReadyWaitMs) == 0) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);
      empty_polls = 0;
    } else {
      ++empty_polls;
    }
  }
  hb_mm_mc_stop(&ctx);  // best-effort; never throw from a destructor
  hb_mm_mc_release(&ctx);
}

}  // namespace

// ---------------------------------------------------------------------------
// VideoEncoder
// ---------------------------------------------------------------------------

namespace {
constexpr int kEncInTimeoutMs = 1000;  // dequeue/queue an input buffer
constexpr int kEncodeWaitMs = 40;      // encode(): wait for one ready packet
constexpr int kEncFlushWaitMs = 300;   // flush(): drain the tail
constexpr int kEncDrainPollMs = 20;    // ~Impl: poll for buffers the codec holds
constexpr uint32_t kEncOutBufMin = 64u * 1024u;  // codec-enforced floor
}  // namespace

// media_codec-backed decoupled encoder. Mirrors VideoDecoder::Impl: inputs are
// queued and compressed packets are dequeued independently, so a rate controller
// that buffers a frame (or any GOP with lookahead) can't deadlock the caller.
struct VideoEncoder::Impl {
  media_codec_context_t ctx{};
  bool started = false;
  bool eos_fed = false;
  int width = 0;
  int height = 0;

  explicit Impl(const VideoEncConfig& cfg) : width(cfg.width), height(cfg.height) {
    const media_codec_id_t id =
        (cfg.type == HB_VP_VIDEO_TYPE_H265) ? MEDIA_CODEC_ID_H265 : MEDIA_CODEC_ID_H264;
    // Populate a valid default context first, THEN override — same reason as the
    // decoder: without it the fields we set are layered onto zeroed internals and
    // the codec runs on its own defaults.
    std::memset(&ctx, 0, sizeof(ctx));
    BCDL_CHECK(hb_mm_mc_get_default_context(id, /*encoder=*/true, &ctx));
    ctx.encoder = true;
    ctx.codec_id = id;

    mc_video_codec_enc_params_t* p = &ctx.video_enc_params;
    p->width = cfg.width;
    p->height = cfg.height;
    p->pix_fmt = MC_PIXEL_FORMAT_NV12;
    // Worst case a packet can't exceed one uncompressed frame; the vendor sizes
    // the bitstream buffer at exactly that, 1 KiB-aligned. The codec also
    // enforces a 64 KiB floor, which a small frame (256x128 = 48 KiB) falls
    // under — clamp, or hb_mm_mc_configure rejects the whole context.
    p->bitstream_buf_size =
        std::max(kEncOutBufMin,
                 alignUp(static_cast<uint32_t>(cfg.width) * cfg.height * 3u / 2u,
                         kDecInBufAlign));
    p->bitstream_buf_count = 5;
    // Internal input frame buffers: feed() copies the caller's NV12 into one of
    // these. The external_frame_buf path (hand the codec our VpImage's phys+virt
    // addresses, zero copy) works too — sunrise_camera uses it — but it makes the
    // caller's buffer codec-owned until the packet pops out, which is the same
    // borrowed-buffer lifetime problem as Step 1h. Left for that work.
    p->external_frame_buf = false;
    p->frame_buf_count = 5;
    p->rot_degree = MC_CCW_0;
    p->mir_direction = MC_DIRECTION_NONE;
    p->frame_cropping_flag = false;
    p->enable_user_pts = 1;

    // Streaming GOP: IDR refresh, no B-frames, single reference. The vendor code
    // notes the wave521 encoder supports neither B-frames nor multi-frame
    // reference, restricting presets to 1 and 9; 9 is the IPPP low-delay preset
    // both vendor apps use.
    p->gop_params.decoding_refresh_type = 2;
    p->gop_params.gop_preset_idx = 9;

    // CBR rate control. The h264/h265 CBR structs are field-for-field identical
    // but they are DISTINCT union members — write through the one that matches
    // the selected mode rather than aliasing, so this stays correct if the SDK
    // ever changes one of them.
    if (id == MEDIA_CODEC_ID_H265) {
      p->rc_params.mode = MC_AV_RC_MODE_H265CBR;
      mc_h265_cbr_params_t& rc = p->rc_params.h265_cbr_params;
      rc.intra_period = static_cast<uint32_t>(cfg.intra_period);
      rc.intra_qp = 30;
      rc.bit_rate = static_cast<uint32_t>(cfg.bitrate_kbps);
      rc.frame_rate = static_cast<uint32_t>(cfg.framerate);
      rc.initial_rc_qp = 20;
      rc.vbv_buffer_size = 20;
      rc.ctu_level_rc_enalbe = 1;
      rc.min_qp_I = 8;  rc.max_qp_I = 50;
      rc.min_qp_P = 8;  rc.max_qp_P = 50;
      rc.min_qp_B = 8;  rc.max_qp_B = 50;
      rc.hvs_qp_enable = 1;
      rc.hvs_qp_scale = 2;
      rc.max_delta_qp = 10;
      rc.qp_map_enable = 0;
    } else {
      p->rc_params.mode = MC_AV_RC_MODE_H264CBR;
      mc_h264_cbr_params_t& rc = p->rc_params.h264_cbr_params;
      rc.intra_period = static_cast<uint32_t>(cfg.intra_period);
      rc.intra_qp = 30;
      rc.bit_rate = static_cast<uint32_t>(cfg.bitrate_kbps);
      rc.frame_rate = static_cast<uint32_t>(cfg.framerate);
      rc.initial_rc_qp = 20;
      rc.vbv_buffer_size = 20;
      rc.mb_level_rc_enalbe = 1;
      rc.min_qp_I = 8;  rc.max_qp_I = 50;
      rc.min_qp_P = 8;  rc.max_qp_P = 50;
      rc.min_qp_B = 8;  rc.max_qp_B = 50;
      rc.hvs_qp_enable = 1;
      rc.hvs_qp_scale = 2;
      rc.max_delta_qp = 10;
      rc.qp_map_enable = 0;
    }

    BCDL_CHECK(hb_mm_mc_initialize(&ctx));
    BCDL_CHECK(hb_mm_mc_configure(&ctx));
    mc_av_codec_startup_params_t su{};
    su.video_enc_startup_params.receive_frame_number = 0;  // unbounded
    BCDL_CHECK(hb_mm_mc_start(&ctx, &su));
    started = true;
  }

  ~Impl() {
    if (started) drainAndRelease(ctx, kEncDrainPollMs);
  }

  // Queue one NV12 frame (eos=false) or an end-of-stream marker (eos=true).
  bool feedFrame(const VpImage* frame, uint64_t pts_us, bool eos) {
    media_codec_buffer_t in{};
    in.type = MC_VIDEO_FRAME_BUFFER;
    if (hb_mm_mc_dequeue_input_buffer(&ctx, &in, kEncInTimeoutMs) != 0) return false;

    if (eos) {
      in.vframe_buf.size = 0;
      in.vframe_buf.frame_end = true;
      return hb_mm_mc_queue_input_buffer(&ctx, &in, kEncInTimeoutMs) == 0;
    }

    // The codec's internal frame buffer is packed NV12: `height` rows of `width`
    // luma, then height/2 rows of interleaved chroma. The source VpImage has its
    // own strides, so copy row-wise rather than as one blob (the vendor's flat
    // memcpy silently assumes stride == width).
    const std::size_t need =
        static_cast<std::size_t>(width) * height * 3u / 2u;
    if (in.vframe_buf.size < need) {
      in.vframe_buf.size = 0;
      hb_mm_mc_queue_input_buffer(&ctx, &in, kEncInTimeoutMs);
      throw Error(-1, "BCDL: VideoEncoder codec input buffer smaller than one NV12 frame");
    }

    const hbVPImage& src = frame->raw();
    auto* y_dst = in.vframe_buf.vir_ptr[0];
    {
      const auto* sp = static_cast<const uint8_t*>(src.dataVirAddr);
      for (int y = 0; y < height; ++y)
        std::memcpy(y_dst + static_cast<std::size_t>(y) * width,
                    sp + static_cast<std::size_t>(y) * src.stride,
                    static_cast<std::size_t>(width));
    }
    // Chroma follows luma in the same allocation unless the codec handed us a
    // separate plane pointer.
    {
      auto* uv_dst = in.vframe_buf.vir_ptr[1] != nullptr
                         ? in.vframe_buf.vir_ptr[1]
                         : y_dst + static_cast<std::size_t>(width) * height;
      const auto* sp = static_cast<const uint8_t*>(src.uvVirAddr);
      for (int y = 0; y < height / 2; ++y)
        std::memcpy(uv_dst + static_cast<std::size_t>(y) * width,
                    sp + static_cast<std::size_t>(y) * src.uvStride,
                    static_cast<std::size_t>(width));
    }

    in.vframe_buf.width = width;
    in.vframe_buf.height = height;
    in.vframe_buf.pix_fmt = MC_PIXEL_FORMAT_NV12;
    in.vframe_buf.stride = width;
    in.vframe_buf.vstride = height;
    in.vframe_buf.size = static_cast<uint32_t>(need);
    in.vframe_buf.pts = pts_us;
    in.vframe_buf.frame_end = false;
    return hb_mm_mc_queue_input_buffer(&ctx, &in, kEncInTimeoutMs) == 0;
  }

  // Dequeue one compressed packet. Returns false on timeout / EOS marker.
  bool recv(std::vector<uint8_t>& out, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    media_codec_buffer_t ob{};
    media_codec_output_buffer_info_t info{};
    for (;;) {
      if (!waitForOutput(ctx, deadline)) return false;
      // The counter goes positive slightly before the buffer reaches the output
      // port, so give the dequeue a couple of ms; the packet is known to exist.
      if (hb_mm_mc_dequeue_output_buffer(&ctx, &ob, &info, kReadyWaitMs) == 0) break;
      if (Clock::now() >= deadline) return false;
    }

    // media_codec_buffer_t is a union: only read vstream_buf once we know this
    // IS a stream buffer. Reading it on a frame/status buffer reinterprets
    // unrelated bytes as a pointer+size and the copy below walks off into the
    // heap — the same trap the decoder hit on reorder/EOS buffers.
    if (ob.type != MC_VIDEO_STREAM_BUFFER) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);
      return false;
    }
    const mc_video_stream_buffer_info_t& sb = ob.vstream_buf;
    if (sb.vir_ptr == nullptr || sb.size == 0) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // empty / stream_end marker
      return false;
    }
    out.assign(sb.vir_ptr, sb.vir_ptr + sb.size);
    hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // return the buffer to the codec
    return true;
  }
};

VideoEncoder::VideoEncoder(const VideoEncConfig& cfg) : cfg_(cfg) {
  if (!isSupportedVideoFormat(cfg.format)) {
    throw Error(-1, "BCDL: VideoEncoder format must be NV12 or YUV420");
  }
  if (cfg.format != HB_VP_IMAGE_FORMAT_NV12) {
    throw Error(-1, "BCDL: VideoEncoder currently only supports NV12 input");
  }
  // Dimensions are fixed for the lifetime of the VPU context, so we REQUIRE
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
  if (cfg.intra_period < 0 || cfg.intra_period > 2047) {
    throw Error(-1, "BCDL: VideoEncoder intra_period must be in [0, 2047]");
  }
  impl_ = std::make_unique<Impl>(cfg);
}

VideoEncoder::~VideoEncoder() = default;

bool VideoEncoder::feed(const VpImage& frame, uint64_t pts_us) {
  if (frame.format() != cfg_.format) {
    throw Error(-1, "BCDL: VideoEncoder.feed source format != encoder format");
  }
  if (frame.width() != cfg_.width || frame.height() != cfg_.height) {
    throw Error(-1, "BCDL: VideoEncoder.feed source size != encoder size");
  }
  return impl_->feedFrame(&frame, pts_us, /*eos=*/false);
}

bool VideoEncoder::receive(std::vector<uint8_t>& out, int timeout_ms) {
  return impl_->recv(out, timeout_ms);
}

bool VideoEncoder::flush(std::vector<uint8_t>& out) {
  feedEndOfStream();
  return impl_->recv(out, kEncFlushWaitMs);
}

void VideoEncoder::feedEndOfStream() {
  if (!impl_->eos_fed) {
    impl_->feedFrame(nullptr, 0, /*eos=*/true);
    impl_->eos_fed = true;
  }
}

std::vector<uint8_t> VideoEncoder::encode(const VpImage& frame) {
  std::vector<uint8_t> out;
  if (!feed(frame)) return out;
  impl_->recv(out, kEncodeWaitMs);
  return out;  // empty => frame buffered, nothing emitted this call
}

// ---------------------------------------------------------------------------
// VideoDecoder
// ---------------------------------------------------------------------------

namespace {
constexpr int kDecInTimeoutMs = 1000;  // dequeue/queue an input buffer
constexpr int kDecodeWaitMs = 40;      // decode(): wait for one ready frame
constexpr int kFlushWaitMs = 300;      // flush(): drain reorder tail
constexpr int kDrainPollMs = 20;       // ~Impl: poll for buffers the codec holds
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
    // FRAME_SIZE: each queued input buffer holds exactly one access unit. This is
    // what both vendor apps use (sample_codec, sunrise_camera), and it is REQUIRED
    // here: with MC_FEEDING_MODE_STREAM_SIZE the decoder's own bitstream-ring
    // update path (updateDecoderBitstream -> vdi_write_memory -> osal_free)
    // corrupts the glibc heap and the process dies mid-decode — measured on S100P
    // / libmultimedia 1.2.3 at 6/20 runs decoding alone, 16/20 with any concurrent
    // load. FRAME_SIZE: 0/20 in both.
    // Callers must therefore feed() whole access units, not arbitrary byte chunks.
    p->feed_mode = MC_FEEDING_MODE_FRAME_SIZE;
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
    if (started) drainAndRelease(ctx, kDrainPollMs);
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
    // Gate on the status counter, never block inside the dequeue — see
    // waitForOutput(). Semantics unchanged: timeout_ms == 0 still never waits.
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    media_codec_buffer_t ob{};
    media_codec_output_buffer_info_t info{};
    for (;;) {
      if (!waitForOutput(ctx, deadline)) return false;
      // The counter is incremented before the frame reaches the output port, so a
      // zero-timeout dequeue right after it turns positive still misses ~1% of the
      // time — and a missed dequeue is exactly what prints. Give it a couple of
      // milliseconds: the frame is known to exist, so this returns at once.
      if (hb_mm_mc_dequeue_output_buffer(&ctx, &ob, &info, kReadyWaitMs) == 0) break;
      if (Clock::now() >= deadline) return false;
    }

    // media_codec_buffer_t is a union. Only interpret it as a decoded picture
    // once we've confirmed (a) it IS a frame buffer — not a stream / EOS /
    // status buffer — and (b) the decode actually succeeded (decode_result != 0;
    // 0 == FAIL). Reading vframe_buf on a non-frame or failed buffer reinterprets
    // unrelated union bytes as width/stride/vir_ptr and the copy below then reads
    // and writes out of bounds, corrupting the heap. The vendor sample
    // (sample_codec.c vp_codec_get_output) gates on exactly these two fields;
    // skipping them is what let reorder/EOS buffers crash the concurrent path.
    if (ob.type != MC_VIDEO_FRAME_BUFFER ||
        info.video_frame_info.decode_result == 0) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // hand the buffer back, skip
      return false;
    }

    const mc_video_frame_buffer_info_t& vb = ob.vframe_buf;
    const bool valid = vb.size > 0 && vb.width > 0 && vb.height > 0 &&
                       vb.vir_ptr[0] != nullptr;
    if (!valid) {
      hb_mm_mc_queue_output_buffer(&ctx, &ob, 0);  // return the empty/EOS buffer
      return false;
    }

    // Reuse the caller's buffer when it already matches; only (re)allocate on a
    // size/format change. A caller that recycles one buffer (or a pool) then
    // pays no per-frame hbUCPMallocCached/hbUCPFree on this hot path.
    if (out.width() != vb.width || out.height() != vb.height ||
        out.format() != HB_VP_IMAGE_FORMAT_NV12) {
      out = VpImage(vb.width, vb.height, HB_VP_IMAGE_FORMAT_NV12);
    }
    const hbVPImage& dst = out.raw();
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

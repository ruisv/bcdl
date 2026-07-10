#include "bcdl/preproc/gdc_letterbox.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gdc_bin.h"  // runtime warp-LUT generation (shared with GdcRemap)

extern "C" {
#include "gdc_cfg.h"
#include "gdc_bin_cfg.h"
#include "hb_mem_mgr.h"
#include "hbn_vpf_interface.h"
}

// These live in SDK sample headers (vp_sensors.h / hbn_isp_cfg.h), not the core
// hobot headers we include — they are plain constants, so define them locally.
#ifndef MAGIC_NUMBER
#define MAGIC_NUMBER 0x12345678
#endif
#ifndef AUTO_ALLOC_ID
#define AUTO_ALLOC_ID (-1)
#endif

namespace bcdl {

namespace {

#define GDC_CHECK(expr)                                                      \
  do {                                                                       \
    int _r = (expr);                                                         \
    if (_r != 0)                                                             \
      throw std::runtime_error(std::string("GdcLetterbox: ") + #expr +       \
                               " failed, ret=" + std::to_string(_r));        \
  } while (0)

inline int align16(int v) { return (v + 15) & ~15; }
inline int even(int v) { return v & ~1; }

// Copy an NV12 plane region [x,y,w,h] (bytes) from src(stride) to dst(stride).
void copyBlock(uint8_t* dst, int dstStride, const uint8_t* src, int srcStride,
               int x, int y, int w, int h) {
  for (int r = 0; r < h; ++r)
    std::memcpy(dst + (size_t)(y + r) * dstStride + x,
                src + (size_t)(y + r) * srcStride + x, w);
}

// Same, mapping each byte through a 256-entry LUT (range conversion rides along
// with the copy-out, so it costs no extra pass over the frame).
void copyBlockLut(uint8_t* dst, int dstStride, const uint8_t* src, int srcStride,
                  int x, int y, int w, int h, const uint8_t* lut) {
  for (int r = 0; r < h; ++r) {
    const uint8_t* s = src + (size_t)(y + r) * srcStride + x;
    uint8_t* d = dst + (size_t)(y + r) * dstStride + x;
    for (int c = 0; c < w; ++c) d[c] = lut[s[c]];
  }
}

uint8_t clampU8(double v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : std::lround(v))); }

}  // namespace

struct GdcLetterbox::Impl {
  int iw = 0, ih = 0, ow = 0, oh = 0;
  int px = 0, py = 0, cw = 0, ch = 0;  // content window (even)
  uint8_t pad = 114;
  bool range_conv = false;             // apply y_lut / uv_lut on copy-out
  uint8_t y_lut[256]{};
  uint8_t uv_lut[256]{};
  hbn_vnode_handle_t vnode = 0;
  hb_mem_common_buf_t bin{};
  hbn_vnode_image_t in{};  // persistent input graph buffer
  bool mem_open = false;
  bool timing = false;     // BCDL_GDC_TIMING (read once at construction)
};

GdcLetterbox::GdcLetterbox(int in_w, int in_h, int out_w, int out_h, uint8_t pad,
                           YuvRange range)
    : p_(new Impl), in_w_(in_w), in_h_(in_h), out_w_(out_w), out_h_(out_h) {
  p_->iw = in_w; p_->ih = in_h; p_->ow = out_w; p_->oh = out_h; p_->pad = pad;

  // Studio->full expansion, the exact composition of cv::cvtColor(YUV2BGR_NV12)
  // with a full-range BGR->NV12 encode: the cross terms cancel and what is left
  // is a per-plane affine map. 255/219 on luma, 255/224 on chroma.
  p_->range_conv = (range == YuvRange::kStudioToFull);
  for (int v = 0; v < 256; ++v) {
    p_->y_lut[v] = clampU8((v - 16) * (255.0 / 219.0));
    p_->uv_lut[v] = clampU8(128.0 + (v - 128) * (255.0 / 224.0));
  }

  // Geometry (aspect-preserving fit + center), identical to bcdl::computeLetterbox
  // so post-processing's inverse map is unchanged.
  lb_ = computeLetterbox(in_w, in_h, out_w, out_h);
  p_->cw = even((int)std::lround(in_w * lb_.scale));
  p_->ch = even((int)std::lround(in_h * lb_.scale));
  p_->px = even((int)std::lround(lb_.padX));
  p_->py = even((int)std::lround(lb_.padY));
  // Clamp the content window so the copy-out never exceeds the output extent.
  p_->cw = even(std::min(p_->cw, out_w - p_->px));
  p_->ch = even(std::min(p_->ch, out_h - p_->py));
  p_->timing = std::getenv("BCDL_GDC_TIMING") != nullptr;

  try {
    GDC_CHECK(hb_mem_module_open());
    p_->mem_open = true;

    // 1) Generate the warp LUT for this geometry. The letterbox is affine, and
    // bilinear interpolation between grid nodes reproduces an affine map
    // EXACTLY — provided no cell straddles the content/pad edge, where nodes get
    // clamped into the input. Hence a grid step that divides the content window
    // as well as the output extent. Pixels outside the content window sample
    // clamped (edge-replicated) input; run() overwrites them with flat pad.
    const int step = gdcbin::gridStepFor({out_w, out_h, p_->px, p_->py, p_->cw, p_->ch});
    const int gw = out_w / step + 1, gh = out_h / step + 1;
    std::vector<point_t> pts(static_cast<size_t>(gw) * gh);
    const double sc = lb_.scale;
    for (int gy = 0; gy < gh; ++gy)
      for (int gx = 0; gx < gw; ++gx) {
        // Evaluate the inverse letterbox AT the node coordinate (which for the
        // last node is the output's exclusive edge, out_w/out_h). Sampling a
        // dense map instead would clamp that node one pixel inward — 1/scale
        // input pixels of error across the whole final cell. Pixel-center
        // convention, matching cv::resize(INTER_LINEAR) and letterboxCpu().
        const double u = gx * step, v = gy * step;
        point_t& p = pts[static_cast<size_t>(gy) * gw + gx];
        p.x = (u - p_->px + 0.5) / sc - 0.5;
        p.y = (v - p_->py + 0.5) / sc - 0.5;
      }
    gdcbin::clampAndDedupeGrid(pts, gw, gh, in_w, in_h);
    gdcbin::allocCustomGridCfg(pts, gw, gh, in_w, in_h, out_w, out_h, &p_->bin);

    // 2) Open + configure the GDC vnode (feedback mode).
    GDC_CHECK(hbn_vnode_open(HB_GDC, 0, AUTO_ALLOC_ID, &p_->vnode));

    gdc_settings_t s{};
    s.gdc_config.config_addr = p_->bin.phys_addr;
    s.gdc_config.config_size = p_->bin.size;
    s.gdc_config.input_width = in_w;
    s.gdc_config.input_height = in_h;
    s.gdc_config.input_stride = align16(in_w);
    s.gdc_config.output_width = out_w;
    s.gdc_config.output_height = out_h;
    s.gdc_config.output_stride = align16(out_w);
    s.gdc_config.div_width = 0;
    s.gdc_config.div_height = 0;
    s.gdc_config.total_planes = 2;
    s.binary_ion_id = p_->bin.share_id;
    s.binary_offset = p_->bin.offset;
    s.magicNumber = MAGIC_NUMBER;

    GDC_CHECK(hbn_vnode_set_attr(p_->vnode, &s));
    GDC_CHECK(hbn_vnode_set_ichn_attr(p_->vnode, 0, &s));
    GDC_CHECK(hbn_vnode_set_ochn_attr(p_->vnode, 0, &s));

    hbn_buf_alloc_attr_t aa{};
    aa.buffers_num = 3;
    aa.is_contig = 1;
    aa.flags = HB_MEM_USAGE_CPU_READ_OFTEN | HB_MEM_USAGE_CPU_WRITE_OFTEN | HB_MEM_USAGE_CACHED;
    GDC_CHECK(hbn_vnode_set_ochn_buf_attr(p_->vnode, 0, &aa));

    GDC_CHECK(hbn_vnode_start(p_->vnode));

    // 3) Persistent input graph buffer (NV12, in_w x in_h). Allocate at the
    // 16-aligned stride we told GDC (input_stride) so the two always agree.
    const int64_t buf_flags = HB_MEM_USAGE_MAP_INITIALIZED |
                              HB_MEM_USAGE_PRIV_HEAP_2_RESERVERD |
                              HB_MEM_USAGE_CPU_READ_OFTEN |
                              HB_MEM_USAGE_CPU_WRITE_OFTEN | HB_MEM_USAGE_CACHED;
    GDC_CHECK(hb_mem_alloc_graph_buf(in_w, in_h, MEM_PIX_FMT_NV12, buf_flags,
                                     align16(in_w), in_h, &p_->in.buffer));
  } catch (...) {
    this->~GdcLetterbox();
    throw;
  }
}

GdcLetterbox::~GdcLetterbox() {
  if (!p_) return;
  if (p_->vnode) {
    hbn_vnode_stop(p_->vnode);
    hbn_vnode_close(p_->vnode);
  }
  if (p_->in.buffer.fd[0]) hb_mem_free_buf(p_->in.buffer.fd[0]);
  if (p_->bin.fd) hb_mem_free_buf(p_->bin.fd);
  if (p_->mem_open) hb_mem_module_close();
  delete p_;
  p_ = nullptr;
}

LetterboxInfo GdcLetterbox::run(const VpImage& src, VpImage& dst) {
  if (src.width() != in_w_ || src.height() != in_h_)
    throw std::runtime_error("GdcLetterbox::run: src size mismatch");
  const bool tm = p_->timing;
  auto now = [] { return std::chrono::high_resolution_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t0 = now();

  // Copy src NV12 (Y + interleaved UV) into the persistent device input buffer.
  const int in_stride = p_->in.buffer.stride;
  uint8_t* in_y = static_cast<uint8_t*>(p_->in.buffer.virt_addr[0]);
  uint8_t* in_uv = static_cast<uint8_t*>(p_->in.buffer.virt_addr[1]);
  const auto& r = src.raw();
  const uint8_t* sy = static_cast<const uint8_t*>(src.data());
  const uint8_t* suv = static_cast<const uint8_t*>(r.uvVirAddr);
  for (int y = 0; y < in_h_; ++y)
    std::memcpy(in_y + (size_t)y * in_stride, sy + (size_t)y * r.stride, in_w_);
  for (int y = 0; y < in_h_ / 2; ++y)
    std::memcpy(in_uv + (size_t)y * in_stride, suv + (size_t)y * r.uvStride, in_w_);
  hb_mem_flush_buf(p_->in.buffer.fd[0], 0, p_->in.buffer.size[0]);
  hb_mem_flush_buf(p_->in.buffer.fd[1], 0, p_->in.buffer.size[1]);
  auto t1 = now();

  // GDC hardware transform.
  hbn_vnode_image_t out{};
  GDC_CHECK(hbn_vnode_sendframe(p_->vnode, 0, &p_->in));
  GDC_CHECK(hbn_vnode_getframe(p_->vnode, 0, 1000, &out));
  auto t2 = now();

  // dst = out_w x out_h NV12, prefilled with flat pad, then the content window
  // copied from the GDC output (guarantees clean borders regardless of GDC fill).
  if (!dst.valid() || dst.width() != out_w_ || dst.height() != out_h_ ||
      dst.format() != HB_VP_IMAGE_FORMAT_NV12)
    dst = VpImage(out_w_, out_h_, HB_VP_IMAGE_FORMAT_NV12);
  const auto& dr = dst.raw();
  uint8_t* dy = static_cast<uint8_t*>(dst.data());
  uint8_t* duv = static_cast<uint8_t*>(dr.uvVirAddr);
  for (int y = 0; y < out_h_; ++y) std::memset(dy + (size_t)y * dr.stride, p_->pad, out_w_);
  for (int y = 0; y < out_h_ / 2; ++y) std::memset(duv + (size_t)y * dr.uvStride, 128, out_w_);

  const int os = out.buffer.stride;
  const uint8_t* oy = static_cast<const uint8_t*>(out.buffer.virt_addr[0]);
  const uint8_t* ouv = static_cast<const uint8_t*>(out.buffer.virt_addr[1]);
  if (p_->range_conv) {
    copyBlockLut(dy, dr.stride, oy, os, p_->px, p_->py, p_->cw, p_->ch, p_->y_lut);
    copyBlockLut(duv, dr.uvStride, ouv, os, p_->px, p_->py / 2, p_->cw, p_->ch / 2,
                 p_->uv_lut);
  } else {
    copyBlock(dy, dr.stride, oy, os, p_->px, p_->py, p_->cw, p_->ch);            // Y content
    copyBlock(duv, dr.uvStride, ouv, os, p_->px, p_->py / 2, p_->cw, p_->ch / 2);  // UV content
  }
  dst.cleanCache();

  hbn_vnode_releaseframe(p_->vnode, 0, &out);
  if (tm) {
    auto t3 = now();
    std::printf("[gdc] copy-in+flush %.2f  gdc-op(send+get) %.2f  copy-out %.2f  total %.2f ms\n",
                ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3));
  }
  return lb_;
}

}  // namespace bcdl

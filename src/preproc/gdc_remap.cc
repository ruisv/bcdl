#include "bcdl/preproc/gdc_remap.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "gdc_bin.h"  // runtime warp-LUT generation (shared with GdcLetterbox)

extern "C" {
#include "gdc_cfg.h"
#include "gdc_bin_cfg.h"
#include "hb_mem_mgr.h"
#include "hbn_vpf_interface.h"
}

#ifndef MAGIC_NUMBER
#define MAGIC_NUMBER 0x12345678
#endif
#ifndef AUTO_ALLOC_ID
#define AUTO_ALLOC_ID (-1)
#endif

namespace bcdl {

namespace {

#define GDC_CHECK(expr)                                                    \
  do {                                                                     \
    int _r = (expr);                                                       \
    if (_r != 0)                                                           \
      throw std::runtime_error(std::string("GdcRemap: ") + #expr +         \
                               " failed, ret=" + std::to_string(_r));      \
  } while (0)

inline int align16(int v) { return (v + 15) & ~15; }

}  // namespace

struct GdcRemap::Impl {
  hbn_vnode_handle_t vnode = 0;
  hb_mem_common_buf_t bin{};
  hbn_vnode_image_t in{};  // persistent input graph buffer
  bool mem_open = false;
  bool timing = false;
};

GdcRemap::GdcRemap(const float* map_x, const float* map_y, int in_w, int in_h,
                   int out_w, int out_h, int grid_step)
    : p_(new Impl), in_w_(in_w), in_h_(in_h), out_w_(out_w), out_h_(out_h) {
  if (grid_step < 1 || out_w % grid_step != 0 || out_h % grid_step != 0)
    throw std::runtime_error("GdcRemap: grid_step must divide out_w and out_h");
  p_->timing = std::getenv("BCDL_GDC_TIMING") != nullptr;

  // CUSTOM transformation grid: (gw x gh) nodes over the regular output grid
  // (tile corners every grid_step px, last node clamped to the last pixel);
  // each node stores the INPUT coordinate to sample — the dense map, sparsified.
  //
  // CUSTOM transformation grid: (gw x gh) nodes over the regular output grid
  // (tile corners every grid_step px, last node clamped to the last pixel);
  // each node stores the INPUT coordinate to sample — the dense map, sparsified.
  // Semantics recovered from libgdcbin's transform_custom (the SDK header
  // comments are wrong/misleading for this build):
  //   - custom.w/h are TILE counts; points holds (w+1)*(h+1) nodes, row-major.
  //   - the hardware maps an output pixel (u, v) to grid coords
  //       gx = (u - out_w/2 - pan) / cell / zoom + centerx
  //       gy = (v - out_h/2 - tilt) / cell / zoom + centery
  //     with cell = max(out_w/custom.w, out_h/custom.h) — so centerx/centery
  //     are in GRID-INDEX units (w/2, h/2 for a full-frame grid), and the two
  //     tile dims must be equal (grid_step divides both) or the smaller axis
  //     is not fully covered.
  //   - the grid node at (gx, gy) then yields the INPUT coordinate (bilinear
  //     between nodes); nodes must be strictly positive and adjacent nodes
  //     must not coincide (validator), hence the epsilon de-duplication on
  //     clamped border nodes.
  int gw = 0, gh = 0;
  const auto pts =
      gdcbin::gridFromDenseMap(map_x, map_y, in_w, in_h, out_w, out_h, grid_step, &gw, &gh);

  try {
    GDC_CHECK(hb_mem_module_open());
    p_->mem_open = true;

    // 1) Generate the warp-LUT cfg at runtime from the CUSTOM grid.
    gdcbin::allocCustomGridCfg(pts, gw, gh, in_w, in_h, out_w, out_h, &p_->bin);

    const int64_t flags = HB_MEM_USAGE_MAP_INITIALIZED | HB_MEM_USAGE_PRIV_HEAP_2_RESERVERD |
                          HB_MEM_USAGE_CPU_READ_OFTEN | HB_MEM_USAGE_CPU_WRITE_OFTEN |
                          HB_MEM_USAGE_CACHED;

    // 2) Open + configure the GDC vnode (feedback mode) — as GdcLetterbox.
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

    // 3) Persistent input graph buffer (NV12, in_w x in_h) at the same
    // 16-aligned stride we told GDC.
    GDC_CHECK(hb_mem_alloc_graph_buf(in_w, in_h, MEM_PIX_FMT_NV12, flags,
                                     align16(in_w), in_h, &p_->in.buffer));
  } catch (...) {
    this->~GdcRemap();
    throw;
  }
}

GdcRemap::~GdcRemap() {
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

void GdcRemap::run(const VpImage& src, VpImage& dst) {
  if (src.width() != in_w_ || src.height() != in_h_)
    throw std::runtime_error("GdcRemap::run: src size mismatch");
  if (src.format() != HB_VP_IMAGE_FORMAT_NV12)
    throw std::runtime_error("GdcRemap::run: src must be NV12");
  const bool tm = p_->timing;
  auto now = [] { return std::chrono::high_resolution_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t0 = now();

  // Copy src NV12 into the persistent device input buffer.
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

  // Copy the full output frame out of the vnode buffer.
  if (!dst.valid() || dst.width() != out_w_ || dst.height() != out_h_ ||
      dst.format() != HB_VP_IMAGE_FORMAT_NV12)
    dst = VpImage(out_w_, out_h_, HB_VP_IMAGE_FORMAT_NV12);
  const auto& dr = dst.raw();
  uint8_t* dy = static_cast<uint8_t*>(dst.data());
  uint8_t* duv = static_cast<uint8_t*>(dr.uvVirAddr);
  const int os = out.buffer.stride;
  const uint8_t* oy = static_cast<const uint8_t*>(out.buffer.virt_addr[0]);
  const uint8_t* ouv = static_cast<const uint8_t*>(out.buffer.virt_addr[1]);
  for (int y = 0; y < out_h_; ++y)
    std::memcpy(dy + (size_t)y * dr.stride, oy + (size_t)y * os, out_w_);
  for (int y = 0; y < out_h_ / 2; ++y)
    std::memcpy(duv + (size_t)y * dr.uvStride, ouv + (size_t)y * os, out_w_);
  dst.cleanCache();

  hbn_vnode_releaseframe(p_->vnode, 0, &out);
  if (tm) {
    auto t3 = now();
    std::printf("[gdc-remap] copy-in+flush %.2f  gdc-op %.2f  copy-out %.2f  total %.2f ms\n",
                ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3));
  }
}

}  // namespace bcdl

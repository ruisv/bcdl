#include "gdc_bin.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

extern "C" {
// libgdcbin internals (exported, but absent from the SDK headers). The public
// hbn_gen_gdc_cfg wrapper NEVER forwards CUSTOM grid points — it calls only
// gdc_init/gdc_calculate (verified by disassembly), so the validator sees a
// zeroed grid. Drive the generator directly and inject the points in between.
void* gdc_init(void);
uint32_t gdc_calculate(void* ctx, const param_t* prm, const window_t* wins, uint32_t wnd_num,
                       void** cfg_buf, uint32_t a, void* b, uint32_t c, uint32_t d);
void gdc_set_custom_points(void* ctx, const custom_tranformation_t* c);
void gdc_cleanup(void* ctx);
}

namespace bcdl {
namespace gdcbin {

std::vector<point_t> gridFromDenseMap(const float* map_x, const float* map_y, int in_w, int in_h,
                                      int out_w, int out_h, int grid_step, int* gw, int* gh) {
  if (grid_step < 1 || out_w % grid_step != 0 || out_h % grid_step != 0)
    throw std::runtime_error("gdcbin: grid_step must divide out_w and out_h");

  const int nw = out_w / grid_step + 1;  // nodes per row
  const int nh = out_h / grid_step + 1;  // node rows
  std::vector<point_t> pts(static_cast<size_t>(nw) * nh);
  for (int gy = 0; gy < nh; ++gy) {
    // The last node sits on the output's exclusive edge, which a dense map has no
    // sample for; clamp to the last pixel. (Callers with an analytic map should
    // evaluate it at the true node coordinate and call clampAndDedupeGrid instead —
    // for a downscale this one-pixel clamp is 1/scale input pixels of error in the
    // final cell.)
    const int oy = std::min(gy * grid_step, out_h - 1);
    for (int gx = 0; gx < nw; ++gx) {
      const int ox = std::min(gx * grid_step, out_w - 1);
      const size_t di = static_cast<size_t>(oy) * out_w + ox;
      point_t& p = pts[static_cast<size_t>(gy) * nw + gx];
      p.x = static_cast<double>(map_x[di]);
      p.y = static_cast<double>(map_y[di]);
    }
  }
  clampAndDedupeGrid(pts, nw, nh, in_w, in_h);
  *gw = nw;
  *gh = nh;
  return pts;
}

void clampAndDedupeGrid(std::vector<point_t>& pts, int gw, int gh, int in_w, int in_h) {
  for (auto& p : pts) {
    p.x = std::min(std::max(0.5, p.x), in_w - 1.0);
    p.y = std::min(std::max(0.5, p.y), in_h - 1.0);
  }
  // The validator rejects adjacent nodes that coincide (happens where the map
  // clamps at a border): nudge duplicates apart by ~1e-3 px — only ever hits
  // out-of-image border samples, visually irrelevant.
  for (int gy = 0; gy < gh; ++gy)
    for (int gx = 0; gx < gw; ++gx) {
      point_t& p = pts[static_cast<size_t>(gy) * gw + gx];
      auto same = [&](const point_t& q) {
        return std::abs(p.x - q.x) < 1e-9 && std::abs(p.y - q.y) < 1e-9;
      };
      if (gx > 0 && same(pts[static_cast<size_t>(gy) * gw + gx - 1])) p.x += 1e-3;
      if (gy > 0 && same(pts[static_cast<size_t>(gy - 1) * gw + gx])) p.y += 1e-3;
    }
}

void allocCustomGridCfg(const std::vector<point_t>& pts, int gw, int gh, int in_w, int in_h,
                        int out_w, int out_h, hb_mem_common_buf_t* bin) {
  param_t prm{};
  prm.format = FMT_SEMIPLANAR_420;
  prm.in = {static_cast<uint32_t>(in_w), static_cast<uint32_t>(in_h)};
  prm.out = {static_cast<uint32_t>(out_w), static_cast<uint32_t>(out_h)};
  prm.x_offset = 0;
  prm.y_offset = 0;
  // diameter/fov only parameterize the fisheye-style transforms; keep the
  // conventional defaults so CUSTOM ignores them.
  prm.diameter = std::min(in_w, in_h);
  prm.fov = 180.0;

  window_t win{};
  win.out_r = {0, 0, out_w, out_h};
  win.transform = CUSTOM;
  win.input_roi_r = {0, 0, in_w, in_h};
  win.zoom = 1.0;
  win.keep_ratio = 1;
  win.custom.full_tile_calc = 0;
  win.custom.tile_incr_x = 0;
  win.custom.tile_incr_y = 0;
  win.custom.w = gw - 1;  // tile counts, NOT node counts
  win.custom.h = gh - 1;
  win.custom.centerx = (gw - 1) / 2.0;  // GRID-INDEX units
  win.custom.centery = (gh - 1) / 2.0;
  win.custom.points = const_cast<point_t*>(pts.data());

  void* ctx = gdc_init();
  if (!ctx) throw std::runtime_error("gdcbin: gdc_init failed");
  gdc_set_custom_points(ctx, &win.custom);
  void* cfg_buf = nullptr;
  uint64_t scratch[2] = {0, 0};
  uint32_t words = gdc_calculate(ctx, &prm, &win, 1, &cfg_buf, 0, scratch, 0, 0);
  gdc_cleanup(ctx);
  const uint64_t cfg_size = static_cast<uint64_t>(words) * 4;
  if (!words || !cfg_buf) throw std::runtime_error("gdcbin: gdc_calculate produced no config");

  const int64_t flags = HB_MEM_USAGE_MAP_INITIALIZED | HB_MEM_USAGE_PRIV_HEAP_2_RESERVERD |
                        HB_MEM_USAGE_CPU_READ_OFTEN | HB_MEM_USAGE_CPU_WRITE_OFTEN |
                        HB_MEM_USAGE_CACHED;
  const int rc = hb_mem_alloc_com_buf(cfg_size, flags, bin);
  if (rc == 0) {
    std::memcpy(bin->virt_addr, cfg_buf, cfg_size);
    hb_mem_flush_buf(bin->fd, 0, cfg_size);
  }
  hbn_free_gdc_cfg(static_cast<uint32_t*>(cfg_buf));
  if (rc != 0)
    throw std::runtime_error("gdcbin: hb_mem_alloc_com_buf failed, ret=" + std::to_string(rc));
}

int gridStepFor(std::initializer_list<int> dims) {
  int g = 0;
  for (int d : dims)
    if (d > 0) g = std::gcd(g, d);
  if (g <= 0) return 1;
  for (int s = std::min(g, 32); s >= 1; --s)
    if (g % s == 0) return s;
  return 1;
}

}  // namespace gdcbin
}  // namespace bcdl

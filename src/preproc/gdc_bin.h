#pragma once

// Internal helper: drive libgdcbin's warp-LUT generator directly to build a
// CUSTOM-grid config at runtime. Shared by GdcRemap (arbitrary dense map) and
// GdcLetterbox (affine letterbox expressed as a map). Not a public header.
//
// The CUSTOM-grid semantics were recovered from libgdcbin's transform_custom —
// the SDK header comments are wrong for this build. See docs/GDC.md.

#include <cstdint>
#include <vector>

extern "C" {
// ORDER MATTERS: gdc_cfg.h must precede gdc_bin_cfg.h. gdc_cfg.h is what drags in
// hbn_vpf_data_info.h -> hb_gdc_data_info.h -> hb_gdc_cfg.h (gdc_settings_t);
// including gdc_bin_cfg.h first suppresses that chain and every gdc_settings_t
// use fails to compile.
#include "gdc_cfg.h"
#include "gdc_bin_cfg.h"
#include "hb_mem_mgr.h"
#include "hbn_vpf_interface.h"  // hbn_free_gdc_cfg
}

namespace bcdl {
namespace gdcbin {

/// Sample a dense output->input map at the grid nodes of a `grid_step` lattice.
///
/// `map_x`/`map_y` are (out_h, out_w) row-major: out(u,v) samples in(map_x[v,u],
/// map_y[v,u]) — cv2.remap semantics. Node coordinates are clamped into
/// [0.5, in-1] (the validator rejects anything else) and coinciding adjacent
/// nodes — which happens wherever the map clamps at a border — are nudged apart
/// by 1e-3 px. `grid_step` must divide out_w and out_h; `gw`/`gh` receive the
/// node counts.
std::vector<point_t> gridFromDenseMap(const float* map_x, const float* map_y, int in_w,
                                      int in_h, int out_w, int out_h, int grid_step,
                                      int* gw, int* gh);

/// Clamp node coordinates into [0.5, in-1] and nudge coinciding adjacent nodes
/// apart by 1e-3 px (the validator rejects duplicates, which appear wherever the
/// map runs off the input). Call on any hand-built grid before allocCustomGridCfg.
void clampAndDedupeGrid(std::vector<point_t>& pts, int gw, int gh, int in_w, int in_h);

/// Generate the warp-LUT for a CUSTOM grid and copy it into a freshly allocated
/// hb_mem common buffer (caller frees with hb_mem_free_buf(bin->fd)).
/// `pts` is row-major (gw * gh) nodes as produced by gridFromDenseMap().
/// hb_mem_module_open() must already have succeeded. Throws std::runtime_error.
void allocCustomGridCfg(const std::vector<point_t>& pts, int gw, int gh, int in_w, int in_h,
                        int out_w, int out_h, hb_mem_common_buf_t* bin);

/// The largest divisor of every argument that is <= 32 (the grid must land on
/// exact pixel boundaries of the content window, or cells straddling the
/// content/pad edge interpolate across clamped nodes). Always >= 1.
int gridStepFor(std::initializer_list<int> dims_that_must_divide);

}  // namespace gdcbin
}  // namespace bcdl

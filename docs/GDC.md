# GDC — the VPS hardware warp engine (`/dev/gdc`)

The S100P's VPS block includes ARM's **GDC** (Geometric Distortion Correction)
engine: a hardware warper that consumes NV12 and a pre-compiled "warp LUT"
config, and produces the warped NV12 with the CPU idle. BCDL wraps it twice:

| class | transform | LUT source | use |
|---|---|---|---|
| `bcdl::GdcLetterbox` | affine letterbox (as a CUSTOM grid) | **generated at runtime** | model-input letterbox |
| `bcdl::GdcRemap` | CUSTOM grid (arbitrary fixed warp) | **generated at runtime** | stereo rectification / dense remap |

Both build the LUT at construction via `src/preproc/gdc_bin.{h,cc}` (which drives
`libgdcbin`'s generator directly — `hbn_gen_gdc_cfg` never forwards CUSTOM grid
points). **Header order landmine:** `gdc_cfg.h` must be included before
`gdc_bin_cfg.h`, or `gdc_settings_t` is never declared.

Both hold a persistent FEEDBACK-mode vnode + device input buffer; `run(src,
dst)` costs two NV12 copies around the hardware op. `BCDL_GDC_TIMING=1` prints
the per-run breakdown.

## GdcLetterbox vs the CPU letterbox: same geometry, different aliasing

`GdcLetterbox` and `letterboxNv12Cpu()` are **not bit-identical**, measured on
S100P (1920×1080 → 640×640, both with `YuvRange::kStudioToFull`):

- **Geometry is identical.** On a linear ramp — which any bilinear kernel
  reproduces exactly — GDC matches the CPU path on 99.69% of pixels (max 1 LSB)
  and matches the analytic value to 1 LSB. Scale, pad offset and the pixel-centre
  convention all agree. Pad borders are byte-identical.
- **High-frequency response differs.** On a single-pixel impulse at 3× downscale,
  the CPU's 2×2 bilinear misses it entirely while GDC retains a trace: both alias,
  differently. On real video: Y 80.8% of pixels identical, mean |Δ| 0.65, p99 13,
  max 135 (texture edges); UV mean |Δ| 0.32.
- **End-to-end this is below the model's noise floor.** Full-stream detections:
  7791 (GDC) / 7718 (CPU NV12) / 7770 (the old BGR chain) — while perturbing the
  model input by a single grey level already moves 6.5% of detections.

Neither is strictly correct for a 3× downscale (that would be `INTER_AREA`), but
the CPU path is closer to the `cv2.resize(INTER_LINEAR)` the model was calibrated
with. `BCDL_NO_GDC=1` switches the pipeline to the CPU path for A/B.

**Detection-preproc geometry (1920×1080 NV12 → 640×640 letterbox), S100P:**
`copy-in 0.23 + gdc-op 0.69 (CPU idle) + copy-out 0.05 = 0.97 ms/frame`, of which
only 0.28 ms is CPU — against **4.71 ms/frame of pure CPU** for the equivalent
`nv12ToBgrCpu` + `letterboxCpu` + `bgrToNv12Cpu` chain the video pipeline uses
today; `AsyncVideoDetectionPipeline` uses the GDC path. (Measured via `GdcRemap`
with an affine-equivalent dense map, since `GdcLetterbox` still needs an offline
`.bin`; the hardware op is the same.)

## GdcRemap

```cpp
// cv2.remap semantics: out(x, y) = in(map_x[y,x], map_y[y,x])
bcdl::GdcRemap remap(map_x, map_y, in_w, in_h, out_w, out_h, /*grid_step=*/16);
remap.run(src_nv12, dst_nv12);
```
```python
g = bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)  # (out_h, out_w) f32 maps
dst = g.run(src)                                           # VpImage NV12 -> VpImage NV12
```

Measured (2448×2048 NV12, real stereo-rectify maps, S100P): **6.3 ms/frame
wall — copy-in 0.5 + GDC op 5.3 (CPU idle) + copy-out 0.5** — vs 14.7 ms
across all 6 CPU cores for `cv2.remap`. Output matches cv2.remap to mean 0.25 /
p99 2 grey-levels over the valid map region. Test/bench:
`tests/test_gdc_remap.py`.

Constraints:
- `grid_step` must divide `out_w` AND `out_h` (square grid cells; for
  2448×2048 that means 16). The dense maps are sampled every `grid_step` px.
- Map values are clamped into `[0.5, in-1]`; coinciding adjacent border nodes
  (a clamped-out corner) are nudged apart by 1e-3 px to satisfy the validator.

## CUSTOM-grid semantics (reverse-engineered — the SDK headers are wrong)

None of this is documented; it was recovered from `libgdcbin.so` disassembly
(`transform_custom`, `parse_input`, `gdc_set_custom_points`) and validated
empirically. The SDK header comments actively mislead:

1. **The public `hbn_gen_gdc_cfg` cannot compile CUSTOM windows.** Its libvpf
   implementation calls only `gdc_init` → `gdc_calculate` and never forwards
   `custom.points` — the validator then sees a zeroed grid ("Wrong coordinate
   (0.000000, 0.000000) …"). Drive libgdcbin directly:
   `gdc_init()` → `gdc_set_custom_points(win*, &custom)` →
   `gdc_calculate(ctx, param, windows, n, &cfg_buf, 0, scratch, 0, 0)`
   (returns the cfg size in u32 words) → `gdc_cleanup(ctx)`.
2. **`custom.w` / `custom.h` are TILE counts, not point counts** (the header
   says "number of points"): the points array holds `(w+1)*(h+1)` nodes,
   row-major from the top-left.
3. **`custom.centerx/centery` are in GRID-INDEX units** (`w/2`, `h/2` for a
   full-frame grid), NOT input pixels. The hardware maps an output pixel
   `(u, v)` to grid coordinates
   `gx = (u - out_w/2 - pan) / cell / zoom + centerx` (same for `gy`), with
   `cell = max(out_w/w, out_h/h)`, then bilinearly interpolates the node
   values (= INPUT coordinates) at `(gx, gy)`. Feeding input-pixel centers
   sends every lookup out of the grid → check_limits returns −1 → an all-black
   output with no error anywhere.
4. Validator rules (these produce the `Wrong coordinate …` stderr lines and a
   2-word error cfg): node coordinates must be strictly positive; adjacent
   nodes must not coincide; degenerate cfgs can also hang the vnode
   (`hbn_vnode_getframe` ret −43).

## Bring-up debugging map (symptom → cause)

| symptom | cause |
|---|---|
| `Wrong coordinate (0,0)×4, adjacent … same` | points never reached libgdcbin (wrapper path) or fewer points than `(w+1)*(h+1)` (over-read → zeros) |
| `… must bigger than 0` | negative/zero input coords in the grid (clamp first) |
| cfg generates but output is all black | centerx/centery in pixels instead of grid units (all lookups out of grid) |
| `hbn_vnode_getframe` ret −43 (timeout) | degenerate cfg reached the hardware (e.g. grid covering a fraction of the output) |
| `hbn_gen_gdc_cfg` ret −393225 | hbn-level INVALID_PARAMETER (module 6, code 9) — e.g. `custom.points == NULL` with `transform=CUSTOM` |

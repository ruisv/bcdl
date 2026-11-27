# GDC —— VPS 硬件 warp 引擎（`/dev/gdc`）

[English](GDC.md) | **简体中文**

S100P 的 VPS 模块里带有 ARM 的 **GDC**（Geometric Distortion Correction，几何畸变校正）
引擎：一个硬件 warp 单元，吃 NV12 和一份预先编译好的 "warp LUT" 配置，输出 warp 后的
NV12，整个过程 CPU 空闲。BCDL 对它做了两层封装：

| 类 | 变换 | LUT 来源 | 用途 |
|---|---|---|---|
| `bcdl::GdcLetterbox` | 仿射 letterbox（以 CUSTOM 网格实现） | **运行时生成** | 模型输入 letterbox |
| `bcdl::GdcRemap` | CUSTOM 网格（任意固定形变） | **运行时生成** | 双目校正 / 密集重映射 |

两者都在构造时通过 `src/preproc/gdc_bin.{h,cc}` 生成 LUT（它直接驱动 `libgdcbin` 的
生成器 —— `hbn_gen_gdc_cfg` 从不转发 CUSTOM 网格点）。**头文件顺序地雷：**
`gdc_cfg.h` 必须在 `gdc_bin_cfg.h` 之前 include，否则 `gdc_settings_t` 永远不会被声明。

两者都常驻一个 FEEDBACK 模式的 vnode 和一块设备端输入缓冲；`run(src, dst)` 的开销是
硬件算子前后各一次 NV12 拷贝。`BCDL_GDC_TIMING=1` 会打印每次运行的耗时拆分。

## GdcLetterbox 与 CPU letterbox：几何相同，走样不同

`GdcLetterbox` 与 `letterboxNv12Cpu()` **并非逐位一致**。在 S100P 上实测
（1920×1080 → 640×640，两者都用 `YuvRange::kStudioToFull`）：

- **几何完全一致。** 在线性斜坡图上（任何双线性核都能精确还原它），GDC 与 CPU 路径在
  99.69% 的像素上一致（最大差 1 LSB），与解析值的差也在 1 LSB 以内。缩放比例、补边偏移
  和像素中心约定全部吻合。补边区域逐字节一致。
- **高频响应不同。** 在 3× 缩小下用单像素脉冲测试，CPU 的 2×2 双线性把它完全漏掉，
  GDC 则还留有一丝痕迹：两者都会走样，只是走样方式不同。在真实视频上：Y 有 80.8% 的像素
  完全一致，平均 |Δ| 0.65，p99 13，最大 135（纹理边缘）；UV 平均 |Δ| 0.32。
- **端到端看，这点差异低于模型自身的噪声底。** 整段视频流的检测数：
  7791（GDC）/ 7718（CPU NV12）/ 7770（旧的 BGR 链） —— 而只要把模型输入扰动一个灰度级，
  就已经会让 6.5% 的检测结果发生变化。

对 3× 缩小来说，两者都不算严格正确（严格正确的是 `INTER_AREA`），但 CPU 路径更接近
模型校准时用的 `cv2.resize(INTER_LINEAR)`。设 `BCDL_NO_GDC=1` 可把流水线切到 CPU 路径
做 A/B 对比。

**检测预处理几何（1920×1080 NV12 → 640×640 letterbox），S100P：**
`copy-in 0.23 + gdc-op 0.69 (CPU idle) + copy-out 0.05 = 0.97 ms/frame`，其中只有
0.28 ms 占用 CPU —— 对比之下，视频流水线在预处理改为 NV12 原生之前所用的等价链路
`nv12ToBgrCpu` + `letterboxCpu` + `bgrToNv12Cpu` 是 **4.71 ms/帧的纯 CPU 开销**。
`AsyncVideoDetectionPipeline` 现在走 GDC 路径，CPU 回退是 `letterboxNv12Cpu`。（这组数据
是用 `GdcRemap` 配一张与该仿射等价的密集映射测出来的，当时 `GdcLetterbox` 还不能自己生成
LUT；硬件算子是同一个。）

## GdcRemap

```cpp
// cv2.remap 语义：out(x, y) = in(map_x[y,x], map_y[y,x])
bcdl::GdcRemap remap(map_x, map_y, in_w, in_h, out_w, out_h, /*grid_step=*/16);
remap.run(src_nv12, dst_nv12);
```
```python
g = bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)  # (out_h, out_w) f32 映射表
dst = g.run(src)                                           # VpImage NV12 -> VpImage NV12
```

实测（2448×2048 NV12，真实的双目校正映射，S100P）：**墙钟 6.3 ms/帧 —— 拷入 0.5 +
GDC 算子 5.3（CPU 空闲）+ 拷出 0.5**；对比 `cv2.remap` 占满全部 6 个 CPU 核要 14.7 ms。
在映射的有效区域内，输出与 cv2.remap 相差平均 0.25 / p99 2 个灰度级。测试与基准：
`tests/test_gdc_remap.py`。

约束：
- `grid_step` 必须同时整除 `out_w` 和 `out_h`（网格单元是正方形；对 2448×2048 来说
  就是 16）。密集映射表每隔 `grid_step` 像素采样一次。
- 映射值会被钳到 `[0.5, in-1]`；相邻边界节点若重合（某个角被钳出范围时会出现），会被
  推开 1e-3 像素，以通过校验器。

## CUSTOM 网格语义（逆向得出 —— SDK 头文件是错的）

这些内容没有任何文档；全部是从 `libgdcbin.so` 的反汇编（`transform_custom`、
`parse_input`、`gdc_set_custom_points`）里还原出来、再经实测验证的。SDK 头文件里的
注释会主动误导你：

1. **公开的 `hbn_gen_gdc_cfg` 编译不了 CUSTOM 窗口。** 它在 libvpf 里的实现只调用
   `gdc_init` → `gdc_calculate`，从不转发 `custom.points` —— 于是校验器看到的是一个
   全零网格（"Wrong coordinate (0.000000, 0.000000) …"）。要直接驱动 libgdcbin：
   `gdc_init()` → `gdc_set_custom_points(win*, &custom)` →
   `gdc_calculate(ctx, param, windows, n, &cfg_buf, 0, scratch, 0, 0)`
   （返回 cfg 的大小，以 u32 字（word）计）→ `gdc_cleanup(ctx)`。
2. **`custom.w` / `custom.h` 是格子（TILE）数，不是点数**（头文件写的是
   "number of points"）：points 数组存放 `(w+1)*(h+1)` 个节点，从左上角开始按行优先排列。
3. **`custom.centerx/centery` 的单位是网格索引**（整帧网格就是 `w/2`、`h/2`），
   不是输入像素。硬件把输出像素 `(u, v)` 映射到网格坐标
   `gx = (u - out_w/2 - pan) / cell / zoom + centerx`（`gy` 同理），其中
   `cell = max(out_w/w, out_h/h)`，再在 `(gx, gy)` 处对节点值（= 输入坐标）做双线性
   插值。如果填的是输入像素中心，每次查找都会落到网格之外 → check_limits 返回 −1 →
   输出全黑，而且哪儿都不报错。
4. 校验器规则（`Wrong coordinate …` 这类 stderr 行和长度只有 2 个字（word）的错误 cfg 就是它们
   产生的）：节点坐标必须严格为正；相邻节点不能重合；退化的 cfg 还可能让 vnode 卡死
   （`hbn_vnode_getframe` 返回 −43）。

## 调通排障对照（现象 → 原因）

| 现象 | 原因 |
|---|---|
| `Wrong coordinate (0,0)×4, adjacent … same` | 点根本没送到 libgdcbin（走了封装那条路），或点数少于 `(w+1)*(h+1)`（越界读 → 全零） |
| `… must bigger than 0` | 网格里有负的或为零的输入坐标（先钳位） |
| cfg 能生成，但输出全黑 | centerx/centery 填的是像素而不是网格单位（所有查找都落在网格外） |
| `hbn_vnode_getframe` 返回 −43（超时） | 退化的 cfg 送到了硬件（例如网格只覆盖了输出的一部分） |
| `hbn_gen_gdc_cfg` 返回 −393225 | hbn 层的 INVALID_PARAMETER（module 6，code 9） —— 例如 `transform=CUSTOM` 时 `custom.points == NULL` |

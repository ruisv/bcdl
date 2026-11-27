# BCDL C++ API 参考

[English](CPP_API.md) | **简体中文**

C++ 库（命名空间 `bcdl`）的完整参考。[`include/bcdl/`](../include/bcdl/) 下的
公共头文件才是唯一权威 —— 本文是它们的摘要加用法。Python 绑定见
[`API.zh.md`](API.zh.md)；两边的名字一一对应（Python snake_case ⇄ C++ camelCase）。

- [链接与构建](#链接与构建) · [约定](#约定) · [完整示例](#完整示例)
- core：[错误处理](#错误处理) · [SysMem](#sysmem) · [Task](#task) · [MemPool](#mempool) · [运行时信息](#运行时信息)
- backend：[Engine](#engine) · [输出读取与输入写入](#输出读取与输入写入)
- 预处理：[几何与 letterbox](#几何与-letterbox) · [GDC 硬件几何变换](#gdc-硬件几何变换) ·
  [VpImage](#vpimage)
- 任务：[检测 Detection](#检测-detection) · [全景驾驶感知 YOLOP](#全景驾驶感知-yolop) ·
  [分类 Classification](#分类-classification) · [图像嵌入 Embedding](#图像嵌入-embedding) ·
  [人脸检测与对齐 Face](#人脸检测与对齐-face) · [姿态 Pose](#姿态-pose) ·
  [全身姿态 WholeBody](#全身姿态-wholebody) ·
  [超分 SuperRes](#超分-superres) · [稀疏特征点 XFeat](#稀疏特征点-xfeat) ·
  [实例分割 InstanceSeg](#实例分割-instanceseg) · [开放词汇检测 YOLOE](#开放词汇检测-yoloe) ·
  [可提示分割 EdgeSAM](#可提示分割-edgesam) · [旋转框 OBB](#旋转框-obb) ·
  [语义分割 Segmentation](#语义分割-segmentation) · [单目 3D 检测 SMOKE](#单目-3d-检测-smoke) ·
  [激光雷达 3D 检测](#激光雷达-3d-检测-lidar-3-d-detection) ·
  [异常检测](#异常检测-anomaly-detection单类) · [光流](#光流-optical-flow) ·
  [端到端驾驶](#端到端驾驶-end-to-end-driving相机--激光雷达--轨迹) ·
  [深度 Depth](#深度-depth) · [深度精修](#深度精修-depth-refinementrgb-d) · [OCR](#ocr)
- 语音：[语音链](#语音链) · [语音识别](#语音识别paraformer)
- 多媒体：[JPEG JPU](#jpeg-jpu) · [视频 VPU](#视频-vpu)
- 跟踪与流水线：[ByteTracker](#bytetracker) · [DetectionPipeline](#detectionpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline) ·
  [AsyncVideoDetectionPipeline](#asyncvideodetectionpipeline) ·
  [TrackingPipeline](#trackingpipeline) · [StereoPipeline](#stereopipeline)

## 链接与构建

总头文件把一切都带进来：

```cpp
#include "bcdl/bcdl.h"     // 或者只 include 你用到的子头文件
```

在 CMake 里，`find_package(bcdl)` 会暴露 `bcdl::bcdl` 目标（传递性地带上头文件
与 hobot SDK 的链接库 —— 用的是板子自己的，位于 `/usr/include/hobot` 与
`/usr/hobot/lib`）：

```cmake
find_package(bcdl CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE bcdl::bcdl)
```

conda 包（`libbcdl`）与板上构建见
[README](../README.md#从源码构建)。这里的一切**只能在 RDK S100 / S100P / S600
板子上运行** —— hobot SDK 与 BPU/JPU/VPU 单元都只存在于板上。

## 约定

- **命名空间** `bcdl`。头文件后缀 `.h`；总头文件是 `bcdl/bcdl.h`。
- **错误** —— 每次 hobot SDK 调用都包在 `BCDL_CHECK(...)` 里，非零时抛出
  `bcdl::Error`（其中带着 SDK 的返回 `code()`）。库函数在被误用时同样抛
  `bcdl::Error`。请用 `try/catch` 包住调用。
- **图像**是交织的 `HxWx3` uint8 BGR（OpenCV 顺序），行 stride 为 `width*3`，
  除非某个 `VpImage` 另有说明。
- **结果坐标是原图像素** —— 需要撤销 letterbox 的任务会接收一个 `LetterboxInfo`。
- **缓存纪律**（最常见的正确性 bug）由 `Engine` 与预处理/编解码辅助类负责；
  只有当你自己手搓缓冲时，才需要碰 `cleanCache()` / `invalidateCache()`。
  见 [SysMem](#sysmem)。
- **所有权** —— 任务类 / 流水线类以 `Engine&` 引用持有它，并**不**拥有它；
  请保证 `Engine` 活得比它们久。RAII 包装（`SysMem`、`Task`、`VpImage`、
  `Engine`、各编解码器）都是 move-only 的。

---

## 错误处理

`bcdl/core/status.h`

```cpp
class Error : public std::runtime_error {
  int code() const noexcept;   // hobot SDK 返回码（0 == 成功）
};

BCDL_CHECK(expr);   // expr != 0 时抛出 bcdl::Error(code, "expr @ file:line")
```

## SysMem

`bcdl/core/sys_mem.h` —— 对 `hbUCPSysMem` 的 RAII 封装。它就是那个被 BPU 张量、
JPU/VPU 图像与 VP 缓冲共用的唯一共享内存缓冲。Move-only。

```cpp
bcdl::SysMem mem(nbytes, /*cached=*/true, /*device_id=*/0);
void*    p   = mem.data();      // 虚拟地址
uint64_t pa  = mem.phyAddr();   // 物理地址
mem.cleanCache();               // CPU 写完之后、设备读之前
mem.invalidateCache();          // 设备写完之后、CPU 读之前
hbUCPSysMem& raw = mem.raw();   // 例如赋给 hbDNNTensor.sysMem
```

带缓存的缓冲（默认）在 CPU ⇄ 设备交接处需要显式 `cleanCache()` /
`invalidateCache()`；传 `cached=false` 可得到一致性（但更慢）的缓冲。

## Task

`bcdl/core/task.h` —— 对 `hbUCPTaskHandle_t` 的 RAII 封装。它就是 `hbDNNInferV2`
与每个 hbVP 算子返回的那个唯一任务句柄。Move-only。

```cpp
bcdl::Task t;
hbDNNInferV2(t.addr(), outputs, inputs, dnn);  // 生产者填充句柄
t.submit(/*priority=*/HB_UCP_PRIORITY_LOWEST);
t.wait(/*timeout_ms=*/0);                       // 0 表示一直阻塞
// t.release() 在析构里执行
```

## MemPool

`bcdl/core/mem_pool.h` —— 可复用的带缓存 `SysMem` 块池，用于"每帧零分配"的
流水线。通过 RAII 的 `Lease` 归还块。

```cpp
bcdl::MemPool pool;                 // cached=true, device_id=0
pool.reserve(/*size=*/4 << 20, /*count=*/4);   // 启动时预热
{
  auto lease = pool.acquire(nbytes);   // 取 >= nbytes 的最小空闲块，没有就新分配
  void* p = lease.data();
  lease.cleanCache();                  // 转发给底层块
}                                      // Lease 析构时把块还回池里
std::size_t free = pool.freeCount(), total = pool.blockCount();
uint64_t pooled = pool.bytesPooled();
pool.clear();                          // 释放所有未被租用的块
```

池必须比每个 `Lease` 活得久。线程安全（acquire/release 有互斥保护）；
回收的块不会被清零。

---

## 运行时信息

`bcdl/core/runtime.h` —— 当前进程实际加载的是哪一份 hobot 运行时。BCDL 链接的是
板子系统镜像自带的 SDK，不自带一份；不同板子上结果不一致时，先比较这个。

```cpp
bcdl::RuntimeInfo r = bcdl::runtimeInfo();   // 可以在任何 Engine 之前调用
// r.dnn_version  "3.13.6_(4.7.5 HBRT)"   hbDNNGetVersion()
// r.ucp_version  "3.13.6"                hbUCPGetVersion()
// r.soc_name     "D-Robotics RDK S100P V1P0"
// r.dnn_library  "/usr/hobot/lib/libdnn.so" —— 在别处说明有另一份 SDK
//                挡住了板上的那份
```

| 成员 | 含义 |
|---|---|
| `runtimeInfo()` | 返回带上述四个字段的 `RuntimeInfo`（链接器给不出路径时 `dnn_library` 为 `""`） |

---

## Engine

`bcdl/backend/engine.h` —— 加载编译好的 `.hbm`，通过 `hbDNN` + `hbUCP` 执行
BPU 推理。张量缓冲在构造时一次性分配并复用；缓存一致性由内部处理
（`setInput` 负责 clean，`infer` 负责 invalidate 输出）。

```cpp
bcdl::Engine engine("model.hbm", /*model_name=*/"");   // "" => 取第一个模型
int ni = engine.numInputs(), no = engine.numOutputs();
std::vector<int> ish = engine.inputShape(0);
int itype = engine.inputType(0);          // HB_DNN_TENSOR_TYPE_*
std::size_t ibytes = engine.inputBytes(0);

engine.setInput(0, host_ptr, nbytes);     // host -> 设备缓冲（并 clean）
engine.infer(/*timeout_ms=*/0);           // 提交 + 等待 + invalidate 输出
const void* out = engine.outputData(0);   // 读取时注意 outputProperties(0).stride
std::size_t obytes = engine.outputBytes(0);
```

| 成员 | 说明 |
|--------|-------------|
| `Engine(hbm_path, model_name="")` | 加载模型（包里有多个时按名字挑）。 |
| `modelName()` | 当前模型名。 |
| `static modelNames(hbm_path)` | 列出 `.hbm` 包里的全部模型名，不用先构造 Engine —— 也就是 `model_name` 该传什么。`.hbm` 是一个包，比如 SigLIP 编码器里就装了两个模型。 |
| `numInputs()` / `numOutputs()` | 张量个数。 |
| `inputName(i)` / `outputName(i)` | 编译进模型的张量名（即 ONNX 里的名字）。 |
| `inputShape(i)` / `outputShape(i)` | `std::vector<int>` 形状。 |
| `inputType(i)` / `outputType(i)` | `hbDNN` 张量类型枚举。 |
| `inputProperties(i)` / `outputProperties(i)` | 完整的 `hbDNNTensorProperties`（dtype、量化、stride）。 |
| `inputBytes(i)` / `outputBytes(i)` | 分配到的设备缓冲大小。 |
| `inputPackedBytes(i)` / `outputPackedBytes(i)` | 该张量作为连续行主序数组的大小 —— 模型对某一维补齐时会小于设备缓冲。 |
| `copyOutput(i, dst, bytes)` | 把输出 `i` 从补齐后的设备 layout 收拢成 `outputPackedBytes(i)` 字节的连续行主序数组；dtype 不变（`outputAsFloat()` 还会做反量化）。 |
| `inputStride(i)` | 输入 `i` 解析后的字节 stride，最内层在最后。 |
| `outputStride(i)` | 输出 `i` 的字节 stride。输出同样会补齐；`outputAsFloat()` 会处理好，直接读原始指针的代码**不能**摊平 reshape。 |
| `setInput(i, data, bytes)` | 把 host 字节拷进输入 `i` 并刷缓存。`bytes` 必须等于 `inputPackedBytes(i)`（按设备 layout 散布写入）或 `inputBytes(i)`（原样拷贝）；其他值一律抛异常。 |
| `inputData(i)` | 输入 `i` 设备缓冲的只读视图，按设备 layout。 |
| `infer(timeout_ms=0)` | 执行一次推理（阻塞；并 invalidate 输出缓存）。 |
| `outputData(i)` | `infer()` 之后输出 `i` 的指针（注意 stride）。 |
| `static elemSize(tensor_type)` | 某个 `hbDNN` 张量元素类型的字节大小。 |

> 相比自己读 `outputData()`，优先用**任务类**（`Detector`、`Segmenter`……） ——
> 它会替你反量化（F16/int8/int16）并处理 stride。只有写自定义解码器时才降到
> 原始输出；即便那样，也请经 `outputAsFloat()` 读取（[见下](#输出读取与输入写入)）。

## 输出读取与输入写入

`bcdl/backend/output_reader.h`、`bcdl/backend/input_writer.h` —— 每个任务类都建在这层
Engine 张量的浮点视图之上。自己解码（或自己喂）模型时用它们，而不是直接读
`outputData()` 或写原始字节。

```cpp
std::vector<float> scratch;            // 跨帧复用
std::vector<int> shape;
const float* out = bcdl::outputAsFloat(engine, /*out_idx=*/0, scratch, shape);
// 逻辑上的行主序浮点；在 `scratch` 销毁或下一次 infer() 之前有效

std::vector<std::uint8_t> stage;       // 每个输入一个，跨帧复用
std::size_t n = bcdl::inputElementCount(engine, /*in_idx=*/0);
bcdl::setInputFromFloat(engine, /*in_idx=*/0, values, n, stage);   // 需要时做量化
```

| 成员 | 说明 |
|---|---|
| `outputAsFloat(engine, out_idx, scratch, shape)` | 输出 `out_idx` 的逻辑行主序浮点视图，并填好 `shape`。连续排布的 F32 张量零拷贝返回（直接指向已 invalidate 的设备缓冲）；F16、量化整型以及 stride 带补齐的张量，会收拢并反量化到 `scratch` 里。 |
| `setInputFromFloat(engine, in_idx, data, count, scratch)` | 把 `count` 个浮点按模型声明的 dtype 写进输入 `in_idx`：F32 原样写入；整型（S8 … U32）按 `round(value / scale) + zero_point` 换算并钳到取值范围内；随后由 `setInput()` 把连续缓冲散布进设备 layout。 |
| `inputElementCount(engine, in_idx)` | 该输入有效形状的元素个数 —— 也就是 `count` 必须等于的值。 |
| `sigmoid(x)` | 所有 YOLO 家族解码器共用的 logistic sigmoid（header-only）。 |

零拷贝路径不是抠细节：检测、姿态、分割、OBB 头每帧输出 10^5–10^6 个元素，省掉这次收拢，
就是后处理约 10 ms 和约 1 ms 的区别。

`setInputFromFloat` 是为**输入也被量化**的模型准备的。这类模型的输入缓冲里装的是带 per-tensor
（或 per-axis）scale 的整数，直接塞 float 原始字节，模型会照常跑，只是在跑噪声，哪里都不报错。
`count` 与有效形状不符时它抛 `Error(-1)`；输入是 F16 时也抛 —— 本库没有哪个模型用 F16 输入，
一条没测过的 float→half 路径还不如一个明确的拒绝。

---

## 几何与 letterbox

`bcdl/preproc/geometry.h` —— 保持长宽比的贴合 + 坐标映射（header-only）。

```cpp
bcdl::LetterboxInfo lb = bcdl::computeLetterbox(srcW, srcH, dstW, dstH,
                                                /*centerPad=*/true);
float mx = lb.fwdX(x), my = lb.fwdY(y);   // 原图 -> 模型像素
float ox = lb.invX(x), oy = lb.invY(y);   // 模型 -> 原图像素（撤销 letterbox）
```

`LetterboxInfo` 字段：`scale`、`padX`、`padY`、`srcW/H`、`dstW/H`；方法
`fwdX/fwdY`、`invX/invY`、`clampX/clampY`。检测家族的每个 `postprocess(lb)` 都要
接收一个，结果才会落在原图像素上。

**CPU 预处理**（`bcdl/preproc/letterbox_cpu.h`） —— 部署板上的 hbVP 几何算子是
vDSP 支撑的、且只在离线/root 下可用，所以流水线走的是这些 CPU 路径。几何与
hbVP 路径完全一致。

```cpp
bcdl::VpImage src(w, h, HB_VP_IMAGE_FORMAT_BGR);
bcdl::VpImage nv12(inW, inH, HB_VP_IMAGE_FORMAT_NV12);
bcdl::LetterboxInfo lb = bcdl::letterboxToNv12Cpu(nv12, src, /*padValue=*/114);
// 另有：letterboxCpu(dstBgrOrY, src, pad) ; bgrToNv12Cpu(dstNv12, srcBgr)
//       letterboxNv12Cpu(dstNv12, srcNv12, pad, range)   // NV12 -> NV12，不经 BGR
```

`bgrToNv12Cpu` 用的是 BT.601 **full-range**（与 cv2 的 `COLOR_BGR2YUV_I420`
一致）；请与你模型标定时的范围对照确认。`letterboxNv12Cpu` 直接对各平面重采样：
亮度精确，色度与 BGR 路径略有差异（UV 平面在半分辨率上重采样）；`range`
（`YuvRange`，默认 `kStudioToFull`）决定要不要把视频解码器输出的 studio-swing 电平展开。

**hbVP 路径**（`bcdl/preproc/letterbox.h`，需要 DSP）：`letterbox(dst, src,
padValue, interp)`、`cvtColor(dst, src)`、`resizeExact(dst, src, interp)`。

> 这三个**未经验证 —— 请优先用上面的 CPU 路径或 GDC 路径**。它们跑在 vDSP 上，
> 而 S100P 4.0.5 系统镜像上 vDSP 并未上线：`/dev/vdsp0` 存在，但即使以 root 运行
> 内核也报 OFFLINE，且 `/lib/firmware` 里没有 DSP 固件，所以这些调用会抛
> `bcdl::Error` 而不是真正执行。特别是 `letterbox()` 的仿射约定（正向 src→dst
> 矩阵、`isInverse=0`）只是照 SDK 头文件写的，从未见它真的出过一帧。若在某块
> DSP 能起来的板子上使用，请先检查首帧有没有镜像或缩放错误再信任它。

**GDC 硬件路径** —— `GdcLetterbox` 与 `GdcRemap` 在 VPS GDC 引擎上做固定的 NV12
变换，期间 CPU 空闲；见 [GDC 硬件几何变换](#gdc-硬件几何变换)。

## GDC 硬件几何变换

`bcdl/preproc/gdc_letterbox.h`、`bcdl/preproc/gdc_remap.h`（仅在带 `BCDL_HAVE_GDC`
构建时可用） —— VPS GDC 引擎上的固定变换，NV12 进 / NV12 出，算子执行期间 CPU 空闲。
两个类都在构造时生成 warp LUT（不需要离线 `.bin`），并常驻一个 vnode 和一块设备输入
缓冲，所以 `run()` 的开销是硬件算子前后的两次 NV12 拷贝；设 `BCDL_GDC_TIMING=1` 会打印
每次的耗时拆分。语义、与 CPU 路径的实测对比，以及逆向出来的 CUSTOM 网格笔记见
[GDC.zh.md](GDC.zh.md)。

```cpp
// letterbox：几何与 computeLetterbox() 完全一致
bcdl::GdcLetterbox lbg(inW, inH, outW, outH, /*pad=*/114, bcdl::YuvRange::kAsIs);
bcdl::LetterboxInfo lb = lbg.run(srcNv12, dstNv12);

// 任意固定的密集重映射（cv2.remap 语义），LUT 运行时生成
bcdl::GdcRemap remap(mapX, mapY, inW, inH, outW, outH, /*gridStep=*/16);
remap.run(srcNv12, dstNv12);   // 2448x2048：墙钟约 6.3 ms，CPU 约 1 ms
```

| 成员 | 说明 |
|---|---|
| `GdcLetterbox(in_w, in_h, out_w, out_h, pad=114, range=kAsIs)` | 为一个固定的 letterbox 打开 GDC。`range` 必须与模型校准时的输入一致（见 `YuvRange`）；`pad` 是在电平转换之后写入的 full-range 亮度值，所以 114 写进去还是 114。任何 hobot/GDC 调用失败都会抛异常。 |
| `GdcLetterbox::run(src, dst)` | NV12 `in_w×in_h` → NV12 `out_w×out_h`，`dst` 尺寸不对时会重新分配；返回用于撤销 letterbox 的 `LetterboxInfo`。 |
| `info()` | 该实例固定的 `LetterboxInfo`。 |
| `GdcRemap(map_x, map_y, in_w, in_h, out_w, out_h, grid_step=16)` | 行主序 `(out_h, out_w)` 的 float32 密集映射，cv2 约定：`out(x, y) = in(map_x[y,x], map_y[y,x])`。每隔 `grid_step` 像素采样进 CUSTOM 网格，所以 `out_w`、`out_h` 都必须是它的整数倍（16 能整除 2448 和 2048）。落在输入范围外的坐标会被钳到范围内。 |
| `GdcRemap::run(src, dst)` | 在 GDC 上做 NV12 `in_w×in_h` → NV12 `out_w×out_h`，`dst` 尺寸不对时会重新分配。 |
| `inputWidth()` · `inputHeight()` · `outputWidth()` · `outputHeight()` | 两个类都有。 |

`GdcRemap` 是为双目立体校正做的：`cv2.initUndistortRectifyMap` 生成的映射很平滑，按网格
采样、节点之间交给 GDC 插值就够了。在真实的校正映射上，它与 `cv2.remap` 的差异是均值 0.25、
p99 为 2 个灰度级，耗时 6.3 ms，而 `cv2.remap` 用满 6 个核要 14.7 ms。`GdcLetterbox`
（1920×1080 → 640×640：0.97 ms，其中 CPU 约 0.3 ms）的几何与 CPU letterbox 完全一致，但不是
逐位相同 —— 两种重采样在高频细节上的混叠不一样，端到端测下来这个差异低于模型自身的噪声水平。

## VpImage

`bcdl/preproc/vp_image.h` —— 由带缓存 `SysMem` 支撑的 RAII 图像；VP 算子与
JPU/VPU 编解码器共用的统一缓冲。Move-only。

```cpp
bcdl::VpImage img(width, height, HB_VP_IMAGE_FORMAT_NV12);  // BGR/RGB/Y/NV12
void* y = img.data();          // 主（Y）平面
int   st = img.raw().stride;   // 行 stride（16 字节对齐）
img.cleanCache(); img.invalidateCache();
```

格式与内存布局：`BGR`/`RGB` 为交织 C3（`stride = align16(w*3)`）；`Y` 为灰度
C1（`align16(w)`）；`NV12` 是先 Y 平面、后交织 UV（宽高需为偶数）。

---

## 检测 Detection

`bcdl/tasks/detection.h` —— 两个解码器家族。

```cpp
struct Detection { float x1, y1, x2, y2, score; int class_id; };  // 原图像素
enum class DecodeLayout { kYoloV8, kYoloV5 };
```

**单个融合张量**（`[1,4+nc,N]` / `[1,N,4+nc]`）：

```cpp
bcdl::DetectConfig cfg;          // input_w/h, num_classes, conf_thresh,
cfg.num_classes = 80;            // iou_thresh, max_dets, layout, channels_first,
                                 // apply_sigmoid
// 纯解码一个浮点张量：
auto dets = bcdl::decode(data, shape /*={1,4+nc,N}*/, cfg, lb);
// 或绑定 Engine（反量化 + 去 stride + 解码）：
bcdl::Detector det(engine, cfg, /*output_index=*/0);
auto dets2 = det.postprocess(lb);     // 在 engine.infer() 之后
float i = bcdl::iou(a, b);
std::vector<int> keep = bcdl::nms(dets, cfg.iou_thresh, cfg.max_dets);
```

**anchor-free LTRB 多尺度**（YOLO26 / 标准 RDK NV12 导出） —— 每个 stride 一对
`(cls, box)` 输出：

```cpp
bcdl::YoloLtrbConfig cfg;        // num_classes, conf/iou_thresh, max_dets,
cfg.strides = {8, 16, 32};       // strides, reg_max（DFL 分箱；0 = 直接 LTRB）
bcdl::YoloLtrbDetector det(engine, cfg, /*output_base=*/0);
auto dets = det.postprocess(lb);      // 读取 2*strides.size() 个输出
// 纯函数形式：
auto d = bcdl::decodeYoloLtrb(cls, box, grid_hw, cfg, lb);
```

`reg_max > 0` 选择 DFL 头（`4*reg_max` 个框通道，softmax 后归约）；
`YoloLtrbDetector` 会从框通道数自动识别。

## 全景驾驶感知 YOLOP

`bcdl/tasks/panoptic_drive.h` —— 库里唯一一个**基于 anchor** 的检测解码（YOLOv5 的写法；
其余全是 anchor-free 的 LTRB/DFL）。YOLOP 用同一次推理喂它，同时产出可行驶区域和车道线
两张掩膜，这两张走普通的 `Segmenter`。

```cpp
struct Anchor { float w, h; };                               // 模型输入像素
struct AnchorDetectConfig { int num_classes=1; float conf_thresh=0.35f, iou_thresh=0.45f;
                            int max_dets=300; std::vector<int> strides={8,16,32};
                            std::vector<std::vector<Anchor>> anchors; };  // 每个尺度 3 个

bcdl::Engine engine("models/yolop_cut_nashm_640x640_nv12.hbm");   // 5 个输出
// 居中 letterbox 到 640x640 NV12，feedNv12Input(engine, nv12)，engine.infer()
bcdl::AnchorDetector det(engine, bcdl::AnchorDetectConfig{}, /*output_base=*/0);
std::vector<bcdl::Detection> cars = det.postprocess(lb);    // 输出 0-2

bcdl::SegConfig sc; sc.num_classes = 2; sc.channels_first = true;
bcdl::SegMask drivable = bcdl::Segmenter(engine, sc, /*output_index=*/3).postprocess();
bcdl::SegMask lanes    = bcdl::Segmenter(engine, sc, /*output_index=*/4).postprocess();
// 纯函数：decodeYoloV5Anchor(raw, grid_hw, cfg, lb)
```

| 成员 | 说明 |
|---|---|
| `AnchorDetector(engine, cfg={}, output_base=0)` | 绑定 `strides.size()` 个连续的原始检测头输出；超出模型输出个数时抛异常。 |
| `postprocess(lb)` | 逐头反量化、去 stride，每个尺度的 `(H, W)` 取自各自的张量，并检查通道数等于 `na * (5 + num_classes)` —— 类别数填错会直接报错，而不是悄悄读错列 —— 然后解码，做一次按类别的 NMS。 |

**要用 `_cut` 版本。** 官方导出把解码做在图里，直接输出现成的 `[1, 25200, 6]`，但那段解码由
`ScatterND` 拼成，过不了 BPU 编译：编译成功，编出来的张量里 objectness 和类别列却**从不被写入**，
检测器在任何阈值下都找不到目标。随库的版本把图切在解码之前，输出三个原始检测头卷积
（`[1, na*(5+nc), H, W]`，未激活），算术放到这里的 CPU 上做。这样还快 10 倍：图内解码在设备上要
28.33 ms，切图后整个推理、三个头一起只要 2.80 ms（358 FPS）。在真实帧上，CPU 解码与官方导出
自带的解码在全部 25200 个候选上相差不超过 1.8e-4。

解码完全照参考实现：**每个**属性都过 sigmoid（xy、wh 也不例外），中心
`(2·s − 0.5 + cell) · stride`，尺寸 `(2·s)² · anchor`，分数 = objectness × 最高类别分。anchor
尺寸的单位是像素而不是 stride，且每个尺度的 anchor 数必须相同。两张掩膜落在模型 640×640 的
letterbox 网格上；`Segmenter` 不接收 `LetterboxInfo`，所以裁掉补边、缩放回原图要自己做。

## 分类 Classification

`bcdl/tasks/classification.h`

```cpp
struct ClsConfig { int top_k = 5; bool apply_softmax = true; };
struct ClsResult { int class_id; float score; };

bcdl::Classifier clf(engine, cfg, /*output_index=*/0);
std::vector<bcdl::ClsResult> top = clf.postprocess();   // 在 infer() 之后
// 纯函数：decodeClassification(logits, num_classes, cfg)
```

## 图像嵌入 Embedding

`bcdl/tasks/embedding.h` —— 把一张图映射成定长的单位向量，外加一张检索这类向量的表。同一个
操作干两件事：**零样本分类**是拿图像向量去点乘一张类别名向量表，**以图搜图**是去点乘一张由
图库图片建成的表。

```cpp
struct EmbedConfig { bool l2_normalize = true; };
struct EmbedMatch { int index; float score; std::string label; };   // 余弦，从高到低

// .hbm 是一个包：按名字绑定池化（pooled）的那个子模型，而不是包里的默认项
std::vector<std::string> names = bcdl::Engine::modelNames(path);
bcdl::Engine engine(path, pooled_name);
bcdl::ImageEmbedder emb(engine);              // 启动时核对一次 emb.dim()

// 挤压式 resize 到输入尺寸（不是 letterbox）、RGB、x / 127.5 - 1、NCHW
engine.setInput(0, x.data(), x.size() * sizeof(float));
engine.infer();
std::vector<float> v = emb.postprocess();     // 已 L2 归一化

bcdl::EmbeddingBank bank;                     // 第一次 add() 定下 dim()
bank.add(gallery_vec, "img_0001.jpg");        // 或某个类别名的文本向量
std::vector<bcdl::EmbedMatch> top = bank.search(v, /*k=*/5);
// 纯函数：decodeEmbedding(data, dim, cfg)
```

| 成员 | 说明 |
|---|---|
| `ImageEmbedder(engine, cfg={}, output_index=0)` | 绑定一个输出；下标越界时抛异常。 |
| `ImageEmbedder::postprocess()` | 把输出摊平成 `dim()` 个浮点（必要时反量化），再按 `cfg` 归一化。 |
| `ImageEmbedder::dim()` | 所绑定输出各维之积。 |
| `EmbeddingBank::add(vec, label="")` | 追加一个向量，归一化后存进表里（调用方手里那份不动）。长度与第一条不同时抛 `Error(-1)`。 |
| `EmbeddingBank::search(query, k=5)` | 按余弦从高到低返回最好的 `k` 条；`query` 不必事先归一化。`k <= 0` 或超过 `size()` 时返回全部条目；空表返回空；长度不对时抛异常。 |
| `size()` · `dim()` · `label(i)` | |

**绑定池化的那个子模型。** 嵌入模型的 `.hbm` 通常同时装着整图池化向量**和**逐 patch 特征图，
构造函数默认取排在前面的那个。池化头是 `[1, D]`，给出的正是该编码器的宽度；patch 头是
`[1, N, D]`，摊平后是 `N*D`，`ImageEmbedder` 分不清它和一个特别宽的嵌入 —— 请拿 `dim()`
去对编码器文档写明的宽度。[`examples/embed_demo.cc`](../examples/embed_demo.cc) 用
`Engine::modelNames()` 列出各个名字，再绑定池化的那个。

**要挤压，不要 letterbox。** 这类编码器训练时用的是直接 resize 成正方形；letterbox 的补边是它们
从没见过的像素，会把向量推偏，足以损失检索精度。嵌入向量不带坐标，事后也没有几何要还原。
RGB 上的 `x / 127.5 - 1` 是示例程序喂给 SigLIP 的归一化；换别的编码器之前先查它的 model card，
别想当然。

板上只跑图像编码器。双塔模型的文本编码器在主机上离线跑一次，把一组固定 prompt 的向量做成表
随模型发布，再 `add()` 进来。`search()` 是线性扫描，对板上这种规模的表（几千条）正合适。
SigLIP base/16 每张图 19.3 ms；与浮点模型相比，逐图余弦 0.9977–0.9995，200 张图的图库上
recall@1/@5 与浮点完全相同（0.9250 / 1.0000）。

## 人脸检测与对齐 Face

`bcdl/tasks/face.h` —— 带 5 个关键点的 SCRFD 人脸检测，加上识别所依赖的对齐这一步。识别本身
就是普通的[图像嵌入](#图像嵌入-embedding)：检测 → 对齐到 112×112 → 嵌入 → 比对。

```cpp
struct FaceDetection { float x1, y1, x2, y2, score;
                       std::vector<std::pair<float, float>> landmarks; };  // 5 个，原图像素
struct FaceDetectConfig { float conf_thresh=0.5f, iou_thresh=0.4f; int max_faces=100;
                          std::vector<int> strides={8,16,32}; int num_anchors=2; };

bcdl::Engine det_engine("models/scrfd_10g_nashm_640x640_nv12.hbm");   // 9 个输出
// SCRFD 在右下补边：图像贴在左上角，所以 letterbox 没有 pad
bcdl::LetterboxInfo lb = bcdl::computeLetterbox(w, h, 640, 640, /*centerPad=*/false);
// 按 lb.scale 缩放后放进全零 640x640 画布的左上角，bgrToNv12Cpu，
// feedNv12Input(det_engine, nv12)，det_engine.infer()
bcdl::FaceDetector det(det_engine, bcdl::FaceDetectConfig{}, /*output_base=*/0);
std::vector<bcdl::FaceDetection> faces = det.postprocess(lb);

std::vector<uint8_t> crop = bcdl::alignFace(bgr, w, h, /*src_stride=*/0,
                                            faces[0].landmarks, /*size=*/112);
bcdl::Engine rec_engine("models/arcface_r50_aligned_nashm_112x112.hbm");
// crop -> RGB、(x - 127.5) / 127.5、NCHW float32 -> rec_engine.setInput()、infer()
bcdl::ImageEmbedder rec(rec_engine);          // dim() == 512
bcdl::EmbeddingBank people;
people.add(rec.postprocess(), "alice");
// 纯函数：decodeScrfd(score, bbox, kps, counts, cfg, lb) · similarityTransform(src, dst) · arcFaceTemplate()
```

| 成员 | 说明 |
|---|---|
| `FaceDetector(engine, cfg={}, output_base=0)` | 把 `3 * strides.size()` 个输出按三组绑定 —— 先是全部 score，再是全部 bbox，最后是全部 kps，即参考导出的输出顺序；模型输出不够时抛异常。 |
| `FaceDetector::postprocess(lb)` | 解码所有尺度、过阈值、做一次 NMS；框和关键点都在原图像素上，框截到图像范围内。若 bbox 或 kps 头比 score 头所对应的尺寸小，抛异常。 |
| `alignFace(bgr, src_w, src_h, src_stride, landmarks, size=112)` | 用相似变换把人脸 warp 到缩放成 `size` 的 `arcFaceTemplate()` 上；双线性采样，越界取边缘像素；返回紧凑排布的 `size*size*3` BGR 裁剪。`src_stride` 为 0 表示紧凑排布；关键点不是正好 5 个时抛异常。 |

关键点顺序与模板一致：左眼、右眼、鼻尖、左嘴角、右嘴角 —— “左”指观看者的左边。

**这个检测器在右下补边。** 它的参考预处理把图缩放到能放下，然后贴在左上角，所以它的
`LetterboxInfo` 是 `padX = padY = 0`；用居中 letterbox 会让每个框和关键点整体偏移一个 pad。
检测头是 anchor-free 的距离回归，每个格子 `num_anchors` 个预测，按（行、列、anchor）排列；
格子中心落在**网格点上**，没有半格偏移，距离以 stride 为单位。`num_anchors` 必须能整除每个
张量的长度，否则抛 `Error(-1)`。

**对齐是正确性要求，不是锦上添花。** 识别模型只见过眼睛、鼻子、嘴角都被 warp 到模板位置上的
人脸；直接裁检测框喂进去，嵌入会严重变差，却哪里都不报错。`similarityTransform()` 是闭式的
Umeyama 解（旋转 + 均匀缩放 + 平移），结果确定，也不可能把脸剪切或拉伸。

与浮点参考在相同帧上比：关键点平均 0.35 px、最大 0.89 px，框角点平均 0.52 px；检测 60.2 ms
（SCRFD-10G），对齐 0.29 ms。识别与浮点的余弦平均 0.975（最低 0.966）。ArcFace 的输入行在
设备上是补齐的（448 → 512 字节）；`setInput()` 会把连续数组散布进这个 layout，而在它这么做之前，
光是这层补齐就让嵌入与浮点的余弦只剩 0.015。

## 姿态 Pose

`bcdl/tasks/pose.h` —— LTRB 多尺度；人体框 + K 个关键点。每个尺度三个输出
`(cls, box, kpt)`。

```cpp
struct Keypoint { float x, y, score; };
struct PoseDetection { float x1,y1,x2,y2,score; int class_id;
                       std::vector<Keypoint> keypoints; };
struct PoseConfig { int num_keypoints=17; float conf_thresh, iou_thresh;
                    int max_dets; std::vector<int> strides; };

bcdl::PoseEstimator est(engine, cfg, /*output_base=*/0);
auto poses = est.postprocess(lb);   // 读取 3*strides.size() 个输出
// 纯函数：decodePose(cls, box, kpt, grid_hw, cfg, lb)
```

`num_keypoints` 取自 kpt 张量（最后一维 / 3），不信任 cfg 里填的值。

## 全身姿态 WholeBody

`bcdl/tasks/wholebody.h` —— **自顶向下**：对**每个人**的裁剪各跑一次推理，所以
前面要接一个检测器，开销随人数增长。单个输出，`[1,K,H,W]` 通道优先的热力图。

```cpp
struct WholeBodyCrop { int x1, y1, pad_left, pad_top, padded_w, padded_h; };
struct WholeBodyConfig { float kpt_thresh; int blur_kernel, box_pad;
                         float mean[3], std[3]; };

std::vector<float> in;
auto crop = bcdl::wholeBodyPreprocess(bgr, w, h, stride, x1, y1, x2, y2,
                                      192, 256, cfg, in);   // BGR 进，RGB 出
engine.setInput(0, in.data(), in.size() * sizeof(float));
engine.infer();
bcdl::WholeBodyEstimator est(engine, cfg);
auto kpts = est.postprocess(crop);   // 133 个 Keypoint，原图像素
// 纯函数：decodeWholeBody(heatmaps, num_kpts, hm_h, hm_w, crop, cfg)
```

排布：`0-16` 身体，`17-22` 脚，`23-90` 人脸，`91-111` 左手，`112-132` 右手。
K/H/W 来自张量而不是 cfg。裁剪走的是参考实现的"加宽-补边-缩放"，**不是** mmpose
的仿射；亚像素精修是在每个峰值周围窗口上做的 DARK-UDP。

## 超分 SuperRes

`bcdl/tasks/superres.h` —— 固定 tile 的放大器，通过重叠 tile + 交叉淡化应用到
任意尺寸的图上。

```cpp
struct SuperResConfig { int overlap; };          // 输入像素
struct SrImage { int width, height; std::vector<uint8_t> data; };  // BGR
struct TilePlacement { int x, y; };              // 一个 tile 的原点，输入像素

bcdl::SuperResolver sr(engine, cfg);
auto big = sr.upscale(bgr, w, h, stride);        // 放大 sr.scale() 倍
int tiles = sr.lastTileCount();                  // 上一次调用跑了几个 tile；开销与它成正比
// 纯函数：planTiles(w, h, tile_w, tile_h, overlap) -> std::vector<TilePlacement>、tileWeight(i, len, ramp)
```

`scale()` 与 `tile()` 都从模型形状里读，永远不由配置指定。混合是累加
`w * pixel` 与 `w` 再相除，所以图像边界不需要特判。注意编译出的 `.hbm` 体积随
tile **面积**增长 —— 同一个网络在 256 tile 下是 148 MB、128 下是 37 MB，而每像素
吞吐相同。

`planTiles` 会把每个方向的最后一个 tile 顶到远端边缘，而不是让它悬在图外，所以最后一段重叠
可能比要求的大、但绝不会更小，也没有 tile 会读出图像范围。比一个 tile 还短的边只得到一个
位于 0 的原点：`upscale()` 会自己把这种图按边缘像素补到 tile 大小，直接调 `planTiles` 就得自己补。

## 稀疏特征点 XFeat

`bcdl/tasks/features.h` —— 关键点 + L2 归一化的 64 维描述子，以及互为最近邻匹配。
三个 1/8 尺度输出（`feats` 64 通道、`keypoints` 65 通道、`reliability` 1 通道）。
输入的 InstanceNorm 与每一个数据相关的步骤（softmax / NMS / top-k / 采样）
都被刻意放在 CPU 上。

```cpp
struct Feature { float x, y, score; };
struct FeatureSet { std::vector<Feature> keypoints;
                    std::vector<float> descriptors; int dim; };
struct XfeatConfig { float detection_thresh; int nms_kernel, top_k; };

bcdl::FeatureExtractor ext(engine, cfg);
auto a = ext.extract(bgr_a, w, h, stride);
auto b = ext.extract(bgr_b, w, h, stride);
auto m = bcdl::matchFeatures(a, b, /*min_cossim=*/0.82f);
// 纯函数：xfeatPreprocess(...) / decodeXfeat(feats, kpts, rel, fh, fw, ...)
```

描述子用**双三次**采样（参考实现采样器的默认值），而可靠性图是双线性。
`matchFeatures` 的复杂度是 `O(|a|*|b|*dim)` 并用 OpenMP 并行 ——
`XfeatConfig::top_k` 就是决定一对图花 130 ms 还是 8 ms 的那个旋钮。

## 实例分割 InstanceSeg

`bcdl/tasks/instance_seg.h` —— LTRB 框 + 由 prototype 张量算出的逐实例二值掩膜。
每个尺度三个输出 `(cls, box, mc)`，外加一个 `proto`。

```cpp
struct InstanceMask { float x1,y1,x2,y2,score; int class_id;
                      int mask_w, mask_h; std::vector<uint8_t> mask; }; // 0/1
struct InstanceSegConfig { float conf_thresh, iou_thresh; int max_dets;
                           std::vector<int> strides; int proto_index=9;
                           bool compute_masks=true;
                           int reg_max=0; };   // 0 = 普通 LTRB；InstanceSegmenter 按框头通道数自动设定

bcdl::InstanceSegmenter seg(engine, cfg, /*output_base=*/0);
auto masks = seg.postprocess(lb, orig_w, orig_h);
// 纯函数：decodeInstanceSeg(cls, box, mc, grid_hw, num_classes, num_coef,
//                           proto, proto_h, proto_w, proto_c, cfg, lb, orig_w, orig_h)
```

设 `compute_masks=false` 可只拿框与分数，跳过掩膜组装。

## 开放词汇检测 YOLOE

`bcdl/tasks/open_vocab.h` —— prompt-free 的 YOLOE 导出，词表在导出时就**固定**了：CLIP 文本
嵌入被折进分类卷积里，所以运行时它就是一个普通的 anchor-free 头，原样走现有的类来解码 ——
检测版用 `YoloLtrbDetector`，分割版用 `InstanceSegmenter`（两者都会按通道数自动识别 DFL 框头）。
唯一新增的状态是类别 id → prompt 名的对照表。

```cpp
struct LabelMap { std::vector<std::string> names; };   // + 下表的成员

bcdl::LabelMap labels = bcdl::LabelMap::fromFile("models/yoloe26m_indoor99_labels.txt");
bcdl::InstanceSegConfig cfg; cfg.compute_masks = false;   // 只要框
bcdl::InstanceSegmenter seg(engine, cfg);
for (const auto& m : seg.postprocess(lb, orig_w, orig_h))
  std::printf("%s %.2f\n", labels.name(m.class_id).c_str(), m.score);

// 检测版：YoloLtrbConfig::num_classes 必须等于 labels.size()
```

| 成员 | 说明 |
|---|---|
| `LabelMap::fromFile(path)` | 每行一个标签，按类别 id 顺序；去掉首尾空白、丢弃空行。文件打不开时抛 `std::runtime_error`。 |
| `LabelMap::fromList(names)` | 从内存里的列表构建。 |
| `size()` · `name(id)` | id 越界时 `name()` 返回 `"?"`，不抛异常。 |

`.hbm` 和它的标签文件要放在一起：每个类别 id 是什么意思，只有这个文件记着；而且**换词表就得
重新导出** —— `fromList()` 只是给模型的类别起名字，并不改变模型检测什么。随库的版本是 COCO-80
类的 YOLOE-11s（检测版和分割版），以及固定 99 类室内词表的 YOLOE-26m（分割版；类名在
`yoloe26m_indoor99_labels.txt`）。

室内版是零样本的 —— 一点都没训练过。在留出的 Open Images 室内图上，它在至少有 20 个框的 25 个
类上 mAP50 为 0.484（浮点 0.478）：整体与有监督检测器持平，但不是每一类都持平。显示器是盲区；
导航类（楼梯、电梯、扶梯）分数偏低，阈值要设得比 0.25 低。只需要框时把 `compute_masks` 关掉 ——
组装掩膜花的 CPU 时间远多于网络在 BPU 上的耗时。详见
[MODELS.zh.md](MODELS.zh.md#室内-yoloe-build-是零样本的)。

## 可提示分割 EdgeSAM

`bcdl/tasks/promptable_seg.h` —— SAM 家族掩膜解码器（EdgeSAM、MobileSAM、SAM）的尾段：从候选
掩膜里挑最好的一张并二值化。整个会话由两个模型组成 —— 每张图跑一次的图像编码器，和每次点击
都对着缓存的 embedding 跑一次的轻量提示解码器。

```cpp
struct SamConfig { float mask_threshold = 0.0f; bool multimask = true; };
struct SamMask { int index; float iou; int mask_w, mask_h;
                 std::vector<uint8_t> mask; };           // Hm*Wm，0/1，低分辨率

bcdl::Engine enc("models/edge_sam_encoder_nashm.hbm");
bcdl::Engine dec("models/edge_sam_decoder_sp1_nashm.hbm");   // 单点；_bp2 = 框
// enc：长边缩放到 1024、右下补边、RGB、原始 0-255 -> 每张图 infer 一次
// dec：缓存的 embedding、该 1024 画布上的提示坐标、SAM 标签 -> infer
bcdl::SamMaskDecoder tail(dec, bcdl::SamConfig{}, /*masks_index=*/0, /*iou_index=*/1);
bcdl::SamMask m = tail.postprocess();
// 纯函数：decodeSamMasks(low_res_logits, num_masks, mask_h, mask_w, iou_pred, cfg)
```

| 成员 | 说明 |
|---|---|
| `SamMaskDecoder(engine, cfg={}, masks_index=0, iou_index=1)` | 绑定解码器的掩膜 logit 输出（`[1,N,Hm,Wm]` 或 `[N,Hm,Wm]`）和 IoU 输出（`[1,N]` 或 `[N]`）。请传你这个 build 的实际下标；掩膜输出是 4 维的那个。 |
| `SamMaskDecoder::postprocess()` | 对当前输出调用 `decodeSamMasks()`；下标越界或掩膜张量维数不对时抛异常。 |

`decodeSamMasks` 在 `multimask` 打开时取预测 IoU 最大的候选（否则取第 0 个；`iou_pred` 为空时
也取第 0 个），并保留 `logit > mask_threshold` 的像素 —— SAM 用的 0.0 对原始 logit 而言就是
sigmoid > 0.5。掩膜停留在解码器的低分辨率上（SAM 是 256×256）；去掉补边、缩放回原图由调用方负责。

**C++ 里只有这个尾段。** 两段式会话是 Python 的 `SamSession`，喂模型的方式以它为准：编码器吃的图
长边缩放到 1024、右下补边、BGR→RGB、原始像素（ImageNet 归一化已编译进编码器）；解码器的输入
0、1、2 依次是 embedding、缩放到该画布上的提示坐标、SAM 标签（前景点为 1，框的两个角为 2 / 3）。
每个解码器 build 的提示点数是固定的：`_sp1` 单点，`_bp2` 两个角点构成的框。

**随库解码器的 IoU 头被 int8 量化坏了**，所以 `SamSession` 不信它，改取面积最大的候选。
`SamMaskDecoder` 没有按面积挑的规则 —— `multimask = true` 时信 IoU 头，`multimask = false` 时
取第 0 个 —— 想按面积挑，就用 `outputAsFloat()` 读出 logit 自己选。

## 旋转框 OBB

`bcdl/tasks/obb.h` —— LTRB + 角度，旋转 IoU NMS。每个尺度三个输出
`(cls, box, angle)`。

```cpp
struct RotatedBox { float cx, cy, w, h, angle; };      // angle 单位为弧度
struct ObbDetection { RotatedBox rrect; float score; int class_id; };
struct ObbConfig { int num_classes=15; float conf_thresh, iou_thresh; int max_dets;
                   std::vector<int> strides; bool regularize=true;
                   float angle_offset_rad=0; int angle_sign=1; };

bcdl::ObbDetector obb(engine, cfg, /*output_base=*/0);
auto dets = obb.postprocess(lb);
float i = bcdl::rotatedIoU(a, b);
std::vector<int> keep = bcdl::rotatedNms(dets, cfg.iou_thresh, cfg.max_dets);
// 纯函数：decodeObb(cls, box, angle, grid_hw, cfg, lb)
```

## 语义分割 Segmentation

`bcdl/tasks/segmentation.h` —— 对 logit 张量取 argmax（或直接透传 id），
得到逐像素标签图。

```cpp
struct SegConfig { int num_classes=0; bool channels_first=true; bool argmaxed=false; };
struct SegMask { int width, height, num_classes; std::vector<int32_t> labels; };

bcdl::Segmenter seg(engine, cfg, /*output_index=*/0);
bcdl::SegMask m = seg.postprocess();
std::vector<uint8_t> bgr = bcdl::segColorize(m);   // (H*W*3) BGR 调色板
// 纯函数：decodeSeg(data, shape, cfg)
```

`num_classes=0` 表示从张量推断 C；模型已经输出 id 时设 `argmaxed=true`。

## 单目 3D 检测 SMOKE

`bcdl/tasks/mono3d.h` —— 一张相机图出 KITTI 风格的 3D 框（Car、Cyclist、Pedestrian）。
CenterNet 式：类别热力图定位每个物体投影后的 3D 中心，该像素处的 8 通道回归图再配合相机内参，
把它抬成完整的 3D 框。

```cpp
struct CameraIntrinsics { float fx, fy, cx, cy; };   // 原图（未缩放）的内参，无 skew
struct Mono3dBox { int class_id; float score; float x, y, z, h, w, l, yaw, alpha;
                   std::array<float, 4> box2d; };    // 相机系，米；box2d 为原图像素
struct Mono3dConfig { int num_classes=3; float conf_thresh=0.25f; int max_dets=50;
                      int nms_kernel=3; bool pred_2d=true;
                      std::array<float, 2> depth_ref;               // {28.01, 16.32}
                      std::vector<std::array<float, 3>> dim_ref; }; // 每类的尺寸先验

// 输入：[1,3,384,1280] F32 RGB，/255 + ImageNet mean/std（没有融进 .hbm），
// 用与解码相同的“按宽缩放”映射做 warp
bcdl::Mono3dDetector det(engine, bcdl::Mono3dConfig{}, /*output_base=*/0);  // 先 cls，后 reg
bcdl::CameraIntrinsics K{fx, fy, cx, cy};            // 来自你的标定
std::vector<bcdl::Mono3dBox> boxes = det.postprocess(orig_w, orig_h, K);
// 纯函数：computeMono3dFeatureXform(origW, origH, featW, featH) · decodeMono3d(cls, reg, H, W, cfg, featXform, K)
```

框在 KITTI 相机坐标系里 —— +x 向右、+y 向下、+z 向前，单位米 —— `(x, y, z)` 是框的**底面**
中心；`yaw` 是绕相机 Y 轴的 rotation_y，`alpha` 是观测角，`box2d` 是用 `K` 投影 3D 框、再截到
图像范围内得到的 2D 框（`pred_2d`）。`Mono3dConfig` 的默认值就是参考 SMOKE 的配置。

**几何不是 letterbox。** SMOKE 按图像**宽度**缩放、高度居中，竖直方向是裁剪或补边：
`scale = featW / origW`、`padX = 0`、`padY = featH/2 − scale·origH/2`，可能为负。
`computeMono3dFeatureXform()` 在特征图分辨率上构建这个映射，384×1280 的输入 warp 是同一个变换
放大 4 倍。预处理与解码必须一致，而且 `K` 始终是原图的内参 —— 缩放完全由这个变换承担，
永远不折进 `K`。

两个输出都是**通道优先的原始 logit**（`[1, nc, 96, 320]` 与 `[1, 8, 96, 320]`，stride 4）：
predictor 的激活已从编译图里剥掉，由解码重新施加 —— 热力图上做截断的 sigmoid，三个尺寸通道做
`sigmoid − 0.5`，`(sin, cos)` 做 L2 归一化 —— 然后是 CenterNet 的 max-pool NMS 与 top-`max_dets`，
与 SMOKE 的后处理一致。`num_classes` 取自 cls 张量，从不信任 `cfg`；`dim_ref` 的行数必须覆盖它；
reg 头不足 8 通道或网格不同时抛 `Error(-1)`，而不是解出一堆像模像样的垃圾框。

## 激光雷达 3D 检测 Lidar 3-D detection

`bcdl/preproc/lidar_voxelize.h` + `bcdl/tasks/lidar_det3d.h` —— 点云到带朝向的 3D 框,
形态是「多段编译模型 + 段间 CPU 处理」的流水线。

```cpp
struct Box3D { float x,y,z, dx,dy,dz, heading; int label; float score; };

bcdl::VoxelConfig vcfg;                                   // KITTI 几何
bcdl::Pillars p = bcdl::voxelizePillars(points, n, /*stride=*/4, vcfg);
std::vector<float> x = bcdl::pillarPointFeatures(p.voxels.data(), p.num_points.data(),
                                                 p.coords.data(), p.num_pillars, vcfg);
// vfe engine -> 柱特征
std::vector<float> canvas = bcdl::scatterPillars(feats, p.coords.data(),
                                                 p.num_pillars, 64, vcfg);
// backbone+head engine
bcdl::LidarDetector3d det(head_engine);
std::vector<bcdl::Box3D> boxes = det.postprocess();
// 纯函数：generateAnchors3d(cfg) · decodeDet3d(...) · bevIou(a,b) · nms3d(...)
```

点必须已经在**自车系**;`VoxelConfig` 里的上限是会真的截断的 —— `Pillars` 会把丢掉的每个点
和每个列都报出来,而不是默默截断。

`pillarPointFeatures()` 只有**重导成 `(1, C, pillars, points)` 上 1×1 卷积**的那种 VFE 才需要;
吃三个原始柱张量的 VFE 把同样的展开做在自己图里。把它搬出来值 179 倍:那个形状是
「巨大柱维 + 4 宽通道维」,图内那版为一个 95 MFLOP 的 MLP 花 290 ms,卷积形态只要 1.62 ms。

anchor 解码放 CPU 是刻意的:官方 head 把它烘进图里用的是 `ScatterND`,那东西**编译不报错、
产物却从不写输出**。`generateAnchors3d` 与该模型自带的 anchor 常量吻合到 8e-06 ——
顺序(类优先再朝向、且在区间上取**闭区间** linspace)是这么定下来的,不是猜的。

## 异常检测 Anomaly detection（单类）

`bcdl/tasks/anomaly.h` —— 只用某产品的**正常**图训练,输出每个 patch「有多不像」。
模型是按产品的、阈值是产线级决策,所以随库的模型是**示例**而不是通用检测器。

```cpp
struct AnomalyConfig { int out_width=0, out_height=0; int blur_kernel=33;
                       float blur_sigma=4.0f, threshold=0.5f; };
struct AnomalyMap { int width, height; std::vector<float> data; float score; };

bcdl::AnomalyDetector det(engine);
bcdl::AnomalyMap m = det.detect(bgr, width, height, stride);
std::vector<uint8_t> mask = bcdl::anomalyMask(m, 0.5f);
// 纯函数：decodeAnomaly(data, shape, cfg) · anomalyPreprocess(bgr, ..., out)
```

`score` 取图的**最大值** —— 零件有多坏取决于最坏那一块,取均值会让大片干净区域盖掉小缺陷。
上采样到输入分辨率与高斯平滑都属于参考后处理;`blur_kernel = 0` 可省掉平滑
(288×288 下约 6 ms),但数值会变。

`anomalyPreprocess` 做 BGR→RGB、`/255`、ImageNet mean/std —— 这套归一化是训练契约的一部分,
默认值就是那组统计量。

## 光流 Optical flow

`bcdl/tasks/optical_flow.h` —— 两帧转稠密位移场。向量单位是源图像素,+u 向右 / +v 向下(OpenCV 约定)。

```cpp
struct FlowConfig { float scale_x=1, scale_y=1; bool channels_first=true; };
struct FlowField { int width, height; std::vector<float> data; /* H*W*2, (u,v) */ };

bcdl::OpticalFlowEstimator est(engine);
bcdl::FlowField f = est.estimate(bgr0, bgr1, width, height, stride);
std::vector<uint8_t> vis = bcdl::flowColorize(f);            // BGR H*W*3
float epe = bcdl::flowEndpointError(f, reference);
// 纯函数：decodeFlow(data, shape, cfg) · flowPreprocess(bgr, ..., out)
```

`estimate()` 从 Engine 读模型输入尺寸,缩放两帧,并把返回向量从模型像素换算回源图像素 ——
留在模型像素的场会让后续 warp 整体位移量偏掉,而且**看不出哪里不对**。

**预处理保持 BGR 与 0-255**:参考实现的 /255 在图内部。`flowPreprocess` 与 Python 路径绑定到同一份
C++ 实现,保证两条路喂给模型的像素完全一致。

## 端到端驾驶 End-to-end driving（相机 + 激光雷达 → 轨迹）

`bcdl/tasks/e2e_drive.h` + `bcdl/preproc/drive_inputs.h` —— 三路前视相机 + 一帧激光雷达进去，
一条规划好的自车轨迹出来（外加检出车辆与 BEV 语义图）。自车系、米、x 向前 / y 向左；
位姿是**未来**的固定间隔采样。

```cpp
struct DrivePose  { float x, y, heading; };
struct DriveAgent { float x, y, heading, length, width, score; };
struct DriveResult { std::vector<DrivePose> trajectory;
                     std::vector<DriveAgent> agents; bcdl::SegMask bev; };

bcdl::DrivePlanner planner(engine, anchors, num_modes, num_poses);

std::vector<float> camera = bcdl::stitchDriveCameras(l, lw, lh, ls, f, fw, fh, fs,
                                                     r, rw, rh, rs);   // (3,H,W) RGB [0,1]
bcdl::LidarBev bev = bcdl::lidarBevHistogram(points, n, /*stride_floats=*/3);

bcdl::DriveStatus st; st.driving_command = 0; st.velocity_x = 5.0f;
bcdl::DriveResult res = planner.plan(camera, bev.data, st);
// 纯函数：decodeDrive(traj, traj_shape, agents, agent_shape, labels, label_shape,
//                    bev, bev_shape, cfg)
```

BEV 栅格**第 0 轴是 x（前向）**、第 1 轴是 y —— 与上游直方图同序。
`LidarBevConfig::use_ground_plane=false`（默认）只出一个通道，装高于 `split_height` 的点。

`DriveResult::bev` 的轴**不是**屏幕轴：行号随**前方距离**增大（row 0 是自车端），
列号随**向左**增大，按「上=前方」画要把两个轴都反过来。上游两个约定都没写，
而直路在旋转 180° 下近似对称 —— **画反了看不出来**。解码原样返回模型输出，不替调用方拿主意。

`DriveAnchorConfig::noise_scale = 0` 只把采样器的随机项清零、保留调度自身的缩放，
这是与另一个实现做数值对比时该用的配置。`DriveIoConfig` 负责张量下标映射，
输出下标设成 `-1` 即跳过该头。

输入经 `setInputFromFloat`（`bcdl/backend/input_writer.h`）写入，按模型声明的 dtype 量化 ——
这个模型的输入是带 per-tensor scale 的 int16，直接塞 float 原始字节它会**照常跑，只是在跑噪声**。

## 深度 Depth

`bcdl/tasks/depth.h` —— 单通道深度/视差转浮点图。

```cpp
struct DepthConfig { int width=0, height=0; bool normalize=true;
                     float clip_lo=0, clip_hi=0; };
struct DepthMap { int width, height; std::vector<float> data; float vmin, vmax; };

bcdl::DepthEstimator est(engine, cfg, /*output_index=*/0);
bcdl::DepthMap dm = est.postprocess();
std::vector<uint8_t> g8  = bcdl::depthToGray8(dm);   // (H*W)
std::vector<uint8_t> bgr = bcdl::depthColorize(dm);  // (H*W*3) Turbo BGR
// 纯函数：decodeDepth(data, shape, cfg)
```

## 深度精修 Depth refinement（RGB-D）

`bcdl/tasks/depth_refine.h` —— 输入是**已有的**带噪、有空洞的深度图加上对齐的
RGB 帧，输出补全后的米制深度 + 可信掩膜（LingBot-Depth）。和单目/双目两个头
不同，它做的是精修而非估计。

```cpp
struct DepthRefineConfig { int encoder_height=420, encoder_width=560;
                           float min_valid_depth=0.01f, mask_threshold=0.5f;
                           bool apply_mask=true; };
struct RefinedDepth { int width, height; std::vector<float> depth;
                      std::vector<uint8_t> mask; float vmin, vmax; };
struct Intrinsics { float fx, fy, cx, cy; };

bcdl::DepthRefiner refiner(engine);            // 编码器网格从模型里读
bcdl::RefinedDepth r = refiner.run(bgr, w, h, stride, depth_m, depth_stride);

bcdl::Intrinsics k = bcdl::scaleIntrinsics(sensor_k, w, h, r.width, r.height);
std::vector<float> pts = bcdl::depthToPointCloud(r, k);   // (H*W*3) 米

// 纯函数部分，想自己喂 engine 时用：
//   preprocessRefineImage(bgr, w, h, stride, cfg, &image_tensor)   // [3,eh,ew]
//   preprocessRefineDepth(depth_m, w, h, stride, cfg, &log_tensor) // [1,eh,ew]
//   depthU16ToMetres(raw16, w, h, stride, /*scale=*/1000.f)
//   decodeRefinedDepth(depth, mask_logit, h, w, cfg)
```

engine 的两个输入必须是 F32 featuremap（否则构造函数抛异常）。深度以 log 深度、
0 表示"没有读数"的形式进图；部署图保留全部深度 token，而上游会丢掉空 patch 的
token —— 代价见 [`MODELS.zh.md`](MODELS.zh.md) 与头文件。

## OCR

`bcdl/tasks/ocr.h` —— 三段互相独立的阶段（det / cls / rec）；由应用来组合
（裁出每个检测框、按需 180° 翻转、再识别）。每一段都有一个纯解码器加一个
绑定 Engine 的包装。默认走 PP-OCRv6，PP-OCRv5 保留为回退。

```cpp
// 识别（CRNN + CTC）：
std::vector<std::string> dict = bcdl::loadCharDict("ppocr_dict.txt"); // blank 在 0
struct RecResult { std::string text; float score; };
bcdl::TextRecognizer rec(engine, "ppocr_dict.txt", /*out_idx=*/0);
bcdl::RecResult r = rec.postprocess();
// 纯函数：decodeCtc(logits, num_steps, num_classes, dict)

// 方向（0°/180°）：
struct ClsDirResult { int label; float score; bool flip180; };
bcdl::TextAngleClassifier cls(engine, /*thresh=*/0.9f, /*out_idx=*/0);
// 纯函数：decodeClsDir(logits, n, thresh)

// 检测（DBNet，纯 C++ 连通域 + unclip）：
struct DbConfig { float bin_thresh=0.3, box_thresh=0.6, unclip_ratio=1.5;
                  int min_size=3, connectivity=8; };
struct TextBox { float pts[8]; float x1,y1,x2,y2; float score; }; // 四点 + 外接框
bcdl::DbTextDetector det(engine, DbConfig{}, /*out_idx=*/0);
std::vector<bcdl::TextBox> boxes = det.postprocess(lb);
// 纯函数：decodeDbnet(prob, H, W, cfg, lb)
```

完整的 det→cls→rec 串接见 [`examples/ocr_demo.cc`](../examples/ocr_demo.cc)
（裁剪顺序与字典约定都在那里处理了）。

---

## 语音链

唤醒词 → 端点检测 → 识别 → 指令匹配 → 语音回复，网络全在 BPU：
[`examples/voice_assistant.cc`](../examples/voice_assistant.cc) 是整个流程的 C++ 写法
（读 WAV、回复写成 WAV；不给输入时先合成一段演示录音）。音频格式、唤醒词 token 和调参见
[语音链总览](API.zh.md#语音链总览)；各个类的说明在下面。

## 语音识别（Paraformer）

`bcdl/preproc/fbank.h` 是与 Kaldi 兼容的前端；`bcdl/tasks/asr.h` 提供 CIF、token 解码和三段式的
`ParaformerAsr`。输入一段 16 kHz 单声道语音，最长 `maxSeconds()`（随库模型为 24 秒），输出文字。

```cpp
bcdl::Engine enc("models/paraformer_encoder_int16.hbm");
bcdl::Engine pred("models/paraformer_predictor_int16_v2.hbm");
bcdl::Engine dec("models/paraformer_decoder_int16_v2.hbm");
bcdl::ParaformerAsr asr(enc, pred, dec,
                        bcdl::loadTokenList("models/paraformer_tokens_8404.json"),
                        bcdl::loadKaldiCmvn("models/paraformer_am.mvn"));
bcdl::AsrResult r = asr.transcribe(pcm, num_samples);   // float，范围 [-1, 1]
// r.text, r.token_ids, r.feature_frames, r.truncated

bcdl::Fbank fbank;                                         // FbankConfig{} = 16 kHz, 80 mel
std::vector<float> feats;
int frames = fbank.compute(pcm, num_samples, feats);       // [帧数, 80]
// 纯函数: lfrStack(feats, frames, dim, lfr_m, lfr_n, out) · parseKaldiCmvn(text)
// 纯函数: cifIntegrate(alphas, hidden, frames, dim, valid_frames, cfg, embeds)
// 纯函数: argmaxTokens(logits, rows, vocab) · tokensToText(ids, vocab)
```

| 成员 | 说明 |
|---|---|
| `transcribe(pcm, samples, timeout_ms)` | 前端 + 三段 BPU + CIF。超过窗口长度的部分截掉，并置 `truncated` |
| `transcribeFeatures(feats, frames, timeout_ms)` | 从 `[帧数, featureDim()]` 的 LFR+CMVN 特征开始 |
| `features(pcm, samples, out)` | 只跑前端 |
| `maxFrames()` · `maxSeconds()` · `featureDim()` · `maxTokens()` · `vocabSize()` | 从 Engine 读出的模型尺寸 |

`ParaformerAsr` 按张量形状、而不是按下标去对应三段模型；token 表大小和 decoder 的类别数对不上时会直接拒绝。
`cifIntegrate` 把 `CifConfig::tail_threshold` 放在第 `valid_frames` 帧，也就是语音真正结束的位置。
定长的 predictor 图只能把它加在补零区的末尾，而补零区随后会被屏蔽掉，尾部权重也跟着没了，
于是凡是最后一个 token 要靠尾部权重才能发放的句子，都会丢掉它。


### 语音端点检测：`bcdl/tasks/vad.h`

```cpp
bcdl::Engine eng("models/silero_vad_16k_int16_nashm.hbm");
bcdl::SileroVad vad(eng);                     // 有状态：一个对象对应一路音频流
float p = vad.probability(chunk);             // 512 个采样点（32 ms）-> 语音概率
bcdl::VadSegmenter seg(vad);                  // VadConfig{}：阈值 0.5，开 250 ms，关 300 ms
for (auto& s : seg.feed(pcm, n))              // 任意长度；返回本次结束的语句
  asr.transcribe(s.pcm.data(), s.pcm.size()); // s.start / s.end：自 reset() 起的采样点
for (auto& s : seg.flush()) { /* 流结束 */ }
```

| 成员 | 说明 |
|---|---|
| `SileroVad::probability(chunk, timeout_ms)` · `reset()` | `kChunk` = 512；内部保存 LSTM 状态和上一块的最后 64 个采样点 |
| `VadSegmenter::feed(pcm, n)` · `flush()` · `segments(pcm, n)` | 返回 `VadSegment{start, end, pcm}`；`segments` = reset + feed + flush |
| `inSpeech()` · `lastProbability()` · `config()` | |
| `VadConfig` | `threshold`、`neg_threshold`（< 0 时等于 threshold）、`min_speech_ms`、`min_silence_ms`、`speech_pad_ms`、`max_speech_s` |

模型自带的 STFT（4 帧 256 点 Hann 窗，帧移 128）在 CPU 上算，BPU 只接收幅度谱：在 BPU 上做平方，
功率值域约 5000，int16 会丢掉所有小幅值频点。概率连续 `min_speech_ms` 不低于 `threshold` 时开始一段，
连续 `min_silence_ms` 低于 `neg_threshold` 时结束；一段达到 `max_speech_s` 就在此切开，接着开新的一段。


### 唤醒词检测：`bcdl/tasks/kws.h`

```cpp
bcdl::Engine enc("models/kws_zipformer_zh_en_3m_c16_int16_nashm.hbm");
bcdl::KeywordSpotter kws(enc, "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                         "models/kws_zipformer_zh_en_3m_tokens.txt");   // KwsConfig{}
kws.setKeywords("n ǐ h ǎo x iǎo ch ē @你好小车\n"
                "x iǎo ch ē x iǎo ch ē :1.5 #0.3 @小车小车\n");    // 或 loadKeywords(path)
for (const bcdl::KwsDetection& d : kws.feed(pcm, n))     // 任意长度，浮点 [-1, 1]
  printf("%s at %.2f s\n", d.keyword.c_str(), d.end);
```

| 成员 | 说明 |
|---|---|
| `setKeywords(text)` · `loadKeywords(path)` · `keywords()` | 每行一个唤醒词：模型 token、可选 `:boost` 和 `#threshold`、`@短语`；会重置音频流 |
| `feed(pcm, n, timeout_ms)` | 返回 `KwsDetection{keyword, index, start, end, tokens}`，时间为自 `reset()` 起的秒数 |
| `flush(timeout_ms)` | 流结束：用静音把最后不满一块的音频推过去，返回检测结果并重置 |
| `reset()` · `encode(pcm, n, frames)` · `tokenId(tok)` · `vocabSize()` | `encode` 返回整段录音的编码器输出（测试用） |
| `KwsConfig` | `keywords_score`（1.0）、`keywords_threshold`（0.25）、`num_trailing_blanks`（1）、`max_active_paths`（4），与 sherpa-onnx 默认值相同 |

编码器在 BPU 上每次处理 160 ms，调用之间带着 38 个缓存张量；解码器、joiner 和关键词搜索
（在唤醒词的 Aho-Corasick 图上做束搜索）在 CPU 上，逐步照搬 sherpa-onnx 的 KeywordSpotter：
同一套搜索的浮点版在 87 段核对片段上与 sherpa-onnx 的检测结果完全一致。特征是 Kaldi fbank，Povey 窗、采样值取 [-1, 1]（`FbankConfig::window`、`input_scale`）。


### 语音指令：`bcdl/tasks/command.h`

```cpp
bcdl::CommandMatcher m({"前进", "后退", "左转", "右转", "打开灯", "关灯"});  // max_distance 0.5
bcdl::CommandMatch c = m.match(r.text);   // "大开灯" -> index 4，distance 0
if (c.index >= 0) { /* c.command */ }
// 纯函数: toPinyin(text) -> std::vector<std::string>，每个汉字一个无声调音节
```

| 成员 | 说明 |
|---|---|
| `match(text)` | 返回 `CommandMatch{index（-1 = 拒绝）, nearest, distance, command}` |
| `distances(text)` | 到每条指令的归一化音节编辑距离 |
| `commands()` · `size()` · `maxDistance()` | |

距离 = 音节编辑距离 ÷ 指令音节数；距离相同时取长度更接近的，再取靠前的。英文单词只和英文单词比。
在 40 条指令上，把 Paraformer 的整句正确率从 84.7% / 82.8%（干净 / 5 dB）提到匹配正确 96.9% / 95.6%，
600 句新闻零误接受。拼音表（U+3400..U+9FFF）编译进库，由 `scripts/gen_pinyin_table.py` 生成。


### 语音合成后端：`bcdl/preproc/istft.h`、`bcdl/tasks/vocoder.h`

```cpp
bcdl::Engine eng("models/vocos_16k_t256_int16_nashm.hbm");
bcdl::VocosVocoder voc(eng);                       // VocosConfig{}：context 32，pad -16
std::vector<float> pcm;
int n = voc.synthesize(mel, frames, pcm);          // mel [80, frames] 行主序 -> 16 kHz PCM
// 纯函数: Istft(IstftConfig{}).compute(real, imag, frames, out)        // 都是 [frames, 513]
// 纯函数: Istft::computePolar(mag, cos, sin, frames, out)               // 都是 [513, frames]
```

| 成员 | 说明 |
|---|---|
| `synthesize(mel, frames, pcm, timeout_ms)` | 按 `chunkFrames()` 分块、每边重叠 `context`；返回 `(frames - 1) * hop` |
| `numMels()` · `chunkFrames()` · `bodyFrames()` · `sampleRate()` | 从 Engine 读出的几何参数 |

Engine 必须按顺序输出 `log_mag` 和 `phase`（`[1, 513, T]`）；`exp`、幅度截断、`cos`、`sin`
和 iSTFT 都在 CPU 上做。`context >= 27` 时块之间的拼接精确；只有整句两端用 `pad_value`
代替网络自身的零填充。


### 语音合成：`bcdl/tasks/tts.h`

```cpp
bcdl::Engine enc("models/matcha_zh_en_enc_l128_int16_nashm.hbm");
bcdl::Engine dec("models/matcha_zh_en_dec_t512_int16_nashm.hbm");
bcdl::Engine voc_eng("models/vocos_16k_t256_int16_nashm.hbm");
bcdl::MatchaAcoustic am(enc, dec, "models/matcha_zh_en_embedding.npy");  // MatchaConfig{}
bcdl::VocosVocoder voc(voc_eng);
bcdl::TextToSpeech tts(bcdl::MatchaFrontend("models/matcha_zh_en_tokens.txt",
                                            "models/matcha_zh_en_lexicon.txt"),
                       am, voc);
bcdl::TtsResult r = tts.synthesize("好的，马上出发。");   // r.pcm，16 kHz
```

| 成员 | 说明 |
|---|---|
| `MatchaFrontend::encode(text)` | 每句一个 id 向量；与 sherpa-onnx 前端一致 |
| `MatchaFrontend::readNumbers(text)` | 把数字读成汉字（静态函数） |
| `MatchaAcoustic::synthesize(ids, n, timeout_ms)` | 返回 `MatchaResult{mel [80, frames], frames, durations, truncated}` |
| `MatchaAcoustic::seed(s)` · `maxTokens()` · `maxFrames()` · `numMels()` · `vocabSize()` | |
| `TextToSpeech::synthesize(text, timeout_ms)` | 返回 `TtsResult{pcm, sample_rate, sentences, truncated}`；停顿按 `TextToSpeechConfig::silence_scale`（0.2）压缩 |
| `scaleSilence(pcm, sample_rate, scale, min_pause_s, level)` | 纯函数；不短于 `min_pause_s` 的停顿保留 `scale` 倍（sherpa-onnx 的规则） |

编码器吃的是词嵌入而不是 id：查表在 CPU 上做，表是 `.npy`（`[vocab, D]`，已预先缩放）。时长为
`ceil(exp(logw)) * length_scale`，在 CPU 上按 Matcha 的 `generate_path` 精确展开；解码器的噪声来自
带种子的 `std::mt19937`。

---

## JPEG JPU

`bcdl/media/jpeg_codec.h` —— JPU 上的硬件 JPEG。Move-only；在一条流上请复用。

```cpp
bcdl::JpegEncoder enc(width, height, /*quality=*/50,
                      HB_VP_IMAGE_FORMAT_NV12);   // width%16==0, height%8==0
std::vector<uint8_t> jpg = enc.encode(src_vpimage);

bcdl::JpegDecoder dec(/*outFormat=*/HB_VP_IMAGE_FORMAT_NV12);
bcdl::VpImage img = dec.decode(jpg.data(), jpg.size());  // 拥有所有权的 NV12 VpImage
```

`encode()`/`decode()` 都从编解码器内部缓冲里拷出来，所以结果可以放心保存。
**请复用 `JpegDecoder`** —— 每次调用都新建要花约 5 ms。

## 视频 VPU

`bcdl/media/video_codec.h` —— VPU 上的硬件 H.264 / H.265。

```cpp
bcdl::VideoEncConfig ec;                 // type, width(%32), height(%8),
ec.type = HB_VP_VIDEO_TYPE_H264;         // bitrate_kbps, framerate, intra_period,
ec.width = 1280; ec.height = 720;        // format
bcdl::VideoEncoder enc(ec);
std::vector<uint8_t> chunk = enc.encode(frame_vpimage);  // 可能为空（被缓冲了）

bcdl::VideoDecConfig dc; dc.type = HB_VP_VIDEO_TYPE_H264;  // 或 _H265；另有 format、in_buf_size
bcdl::VideoDecoder dec(dc);
bcdl::VpImage out;
if (dec.decode(nal_data, nal_size, out)) { /* 帧就绪（NV12） */ }
// 否则解码器还在缓冲（重排序 / 参考帧）
```

**解码器**建立在 `media_codec`（`hb_mm_mc_*`）流式 API 之上，它**把喂入与取出解耦
并正确处理重排序** —— 这是 H.265 必需的（逐 AU 的"解码 → 立刻拿这一帧"模型在
重排序流上会超时）。两种驱动方式：

- `decode(data, size, out) -> bool` —— 便捷式：喂一个访问单元，短暂等待一帧。
  适合低延迟 H.264；尾部的重排序帧需要 `flush()`。
- **解耦式**（喂入线程 + 取出线程，`AsyncVideoDetectionPipeline` 就这么用）：
  `feed(data, size)` 入队一个 AU；`receive(out, timeout_ms)` 按显示顺序取一帧
  （`timeout_ms=0` 为非阻塞）；`feedEndOfStream()` 之后用 `flush(out)` 排空
  重排序尾巴。

**编码器**同样有两种用法：`encode(frame)` 喂一帧 NV12 并短暂等待一个码流包（返回的是
*这一次*调用产出的字节，可能为空）；处理一条流时，`feed(frame, pts_us = 0) -> bool`
把一帧排进队列、不等输出，`receive(out, timeout_ms = 0) -> bool` 取一个包，最后一帧之后
反复调用 `flush(out)` 直到返回 `false`，它会发出流结束信号并取出剩余的包
（`feedEndOfStream()` 只排入结束标记，可重复调用）。每次 `feed()` 之后都要取包；
它因输入队列满而返回 `false` 时，先取包再重试，别当成致命错误 —— 生成一帧 1080p
比编码一帧便宜得多，生产者很容易跑在编码器前面。

> **喂进去的必须是一个访问单元，不是任意字节。** 解码器运行在
> `MC_FEEDING_MODE_FRAME_SIZE`：每次 `feed()`/`decode()` 必须正好携带一张图
> 的、带起始码前缀的 NAL。（`AsyncVideoDetectionPipeline` 替你做了这个重组 ——
> 它的 `submit()` 吃任意 Annex-B 字节。）旧的 `STREAM_SIZE` 模式虽然接受任意
> 分片，但会从编解码器内部把堆写坏。

> **只接受 Annex-B 裸流 —— MP4 不是。** MP4/MOV 里的 NAL 是 AVCC 形式
> （4 字节长度前缀、没有起始码），SPS/PPS 藏在 `avcC` box 里，所以 AU 切分器
> 什么都找不到，解码器会悄无声息地一帧不出。请先解封装，且不碰像素：
> `ffmpeg -i in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 -`
> （H.265 用 `hevc_mp4toannexb` / `-f hevc`）。`examples/video_det_demo.py` 里的
> `load_annexb()` 做的正是这件事；VPU 仍然是唯一的解码器。

> **流中途销毁解码器是安全的，但仅仅因为析构函数会把它排空。**
> `hb_mm_mc_stop()` 会一直阻塞到编解码器的 `vdec_render` 组件清空其输出端口，
> `hb_mm_mc_flush()` 同理 —— 而只有应用才能清空它，所以对一个还攥着已解码帧的
> 解码器直接 `stop()` 会永远挂住。`~VideoDecoder` 会先把每个待处理的输出缓冲
> 还回去。任何直接驱动 `hb_mm_mc_*` 的代码都必须这么做。

> **H.265 注意（分层 GOP）：** 带时域子层的流（例如开了 SVC / "H.265+"/smart-codec
> 的海康相机）只会解出**基础时域层**（约 1/4 的帧） ——
> `target_dec_temporal_id_plus1` 这个控制项在当前 SoC SDK 上无效。非分层的 HEVC
> 与 H.264 都能完整解出。

---

## ByteTracker

`bcdl/tracks/byte_tracker.h` —— 不依赖模型的多目标跟踪器（Kalman + 两阶段 IoU
关联）。Move-only。

```cpp
struct Track { int track_id; float x1,y1,x2,y2,score; int class_id; };
struct ByteTrackConfig { float track_thresh=0.5, high_thresh=0.6, match_thresh=0.8;
                         int track_buffer=30, frame_rate=30;
                         float proximity_thresh=0.5, appearance_thresh=0.25,
                               ema_alpha=0.95;
                         bcdl::BoostConfig boost; };

bcdl::ByteTracker tracker(cfg);
for (auto& frame : stream) {
  std::vector<bcdl::Detection> dets = /* 检测器 */;
  std::vector<bcdl::Track> tracks = tracker.update(dets);  // 每帧一次
}
tracker.reset();   // 流断开时
```

传入的检测必须已经在原图像素坐标上（各检测器已经做好了）。

### 外观（ReID）

`bcdl/tracks/reid.h`。每个检测各传一个嵌入向量，第一轮关联的代价就变成
`min(IoU 距离, 门控后的余弦距离)` —— 所以外观只可能**挽救**几何本来要漏掉的
匹配，永远不会破坏几何本来已经拿到的匹配。某一项传**空**表示"这个检测没有
外观"，这正是你在廉价裁剪上跳过 ReID 模型的方式。

```cpp
std::vector<std::vector<float>> embs(dets.size());
for (size_t i = 0; i < dets.size(); ++i) {
  if (dets[i].score < 0.5f) continue;                 // 留空
  bcdl::reidPreprocess(bgr, w, h, w * 3, dets[i].x1, dets[i].y1,
                       dets[i].x2, dets[i].y2, 128, 256, reid_cfg, crop);
  reid_engine.setInput(0, crop.data(), crop.size() * sizeof(float));
  reid_engine.infer();
  embs[i] = embedder.postprocess();                   // bcdl::ImageEmbedder
}
std::vector<bcdl::Track> tracks = tracker.update(dets, embs);
```

`reidPreprocess()` 做的是到模型裁剪尺寸的**挤压式** resize（不是 letterbox ——
这类模型就是在挤压过的裁剪上训练的）、BGR→RGB、ImageNet 归一化、NCHW float32。
读出复用 `ImageEmbedder`；`l2Normalize()` / `cosineSimilarity()` 是 header-only
的基础函数。

若 `embeddings.size()` 与 `detections.size()` 不一致、或嵌入维度在流中途变化，
会抛 `Error(-1)`。

### 相机运动

```cpp
float affine[6] = {1,0,dx, 0,1,dy};   // 把上一帧映射到这一帧
tracker.applyCameraMotion(affine);    // 在下一次 update() 之前
```

位置与尺寸会被 warp，速度不会 —— 因为速度描述的是目标在世界中的运动，而相机抖
一下并不等于目标在加速。变换由调用方提供，因为这个类只看得见框。静止相机请
直接不调用，而不是传一个单位矩阵。

### BoostConfig

BoostTrack++ 的增补项，**默认全部关闭**，每一项都可独立开关，这样它的贡献是被
测出来的而不是假设出来的：`rich_similarity`（马氏 + 形状项，用 `min_iou` 兜住
结果）、`soft_biou`（按 `1 - 轨迹置信度` 同时放大两个框）、`boost_detections`
（在高/低分切分之前做 DLO/DUO 抬分）。最后一项**看场景** —— 当检测器的瓶颈是漏检
时它能捞回真检测，否则它制造假轨迹。

## DetectionPipeline

`bcdl/pipeline/detection_pipeline.h` —— 同步、每帧零分配的
BGR→NV12→推理→解码。需要一个 NV12 输入的 YOLO `.hbm`。

```cpp
struct PipelineConfig {
  int input_w=0, input_h=0;          // 0 => 从 Engine input[0] 推导
  bcdl::DetectConfig detect;         // 两个头共用的阈值
  int output_index=0;                // kSingleTensor 的输出下标
  uint8_t pad_value=114;
  bcdl::DetectHead head = bcdl::DetectHead::kAuto;  // kAuto/kSingleTensor/kYoloLtrb
  std::vector<int> ltrb_strides = {8,16,32};
};

bcdl::DetectionPipeline pipe(engine, cfg);
std::vector<bcdl::Detection> dets = pipe.process(bgr, width, height);
const bcdl::LetterboxInfo& lb = pipe.lastLetterbox();
bcdl::DetectHead resolved = pipe.head();

// 跨每次 process() 调用累计的各阶段耗时（用于性能剖析）。
const bcdl::StageProfile& sp = pipe.profile();
// sp.preproc_ms / infer_ms / postproc_ms（累计）、sp.frames、sp.totalMs()、
// sp.preprocPerFrame() / inferPerFrame() / postprocPerFrame()。
pipe.resetProfile();
```

`DetectHead::kAuto` 在构造时根据 Engine 的输出签名解析。共享辅助
（`resolveDetectionConfig`、`requireNv12InputModel`、`feedNv12Input`、
`preprocBgrToNv12`、`HeadDecoder`）也对外暴露，方便自定义流水线。

`StageProfile`（与 `AsyncDetectionPipeline` 共用）把每帧成本拆成 `preproc`
（letterbox BGR→NV12，CPU）、`infer`（喂入 + BPU 提交/等待）、`postproc`
（解码 + NMS，CPU）；视频流水线还会填 `decode_ms` 与 `cvt_ms`。在**同步**流水线
里三者背靠背执行，所以之和就是 `process()` 的成本；在**异步**流水线里它们是各自
线程上测的*服务*时间，所以之和会超过墙钟时间，且**最慢**的那一段决定吞吐。

## AsyncDetectionPipeline

`bcdl/pipeline/async_detection_pipeline.h` —— 两个工作线程把后面帧的 CPU 预处理
与前面帧的 BPU 推理+解码重叠起来。`next()` 按提交顺序返回结果。

```cpp
bcdl::AsyncDetectionPipeline p(engine, cfg, /*depth=*/3);
std::vector<bcdl::Detection> dets;
int i = 0;
for (auto& f : stream) {
  p.submit(f.bgr, f.w, f.h);          // 字节已拷贝；队满则阻塞（背压）
  if (i++ >= 3) p.next(dets);         // 保持流水线满载，按序取走
}
p.finish();                           // 标记流结束（析构里也会做）
while (p.next(dets)) { /* 最后在途的结果 */ }
```

- `submit(bgr, w, h) -> bool` —— `finish()` 之后返回 `false`（帧未被接受）。
- `next(out) -> bool` —— 已 finish **且**排空之后返回 `false`。
- `depth` ≥ 2 才能重叠；更大能容忍更多抖动，代价是延迟。
- `profile() -> StageProfile` —— 各阶段的*服务*时间（每段在各自线程上；最慢的
  决定吞吐）。在 `finish()` + 完全排空之后再读。

把解码出来的视频帧经一个解码线程送进 `AsyncDetectionPipeline`，可以得到完整的
`解码 ‖ 预处理 ‖ 推理+NMS` 重叠；见 `examples/video_det_demo_async.cc`
（yolo26n 1080p：串行 119 → 重叠 234 FPS，瓶颈在解码）。如果你想要的是
"喂压缩字节"的接口，请用下面的 `AsyncVideoDetectionPipeline` —— 那个重叠由它
自己拥有，调用方只管搬字节。

## AsyncVideoDetectionPipeline

`bcdl/pipeline/async_video_detection_pipeline.h` —— 整条"压缩视频 → 检测结果"
的链路收在一个 C++ 对象里：它把 Annex-B 字节切成访问单元、用 VPU 解码、
NV12→BGR 转换、然后检测，**四段重叠**（`解码 ‖ nv12→bgr ‖ 预处理 ‖ 推理+NMS`）。
调用方只喂字节，所以一个很薄的驱动（例如 Python 搬运 `ffmpeg -c copy` 的流）
也能打到 C++ 的解码上限，而不是被自己的编排拖住。

```cpp
bcdl::AsyncVideoDetectionPipeline p(engine, cfg, HB_VP_VIDEO_TYPE_H264, /*depth=*/4);
std::vector<bcdl::Detection> dets;
while (int n = read(ffmpeg_stdout, buf, sizeof buf)) {
  p.submit(buf, n);                    // Annex-B 字节；背压时阻塞
  while (p.tryNext(dets)) { /* 画框 / 计数 */ }   // 非阻塞排空
}
p.finish();
while (p.next(dets)) { /* 排空最后在途的帧 */ }
```

- `submit(data, n) -> bool` —— 喂一段 Annex-B 字节（任意长度；AU 由内部切分）。
  队满时阻塞；`finish()` 之后返回 `false`。
- `next(out) -> bool` —— 按解码顺序阻塞弹出；已 finish 且排空后返回 `false`。
- `tryNext(out) -> bool` —— 非阻塞弹出；边喂边排空。
- `profile() -> StageProfile` —— 在 preproc/infer/postproc 之外，还有 `decode_ms`
  （VPU 解码）与 `cvt_ms`（NV12→BGR）。五项都是各线程的*服务*时间，所以**最慢**
  的那一段决定吞吐。

实测：**yolo26n 1080p H.264 → 约 441 FPS**，瓶颈在解码（`decode 0.29 |
preproc 1.32 | infer 1.47 | postproc 0.65` ms/帧）。流水线把解码出的 NV12 直接
在 GDC 硬件引擎上 letterbox 进模型输入 —— 没有 BGR 往返，`cvt_ms` 保持为 0。
H.265 跑 300/300 帧、439–451 FPS。视频解码支持 **H.264 与 H.265**（分层 GOP 的
HEVC 流只解出基础时域层）。想从解码帧里拿到 BGR 的调用方仍可用
`nv12ToBgrCpu()`（preproc/letterbox_cpu.h）。它接收一个 `YuvRange`：
`kStudioToFull`（默认 —— 视频解码器输出的就是这个，也是
`cv::cvtColor(COLOR_YUV2BGR_NV12)` 的行为）或 `kAsIs`（full-range 输入，是
`bgrToNv12Cpu()` 的逐位逆变换）。OpenCV SIMD 路径与手写回退实现对两种范围的
处理完全一致。

## TrackingPipeline

`bcdl/pipeline/tracking_pipeline.h` —— `DetectionPipeline` 接一个 `ByteTracker`，
一帧进、轨迹出。

```cpp
bcdl::TrackingPipeline pipe(engine, det_cfg /*=PipelineConfig*/,
                                    track_cfg /*=ByteTrackConfig*/);
std::vector<bcdl::Track> tracks = pipe.process(bgr, width, height);
const auto& dets = pipe.lastDetections();   // 关联之前的检测，便于叠加显示
pipe.reset();

// 带外观：传入一个 ReID Engine（它必须活得比流水线久）。裁剪尺寸来自那个模型，
// 所以换模型就是换路径。
bcdl::TrackingPipeline pipe2(engine, reid_engine, det_cfg, track_cfg,
                             bcdl::TrackingReidConfig{});
pipe2.hasReid();          // true
pipe2.lastEmbedCount();   // 上一帧嵌入了多少个裁剪
```

`TrackingReidConfig{min_score, max_crops, crop}` 都是成本旋钮：ReID 模型
**每个合格裁剪跑一次**，所以人多的帧里决定帧耗时的是它而不是检测器。

## StereoPipeline

`bcdl/pipeline/stereo_pipeline.h` —— 两张已校正的图 → 视差（+ 可选米制深度 /
有效性掩膜）。像素归一化已经融进 `.hbm`。

```cpp
enum class StereoFit { kResize, kCrop };   // 必须与离线标定一致
struct StereoConfig {
  int input_w=0, input_h=0; bcdl::StereoFit fit = bcdl::StereoFit::kResize;
  bool to_rgb=true; int left_index=0, right_index=1, output_index=0;
  float fx=0, baseline=0;                 // 两者都 > 0 时填充 StereoResult.depth
  bool valid_mask=false; float disp_min=0, max_disp=192; int left_margin=0;
  bool lr_check=false; float lr_thresh=1.5;
};
struct StereoResult { bcdl::DepthMap disparity; std::vector<float> depth;
                      std::vector<uint8_t> valid; };

bcdl::StereoPipeline pipe(engine, cfg);
bcdl::StereoResult res = pipe.process(left_bgr, right_bgr, width, height);
// res.disparity.data（总是有）、res.depth / res.valid（启用时才有）
```

纯辅助函数：`packStereoInputCHW(bgr, w, h, out_h, out_w, fit, to_rgb, dst)`、
`disparityToDepth(disp, fx, baseline)`、
`stereoValidMask(disp, disp_min, max_disp, left_margin, disp_right, lr_thresh)`。

---

## 完整示例

对一张 BGR 图做端到端的 LTRB 检测（与 `examples/` 对应）：

```cpp
#include "bcdl/bcdl.h"

int main(int argc, char** argv) {
  try {
    bcdl::Engine engine(argv[1]);            // 一个 NV12 输入的 YOLO26 .hbm

    // 最省事：让流水线接管预处理与检测头选择。
    bcdl::PipelineConfig cfg;
    cfg.detect.num_classes = 80;
    bcdl::DetectionPipeline pipe(engine, cfg);

    // bgr：交织的 HxWx3 uint8（例如 cv::imread 之后取 .data）
    std::vector<bcdl::Detection> dets = pipe.process(bgr, width, height);
    for (const auto& d : dets)
      std::printf("cls=%d score=%.3f [%.1f,%.1f,%.1f,%.1f]\n",
                  d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
  } catch (const bcdl::Error& e) {
    std::fprintf(stderr, "bcdl error %d: %s\n", e.code(), e.what());
    return 1;
  }
}
```

更多内容见 [`examples/`](../examples/) 目录里的可运行程序：`det_demo`、
`ocr_demo`、`track_demo`、`video_det_demo`（串行，打印各阶段耗时分布）、
`video_det_demo_async`（三段重叠的 解码‖预处理‖推理，并做串行/异步对比）、
`stereo_demo`、`jpeg_roundtrip`、`video_roundtrip`、`mempool_demo`，
以及各个 `*_bench` 驱动程序。

# BCDL C++ API 参考

[English](CPP_API.md) | **简体中文**

C++ 库（命名空间 `bcdl`）的完整参考。[`include/bcdl/`](../include/bcdl/) 下的
公共头文件才是唯一权威 —— 本文是它们的摘要加用法。Python 绑定见
[`API.zh.md`](API.zh.md)；两边的名字一一对应（Python snake_case ⇄ C++ camelCase）。

- [链接与构建](#链接与构建) · [约定](#约定) · [完整示例](#完整示例)
- core：[错误处理](#错误处理) · [SysMem](#sysmem) · [Task](#task) · [MemPool](#mempool)
- backend：[Engine](#engine)
- 预处理：[几何与 letterbox](#几何与-letterbox) · [VpImage](#vpimage)
- 任务：[检测 Detection](#检测-detection) · [分类 Classification](#分类-classification) ·
  [姿态 Pose](#姿态-pose) · [全身姿态 WholeBody](#全身姿态-wholebody) ·
  [超分 SuperRes](#超分-superres) · [稀疏特征点 XFeat](#稀疏特征点-xfeat) ·
  [实例分割 InstanceSeg](#实例分割-instanceseg) · [旋转框 OBB](#旋转框-obb) ·
  [语义分割 Segmentation](#语义分割-segmentation) · [深度 Depth](#深度-depth) · [OCR](#ocr)
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
与 hobot SDK 的链接库）：

```cmake
find_package(bcdl CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE bcdl::bcdl)
```

conda 包（`libbcdl` + `hobot-dnn` + `hobot-media`）与板上构建见
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
| `numInputs()` / `numOutputs()` | 张量个数。 |
| `inputShape(i)` / `outputShape(i)` | `std::vector<int>` 形状。 |
| `inputType(i)` / `outputType(i)` | `hbDNN` 张量类型枚举。 |
| `inputProperties(i)` / `outputProperties(i)` | 完整的 `hbDNNTensorProperties`（dtype、量化、stride）。 |
| `inputBytes(i)` / `outputBytes(i)` | 分配到的设备缓冲大小。 |
| `inputPackedBytes(i)` | 输入 `i` 作为连续行主序数组的大小 —— 模型对某一维补齐时会小于 `inputBytes(i)`。 |
| `inputStride(i)` | 输入 `i` 解析后的字节 stride，最内层在最后。 |
| `outputStride(i)` | 输出 `i` 的字节 stride。输出同样会补齐；`outputAsFloat()` 会处理好，直接读原始指针的代码**不能**摊平 reshape。 |
| `setInput(i, data, bytes)` | 把 host 字节拷进输入 `i` 并刷缓存。`bytes` 必须等于 `inputPackedBytes(i)`（按设备 layout 散布写入）或 `inputBytes(i)`（原样拷贝）；其他值一律抛异常。 |
| `inputData(i)` | 输入 `i` 设备缓冲的只读视图，按设备 layout。 |
| `infer(timeout_ms=0)` | 执行一次推理（阻塞；并 invalidate 输出缓存）。 |
| `outputData(i)` | `infer()` 之后输出 `i` 的指针（注意 stride）。 |
| `static elemSize(tensor_type)` | 某个 `hbDNN` 张量元素类型的字节大小。 |

> 相比自己读 `outputData()`，优先用**任务类**（`Detector`、`Segmenter`……）——
> 它会替你反量化（F16/int8/int16）并处理 stride。只有写自定义解码器时才降到
> 原始输出。

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

**CPU 预处理**（`bcdl/preproc/letterbox_cpu.h`）—— 部署板上的 hbVP 几何算子是
vDSP 支撑的、且只在离线/root 下可用，所以流水线走的是这些 CPU 路径。几何与
hbVP 路径完全一致。

```cpp
bcdl::VpImage src(w, h, HB_VP_IMAGE_FORMAT_BGR);
bcdl::VpImage nv12(inW, inH, HB_VP_IMAGE_FORMAT_NV12);
bcdl::LetterboxInfo lb = bcdl::letterboxToNv12Cpu(nv12, src, /*padValue=*/114);
// 另有：letterboxCpu(dstBgrOrY, src, pad) ; bgrToNv12Cpu(dstNv12, srcBgr)
```

`bgrToNv12Cpu` 用的是 BT.601 **full-range**（与 cv2 的 `COLOR_BGR2YUV_I420`
一致）；请与你模型标定时的范围对照确认。

**hbVP 路径**（`bcdl/preproc/letterbox.h`，需要 DSP）：`letterbox(dst, src,
padValue, interp)`、`cvtColor(dst, src)`、`resizeExact(dst, src, interp)`。

**GDC 硬件路径**（`bcdl/preproc/gdc_letterbox.h`、`gdc_remap.h`；仅在带
`BCDL_HAVE_GDC` 构建时可用）—— VPS GDC 引擎上的固定变换，NV12 进 / NV12 出，
算子执行期间 CPU 空闲。语义与逆向出来的 CUSTOM 网格笔记见
[docs/GDC.md](GDC.md)。

```cpp
// 用离线生成的 AFFINE warp bin 做 letterbox
bcdl::GdcLetterbox lbg(binPath, inW, inH, outW, outH, /*pad=*/114);
lbg.run(srcNv12, dstNv12);

// 任意固定的密集重映射（cv2.remap 语义），LUT 运行时生成
bcdl::GdcRemap remap(mapX, mapY, inW, inH, outW, outH, /*gridStep=*/16);
remap.run(srcNv12, dstNv12);   // 2448x2048：墙钟约 6.3 ms，CPU 约 1 ms
```

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

**anchor-free LTRB 多尺度**（YOLO26 / 标准 RDK NV12 导出）—— 每个 stride 一对
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

## 分类 Classification

`bcdl/tasks/classification.h`

```cpp
struct ClsConfig { int top_k = 5; bool apply_softmax = true; };
struct ClsResult { int class_id; float score; };

bcdl::Classifier clf(engine, cfg, /*output_index=*/0);
std::vector<bcdl::ClsResult> top = clf.postprocess();   // 在 infer() 之后
// 纯函数：decodeClassification(logits, num_classes, cfg)
```

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

bcdl::SuperResolver sr(engine, cfg);
auto big = sr.upscale(bgr, w, h, stride);        // 放大 sr.scale() 倍
// 纯函数：planTiles(w, h, tile_w, tile_h, overlap)、tileWeight(i, len, ramp)
```

`scale()` 与 `tile()` 都从模型形状里读，永远不由配置指定。混合是累加
`w * pixel` 与 `w` 再相除，所以图像边界不需要特判。注意编译出的 `.hbm` 体积随
tile **面积**增长 —— 同一个网络在 256 tile 下是 148 MB、128 下是 37 MB，而每像素
吞吐相同。

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
                           bool compute_masks=true; };

bcdl::InstanceSegmenter seg(engine, cfg, /*output_base=*/0);
auto masks = seg.postprocess(lb, orig_w, orig_h);
// 纯函数：decodeInstanceSeg(cls, box, mc, grid_hw, num_classes, num_coef,
//                           proto, proto_h, proto_w, proto_c, cfg, lb, orig_w, orig_h)
```

设 `compute_masks=false` 可只拿框与分数，跳过掩膜组装。

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
> 的海康相机）只会解出**基础时域层**（约 1/4 的帧）——
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

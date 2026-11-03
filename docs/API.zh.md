# BCDL Python API 参考

[English](API.md) | **简体中文**

`bcdl` Python 模块（C++ 核心之上的 nanobind 绑定）的完整参考。C++ 接口见
[`CPP_API.zh.md`](CPP_API.zh.md)（[`include/bcdl/`](../include/bcdl/) 下的公共头文件
才是唯一权威）；两边的名字一一对应（Python snake_case ⇄ C++ camelCase）。

- [安装](#安装) · [两种使用方式](#两种使用方式) · [约定](#约定)
- 引擎：[Engine](#engine)
- 预处理：[Letterbox 与几何](#letterbox-与几何) · [NV12 辅助](#nv12-辅助) ·
  [GDC 硬件算子](#gdc-硬件算子-仅板端)
- 任务：[检测 Detection](#检测-detection) · [分类 Classification](#分类-classification) ·
  [姿态 Pose](#姿态-pose) · [全身姿态 WholeBody](#全身姿态-wholebody) ·
  [超分 SuperRes](#超分-superres) · [稀疏特征点 XFeat](#稀疏特征点-xfeat) ·
  [实例分割 InstanceSeg](#实例分割-instanceseg) · [旋转框 OBB](#旋转框-obb) ·
  [语义分割 Segmentation](#语义分割-segmentation) · [深度 Depth](#深度-depth) ·
  [双目 Stereo](#双目-stereo) · [OCR](#ocr)
- 语音：[语音链总览（从这里开始）](#语音链总览) · [唤醒词](#唤醒词检测) ·
  [端点检测](#语音端点检测silero-vad) · [识别](#语音识别paraformer) ·
  [指令匹配](#语音指令固定指令集) · [合成](#语音合成matcha-tts--vocos)
- 流式与跟踪：[跟踪 ByteTrack](#跟踪-bytetrack) · [TrackingPipeline](#trackingpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline) ·
  [AsyncVideoDetectionPipeline](#asyncvideodetectionpipeline)
- 多媒体：[图像与编解码](#图像与编解码)

> 命名：snake_case 的函数名与 `lower_case` 的配置字段，是 C++ camelCase 成员的
> Python 写法，无一例外。曾有九个解码器另带 camelCase 别名；这些名字已废弃，
> 调用时会给出 `DeprecationWarning`，本手册不再收录。

## 安装

完整表格见 [README](../README.md#快速上手)。最短路径：

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bcdl
python -c "import bcdl; print(bcdl.__version__)"
```

`import bcdl` 会传递性地带入 C++ 库与打包好的 hobot SDK，所以在一块 RDK S100 /
S100P / S600 板子上不需要再装别的东西。

## 两种使用方式

每个任务都有两个层次 —— 按你想让 BCDL 接管多少来选：

1. **高层任务类**（`Detector`、`Classifier`、`PoseEstimator`、`Segmenter`、
   `DepthEstimator`……）。给它一个 `Engine` 和一份配置，它一次调用完成推理与
   后处理。设备缓冲、缓存刷新、反量化都由 BCDL 负责。

2. **纯 `decode_*` 函数**（`decode`、`decode_pose`、`decode_obb`、`decode_seg`、
   `decode_ctc`……）。它们吃 **float32 NumPy 数组**（你手上已有的模型输出），
   返回同样的结果对象 —— **不需要 Engine、不需要板子、不需要模型。**
   确定性测试走的就是这条路；想把 BCDL 的后处理接到别处产生的输出上，也走这条。

   > **量化注意：** `decode_*` 假定输入是**已反量化的 float32**。`F16` 输出请先
   > 转换（`np.asarray(out, np.float32)`）。若是**量化的 int8/int16** 输出，它们会
   > 把原始整数当浮点读，结果必然是错的 —— 这种情况请用高层任务类，它会按张量的
   > scale / zero-point 反量化。

## 约定

- **图像是 `HxWx3` uint8 BGR**（OpenCV 顺序）。流水线遇到非 uint8 输入会**明确报错**，
  而不是悄悄什么都检不出来。
- **坐标返回的是原图像素。** 需要撤销 letterbox 的任务，在 `postprocess(...)` 里
  接收一个 `LetterboxInfo`。
- **NV12 双输入模型：** 标准 RDK YOLO 导出接受两个输入 `[Y, UV]`；
  `*Detector.detect([...], lb)` 这类包装会按顺序设置输入。
- **预处理的归属：** 高层*任务类*（`Detector` 等）把预处理留给你（各模型的
  layout 不同）；而*流水线*（`TrackingPipeline`、`AsyncDetectionPipeline`、
  `StereoPipeline`）在 C++ 里自己做预处理，直接吃一帧原始 BGR。
- **`__version__`** 即 `bcdl.__version__`。

---

## Engine

加载编译好的 `.hbm` 并执行 BPU 推理，替你处理缓存纪律（推理前 clean 输入、
读取前 invalidate 输出）。

```python
import numpy as np, bcdl

engine = bcdl.Engine("model.hbm")            # 可选 model_name=""
print(engine.num_inputs, engine.num_outputs)
print(engine.input_shape(0), engine.input_dtype(0))
print(engine.output_shape(0), engine.output_dtype(0))

# 通用推理：输入数组列表进 -> 输出数组列表出。
outs = engine.infer([x])                     # outs[i] 已 reshape 成 output_shape(i)
```

| 成员 | 说明 |
|--------|-------------|
| `Engine(hbm_path, model_name="")` | 加载模型。`.hbm` 里打包了多个模型时，用 `model_name` 选其中一个。 |
| `model_name -> str` | 当前模型名。 |
| `num_inputs / num_outputs -> int` | 张量个数。 |
| `input_name(i) / output_name(i) -> str` | 编译进模型的张量名（即 ONNX 里的名字）。 |
| `input_shape(i) / output_shape(i) -> list[int]` | 张量形状。 |
| `input_bytes(i) -> int` | 输入 `i` 分配到的设备缓冲大小（已按 BPU stride 对齐）。 |
| `input_packed_bytes(i) -> int` | 输入 `i` 作为连续行主序数组的大小。模型对某一维做了补齐时会小于 `input_bytes(i)`。 |
| `input_stride(i) -> list[int]` | 输入 `i` 解析后的字节 stride，最内层在最后。 |
| `output_stride(i) -> list[int]` | 输出 `i` 的字节 stride。**输出同样会补齐** —— 把原始缓冲直接摊平 reshape 会把张量剪切错位。 |
| `output_packed_bytes(i) -> int` | 输出 `i` 作为连续行主序数组的大小。模型对某一维做了补齐时会小于设备缓冲。 |
| `input_dtype(i) / output_dtype(i) -> str` | NumPy dtype 字符串（如 `"float32"`、`"float16"`、`"int8"`）。 |
| `infer(inputs, timeout_ms=0) -> list[np.ndarray]` | 把输入拷进设备缓冲、执行、每个输出返回一个数组。`timeout_ms=0` 表示阻塞。 |

> **补齐的输出已经替你收拢过了。** 即使模型对某一维做了补齐，`infer()` 返回的也是
> 连续行主序数组（ViTPose 的 `[1,133,64,48]` 热图每行 192 字节有效、stride 256）。
> 直接摊平读设备缓冲会把第 `r` 行错位 `r * pad`，**得到的张量看上去仍然像数据** ——
> 这曾让一个其实正确的模型被当成量化失败查了一天。

> 高层任务类内部包住一个 `Engine`，直接调用原生后处理（不走每次调用的 NumPy 往返），
> 所以单个任务优先用它们；只有当你想拿到原始输出张量时才用 `Engine.infer()`。

---

## Letterbox 与几何

保持长宽比的贴合，以及原图 ⇄ 模型空间的坐标映射。

```python
lb = bcdl.compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True)
mx, my = lb.fwd_x(x), lb.fwd_y(y)   # 原图 -> 模型像素
ox, oy = lb.inv_x(x), lb.inv_y(y)   # 模型 -> 原图像素

# 对 uint8 图像一次性做 CPU letterbox（需要 OpenCV）：
canvas, lb = bcdl.letterbox_numpy(img, dst_w, dst_h, pad=114)
```

- **`compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True) -> LetterboxInfo`**
- **`LetterboxInfo`** —— 字段 `scale`、`pad_x`、`pad_y`、`src_w/h`、`dst_w/h`
  （均可读写）；方法 `fwd_x/fwd_y`（原图→模型）与 `inv_x/inv_y`（模型→原图）。
  把它传给检测家族的每个 `postprocess(...)`，框才会回到原图像素坐标。
- **`letterbox_numpy(img, dst_w, dst_h, pad=114) -> (canvas, LetterboxInfo)`** ——
  几何与 C++ VP letterbox 一致；需要 OpenCV。

## NV12 辅助

```python
nv12 = bcdl.bgr_to_nv12(bgr)         # (H*3//2, W) uint8；尺寸需为偶数；需要 OpenCV
```

- **`bgr_to_nv12(bgr) -> np.ndarray`** —— 打包的 NV12，可直接喂给
  `vp_image_from_nv12` 与 JPU 编码器。另见[图像与编解码](#图像与编解码)。

## GDC 硬件算子 仅板端

VPS GDC 引擎上的固定几何变换 —— NV12 进 / NV12 出，算子执行期间 CPU 空闲。
不在板上、或不支持 GDC 时为 `None`。完整语义与逆向出来的 CUSTOM 网格笔记见
[docs/GDC.md](GDC.md)。

```python
# 硬件 letterbox（warp LUT 在构造时生成）
g = bcdl.GdcLetterbox(in_w, in_h, out_w, out_h, pad=114)
dst = g.run(src_vpimage)                    # NV12 VpImage -> NV12 VpImage

# 硬件密集重映射（cv2.remap 语义；LUT 运行时生成）
g = bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)   # (out_h,out_w) f32 映射表
dst = g.run(src_vpimage)
```

- **`GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)`** —— 任意**固定**形变，
  `out(x,y) = in(map_x[y,x], map_y[y,x])`；为双目立体校正而做。`grid_step` 必须
  能整除输出的两个维度。2448×2048：墙钟 6.3 ms / CPU 约 1 ms，对比全核
  `cv2.remap` 的 14.7 ms；与 cv2 的差异在 p99 ≤ 2 个灰阶。设
  `BCDL_GDC_TIMING=1` 可打印拷贝/算子的耗时拆分。
- **`GdcLetterbox(in_w, in_h, out_w, out_h, pad=114)`** —— GDC 上的 letterbox；
  warp LUT 在构造时生成（不需要离线 `.bin`）。`.info` 返回对应的 `LetterboxInfo`。
  1920×1080 → 640×640：墙钟 0.97 ms，其中约 0.3 ms 占 CPU。几何与 CPU letterbox
  完全一致；两者的重采样器在高频细节上的走样方式不同 —— 见 [docs/GDC.md](GDC.md)。

---

## 检测 Detection

两个解码器家族：

- **单张量**（`DecodeLayout.YoloV8` / `YoloV5`）—— 一个浮点输出张量。
  用 `DetectConfig` + `decode(...)` / `Detector`。
- **anchor-free LTRB 多尺度**（YOLO26 / 标准 RDK NV12 导出）—— 每个 stride 一对
  `(cls, box)` 输出。用 `YoloLtrbConfig` + `YoloLtrbDetector`。

```python
# 单张量，numpy 路径：
cfg = bcdl.DetectConfig(); cfg.num_classes = 80; cfg.input_w = cfg.input_h = 640
out = engine.infer([x])[0].astype(np.float32)      # (N, 4+nc) 或 (4+nc, N)
dets = bcdl.decode(out, cfg, lb)
for d in dets:
    print(d.class_id, d.score, d.x1, d.y1, d.x2, d.y2)

# anchor-free LTRB，高层用法（NV12 双输入）：
det = bcdl.YoloLtrbDetector(engine, bcdl.YoloLtrbConfig())
dets = det.detect([y_plane, uv_plane], lb)
```

**`DetectConfig`** —— `input_w`、`input_h`、`num_classes`、`conf_thresh`、
`iou_thresh`、`max_dets`、`layout`（`DecodeLayout`）、`channels_first`、
`apply_sigmoid`。

**`DecodeLayout`** 枚举 —— `YoloV8`、`YoloV5`。

**`Detection`** —— `x1, y1, x2, y2, score, class_id`（均可读写）；带 `__repr__`。

**`YoloLtrbConfig`** —— `num_classes`、`conf_thresh`、`iou_thresh`、`max_dets`、
`strides`（如 `[8,16,32]`）、`reg_max`（DFL 分箱数；`0`/`1` 表示直接 LTRB）。

函数 / 类：

| 符号 | 签名 |
|--------|-----------|
| `decode` | `decode(output: f32 ndarray, config: DetectConfig, letterbox: LetterboxInfo) -> list[Detection]` |
| `nms` | `nms(dets, iou_thresh, max_dets=300) -> list[int]`（保留下来的下标） |
| `iou` | `iou(a: Detection, b: Detection) -> float` |
| `Detector` | `Detector(engine, config, output_index=0)`；`.detect(model_input, lb, timeout_ms=0)`、`.config` |
| `YoloLtrbDetector` | `YoloLtrbDetector(engine, config=None, output_base=0)`；`.detect([inputs], lb, timeout_ms=0)`、`.postprocess(lb)`、`.config` |

---

## 分类 Classification

```python
cfg = bcdl.ClsConfig(); cfg.top_k = 5; cfg.apply_softmax = True
clf = bcdl.Classifier(engine, cfg)
for r in clf.classify(x):                 # x：单个数组，或 [Y, UV]
    print(r.class_id, r.score)

# numpy 路径：
top = bcdl.decode_classification(logits.astype(np.float32), cfg)
```

- **`ClsConfig`** —— `top_k`、`apply_softmax`。
- **`ClsResult`** —— `class_id`、`score`；带 `__repr__`。
- **`decode_classification(logits, config) -> list[ClsResult]`** —— 一维 logit
  向量取 top-k。
- **`Classifier(engine, config=None, output_index=0)`** —— `.classify(inputs, timeout_ms=0)`、
  `.postprocess()`、`.config`。

## 姿态 Pose

LTRB 多尺度头：一个人体框加 K 个关键点。NV12 双输入。

```python
cfg = bcdl.PoseConfig(); cfg.num_keypoints = 17
est = bcdl.PoseEstimator(engine, cfg)
for p in est.detect([y_plane, uv_plane], lb):
    print(p.score, [(k.x, k.y, k.score) for k in p.keypoints])
```

- **`PoseConfig`** —— `num_keypoints`、`conf_thresh`、`iou_thresh`、`max_dets`、
  `strides`。
- **`PoseDetection`** —— `x1, y1, x2, y2, score, class_id`、`keypoints: list[Keypoint]`。
- **`Keypoint`** —— `x, y, score`。
- **`PoseEstimator(engine, config=None, output_base=0)`** —— `.detect([inputs], lb, timeout_ms=0)`、
  `.postprocess(lb)`、`.config`。
- **`decode_pose(cls, box, kpt, config, letterbox) -> list[PoseDetection]`** ——
  numpy 路径；`cls/box/kpt` 是逐 stride 的 `[H,W,1]` / `[H,W,4]` / `[H,W,K*3]`
  浮点数组列表。

## 全身姿态 WholeBody

**自顶向下**，与 `PoseEstimator` 不同：对**每个人**的裁剪各跑一次推理，所以前面
必须接一个检测器，开销随人数增长。换来的是脚、68 点人脸和双手。

```python
est = bcdl.WholeBodyEstimator(engine)
for box in person_boxes:                     # 来自上面任意一个检测器
    kpts = est.estimate(bgr, box)            # 133 个关键点，原图像素
    print(sum(k.score > 0.2 for k in kpts), "可见")
```

关键点排布（COCO-WholeBody）：`0-16` 身体，`17-22` 脚，`23-90` 人脸，
`91-111` 左手，`112-132` 右手。

- **`WholeBodyConfig`** —— `kpt_thresh`、`blur_kernel`（DARK 调制，需为奇数）、
  `box_pad`、`mean`、`std`（RGB，在缩放到 `[0,1]` 之后施加）。
- **`WholeBodyCrop`** —— `x1, y1, pad_left, pad_top, padded_w, padded_h`：
  一个人体框是怎么变成模型画布的，以及怎么反变换回去。
- **`WholeBodyEstimator(engine, config=None, output_index=0)`** ——
  `.estimate(bgr, box, timeout_ms=0)`、`.postprocess(crop)`、`.config`。
- **`wholebody_preprocess(bgr, x1, y1, x2, y2, in_w=192, in_h=256, config=None)`**
  → `(input, crop)`。和本 API 其余部分一样吃 **BGR**，内部翻成模型要的 RGB。
  框先按 `box_pad` 加宽，再零填充到模型的 3:4 比例并 resize —— **不是** mmpose
  的 center/scale 仿射。
- **`decode_wholebody(heatmaps, crop, config=None) -> list[Keypoint]`** ——
  numpy 路径；`heatmaps` 是通道优先的 `[K,H,W]`（或 `[1,K,H,W]`）浮点数组。

## 超分 SuperRes

模型只放大一块固定尺寸的 tile；任意尺寸的图会被切成重叠的 tile，再交叉淡化拼回去。

```python
sr = bcdl.SuperResolver(engine)          # 放大倍数与 tile 尺寸都从模型里读
big = sr.upscale(img)                    # BGR 进、BGR 出，放大 sr.scale 倍
print(sr.last_tile_count, "块")
```

- **`SuperResConfig`** —— `overlap`（输入像素数；同时也是交叉淡化的宽度）。
- **`SuperResolver(engine, config=None, input_index=0, output_index=0)`** ——
  `.upscale(bgr, timeout_ms=0)`、`.scale`、`.tile`、`.last_tile_count`。
- **`plan_tiles(width, height, tile_w, tile_h, overlap)`** → 各 tile 的原点。
  每个轴上最后一块会与远端边缘对齐，所以那里的实际重叠可能超过请求值。
- **`tile_weight(i, len, ramp)`** —— 交叉淡化权重，恒 > 0，所以归一化混合不需要
  对边界做特判。

> tile 尺寸是**部署决策**，不只是内存问题：编译出的 `.hbm` 体积随 tile **面积**
> 增长（同一个网络，256×256 是 148 MB，128×128 是 37 MB），而每像素吞吐相同。

## 稀疏特征点 XFeat

可重复的关键点 + L2 归一化的 64 维描述子，外加互为最近邻匹配。只有卷积主干在
BPU 上（约 1.0 ms）；输入归一化以及全部 softmax / NMS / top-k / 采样都在 CPU。

```python
ext = bcdl.FeatureExtractor(engine)
fa, fb = ext.extract(img_a), ext.extract(img_b)
for m in bcdl.match_features(fa, fb, min_cossim=0.82):
    print(fa.xy[m.a], "->", fb.xy[m.b], m.score)
```

- **`XfeatConfig`** —— `detection_thresh`、`nms_kernel`（奇数）、`top_k`。
- **`FeatureSet`** —— `keypoints: list[Feature]`、`descriptors`（`(N, 64)` 数组）、
  `xy`（`(N, 2)` 数组）、`dim`、`len()`。
- **`Feature`** —— `x, y, score`。**`FeatureMatch`** —— `a, b, score`。
- **`FeatureExtractor(engine, config=None, output_base=0)`** ——
  `.extract(bgr, timeout_ms=0)`、`.postprocess(scale_x, scale_y)`、`.config`。
- **`xfeat_preprocess(bgr, in_w=640, in_h=480)`** → `(input, scale_x, scale_y)`。
  灰度用的是**通道均值**，不是亮度加权。
- **`decode_xfeat(feats, keypoints, heatmap, config=None, scale_x=1, scale_y=1)`**
  —— numpy 路径；三张图都是 `[C,H,W]`（或 `[1,C,H,W]`）浮点数组，通道数分别
  64 / 65 / 1。
- **`match_features(a, b, min_cossim=0.82)`** —— 互为最近邻。
  **代价是 `O(|a|*|b|*64)`**：默认 `top_k=4096` 时每对图约 130 ms，降到 1024 只要
  8 ms。如果匹配成了瓶颈，第一件该调的就是 `top_k`。

## 实例分割 InstanceSeg

LTRB 框，加上由 prototype 张量组装出的逐实例二值掩膜。

```python
cfg = bcdl.InstanceSegConfig(); cfg.compute_masks = True
seg = bcdl.InstanceSegmenter(engine, cfg)
for m in seg.detect([y_plane, uv_plane], lb, orig_w, orig_h):
    print(m.class_id, m.score, m.mask.shape)     # m.mask: (H, W) uint8 0/1
```

- **`InstanceSegConfig`** —— `conf_thresh`、`iou_thresh`、`max_dets`、`strides`、
  `proto_index`、`compute_masks`（设 `False` 可跳过掩膜组装以提速）。
- **`InstanceMask`** —— `x1, y1, x2, y2, score, class_id`、`mask_w, mask_h`、
  `mask`（`(H,W)` uint8 0/1；`compute_masks=False` 时为空）。
- **`InstanceSegmenter(engine, config=None, output_base=0)`** ——
  `.detect([inputs], lb, orig_w, orig_h, timeout_ms=0)`、
  `.postprocess(lb, orig_w, orig_h)`、`.config`。
- **`decode_instance_seg(cls, box, mc, proto, config, letterbox, orig_w, orig_h) -> list[InstanceMask]`**
  —— numpy 路径；`cls/box/mc` 为逐 stride 的 `[H,W,nc]`/`[H,W,4]`/`[H,W,np]`，
  `proto` 为 `[mH,mW,np]`。

## 旋转框 OBB

LTRB 加一个角度；旋转 IoU NMS。

```python
cfg = bcdl.ObbConfig(); cfg.num_classes = 15
obb = bcdl.ObbDetector(engine, cfg)
for o in obb.detect([y_plane, uv_plane], lb):
    r = o.rrect
    print(o.class_id, o.score, r.cx, r.cy, r.w, r.h, r.angle)  # angle 单位为弧度
```

- **`ObbConfig`** —— `num_classes`、`conf_thresh`、`iou_thresh`、`max_dets`、
  `strides`、`regularize`、`angle_offset_rad`、`angle_sign`。
- **`RotatedBox`** —— `cx, cy, w, h, angle`（弧度）。
- **`ObbDetection`** —— `rrect: RotatedBox`、`score`、`class_id`。
- **`ObbDetector(engine, config=None, output_base=0)`** —— `.detect([inputs], lb, timeout_ms=0)`、
  `.postprocess(lb)`、`.config`。
- **`decode_obb(cls, box, angle, config, letterbox) -> list[ObbDetection]`** ——
  numpy 路径；逐 stride 的 `[H,W,nc]`/`[H,W,4]`/`[H,W,1]`。
- **`rotated_iou(a_cx,a_cy,a_w,a_h,a_angle, b_cx,b_cy,b_w,b_h,b_angle) -> float`**。

## 语义分割 Segmentation

对 logit 张量取 argmax（或直接透传已经 argmax 过的 id 张量），得到逐像素标签图。

```python
cfg = bcdl.SegConfig(); cfg.num_classes = 19
seg = bcdl.Segmenter(engine, cfg)
mask = seg.segment(x)                 # SegMask
labels = mask.labels                  # (H, W) int32 类别 id
bgr = bcdl.seg_colorize(mask)         # (H, W, 3) uint8 调色板图

# numpy 路径：
mask = bcdl.decode_seg(logits.astype(np.float32), cfg)
```

- **`SegConfig`** —— `num_classes`、`channels_first`（NCHW vs NHWC）、
  `argmaxed`（模型已经输出 id 时设 `True`）。
- **`SegMask`** —— `width`、`height`、`num_classes`、`labels`（`(H,W)` int32）。
- **`decode_seg(output, config) -> SegMask`**。
- **`seg_colorize(mask) -> np.ndarray`** —— `(H,W,3)` uint8 固定调色板。
- **`Segmenter(engine, config=None, output_index=0)`** —— `.segment(model_input, timeout_ms=0)`、
  `.postprocess()`、`.config`。

## 激光雷达 3D 检测 Lidar 3-D detection

点云进,带朝向的 3D 框出(自车系的中心、尺寸、偏航角,单位米)。柱式检测器是一条**流水线**
而不是一个模型,各段成本差别极大,所以刻意不合并:

```python
cfg = bcdl.VoxelConfig()                               # 默认就是 KITTI 那套几何
vox, npts, coords, stats = bcdl.voxelize_pillars(points_xyzi, cfg)
x = bcdl.pillar_point_features(vox, npts, coords, stats.num_pillars, cfg)
feats = vfe_engine.infer([x])[0]                       # BPU,出 (1, C, P, 1)
canvas = bcdl.scatter_pillars(feats.reshape(feats.shape[1], -1).T[:stats.num_pillars],
                              coords[:stats.num_pillars], stats.num_pillars, cfg)
head_engine.infer([canvas])                            # BPU(backbone 已并进去)
boxes = bcdl.LidarDetector3d(head_engine).postprocess()
```

**点必须已经在自车系** —— 后面没有任何一环能看出你还停在传感器坐标系里。
**上限是真的会截断的**:`stats` 会报出超出范围被丢的点、因所在柱已满被丢的点、
以及没有槽位的占据列;这样一帧过密的扫描会**说出来**,而不是悄悄变瞎。

**VFE 有两种形态,差 179 倍**。吃三个原始柱张量的那种把逐点特征展开做在自己图里,
为一个 95 MFLOP 的 MLP 付 290 ms —— `(pillars, points, 4)` 是「巨大柱维 + 4 宽通道维」,
正是卷积加速器最不擅长的形状。同一个 MLP 改写成 `(1, 10, P, 32)` 上的 1×1 卷积只要
1.62 ms,改吃 `pillar_point_features()`。**按模型自己要什么喂**:单输入的 VFE 就是卷积形态;
同理,输入是 scatter 画布的 head 已经把 backbone 并了进去,不需要再单独跑一段。
两条链的取法见 `docs/MODELS.md`。

- **`VoxelConfig`** —— `min_x`/`max_x`/`min_y`/`max_y`/`min_z`/`max_z`、
  `pillar_x`/`pillar_y`、`max_points_per_pillar`、`max_pillars`、`num_features`;
  只读的 `grid_x`/`grid_y`。
- **`voxelize_pillars(points, config) -> (voxels, num_points, coords, Pillars)`** ——
  `points` 是 `(N, >=3)` float32 的 x, y, z[, 强度…]。
- **`pillar_point_features(voxels, num_points, coords, num_pillars, config) -> np.ndarray`**
  —— 出 `(1, num_features+6, max_pillars, max_points)`:原始点、相对柱质心的偏移、
  相对柱几何中心的偏移。**没填满的点位保持为零** —— 它们同样要过 MLP,否则点间取 max
  会把偏置映射出来的那个值当成一个真点。
- **`scatter_pillars(features, coords, num_pillars, config) -> np.ndarray`** ——
  出 `(1, C, grid_y, grid_x)`。
- **`Box3D`** —— `x`、`y`、`z`、`dx`、`dy`、`dz`、`heading`、`label`、`score`。
- **`Anchor3dConfig`** —— 栅格 + 每类 `sizes`/`centers` + `rotations`。
  **`generate_anchors_3d(config)`** 出 `(N, 7)`;它与官方模型图里自带的 anchor 常量
  **逐位吻合到 8e-06**。
- **`Det3dConfig`** —— `score_thresh`、`nms_iou`、`max_boxes`、`apply_sigmoid`、`anchors`。
- **`decode_det3d(cls, box, dir=None, anchors=..., config=...) -> list[Box3D]`**
  `dir=None` 时每个朝向**只在模 π 意义下正确**。
- **`bev_iou(a, b)`** · **`nms3d(boxes, iou_thresh, max_boxes)`** ——
  这一族的 NMS 定义在 **BEV 投影**上,不是体积。
- **`LidarDetector3d(engine, config=None, cls_index=0, box_index=1, dir_index=2)`** ——
  `.postprocess()`、`.anchors`。

## 异常检测 Anomaly detection（单类）

回答的是「这个零件像不像那批好的」。没有类别表,也没有缺陷训练集——模型只见过某一个产品的
**正常**样本,输出的是每个 patch「有多不像」。

两条 API 藏不住的后果:模型是**按产品**的,所以随库的 `.hbm` 是**示例而不是通用检测器**
(换零件就要重训);阈值是**产线级决策**,按该产品正常波动标定,不是库的常量。

```python
det = bcdl.AnomalyDetector(engine)
m = det.detect(bgr)                  # 一张 (H, W, 3) uint8 BGR
m.score                              # 图像级判定 = 图的最大值
heat = m.data                        # (H, W) float32,模型输入分辨率
mask = bcdl.anomaly_mask(m, 0.5)     # (H, W) uint8 0/255

# numpy 路径：
m = bcdl.decode_anomaly(grid.astype(np.float32), cfg)
```

图像分数取**最大值**而不是均值:一个零件有多坏,取决于它最坏的那一块;**取均值会让一大片
干净区域把小缺陷盖掉**——而那恰恰是这套方法存在的理由。

- **`AnomalyConfig`** —— `out_width`/`out_height`(0 ⇒ 保持原始网格;检测类默认设成模型输入尺寸)、
  `blur_kernel`(奇数,0 关闭)、`blur_sigma`、`threshold`。上采样**和**高斯平滑都是参考实现
  后处理的一部分,不是装饰。
- **`AnomalyMap`** —— `width`、`height`、`score`、`data`(`(H,W)` float32)。
- **`decode_anomaly(output, config=AnomalyConfig()) -> AnomalyMap`**。
- **`anomaly_mask(map, threshold) -> np.ndarray`** —— `(H,W)` uint8 0/255。
- **`anomaly_preprocess(bgr, in_w, in_h) -> np.ndarray`** —— resize + BGR→RGB + `/255` +
  ImageNet mean/std,输出 `(3,H,W)`。
- **`AnomalyDetector(engine, config=None, output_index=0, input_index=0)`** ——
  `.detect(bgr, timeout_ms=0)`、`.postprocess()`、`.input_size`。

## 光流 Optical flow

两帧进,逐像素位移场出。向量单位是**源图像素**、+u 向右 +v 向下(OpenCV 约定),
拆成 x/y 两张图后可直接喂 `cv2.remap`。

```python
est = bcdl.OpticalFlowEstimator(engine)
flow = est.estimate(frame0, frame1)     # 两张 (H, W, 3) uint8 BGR
uv = flow.data                          # (H, W, 2) float32,源图像素
bgr = bcdl.flow_colorize(flow)          # Middlebury 色轮

# numpy 路径：
flow = bcdl.decode_flow(out.astype(np.float32), cfg)
epe = bcdl.flow_endpoint_error(flow_a, flow_b)
```

预处理特意放在类里面:参考实现喂的是 **BGR、0-255**,除以 255 在**图内部**做——
给它 RGB 或者先归一化过的像素,**不会报错,只会给你一个看着挺合理的错场**。

- **`FlowConfig`** —— `scale_x`、`scale_y`(模型像素 → 源图像素,`estimate()` 会按传入尺寸自动设)、
  `channels_first`(`[1,2,H,W]` vs `[1,H,W,2]`)。
- **`FlowField`** —— `width`、`height`、`data`(`(H,W,2)` float32,uv 交错)。
- **`decode_flow(output, config=FlowConfig()) -> FlowField`**。
- **`flow_colorize(flow, max_magnitude=0.0) -> np.ndarray`** —— `(H,W,3)` uint8 BGR;
  传 `0` 用该场自身 99 分位幅值归一化,免得一个快速物体把整帧压成灰的。
- **`flow_endpoint_error(a, b) -> float`** —— 平均端点误差(像素),拿量化版对浮点参考打分就用它。
- **`flow_preprocess(bgr, in_w, in_h) -> np.ndarray`** —— 单帧转 `(3,H,W)` float,保持 BGR 与 0-255。
- **`OpticalFlowEstimator(engine, config=None, output_index=0, input0_index=0, input1_index=1)`** ——
  `.estimate(frame0, frame1, timeout_ms=0)`、`.postprocess()`、`.input_size`。

## 端到端驾驶 End-to-end driving（相机 + 激光雷达 → 轨迹）

本库唯一一个输出「决策」而不是「观测」的头：三路前视相机 + 一帧激光雷达进去，
一条规划好的自车轨迹出来，外加与它联合训练的感知头。多模态扩散解码器在编译图内部——
喂进聚类轨迹锚点，出来的已经是选好的那一条。

所有位姿都是自车系、米、x 向前 / y 向左、heading 弧度逆时针，且是**未来**的固定间隔采样
（现货模型：4 s 内 8 个点）。

```python
anchors = np.load("kmeans_navsim_traj_20.npy").astype(np.float32)   # (20, 8, 2)
planner = bcdl.DrivePlanner(engine, anchors)

camera = bcdl.stitch_drive_cameras(cam_left, cam_front, cam_right)  # (3, H, W) RGB [0,1]
lidar = bcdl.lidar_bev_histogram(points_xyz)                        # (C, H, W)

status = bcdl.DriveStatus()
status.driving_command = 0          # 路径指令，one-hot 槽位
status.velocity_x = 5.0             # m/s

r = planner.plan(camera, lidar, status)
[(p.x, p.y, p.heading) for p in r.trajectory]   # 规划轨迹
[(a.x, a.y, a.score) for a in r.agents]         # 检出的车辆
bgr = bcdl.seg_colorize(r.bev)                  # BEV 语义图（SegMask）
```

两个传感器构造函数是**几何契约**而不是便利封装：BEV 网格转置了、status 顺序错了，
模型照样推理、照样给你一条看着很顺的轨迹。

- **`lidar_bev_histogram(points, config=LidarBevConfig()) -> np.ndarray`** ——
  `(N,>=3)` 自车系 xyz → `(C,H,W)` 占据栅格，取值 `[0,1]`。网格**第 0 轴是 x（前向）**，
  第 1 轴是 y。**`LidarBevConfig`** —— `min_x`/`max_x`/`min_y`/`max_y`、
  `pixels_per_meter`、`max_height`、`split_height`、`hist_max_per_pixel`、
  `use_ground_plane`（false ⇒ 单通道，只要高于分界的点）。
- **`stitch_drive_cameras(left, front, right, config=DriveCameraConfig()) -> np.ndarray`** ——
  三张 `(H,W,3)` BGR → `(3,out_h,out_w)` RGB float `[0,1]`。**`DriveCameraConfig`** ——
  `crop_top`、`crop_bottom`、`crop_side`（只作用于左右两路）、`out_width`、`out_height`。
  缩放发生在拼接**之后**，所以像素会跨接缝插值。
- **`DriveStatus`** —— `driving_command`（one-hot 槽位）、`velocity_x`、`velocity_y`、
  `accel_x`、`accel_y`。**`drive_status_vector(status, num_commands=4)`** 负责打包。
- **`DriveAnchorConfig`** —— 归一化（`x_offset`/`x_span`/`y_offset`/`y_span`）、DDIM 调度
  （`truncated_timestep`、`num_train_timesteps`、`beta_start`、`beta_end`）、`noise_scale`、
  `seed`。**`drive_anchor_noise(anchors, config)`** 应用它；`noise_scale=0` 只把随机项清零
  （便于复现对比），调度自身的缩放仍然保留。
- **`DriveResult`** —— `trajectory`（`DrivePose` 列表）、`agents`（`DriveAgent`：`x`、`y`、
  `heading`、`length`、`width`、`score`）、`bev`（就是 `SegMask`，可直接 `seg_colorize`）。
  **BEV 图的轴不是屏幕轴**：行号随**前方距离**增大（row 0 是自车端），列号随**向左**增大，
  所以要按「上=前方」画必须 `bev.labels[::-1, ::-1]`。上游两个约定都没写，而直路在旋转 180°
  下近似对称——**画反了看着完全合理**。
- **`DriveConfig`** —— `agent_score_thresh`、`bev`（一个 `SegConfig`）。
- **`DriveIoConfig`** —— 哪个张量是哪个，默认值对齐现货导出；把某个输出下标设成 `-1` 即跳过该头。
- **`decode_drive(trajectory=None, agent_states=None, agent_labels=None, bev=None, config=DriveConfig()) -> DriveResult`**
  —— numpy 路径，任一头都可以传 `None`。
- **`DrivePlanner(engine, anchors, config=None, anchor_config=None, io=None)`** ——
  `.plan(camera, lidar, status, timeout_ms=0)`、`.postprocess()`、`.diffusion_input`。

## 深度 Depth

单通道的深度 / 视差转成浮点图，并附着色辅助函数。

```python
cfg = bcdl.DepthConfig(); cfg.normalize = True
est = bcdl.DepthEstimator(engine, cfg)
dm = est.estimate(x)                  # DepthMap
arr = dm.data                         # (H, W) float32
bgr = bcdl.depth_colorize(dm)         # turbo 色图 (H, W, 3) uint8
g8 = bcdl.depth_to_gray8(dm)          # (H, W) uint8

# numpy 路径：
dm = bcdl.decode_depth(out.astype(np.float32), cfg)
```

- **`DepthConfig`** —— `width`、`height`、`normalize`（缩放到 [0,1]）、
  `clip_lo`、`clip_hi`。
- **`DepthMap`** —— `width`、`height`、`vmin`、`vmax`、`data`（`(H,W)` float32）。
- **`decode_depth(output, config) -> DepthMap`**。
- **`depth_colorize(dm) -> np.ndarray`** · **`depth_to_gray8(dm) -> np.ndarray`**。
- **`DepthEstimator(engine, config=None, output_index=0)`** —— `.estimate(model_input, timeout_ms=0)`、
  `.postprocess()`、`.config`。

## 深度精修 Depth refinement（RGB-D）

输入是你**已经有的**深度图——双目或 ToF 的读数，带噪、有空洞——加上对齐的 RGB
帧，输出干净的米制深度和逐像素可信掩膜。这是 LingBot-Depth 头；它不像
`DepthEstimator`（单目）或 `StereoPipeline` 那样从零估计深度。

```python
refiner = bcdl.DepthRefiner(engine)          # 输入：image、depth_log
r = refiner.run(bgr, depth_m)                # BGR (H,W,3) uint8 + (H,W) float32 米
r.depth                                      # (H, W) float32 米，被拒绝处为 0
r.mask                                       # (H, W) uint8，1 = 可信
r.vmin, r.vmax                               # 仅统计可信像素的范围

k = bcdl.Intrinsics(fx, fy, cx, cy)          # 用**传感器原图**的像素单位
k = bcdl.scale_intrinsics(k, bgr.shape[1], bgr.shape[0], r.width, r.height)
pts = bcdl.depth_to_pointcloud(r, k)         # (H, W, 3) float32 米，相机坐标系

# numpy 路径 —— 自己喂图：
img = bcdl.preprocess_refine_image(bgr)      # (1, 3, eh, ew) float32
dep = bcdl.preprocess_refine_depth(depth_m)  # (1, 1, eh, ew) float32
r = bcdl.decode_refined_depth(depth_out, mask_logit_out)
```

- **`DepthRefineConfig`** —— `encoder_height`、`encoder_width`（构造时从模型读取）、
  `min_valid_depth`、`mask_threshold`、`apply_mask`。
- **`RefinedDepth`** —— `width`、`height`、`vmin`、`vmax`、`depth`、`mask`。
- **`Intrinsics(fx, fy, cx, cy)`** · **`scale_intrinsics(k, from_w, from_h, to_w, to_h)`**。
- **`preprocess_refine_image(bgr, config=None)`** · **`preprocess_refine_depth(depth_m, config=None)`**
  —— 与模型标定时**完全一致**的前处理。RGB 降采样走面积平均，换成普通双线性会实测变差。
- **`decode_refined_depth(depth, mask_logit=None, config=None) -> RefinedDepth`**。
- **`depth_to_pointcloud(refined, intrinsics) -> np.ndarray`**。
- **`DepthRefiner(engine, config=None, image_input=0, depth_input=1, depth_output=0, mask_output=1)`**
  —— `.run(bgr, depth_m)`、`.postprocess()`、`.config`。

> 深度是以 **log 深度、0 表示"没有读数"** 的形式进图的，这是上游自己的约定——
> 真实的 1 m 读数和缺失像素落在同一个值上，靠 RGB token 去区分。部署图还保留了
> 全部深度 token，而上游会把"整块 patch 都没有有效读数"的 token 丢掉（变长序列
> 编不出静态图）。在参考场景上这一改动的代价是 0.06% 的绝对相对误差；但那些场景
> 的深度有效率是 87–100%，**极稀疏输入不在实测范围内**。

## 双目 Stereo

两张已校正的图像 → 视差图，可选输出米制深度与有效性掩膜。像素归一化已经融进
`.hbm`；C++ 核心负责贴合 + BGR→RGB + F32 NCHW 打包。
**`fit` 模式必须与模型标定时的做法一致。**

```python
cfg = bcdl.StereoConfig()
cfg.fit = bcdl.StereoFit.Crop      # 或 Resize —— 必须与标定一致
cfg.fx, cfg.baseline = 700.0, 0.12 # 打开米制深度（z = fx*baseline/disp）
cfg.valid_mask = True
pipe = bcdl.StereoPipeline(engine, cfg)

res = pipe.process(left_bgr, right_bgr)
disp = res.disparity.data          # (H, W) float32 视差（一个 DepthMap）
depth = res.depth                  # (H, W) float32 米制深度；关闭时形状为 (0,)
valid = res.valid                  # (H, W) uint8 掩膜；关闭时形状为 (0,)
```

- **`StereoFit`** 枚举 —— `Resize`、`Crop`。
- **`StereoConfig`** —— `input_w`、`input_h`、`fit`、`to_rgb`、`left_index`、
  `right_index`、`output_index`、`fx`、`baseline`、`valid_mask`、`disp_min`、
  `max_disp`、`left_margin`、`lr_check`、`lr_thresh`。
- **`StereoResult`** —— `disparity: DepthMap`、`depth`（numpy，未给 `fx`/`baseline`
  时为空）、`valid`（numpy uint8，`valid_mask=False` 时为空）。
- **`StereoPipeline(engine, config=None)`** —— `.process(left, right) -> StereoResult`、
  `.input_w`、`.input_h`。
- numpy 辅助函数：**`pack_stereo_input(bgr, out_h, out_w, fit=StereoFit.Resize, to_rgb=True)`**、
  **`disparity_to_depth(disp, fx, baseline)`**、
  **`stereo_valid_mask(disp, disp_min=0.0, max_disp=192.0, left_margin=..., lr_check=False, lr_thresh=...)`**。

## OCR

完整的 PP-OCR 三段式流水线（默认 PP-OCRv6，PP-OCRv5 保留为回退），每一段都可以
单独使用。纯解码器（`decode_dbnet`、`decode_cls_dir`、`decode_ctc`、
`load_char_dict`）不需要模型。

```python
chars = bcdl.load_char_dict("ppocr_dict.txt")

det = bcdl.DbTextDetector(engine_det)            # DBNet 检测
boxes = det.postprocess(lb)                      # list[TextBox]，四点旋转框
cls = bcdl.TextAngleClassifier(engine_cls)       # 0/180 方向
rec = bcdl.TextRecognizer(engine_rec, "ppocr_dict.txt")  # CRNN/CTC
# （裁剪每个框，在裁剪图上依次跑 cls 与 rec；见 examples/ocr_demo）

# numpy 解码器：
boxes = bcdl.decode_dbnet(prob.astype(np.float32), bcdl.DbConfig(), lb)
dir_  = bcdl.decode_cls_dir(logits.astype(np.float32), thresh=0.9)
text  = bcdl.decode_ctc(logits.astype(np.float32), chars)
```

- **`DbConfig`** —— `bin_thresh`、`box_thresh`、`unclip_ratio`、`min_size`、
  `connectivity`。
- **`TextBox`** —— `x1, y1, x2, y2`、`score`、`points`（`(4,2)` float，顺时针，
  原图像素）。
- **`RecResult`** —— `text`、`score`。
- **`ClsDirResult`** —— `label`、`score`、`flip180`。
- **`load_char_dict(path) -> list[str]`** —— 每行一个 token。
- **`decode_dbnet(prob, config, letterbox) -> list[TextBox]`** —— 在 `(H,W)` 概率图上
  做连通域 + unclip。
- **`decode_cls_dir(logits, thresh=0.9) -> ClsDirResult`**。
- **`decode_ctc(logits, dict) -> RecResult`** —— 对 `(T,C)` 数组做 CTC 贪心解码。
- 绑定 Engine 的：**`DbTextDetector(engine, config=None, output_index=0)`**、
  **`TextAngleClassifier(engine, thresh=0.9, output_index=0)`**、
  **`TextRecognizer(engine, dict_path, output_index=0)`** —— 各自都有
  `.postprocess(...)`。

完整的 det→cls→rec 串接见 [`examples/ocr_demo.cc`](../examples/ocr_demo.cc)
（裁剪顺序与字典的坑都在那里处理了）。

---

## 语音链总览

做一台能听懂指令、会回话的机器人，需要下面五个部件，网络全部跑在 BPU 上。BCDL 只接收和返回
16 kHz PCM，麦克风采集和扬声器播放由你的代码负责。

```
 麦克风 ──> KeywordSpotter ──唤醒──> VadSegmenter ──整句──> ParaformerAsr ──文字──> CommandMatcher
            （一直在听）              （听到停顿为止）        （一次前向）            （按拼音匹配）
                                                                                       │
 扬声器 <────────────────────────── TextToSpeech <────────── 回复文本 <─────────────────┘
```

| 部件 | 作用 | 板上开销 |
|---|---|---|
| `KeywordSpotter` | 一直监听唤醒词 | 每 160 ms 音频占 BPU 2.0 ms |
| `VadSegmenter`（`SileroVad`） | 唤醒后切出完整的一句话 | 每 32 ms 音频 0.39 ms |
| `ParaformerAsr` | 一句话 → 中文文本 | 每句约 35 ms |
| `CommandMatcher` | 文本 → 你定义的某条指令，或"不是指令" | 几微秒 |
| `TextToSpeech` | 回复文本 → 语音 | 每句约 15–25 ms |

**先跑起来。** [`examples/voice_assistant.py`](../examples/voice_assistant.py)（C++ 版是
[`voice_assistant.cc`](../examples/voice_assistant.cc)）跑通整个流程。不带参数时，它会先用
机器人自己的声音合成一段演示录音：一句不带唤醒词的话、两条指令、一个不是指令的请求。所以板子上
没有麦克风、没有录音也能直接跑：

```bash
scripts/fetch_models.sh                       # 只需一次
python examples/voice_assistant.py            # 或 ./build/voice_assistant
python examples/voice_assistant.py my.wav     # 你自己的 16 kHz 单声道录音
```

```
    time  event
    4.36 s  wake word '你好小车'
    4.60 s  robot: 我在。  (14 ms to synthesise 0.7 s -> voice_out/reply_01.wav)
    5.28 s  heard '向左转' (0.8 s, 34 ms to recognise) -> command: 向左转
    6.30 s  robot: 好的，向左转。  (16 ms to synthesise 1.4 s -> voice_out/reply_02.wav)
    ...
   13.71 s  heard '给我唱首歌吧' (1.1 s, 33 ms to recognise) -> not a command
```

**获取模型。** `scripts/fetch_models.sh` 会下载语音识别模型（D-Robotics 官方预编译的
Paraformer）。另外四个——唤醒词、VAD、Matcha、Vocos——是从上游模型自己转换的，每个在
[bcdl-model-zoo](https://github.com/ruisv/bcdl-model-zoo) 里都有转换配方（`models/kws_zipformer`、
`silero_vad`、`matcha_tts`、`vocos`）：在装有 OpenExplorer 工具链的 x86 机器上导出、校准、
`hb_compile`。转换好后，让获取脚本指向产物所在目录，它会按 BCDL 需要的文件名复制到 `models/`：

```bash
BCDL_SRC_KWS=/path/to/kws_build BCDL_SRC_VAD=/path/to/vad_build \
BCDL_SRC_MATCHA=/path/to/matcha_build BCDL_SRC_VOCOS=/path/to/vocos_build \
  scripts/fetch_models.sh
```

找不到的文件会显示 `MISSING`，并给出对应的配方；依赖它们的测试会自动跳过，不会报错。

主循环很短，关键是这几行：

```python
awake = False
for block in mic_blocks():                      # 长度任意；float32 或 int16，16 kHz 单声道
    if not awake:
        if kws.feed(block):                     # 听到唤醒词
            awake = True; vad.reset()
        continue
    for seg in vad.feed(block):                 # 一整句话
        m = matcher.match(asr.transcribe(seg.pcm).text)
        play(tts.synthesize(f"好的，{m.command}。" if m else "没听懂，请再说一遍。").pcm)
        awake = False; kws.reset()              # 回到待机，换一路新的流
        break
```

**音频输入。** 16 kHz、单声道，浮点 [-1, 1] 或 int16（int16 会按比例换算，不是直接转型）。
其他采样率要先重采样（`ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`，或在声卡驱动层做）。每次送多长
都可以。`KeywordSpotter`、`SileroVad`、`VadSegmenter` 在调用之间保存状态，所以每路音频要用
各自的对象，并按顺序送入。`reset()` 开始一路新的流，`flush()` 结束一路流：不调用 `flush()`，
录音最后一小段里的唤醒词或语句就不会被报告。

**换成你的唤醒词。** 唤醒词是一行模型 token（声母和带声调的韵母），换词不需要训练。先按
`你好小车 @你好小车` 的格式每行写一个短语，再用 sherpa-onnx 的工具转成 token：

```bash
pip install sherpa-onnx
sherpa-onnx-cli text2token --tokens models/kws_zipformer_zh_en_3m_tokens.txt \
    --tokens-type ppinyin keywords_raw.txt keywords.txt
# keywords.txt：n ǐ h ǎo x iǎo ch ē @你好小车
```

唤醒词选三到四个音节、日常说话和指令里都不常出现的；太短或太常见的词容易误唤醒。英文唤醒词要用
上游模型包里的 CMU 格式词典（`en.phone`），并改用 `--tokens-type phone+ppinyin`。

**调参。**

| 现象 | 调整 |
|---|---|
| 唤醒词叫了没反应 | 调低 `KwsConfig.keywords_threshold`（0.25），或只给这一行加 `#0.15` |
| 说相近的词也会唤醒 | 调高阈值，或给这一行加 `:0.5` 降低加分 |
| 一条指令在停顿处被切成两段 | 调大 `VadConfig.min_silence_ms`（300） |
| 机器人回应太慢 | 调小 `min_silence_ms`：它占了延迟的大头，说话人停下后大约要过这么久才判定说完 |
| 指令经常被拒绝 | 把大家实际会说的说法都加成单独的条目；`CommandMatcher(..., max_distance=0.5)` |
| 别的话被当成指令 | 调小 `max_distance` |
| 回复语速太快或太慢 | `MatchaConfig.length_scale`（> 1 变慢） |
| 回复里的停顿太长或被截断 | `TextToSpeechConfig.silence_scale`（0.2；设为 1 保留模型原样） |

**需要知道的几点。**

- **机器人会听到自己的声音。** 播放回复时，暂停把麦克风音频送给 VAD 和唤醒词检测，或者在音频
  栈里做回声消除。
- **中文。** 识别和合成都是中文；合成会跳过英文单词（写"无线网"而不是 WiFi），数字会读出来
  （35% → 百分之三十五）。
- **长度上限。** 一句话最长 24 s（`VadConfig.max_speech_s` 在 20 s 处切开）。回复的单句最长约
  8 s，更长的文本会自动按句子切开合成。
- **下文各节的精度数字**来自合成数据和公开测试集。正式使用前，建议在你的机器人上、电机运转时
  录一百条真实指令，用示例程序跑一遍。

## 语音识别（Paraformer）

输入一段 16 kHz 单声道语音，一次前向就出文字。Paraformer 是非自回归模型，没有逐字解码的循环。
模型编成三段，后两段之间在 CPU 上做 CIF（连续积分发放）：
fbank → LFR → CMVN → **encoder** → **predictor** → CIF → **decoder** → argmax。

```python
asr = bcdl.ParaformerAsr.from_files(
    "models/paraformer_encoder_int16.hbm",
    "models/paraformer_predictor_int16_v2.hbm",
    "models/paraformer_decoder_int16_v2.hbm",
    "models/paraformer_tokens_8404.json",
    "models/paraformer_am.mvn")
r = asr.transcribe(pcm)          # float32，范围 [-1, 1]（int16 也行），16 kHz 单声道
print(r.text, r.truncated)

# 前端和 CIF 也能单独用（不需要模型）：
feats = bcdl.Fbank().compute(pcm)                  # (帧数, 80) Kaldi log-mel
lfr = bcdl.lfr_stack(feats, 7, 6)                  # (ceil(帧数/6), 560)
emb, n = bcdl.cif_integrate(alphas, hidden, valid_frames)
```

**每次只喂一句话，最长 `asr.max_seconds`（24 秒）。**超长的部分会被截掉，并把 `r.truncated`
置为真，所以长音频要先用 VAD 切句。库里没有重采样，调用前请先转成 16 kHz。

前端复现 `torchaudio.compliance.kaldi.fbank`（dither 0）：和 float64 参考实现的误差约 1e-5，和 torchaudio 自身的 float32 结果相差约 2e-4；LFR 与 FunASR 逐位一致。
这么较真是因为：模型是在这套特征上训练的，换成"差不多"的特征不会报错，只会悄悄地识别变差。

- **`FbankConfig`**：`sample_rate`、`num_mel_bins`、`frame_length_ms`、`frame_shift_ms`、
  `preemphasis`、`remove_dc_offset`、`low_freq`、`high_freq`、`input_scale`、`window`。
  `input_scale` 默认 32768，float PCM 会先乘上它，因为 Kaldi 处理的是 int16 量级的采样；icefall / k2
  的模型要设成 1。`window` 为 `"hamming"`（torchaudio 和 FunASR 用的）或 `"povey"`（Kaldi 默认）。
- **`Fbank(config=FbankConfig())`**：`.compute(pcm) -> (帧数, num_mel_bins)`、
  `.num_frames(samples)`、`.frame_length`、`.frame_shift`、`.fft_size`。
- **`lfr_stack(feats, lfr_m=7, lfr_n=6) -> np.ndarray`**：低帧率拼帧，与 FunASR 逐位一致。
- **`Cmvn`**：`dim`、`shift`、`scale`、`.apply(feats)`。`am.mvn` 用
  **`load_kaldi_cmvn(path)`** 或 **`parse_kaldi_cmvn(text)`** 读入。
- **`CifConfig`**：`threshold`（默认 1.0）、`tail_threshold`（默认 0.45）、`max_tokens`。
- **`cif_integrate(alphas, hidden, valid_frames, config=CifConfig())`** →
  `((max_tokens, D) 嵌入, token 数)`。尾部权重放在第 `valid_frames` 帧，也就是语音真正结束的位置。
  定长的 predictor 图只能把它放在补零区的末尾，而补零区随后会被屏蔽掉，尾部权重也就跟着没了，
  结果是凡是最后一个字要靠尾部权重才能发放的句子，都会丢掉这个字。
- **`load_token_list(path) -> list[str]`**：读 JSON 数组（`tokens.json`）或每行一个 token 的文本。
  另有 **`argmax_tokens(logits)`** 和 **`tokens_to_text(ids, vocab)`**（去掉 `<特殊符号>`、
  拼接 `@@` 子词、英文单词之间加空格）。
- **`AsrResult`**：`text`、`token_ids`、`feature_frames`、`truncated`。
- **`ParaformerConfig`**：`fbank`、`lfr_m`、`lfr_n`、`cif`。
- **`ParaformerAsr(encoder, predictor, decoder, vocab, cmvn, config=None)`**：
  传入三个 `Engine`，也可以用 `ParaformerAsr.from_files(encoder, predictor, decoder, tokens, cmvn,
  config=None)` 从文件构建。方法和属性有 `.transcribe(pcm, timeout_ms=0)`、
  `.transcribe_features(feats)`、`.features(pcm)`、`.max_seconds`、`.max_frames`、
  `.feature_dim`、`.max_tokens`、`.vocab_size`。三段模型按张量形状自动对应；
  token 表的大小和 decoder 对不上时会直接报错。


### 语音端点检测（Silero VAD）

在 16 kHz 音频流里找出语音的起止，把每句话完整地交给识别。Silero VAD v6 跑在 BPU 上，每 32 ms
调用一次；`VadSegmenter` 把概率切成一句一句。

```python
vad = bcdl.SileroVad("models/silero_vad_16k_int16_nashm.hbm")
seg = bcdl.VadSegmenter(vad)                 # VadConfig()：阈值 0.5，开 250 ms，关 300 ms
for block in mic_blocks():                   # float32 或 int16，长度任意
    for s in seg.feed(block):                # 本块里结束的语句
        print(s.start / 16000, asr.transcribe(s.pcm).text)
for s in seg.flush(): ...                    # 流结束

segs = seg.segments(recording)               # 整段录音一次处理
p = vad.probabilities(recording)             # 每 512 个采样点一个概率
```

在 100 段带噪会话（每段 30 s：AISHELL-1 语句间隔 0.3–3 s，叠加 10 dB 和 0 dB 的 MUSAN 噪声）和
24 分钟纯噪声上，用默认 `VadConfig` 测得：

| | 帧级 F1，10 / 0 dB | 漏句率 | 判停延迟（中位 / p90，10 dB） | 纯噪声误报 |
|---|---|---|---|---|
| 浮点 Silero（ONNX） | 0.929 / 0.870 | 2.3% / 11.5% | 0.36 / 0.44 s | 0 |
| **int16，BPU** | **0.924 / 0.863** | **3.4% / 13.2%** | **0.36 / 0.43 s** | **0** |

25.3 万个块上两者的语音 / 非语音判定 99.0% 一致（概率平均差 0.012）。板上每块 0.39 ms，
含 STFT 和 Python 开销，占实时的 1.2%。把 `threshold` 调到 0.4，BPU 版的 F1 能追平浮点
（0.929 / 0.872），但判停更晚（0 dB 下 p90 从 0.52 s 变成 0.61 s）。

- **`SileroVad(engine_or_path)`** —— `.probability(chunk)`（512 个采样点 → 概率）、
  `.probabilities(pcm)`（整段录音，从 reset 开始）、`.reset()`、`.chunk`（512）、
  `.sample_rate`（16000）。有状态：LSTM 状态和上一块最后 64 个采样点会带到下一次，
  所以一个对象只读一路音频流，且要按顺序。
- **`VadConfig`** —— `threshold`（0.5）、`neg_threshold`（−1 表示等于 threshold；调低即滞回）、
  `min_speech_ms`（250）、`min_silence_ms`（300）、`speech_pad_ms`（30）、
  `max_speech_s`（20；识别窗口是 24 s）。
- **`VadSegmenter(vad, config=None)`** —— `.feed(pcm)`、`.flush()`、`.segments(pcm)`、
  `.reset()`、`.in_speech`、`.last_probability`。概率连续 `min_speech_ms` 不低于 `threshold`
  时开始一段，连续 `min_silence_ms` 低于 `neg_threshold` 时结束，所以判停大约在说话人停下后
  `min_silence_ms`。
- **`VadSegment`** —— `start`、`end`（自上次 reset 起的采样点，含补边）、`pcm`。

模型自带的 STFT 在 CPU 上算，BPU 只接收幅度谱。整图编译时，实部虚部的平方在 BPU 上做，
功率值域约 5000，int16 会丢掉所有小幅值频点，句子中段的概率掉到 0.2–0.5。

### 唤醒词检测

持续监听唤醒词，听到后再交给 VAD 和识别。模型是 sherpa-onnx 的开放词表 zipformer transducer
（中英文，300 万参数）：唤醒词是一行 token，不需要重新训练模型。编码器在 BPU 上每次处理 160 ms，
解码器、joiner 和关键词搜索在 CPU 上，逐步照搬 sherpa-onnx 的 KeywordSpotter。

```python
kws = bcdl.KeywordSpotter("models/kws_zipformer_zh_en_3m_c16_int16_nashm.hbm",
                          "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                          "models/kws_zipformer_zh_en_3m_tokens.txt",
                          "n ǐ h ǎo x iǎo ch ē @你好小车\n"
                          "x iǎo ch ē x iǎo ch ē @小车小车\n")
for block in mic_blocks():                   # float32 或 int16，长度任意
    for d in kws.feed(block):
        print(d.keyword, d.start, d.end)     # 自 reset() 起的秒数
```

每行唤醒词依次是：模型的 token（中文是声母和带声调的韵母，英文是 ARPAbet 音素），可选的
`:boost` 和 `#threshold`（只作用于这一行），最后是 `@短语`。sherpa-onnx 的 `text2token` 工具
（`pip install sherpa-onnx`）可以把普通文本转成这种格式。

用"你好小车""小车小车"测试（各 202 段，干净和叠加 MUSAN 噪声各一套，9 种合成音色），负样本是
AISHELL-1 语音、语音指令和 2.3 小时带噪会话，另有 350 段易混淆短语（你好小菜、校车校车……），
默认阈值下：

| | 唤醒词召回（干净 / 噪声） | 误唤醒 | 易混淆短语误触发（干净 / 噪声） |
|---|---|---|---|
| 浮点（sherpa-onnx） | 96.5% / 87.1% | 4.1 小时 0 次 | 5.4% / 4.6% |
| **int16，BPU** | **97.5% / 89.1%** | **4.4 小时 0 次** | **5.7% / 4.0%** |

编码器每 160 ms 音频占用 BPU 2.0 ms；加上 CPU 解码，整体实时率 0.023。

- **`KeywordSpotter(encoder, decoder_joiner, tokens, keywords=None, config=None)`** ——
  `keywords` 可以是唤醒词文本，也可以是文件路径。`.feed(pcm)`、`.flush()`（流结束时调用：最后约
  0.3 s 里的唤醒词要靠它才检测得到）、`.set_keywords(text)`、
  `.load_keywords(path)`、`.keywords`、`.reset()`、`.encode(pcm)`（整段录音的编码器输出，测试用）、
  `.token_id(token)`。有状态：一个对象只读一路音频流，且要按顺序。
- **`KwsConfig`** —— `keywords_score`（1.0，每个匹配 token 的加分）、`keywords_threshold`（0.25，
  检测所需的平均 token 概率）、`num_trailing_blanks`（1）、`max_active_paths`（4），与 sherpa-onnx
  默认值相同。想减少误唤醒就调高阈值，想提高召回就调低。
- **`KwsDetection`** —— `keyword`、`index`（第几行）、`start`、`end`、`tokens`。

### 语音指令（固定指令集）

机器人听口令，要的是知道说的是**哪一条**指令，不需要一字不差的转写。`CommandMatcher`
把识别结果和固定指令表放到无声调拼音里比较，同音字（打开灯 → 大开灯、出发 → 处罚）和
句尾多出的语气词（前进 → 前进了）都能吸收，不是指令的话会被拒绝。纯 CPU，每次几微秒。

```python
m = bcdl.CommandMatcher(["前进", "后退", "左转", "右转", "打开灯", "关灯"])
r = m.match(asr.transcribe(pcm).text)    # 比如 "大开灯"
if r:                                    # 即 r.index >= 0
    print(r.index, r.command, r.distance)   # 4 打开灯 0.0
```

在 40 条指令 × 9 个合成音色上，Paraformer 整句完全正确率是干净 84.7% / 5 dB 噪声 82.8%；
这样匹配后是 96.9% / 95.6%，错配到别的指令 0.6% / 1.1%，600 句新闻没有一句被当成指令。

- **`to_pinyin(text) -> list[str]`**：每个汉字一个无声调音节（取最常用读音，ü 写作 `v`）；
  连续的 ASCII 字母算一个小写 token；连续数字按中文读法展开（`"90"` → `jiu shi`）；
  全角字母数字折成半角；其余字符丢弃。
- **`CommandMatcher(commands, max_distance=0.5)`**：距离 = 音节级编辑距离 ÷ 该指令的音节数；
  距离相同时取长度更接近的，再取靠前的。英文单词只和英文单词比（"you" 不等于“右”）。
  方法有 `.match(text) -> CommandMatch`、`.distances(text) -> list[float]`、`.commands`、
  `.max_distance`、`len()`。
- **`CommandMatch`**：`index`（-1 表示拒绝）、`nearest`（被拒绝时也给出最近的指令；文本没有
  任何拼音时为 -1）、`distance`、`command`；接受时布尔值为真。同一意图的多种说法就写成多条。


### 语音合成后端（Vocos 声码器）

mel 谱进、波形出：Vocos 跑在 BPU，逆 STFT 在 CPU。它是非自回归 TTS 的后半段，
前面接一个输出 80 维对数 mel（16 kHz、hop 256）的声学模型，比如 Matcha-TTS zh-en。

```python
voc = bcdl.VocosVocoder("models/vocos_16k_t256_int16_nashm.hbm")
pcm = voc.synthesize(mel)        # (80, frames) float32 对数 mel -> float32 PCM，16 kHz

# 单独用逆 STFT（不需要模型）：
ist = bcdl.Istft()               # n_fft 1024、hop 256、center、周期 Hann 窗
wav = ist.compute(real, imag)    # 都是 (frames, 513)
wav = ist.compute_polar(mag, cos, sin)   # 都是 (513, frames)，即 Vocos 的输出布局
```

编译好的模型一次吃 256 帧（4.1 s）；更长的 mel 切成两边各重叠 32 帧的块，拼接处精确无误差。
每 3 s 音频：BPU 1.3 ms、CPU 6.6 ms。和浮点模型相比，可听部分的对数谱距离 0.54 dB，
Paraformer 识别它的输出一字不差。

- **`IstftConfig`**：`n_fft`（2 的幂）、`hop_length`、`center`。
- **`Istft(config=IstftConfig())`**：`.compute(real, imag)`、`.compute_polar(mag, cos, sin)`、
  `.num_samples(frames)`（center 时为 `(frames - 1) * hop`）、`.bins`。与周期 Hann 窗的
  `torch.istft` 一致；DC 和 Nyquist 的虚部被忽略。
- **`VocosConfig`**：`stft`、`context`（每边重叠帧数，≥ 27 拼接才精确）、`pad_value`
  （两端之外补的对数 mel 值，−16）、`max_magnitude`（100）、`sample_rate`。
- **`VocosVocoder(engine_or_path, config=None)`**：`.synthesize(mel, timeout_ms=0)` 接受
  `(num_mels, frames)` 或 `(1, num_mels, frames)`；属性有 `.num_mels`、`.chunk_frames`、
  `.body_frames`、`.sample_rate`。模型必须按顺序输出 `log_mag` 和 `phase`：`exp`/`cos`/`sin`
  放在 CPU 上算，因为编译器量化后的 `cos`/`sin` 会把相位算坏。


### 语音合成（Matcha-TTS + Vocos）

中文文本进、16 kHz 语音出，所有网络都跑在 BPU 上。非自回归：每句话只跑一次编码器、一次解码器
（三步 flow matching 编译在图里）和一到两块声码器。

```python
tts = bcdl.TextToSpeech.from_files(
    "models/matcha_zh_en_enc_l128_int16_nashm.hbm",
    "models/matcha_zh_en_dec_t512_int16_nashm.hbm",
    "models/matcha_zh_en_embedding.npy",
    "models/vocos_16k_t256_int16_nashm.hbm",
    "models/matcha_zh_en_tokens.txt",
    "models/matcha_zh_en_lexicon.txt")
r = tts.synthesize("好的，马上出发。")      # r.pcm 为 float32，r.sample_rate 为 16000

# 也可以逐段调用：
fe = bcdl.MatchaFrontend("models/matcha_zh_en_tokens.txt", "models/matcha_zh_en_lexicon.txt")
for ids in fe.encode(text):                  # 每句一个 id 列表
    m = acoustic.synthesize(ids)             # MatchaAcoustic -> m.mel (80, frames)
    pcm = vocoder.synthesize(m.mel)
```

在 60 句小车回复上，BPU 整条链路的合成语音用 FireRedASR2 回识别，字错率 1.73%；浮点模型在 CPU
上（sherpa-onnx）是 1.61%。实时率 0.009，平均每句 26 ms。

- **`MatchaFrontend(tokens, lexicon, config=MatchaFrontendConfig())`**：`.encode(text) ->
  list[list[int]]`、`.sentences(text)`、`MatchaFrontend.read_numbers(text)`、`.vocab_size`、
  `.lexicon_size`。与 sherpa-onnx 针对该模型的词典前端逐个 id 一致（标点折叠、切句与合并、
  词典最长匹配、句尾标点 token）。数字先读成汉字（`90` → 九十、`12.6` → 十二点六、
  `35%` → 百分之三十五、`2026年` → 二零二六年）。英文单词会被丢弃——词典里没有英文。
- **`MatchaFrontendConfig`**：`min_words`（5）、`max_words`（20）。
- **`MatchaConfig`**：`noise_scale`（1.0；设为 0 时输出确定）、`length_scale`（> 1 更慢）、`seed`。
- **`MatchaAcoustic(encoder, decoder, embedding, config=None)`**：`.synthesize(ids) ->
  MatchaResult`（`mel` 为 `(80, frames)`、`frames`、`durations`、`truncated`）、`.seed(n)`、
  `.max_tokens`（128）、`.max_frames`（512 = 8.2 s）、`.num_mels`、`.vocab_size`。超长输入会被
  截断并打上标记。token 的词嵌入在 CPU 上从 `.npy` 查表。
- **`TextToSpeech(frontend, acoustic, vocoder, tts_config=None)`** /
  **`.from_files(..., config=None, tts_config=None)`**：`.synthesize(text) -> TtsResult`（`pcm`、
  `sample_rate`、`sentences`、`truncated`）。每句单独合成后拼接，再把每段不短于 0.2 s 的停顿压到
  `TextToSpeechConfig.silence_scale` 倍（默认 0.2，与 sherpa-onnx 相同；设为 1 保留模型原样的停顿，
  标点处偏长）。60 句测试文本的总时长从 225 s 降到 172 s（sherpa-onnx 是 176 s），回识别字错率不变。
- **`scale_silence(pcm, sample_rate, scale, min_pause_s=0.2, level=0.01)`**：单独做停顿压缩
  （`|x| <= level` 算静音），与 sherpa-onnx 逐样本一致。

---

## 跟踪 ByteTrack

不依赖模型：喂进每帧的检测结果，拿回稳定的 track id。

```python
tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
for frame in stream:
    dets = detector.detect(...)              # list[Detection]
    for t in tracker.update(dets):
        print(t.track_id, t.class_id, t.x1, t.y1, t.x2, t.y2)
```

- **`ByteTrackConfig`** —— `track_thresh`、`high_thresh`、`match_thresh`、
  `track_buffer`、`frame_rate`；外观相关：`proximity_thresh`、
  `appearance_thresh`、`ema_alpha`；以及 `boost`（`BoostConfig`）。
- **`Track`** —— `track_id`、`x1, y1, x2, y2`、`score`、`class_id`；带 `__repr__`。
- **`ByteTracker(config=None)`** —— `.update(detections) -> list[Track]`、
  `.update(detections, embeddings) -> list[Track]`、
  `.apply_camera_motion(affine)`、`.reset()`、`.config`。

### 外观（ReID）

每个检测各给一个嵌入向量，就会打开外观关联：代价变成
`min(IoU 距离, 门控后的余弦距离)`，所以外观只可能**挽救**几何本来要漏掉的匹配，
**永远不会破坏**几何本来已经拿到的匹配。某一项传**空**表示"这个检测没有外观"，
这正是你在廉价裁剪上跳过 ReID 模型的方式。

```python
reid = bcdl.ReIDExtractor(bcdl.Engine("osnet.hbm"))
tracker = bcdl.ByteTracker(bcdl.ByteTrackConfig())
for frame in stream:
    dets = detector.detect(...)
    embs = reid.embed_detections(frame, dets, min_score=0.5)   # 与 dets 一一对应
    for t in tracker.update(dets, embs):
        print(t.track_id)
```

- **`ReIDExtractor(engine, config=None, output_index=0)`** —— `.dim`、
  `.embed(model_input)`、`.embed_crop(crop_bgr)`、
  `.embed_detections(frame_bgr, detections, min_score=0.0)`。
- **`reid_preprocess(crop_bgr, width=128, height=256, config=None)`** 与
  **`reid_crop_preprocess(bgr, x1, y1, x2, y2, in_w, in_h, config)`** ——
  **挤压式** resize（不是 letterbox：这类模型就是在挤压过的裁剪上训练的）、
  BGR→RGB、ImageNet 归一化、NCHW float32。
- **`ReidConfig`** —— `mean`、`std`。

模型是**每个裁剪跑一次**，所以人多的帧里决定帧耗时的是它而不是检测器。注意
**int8 PTQ 对 OSNet 是不够的** —— 见 `CHANGELOG.md` 里 0.4.0 的条目。

### 相机运动

`apply_camera_motion(affine)` 在下一次 `update()` 之前，用一个把上一帧映射到当前帧的
2x3 变换去 warp 每条轨迹。位置与尺寸会被 warp，速度不会 —— 因为速度描述的是目标
在世界中的运动，而相机抖一下并不等于目标在加速。静止相机请**直接不调用**，
而不是传一个单位矩阵。

```python
M, _ = cv2.estimateAffinePartial2D(prev_pts, cur_pts, method=cv2.RANSAC)
tracker.apply_camera_motion(np.ascontiguousarray(M, np.float32))
```

### BoostConfig

BoostTrack++ 的增补项，**默认全部关闭** —— 每一项都可独立开关，这样它的贡献是
被测出来的，而不是假设出来的。

- `rich_similarity` —— 在代价里加入马氏距离与形状一致性项；
  `lambda_iou`、`lambda_mhd`、`lambda_shape`、`min_iou`。
- `soft_biou` —— 按 `1 - 轨迹置信度` 同时放大两个框，让一条已经在惯性滑行的轨迹
  搜索更大的范围。
- `boost_detections` —— 在高/低分切分之前，抬高被轨迹背书的检测分；
  `dlo_alpha`、`vt_start`、`vt_end`、`vt_steps`、`duo`、`duo_iou`。
  **看场景**：当检测器的瓶颈是漏检时它能捞回真检测，否则它制造假轨迹。

## TrackingPipeline

一次调用完成检测 + 跟踪；预处理全在 C++ 里。需要一个 NV12 输入的 YOLO `.hbm`。

```python
pipe = bcdl.TrackingPipeline(engine)          # det_config、track_config 可选
for frame in stream:                          # frame: HxWx3 uint8 BGR
    for t in pipe.process(frame):
        print(t.track_id, t.class_id, t.score)
print(pipe.last_detections)                   # 上一帧关联之前的检测
```

- **`PipelineConfig`** —— `input_w`、`input_h`、`detect`（`DetectConfig`）、
  `output_index`、`pad_value`、`head`（`DetectHead`）、`ltrb_strides`。
- **`DetectHead`** 枚举 —— `Auto`、`SingleTensor`、`YoloLtrb`。
- **`TrackingPipeline(engine, det_config=None, track_config=None,
  reid_engine=None, reid_config=None)`** —— `.process(bgr) -> list[Track]`、
  `.last_detections`、`.has_reid`、`.last_embed_count`、`.reset()`。传入
  `reid_engine` 即为原生路径加上外观；裁剪尺寸从那个模型里读。
- **`TrackingReidConfig`** —— `min_score`、`max_crops`、`crop`（`ReidConfig`）。
  两者都是成本旋钮：ReID 模型每个合格裁剪跑一次，`last_embed_count` 会告诉你
  这一帧跑了多少次。

## AsyncDetectionPipeline

流式检测：后面帧的 CPU 预处理与前面帧的 BPU 推理+解码重叠。`submit()` / `next()`
会阻塞（背压 / 等待）但**释放 GIL**。结果按提交顺序返回。需要一个 NV12 输入的
YOLO `.hbm`。

```python
cfg = bcdl.PipelineConfig(); cfg.detect.num_classes = 80
pipe = bcdl.AsyncDetectionPipeline(engine, cfg, depth=3)

for i, frame in enumerate(stream):            # frame: HxWx3 uint8 BGR
    pipe.submit(frame)                        # 队满则阻塞
    if i >= 3:
        for d in pipe.next():                 # 按提交顺序
            ...
pipe.finish()
while (dets := pipe.next()) is not None:
    ...                                       # 排空在途的帧
```

- **`AsyncDetectionPipeline(engine, config=None, depth=3)`**：
  - `submit(bgr) -> bool` —— 入队（字节已拷贝，数组可立即复用）。队满时阻塞；
    `finish()` 之后返回 `False`。
  - `next() -> list[Detection] | None` —— 弹出下一个结果；已 finish **且**排空后
    返回 `None`。
  - `finish()` —— 标记流结束（幂等；GC 时也会执行）。
  - `head -> DetectHead` —— 解析出来的解码器家族。
  - `profile() -> StageProfile` —— 各阶段的*服务*时间
    （`preproc_ms`/`infer_ms`/`postproc_ms` 总量 + `*_per_frame()`）；最慢的那一段
    决定吞吐上限。在 `finish()` + 排空之后再读。

## AsyncVideoDetectionPipeline

整条"压缩视频 → 检测结果"的链路都在 C++ 里（`解码 ‖ nv12→bgr ‖ 预处理 ‖
推理+NMS`，四段重叠）。**Python 只负责搬字节** —— 喂 Annex-B 分片（例如
`ffmpeg -c copy` 出来的 RTSP/mp4 流）并读走检测结果；所有解码/转换/检测线程都在
C++ 里跑且释放 GIL。这正是一个很薄的 Python 驱动也能打到 C++ 解码上限
（**1080p H.264 约 441 FPS**）、而不是被 Python 编排的解码循环卡在约 81 FPS 的原因。

```python
cfg = bcdl.PipelineConfig(); cfg.detect.num_classes = 80
pipe = bcdl.AsyncVideoDetectionPipeline(engine, cfg, bcdl.VideoType.H264, depth=4)

while chunk := ffmpeg.stdout.read(65536):     # ffmpeg -c copy（不做软解）
    pipe.submit(chunk)                        # AU 切分 + VPU 解码都在 C++ 里
    while (dets := pipe.next_nowait()) is not None:
        ...                                   # 把已就绪的取走
pipe.finish()
while (dets := pipe.next()) is not None:      # 阻塞式排空最后几帧
    ...
```

- **`AsyncVideoDetectionPipeline(engine, config=None, codec=VideoType.H264, depth=4)`**：
  - `submit(bytes) -> bool` —— 喂 **Annex-B** 字节；背压时阻塞。
    MP4 **不是** Annex-B（AVCC 长度前缀、SPS/PPS 藏在 `avcC` box 里），直接喂会
    一帧都出不来。请先解封装 —— 只拆容器、不碰像素，VPU 仍是唯一的解码器：
    `ffmpeg -i in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 -`。参见
    [`examples/video_det_demo.py`](../examples/video_det_demo.py) 里的 `load_annexb()`。
  - `next() -> list[Detection] | None` —— 按解码顺序阻塞弹出。
  - `next_nowait() -> list[Detection] | None` —— 非阻塞弹出。
  - `finish()`、`profile() -> StageProfile`（含 `decode_ms`、`cvt_ms`）。
- 视频解码支持 **H.264 与 H.265**（分层 GOP 的 HEVC 流只会解出基础时域层 ——
  见下文 VideoDecoder 的说明）。
- 参见 [`examples/rtsp_det_demo.py`](../examples/rtsp_det_demo.py) —— 基于它写的
  一个很薄的 RTSP 驱动（ffmpeg 管道 → `submit` → `next_nowait`）。

### 视频示例（mp4/h264 进 → 检测 → mp4 出）

[`examples/video_det_demo.py`](../examples/video_det_demo.py) 是端到端的 Python
示例：**VPU** 解码 → `AsyncDetectionPipeline`（BPU）→ 画框 → **VPU** 编码 →
`.mp4`。它能读裸 `.h264/.h265` **或** `.mp4/.mov`（MP4 会用 `ffmpeg -c copy`
解封装成 Annex-B，真正的解码仍由 VPU 做），并把 VPU 出的裸流用 `ffmpeg -c copy`
封装成 `.mp4`（只封装容器，不重编码）。它会打印各阶段的 `profile()` 分布。

```bash
python examples/video_det_demo.py det.hbm in.mp4 out.mp4          # mp4 -> mp4
python examples/video_det_demo.py det.hbm in.h264 out.mp4 300 4   # [max_frames] [depth]
```

注意：解码与编码共用同一个 VPU 核，所以整条回环会比只解码的检测路径慢
（1080p yolo26n 约 104 FPS，瓶颈在编码，3.56 ms/帧；只解码的检测路径约 441 FPS）。

---

## 图像与编解码

统一的共享内存图像，加上 JPU（JPEG）与 VPU（H.264/H.265）硬件编解码器。它们会
分配带缓存的设备缓冲并驱动多媒体单元，所以只能在板上运行。

```python
# JPEG（JPU）—— 一次性辅助函数：
jpg = bcdl.jpeg_encode(bgr, quality=80)       # bytes
img = bcdl.jpeg_decode(jpg)                    # VpImage（NV12）
planes = img.to_numpy()                        # 见下面 VpImage.to_numpy

# 处理一条流时请复用解码器（每次调用都新建会多约 5 ms 的 JPU 初始化）：
dec = bcdl.JpegDecoder()
for blob in blobs:
    vp = dec.decode(blob)

# H.264 / H.265（VPU）：
ec = bcdl.VideoEncConfig(); ec.type = bcdl.VideoType.H264
ec.width, ec.height, ec.bitrate_kbps, ec.framerate = 1280, 720, 4000, 30
enc = bcdl.VideoEncoder(ec)
chunk = enc.encode(vp_image)                   # bytes（被缓冲时可能为空）

dc = bcdl.VideoDecConfig(); dc.type = bcdl.VideoType.H265   # 或 .H264
dec = bcdl.VideoDecoder(dc)
frame = dec.decode(nal_bytes)                  # VpImage；仍在缓冲时为 None
# 解耦的喂入/取出（正确处理 H.265 重排序）：喂任意字节，按显示顺序取，
# 流结束时把重排序尾巴 flush 出来。
dec.feed(chunk_bytes)                           # 入队字节（不需要自己切 AU）
while (f := dec.receive(0)) is not None: ...    # 非阻塞取（0 = 不等待）
while (f := dec.flush()) is not None: ...        # 最后一次 feed 之后：重排序尾巴
```

> **H.265 说明：** 解码器走的是正确处理重排序的 `media_codec` API，所以 H.265
> 可用（早期那种"逐 AU"的模型在它上面会超时）。**分层 GOP** 的 HEVC 流
> （海康 SVC / "H.265+"/smart-codec）只会解出**基础时域层**（约 1/4 的帧）——
> 这是 SoC SDK 的限制；要满帧率请关掉相机的 SVC 模式。H.264 与非分层 HEVC
> 都能完整解出。

- **`ImageFormat`** 枚举 —— `Y`、`NV12`、`RGB`、`BGR`。
- **`VpImage(width, height, format)`** —— `width`、`height`、`format`、`valid`；
  `.to_numpy()` 按设备行 stride 拷出：`BGR/RGB -> (H,W,3)`、`Y -> (H,W)`、
  `NV12 -> 一维 (W*H*3//2)`（先 Y 平面，后交织的 UV）。
- **`vp_image_from_bgr(bgr) -> VpImage`** · **`vp_image_from_nv12(nv12, width, height) -> VpImage`**。
- **`JpegEncoder(width, height, quality=50, format=NV12)`** —— `.encode(VpImage) -> bytes`
  （宽需 16 对齐，高需 8 对齐）。
- **`JpegDecoder(out_format=NV12)`** —— `.decode(bytes) -> VpImage`。一条流上请
  **复用同一个实例**；每次调用都新建要花约 5 ms。
- **`jpeg_encode(bgr, quality=50) -> bytes`** · **`jpeg_decode(data) -> VpImage`**
  —— 一次性的便捷包装（需要 OpenCV，且尺寸为偶数）。
- **`VideoType`** 枚举 —— `H264`、`H265`。
- **`VideoEncConfig`** —— `type`、`width`、`height`、`bitrate_kbps`、`framerate`、
  `intra_period`、`format`。
- **`VideoEncoder(config)`** —— `.encode(frame: VpImage) -> bytes`（编码器把这帧
  缓冲起来时返回空）；`.type/.width/.height/.format`。
- **`VideoDecConfig`** —— `type`、`format`、`in_buf_size`。
- **`VideoDecoder(config)`** —— `.decode(data) -> VpImage | None`（还在缓冲参考帧
  时为 `None`）；`.type/.format`。

---

## C++ API

C++ 接口面与上面一一对应；配置与结果结构体共用同样的字段名（camelCase）。
完整参考见 [`CPP_API.zh.md`](CPP_API.zh.md)，入口是
[`include/bcdl/bcdl.h`](../include/bcdl/bcdl.h) 以及各领域的头文件：

| 领域 | 头文件 |
|------|--------|
| core（SysMem、Task、Status、MemPool） | [`core/`](../include/bcdl/core/) |
| backend（Engine、输出读取器） | [`backend/engine.h`](../include/bcdl/backend/engine.h) |
| 预处理（letterbox、VpImage） | [`preproc/`](../include/bcdl/preproc/) |
| 多媒体（JPEG/视频编解码） | [`media/`](../include/bcdl/media/) |
| 任务（det/cls/pose/seg/obb/depth/ocr） | [`tasks/`](../include/bcdl/tasks/) |
| 跟踪 | [`tracks/byte_tracker.h`](../include/bcdl/tracks/byte_tracker.h) |
| 流水线 | [`pipeline/`](../include/bcdl/pipeline/) |

错误通过 `BCDL_CHECK(...)` 抛出 `bcdl::Error`。可运行的程序见
[`examples/`](../examples/)。

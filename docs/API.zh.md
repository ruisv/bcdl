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
- 流式与跟踪：[跟踪 ByteTrack](#跟踪-bytetrack) · [TrackingPipeline](#trackingpipeline) ·
  [AsyncDetectionPipeline](#asyncdetectionpipeline) ·
  [AsyncVideoDetectionPipeline](#asyncvideodetectionpipeline)
- 多媒体：[图像与编解码](#图像与编解码)

> 命名：snake_case 的函数名与 `lower_case` 的配置字段，是 C++ camelCase 成员的
> Python 写法。少数解码器另有 camelCase 别名（`decodeDepth`、`decodeSeg`），
> 以便与 C++ 名字对齐。

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
| `input_shape(i) / output_shape(i) -> list[int]` | 张量形状。 |
| `input_bytes(i) -> int` | 输入 `i` 分配到的设备缓冲大小（已按 BPU stride 对齐）。 |
| `input_packed_bytes(i) -> int` | 输入 `i` 作为连续行主序数组的大小。模型对某一维做了补齐时会小于 `input_bytes(i)`。 |
| `input_stride(i) -> list[int]` | 输入 `i` 解析后的字节 stride，最内层在最后。 |
| `output_stride(i) -> list[int]` | 输出 `i` 的字节 stride。**输出同样会补齐** —— 把原始缓冲直接摊平 reshape 会把张量剪切错位。 |
| `input_dtype(i) / output_dtype(i) -> str` | NumPy dtype 字符串（如 `"float32"`、`"float16"`、`"int8"`）。 |
| `infer(inputs, timeout_ms=0) -> list[np.ndarray]` | 把输入拷进设备缓冲、执行、每个输出返回一个数组。`timeout_ms=0` 表示阻塞。 |

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
- **`decode_seg(output, config) -> SegMask`**（别名 `decodeSeg`）。
- **`seg_colorize(mask) -> np.ndarray`** —— `(H,W,3)` uint8 固定调色板。
- **`Segmenter(engine, config=None, output_index=0)`** —— `.segment(model_input, timeout_ms=0)`、
  `.postprocess()`、`.config`。

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
- **`decode_depth(output, config) -> DepthMap`**（别名 `decodeDepth`）。
- **`depth_colorize(dm) -> np.ndarray`** · **`depth_to_gray8(dm) -> np.ndarray`**。
- **`DepthEstimator(engine, config=None, output_index=0)`** —— `.estimate(model_input, timeout_ms=0)`、
  `.postprocess()`、`.config`。

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

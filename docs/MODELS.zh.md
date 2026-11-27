# 模型清单

[English](MODELS.md) | **简体中文**

示例、测试与基准套件所用的已编译 BPU 模型（`.hbm`）清单。每个模型**是什么、从哪来、
该取哪个 build、遵循什么许可证**，只在本文件里记录一处 —— 代码和其他文档都链接到这里，
而不是各自重述一遍。

本文件刻意**不放实测数据**：耗时、模型体积和内存占用都在
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)，它由 `scripts/board_bench.py`
在干净的板子上生成。同一组数字手工维护两份，迟早会对不上。

- [模型总表](#模型总表)
- [该取哪个 build](#该取哪个-build)
- [获取模型](#获取模型)
- [Hugging Face 上的预编译模型](#hugging-face-上的预编译模型)
- [怎么读激光雷达校验图](#怎么读激光雷达校验图)
- [许可证](#许可证)

## 模型总表

**来源**一列取以下三者之一：**板载**（随板子系统镜像提供，位于 `/opt/hobot` 下）、
**zoo**（D-Robotics `rdk_model_zoo` 的预编译模型），或**转换**（用 OpenExplorer
工具链从上游 ONNX 离线编译 —— 本仓库不分发，见[获取模型](#获取模型)）。

| 任务 | 文件 | 上游 | 输入 | 来源 |
|---|---|---|---|---|
| OCR —— 检测 | `ppocrv6_medium_det_960x960.hbm` | PP-OCRv6 medium det | 960×960 | 转换 |
| OCR —— 识别 | `ppocrv6_medium_rec_int16_48x320.hbm` | PP-OCRv6 medium rec (int16) | 48×320 | 转换 |
| OCR —— 方向分类 | `ppocrv5_lcnet_cls_80x160.hbm` | PP-LCNet textline ori | 80×160 | 转换 |
| OCR —— 检测（v5） | `ppocrv5_server_det_960x960.hbm` | PP-OCRv5 server det | 960×960 | 转换 |
| OCR —— 识别（v5） | `ppocrv5_server_rec_48x320.hbm` | PP-OCRv5 server rec | 48×320 | 转换 |
| 检测 | `yolo26s_det_nashm_640x640_nv12.hbm` | YOLO26s | 640×640 | 转换 |
| 检测 | `yolo26n_detect_nashm_640x640_nv12.hbm` | YOLO26n | 640×640 | zoo |
| 检测（DFL 头） | `yolov8_640x640_nv12.hbm` | YOLOv8 | 640×640 | 板载 |
| 分类 | `yolo26n_cls_nashm_224x224_nv12.hbm` | YOLO26n-cls | 224×224 | zoo |
| 姿态 | `yolo26n_pose_nashm_640x640_nv12.hbm` | YOLO26n-pose | 640×640 | zoo |
| 实例分割 | `yolo26n_seg_nashm_640x640_nv12.hbm` | YOLO26n-seg | 640×640 | zoo |
| 旋转框 | `yolo26n_obb_nashm_640x640_nv12.hbm` | YOLO26n-obb | 640×640 | zoo |
| 语义分割 | `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | DeepLabV3+ | 2048×1024 | 板载 |
| 语义分割（实时） | `pidnet_s_nashm_1024x2048_nv12_v3.hbm` | PIDNet-S | 2048×1024 | 转换 |
| 语义分割（YOLO26 原生分辨率） | `yolo26n_sem_nashm_640x640_nv12.hbm` | YOLO26n-sem | 640×640 | 转换 |
| 单目深度 | `depth_any.hbm` | Depth-Anything-V2 | 686×518 | zoo |
| 双目视差 | `las2_m_crop2_nashm.hbm`、`las2_m_resize2_nashm.hbm` | Lite Any Stereo V2 (M) | 640×480 | 转换 |
| 双目视差（更准） | `las2_l_crop2_nashm.hbm`、`las2_l_resize2_nashm.hbm` | Lite Any Stereo V2 (L) | 640×480 | 转换 |
| RGB-D 深度精修 | `lingbot_depth_v05_int16_nashm.hbm` | LingBot-Depth v0.5 (MDM, DINOv2 ViT-L/14 RGBD) | 640×480 RGB + 深度 | 转换 |
| 开放词汇检测 | `yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm` | YOLOE-11s | 640×640 | 转换 |
| 开放词汇分割 | `yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm` | YOLOE-11s | 640×640 | 转换 |
| 室内物体 + 掩膜（99 类） | `yoloe26m_indoor99_seg_nashm_640x640_nv12.hbm` + `yoloe26m_indoor99_labels.txt` | YOLOE-26m，固定的室内词表（见[下文](#室内-yoloe-build-是零样本的)） | 640×640 | 转换 |
| 场景分类（365 类场景） | `places365_resnet50_nashm_224x224_nv12.hbm` + `categories_places365.txt`、`IO_places365.txt` | Places365 ResNet50 (CSAIL) | 224×224 | 转换 |
| 可提示分割 | `edge_sam_encoder_nashm.hbm` | EdgeSAM 编码器 | 1024×1024 | 转换 |
| 可提示分割 | `edge_sam_decoder_sp1_nashm.hbm` | EdgeSAM 解码器，1 个点 | — | 转换 |
| 可提示分割 | `edge_sam_decoder_bp2_nashm.hbm` | EdgeSAM 解码器，框（2 个点） | — | 转换 |
| 人脸检测 | `scrfd_10g_nashm_640x640_nv12.hbm` | SCRFD-10G (insightface) | 640×640 | 转换 |
| 人脸识别 | `arcface_r50_aligned_nashm_112x112.hbm` | ArcFace R50 (insightface) | 112×112 | 转换 |
| 行人 ReID | `osnet_ain_qat_nashm_256x128.hbm` | OSNet-AIN | 256×128 | 转换 |
| 全身姿态 | `vitpose_s_wholebody_nashm_256x192.hbm` | ViTPose-S，133 关键点 | 256×192 | 转换 |
| 全景驾驶感知 | `yolop_cut_nashm_640x640_nv12.hbm` | YOLOP | 640×640 | 转换 |
| 端到端驾驶 | `diffusiondrive_r34_256x1024_s100p.hbm` | DiffusionDrive (ResNet-34) | 1024×256 相机 + 256×256 激光雷达 | zoo（下载） |
| 光流 | `neuflow_v2_int16_nashm_512x384.hbm` | NeuFlow v2 | 512×384 ×2 帧 | 转换 |
| 异常检测 | `glass_bottle_int8_nashm_288x288.hbm` | GLASS (WideResNet-50) | 288×288 | 转换 |
| 激光雷达 3-D 检测 | `pointpillars_vfe.hbm` + `pointpillars_backbone.hbm` + `pointpillars_head_int16_nashm_248x216.hbm` | PointPillars (KITTI) | 11546 个 pillar → 496×432 BEV | zoo + 转换的头 |
| 激光雷达 3-D 检测 —— 样例帧 | `models/kitti_frame/`（点云 + 图像 + 标定 + 标注） | 一帧 KITTI，从你自己的副本拷入 | — | **不分发**（见[怎么读激光雷达校验图](#怎么读激光雷达校验图)） |
| 稀疏特征点 | `xfeat_nashm_640x480.hbm` | XFeat | 640×480 | 转换 |
| 超分 ×4 | `realesr_general_x4v3_nashm_128.hbm` | Real-ESRGAN Compact | 128×128 分块 | 转换 |
| 超分 ×4 | `spanx4_ch48_nashm_128.hbm` | SPAN ch48 | 128×128 分块 | 转换 |
| 图像嵌入 | `bpu-siglip-base-patch16-224.hbm` | SigLIP base/16 | 224×224 | zoo（下载） |
| 语音识别 —— 编码器 | `paraformer_encoder_int16.hbm` | Paraformer-large，中文（contextual） | 400 个 LFR 帧（24 s）× 560 | zoo（下载） |
| 语音识别 —— 预测器 | `paraformer_predictor_int16_v2.hbm` | Paraformer-large CIF 预测器 | 400 × 512 | zoo（下载） |
| 语音识别 —— 解码器 | `paraformer_decoder_int16_v2.hbm` | Paraformer-large 解码器，8404 个 token | ≤ 100 个 token | zoo（下载） |
| 语音识别 —— 配套文件 | `paraformer_tokens_8404.json`、`paraformer_am.mvn` | token 表（取自该 build）+ CMVN（取自上游模型） | — | 下载 |
| 唤醒词 —— 编码器 | `kws_zipformer_zh_en_3m_c16_int16_nashm.hbm` | sherpa-onnx zipformer2 KWS zh-en 3M（2025-12-20），带 38 个缓存张量的流式编码器；图为 BPU 做过改写（见下文） | 每次调用 29 个 fbank 帧（步长 160 ms） | 转换 |
| 唤醒词 —— 配套文件 | `kws_zipformer_zh_en_3m_decoder_joiner.npy`、`kws_zipformer_zh_en_3m_tokens.txt` | decoder + joiner 权重合成的一张表（在 CPU 上跑）、token 表 | — | 转换 / 上游 |
| 语音端点检测 | `silero_vad_16k_int16_nashm.hbm` | Silero VAD v6.2.3，16 kHz；从幅度谱开始的单个流式步（STFT 在 CPU 上），LSTM 状态作为输入和输出 | 每次调用 512 个采样点（32 ms） | 转换 |
| 语音合成 —— 编码器 | `matcha_zh_en_enc_l128_int16_nashm.hbm` | Matcha-TTS zh-en 文本编码器 + 时长预测器；输入为 embedding | 128 个 token | 转换 |
| 语音合成 —— 解码器 | `matcha_zh_en_dec_t512_int16_nashm.hbm` | Matcha-TTS zh-en U-Net 解码器，3 步 flow-matching 已编译进图 | 512 帧（8.2 s） | 转换 |
| 语音合成 —— 配套文件 | `matcha_zh_en_embedding.npy`、`matcha_zh_en_tokens.txt`、`matcha_zh_en_lexicon.txt` | token embedding 表（CPU 查表）+ 前端 token 表与词典 | — | 转换 / 上游 |
| 语音合成 —— 声码器 | `vocos_16k_t256_int16_nashm.hbm` | Vocos 16 kHz（n_fft 1024，hop 256），Matcha-TTS zh-en 配套的声码器；图止于 log 幅度 + 相位 | 80 mel × 256 帧（4.1 s；更长的输入分块处理） | 转换 |

代码**只**通过仓库内相对路径 `models/<name>.hbm` 引用上面每一个模型，每个任务都可以用
环境变量覆盖（`BCDL_MODELS`、`BCDL_OCR_DET`……）。库、示例和测试里不出现任何绝对路径。

**OCR 默认用 PP-OCRv6**；v5 那几行保留作回退，也是为了方向分类器 —— v6 没有发布这个
模型。v6 识别模型特意用**全 int16** build：在 S100P 上，它比编译器默认的混合精度 int8
build 字错率约减半，*而且*快约 2 倍，因为全 int16 的图在边界上没有 int8↔int16 的 requant。
对于长宽比超过约 6.7:1 的文本行，还有一个 960 宽的 v6 识别 build
（`ppocrv6_medium_rec_int16_48x960.hbm`，延迟约为 2.4 倍），把 `BCDL_OCR_REC` 指向它即可。
识别器从模型里读取类别数，所以字典从 18385 换到 18710 不需要改代码 —— 只换了
`data/ppocr_keys_v6_18710.txt`。所有 v6 build 的配方都在配套仓库 `bcdl-model-zoo` 里。

**`yolo26n_sem` 发布的是在真实 Cityscapes 帧上校准的 int16，这两个选择缺一不可。**
这个模型的全 int8 build 编译毫无报错，逐张量的余弦闸门全部通过（每层 >0.95，输出 ≥0.98），
逐像素的类别判定却只有 61% 与浮点模型一致 —— 整类整类地互换。逐像素 19 路 softmax 的
稠密分类器对决策边界很敏感，这是单个汇总余弦看不出来的。只靠 `set_all_nodes_int16`
能到 85% 一致；剩下的差距原来是校准数据的域问题，而不是量化 —— 改用真实 Cityscapes 帧
重新校准（而不是拿一般街景数据充当）后，一致率提高到 **96%**。完整细节见该配方的 README。

**光流模型用 int16，因为 int8 会变成一层 2 像素的噪声底，而没有哪个余弦闸门会报出来。**
光流不需要数据集就能对着精确真值打分 —— 把一张图平移已知距离，正确的流场就是处处等于
这个常量 —— 在 S100P 上这样测得：

| build | 静止图对的噪声底 | 已知 2–16 px 平移上的 EPE | 纯 BPU | 端到端 |
|---|---|---|---|---|
| 浮点（ONNX） | 0.025 px | 0.036–0.076 px | — | — |
| int8 | **1.936 px** | 1.61–2.15 px | 14.4 ms | 21.3 ms |
| int16 | **0.070 px** | 0.055–0.081 px | 19.3 ms | 26.3 ms |

int8 的误差是一层**平的噪声底**：平移 2 像素和平移 16 像素，得到的误差都是约 1.9 px，
所以快速运动的像素看上去没问题（相对误差约 12%），慢速的则全是噪声。这在编译器的输出
余弦里完全看不出来 —— int8 build 的输出余弦是 **0.9954**，幅值大的场吸收 2 px 噪声，余弦
几乎不动。int16 多花 34% 的延迟，精度基本回到浮点水平。**当一个头回归的是大动态范围的
物理量时，要用那个量的单位打分，而不是用余弦。**

**异常检测模型用 int8，理由与光流模型恰恰相反 —— 这正是要点所在。** 它的 int8 build
编译出来输出余弦只有 **0.9409**，远低于通常的闸门，分数图与浮点模型的逐像素平均差为
0.024。但按任务真正衡量的东西打分 —— 官方 MVTec-AD `bottle` 测试集，83 张带像素级真值
的图 —— 它与浮点没有区别：

| build | 图像 AUROC | 像素 AUROC | 输出余弦 | 纯 BPU | 端到端 |
|---|---|---|---|---|---|
| 浮点（ONNX） | 0.9905 | 0.9907 | — | — | — |
| **int8（发布版）** | **0.9905** | **0.9905** | 0.9409 | 72.5 ms | 81.0 ms |
| int16 | 0.9897 | 0.9905 | 0.9918 | **6619.9 ms** | 6626 ms |

AUROC 基于排序，量化噪声把每个分数都挪一点、但保持排序不变时，就没有任何代价 —— 这就是
发布 int8 的原因。

> **表中 int16 的延迟未经核实，而且很可能是错的。** 它是在板子同时跑着一个无关服务的
> 时段测的，而在那种状态下，*同一个*模型背靠背重复测，结果最多能差 46 倍 —— 正是本仓
> 已经记录过的板端计时陷阱。在这里引用任何 int8 对 int16 的延迟比之前，先在空闲的板子上
> 用 min-of-N 把两个 build 都重测一遍。AUROC 数字不受影响：它们是结果闸门，不是计时。

和上面光流那一行放在一起看，两者正好成对：**0.9954 的余弦藏着一个不可用的模型，0.9409
的余弦什么也没藏。** 无论往哪个方向，余弦都不能代表任务质量。

延迟主要花在骨干上（81 ms 中占 72.5 ms）；解码里的高斯平滑另占 6.3 ms，设
`blur_kernel = 0` 可以省掉，代价是与参考后处理不再一致。

**驾驶模型的 int16 误差表现为模式翻转，而不是漂移。** 对照上游 FP32 模型，在 120 个真实
nuScenes 关键帧上测量，两边喂的是逐比特相同的张量：轨迹余弦 **0.9994**，4 s 时域内终点
偏差中位数 **0.47 m**，BEV 语义 argmax 一致率 **94.4%**，车辆检测在 **90.5%** 的槽位上
一致。但 120 帧里有 2 帧偏差超过 2 m，而最差的一帧（6.2 m）并不是同一个规划漂移了，而是
换成了 20 个候选模式中的另一个。在这些模式上取 argmax 发生在编译好的图里，所以在接近
平局时，量化噪声会换掉整条轨迹，而不是轻推它一下。要按这条长尾留余量，而不是按均值。
（作为参照，这些帧上对照真实未来轨迹的开环 L2 在 1 s 时为 0.28 m、3 s 时为 1.85 m ——
但这是一个 NAVSIM 训练的模型在 nuScenes 上评估，传感器配置不同，所以它衡量的是跨域迁移，
而不是模型本身的精度。）

**语音识别模型若 CIF 处理不对，会吞掉很多句子的最后一个字 —— 修复在 CPU 上，不在 build
里。** Paraformer 的预测器会在语音结束处之后加一个小小的"尾巴"权重，好让最后一个 token
能触发。定长的预测器图只知道补零后的长度，所以把尾巴加在了补零段的末尾；随后对补零段做
掩膜时，尾巴就被清掉了。`ParaformerAsr` 会在真实的语音结束处重新加上尾巴。在 600 条随机
抽取的 AISHELL-1 测试语句上，以及其中最短的 300 条上，两种运行方式喂的是同一个 int16 build：

| | 浮点（FunASR） | int16，尾巴丢失 | int16，`ParaformerAsr` |
|---|---|---|---|
| AISHELL-1，随机 600 条 | 3.41% | 4.57% | 3.78% |
| AISHELL-1，最短 300 条 | 1.76% | 3.88% | 2.72% |
| 同样的 600 条 + MUSAN 噪声，5 dB | 5.08% | 6.20% | 5.58% |

（字错率，按字计；打分前两边的数字都转写成中文。）补零到 24 s 窗口、把上下文偏置
（contextual-bias）输入置零，都不是原因 —— 两者都在浮点下做过消融，没有可测量的变化。
`ParaformerAsr` 的前端不加 dither，所以转写结果是确定的；在这些测试集上这没有带来任何损失。

**TTS 声学模型要做两处图改写，定长 build 才与原模型等价。** Matcha 按每句自己的长度运行
（向上取整到 4 的倍数）。补零到固定的 512 帧后，有两处会改变结果：解码器的 GroupNorm
在整条时间轴上做归一化，补零段也算在内；它的注意力掩膜是 0/1 的浮点数，
`scaled_dot_product_attention` 把它*加*到分数上 —— 有效帧 +1，补零帧 +0，于是补零段只是
被降权，而不是被排除。这两处都不在乎补零段里装的是什么，只在乎补了多少，所以"输出对
补零段的取值是否敏感？"这项检查会通过，而输出在 log-mel 上最多能差 10。发布的解码器恰好
只对未补零模型看到的那些帧做归一化和注意力，在浮点下与原模型吻合到 9e-5。另外，token id
不能作为图的输入：编译器会像对待任何激活值一样量化浮点 id 输入（它为一个 2190 项的词表
选了 127 的范围），这让编码器的时长输出余弦只有 0.83。因此 embedding 查表改在 CPU 上做，
编码器输出达到 0.99999。时长是 `ceil(exp(logw))`，所以 int16 build 会改动某个 token 的
时长：在 19 句话的 147 个 token 中，有 30 个与浮点不同 —— 时长恰好落在整数边界上的 token
翻动一帧，长停顿 token（8–58 帧）移动 1–3 帧 —— 而没有哪句话的总长度变化超过 2 帧。在
同样的 60 句话上打分：

| | 回环字错率（FireRedASR2） | UTMOS | 板端实时率 |
|---|---|---|---|
| 浮点 Matcha + Vocos，CPU（sherpa-onnx，4 线程） | 1.61% | 3.29 | 0.085 |
| int16 Matcha + Vocos，BPU，停顿保持模型原样 | 1.73% | 3.33 | 0.0070 |
| int16 Matcha + Vocos，BPU（`TextToSpeech`，像 sherpa-onnx 那样把 ≥ 0.2 s 的停顿缩短到 0.2×） | 1.73% | 3.29 | 0.0092 |

BPU 链平均每句 26 ms（最慢的一段是两句话的文本，58 ms），其中解码器图 8.2 ms、编码器
1.2 ms。模型在标点处的停顿偏长 —— 60 句话共 225 s 音频，而 sherpa-onnx 是 176 s，因为
sherpa-onnx 把每个 0.2 s 及以上的停顿缩短到原长的 0.2×。`TextToSpeech` 默认也这么做
（172 s）；实时率上升只是因为音频变少了。

**声码器用 int16，其相位不在 BPU 上计算。** Vocos 的末端是对幅度做 `exp`、对一个跨度约
±42 rad 的原始相位做 `cos`/`sin`。按发布原样编译时，编译器把 `cos`/`sin` 降为量化的范围
归约（range reduction），相位对浮点的余弦在 int8 下只有 0.58，int16 下 0.97–0.99 —— 足以
听得出来。发布的图切在头部的线性输出处，所以 BPU 返回 log 幅度和相位，其余交给 CPU。在
板上每个 192 帧的块（3.1 s 音频）：BPU 1.3 ms，exp/cos/sin 2.5 ms，iSTFT 4.1 ms —— 共
7.9 ms。在 60 句话上对照浮点模型打分，只统计每句峰值 40 dB 以内的时频单元：

| | 对数谱距离 | 偏差 > 3 dB 的单元 | 回环字错率（FireRedASR2） | 每 256 帧块的 BPU 耗时 |
|---|---|---|---|---|
| 浮点，整句 | — | — | 1.98% | — |
| 浮点，按板上方式分块 | 0.01 dB | 0.0% | 1.98% | — |
| int8 | 5.38 dB | 49.8% | 2.10% | 0.76 ms |
| **int16（发布版）** | **0.54 dB** | **0.3%** | 2.48% | 1.29 ms |

int8 build 仍然听得懂 —— 识别器几乎察觉不到 —— 但你听到的内容有一半偏差超过 3 dB，这就是
一层噪声底；int16 的差别是 60 句话里多了一处同音字（它 → 他）。分块本身是精确的：网络
两侧最多只看 27 帧，块之间重叠 32 帧，接缝处与整句一次推理逐比特相同。含 iSTFT 在内的
实时率为 0.0026。

**唤醒词编码器要做四处图改写，编译器才肯编译它；其中一处不改，int16 就毫无意义。**
发布的流式编码器接收一个 int64 的 `processed_lens`，再把它转成布尔注意力掩膜；这一步改成
在 CPU 上计算的浮点 `pad_mask` 输入，所有作用在它上面的 `Where` 都改写为
`x·(1−m) + c·m`。Swoosh 里的 softplus `log(1 + exp(s))` 让 `exp(s)` 跨越许多个数量级，
量化后余弦为 0，把整个编码器拖到 −0.21；改写成 `relu(s) + log(1 + exp(−|s|))` 后，其中的
`exp` 落在 (0, 1] 内。相对位置平移是六个带常量索引的 `GatherElements` —— 每一行都是一段
连续区间，偏移取决于查询位置 —— 改成了逐查询的切片。有十个 depthwise 卷积的核长 15、
补边 7，而输入只有 1–4 帧，代码生成器无法处理（"invalid storage"）；在 L 帧上，只有中间
2L−1 个抽头会碰到数据，所以把核裁剪到这些抽头。每处改写都在浮点下与发布的图对照过
（编码器输出相差在 1.5e-4 以内，87 段音频上检测结果相同）。优化器的某个 pass 会让
onnxruntime 在这个图上崩溃，编译时把它关掉了。以 int16 编译，校准数据是真实的流式状态
（唤醒词片段、易混淆短语、带噪语音、噪声）。在 87 段音频上流式推理并回灌缓存，编码器
输出对浮点的逐帧平均余弦为 0.993，个别帧低至 0.5–0.65；87 段中有 86 段的检测结果与浮点
一致。余弦在这里说明不了什么：重要的是 joiner 的 tanh 和 softmax 很少改变判定。在完整
测试集上：

| | 唤醒词检出率，干净 / 噪声 | 误唤醒 | 易混淆短语误触发率，干净 / 噪声 |
|---|---|---|---|
| 浮点（sherpa-onnx） | 96.5% / 87.1% | 4.1 h 内 0 次 | 5.4% / 4.6% |
| **int16，BPU** | **97.5% / 89.1%** | **4.4 h 内 0 次** | **5.7% / 4.0%** |

**VAD 的频谱在 CPU 上计算。** Silero VAD 开头是一个写成带 stride 卷积的 STFT，后接
`sqrt(re² + im²)`。整图编译时，平方项的跨度约 5000，int16 的分辨率是 0.15 一档，每个
安静的频点都被舍入掉，句子中间的语音概率塌到 0.2–0.5。发布的图从幅度谱开始（4 帧 ×
129 个频点，由 CPU 上的 FFT 计算）。在 13,736 个留出音频块上流式推理，它的概率与浮点
平均相差在 0.015 以内，98.4% 的时间做出相同的语音 / 非语音判定。`hb_compile` 为这个模型
打印的输出余弦说明不了什么：它们只在第一个校准样本上计算（有一个 build 对某个状态打印了
0.925，而该状态在全部样本上的中位数是 0.9998），而概率只是单个数，永远打印 1.0。校准请用
`calibration_type: max`：在一套新的校准集上，默认搜索选了 KL，把频谱输入截断在 1.8，而
语音能到 13 —— 编译毫无报错，最坏情况误差却翻倍。
在 100 段带噪的 AISHELL-1 会话和 24 分钟 MUSAN 噪声上，使用默认分段器（阈值 0.5，250 ms
开启，300 ms 关闭）：

| | 帧级 F1，10 / 0 dB | 漏掉的语句，10 / 0 dB | 判定语音结束的滞后（中位数，10 dB） | 纯噪声上的误报语音 |
|---|---|---|---|---|
| 浮点（ONNX） | 0.929 / 0.870 | 2.3% / 11.5% | 0.36 s | 0 |
| **int16，BPU（发布版）** | **0.924 / 0.863** | **3.4% / 13.2%** | **0.36 s** | **0** |

校准集是真实的流式状态 —— 每个块实际到达时带着的 LSTM 状态 —— 而不是零状态。

## 该取哪个 build

**模型的文件名与其内部的模型名一致，这是刻意的。** 上面有好几项都有兄弟 build，区别只在
校准方式或图的切点，而用错了不会大声报错 —— 它照样编译通过、能加载，解出来的却是乱码，
或者精度悄悄下降。后缀标明的是你手上是哪个 build，不是可以忽略的噪音：

| 后缀 | 模型 | 另一个 build 会怎样 |
|---|---|---|
| `_aligned` | ArcFace | 在**对齐后**的人脸裁剪上校准。同形状、在中心裁剪上校准的 build 是另一个文件，公开的精度数字是对齐版的。 |
| `_cut` | YOLOP | 官方导出通过 `ScatterND` 把 anchor 解码烘进图里。这样编译不报任何错，产出的模型 objectness 列和类别列却**从不被写入**。可行的做法是在解码之前切图，输出三个原始头。 |
| `_v3` | PIDNet-S | 更早的 build 用的校准数据**没有**预先归一化。当校准数据是 float32 时，编译器的 `norm_type` 不作用于它，于是输入阈值是错的 —— 同样是编译毫无报错。 |
| `_qat` | OSNet-AIN | int8 PTQ 在这个网络上完全不行：编译毫无报错，返回的单位向量格式规整，Market-1501 Rank-1 却只有 51%，而浮点模型是 85%。发布的 build 靠 QAT 自蒸馏恢复到 84.6%。 |
| `_s100p` | DiffusionDrive | 第一个**按板子**而不是按 march 发布的预编译模型 —— 还有一个 `_s600` 兄弟版。它也是直接来自上游的全 int16 build，所以平时"后缀优先取 int8"的读法在这里不适用。配套的 anchor 文件不是可选的：没有 `kmeans_navsim_traj_20.npy`，扩散解码器就没有东西可去噪。 |
| `_crop2` / `_resize2` | Lite Any Stereo V2 | 两种校准模式 —— 中心裁剪适配（centre-crop-fit）与缩放适配（resize-fit）。按你对图对做 letterbox 的方式来选；本仓的双目测试用的是中心裁剪的图对，优先用 `_crop2`。后缀里的 `2` 指的是校准**范围**。输出层的 int16 阈值由校准图对中包含的最大视差决定，而更早的 `_crop` / `_int16` build 是在没有近景的场景上校准的，所以它们**把视差截断在 131 px 和 47.7 px** —— 比这更近的东西都会读成更远，而且不报任何错。`2` 系列 build 加入了同样图对的副本，并把右视图左移，这样每个视差都增加了平移量而内容不变，阈值因此覆盖了模型完整的 0–192 px。第二个原因叠加在第一个之上：编译器按 8 个一批做校准，而对这个图，它只从每批的第一对取输出范围，所以 `2` 系列 build 还把每个校准图对重复了 8 次。在留出的 ETH3D 与 Middlebury 2021 场景上对照真值测量：crop（33 个场景）下，最差场景相对浮点模型的误差从 14.7 px 降到 0.30–0.43 px，落在旧范围以内的场景为更宽的范围付出约 0.07 px；resize（51 个场景）下，EPE 从 2.17 px 降到 0.60 px，Middlebury 的 bad-3px 比例从 17.3% 降到 3.6%。 |
| `_int16` | LingBot-Depth | int8 PTQ 在这个网络上撑不住。编译毫无报错，返回的深度图格式规整，动态范围却塌了 —— 2.9–13.8 m，而浮点模型是 0.97–45.8 m；平均绝对相对误差 233%，余弦 0.53，掩膜 IoU 0.56。逐层报告说明了原因：LayerNorm 和 softmax 的 `Mul` 节点余弦只有 0.006，1418 个节点中有 1056 个低于 0.90。24 层的 ViT-L 正是"全 int16"这条规则要应对的情形。 |
| token 数 | LingBot-Depth | 上游网络没有固定分辨率：token 预算加上长宽比决定编码器网格。发布的 build 用 1200 个 token，即 30×40 网格、420×560 的编码器输入，解码器金字塔恰好落在 480×640，所以最后的 resize 是恒等变换。更大的预算（最多 3600）能换来精度，代价是注意力的平方级开销 —— 而这个图已经是本清单中最慢的模型了。 |

### LAS2 选 M 还是 L

同一网络家族，同样只用 2D 的代价聚合，没有迭代；L 在每个尺度上的聚合深度是 M 的两倍。
在板上以 int16 测量，对照不在校准集中的 ETH3D 与 Middlebury 2021 场景的真值（crop 33 个，
resize 51 个；两套场景不同，所以只在同一模式内比较，不要跨模式比较）：

| build | EPE | bad 1 px | bad 2 px | 延迟 | 体积 |
|---|---|---|---|---|---|
| `las2_m_crop2_nashm.hbm` | 0.998 px | 14.2% | 6.0% | 15.0 ms / 67 FPS | 40 MB |
| `las2_l_crop2_nashm.hbm` | 0.928 px | 11.0% | 5.0% | 24.0 ms / 42 FPS | 71 MB |
| `las2_m_resize2_nashm.hbm` | 0.597 px | 8.5% | 3.4% | 15.0 ms / 67 FPS | 41 MB |
| `las2_l_resize2_nashm.hbm` | 0.568 px | 6.7% | 2.9% | 24.0 ms / 42 FPS | 71 MB |

延迟是一次推理加 Python I/O，单线程。收益全在室内和近距离：在 Middlebury 场景上，crop 的
bad-1px 从 19.2% 降到 14.5%（resize：从 14.4% 降到 10.7%），而在 ETH3D（视差小）上两者持平。
近景细节重要、且 42 FPS 够用时选 L；否则选 M。

### PointPillars 有两条链，其中一条快 46 倍

`fetch_models.sh` 取到的是官方发布的各段，一帧点云要 **403.7 ms 的 BPU 时间**。其中 VFE
占 290 ms，而它只是一个约 95 MFLOP 的 4→64 MLP —— 折合约 0.3 GFLOPS，这个数字本身就说明了
问题。它不是受算力限制，而是受布局限制：它的输入是 `(11546, 32, 4)`，巨大的 pillar 轴对着
4 宽的通道轴，这是能交给卷积引擎的最差形状。

同样的运算重新导出为**在 `(1, 10, P, 32)` 上的 1×1 卷积**，对点取 max 用池化窗口实现，
只要 **1.62 ms** —— 逐点 MLP 变成了通道卷积，特征扩展挪到 CPU 上
（`pillarPointFeatures()`，1.1 ms）。从同一个 checkpoint 把骨干和头重新导出成一个 int8 图，
耗时从 113.7 ms 降到 7.20 ms。合起来：

| | 官方发布 | 重新导出 |
|---|---|---|
| VFE | 290.0 ms | 1.62 ms |
| 骨干 + 头 | 106.2 + 7.5 ms | 7.20 ms（一个图） |
| **BPU 合计** | **403.7 ms** | **8.82 ms** |
| 端到端，本仓的链 | 468.8 ms | **43.0 ms** |

**量化要对照这条链*自己的*浮点打分，而不是对照另一条链。** 两条链的权重不同，所以两者之间
的任何比较都把"checkpoint 不同"和"int8 代价"混在一起，读起来比实际更乐观。对照它自己的
浮点 ONNX，在样例点云上，这条链找到 18 个框，浮点也是 18 个；匹配上的中心相差 0.07 m，
分数整体低 0.035，浮点打分高于 0.3 的 anchor 有 93% 保留了下来。与官方链剩下的差别
（18 个框对 16 个，其中 13 个相距 1 m 以内）来自 checkpoint，而不是量化。

有两条校准事实是关键的，都经过实测：

- **头要用不止一帧来校准。** 第一个 build 只用了一帧点云，丢了 18 个框中的 4 个，分数被
  压低 0.135，两个朝向翻转了 π —— 而且*降低阈值也找不回来*，因为退化的是排序（板端/浮点：
  阈值 0.30 时 14/18，0.15 时 24/34）。用八帧真实点云校准，框头余弦从 0.897 升到 0.9745，
  方向头从 0.848 升到 0.9584，**延迟完全不变**。
- **对卷积形式的 VFE，int8 胜过 int16**（相对其自身浮点为 0.9988 对 0.9567），这与通常的
  方向相反。

KITTI mAP 则*没有*定论，无论往好还是往坏的方向：这些是单帧保真度数字，回答的是"板子能否
复现浮点模型"，而不是"这个检测器有多好"。

**这些 build 不随本库分发。** 它们是从一个 OpenPCDet checkpoint 重新导出的，权重在 KITTI
上训练，而 KITTI 是 CC BY-NC-SA —— 所以随库发布的是运行时和转换配方，不是权重。请自己
编译，以 `pointpillars_vfe_conv_int8_nashm.hbm` 和 `pointpillars_bbhead_int8_nashm.hbm`
为文件名放进 `models/`，或者把 `BCDL_LIDAR_VFE` / `BCDL_LIDAR_HEAD` 指向任意位置。没有任何
逻辑按文件名分派：只有一个输入的 VFE 被当作卷积形式，输入是 scatter 画布的头被当作已融合
了骨干。

这样一来，整条链的大头是**主机到设备的拷贝，而不是计算** —— 55 MB 的 BEV 画布交给融合图
要约 12 ms，15 MB 的展开 pillar 张量约 5 ms，而真正的 BPU 工作只有 8.8 ms。

表中还有两处选择**不是**一好一坏的关系：

- **Real-ESRGAN Compact 与 SPAN** —— Compact 是感知型，在模糊或压缩过的输入上胜出；SPAN
  是保真型，在干净输入上胜出，体积只有前者的六分之一。两个都留着。
- **128×128 的超分分块是刻意的。** 编译出的 `.hbm` 主体是指令流而不是权重，它随分块
  **面积**增长 —— 同一个网络在 256×256 下是 148 MB 的模型，这里是 37 MB，每像素吞吐完全
  相同。如果运行时本来就分块，就用小分块。

要确认一次 fetch 真的拿到了这些 build：清空 `models/`，重跑 fetch 脚本，再跑测试套件。

### 室内 YOLOE build 是零样本的

`yoloe26m_indoor99_seg` 是把**固定的 99 类词表**折进检测头的 YOLOE-26m：前十类是导航
地标（门、开着的门、玻璃门、电梯、电梯门、自动扶梯、楼梯、扶手、出口标志、小心地滑标志），
其后是家居物品（家具、家电、电子产品、餐具、卫浴设施、灯具与装饰）。类别顺序就是
`yoloe26m_indoor99_labels.txt` 的行序。没有做任何训练：每个类都是 YOLOE 对该提示词的开放
词汇先验。它通过普通的 `InstanceSegmenter` 解码（框和掩膜）；只需要框时请关掉
`compute_masks`，因为拼装掩膜花的 CPU 时间远多于 BPU 跑网络的时间。

这换来了什么、付出了什么 —— 对照留出的真值测量，而不是从余弦上读出来的：

- **整体上与有监督检测器持平，但不是逐类持平。** 在 1952 张 Open Images V7 室内图像上
  （validation + test 划分，都没有用于校准），在至少有 20 个框的 25 个类上，板端 build 的
  mAP50 是 0.484，浮点模型 0.478，而在 Open Images 本身上训练的 Ultralytics YOLOv8s 是
  0.481 —— 后者还占着熟悉这些图像标注惯例的便宜。YOLOE 在床、马桶、家电和瓶子上领先，在
  门上持平（0.44 对 0.42），在楼梯（0.33 对 0.45）、窗户、搁架、橱柜和书上落后。
- **屏幕是盲区。** 在尝试过的每一种提示词下（`computer monitor`、`monitor`、
  `computer screen`、`lcd monitor`……），电脑显示器的分数都接近零，即使词表只有三类也一样。
  这是模型的问题，不是词表的问题。
- **导航类的分数偏低。** 拿 Places365 场景图像当"是否出现"的标签，图像级 AP 是自动扶梯
  0.74、电梯 0.46、楼梯 0.35，但在常用的 0.25 阈值下，只有约五分之一的电梯图像有任何检测。
  对这些类请用更低的阈值，并在你自己的帧上验证；在漏掉楼梯会出问题的场合，请用专门在楼梯上
  训练过的检测器。
- **int8 会挪动分数接近的类。** 板端 build 总体上不比浮点差，但单个类会双向变动（楼梯和
  窗户上升；百叶窗和瓶子下降），因为量化会翻转某个格子上两个接近的类谁胜出。两边都找到的
  实例，掩膜高度吻合（平均 IoU 0.93）。

`places365_resnet50` 返回 365 个 Places365 场景上的 logits；场景名取自
`categories_places365.txt`，室内/室外标记（1 = 室内）取自 `IO_places365.txt`。按训练时的
方式预处理 —— 拉伸到 256×256 再中心裁剪 224×224，不要用 letterbox（对 4:3 的帧做
letterbox，浮点下 top-1 会掉 3.6 个百分点）。在 36,380 张未用于校准的 Places365 验证图像上，
板端 build 得到 54.5% top-1 / 84.6% top-5（浮点：54.6% / 85.0%）；365 类的场景标签天生
有歧义，所以 top-5 是更有用的数字。

## 获取模型

`.hbm` 文件**不纳入版本库**（太大；`.gitignore` 排除了 `*.hbm`）。在板子上用下面的命令
填充 `models/`：

```bash
scripts/fetch_models.sh
```

这个脚本是本地源路径唯一出现的地方 —— 在 `scripts/local.env` 里设置它们（已被 gitignore；
见 `scripts/local.env.example`）。没有设置的项会回退到占位路径，并直接报告为 `MISSING`。

模型转换（ONNX → `.hbm`、PTQ 校准）用 D-Robotics OpenExplorer 工具链**在 x86 主机上
离线**完成。完整的转换配方 —— ONNX 导出、校准、`hb_compile` 配置，以及精度/延迟数字 ——
在配套的 [**bcdl-model-zoo**](https://github.com/ruisv/bcdl-model-zoo) 仓库里，每个模型
一个目录，位于 `models/<name>/`。上面每个**转换**项在那里都有配方：`las2`、`ppocr_v6`、
`ppocr_v5`、`osnet`、`yolop`、`pidnet`、`superres`、`span`、`xfeat`、`yoloe`、
`yoloe26_indoor`、`places365`、`vitpose`、`yolo26`、`yolo26_sem`、`face`、`edgesam`，
外加 `pointpillars` —— 它是那里唯一一个产物完全不在上表中的配方，因为它的权重不能再分发
（见上文的两条链）。
**板载**和 **zoo** 项是预编译的下载，不是转换产物，所以它们的"配方"就是来源 URL，由
`scripts/fetch_models.sh` 下载，而不是重新构建。

每个 `.hbm` 都为特定的 BPU **march** 编译，所以请运行为你的板子编译的 build —— S100 和
S100P 共用 Nash march，S600 有为它自己编译的版本。BCDL 运行时本身在整个 S 系列上完全相同。

## Hugging Face 上的预编译模型

上面大多数**转换**行都以可直接加载的 `.hbm` 形式发布，所以你可以完全跳过工具链 —— 不用
导出 ONNX、不用校准、不用 `hb_compile`、不用 x86 主机。`scripts/fetch_models.sh` 会自动
拉取它们；列在这里，是为了让你也可以手动下载单个模型。

| 仓库 | 内容 | 大小 | 许可证 |
|---|---|---|---|
| [`ruisv/bcdl-ppocr-v6`](https://huggingface.co/ruisv/bcdl-ppocr-v6) | OCR 检测 + 识别 + 方向分类，**附 18710 类字典** | 49 MB | Apache-2.0 |
| [`ruisv/bcdl-ppocr-v5`](https://huggingface.co/ruisv/bcdl-ppocr-v5) | OCR v5 检测 + 识别 + 分类，附其 18385 类字典 | 61 MB | Apache-2.0 |
| [`ruisv/bcdl-las2-stereo`](https://huggingface.co/ruisv/bcdl-las2-stereo) | 零样本双目，M 和 L，两种校准模式 | 213 MB | MIT |
| [`ruisv/bcdl-lingbot-depth`](https://huggingface.co/ruisv/bcdl-lingbot-depth) | RGB-D 深度精修，int16 | 1015 MB | Apache-2.0 |
| [`ruisv/bcdl-superres`](https://huggingface.co/ruisv/bcdl-superres) | ×4 超分，Compact **和** SPAN | 43 MB | BSD-3 / Apache-2.0 |
| [`ruisv/bcdl-xfeat`](https://huggingface.co/ruisv/bcdl-xfeat) | 稀疏特征点 | 3 MB | Apache-2.0 |
| [`ruisv/bcdl-glass-bottle`](https://huggingface.co/ruisv/bcdl-glass-bottle) | 单类异常检测，MVTec `bottle` | 92 MB | **CC BY-NC-SA 4.0** |
| [`ruisv/bcdl-vitpose-wholebody`](https://huggingface.co/ruisv/bcdl-vitpose-wholebody) | 133 关键点全身姿态 | 28 MB | **非商用** |
| [`ruisv/bcdl-pidnet`](https://huggingface.co/ruisv/bcdl-pidnet) | 实时语义分割 | 18 MB | **非商用** |
| [`ruisv/bcdl-yolop`](https://huggingface.co/ruisv/bcdl-yolop) | 全景驾驶感知，3 个头 | 11 MB | **非商用** |

```bash
# 手动下载单个模型
curl -fL -o xfeat_nashm_640x480.hbm \
  https://huggingface.co/ruisv/bcdl-xfeat/resolve/main/xfeat_nashm_640x480.hbm
```

**这里只收录*权重*可再分发的模型**，这和上游仓库挂的许可证徽章是两回事 —— 见
[许可证](#许可证)。这也是这份列表比总表短的原因：源自 Ultralytics 的 build 是 AGPL，人脸
模型和 PointPillars 的*权重*是非商用的，而 **NeuFlow v2**、EdgeSAM 和 OSNet 尚无定论。
这些只提供配方，`fetch_models.sh` 仍会从你自己的转换产物中把它们拷入。

NeuFlow 最能说明问题：Apache-2.0 的代码，*权重就提交在同一个仓库里*，通常这就足以定论
—— 但它是在 FlyingThings3D 上训练的，而该数据集的条款仅限研究使用，并且对用这些数据训练
出的模型**只字未提**：既没有把限制延伸到模型，也没有给出许可。沉默不等于许可，所以尽管
它表面上看起来处处宽松，光流模型仍然只提供配方。

标为**非商用**的行可以再分发，可以用于研究；不得用于商业用途。每个模型卡都在顶部的横幅
里说明原因，而不是放在脚注里。GLASS 另外带有**相同方式共享**（share-alike）条款（MVTec AD
是 CC BY-NC-SA 4.0） —— 不过要注意，这一限制不会延伸到在你自己的零件上重新训练后的部署，
而这正是这个模型实际的预期用法。

## 怎么读激光雷达校验图

`benchmarks/figures/lidar3d.jpg` 已提交到仓库，所以不用板子也能看。本节说明它画的是什么，
以及如何在你自己的数据上重新生成 —— 这是看检测器是否正常工作最快的方法。

### 图里画的是什么

**一帧激光雷达点云**的两个视图，外加一张记分卡。

**上方 —— 相机图像，叠加模型输出的 3-D 框。** 这些框不是在图像平面上画出来的；它们是
检测器自己的 3-D 输出，经该帧的标定（`P2 · R0_rect · Tr_velo_to_cam`）投影而来。正因如此，
这才是决定性的视图：一个框要么以正确的尺寸和朝向包住车，要么明显没有。俯视的占地轮廓
可能掩盖一个低了半米或转了 180° 的框；投影出的立方体掩盖不了。每个框的**正面涂了阴影**，
朝向一目了然。

**左下 —— 同样的框叠在点云上，鸟瞰视角。** 点按**高度**着色，色阶拉伸到该帧高度的第
2–98 百分位（右侧色条标注的是实际米数）。高度着色能区分一个框是落在车上，还是落在墙或
树篱上。圆环间隔 10 m，自车按比例画在原点，+x 朝上。

**右下 —— 记分卡**，前提是该帧有标注。

| 颜色 | 含义 |
|---|---|
| 绿色 | **与标注匹配上**的检测 |
| 粉色，虚线 | 什么都没匹配上的检测 —— **误检** |
| 琥珀色，虚线 | **没有被任何检测找到**的标注 —— 漏检 |
| 灰色 | **在相机视野之外**的检测，画出来但不计分 |

最后一行很重要，也很容易弄错。KITTI 只标注相机能看到的东西，而检测器拿到的是完整的 360°
点云 —— 所以在自车后方检测到的车没有标注，但*不是*错误。图中把它画成灰色，并排除在计算
之外，KITTI 自己的评测也是这么做的。

匹配用的是鸟瞰 IoU，采用 **KITTI 自己的阈值** —— 车 0.7，行人和骑车人 0.5 —— 所以
"匹配上"的含义与基准中一致，而不是挑一个足够宽松、能让图好看的标准。

> **一帧不算评测。** 这里打印的召回率和精确率只描述这一帧点云。它们是用来发现哪里坏了的，
> 不是用来报告检测器有多好的；那需要 KITTI 验证集划分和官方评测。

### 在你自己的帧上重新生成

校验图已提交，但**数据集没有**。KITTI 是 CC BY-NC-SA，所以它需要的四个文件要从你已有的
KITTI 副本中拷出来：

```bash
# scripts/local.env  （已被 gitignore）
BCDL_SRC_KITTI=/path/to/kitti/object/training   # 内含 velodyne/ image_2/ calib/ label_2/
BCDL_KITTI_FRAME=000736                         # 任选一个帧号
```

然后在板子上：

```bash
scripts/fetch_models.sh                          # 放置 models/kitti_frame/
PYTHONPATH=build:python python scripts/board_bench.py --tasks lidar3d
```

它会写出 `benchmarks/figures/lidar3d.jpg`。放置好的目录就是：

```
models/kitti_frame/
  velodyne.bin     点云                          （必需）
  calib.txt        P2, R0_rect, Tr_velo_to_cam  （必需）
  image_2.png      左侧彩色相机                  （相机视图需要）
  label_2.txt      真值                          （可选 —— 没有标注就没有记分卡）
  frame_id.txt     标题里要打印的内容
```

**任何标定好的传感器组合都行**，不只是 KITTI：按这个布局写出你自己的四个文件，图就会随之
生成。加载器假定的是 KITTI 的约定，这部分才是要弄对的 —— 激光雷达是 x 向前 / y 向左 /
z 向上，标注文件用的是*相机*坐标系，`y` 在框的**底部**，旋转 `ry` 绕相机的向下轴。
`tests/kitti_frame.py` 在依赖每条约定的那一行都写了说明。

**没有这些也照样能用。** 没有放置任何帧时，图回退为仅鸟瞰视图，由 `fetch_models.sh` 从
model zoo 下载的样例点云驱动 —— 没有相机视图，也没有记分卡，因为未标定的点云两者都支撑
不了。

### 框的位置不对时

先跑几何测试 —— 它们能把标定错误和模型错误区分开，而且开销很小：

```bash
PYTHONPATH=build:python pytest -s tests/test_kitti_frame.py
```

它们先用一个可以手算的合成标定检查投影，再检查你的标注和检测器对车的位置是否一致 —— 两条
独立的路径通向同一个答案，所以一致就说明两者都对。

## 许可证

上面的模型都来自第三方，带有**各自的许可证**。BCDL 自身的代码（本仓库，Apache-2.0）与它们
无关：它是一个能加载任意 `.hbm` 的通用运行时。下面的许可证约束的是**模型权重及其编译出的
`.hbm`**，所以在再分发或商用发布之前，请逐一核对。

**仓库的许可证不等于权重的许可证。** 这是最常见的出错方式，也是为什么下面有好几行同时带着
宽松的徽章和一条限制。一个 checkpoint 可能受到代码许可证从未提及的两样东西约束：它**微调
所基于的预训练权重**（ViTPose 从 MAE 起步，而 MAE 是 CC-BY-NC），以及它**训练所用的数据集**
（Cityscapes 白纸黑字地禁止将在其上训练的模型用于商业用途）。凡是上游对权重保持沉默的地方，
本表就如实写明，而不是从代码许可证推断出授权 —— 沉默不等于许可。每一行都对照过上游的
LICENSE 文件和权重自己的模型卡；标为*未定论*的行，是没有任何一手来源能回答的，而不是没人
查过。

| 上游 | 许可证 |
|---|---|
| PP-OCRv6 det/rec, PP-OCRv5 det/rec, PP-LCNet cls (PaddleOCR) | Apache-2.0，代码**和**权重都是（每个模型卡都有声明） |
| YOLO26 / YOLOv8 / YOLOE (Ultralytics) | **AGPL-3.0** —— 著佐权（copyleft）；商用需要其 Enterprise 许可证。室内 YOLOE build 还带有用 Ultralytics 分发的 MobileCLIP2 文本编码器计算出的类别 embedding；该编码器自身的条款**未定论** —— 未审阅 |
| Places365 CNNs (CSAIL) | **代码** MIT；上游 README 以 **CC BY**（署名）发布预训练的 Places365-CNN 权重。训练图像的版权仍归其所有者 |
| Depth-Anything-V2 | Apache-2.0（Small）/ CC-BY-NC-4.0（Base 及以上） —— 请核对所用变体 |
| DeepLabV3+, SigLIP | 经由 D-Robotics `rdk_model_zoo` —— 请核对其条款 |
| SCRFD, ArcFace (insightface) | **代码** MIT，但预训练**权重仅限学术 / 非商用**。商用 build 请换成 YuNet（OpenCV Zoo —— 普通 CNN，int8 安全），并另选一个识别骨干 |
| Lite Any Stereo V2 | MIT，代码和权重都是。有一点要注意：发布的权重是从 FoundationStereo 蒸馏来的，而后者的 NVIDIA 许可证仅限研究使用。这属于上游自己的合规问题，而不是他们传递下来的条款，但在做商用 build 之前值得看一眼 |
| LingBot-Depth (MDM) | Apache-2.0，代码和权重都是；其 DINOv2 骨干也是 Apache-2.0 |
| EdgeSAM | **未定论** —— 未审阅上游条款 |
| OSNet (torchreid) | **代码** MIT。对 ReID 有用的 checkpoint 是在 MSMT17 / Market-1501 上训练的，这两个都是仅限学术使用的数据集，而没有任何上游来源说明这一限制是否延及权重 —— **未定论**。干净的路线是用 ImageNet 预训练的骨干，加上你自己的、许可宽松的 ReID 数据 |
| ViTPose（easy_ViTPose 导出） | **代码** Apache-2.0，但全身 checkpoint 是从 **MAE（CC-BY-NC-4.0）**微调而来，并在 COCO-WholeBody（研究 / 非商用）上训练。请把权重视为**非商用** |
| XFeat | Apache-2.0 —— 权重就提交在同一个仓库里，所以也在其覆盖范围内 |
| PointPillars（官方发布的各段） | 经由 D-Robotics `rdk_model_zoo` —— 请核对其条款。网络是 OpenPCDet（Apache-2.0），权重在 KITTI 上训练；KITTI 是 CC BY-NC-SA，所以请把权重视为**非商用** |
| KITTI（放置的校验帧） | **CC BY-NC-SA 3.0** —— 非商用、相同方式共享、署名。这里不再分发其中任何内容：`fetch_models.sh` 从*你的* KITTI 副本中拷出四个文件到 `models/kitti_frame/`，该目录已被 gitignore。渲染出的校验图已提交；其背后的数据没有 |
| GLASS | MIT，代码**和**发布的权重都是。这些权重是在 **MVTec AD** 上训练的，而它是 CC BY-NC-SA 4.0 —— 所以随附的示例模型是**非商用**的。这一约束不会传递到部署上：这种方法要在客户自己的正常零件上重新训练，而本库发布的是运行时，不是权重 |
| NeuFlow v2 | Apache-2.0，checkpoint 就提交在仓库里，所以许可证覆盖它们。训练数据是常见的光流数据组合（FlyingThings / Sintel / KITTI），它们自身的条款偏向研究用途 —— 每个光流模型都带着这条同样未定论的数据链 |
| SPAN | 项目整体是 Apache-2.0。checkpoint 从另一个文件托管处发布，没有附带任何条款，所以授权依据的是项目声明，而不是附在产物本身上 |
| Real-ESRGAN (realesr-general-x4v3) | BSD-3-Clause —— 权重是同一仓库的 release 资产，由版权持有人发布 |
| YOLOP | **代码** MIT，而且少见地把权重提交在仓库里，所以 MIT 覆盖它们。但它是在 **BDD100K** 上训练的，其协议把商业使用限制在成员组织内 —— 非商用是清楚的，商用则需要仔细查看 |
| DiffusionDrive | **代码** MIT（navsim 本身是 Apache-2.0），但编译后的模型和轨迹 anchor 都源自 **NAVSIM / OpenScene**，它们建立在 nuPlan 之上 —— 非商用条款。请把权重和 anchor 文件视为**非商用** |
| Paraformer-large contextual (FunASR) | 模型卡声明为 **Apache-2.0**；FunASR 的模型许可证（v1.1）补充要求再分发时保留署名和模型名称。两者都允许再分发。训练数据是未公开的内部音频。编译后的各段来自 D-Robotics `rdk_model_zoo` —— 也请核对其条款 |
| sherpa-onnx KWS zipformer zh-en 3M (2025-12-20) | 由 k2-fsa 项目（sherpa-onnx 是 Apache-2.0）以 release 压缩包形式发布，没有针对该模型的许可证，release 页面也没有说明 zh-en 模型用什么数据训练；它的纯中文兄弟模型是在 WenetSpeech 上训练的。再分发权重前请核对数据条款 |
| Silero VAD v6.2.3 | 代码和发布的权重均为 **MIT**（本 build 即转换自其 ONNX release）。训练数据未公开 |
| Matcha-TTS zh-en（ModelScope `dengcunqin/matcha_tts_zh_en_20251010`） | 模型卡声明为 **Apache-2.0**；Matcha-TTS 代码是 MIT。训练数据未公开：模型卡写的是 "mixed Chinese-English data"（中英混合数据），是从同一作者更早的 zh-en 模型微调而来，而那个模型是用 icefall 的 `baker_zh` 配方训练的 —— 该配方的 Baker 语料（BZNSYP，**非商用**）是否在训练数据中，并未说明。做商用 build 前请核对。词典和 token 表来自同一模型的 sherpa-onnx 打包 |
| Vocos 16 kHz（`vocos-16khz-univ`，随 Matcha-TTS zh-en 发布） | 模型卡声明为 **Apache-2.0**；Vocos 架构和代码是 MIT。训练数据未公开。与它搭配的声学模型（Matcha-TTS zh-en，同一作者，模型卡为 Apache-2.0）同样没有公开其数据 —— 它是从一个在中英混合语音上训练的模型微调而来 |
| PIDNet | **代码** MIT，在 **Cityscapes** 上训练，而 Cityscapes 明确允许分发在其上训练的模型，*同时*明确禁止对任何衍生作品做商业使用。所以：可再分发，**非商用** |

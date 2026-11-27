# 更新日志

[English](CHANGELOG.md) | **简体中文**

BCDL 的所有重要变更都记录在这里。格式遵循
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，版本号遵循
[语义化版本](https://semver.org/spec/v2.0.0.html)。

<!-- 本文件随发布包原样发布：不另做公开版本，写的时候就要达到可公开发布的标准。
     不得出现本地文件系统路径、机器名、客户信息，也不得链接不随它一起发布的文档。
     达不到这个标准的细节，放进路线图/设计笔记，不要写在这里。
     中文版与英文版（CHANGELOG.md）逐版本对应：英文版改了，就同步改这里。 -->

## [Unreleased]

### 新增
- **API 参考之外的中文文档。** `CHANGELOG.zh.md`、`docs/MODELS.zh.md`、
  `docs/GDC.zh.md` 和 `CONTRIBUTING.zh.md` 完整翻译了对应的英文文档，中文 README
  也改为链接它们。中文更新日志与英文版逐版本对应；任何一边缺了某个版本或某一条，
  `tests/test_changelog_zh.py` 都会失败。
- **API 参考覆盖了库导出的每一类任务。** 有六类任务随库发布了，却在
  `docs/API.md` 和 `docs/CPP_API.md` 里一字未提：人脸检测与对齐（SCRFD、
  `align_face`）、图像嵌入（`ImageEmbedder`、`EmbeddingBank`）、单目 3D 检测
  （SMOKE）、可提示分割（EdgeSAM）、开放词汇检测（YOLOE 的 `LabelMap`）以及全景
  驾驶感知（YOLOP 的 `AnchorDetector`）。现在它们在两种语言、两份参考里都各有
  一节；`TilePlacement`、`jpeg_save`、`letterbox_chw_float`、输出读取与输入写入、
  GDC 硬件几何变换，以及 `VideoEncoder` 的流式接口（`feed`、`receive`、`flush`）
  也都补上了。

### 修复
- **`VideoDecoder.feed()` 每次只接收一个 access unit。** Python 参考原先说可以
  喂任意切分的字节；而解码器跑在按帧喂入模式，每次调用必须恰好是一帧画面的
  NAL。能接收任意切分的 Annex-B 字节的是 `AsyncVideoDetectionPipeline.submit()`。
- C++ 参考里的 `GdcLetterbox` 示例还在传 0.3.0 就已删除的 `binPath` 参数。
- 本文件末尾的版本链接原先指向一个占位域名；现在改为对比真实的 tag。

## [0.9.0] — 2026-11

### 新增
- **带实例掩膜的室内物品：`yoloe26m_indoor99_seg`。** YOLOE-26m，检测头里折进了
  一套固定的 99 类室内词表 —— 先是十个导航地标（门、开着的门、玻璃门、电梯、
  电梯门、自动扶梯、楼梯、扶手、出口标志、小心地滑标志），之后是家具、家电、
  电子产品、餐具、卫浴设施、灯具和装饰品。它走现有的 `InstanceSegmenter` 解码成
  框和掩膜；类名随模型一起发布，即 `yoloe26m_indoor99_labels.txt`。在 1952 张
  留出的 Open Images 室内图片上，板端 int8 build 在框数不少于 20 的 25 个类上
  mAP50 为 0.484（浮点：0.478）。它是零样本模型，`docs/MODELS.md` 列出了这一点
  会在哪里暴露：显示器是盲区，楼梯不如有监督的检测器，导航类需要比 0.25 更低的
  分数阈值。
- **场景分类：`places365_resnet50`。** CSAIL 的 Places365 ResNet50，365 类场景，
  附带室内/室外标记，走现有的 `Classifier`。在板上用 Places365 验证集中未参与
  校准的图片测得 top-1 54.5% / top-5 84.6%（浮点：54.6% / 85.0%）。输入要做缩放
  加中心裁剪，不要用 letterbox —— 对 4:3 画面做 letterbox 会让 top-1 掉 3.6 个点。
- `scripts/fetch_models.sh` 会下载这两个模型，并从上游获取 Places365 的类名和
  室内标记；板端测试和 `board_bench.py` 都覆盖了这两个模型。转换配方：
  bcdl-model-zoo 中的 `yoloe26_indoor` 与 `places365`。
- **`runtime_info()` / `bcdl::runtimeInfo()`** —— 报告进程实际加载的是哪一份 hobot
  运行时：DNN 与 UCP 版本、SoC 名称，以及 libdnn 是从哪个文件解析出来的。正常
  情况下该文件在 `/usr/hobot/lib` 下；在别处就说明有另一份 SDK 挡住了板子自带的
  那份，而这一点光看结果是看不出来的。

### 变更
- **conda 包不再捆绑 hobot SDK。** `libbcdl` 以前依赖 `hobot-dnn` /
  `hobot-media`，它们会把一份为 S100 构建的运行时（`libdnn`、`libhbucp`、
  `libhbrt4`、`libhbmem`、`libmultimedia`……）装进环境，排在板子自带的
  `/usr/hobot/lib` 前面。这份运行时必须与板子固件匹配，所以在 S600 上它跑在一个
  并非为它构建的平台上，`Engine.infer()` 返回的输出会错位或是 NaN。现在
  `libbcdl` 链接板子自带的 SDK —— 头文件在整个 S 系列上完全一致，所以一个包同时
  适用于 S100、S100P 和 S600，系统升级后也照样能用。升级已有环境时会从中移除
  `hobot-dnn` 和 `hobot-media`。在 S100P 上，去掉这两个包之后，已发布的 0.8.1
  二进制不做任何改动即可通过完整的板端测试套件。

### 修复
- **输入行宽不是 64 字节整数倍的模型在 S600 上跑不了。** `Engine` 自行确定模型的
  动态输入 stride，此前按 32 字节对齐。S100 和 S100P 接受这种对齐；S600 运行时
  要求 64，会在推理时拒绝该 stride（"stride should be align to 64, but get
  224"），所以 224 宽的分类器、672 宽的检测器在那上面根本跑不起来。640 宽的模型
  不受影响，所以一直没被发现。现在 stride 按 64 对齐，所有 S 系列板子都接受。在
  S100/S100P 上，这只改变这类行的布局（224 宽的 NV12 平面现在每行 256 字节），
  输出与 32 字节布局逐位一致。不经 `setInput()`、直接写输入缓冲的代码，必须从
  `inputStride()` / `input_stride()` 取行 stride，不能自己假定一个。

## [0.8.1] — 2026-11

### 修复
- **在板子的"balanced" ION 布局下，`JpegDecoder` 每次解码都失败。** JPU 映射输入
  时，不管缓冲多大，总是从输入缓冲起始处映射一个固定 10 MiB 的窗口；而解码器给
  每个 JPEG 准备的暂存缓冲只是刚好够装下它 —— 小图只有 64 KiB，即一页。balanced
  布局下 carveout 堆只有 128 MiB，再有另一个 BPU 程序常驻时几乎是满的，于是这块
  缓冲落在距堆尾 320 KiB 处，它的窗口伸进了下一个 ION 区域，JPU 驱动拒绝映射
  （`not in ion range`），解码超时。现在输入缓冲至少 10 MiB，整个窗口都落在它
  自己内部；每个解码器在生命周期内占用 10 MiB ION 内存。构造一个解码器仍需约
  5 ms。在这种布局下，整套测试现在能在一个进程里全部通过（512 passed，
  3 skipped）。

## [0.8.0] — 2026-11

### 新增
- **`examples/voice_assistant.py` 与 `voice_assistant.cc`**：端到端的语音控制
  机器人 —— 唤醒词、VAD、识别、命令匹配和语音回复，像麦克风那样按 100 ms 的块
  输入。没有录音时它会先合成一段演示会话，所以没有麦克风的板子也能跑。API 参考
  新增了语音流水线概览：音频格式、如何制作唤醒词、调参表和常见的坑。
- **语音识别：`ParaformerAsr`**（中文，非自回归）。输入一段 16 kHz 单声道语音，
  输出文本：C++ 实现的 Kaldi 兼容前端（`Fbank`、`lfr_stack`、`Cmvn` —— 与
  `torchaudio.compliance.kaldi.fbank` 相差约 2e-4，即 float32 的精度下限，与
  FunASR 的 LFR 逐位一致）、BPU 上官方的三段式 int16 Paraformer-large build，
  以及在 CPU 上、位于预测器与解码器之间的连续积分发放（`cif_integrate`）。在
  S100P 上每段语音约 36 ms，含前端。`scripts/fetch_models.sh` 会下载该 build；
  `data/audio/` 中的四段 AISHELL-1 音频用于驱动板端测试。
- 这里的 CIF 把预测器的尾部权重放在**真正的语音结束处**。定长 build 把它加在
  24 s 补零的末尾，而对补零部分做掩膜时会把它一并清掉，所以凡是最后一个字需要
  这份尾部权重的语音，都会丢掉这个字：在 AISHELL-1 测试集最短的 300 条语音上，
  这样的字错率是 3.88%，恢复尾部权重后为 2.72%（浮点模型：1.76%）。
- **唤醒词：`KeywordSpotter`**。sherpa-onnx 的开放词表 zipformer2 KWS 模型
  （zh-en，3M）：编码器在 BPU 上，每次调用处理一个 160 ms 的块；解码器、joiner
  和关键词搜索在 CPU 上，逐步与 sherpa-onnx 的 KeywordSpotter 保持一致（在浮点下，
  同一套搜索在核对过的全部 87 段音频上给出了与 sherpa-onnx 相同的检测结果）。
  唤醒词是一行 token（`n ǐ h ǎo x iǎo ch ē @你好小车`），不是一个模型。召回率
  97.5% / 89.1%（干净 / 带噪），浮点为 96.5% / 87.1%；在 4.4 h 的语音与噪声中
  零误唤醒；易混短语有 5.7% / 4.0% 会触发（浮点 5.4% / 4.6%）；每 160 ms 占
  2.0 ms BPU，全链路实时率 0.023。发布的编码器要经过四处改写才能编译（int64 掩膜
  输入、softplus、相对位置 gather、补边过多的 depthwise 卷积）；见 MODELS.md。
  `FbankConfig.window`（`"povey"`）是新增的，供 icefall 模型使用。
- **语音活动检测：`SileroVad`** 与 `VadSegmenter`。Silero VAD v6.2.3 跑在 BPU
  上，每 32 ms 一块调用一次，配一个流式分段器，把完整的语句交给识别器。在 100 段
  带噪 AISHELL-1 会话（SNR 10 / 0 dB）上，帧级 F1 为 0.924 / 0.863，浮点模型为
  0.929 / 0.870；说话人停下后，判定语音结束的中位延迟为 0.36 s；24 分钟纯噪声
  不产生任何语音段；每块 0.39 ms。模型的 STFT 放在 CPU 上：整图编译时，平方后的
  频谱在 BPU 上的跨度约为 5000，int16 丢掉了安静的频点，概率在句子中段往下掉。
  `Engine.input_name` / `output_name`（C++ `inputName` / `outputName`）是新增的。
- **语音命令：`CommandMatcher`** 与 `to_pinyin`。在无声调拼音上把识别文本与一份
  固定的命令列表做匹配，因此同音字（打开灯 → 大开灯）和句尾语气词（前进 →
  前进了）仍能映射到正确的命令，非命令则会被拒绝。在 40 条命令 × 9 种合成音色
  上，它把 Paraformer 从 84.7% / 82.8% 完全一致（干净 / 5 dB SNR）提高到
  96.9% / 95.6% 匹配正确，错配命令 0.6% / 1.1%，在 600 句新闻上零误接受；每次
  调用几 µs。拼音表编译进了库（`scripts/gen_pinyin_table.py`，数据来自
  pypinyin，MIT）。
- **语音合成：`TextToSpeech`** —— 输入中文文本，输出 16 kHz 语音，所有网络都在
  BPU 上：`MatchaFrontend`（sherpa-onnx 为 Matcha-TTS zh-en 写的词典前端，在
  100 句上逐 id 复现，另加数字朗读）、`MatchaAcoustic`（两个 int16 图，长度调节
  在 CPU 上）以及 `VocosVocoder`。在 60 句上回环字错率 1.73%，CPU 上的浮点模型为
  1.61%；实时率 0.009（每句 26 ms）。原版 Matcha 的定长导出并不等价于原模型：
  它的 GroupNorm 和加性 0/1 注意力掩膜都取决于补零有多少，所以解码器中的这两处
  被改写为只看未补零的帧（与浮点相差 9e-5）。token id 在 CPU 上查表，因为编译器
  把浮点的 id 输入量化到了只有 127 的范围。0.2 s 及以上的停顿按
  `TextToSpeechConfig.silence_scale` 缩短（0.2，与 sherpa-onnx 相同；也可单独
  调用 `scale_silence`）：60 句的音频从 225 s 缩到 172 s（sherpa-onnx：176 s），
  回环字错率不变。
- **语音合成后端：`VocosVocoder`** 与 `Istft`。输入梅尔频谱，输出 16 kHz 波形：
  Vocos（Matcha-TTS zh-en 的声码器）转成固定 256 帧窗口的 int16 BPU build，逆
  STFT 在 CPU 上用 C++ 实现。更长的梅尔频谱分块处理，接缝处精确无误差。每 3 s
  音频 7.9 ms（实时率 0.0026）；在可听的时频单元上与浮点的对数谱距离为
  0.54 dB，Paraformer 对输出的转写完全正确。图在 `exp` 和 `cos`/`sin` 之前切开：
  按发布原样编译时，量化后的 `cos`/`sin` 让相位的余弦在 int8 下只有 0.58、在
  int16 下为 0.97–0.99。int8 同样被否决 —— 仍听得懂，但与浮点差 5.4 dB，等于一层
  底噪。
- **LAS2-L 双目 build**（`las2_l_crop2_nashm.hbm`、
  `las2_l_resize2_nashm.hbm`）：聚合深度是 M 的两倍，在 crop 集上 bad-1px 从
  14.2% 降到 11.0%，44 FPS，M 为 72 FPS。收益集中在近距离；见
  `docs/MODELS.md` 的"LAS2: M or L"一节。

### 变更
- **BCDL 现在是 BPU Common Deep Learning 的缩写**，README 把它介绍为一个同时面向
  视觉*和*语音的框架；能力清单里的语音部分单独归为一组。

### 修复
- **随库发布的 LAS2 双目 build 把视差截断在 131 px 和 47.7 px。** int16 输出阈值
  就是校准集产生过的最大值，而 crop 和 resize 两个 build 的校准场景里都没有近景
  —— 所以模型本该覆盖 0–192 px，比截断值更近的东西却都被读成更远，而且不报任何
  错。仓库自带的双目图对视差范围是 44–129 px，正好在 crop 的上限之下，所以没有
  测试发现。替换版 `las2_m_crop2_nashm.hbm` 和 `las2_m_resize2_nashm.hbm` 在完整
  范围上校准（输出阈值 180–187 px）。在带真值的留出 ETH3D 与 Middlebury 2021
  场景上，与浮点模型相比，crop 最差的场景从 14.7 px 降到 0.30–0.43 px，resize
  的 EPE 从 2.17 px 降到 0.60 px。新增的测试把仓库图对平移到约 175 px，旧 build
  在它上面会失败。

## [0.7.5] — 2026-08

### 修复
- **高度为零的检测框让跟踪器的分配求解器越界访问。** 完全位于画面上方的框会被
  钳制成 `y1 == y2 == 0`，而卡尔曼状态是 `[cx, cy, w/h, h]` —— 于是 `h == 0` 让
  宽高比一项变成无穷大，框再出来时就成了 NaN。形状相似度一项没有 NaN 保护，于是
  分配代价矩阵的一整行都成了 NaN；在匈牙利求解器里，任何与 NaN 的比较都为假，
  目标列下标停在初始值 `-1`，代码就读了 `p[-1]`、写了 `used[-1]`。
  AddressSanitizer 把它报告为一次普通 `update()` 调用引发的
  heap-buffer-overflow。现在在构建状态的地方把框的边长钳制到一个亚像素下限，
  求解器无论收到什么输入都不会越界。除了修掉越界，这也恢复了关联：同样的输入下，
  跟踪器恢复出四条轨迹，而此前 NaN 代价行把它们塌缩成了一条。

- **`Engine` 有十个访问器索引张量 vector 时不做范围检查**，而旁边另外五个是做的。
  这十个全都重新导出到了 Python，在那里 `engine.output_bytes(99)` 会基于一个垃圾
  指针构造 bytes 对象，让解释器段错误。还有三个任务头在校验之前就把调用方传入的
  下标传了下去。

- **SCRFD 人脸解码拿一个张量的形状去套三个张量。** 每个尺度的预测数取自分数头，
  两个回归头随后按每个预测 4 个和 10 个 float 去索引，却从未核对过它们的实际
  大小。如果模型按尺度而不是按类型分组，或者关键点少于五个，就会在不报错的
  情况下越界读堆，最多读到十倍。单目 3D 检测器也是同样的毛病：它取了回归张量的
  形状却从未使用，而是用*分类*张量来确定八个通道平面的大小，返回看起来像模像样
  的垃圾框。

- **张量元素读取器有三份拷贝漏掉了一次越界修复。** 深度、细化深度和分割各自带着
  一份复制粘贴来的 `readElement`，早于共享读取器后来加上的钳制，所以逐通道量化的
  输出会读过 SDK 的 scale 数组末尾。现在三者都调用共享的 `outputAsFloat()`，遇到
  紧密排列的 F32 输出头时它还会走零拷贝路径。

- **编解码器构造失败会泄漏一个 VPU 实例。** 被拒绝的配置抛异常时，
  `hb_mm_mc_initialize` 已经占了一个实例，而抛异常的构造函数不会执行析构 —— 一个
  重试循环就能把全部 32 个耗尽，而且重启之前都回不来。GDC 路径在拷出抛异常时也
  以同样方式泄漏一个输出帧：泄漏三个之后，此后每次帧请求都会一直阻塞到超时。

- **JPU 的输出拷出时没有做缓存失效**，违反了本项目在 `core/sys_mem.h` 中写明、
  在别处都遵守的规则。编解码器自有的缓冲每次调用都重新分配，所以同样的地址回来时，
  上一帧的缓存行仍然有效 —— 偏偏就在复用解码器这条路径上，会偶发读到陈旧像素，而
  这个类存在的意义正是让这条路径变快。

- **总头文件只重新导出了 46 个公开头文件中的 27 个。** C++ API 参考说包含
  `bcdl/bcdl.h` 就够了；对 `TrackingPipeline`、`outputAsFloat`、
  `AnomalyDetector` 以及另外十六个来说并非如此，示例代码则悄悄绕过了这个问题，
  直接包含子头文件。现在有测试从两个方向比对总头文件与安装树。

- **截断的视频块会多读缓冲末尾之后的一个字节。** Annex-B 起始码扫描会接受一个
  其后没有 NAL 头字节的偏移，`finish()` 随后对它解引用。异步流水线文档里的三步
  槽位协议，还演示了把 `acquireSlot()` 的结果直接喂给 `letterboxIntoSlot()`、
  中间不做任何检查 —— 就在说明 `acquireSlot` 结束后返回 `-1` 的那句话下面一行；
  现在这些调用都会做校验。

### 变更
- **Python 在每个阻塞调用期间释放 GIL。** `Engine.infer` 以前在整个推理期间都
  持有 GIL —— 深度模型 116 ms，单目 3D 模型 271 ms —— JPU 与 VPU 编解码器、GDC
  letterbox、深度细化器以及双目和跟踪流水线也是如此。摄像头读取线程或 HTTP 线程
  在每一帧期间都完全停住。在板上用一个每 0.5 ms 计一次数的辅助线程、跨十次深度
  推理（1046 ms）测量：之前 10 次，之后 1851 次，而解释器完全空闲时为 1871 次。

- **九个 camelCase 解码器别名已废弃。** `decodeDepth`、`decodeSeg`、
  `decodeDet3d`、`decodeAnomaly`、`decodeFlow`、`decodeDrive`、
  `decodeRefinedDepth`、`depthToPointCloud` 和 `scaleIntrinsics` 只覆盖了大约
  四十个同类 `decode_*` 函数中的九个，而整个包其余部分严格使用 snake_case，
  所以它们读起来像是在刻意区分什么，而不是一种约定。它们仍然可以调用，现在会
  发出警告并给出应改用的名字；去掉的是这层重复的接口：它们移出了 `__all__`，
  也移出了 API 参考。

- **hbVP 预处理路径标注为未经验证。** `letterbox()`、`cvtColor()` 和
  `resizeExact()` 运行在 vDSP 上，而在当前软件版本上 vDSP 不在线：设备节点存在，
  但即使以 root 身份，核心也报告 OFFLINE，也没有安装 DSP 固件，所以这些调用会
  抛异常而不会执行。因此这套仿射约定从未被实际观察到产出过一帧。头文件和两份
  API 参考现在都写明了这一点，并给出两个实测过的替代方案 —— CPU letterbox 和 GDC
  引擎。

## [0.7.4] — 2026-08

### 变更
- **conda 包声明了它一直在用的 OpenCV Python 绑定。** 七个有文档的函数 ——
  `letterbox_numpy`、`letterbox_chw_float`、`bgr_to_nv12`、`jpeg_encode`、
  `face_letterbox`、`embed_preprocess` 和 `SamSession` —— 会延迟导入 `cv2`，而包
  从未声明这一依赖。之所以没被发现，是因为频道里原先那份 OpenCV build 恰好把
  Python 绑定打进了 C++ 包，于是 `cv2` 随着每次安装都要装的那个库意外地一并到位。
  现在频道改用上游 OpenCV，它像 conda-forge 一样把两者拆开，`bcdl` 也显式依赖
  `py-opencv`。额外下载的只有绑定本身：C++ 那部分本来就因 `libbcdl` 而装上了。

  调用方如果把自己在别处准备好的张量喂给 engine，就永远碰不到这七个函数；但
  letterbox 辅助函数几乎是这里每条流水线的入口，一个有文档的函数在默认安装下抛
  `ImportError`，在用户能理解的任何意义上都谈不上"可选"。

### 修复
- **0.7.0 新增的所有内容都不在 `bcdl.__all__` 里** —— 四个任务头（激光雷达 3D、
  异常检测、光流、端到端驾驶）共 45 个名字，包括四个包装类。这些类是普通的模块
  属性，所以 `bcdl.AnomalyDetector` 能用，每个示例和板端测试都照样通过，
  `docs/API.md` 中的文档示例也完全照写就能跑。与参考文档不一致的是
  `from bcdl import *`，以及任何把 `__all__` 当作 `py.typed` 包公开接口来读的
  类型检查器。这是把已发布的 conda 包装进一个空环境、问包导出了什么才发现的，
  而不是靠调用它。

  `tests/test_python_exports.py` 从两个方向堵上这个洞：API 参考提到的每个
  `bcdl.Something` 都必须在 `__all__` 里（需要包可导入 —— 这正是用户会碰到的
  情形）；包装层中定义的每个公开类都必须在 `__all__` 里（静态检查，这样下一个
  漏掉的会在写它的那台机器上就被抓到，而不是等到上了板子）。这与 0.7.2 的修复和
  0.7.3 的 C++ 检查属于同一类问题：文档提到了某个符号，却没有任何东西断言包与之
  一致。

## [0.7.3] — 2026-08

### 变更
- **社区交流一节上移，放到快速上手之后。** 它原来在测试表格下面，那不是它该在的
  位置：一个人最想找地方提问的时刻，恰恰是刚跑完那三条安装命令的时候 —— 要么成功
  了想了解更多，要么失败了卡住了。现在排在它后面的那些章节，都是可以回头再翻的
  浏览材料。

### 新增
- **C++ API 参考现在会与头文件比对检查。** `docs/CPP_API.md` 及其中文版承诺了
  一组成员和 `bcdl::` 类型；有个测试读取成员表和 `// pure:` 行，断言每一项都在
  某个公开头文件中有声明。它还断言两种语言承诺的是*同一套* API —— 译文一旦走样，
  就会告诉一类读者另一类读者从未听说过的东西。

  这是 0.7.2 在 Python 一侧补上的那个漏洞的 C++ 另一半，而且是更隐蔽的那一半：
  文档里有、实际没有的 Python 属性会在调用时报错，而文档里有、实际没有的 C++
  成员，只有在有人试图针对它编译时才会被发现。这项检查在任何机器上都能跑 —— 它只
  读文件，不需要板子也不需要模型 —— 并且会自我验证：故意改坏一份文档副本，确认
  能被抓到，因为一个不可能失败的成员存在性检查什么也证明不了。

## [0.7.2] — 2026-08

### 修复
- **`Engine.output_packed_bytes()` 和 `Engine.output_packed()` 有文档却不存在。**
  0.7.1 把两者都加进了 nanobind 绑定，并在 `docs/API.md` 的 `Engine` 成员表里
  列出了 `output_packed_bytes(i)`，就在*确实*转发了的 `input_packed_bytes(i)`
  旁边 —— 但 Python 包装类从未转发它们，所以照着文档用会得到 `AttributeError`。
  现在两者都在包装类上了。

  它之所以一直没被发现，是因为每个测试都通过私有的 `engine._e` 句柄而不是公开类
  来访问，所以绑定被覆盖到了，用户实际调用的接口却没有。新测试断言 API 参考承诺
  的每个成员都存在于包装类上。

## [0.7.1] — 2026-08

编译好的模型已经发布，所以现在你唯一需要的硬件就是一块板子。

### 新增
- **Hugging Face 上的预编译模型 —— 不需要工具链。**
  [`ruisv/bcdl-*`](https://huggingface.co/ruisv) 下的九个包提供可直接加载的
  `.hbm`，覆盖 OCR（v6 和 v5，**附带匹配的字典**）、零样本双目、RGB-D 深度细化、
  ×4 超分辨率、稀疏特征、全身姿态、实时分割和全景驾驶感知。
  `scripts/fetch_models.sh` 会自动拉取它们，并可回退到本地暂存的 build，所以
  重新构建模型的维护者仍能拿到自己的那一份。

  只发布**权重**可以再分发的模型 —— 这与上游仓库的许可证是两回事，在转换配方里
  逐个模型确定。Ultralytics 派生的 build（AGPL）、人脸模型和 PointPillars（权重
  限非商用），以及 EdgeSAM 和 OSNet（尚未厘清）都只提供配方。十个当中有四个可以
  再分发但**限非商用**，每一个的模型卡都在顶部用横幅写明，而不是放在脚注里。
- **异常检测器也发布了**
  （[`ruisv/bcdl-glass-bottle`](https://huggingface.co/ruisv/bcdl-glass-bottle)），
  许可证为 CC BY-NC-SA 4.0 —— 代码和权重是 MIT，但它在 MVTec AD 上训练，而
  MVTec AD 是 CC BY-NC-SA，这一限制会传到训练出的模型上，相同方式共享也一并
  传过来。这一限制不会波及真实部署：实际部署的 GLASS 要在你自己的无缺陷零件上
  重新训练，所以受限的只是这个示例 checkpoint。

  **光流模型没有发布，原因值得一读**，别以为一个宽松许可证的徽章就能说明一切。
  NeuFlow v2 是 Apache-2.0，*权重就提交在同一个仓库里* —— 通常问题到这里就结束了
  —— 但它是在 FlyingThings3D 上训练的，而后者的条款仅限研究使用，对用这些数据
  训练出的模型只字未提：既没有把限制延伸过去，也没有给予许可。沉默不等于许可，
  所以它只提供配方。两份配方现在都已收进配套的 model-zoo。

## [0.7.0] — 2026-08

四个新任务头 —— 光流、工业异常检测、端到端驾驶和激光雷达 3D 检测；一条在 BPU 上
快 46 倍的激光雷达链路；一个输出侧 stride 修复，它此前一直悄悄让五分之一的模型
发生错切；以及围绕真正能判定检测好坏的东西重做的校验图。

### 变更
- **OCR 解码快了 2.2 倍 —— 从 205 ms 到 93 ms** —— 而 OCR 本身什么都没改。
  `outputAsFloat()` 的慢路径逐元素分解下标，而 PP-OCRv6 识别器的 `[1,40,18710]`
  输出按行补齐（stride 74880 字节，有效 74840），所以走了这条路径。改为整行
  拷贝后只要 4 ms，原来要 56 ms。所有带补齐的输出都受益：全身姿态热图、分割
  logits、XFeat 的三个头、YOLOP 的三个头。

### 修复
- **`Engine.infer()` 不再让带补齐的输出发生错切。** 输出按设备布局分配，而在本库
  所配套的模型中，有 8 个会在某个维度上补齐 —— 一个 `[1,133,64,48]` 的 ViTPose
  热图，行 stride 是 256 字节，有效字节只有 192。`infer()` 取的是这块缓冲中扁平的
  有效前缀，这会让第 `r` 行偏移 `r * pad`；结果看起来仍像数据，所以什么都没报错。
  这是 0.4.0 中输入侧 stride 修复的镜像，也是同一类 bug：一个重新编译的
  PointPillars 阶段与它自己的浮点参照相比余弦只有 0.70，被当成量化失败放弃了，
  而实际上产物的保真度是 0.9988，错的是读数的一方。高层任务类从未受影响 —— 它们
  通过原生的 `outputAsFloat()` 解码，而它一直遵守 stride。

### 新增
- **四个新任务头**，即上面摘要中提到的四个。
  - **`tasks/optical_flow.h`** —— 稠密光流。本库第一个接收*两张*图的任务，它回答
    的是关于这一对图的问题，而不是关于其中任一张：每个像素去了哪里。本库其余
    所有任务都把一帧归约成一段描述；这个任务保留帧的形状，并用运动把它填满。
  - **`tasks/anomaly.h`** —— 工业异常检测，单类。没有类别列表，也没有缺陷训练集：
    模型只看过某一种产品的*正常*样本，然后报告每个 patch 与它们有多不像。调用方
    必须遵守两个推论：模型是按产品的，所以随附的 `.hbm` 只是示例而不是通用
    检测器，换新零件就意味着重新训练；分数是相对该产品自身的正常波动校准的，
    所以阈值是每条产线自己的决定，而不是库的默认值。
  - **`tasks/e2e_drive.h`** —— 端到端驾驶，从摄像头加激光雷达到一条规划轨迹。与
    本库其他所有任务不同，这个模型不是报告它看到了什么、再把决策留给你；它直接
    输出要行驶的自车轨迹。对聚类轨迹锚点去噪的扩散解码器就在编译好的图里 —— 锚点
    进去，出来的是一条已经选好的轨迹，同时还有与规划器联合训练的那些感知头的
    输出。位姿是按固定时间间隔排列的未来时刻（原版模型：4 s 内 8 个位姿），而
    不是一条空间路径。
  - **`tasks/lidar_det3d.h`** —— 激光雷达 3D 检测，基于锚框，PointPillars 系列。
    本库第一个输出*空间*中而非图像中的框的任务：自车坐标系下的中心、尺寸和偏航
    角，单位是米。检测头预测的是相对一组固定锚框网格的残差，所以解码需要与模型
    训练时相同的锚框集合 —— 弄错了，框依然看起来合理、格式规整，只是位置不对。
    解码是特意放在 CPU 上的：发布的检测头用 ScatterND 把解码烘进了图里，它能一个
    错都不报地编译成一个输出永远不会被写入的模型。

  `docs/MODELS.md` 给出了每个任务实测的延迟和精度，以及其中哪些以可下载模型提供、
  哪些只有转换配方。
- **能用来评判模型、而不只是好看的校验图。** 两张鸟瞰图现在用 SVG 绘制，再用
  librsvg 栅格化（`tests/bev_svg.py`） —— 真正的排版、矢量和 CSS，只有确实是栅格
  的图层才以图片嵌入。更重要的是，激光雷达图展示的内容变了：暂存好一帧带标定的
  数据后，它会把模型自己的 3D 框投影回摄像头照片上，并按 KITTI 自己的 BEV IoU
  阈值与标注比对打分 —— 匹配、误检、漏检 —— 且限定在相机视锥内，因为标注只覆盖
  相机看到的部分。俯视的占地框看不出一个框低了半米或者方向反了；投影出来的
  立方体一眼就能同时看出这两点。不分发任何数据集：`scripts/fetch_models.sh`
  从你自己的 KITTI 副本中暂存四个文件，没有它们时，图会回退为只画鸟瞰。
  `docs/MODELS.md` 逐一讲解了每种颜色的含义，以及如何在你自己的传感器平台上重新
  生成这张图。
- **`tests/kitti_frame.py`** —— 一帧带标定数据的点云扫描、图像、标定和标注，投影
  方式和相机↔激光雷达的坐标约定都在依赖它们的那几行旁边写明，另附贪心 BEV-IoU
  匹配。
- **`pillarPointFeatures()`（`preproc/lidar_voxelize.h`，Python 中为
  `pillar_point_features`）**，并借此支持一条**在 BPU 上快 45.9 倍**的
  PointPillars 链路。接收三个原始 pillar 张量的 VFE 会在自己的图内把它们展开成
  逐点特征，在一个 95 MFLOP 的 MLP 上花掉 290 ms：`(pillars, points, 4)` 是一条
  巨大的 pillar 轴对上一条只有 4 宽的通道轴，这是能交给卷积引擎的最差形状。改为
  导出成 `(1, 10, P, 32)` 上的 1×1 卷积、把对点取 max 做成池化窗口后，同样的
  计算只要 1.62 ms，而本函数在 CPU 上做展开只要 1.13 ms。把 backbone 和检测头
  重新导出为一个 int8 图，又把 113.7 ms 降到 7.20 ms。在示例扫描上，整条链路从
  468.8 ms 降到 43.0 ms。

  量化以这条链路**自己的**浮点版为参照来评分，因为发布的链路是另一套权重，拿它
  比较会把 checkpoint 的变化混进 int8 的代价里：浮点找到 18 个框，这里也是 18
  个；匹配上的框中心相差不超过 0.07 m；分数低 0.035；浮点打分高于 0.3 的锚框有
  93% 保留了下来。要做到这一步，检测头需要用八帧真实扫描而不是一帧来校准 —— 单帧
  校准集丢了 18 个框中的 4 个，还出现两次 π 的航向翻转，而且*降低阈值也救不
  回来*，因为排序本身变差了。这些是单帧扫描上的保真度数字，不是 KITTI mAP。

  两条链路都支持，且都不靠文件名来选：只有一个输入的 VFE 被视为卷积形式，输入是
  scatter 画布的检测头被视为已融合 backbone。更快的这些 build **不分发** —— 它们
  来自一个在 KITTI 上训练的 checkpoint，所以随库提供的是运行时和转换配方；
  `docs/MODELS.md` 覆盖了这两条链路。

  剩下的瓶颈已经不是计算了：8.8 ms 的 BPU 工作现在排在约 17 ms 的主机到设备
  拷贝之后，其中大部分是 55 MB 的 BEV 画布。
- `Engine::outputPackedBytes()` / `Engine::copyOutput()`（C++）和
  `output_packed_bytes()` / `output_packed()`（Python）：把输出读成按其声明形状和
  dtype 连续排列的行主序数组，补齐部分被剔除，量化保持原样。与
  `inputPackedBytes()` / `setInput()` 对称。

## [0.6.0] — 2026-08

### 新增
- **RGB-D 深度精修（`tasks/depth_refine.h`）。** 新的任务头：输入一张你已有的深度图
  （双目或 ToF 测得的读数，噪声大、空洞多）和对齐的 RGB 帧，输出补好空洞的度量深度
  以及逐像素的可信掩膜，可直接反投影成相机坐标系下的点云。它做的是精修而不是估计，
  所以是单目、双目深度头的补充，不取代其中任何一个。`DepthRefiner` 自带预处理，因为图的
  两路输入（按面积缩放并归一化的 RGB，以及用 0 表示"无读数"的对数深度）必须与模型校准
  时完全一致；`preprocessRefineImage` / `preprocessRefineDepth` /
  `decodeRefinedDepth` / `depthToPointCloud` 以纯函数形式公开，供自己驱动引擎的调用方
  使用。新增 `examples/depth_refine_demo.cc`，输出输入与精修结果的对比图，并可选输出
  `.xyz` 点云。

  部署图有两个特性值得了解。上游会丢掉 patch 内没有任何有效读数的深度 token，这让
  transformer 的序列长度随数据变化，无法静态编译；部署图保留全部 token，在参考场景上
  实测平均绝对相对深度误差 0.06%、掩膜 IoU 0.9999 —— 但这些场景的有效率在 87–100%，
  所以非常稀疏的输入深度不在实测范围内。另外，RGB 缩小用的是面积平均而不是双线性：
  对照参考实现的抗锯齿 resize，面积平均的平均绝对相对误差为 6.1e-4，普通双线性为 9.1e-4。

  发布的 build 是全 int16，因为 int8 PTQ 扛不住 24 层的 ViT-L：编译顺利，返回的深度图
  动态范围却已塌缩（2.9–13.8 m，浮点模型是 0.97–45.8 m；平均绝对相对误差 233%，余弦
  0.53）。见 `docs/MODELS.md`。

  **这是模型目录里最慢的模型，慢了一个数量级：在 S100P 上每帧 1457 ms（0.69 FPS）**，
  相对浮点模型的平均绝对相对误差 2.9%、掩膜 IoU 0.959。把它当作对关键帧的按需精修，
  别放进视频循环。开销来自注意力分数矩阵：它随序列长度 1 + 2N 呈平方增长（图像 token
  与深度 token 拼接在一起），int16 下每帧搬运 23 GB DDR —— 注意力矩阵乘在 21.9 ms 的
  时间片里只有 4.3 ms 在计算。因此 token 预算就是调延迟的杠杆，而且精度代价不大：对照
  3600 token 的浮点参考，发布版的 1200 token 平均绝对相对误差 1.20%，768 token 为
  1.75%（注意力开销低 2.4×），432 token 为 2.56%（低 7.5×）。量化误差本身已经比这些
  都大。
- **YOLO26n-sem 语义分割（Cityscapes，19 类）。** 第二个实时分割模型，没有新增 C++ 代码：
  它和 DeepLabV3+、PIDNet-S 一样，走现有 `Segmenter` 的 argmax 路径解码。它带来的是
  原生 640×640 输出，所以不同于 1024×2048 的 `semseg_rt` build，调用方不必自己对标签图
  上采样。

  发布的 build 是 int16，**并且**用真实 Cityscapes 帧校准，两者缺一不可。全 int8 的
  build 编译顺利，逐张量余弦闸门全部通过（每层 >0.95，输出 ≥0.98），但像素级类别判定
  与浮点模型只有 61% 一致 —— 整类整类地互换。稠密的 19 路逐像素分类器对决策边界很敏感，
  而聚合的余弦看不到这种敏感；下次余弦闸门说某个分割模型没问题时，值得记起这一点。统一
  int16 能恢复到 85%；剩下的差距出在校准数据的域而不在精度，改用真实 Cityscapes 帧
  （而不是通用街景替代数据）重新校准后，一致率达到 96%。
- **中文 API 参考。** `docs/API.zh.md`（Python）与 `docs/CPP_API.zh.md`（C++）是英文两份
  文档的完整译本，不是摘要：每个类、配置字段、解码器签名、坑点和实测数据都译了过来。
  每份文档都链接到另一语言的对应版本，两个 README 也各自指向同语言的参考文档。

### 变更
- API 参考把 OCR 描述为默认 PP-OCRv6、以 v5 作为回退，与 0.5.0 的行为一致。

## [0.5.0] — 2026-07

### 变更
- **OCR 默认改为 PP-OCRv6。** demo、测试和拉取脚本中的检测与识别默认模型统一从
  PP-OCRv5 换成 PP-OCRv6 medium。改动刻意做得很小：识别器从模型输出读取 CTC 类别数，
  从文件加载字符集，所以从 18385 类换到 18710 类只需要一份新字典
  （`data/ppocr_keys_v6_18710.txt`）和新的模型路径 —— 识别代码一行没改，预处理本来就是
  PP-OCRv6 要求的保持宽高比的补边。PP-OCRv6 不带文本行方向分类器，所以方向分类这一段
  沿用 v5 的 PP-LCNet 模型；v5 的检测/识别 build 也继续保留，作为回退。
- **默认识别器是全 int16 build**，而不是编译器默认的混合精度 build。在板上，统一 int16
  的图既更准（字错率约减半）又更快（约 2 倍），因为层边界上没有 int8/int16 重量化。
  另有 960 宽的 int16 build 供长文本行使用，可通过 `BCDL_OCR_REC` 选择。见
  [`docs/MODELS.zh.md`](docs/MODELS.zh.md)；每个 build 的转换配方都在配套的 model-zoo
  仓库里。

## [0.4.0] — 2026-07

### 新增
- **利用外观特征的多目标跟踪。** `ByteTracker::update()` 新增一个重载，为每个检测接收
  一条 ReID 嵌入向量，第一轮关联的代价变为 `min(IoU distance, gated cosine distance)`。
  取**最小值**正是它安全的原因：外观只能降低代价，所以坏的嵌入向量永远拆不掉几何已经
  配上的匹配 —— 只能救回几何即将错过的匹配。三道闸门防止它凭空造出匹配：基于**原始**
  IoU 距离的邻近闸门（在分数融合之前计算，否则高置信度的检测会放宽自己的空间闸门）、
  余弦闸门，以及类别闸门（不同类别之间没有共享的外观空间）。低分检测的第二轮关联仍是
  纯 IoU，因为这些裁剪块正是被遮挡、模糊的那些，它们的嵌入向量价值最低。

  轨迹模板采用置信度自适应的 EMA：检测分数越低，平滑系数越趋近 1（即完全不更新），
  这样半遮挡的裁剪块就无法污染目标被记住的样子。

  **空的**嵌入向量表示"该检测没有外观特征"，这是对廉价裁剪块跳过 ReID 模型的受支持
  做法。单参数的 `update()` 保持不变，逐位一致。
- **行人 ReID。** 新增 `tracks/reid.h` / `src/tracks/reid.cc`：`reidPreprocess()` 把
  检测框裁成模型的 256x128 输入（拉伸 resize，**不是** letterbox —— 这些模型就是用
  拉伸过的裁剪块训练的）。读出复用嵌入任务。Python：`bcdl.ReIDExtractor`、
  `bcdl.reid_preprocess`、`bcdl.reid_crop_preprocess`。Python 的预处理绑定的就是 C++
  实现，两条路径不会逐渐偏离、给模型喂不同的像素。6.6 MB 的模型，**每个裁剪块
  0.84 ms**。
- **`TrackingPipeline` 可选传入 ReID Engine**，原生 C++ 路径也能用上外观特征。裁剪
  尺寸从模型读取。`lastEmbedCount()` 暴露了让帧耗时随人群规模增长的那一项 —— ReID 模型
  每个裁剪块跑一次，所以在拥挤的帧上决定节奏的是它而不是检测器；`TrackingReidConfig`
  给它设上限。
- **BoostTrack++ 新增项**，各有独立开关，默认全部关闭（`BoostConfig`）：软缓冲 IoU
  （soft buffered IoU），按 `1 - tracklet confidence` 的比例同时放大两个框，让一直在
  惯性滑行（coasting）的轨迹搜得更宽；关联代价中的马氏距离项与形状一致性项；以及
  DLO/DUO 检测置信度提升，在高/低分拆分**之前**提高被轨迹"担保"的检测的分数。

  逐个开关实测，而不是凭信心直接发布。在检测器表现正常的序列上，soft-BIoU 和相似度项
  带来小而稳定的提升（IDF1 从 0.510 到 0.554，MOTA 也上升）；检测置信度提升则**视情况
  而定** —— 在那条序列上它把检测器噪声变成新轨迹，MOTA 降了 0.084，但在检测严重不足的
  序列上，它是三项中唯一有帮助的。往哪边走取决于你的检测器主要受漏检还是误检限制，这也
  是它默认关闭的原因。
- **相机运动补偿。** `ByteTracker::applyCameraMotion(affine)` 在下一次更新前用一个 2x3
  变换扭曲所有轨迹。位置和尺寸会被变换，速度不会：速度是目标在世界中的属性，相机一帧的
  抖动不该被理解成目标在加速。变换由调用方提供，因为这个类只看得到框；Python 端直接接收
  OpenCV `estimateAffinePartial2D` 的返回值。在一段让身份切换从 15 次增至 38 次的相机
  运动下，补偿后完全恢复静止相机的结果；相机不动时没有任何开销。
- **第二个 x4 超分模型（SPAN），它不是第一个的替代品。** 加入它完全不需要库代码 ——
  `SuperResolver` 从模型读取放大倍数和 tile，所以换一个超分模型只是换一个路径。两者是
  互补关系，不分高下：SPAN 偏保真，在干净的下采样图上胜出（32.95 dB，bicubic 为 32.29，
  Compact 模型为 30.01），而且**只有 5.8 MB、每 tile 1.13 ms**；面向感知质量、用退化
  数据训练的 Compact 模型则在模糊和 JPEG 压缩的输入上胜出（28.54 vs 27.60 dB）。按你的
  输入长什么样来选。
- **x4 超分辨率，通过 tiling 实现。** 新增 `tasks/superres.{h,cc}`：`planTiles()`、
  `tileWeight()` 和 `SuperResolver`。Python：`bcdl.SuperResolver`、`bcdl.plan_tiles`、
  `bcdl.tile_weight`。模型只放大一个固定尺寸的 tile；任意图像被切成相互重叠的 tile，
  再交叉淡化拼接回去。**每 tile 2.04 ms**；202x270 的帧放大到 808x1080 用时 27.7 ms。

  融合是加权平均 —— 每个 tile 向两个累加器分别贡献 `w * pixel` 和 `w`，结果取两者之比
  —— 正因如此，图像边界不需要特殊处理：边界像素只被一个 tile 覆盖，除以该 tile 自己的
  权重后原样返回。

  **选 tile 尺寸时要把 `.hbm` 考虑进去。** 编译出的指令流随 tile **面积**增长，与权重
  无关：这个网络只有 1.2 MB 的 int8 权重，但按 256x256 编译出的文件有 **147.9 MB**，
  128x128 是 **37.1 MB**，64x64 是 **11.4 MB**。三者的每像素吞吐完全相同（每输入像素
  0.126 vs 0.127 us），所以既然软件里本来就要做 tiling，用更小的模型 tile 几乎没有
  代价 —— 唯一的代价是固定的重叠区在小 tile 里占比更大。发布的模型用 128。

  关于能期待什么：这是一个面向感知质量、GAN 训练的超分模型，所以在干净的下采样图上
  PSNR 反而*低于* bicubic（20.5 vs 21.0 dB），细节却明显更多 —— 梯度能量是真值的
  0.56x，bicubic 只有 0.40x。在它真正针对的退化输入上（模糊加 JPEG q40），它的 PSNR
  也胜出，28.5 vs 27.9 dB。基准程序两个数字都打印，因为单看任何一个都会误导。
- **稀疏局部特征 + 匹配（XFeat）。** 新增 `tasks/features.{h,cc}`：`xfeatPreprocess()`、
  `decodeXfeat()`、`FeatureExtractor` 和 `matchFeatures()`。Python：
  `bcdl.FeatureExtractor`、`bcdl.decode_xfeat`、`bcdl.match_features`、
  `bcdl.xfeat_preprocess`。可重复检出的关键点，带 L2 归一化的 64 维描述子，用于标定、
  拼接、跟踪和 SLAM 前端 —— 这是几何视觉层里与双目、GDC 并列的"检测/描述子"那一半。

  编译后的模型**只有 3.1 MB，约 1.0 ms 跑完**，因为只有卷积主干放在 BPU 上。所有依赖
  数据的部分都刻意放在图外：输入端的 InstanceNorm（逐图像的统计量，最不适合量化的那类
  东西）挪到预处理，softmax、非极大值抑制、top-k 和稀疏采样都放在解码里。这样图才能
  保持直线结构、没有动态形状。

  **开销主要在 CPU 而不是 BPU**，`XfeatConfig::top_k` 就是调节旋钮：每张 640x480 图像，
  预处理 3.7 ms、推理 1.2 ms 是固定的，解码 21-28 ms，匹配一对图像在默认 4096 个特征时
  为 130 ms，1024 个时为 8.4 ms，512 个时为 2.2 ms —— 它随两边特征数的乘积增长。解码
  约 21 ms 的下限来自全分辨率热力图的 scatter 加上对 307k 像素做 NMS，与 top_k 无关。

  在相同输入上对照参考实现：关键点 99.95% 一致，描述子余弦 **1.000000**；板上的特征有
  93% 在 2 px 内复现了浮点结果；匹配找到 326 对，参考实现是 325 对。如果你要自己写解码，
  有个细节值得照抄 —— 描述子用**双三次**采样（参考采样器的默认值），而旁边的可靠性图
  用的是双线性。若把描述子也按双线性读取，所有形状、数量和关键点都对，只是描述子余弦
  掉 0.4%，看起来和浮点噪声一模一样。
- **全身姿态 —— 133 个关键点（身体、脚、面部、双手）。** 新增 `tasks/wholebody.{h,cc}`：
  `wholeBodyPreprocess()` 把人体框转换成模型的画布，`decodeWholeBody()` 把热力图还原成
  原图上的关键点，`WholeBodyEstimator` 把两者绑定到一个 Engine。Python：
  `bcdl.WholeBodyEstimator`、`bcdl.wholebody_preprocess`、`bcdl.decode_wholebody`。
  28 MB 的模型，每人 **1.68 ms / 589 FPS**。

  这是本库第一个**自顶向下**（top-down）的任务：它对每个人的裁剪块各跑一次，所以前面
  需要一个检测器，开销随人数增长；而自底向上的 `PoseEstimator` 一帧只读一次就覆盖所有
  人。换来的是脚、68 点面部和双手，这些是 17 关节的头根本没有的。

  参考实现的几何有两个细节不能错、又容易弄错：模型吃的是 **RGB**（只有前面的检测器
  要 BGR —— 不过 `wholeBodyPreprocess` 和本库其他接口一样接收 BGR，在内部翻转），而且
  裁剪**不是**常见的 mmpose 仿射 —— 框先被加宽、补零到 3:4，再 resize，所以逆映射是纯粹
  的缩放加平移。亚像素精化用 DARK-UDP；bcdl 只对每个峰值周围 13x13 的窗口做模糊，而不是
  对全部 133 张图整图模糊，算术结果在 2e-3 px 以内一致，工作量约为 1/300。

  在相同输入上对照浮点模型：热力图余弦 0.996/0.988；对浮点模型有把握的关键点（score >=
  0.6），误差为平均 2.07 px / 中位数 0.89 px。大误差的长尾全部是被遮挡或出画肢体上的
  低置信度关键点，那里热力图是平的，浮点和 int8 只是各自挑中了不同的噪声峰 —— 常用的
  可见性阈值会把它们滤掉。
- **人脸检测（5 个关键点）+ 对齐。** 新增 `tasks/face.{h,cc}`：`decodeScrfd()`（多尺度
  距离回归加关键点）、`similarityTransform()`（闭式解 Umeyama）和 `alignFace()`（扭曲到
  标准的 112x112 模板），Python 绑定为 `bcdl.FaceDetector`、`bcdl.decode_scrfd`、
  `bcdl.align_face` 和 `bcdl.face_letterbox`。在相同帧上对照浮点参考：关键点误差平均
  0.35 px / 最大 0.89 px，框角点平均 0.52 px。检测 60.2 ms，对齐 0.29 ms。

  识别不需要新任务 —— 对齐后的裁剪块走 `ImageEmbedder`，匹配用 `EmbeddingBank`。有两点
  要知道：这个检测器把补边放在**右下**，而不是居中，所以 `face_letterbox()` 返回的
  LetterboxInfo 补边为零（居中的会让每个框和关键点都偏移）；另外，对齐是正确性的硬性
  要求而不是锦上添花，因为识别模型只见过扭曲到模板上的人脸。

  识别已在板上端到端验证，对照浮点参考**平均余弦 0.975**（最小 0.966），同人对与非同人
  对之间的区分度与浮点一致。为此绕了个弯：嵌入向量起初与浮点模型几乎正交（余弦
  0.015），后来查明是下文修复的 `setInput` stride bug，与识别主干无关。
- **全景驾驶感知 —— 检测 + 可行驶区域 + 车道线。** 新增 `tasks/panoptic_drive.{h,cc}`：
  `decodeYoloV5Anchor()` 和绑定 Engine 的 `AnchorDetector`，这是本库第一个**基于
  anchor** 的解码（其余全是 anchor-free 的 LTRB/DFL）。一次推理同时供给三个头，
  **2.80 ms / 358 FPS**，CPU 后处理 2.64 ms，模型 12 MB。Python 绑定为
  `bcdl.AnchorDetector` / `bcdl.decode_yolov5_anchor`。

  上游公开的导出模型输出一个已经解码好的 `[1,25200,6]` 张量，看上去似乎省掉了 CPU
  解码。**它过不了 BPU 编译**：那段解码由 ScatterND 拼成，编译成功、没有任何报错，但
  结果张量的 objectness 列和类别列从未被写入 —— 每 8 个浮点数只有 2 个有数据，全是
  坐标 —— 所以检测器在任何阈值下都读到零个目标。因此图在解码之前切开，导出三个原始的
  头部卷积，算术放到 CPU 上做。这样还**快 10 倍**：图内解码在设备上耗时 28.33 ms，
  切图后是 2.80 ms。
- Python `Detector.postprocess(lb)`，用于一次推理供给多个解码器的多头模型 —— `detect()`
  表达不了这种情况。
- **实时语义分割头（PIDNet-S）。** 与现有 deeplabv3plus 头相同的 2048x1024 输入，
  **4.52 ms / 221 FPS，对比 49.6 ms / 20 FPS —— 快 11 倍**，模型 19 MB 而不是 38 MB，
  对照浮点参考保持 0.9859 余弦、94.63% 的逐像素一致率。没有新解码器：它原样走现有的
  `Segmenter`，注册为 `semseg_rt` 任务。唯一要注意的区别是它输出 1/8 分辨率的 logits
  （`[1,19,128,256]`），所以标签图是 256x128，由调用方上采样；对它要设置
  `SegConfig::channels_first = true`（NCHW），而 deeplabv3plus 是 NHWC。
- **图像嵌入 —— 检索与零样本分类。** 新增 `tasks/embedding.{h,cc}`：`decodeEmbedding()`
  （读出 + L2 归一化）、`EmbeddingBank`（在向量表上做余弦 top-k），以及绑定 Engine 的
  `ImageEmbedder`。Python 端有 `bcdl.ImageEmbedder`、`bcdl.EmbeddingBank` 和
  `bcdl.embed_preprocess()`，另有 `examples/embed_demo.cc`（图库检索）。在相同输入张量
  上对照模型作者的浮点实现：逐图余弦 0.9977–0.9995，在 200 张图的图库上 recall@1/@5 与
  浮点基线完全相同（0.9250 / 1.0000）。

  这个 API 是围绕两个坑设计的。一个嵌入模型的 `.hbm` 打包了*多个*子模型 —— 一个池化后
  的整图向量和一个逐 patch 的特征网格 —— 所以要按名字绑定池化的那个
  （`Engine::modelNames()` 会列出它们），并检查 `ImageEmbedder::dim()`；否则 patch 网格
  会被展平成一个非常宽、毫无意义的"嵌入向量"。另外，`embed_preprocess()` 刻意做*拉伸*
  resize 而不是 letterbox：这些编码器塔从没见过补边条，事后也没有需要反算的几何。
- **`VideoEncoder` 解耦 API + H.265 编码。** `feed(frame, pts_us)` /
  `receive(timeout_ms)` / `flush()` / `feed_end_of_stream()`，与现有的 `encode()` 一起
  绑定到 Python。H.265 编码现在可用，由
  `tests/test_codec_py.py::test_video_encode_decoupled` 覆盖（对两种编解码格式参数化）；
  此前编码器只以 H.264 跑过。
- **`Engine::modelNames(path)` / `bcdl.Engine.model_names(path)`** —— 不加载即可列出
  `.hbm` 里打包的所有模型。`.hbm` 是一个*包*，可能装着多个模型（官方 SigLIP 编码器就在
  一个文件里放了全局嵌入和 patch 特征两个子模型）。构造函数早就能按名字选其中一个，但
  之前没有办法知道有哪些名字。

### 变更
- **`VideoEncoder` 基于 `media_codec`（`hb_mm_mc_*`）流式 API 重写**，取代
  `hbVPVideoEncode` + 紧接着调用 `hbVPGetVideoEncOutputBuffer(task)` 的做法 —— 也就是
  "进一帧，从同一个 task 取回这一帧的输出"的模型；它在真实 HEVC 解码上超时，用在编码器
  上同样是错的，因为码率控制器可能缓存一帧、什么也不输出。现在输入入队与码流包取出彼此
  独立。`encode()` 的签名和语义不变（feed + 短暂等待，可能返回空）。

  已在 640×384 和 1920×1088 下验证 H.264 与 H.265，对照源图案最差 Y MAE 为 0.09–0.16
  个灰阶；1080p HEVC 码流用 `ffprobe` 做了外部验证，关键帧与配置的 `intra_period` 吻合。

  **调用方必须持续取出（drain）**：并不是每喂一帧就一定出一个包；`feed()` 返回 `false`
  时（输入队列暂时满了 —— 在 1080p 下很容易遇到，因为生成一帧远比编码一帧便宜），正确的
  做法是先取出再重试，而不是当成致命错误。收尾需要 `flush()`。
  `examples/video_roundtrip.cc` 和 `examples/video_det_demo.py` 演示了这个节奏；后者以前
  假设每帧一个包，因此在高负载下丢帧，还丢掉了尾部。

### 修复
- **`Engine::setInput` 忽略了输入的行 stride。** 输入缓冲区按模型解析出 stride 后的
  属性分配，但往里拷贝用的是一次平铺的 `memcpy`，只靠 `bytes > size` 做保护 —— 所以比
  设备缓冲区短的紧密数组会被悄无声息地写进去。输入行连续的模型（几乎全部）不受影响；
  行有补齐的模型就中招了。ArcFace R50 正是例子：112 个浮点列是 448 B/行，补齐到
  **512 B stride**，于是第一行之后的每一行都错位，不管喂进去什么，模型看到的都是同一张
  打乱的图，产出近乎共线的嵌入向量。

  `setInput` 现在只接受两种大小：`inputPackedBytes(i)` —— 连续的行主序数组，逐行散布到
  设备布局，补齐字节置零；或 `inputBytes(i)` —— 已经是设备布局的数据。**其他任何大小
  现在都会抛异常**，不再被当作短前缀拷进去。新增访问器 `inputPackedBytes()`、
  `inputStride()` 和 `inputData()`（Python：`input_packed_bytes`、`input_stride`、
  `input_buffer_bytes`），让布局可以查询；任何不经 `setInput`、直接写输入缓冲区的代码
  都必须遵守 `inputStride`。`outputStride()` / `output_stride()` 在输出侧做同样的事，
  输出侧也有补齐：一个 `[1,133,64,48]` 的浮点热力图返回时行 stride 为 256 字节，所以把
  原始输出缓冲区直接平铺 reshape 会让每张图都错切。（`outputAsFloat()` 和各任务解码器
  早已处理了这一点 —— 这个访问器是给直接读缓冲区的代码用的。）
- **编码器 `bitstream_buf_size` 下限钳到编解码器要求的 64 KiB。** 原先按 `w*h*3/2`
  计算，所以小帧（256×128 = 48 KiB）会让 `hb_mm_mc_configure` 拒绝整个 context。
- **一个进程里放不下两个 1080p 编解码 context。** 编码器存活时（5 个帧缓冲），解码器
  （8 个）在 1920×1088 下始终拿不到缓冲：它不出帧，输入队列被填满，`feed()` 开始返回
  false。这是容量限制而不是并发问题 —— 在 640×384 下两者都放得下，因而掩盖了它。
  `examples/video_roundtrip.cc` 在创建解码器之前先销毁编码器；真正的转码器应当有意识地
  设置 `frame_buf_count`，而不是假设两边的默认值都放得下。
- **`scripts/fetch_models.sh` 可自愈。** 有五个模型不在脚本里，也从没放到板子上过，所以
  "重新运行它即可补齐"这句话并不成立。现在板上有本地副本时直接使用，否则从转换机拉取，
  并暂存下来供下次使用；传输时用 `.part` 临时文件名，这样截断的传输就无法冒充完好的
  模型。

### 说明
- **ReID 模型需要 QAT；int8 PTQ 对这个网络不够用。** PTQ 编译顺利，返回格式完全正确的
  单位向量，但 Market-1501 Rank-1 只有 51%，浮点模型是 85% —— 三种架构变体、三种校准
  方法、六倍大的校准集，全都以同样的方式失败。以浮点模型的嵌入向量为目标做 QAT 自蒸馏
  （不需要身份标签，也不需要 ReID 训练集：缺陷在于函数拟合，而不在于身份学习），恢复到
  **Rank-1 84.6%、板端对浮点余弦 0.980**。一个真正塌缩了的量化嵌入塔依然会返回单位
  长度、确定性、形状正确的向量，所以能抓住它的是对照浮点参考的余弦和检索指标，而不是
  形状检查。

## [0.3.1] — 2026-07

打包修复。库与绑定均无改动。

### 修复
- **重新生成 `python/bcdl_py.pyi`。** 已提交的类型存根落后于 nanobind 绑定，缺了 14 个
  类/枚举、8 个模块函数和 `InstanceSegConfig.reg_max` —— 包括双目流水线、Mono3D 头、两个
  GDC 类，以及 SOTA 头（`LabelMap`、SAM 掩膜解码器、ReID 嵌入向量辅助函数）。
  `GdcLetterbox` 也仍然声明着 0.3.0 已删除的 `bin_path` 参数。由于存根随 wheel 和 conda
  包一起发布，这个缺口在用户那里表现为类型信息缺失。类型和模块函数现在与绑定一一对应：
  67/67 与 28/28。

  重新生成需要在板上、在开启 GDC 的 build 中使用 `-DBCDL_BUILD_STUBS=ON`（存根是通过导入
  模块生成的，所以不带 GDC 的 build 会悄悄丢掉 `GdcLetterbox`/`GdcRemap`）。注意，普通的
  重新构建不会刷新它：存根目标以 `bcdl_py.so` 为依赖，一旦 `.pyi` 比模块新，ninja 就会
  跳过它。删掉 `.pyi` 即可强制重新生成。

## [0.3.0] — 2026-07

视频路径的加固，以及 NV12 原生预处理。压缩视频流水线快了 1.8 倍，不再绕道 BGR。

### 变更（不兼容）
- **`GdcLetterbox` 在构造时生成 warp LUT** —— `bin_path` 参数去掉了，也不再需要离线的
  `.bin`。仿射 letterbox 被表达为一个 CUSTOM 采样网格，在运行时生成。
- **`VideoDecoder::feed()` / `decode()` 每次只接收一个访问单元（AU）。** 解码器现在运行在
  `MC_FEEDING_MODE_FRAME_SIZE` 下；不再支持喂任意字节块。
  （`AsyncVideoDetectionPipeline::submit()` 仍接收任意 Annex-B 字节，并替你完成重组。）
- **`StageProfile` 新增 `cvt_ms`**，`totalMs()` 现在会把它计入。

### 新增
- **`AsyncDetectionPipeline::submitNv12()`** —— 把解码出的 NV12 帧直接 letterbox 进模型
  输入槽位，有 GDC 硬件引擎时用它，否则用 CPU。没有 BGR 往返，也不拷贝帧。拆分为
  `acquireSlot()` / `letterboxIntoSlot()` / `commitSlot()`，供那些在等待流水线容量时不能
  持有回收缓冲区的调用方使用。
- **`letterboxNv12Cpu()`** —— NV12→NV12 的 letterbox，不做颜色转换；GDC 不可用时的
  回退。`BCDL_NO_GDC=1` 可强制使用它，便于 A/B 对比。
- **`YuvRange`**（`preproc/geometry.h`） —— 把 studio range（有限范围）→ full range
  （全范围）的扩展显式化。视频携带的是 studio swing 的 NV12（Y 在 [16,235]），而模型是在
  全范围像素上校准的；旧的 BGR 往返是碰巧完成了这个转换，只是 `cv::cvtColor` 的副作用。
- `StageProfile::cvt_ms` / `cvtPerFrame()`（Python：`cvt_ms`、`cvt_per_frame`）。

### 修复
- **VPU `INTERRUPT TIMEOUT` 卡死。** 喂得比取得快会填满解码器的帧缓冲，下一次硬件解码
  就会卡住。现在每条解码路径都是喂一个 AU、取出所有就绪帧，最后再 flush 重排序的尾部。
- **堆损坏 / 随机 `SIGSEGV`。** 解码器原先运行在 `MC_FEEDING_MODE_STREAM_SIZE` 下，其
  码流环形缓冲的更新路径会从编解码器内部破坏 glibc 堆。即使进程里没有 BPU，单线程解码
  本身就 20 次里崩 6 次；`FRAME_SIZE` 没有这个问题，而且更快（纯解码 452 vs 281 FPS），
  也不再丢帧。
- **H.265 卡在 16 FPS 并丢帧。** 阻塞式的输出出队会持有编解码器的内部锁，所以只要解码器
  没有就绪帧，喂数据线程与接收线程之间的竞争就会卡住。现在由一个线程独占编解码器
  context。
- **背压下的死锁。** 面向调用方的结果队列是有界的，队列满了就阻塞推理阶段，推理阶段一
  阻塞就不再回收槽位，进而卡住所有上游阶段 —— 而唯一能取空队列的调用方，此时正阻塞在
  `submit()` 里。
- **`~VideoDecoder` 卡死**，发生在解码器仍持有已解码帧时：`hb_mm_mc_stop()` 会等编解码器
  清空输出端口，而只有应用能清空它。析构函数现在先归还所有待处理的缓冲区。
- **每解码一帧就打印一行 `Component vdec_render isn't ready!` 错误。** 出队时没取到帧就会
  （向 stdout）打印这一行，而"取到空为止"的节奏会让每个访问单元都以这样一次调用结束。
  `recv()` 现在先调用 `hb_mm_mc_get_status()`，它不会打印。
- **`AsyncVideoDetectionPipeline` 在流结束时丢掉最后一帧**：末尾的缓冲区可能装着两个
  访问单元，而在 `FRAME_SIZE` 模式下只有第一个会被解码。
- **视频流水线中的逐帧设备内存分配**（319 帧 334 次分配 → 22 次，全部在预热阶段）。
- **`nv12ToBgrCpu()` 的两种构建配置产出不同的像素** —— OpenCV 快速路径做了 studio swing
  扩展，手写的回退路径没做。现在它接收一个 `YuvRange`，两个分支都实现了每种范围。

### 性能
- `AsyncVideoDetectionPipeline`，1080p H.264，yolo26n：**249 → 441 FPS**（1385/1385
  帧），瓶颈在解码。H.265：**439–451 FPS**（300/300 帧）。
- 纯 VPU 解码：**452 FPS**（H.264）/ **481 FPS**（H.265）。
- GDC 硬件 letterbox，1920×1080 → 640×640：墙钟 0.97 ms/帧，其中 CPU 约 0.3 ms；它所
  取代的 BGR 链路纯 CPU 需要 4.71 ms。

## [0.2.0] — 2026-07

### 新增
- **视频目标检测 —— `AsyncVideoDetectionPipeline`** —— 压缩视频 → 检测结果的完整路径都在
  C++ 里：`submit(Annex-B bytes)` → 内部 AU 切分 → 四个相互重叠的阶段（VPU 解码 ‖
  NV12→BGR ‖ CPU 预处理 ‖ BPU 推理+NMS）→ `next()`/`next_nowait()`。调用方只需薄薄一层，
  把 `ffmpeg -c copy` 的字节流泵进去，就能达到解码瓶颈的上限（yolo26n 1080p H.264 约
  240 FPS）；Python 绑定会释放 GIL。新增 demo `examples/video_det_demo.{cc,py}`、
  `examples/video_det_demo_async.cc`，以及轻量的 RTSP 驱动 `examples/rtsp_det_demo.py`。
- **H.265 解码** —— `VideoDecoder` 基于 `media_codec`（`hb_mm_mc_*`）流式 API 重写，
  支持重排序，接口解耦为 `feed()` / `receive()` / `flush()`（旧的逐 AU 模型在 HEVC 上
  超时）。注意：分层 GOP 的 HEVC 码流（相机 SVC / 智能编码）只解码基础时域层。
- **逐阶段 `StageProfile` 计时** —— `DetectionPipeline` 和 `AsyncDetectionPipeline` 给出
  每帧 `decode` / `preproc` / `infer` / `postproc` 的毫秒数（异步路径中最慢的阶段决定
  吞吐），另加 `AsyncDetectionPipeline::tryNext()`（非阻塞弹出）和 `nv12ToBgrCpu()`
  （NV12→BGR，OpenCV SIMD + BT.601 回退）。
- **`bcdl::GdcRemap` —— 基于 VPS GDC 引擎的硬件稠密 remap**（+ Python 绑定
  `bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)`）。任意的**固定** warp，语义
  与 cv2.remap 相同（为双目校正而做），在构造时编译成 GDC CUSTOM 网格 warp LUT —— 通过
  直接驱动 `libgdcbin`（`gdc_init`/`gdc_set_custom_points`/`gdc_calculate`）**在运行时**
  生成；不需要离线 bin 文件。公开的 `hbn_gen_gdc_cfg` 做不到这一点（它从不转发自定义
  网格点 —— 已通过反汇编验证）。从 `transform_custom` 中还原出的关键语义（SDK 头文件
  注释是错的/有误导性）：`custom.w/h` 是 **tile 数**（点数 = `(w+1)*(h+1)` 个节点），
  `centerx/centery` 以**网格索引**为单位（`w/2, h/2`），而不是输入像素。实测
  （2448×2048 NV12，真实双目校正映射，S100P）：**墙钟 6.3 ms/帧，其中 CPU 约 1 ms**
  （拷入 0.5 + 拷出 0.5；5.3 ms 的 GDC 操作期间 CPU 空闲），对比 `cv2.remap` 占满全部
  6 个核要 14.7 ms；在映射有效区域内与 cv2.remap 的差异为平均 0.25 / p99 2 个灰阶。
  正确性/基准套件：`tests/test_gdc_remap.py`。`GdcRemap.run` 会释放 GIL，所以两个实例
  （双目上/下）可以在 Python 线程里让一路相机的 CPU 转换与另一路的硬件操作重叠
  （一个双相机立体视觉应用实测两路相机合计 24.3 → 18.6 ms）。
- **开放词表检测 / 分割（YOLOE）** —— 无提示词（prompt-free）的 YOLOE 复用现有的
  LTRB / DFL 解码，配合 `bcdl::LabelMap`（默认 COCO-80，可用 `from_file` /
  `from_list`）；没有新的后处理数学。板端基准 + 校验图：`yoloe_det`、`yoloe_seg`
  （S100P 上推理约 520 / 约 397 FPS）。
- **可提示分割（EdgeSAM / SAM 掩膜解码器尾段）** —— `SamConfig`、`SamMask`、
  `SamMaskDecoder`、`decode_sam_masks`，以及两阶段的 `SamSession` Python 封装（RepViT
  编码器 → 缓存的嵌入 → 点/框提示解码器）。`tests/test_promptable_seg.py`。
- **ReID 外观嵌入向量** —— `normalize_embedding`（L2）+ `cosine_similarity`，BoT-SORT
  关联所用的基础原语。`tests/test_reid.py`。
- **QAT 浮点输入检测路径** —— `YoloLtrbDetector.detect_float()` 和
  `letterbox_chw_float()` 用于喂 `[1,3,H,W]` float32（QAT 导出）模型；LTRB/DFL 解码与
  NV12 PTQ 路径完全相同。
- **实例分割 DFL 框头** —— `InstanceSegmenter` 自动识别普通 LTRB（4 通道）与 DFL
  （`4*reg_max`，如 64）框通道，所以 yolov8/v11/YOLOE 的分割头无需配置即可解码；为底层的
  `decode_instance_seg()` 入口公开了 `InstanceSegConfig.reg_max`。

### 变更
- **语义分割 argmax 快了 73×** —— `Segmenter::postprocess()` 直接在 F32 设备缓冲区上沿
  通道轴做 argmax（感知字节 stride，OpenMP），而不是先把整个 logit 张量逐元素物化出来。
  全分辨率 deeplabv3plus `[1,1024,2048,19]` 解码从约 660 ms 降到约 8.7 ms，标签逐字节
  一致。模型未变（物化的回退路径覆盖 F16/量化输出）。
- **文档大修** —— README 重新组织：围绕 12 个任务的校验图展示（Gallery）、文档导航中心、
  徽章/目录、更完整的私有 conda 安装说明（环境版本锁定、channel 配置、冒烟测试）和快速
  上手。平台支持写明为 RDK **S100 / S100P / S600**（同一套 S 系列运行时；编译出的
  `.hbm` 与 march 绑定）。`scripts/board_bench.py` 现在会在 JPU 校验图上标注实测的 cv2
  与 JPU 对比柱状条，并生成 ByteTrack 校验图（`benchmarks/figures/track.jpg`）。

### 新增
- **双语 README** —— 默认简体中文（`README.md`），附完整英文译本（`README.en.md`），两个
  文件顶部都有语言切换（`English | 简体中文`）。
- **双目** —— `StereoPipeline`：两张图（左/右）的 F32 NCHW RGB 输入（resize **或**中心
  裁剪适配）→ 视差，可选度量深度（`z=fx·baseline/disp`）和几何有效性掩膜（视差范围 +
  左边界 + 可选的左右一致性）。纯函数辅助 `pack_stereo_input` / `disparity_to_depth` /
  `stereo_valid_mask` + nanobind 绑定 + `examples/stereo_demo.cc`。已在 LAS2-M 上验证
  （对照参考预处理余弦 1.0 / EPE 0 px，480×640 下 70 FPS）。
- 打包：`find_package(bcdl)` CMake 导出（`bcdlConfig.cmake` + `bcdlTargets.cmake`，内置
  `FindHobot.cmake`）、`install()` 规则，以及生成的 `bcdl/version.h`（`BCDL_VERSION*`
  宏）。
- Python：通过 scikit-build-core 支持 `pip install .`（`pyproject.toml`）、`py.typed`
  标记，以及 `bcdl.__version__`。
- 文档：`CONTRIBUTING.md`，以及完整的 Python（`docs/API.md`）与 C++（`docs/CPP_API.md`）
  API 参考 —— 涵盖每个类、配置和解码器，每个任务配一段用法示例。

### 修复
- 文档：修正了过时的基准数字（argmax 提速后语义分割解码 `660 → 58 ms`；JPU 解码
  `2.6–3.5× → ≈3.6–5.3×`，现与实测表格一致）、跟踪模块的位置（`tracks/`，不是
  `tasks/`），以及 API 参考中的签名不符（`Detector` 封装没有 `postprocess()`；`decode_*`
  的 letterbox 参数名）。

## [0.1.0] — 2026-06

首个打 tag 的版本。已在 RDK S100P 上板端验证（79 通过，1 跳过）。

### 新增
- **核心** —— `SysMem` / `Task` / `MemPool`（`hbUCP` 之上的 RAII），基于 `hbDNN` 的
  `Engine`，自动维护缓存纪律，零拷贝 / 反量化的输出读取器。
- **任务** —— 检测（LTRB 多尺度 + DFL：YOLO26 / YOLOv8 / v5 / v11）、分类、姿态、实例
  分割、旋转框（OBB）、语义分割、单目深度，以及三段式 **PP-OCRv5** OCR 流水线（DBNet
  检测 → PP-LCNet 方向分类 → CRNN/CTC 识别）。每个解码器也都以不依赖 Engine 的纯
  `decode_*` 函数形式公开。
- **跟踪** —— ByteTrack 多目标跟踪器（Kalman + 匈牙利算法 + 两阶段关联）。
- **多媒体** —— 硬件 JPEG（JPU）和 H.264 / H.265（VPU）编解码。
- **流水线** —— 同步、复用缓冲区的 `DetectionPipeline`，多线程的 `AsyncDetectionPipeline`
  （预处理 ‖ 推理重叠，在 Python 中释放 GIL），`TrackingPipeline`，以及视频文件 → 解码
  → 检测的路径。
- **Python** —— 覆盖每个解码器 + 流水线的 nanobind 绑定（NumPy 输入/输出）。
- **工具** —— 板端基准 + 校验图生成器（`scripts/board_bench.py`）、仓库内的模型目录
  （`scripts/fetch_models.sh`）、两层测试套件（不依赖 Engine 的 NumPy 测试 + 板上真实
  模型测试）。

[Unreleased]: https://github.com/ruisv/bcdl/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/ruisv/bcdl/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/ruisv/bcdl/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ruisv/bcdl/compare/v0.7.5...v0.8.0
[0.7.5]: https://github.com/ruisv/bcdl/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/ruisv/bcdl/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/ruisv/bcdl/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/ruisv/bcdl/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/ruisv/bcdl/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ruisv/bcdl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ruisv/bcdl/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ruisv/bcdl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ruisv/bcdl/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ruisv/bcdl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ruisv/bcdl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ruisv/bcdl/compare/v0.1.1...v0.2.0
[0.1.0]: https://github.com/ruisv/bcdl/tree/v0.1.1

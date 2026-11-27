# 参与贡献 BCDL

[English](CONTRIBUTING.md) | **简体中文**

感谢你对 BCDL 的关注 —— 欢迎任何形式的贡献：bug 报告、修复、新的任务解码器、基准测试，
或是文档改进。本指南介绍如何搭建环境、构建、测试和提交改动。

提交贡献即表示你同意你的贡献以本项目的 [Apache License 2.0](LICENSE) 授权。

## 目录

- [唯一的硬约束：在板上构建](#唯一的硬约束在板上构建)
- [项目结构](#项目结构)
- [环境搭建](#环境搭建)
- [构建](#构建)
- [测试](#测试)
- [编码约定](#编码约定)
- [提交与更新日志](#提交与更新日志)
- [发起 Pull Request](#发起-pull-request)
- [报告 bug](#报告-bug)
- [范围：哪些东西属于本仓](#范围哪些东西属于本仓)

## 唯一的硬约束：在板上构建

BCDL 链接 D-Robotics 的 **hobot SDK**（`hbDNN`、`hbUCP`、`hb_vp`、media/codec），
而它**只存在于 RDK S100 / S100P / S600 硬件上**（`/usr/include/hobot`、
`/usr/hobot/lib`，或 `hobot-dnn` / `hobot-media` conda 包）。因此：

- **C++ 库、示例和 `bcdl_py` 模块必须在 S100 / S100P / S600 板上构建和运行**
  （aarch64、Ubuntu 22.04、GCC 11、CMake ≥ 3.22）。
- **不要试图在板子以外的主机上编译。** 编辑、代码评审，以及不依赖 Engine 的 NumPy
  解码测试在哪儿都能做；凡是碰到 SDK 的都不行。

典型的工作循环是"在工作站上编辑，同步到板子，在板上构建和测试"。板上的 Python 环境
由 conda 管理（`bcdl` 环境）。

## 项目结构

```
include/bcdl/, src/   库：core / backend / preproc / media / tasks / tracks / pipeline
  core/        SysMem · Task · Status · MemPool        (libhbucp -> hbUCP*)
  backend/     Engine、输出读取器                      (libdnn  -> hbDNN*)
  preproc/     CPU letterbox + BGR->NV12；VP 单元      (hb_vp)
  media/       JpegCodec (JPU) · VideoCodec H.264/H.265 (VPU)
  tasks/       det · cls · pose · seg · obb · semseg · depth · ocr
  tracks/      ByteTrack 多目标跟踪器
  pipeline/    同步 / 异步检测 · 跟踪 · 双目
python/        nanobind 绑定（NumPy <-> 张量）
examples/      独立的 C++ 程序
tests/         pytest 测试套件（NumPy 解码 + 板端端到端）
benchmarks/    板端基准测试驱动 + 结果
scripts/       同步 / 构建 / 初始化 / 拉取模型 / 基准测试
docs/          API.md、CPP_API.md（Python + C++ API 参考）
```

## 环境搭建

在板上做一次性初始化，按 [`env/environment.yml`](env/environment.yml) 创建 conda 环境：

```bash
scripts/bootstrap_board.sh
```

如果要跑板端测试 / 基准测试，接着把模型拉下来（`.hbm` 文件很大，**不入库**；BCDL 本身
不分发任何权重）：

```bash
scripts/fetch_models.sh        # 写入 models/（已被 gitignore）
```

辅助脚本默认使用 ssh 主机别名 `rdk` 和远端路径 `projects/bcdl`；用环境变量覆盖，
即可指向你自己的板子：

```bash
BOARD=myboard DEST=path/to/bcdl scripts/sync.sh
```

## 构建

在工作站上迭代，在板上构建：

```bash
scripts/sync.sh                # rsync 工作树 -> 板子（快，适合未提交的改动）
scripts/board_build.sh         # 在板上的 build/ 里跑 cmake + ninja
scripts/board_build.sh --run   # 构建，然后运行 det_infer ($HBM)
scripts/board_build.sh --clean # 先清空 build/
```

或者直接在板上、conda 环境里：

```bash
conda activate bcdl
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

这会构建 `bcdl` 库、C++ 示例和 `bcdl_py` Python 模块。要构建可 pip 安装的 wheel
（scikit-build-core）：`pip install .`。

## 测试

**每一处行为变更**都要新增或更新测试。测试套件分两层：

1. **解码数学 —— 不依赖 Engine 的 NumPy 测试。** 它们通过 `decode_*` 绑定把后处理
   （NMS、掩膜拼装、关键点、CTC、旋转 IoU、argmax……）钉死，**不需要板子，也不需要
   模型**。在哪儿都能跑：

   ```bash
   PYTHONPATH=build:python pytest tests/test_detection.py tests/test_pose.py \
       tests/test_obb.py tests/test_instance_seg.py tests/test_depth_seg.py \
       tests/test_depth_seg_py.py tests/test_classification.py tests/test_ocr.py
   ```

2. **板端、真实模型端到端。** 它们在真实的 `.hbm` 模型上验证完整的 BPU / 编解码 /
   流水线路径；模型或图片缺失时，每个测试都会**干净地跳过**。在板上、`models/` 已填充
   的情况下运行：

   ```bash
   PYTHONPATH=build:python pytest tests/
   ```

装好 BCDL 之后（conda 频道或 pip wheel），可以不设 `PYTHONPATH`，直接对已安装的模块跑
`pytest tests/`。

**准则**

- 新的后处理逻辑要配一个通过 `decode_*` 绑定的确定性 NumPy 测试，离开板子也能复现。
- 新的、依赖模型的行为要配一个板端测试，并在模型缺失时跳过（照
  `tests/test_board_models.py` 里现有的写法来）。
- 保持测试的确定性 —— 不联网、固定随机种子；在合适的地方用容差比较，而不是要求浮点数
  完全相等。

## 编码约定

- **语言** —— C++17。头文件用 `.h`，实现用 `.cc`。所有代码都在命名空间 `bcdl` 里。
- **错误** —— 用 `BCDL_CHECK(...)`，它会抛出 `bcdl::Error`；不要另起临时的错误路径。
- **内存 / 缓存纪律** —— 缓冲区用 `hbUCPMallocCached` 分配；CPU 写完输入后必须
  `cleanCache()`，读取设备写出的输出前必须 `invalidateCache()`。`Engine` 在推理路径上
  会处理这些 —— 请经由它走，不要绕过缓存刷新。
- **默认零拷贝** —— BPU 张量、JPU/VPU 图像和 VP 缓冲都共享 `hbUCPSysMem`；优先让缓冲
  在这套共享内存体系中直接传递，而不是拷贝。
- **OpenCV 是可选的** —— 图像操作用 `#ifdef BCDL_HAVE_OPENCV` 守卫，并配有手写的回退
  实现；改到这些地方时，两条路径都要保持可用。
- **风格** —— 与周围代码保持一致（命名、注释密度、结构）。从 CUDA 技术栈（ccdl）
  移植来的后处理，在这里改为 CPU/NEON 实现。

## 提交与更新日志

- 每个提交保持聚焦，提交信息用祈使语气。
- 遵循日志中已有的 [Conventional Commits](https://www.conventionalcommits.org/) 风格
  （`feat:`、`fix:`、`build:`、`docs:`、`test:`……）；鼓励加上 scope，例如
  `fix(tasks): ...`。
- 对用户可见的变更记录在 [`CHANGELOG.md`](CHANGELOG.md) 的 `[Unreleased]` 下，同时
  更新 [`CHANGELOG.zh.md`](CHANGELOG.zh.md) 的对应条目。本项目遵循
  [Keep a Changelog](https://keepachangelog.com/) 和
  [Semantic Versioning](https://semver.org/)（语义化版本）。版本号的唯一真源是
  `CMakeLists.txt` 里的 `project(... VERSION ...)`，并与 `pyproject.toml` 保持同步。

## 发起 Pull Request

发起 PR 之前，确认：

1. 相关测试通过 —— 至少要跑离板的解码测试；如果改动涉及 SDK / 模型 / 流水线路径，
   板端套件也要跑。
2. 新增或改动的行为有测试覆盖。
3. `CHANGELOG.md` 记下了所有对用户可见的变更（同时更新 `CHANGELOG.zh.md` 的对应条目）。
4. diff 保持聚焦；无关的清理另开 PR。

在 PR 描述里写清改了**什么**、**为什么**改，注明是否在板上验证过（以及是哪块板：
S100 / S100P / S600，软件版本多少）；凡是和性能相关的改动，附上改动前后的数字。

## 报告 bug

提 issue 时请附上：

- 板子型号和软件版本（例如 S100P，sw 4.0.5），以及 BCDL 的获取方式（conda 包版本，
  或源码构建所用的 commit）。
- 涉及的模型 / 任务和输入形状、一个最小复现（命令或简短脚本），以及完整的报错 / 异常输出。
- 如果是正确性问题，写明你期望得到什么、实际得到了什么。

## 范围：哪些东西属于本仓

BCDL 是一个**板端运行时**库：加载编译好的 `.hbm`、运行推理、预处理 / 后处理、硬件编解码
以及流水线 —— 全部在板上完成。

**不在范围内：** 模型转换。ONNX → `.hbm`、PTQ 校准和余弦验证，由独立项目用 D-Robotics
OpenExplorer 工具链离线完成；BCDL 只消费最终的 `.hbm`。新增任务解码器、媒体路径、
流水线或绑定的改动在范围内；转换 / 校准工具链则不在。

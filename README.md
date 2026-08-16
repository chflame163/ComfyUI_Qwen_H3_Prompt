# ComfyUI Qwen H3 Prompt

在 ComfyUI 内使用本地 Qwen3.8-27B 和九个 MiniMax-H3 官方 Skill 生成 H3 提示词与制作方案。节点在 ComfyUI 内启动独立的本地 `llama-server`，不依赖外部服务；完成运行时安装后，推理过程不会访问互联网。

## 运行时自动安装

项目只提供一个跨平台安装入口 `install_runtime.py`。它仅使用 Python 标准库，自动识别 Windows、Linux 或 macOS、x64/ARM64 架构及显卡，并为当前机器部署固定的 llama.cpp `b10436` runtime，不需要安装额外的 pip 包。

Windows ComfyUI 官方便携包在节点目录中执行：

```powershell
cd ComfyUI\custom_nodes\ComfyUI_Qwen_H3_Prompt
..\..\..\python_embeded\python.exe install_runtime.py
```

Linux 与 macOS 在节点目录中使用 ComfyUI 所在的 Python 环境执行：

```bash
python3 install_runtime.py
```

首次启动 ComfyUI 前必须先执行安装器。安装器会依次完成硬件检测、后端选择、下载、SHA256 校验、安全解压、`llama-server --version` 与 `--list-devices` 验证，最后生成本机专用的 `runtime_config.json`。下载缓存、源码缓存和安装结果都保存在节点的 `runtime/` 目录中，不会提交到 Git。节点会拒绝加载其他操作系统生成的配置；缺少配置时会提示先运行安装器。

常用命令在三个系统中一致：

```bash
# 只显示检测结果和计划，不下载或安装
python3 install_runtime.py --dry-run

# 查看当前系统和架构支持的后端
python3 install_runtime.py --list-backends

# 手动指定后端；手动指定时不自动回退
python3 install_runtime.py --backend vulkan

# 强制重新安装
python3 install_runtime.py --force

# 只使用 runtime/.downloads/b10436 中已经校验的文件
python3 install_runtime.py --offline
```

Windows 便携包用户可将上述 `python3` 替换为 `..\..\..\python_embeded\python.exe`。

### 自动后端选择

| 系统和设备 | 自动顺序 |
| --- | --- |
| Windows x64 NVIDIA | CUDA 12.4 → Vulkan → CPU |
| Windows x64 AMD | ROCm 7.14 → Vulkan → CPU |
| Windows x64 Intel | SYCL → Vulkan → CPU |
| Windows x64 未识别 GPU | Vulkan → CPU |
| Windows ARM64 NVIDIA | CUDA 13.4 preview → CPU |
| Windows ARM64 Qualcomm Adreno | OpenCL → CPU |
| Linux x64 Intel | SYCL FP16 → Vulkan → CPU |
| Linux x64 NVIDIA/AMD/未识别 GPU | Vulkan → CPU |
| Linux ARM64 | Vulkan → CPU |
| macOS Apple Silicon 或 Intel | Metal → CPU |

Windows 可用后端取决于架构，包括 `cuda12`、`cuda13`、`rocm`、`sycl`、`vulkan`、`opencl` 和 `cpu`。Windows CUDA 官方包同时包含对应 CUDA runtime，下载量会大于 Vulkan。仓库不包含预编译 `bin` 目录，所有 Windows 用户也必须先运行 `install_runtime.py`。

Linux 官方预编译后端包括 `vulkan`、`sycl`、`sycl-fp32`、`openvino` 和 `cpu`，具体选项取决于架构。官方文件是 Ubuntu 构建，其他发行版需要兼容的 glibc/libstdc++。Vulkan、Intel SYCL/OpenVINO 还需要系统中已经存在相应的显卡驱动和运行库。

macOS 使用对应架构的官方包。Metal 不需要额外 GPU 计算框架，但 Qwen3.8-27B Q4 和视觉投影器仍需要足够的统一内存；CPU 只作为功能回退。

### Linux CUDA 源码部署

官方 `b10436` 没有提供 Linux CUDA 压缩包。NVIDIA Linux 用户可显式要求安装器从源码构建：

```bash
python3 install_runtime.py --backend cuda --build-from-source
```

源码构建不会调用 `sudo` 或自动修改系统。运行前必须已经具备：

- 可正常运行 `nvidia-smi` 的 NVIDIA 驱动；
- 包含 `nvcc` 的 NVIDIA CUDA Toolkit，`nvcc` 位于 `PATH` 中或已经设置 `CUDA_HOME`；
- Git、CMake、C 编译器和 C++ 编译器，例如 Ubuntu/Debian 的 `git cmake build-essential`。

安装器固定检出 tag `b10436`，并校验完整 commit `6fed9f6ff7a603b124cb8c5864fca6ea879f9f99`。构建使用 `GGML_CUDA=ON` 和 `GGML_NATIVE=OFF`，生成可覆盖不同 NVIDIA CUDA GPU 的 runtime。默认最多使用 8 个并行任务，可以根据主机内存调整：

```bash
python3 install_runtime.py --backend cuda --build-from-source --jobs 4
```

源码缓存在 `runtime/.sources/llama.cpp-b10436/`，临时编译目录会在成功或失败后清理，最终产物安装到 `runtime/linux-<arch>/cuda/b10436/`。只有源码 commit、程序版本和 CUDA 设备全部验证成功后才会写入配置。

强制离线重编译要求源码缓存存在、commit 正确且工作区无修改：

```bash
python3 install_runtime.py --backend cuda --build-from-source --offline --force
```

源码构建不会加入默认回退链。缺少工具链、编译失败或未检测到 CUDA 设备时，安装器会以红色错误信息退出，不会自动改用 Vulkan/CPU。安装后的 runtime 仍依赖本机 NVIDIA 驱动和构建时使用的 CUDA Toolkit 动态库。

## 模型目录

模型固定从以下目录读取：

```text
ComfyUI/models/LLM/Qwen3.8/
  Qwen3.8-27B-Q4_K_M.gguf
  mmproj-F16.gguf
```

## 节点

在 `😺dzNodes/Qwen_H3_Prompt` 分类中添加 `Qwen H3 Prompt (Local)`。

- `skill`：`auto` 或九个官方 Skill 原名。
- `duration`：目标 H3 视频时长，默认 `10.0` 秒。
- `think_mode`：布尔开关，默认关闭；自动在 Qwen 官方 instruct 和 thinking 推荐参数之间切换。
- `reasoning_effort`：仅在 `think_mode` 开启时生效，默认 `medium`。
- `seed`：ComfyUI 标准种子控件，支持运行后固定、递增、递减或随机；传入 llama.cpp 前会稳定映射到其 32 位种子范围。
- `reference_images`：动态添加 0-9 张独立 IMAGE。每个插槽只接受一张图片；图片角色由数量和用户提示词共同决定。
- `reference_videos`：动态添加 0-3 路 IMAGE 批次，可连接 VHS `Load Video` 的 IMAGE 输出，内部固定按 24 fps 计算时长和采样时间。
- `video_sample_frames_per_sec`：每路参考视频每秒抽取的帧数，默认值和 RTX 3090 推荐值为 2。
- `force unload model`：默认开启。开启时每次完成后终止内置 llama.cpp 并强制清理内存和显存；关闭时保留模型供下一次运行复用。发生错误时无论选项状态都会强制释放。

### 图片输入与模式路由

节点按照以下规则选择 H3 模式：

| 输入 | 判定方式 |
| --- | --- |
| 没有图片和视频 | 直接使用 T2VA |
| 1 张图片、没有视频 | Qwen 根据用户提示词选择 I2VA、L2VA 或 Ref2VA |
| 2 张图片、没有视频 | Qwen 根据用户提示词选择 FL2VA 或 Ref2VA |
| 3-9 张图片 | 直接使用 Ref2VA |
| 连接任意参考视频 | 直接使用 Ref2VA |

一到两张图片时，节点不会仅根据图片数量猜测首尾帧。要生成关键帧模式提示词，必须在用户提示词中明确说明图片的时间角色：

```text
# 一张首帧图，路由为 I2VA
将 Picture 1 作为目标视频的首帧，从这个画面继续向前发展。

# 一张尾帧图，路由为 L2VA
将 Picture 1 作为目标视频最后一帧，设计一个自然抵达该画面的过程。

# 两张首尾帧图，路由为 FL2VA
Picture 1 是目标视频首帧，Picture 2 是目标视频尾帧，生成从前者连续过渡到后者的过程。
```

如果一到两张图片只是人物、产品、场景、构图或风格参考，也应明确说明其参考作用，节点会选择 Ref2VA：

```text
参考 Picture 1 的人物身份和服装，参考 Picture 2 的场景与灯光，创作一段产品宣传短片；两张图都不是首尾帧。
```

`reference_images` 的编号按照动态插槽顺序对应 `<Picture 1>` 至 `<Picture 9>`。不要将包含多张图片的 IMAGE batch 接入单个图片插槽；需要多张图片时应分别增加插槽。旧版的 `first_frame` 和 `last_frame` 已合并为统一的 `reference_images` 输入，旧工作流需要删除并重新添加节点后接线。

### 参考视频

`reference_videos` 最多连接 3 路，每一路都是按时间顺序排列的 IMAGE batch。单段视频按 24 fps 计算后必须为 2-15 秒，全部参考视频总时长不能超过 15 秒。

`video_sample_frames_per_sec` 表示从每路视频的每一个一秒时间窗中均匀抽取多少帧，它不是整段视频的总抽帧数，也不会改变最终 H3 视频的帧率。例如：

- 10 秒视频设置为 `3`：共抽取约 30 帧，每秒 3 帧。
- 10 秒视频设置为 `1`：共抽取约 10 帧，每秒 1 帧。
- 连接多路视频时，对每一路分别按相同的每秒采样率处理。

为避免大量独立图片占满 Qwen 的上下文，同一秒内抽出的帧会按时间顺序合成一张联系表，再作为一个视觉输入发送。上面的 10 秒、每秒 3 帧示例会抽取 30 帧并组成 10 张逐秒联系表，而不是发送 30 个独立视觉消息。联系表中的画面按从左到右、从上到下的顺序排列，并附带各帧时间戳。

默认值 `2` 适合大多数动作和运镜。静态产品展示可以使用 `1`；快速动作、复杂转场可提高到 `3-8`。数值越高，每秒保留的动作细节越多，但联系表中单帧面积会变小，同时增加图片编码、内存和推理开销。

连接参考视频后始终使用 Ref2VA。当前接口只接收视频画面帧，不会读取视频文件中的音轨。

### 原创声音与配乐描述

Qwen 会根据用户提示词、目标时长、参考图片以及参考视频的抽样画面，自动创作与画面匹配的声音设计：

- `overall_soundscape` 描述环境声、动作声、空间感和随画面变化的声音事件。
- `non_diegetic_music` 描述原创器乐配乐的情绪、乐器、速度、发展、关键画面同步点和收束方式。
- 没有用户明确提供时，不会自行编造对白、旁白、歌词或人声演唱。
- 用户只要求“无背景音乐”时，输出 `non_diegetic_music: N/A`，但仍保留合理的环境声和动作声。
- 用户明确要求“完全静音”“全程静音”或“不要任何声音”时，强制输出 `overall_soundscape: N/A` 和 `non_diegetic_music: N/A`。

这属于根据视觉内容创作新的音频描述，不是分析、复刻或保留参考视频原声。生成的文本最终交给 MiniMax H3 合成原生声音。

节点输出 `h3_prompt`、`selected_skill` 和 `detected_mode` 三个字符串。可将 `h3_prompt` 连接到 ComfyUI 的 `PreviewAny` 查看结果，也可直接连接 MiniMax H3 Conditioning 的 `prompt` 输入。

## 推荐参数

`think_mode` 是布尔开关，默认为关闭。节点不再显示 `temperature`、`top_p`、`top_k`、`min_p`、`presence_penalty` 和 `repetition_penalty`；它会根据开关状态自动使用 Qwen3.8-27B 官方推荐值：

| 模式 | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| instruct | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |

关闭 `think_mode` 时使用 instruct 模式；开启时使用 thinking 模式，并将 `reasoning_effort` 发送给 llama.cpp。默认使用 `think_mode=false`、`reasoning_effort=medium`、`max_tokens=8192`。提示词改写通常不需要长思维链，复杂长片策划可以开启思考模式。自动 H3 模式路由与 Skill 路由始终关闭思考并使用确定性参数，不受这个开关影响。

本次输入结构调整删除了六个采样插槽，并将旧的 `thinking_mode` 和 `force_release_memory` 分别改名为 `think_mode` 和 `force_unload_model`。旧工作流应删除并重新添加本节点后接线。

## 显存和生命周期

节点执行前会请求 ComfyUI 卸载已加载模型，然后根据 `runtime_config.json` 从 `runtime/` 目录启动当前平台的独立本地 llama.cpp 服务。Windows、Linux 和 macOS 都必须先运行 `install_runtime.py`。默认在提示词生成结束后终止服务，执行 Python 垃圾回收并清理 GPU 缓存；发生错误时也执行相同的强制释放。同一时间只允许一个本节点推理，防止重复加载导致显存溢出。

若关闭 `force unload model`，内置服务会继续占用约 20 GB 显存，模型配置不变时下一次可直接复用；再次开启并运行节点、模型配置发生变化或退出 ComfyUI 时会终止驻留服务。3090 上如需在后续节点加载 H3 视频模型，应保持此选项开启。

控制台会输出多模态输入准备、ComfyUI 模型清理、Qwen 模型加载、Skill 路由、推理、格式校验和资源释放等关键阶段及耗时。最终成功信息使用绿色字符，失败信息使用红色字符。

## 官方 Skill 的本地适配边界

`h3-prompt-writing` 官方声明可直接移植。其余八个 Skill 原本依赖 MiniMax Hub 的画布、选择卡和媒体生成工具。本节点保留官方 Skill 内容作为创意与领域指导，但统一跳过素材收集、选择卡、审批门槛和媒体生成步骤，只输出符合当前输入模式与时长的最终 H3 prompt。

节点不会输出澄清问题、确认请求、备选方案、pre-production package 或下一步建议。用户未提供的可选内容会直接省略，例如画面文案、口号、品牌声明、对白、旁白和歌词；生成视频不可缺少但未指定的纯创意细节，则根据请求与参考媒体采用一个保守的确定值。九个 Skill 的结果都会经过同一套格式和交互语句检查，不合格时自动修复一次。

## 上游来源与许可证

- 本项目采用 MIT 许可证，详见根目录 `LICENSE`。
- 九个 MiniMax H3 Skill 来自 [`MiniMax-AI/MiniMax-H3`](https://github.com/MiniMax-AI/MiniMax-H3) commit `d21241f0a4b3acbb34c97dae47fa417b7065e438`，官方文件保存在 `skills/`。
- 跨平台安装器固定使用 llama.cpp `b10436`，对应完整 commit `6fed9f6ff7a603b124cb8c5864fca6ea879f9f99`，并对下载文件执行 SHA256 校验。
- Linux CUDA 源码构建使用同一固定 tag 和 commit。llama.cpp 的 MIT 许可证保存在 `third_party/llama.cpp-LICENSE`。

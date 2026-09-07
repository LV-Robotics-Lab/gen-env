# SimFoundry 视频转 OmniGibson

上游源码以子模块保存在 `external/SimFoundry`，当前固定
`NVlabs/SimFoundry@9e34ebefcd020583fbb755a8b57268dce78eca26`。
本目录只管理本机环境和命令入口，重建算法仍由上游提供。
输出是 OmniGibson 场景，不是稳定 `/gen-env` 的 `ResolvedSceneSpec`，
也没有自动转换成 Genesis、RoboTwin 或通过本仓库的物理验收。

## 安装

在仓库根目录运行。GPU 环境独立安装在 `.cache/simfoundry/miniforge`，
不会替换原有 Miniconda 环境。完整上游说明见
[README](../../../external/SimFoundry/README.md)、
[agent 安装指南](../../../external/SimFoundry/docs/AGENT_INSTALL.md)。

```bash
git submodule update --init external/SimFoundry
bash self_improving/sim_adapters/simfoundry/bootstrap.sh
source self_improving/sim_adapters/simfoundry/env.sh
mkdir -p logs/simfoundry
cd external/SimFoundry
nohup bash scripts/installation/install_everything.sh \
  --only "simfoundry hunyuan any6d da3" \
  > ../../logs/simfoundry/install-core.log 2>&1 < /dev/null &
```

已有安装正在运行时不要重复执行。上游总安装器会跳过已存在环境，
所以目录存在不代表安装完成；中断后应检查失败位置并针对性恢复，
不要用 `--fresh` 删除其他环境。CUDA 12.8 可由当前上游脚本安装进环境；
RTX 5090 架构由安装器自动检测。可选背景三环境和 articulation 暂不安装。
Miniforge 固定版本与校验值来自
[官方 release](https://github.com/conda-forge/miniforge/releases/tag/26.5.3-0)。

在下载模型前准备 Hugging Face token，并申请所需仓库的访问权限：
[SAM3](https://huggingface.co/facebook/sam3)、
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)、
[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)。
默认 Hunyuan 路径使用公开 DINOv2；DINOv3 / RMBG-2.0 是其他后端依赖。
默认分割需要 SAM3。token 存在不代表已获模型授权。本机通过 `HF_TOKEN_PATH` 指向权限 600 的
`.cache/simfoundry/hf_token`，保留原全局 token。新机器请把获授权 token 写入此文件
或显式设置 `HF_TOKEN`。`HF_HUB_CACHE` 和 `HF_XET_CACHE` 也位于项目
`.cache/simfoundry/` 下；上游各后端在运行时还可能按需下载模型。

VLM 默认使用 Google Gemini 原生接口。可以在被上游 Git 忽略的
`external/SimFoundry/api_keys.txt` 写入 `GEMINI_API_KEY=...`，文件权限设为 600，
或配置 Vertex AI 的 `GCLOUD_PROJECT` 和 ADC。OpenAI 兼容代理的 key 不能
直接假定适用于该接口；需确认服务协议、地址及图像生成模型支持。

```bash
source self_improving/sim_adapters/simfoundry/env.sh
cd external/SimFoundry
bash scripts/installation/download_checkpoints.sh --default
```

注意：当前上游下载脚本还会下载可选 VOID 背景权重。只复现核心流程时，
可按该脚本列出的官方 URL 获取 FoundationStereo、FoundationPose 两组、
SAM2.1、DepthPro 和 RealESRGAN；待对应 `deps` 仓库已克隆后再写入权重，
提前创建同名空目录会让部分上游安装器误判仓库已克隆。

## 本机代理配置

本机已验证 `https://api2.aigcbest.top/v1/chat/completions` 的文字和图片输入。
该服务也支持 Gemini 原生协议，因此流水线继续使用上游 `google.genai` 客户端。
`run.sh` 仅在启动 pipeline 时读取被忽略的 `.cache/simfoundry/service.env`；
也可用 `SIMFOUNDRY_SERVICE_ENV` 指定另一个本地配置文件。
配置为受信任的 shell 文件，权限设为 600，不应提交到 Git：

```bash
export GOOGLE_GEMINI_BASE_URL='https://api2.aigcbest.top'
export GEMINI_API_KEY='<该平台的密钥>'
export SIMFOUNDRY_GEMINI_BACKEND='api_key'
export SIMFOUNDRY_GEMINI_NONSTREAM_TEXT=1
```

这里的原生 API base URL 不带 `/v1`；完整 chat/completions 地址属于另一种协议。
该代理的部分流式文字响应含非标准 `thoughtSignature`，会被 Google SDK 拒绝。
启用最后一项时，启动器将本目录 `runtime/` 加入子进程 `PYTHONPATH`，
把文字响应改为 SDK 非流式调用，再作为一条完整响应交给上游。
图片生成仍使用原始流式接口；finish reason、安全拦截和异常保留。
这是显式启用的平台进程适配，没有改写上游或已安装 SDK 文件。

当前用户已授权本次公开 Fruits 示例帧及中间图片向该服务传输。
运行时会将相应图片发送给此代理。模型缓存请使用独立的
`--model-cache-dir "$PWD/.cache/simfoundry/model_calls/aigcbest_fruits"`，
避免与其他服务的同名模型缓存混用。

## 单张图片输入

固定上游的 `1b_process_raw_video.py` 已实现 `process_single_image()`。
仍用 `--video-fpath /absolute/path/scene.png` 传图片路径，在 `--` 后追加
`s1_video.single_image_input=true s1_video.n_subsampled_frames=1 s3_ground.img_idx=0`，
其余模型覆盖项沿用下文。上游会保留唯一图片并生成兼容目录所需的单帧 MP4；
这不提供新的视角或运动信息。当前只核实了源码入口，本机单图端到端尚未验证。

流程先估计深度和支撑平面，再从选定参考图分解物体、补全图片、生成网格，
对齐尺度和位姿，估计质量/摩擦并生成碰撞几何，最终导出 OmniGibson 场景。
隐藏形状与物理参数包含模型推断，不能当作真实测量；单图没有视频的多视角约束。

## Agent 调用

从仓库根目录可通过 shell 工具调用；这是 CLI 集成，尚不是 MCP server。
`doctor` 仅输出环境存在性、模块路径和凭据存在性，不启动 Isaac Sim，
不验证服务权限，也不宣布端到端就绪。

```bash
bash self_improving/sim_adapters/simfoundry/run.sh doctor
bash self_improving/sim_adapters/simfoundry/run.sh reconstruct --help

# 先查看上游执行计划。正常运行时移除 --dry-run。
bash self_improving/sim_adapters/simfoundry/run.sh reconstruct \
  --scene-name fruits_trial_001 \
  --video-fpath "$PWD/external/SimFoundry/docs/assets/example_videos/Fruits.mp4" \
  --root-dir "$PWD/data/simfoundry" \
  --dry-run -- s7_mesh.low_vram=true

# 在真实重建成功后，单独执行 OmniGibson 随机动作 smoke test。
bash self_improving/sim_adapters/simfoundry/run.sh smoke \
  --scene-name fruits_trial_001 --root-dir "$PWD/data/simfoundry" \
  --mode smoke-random
```

本机这次复验将文字模型设为 `gemini-2.5-flash`，图片模型保留上游默认值。
若要沿用同一组模型，在 `--` 后追加以下覆盖项：

```text
s3_ground.detection_model=gemini-2.5-flash
s5_scene.detection_model=gemini-2.5-flash
s8_pose.front_pick_model=gemini-2.5-flash
s11_sim.vlm_model=gemini-2.5-flash
```

这些是同一条 shell 命令的四个参数，分行展示仅为便于阅读。
本次既有深度输入还使用了 `s3_ground.frame_selection.mode=heuristic`。

本机 Fruits 完整重建已完成，产物位于 `data/simfoundry/fruits_da3_20260907/`。
以下是一条可直接交给 agent 执行的重建命令（新 scene name）：

```bash
bash self_improving/sim_adapters/simfoundry/run.sh reconstruct \
  --scene-name fruits_agent_001 \
  --video-fpath "$PWD/external/SimFoundry/docs/assets/example_videos/Fruits.mp4" \
  --root-dir "$PWD/data/simfoundry" --no-stream --cache-mode \
  --model-cache-dir "$PWD/.cache/simfoundry/model_calls/aigcbest_fruits" -- \
  s3_ground.frame_selection.mode=heuristic \
  s3_ground.detection_model=gemini-2.5-flash \
  s5_scene.detection_model=gemini-2.5-flash \
  s8_pose.front_pick_model=gemini-2.5-flash \
  s11_sim.vlm_model=gemini-2.5-flash s7_mesh.low_vram=true
```

每次新输入使用新的 `scene-name`，保留全部阶段输出。
`--skip-successful` 只代表上游曾完成该阶段；修改视频或上游阶段后不要
把旧成功标记当成缓存有效性证明。所有参数原样转交上游，包括
`--include`、`--exclude`、`--max-vram-gb` 和 `--` 后的 Hydra overrides。
示例默认低显存模式以给当前桌面及其他进程留下余量。

Agent 应读取退出码、阶段 `stage_info.json`、`pipeline_run_report.json`，
并检查本次输出的 `s14_og/reconstructed_og_scene.json`。
单独完成抽帧或生成执行计划不能报告为重建成功；上游 smoke 成功也不能
替代稳定核心要求的接触、支撑、包含和哈希绑定物理证据。

`env.sh` 设置 `HY3DGEN_MODELS` 到项目缓存；本机 Hunyuan 目录通过符号链接
指向已下载 snapshot。复制安装时应保留该链接和目标缓存，或让上游按需重新下载。

当前安装与复现记录见 [REPRODUCTION.md](REPRODUCTION.md)。

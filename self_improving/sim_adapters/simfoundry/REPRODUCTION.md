# 本机复现记录

记录时间：2026-09-07（Asia/Singapore）。本页区分已运行证据与待完成步骤。

| 项目 | 状态与证据 |
| --- | --- |
| 上游 | 子模块 `9e34ebefcd020583fbb755a8b57268dce78eca26`，源码保持干净 |
| GPU | RTX 5090 32 GB；驱动 580.173.02；真实 CUDA tensor 运算通过 |
| 环境管理 | 项目内 Miniforge 26.5.3-0 / Mamba 2.5.0 已安装，官方 SHA-256 核验通过 |
| 主环境 | PyTorch `2.7.0+cu128`、OmniGibson 3.8.0（本 checkout 路径）导入通过；机器人资源安装完成，FAISS GPU 搜索通过；Hunyuan 和 Any6D 的安装、原生扩展导入及 CUDA/NumPy 桥接通过；DA3 独立环境安装及 FAISS GPU 搜索通过；四环境安装最终退出码 0 |
| Gemini | 原 Google 官方 key 免费层配额为 0；用户代理的 `gemini-2.5-flash` 文字/视觉调用及 `gemini-3-pro-image` 图片生成已通过，平台非流式文字适配已验证 |
| Hugging Face | 授权后复查：SAM3 与 RMBG-2.0 配置文件 HTTP 200；DINOv3 后续复查也已 HTTP 200；三项模型权重均已下载，SAM3 实际分割通过 |
| 上游轻量测试 | 20 passed、1 skipped；skip 为测试报告 articulation stage 9 不可用 |
| 主仓库回归 | `.venv/bin/python -m pytest -q`：363 passed（含 7 项代理兼容测试） |
| Agent CLI | `doctor`、help、A 全流程 dry-run、C smoke dry-run 已执行 |
| 真实视频输入 | 上游 Fruits.mp4 的 stage 1b 通过；452 帧、452 互异帧、15 张采样图，耗时约 9.53 秒 |
| SAM3 分割 | 授权后离线加载及真实首帧推理通过；手动文本提示 `fruit` 得到 4 个非空掩码，约 7.94 秒，峰值 GPU allocated 4.97 GiB |
| Hunyuan 模型 | 离线加载完整形状和纹理模型通过，约 39.14 秒，峰值 GPU allocated 6.86 GiB；后续 stage 7 已执行网格生成 |
| 真实深度 | DA3 处理 15 帧，通过正深度与有限值检查，输出深度/置信度和相机内外参；主环境约 16.09 秒，默认独立 DA3 环境约 15.61 秒 |
| 地面与世界坐标系 | stage 3 / 4 通过；显式使用本地 heuristic 选帧，选中采样图索引 11；约 14.39 / 3.59 秒，4×4 相机变换为有限值 |
| 物体分解与图片 | stage 5 / 6 均 success；共 7 个物体（包括原图盘下黑色边框）；图片有效性复查使用上游默认 check_valid=false |
| 网格导出 | stage 7 success；7 个带纹理 GLB，顶点均有限、几何非空，共 279,999 个三角面；见 mesh_evidence.json |
| 完整视频→场景 | stage 1b–14（不含可选 stage 9）均 success；后半段 5–14 耗时 57 分 56.42 秒，退出码 0；最终场景含 7 个重建物体 |
| 原生运行时 smoke | 无模型依赖的动态立方体场景真实运行 120 步，Z 从 0.78365 m 降到 0.05001 m，约 178.97 秒；只验证原生运行时 |
| 重建场景的 C smoke | 原生 OmniGibson 随机动作 120 步通过，退出码 0，约 55.91 秒；视频 121 帧、121 互异帧、15 fps，3280×720 |

深度初次记录见 [depth_evidence.json](depth_evidence.json)。当时为提早验证，显式使用
`--env-da3 simfoundry`（主环境中上游已安装的 DA3 runtime），使用已下载的
`depth-anything/DA3NESTED-GIANT-LARGE-1.1@b2359bdf726fb44ef62acca04d629dcf158053e7`
并设置 `HF_HUB_OFFLINE=1`。四环境安装结束后，使用无环境覆盖的默认
`da3` 路径再次离线执行 stage 1b、2，退出码 0，总耗时约 23.74 秒；
深度、置信度、相机内外参均为有限值，深度全部为正，
见 [独立 DA3 证据](depth_dedicated_evidence.json)。
新产物位于 `data/simfoundry/fruits_da3_20260907/`，原主环境验证目录保留。

原生运行时记录见 [runtime_evidence.json](runtime_evidence.json)，不等价于重建视频场景通过。
抽帧详细摘要和输入/输出哈希见 [preprocessing_evidence.json](preprocessing_evidence.json)。
真实产物保存在被忽略的 `data/simfoundry/fruits_preprocess_20260907/`。
只完成 stage 1b 的 `stage_info.json.success=true` 不表示完整 pipeline 成功。

首次安装日志：`logs/simfoundry/install-core.log`，退出码 1。
恢复安装日志：`logs/simfoundry/install-recovery.log`，最终退出码 0；
四环境均已完成，上游 FAISS GPU 校验通过。
公开权重下载日志：`logs/simfoundry/public-checkpoints.log`。
服务检查仅保留 HTTP 状态与模型名称，不保存 key 到报告或日志。

四环境与默认公开模型已安装完成。以下为代理接入前的服务诊断历史；
代理现已验证，Fruits 完整重建已完成。其他新输入使用新的 scene name
运行 A 全流程，读取 `s14_og/reconstructed_og_scene.json`，最后运行 C smoke。

- SAM3 下载权限已通过复查，RMBG-2.0 也已开放。
  [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 仍返回
  `GatedRepo`，但它不阻塞当前默认 Hunyuan 路径。
- 为当前 Gemini key 所属项目提供默认 `gemini-3.1-pro-preview` 的可用生成配额；
  当前请求返回 429，相关免费层额度为 0。models 接口 200 只证明列表可访问，
  不代表检测或图像生成可用。额度恢复后仍需验证实际视频重建调用。

不要把已有的抽帧示例或 dry-run 目录误当成已重建场景。

## 安装恢复

首轮安装在 `install_simfoundry.sh` 的机器人资源导入处退出 1：
`charset_normalizer.cd` 报 `charset_normalizer.md` 缺少 `CharInfo`。
这是混合安装残留的扩展文件；单纯 force-reinstall 仍失败。
本次把新建环境中的 `charset_normalizer/` 和其 dist-info 移到
`.cache/simfoundry/quarantine/charset-normalizer-20260907/` 保留，再通过 pip
安装 Isaac Sim 所用 `charset-normalizer==3.3.2`，requests 和 OmniGibson 导入恢复。
没有删除 conda 环境或为修复此故障改写上游 `deps` 源码。

随后从已检视的原安装脚本 `copy_robot_asset_from_fallback()` 定义开始，
原样执行其机器人资源、校验与 FAISS 尾段；本机恢复脚本为
`.cache/simfoundry/resume-main-assets.sh`，余下三环境仍调用原上游总安装器。
恢复日志及最终退出码分别是 `logs/simfoundry/install-recovery.log` 和
`logs/simfoundry/install-recovery.exit`，原失败日志保留。
`doctor` 在恢复日志存在时展示恢复尝试的日志和退出码。

六项已下载公开权重共 6,426,548,219 字节，SHA-256 本地清单保存在
`logs/simfoundry/public-checkpoints-manifest.json`。默认 DA3、Hunyuan 和 DINOv2 模型也已下载到 `.cache/simfoundry/hub/`：

| 模型 | revision | 下载字节数 |
| --- | --- | --- |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | `b2359bdf726fb44ef62acca04d629dcf158053e7` | 6,759,561,213 |
| `tencent/Hunyuan3D-2.1` | `0b94677654c57bb9a6b6845cd7b704ccf551d327` | 14,909,628,047 |
| `facebook/dinov2-giant` | `611a9d42f2335e0f921f1e313ad3c1b7178d206d` | 4,546,006,416 |

下载回执在 `logs/simfoundry/default-models.json` 与 `dinov2-model.json`。
`HY3DGEN_MODELS/tencent/Hunyuan3D-2.1` 指向对应已固定的本地 snapshot；
模型权重不进入 Git。Hunyuan 初始化验证复用上游 stage 7 的
`torchvision_fix.apply_fix()`，同时加载形状和纹理模型，结果见
[hunyuan_model_evidence.json](hunyuan_model_evidence.json)。默认路径使用公开
DINOv2；上述 DINOv3 / RMBG-2.0 属于 README 列出的其他后端依赖，
三项 HF 权限均已在后续复查通过；原生 Google 配额问题通过用户提供的代理接入另行解决。

复验入口（从仓库根目录运行）：

```bash
HF_HUB_OFFLINE=1 bash self_improving/sim_adapters/simfoundry/run.sh reconstruct \
  --scene-name fruits_depth_new \
  --video-fpath "$PWD/external/SimFoundry/docs/assets/example_videos/Fruits.mp4" \
  --root-dir "$PWD/data/simfoundry" --include 1b,2
```

该命令只执行抽帧和深度估计；选择新的 scene name 保留旧产物。

主环境还有上游安装器的依赖声明冲突（见
`logs/simfoundry/pip-check-before-repair.log`）；不能把已通过的特定 smoke 路径
表述成全部可选依赖一致。当前实际导入检查通过 google.genai、上游 Gemini 的
`api_key` 路由、rembg 和 SAM3 builder。可选 FLUX 的 Flash-Attention 版本警告
符合上游已知限制，本次默认 Hunyuan/Gemini 路径没有启用 FLUX。

## 模型授权后的复验

授权后 SAM3 和 RMBG-2.0 下载成功，分别为 3,450,121,474 与
884,980,745 字节。版本和逐文件 SHA-256 见
[授权模型回执](authorized_model_evidence.json)。
SAM3 的真实 CUDA 推理通过，输入是 Fruits 首帧，手动提示词为 `fruit`，
见 [分割证据](sam3_evidence.json)。该独立测试不等价于上游 stage 5 的
自动物体分解；RMBG-2.0 本次只下载，没有加载其远程代码或执行推理。

服务复查见 `logs/simfoundry/services-recheck.json`：DINOv3 仍为
403 / `GatedRepo`，Gemini 默认模型仍为 429，免费层配额为 0。

随后在同一 Fruits 输入的既有深度产物上运行 stage 3、4，退出码 0；
显式设置 `s3_ground.frame_selection.mode=heuristic`，没有改变几何阈值。
地面分割和世界坐标系产物的哈希见
[地面与坐标系证据](ground_frame_evidence.json)。这不是完整仿真场景。

按用户指定格式另建 `.cache/simfoundry/gemini-legacy/` 测试环境，安装
`google-generativeai==0.8.6`，实际调用
`genai.configure(api_key=...)` → `GenerativeModel("gemini-2.5-flash")` →
`generate_content(...)`。新候选密钥在 Google 官方接口返回
HTTP 400 / `API_KEY_INVALID`；测试结果见
`logs/simfoundry/gemini-legacy-probe.json`。该候选密钥未用于替换原配置；
后续用户提供代理地址，现已完成适配，见下节。

## 代理接入与继续重建

DINOv3 权重已下载：`ea8dc2863c51be0a264bab82070e3e8836b02d51`，
1,212,583,161 字节；详见 `logs/simfoundry/dinov3-model.json`。
代理 chat 文本、图片输入及原生 Gemini 协议均返回 200。
原生流式文字曾因非标准 `thoughtSignature` 被 SDK 拒绝；平台采用非流式
文字兼容方式，保留完整 SDK 响应，实际 `Gemini` 封装复验通过，约 6.33 秒。
原始流式图片生成成功得到 1376×768 图片，约 36.97 秒。
证据分别在 `logs/simfoundry/aigcbest-text.json`、
`aigcbest-vision-probe.json`、`proxy-upstream-probe.json`、`proxy-adapted-probe.json`。

新增兼容测试覆盖完整响应/截断/安全状态保留、图片流式路由及错误传播，
主仓库回归为 363 passed。服务配置和进程适配说明见 [调用指南](README.md)。
实际继续命令保存在 `.cache/simfoundry/run-fruits-full.sh`，使用顺序执行、
`s7_mesh.low_vram=true`，并将 ground / scene detection、front pick、sim VLM
显式设为 `gemini-2.5-flash`；图片模型保留 `gemini-3-pro-image`。
这组配置区别于 README 默认的 `gemini-3.1-pro-preview` 文字模型。

## 完整 Fruits 重建与回放

A 的 stage 1b–14（不含可选 articulation stage 9）均记录 success。
后半段 5–14 顺序执行耗时 57 分 56.42 秒，最终退出码 0；生成 7 个物体的
带纹理网格、位姿、碰撞网格、URDF、USD 与 OmniGibson 场景。
[完整证据](reconstruction_evidence.json) 绑定源视频、各阶段记录、最终场景和回放视频 SHA-256。

当前真实产物：

- 场景：`data/simfoundry/fruits_da3_20260907/s14_og/reconstructed_og_scene.json`
- 预览：`data/simfoundry/fruits_da3_20260907/s14_og/reconstructed_scene.png`
- 回放：`data/simfoundry/fruits_da3_20260907/application_smoke/random_action_smoke.mp4`
- 导入资产：`external/SimFoundry/deps/BEHAVIOR-1K/datasets/real2sim-assets/`

产物从旧 `output/simfoundry/` 移至 `data/simfoundry/` 后，最终场景字节未变，
已从新目录成功加载并运行 C smoke。首次使用旧路径的 C 失败日志保留在
`logs/simfoundry/fruits-application-smoke.log`；成功日志为
`logs/simfoundry/fruits-application-smoke-data.log`。原始 stage_info 保留执行时路径，
本页机读摘要指向当前目录。上游会覆盖顶层 pipeline_run_report.json，当前它记录最近的 C；
A 的结果由各 stage_info、fruits-full.log 与完整证据保留。

C 实际执行 120 次随机动作，动作频率 15 Hz、物理频率 120 Hz；
输出初始观测及每步真实观测，共 121 帧，解码后 121 互异帧。
视频已抽帧查看；该测试验证加载、步进和连续渲染，未测量重建精度或任务成功率。
上游 stage 12 调整会逐步重置速度，不能视为自由动力学稳定性验证，
本次没有执行稳定核心的支撑、接触或包含验收。

复验已有场景（无需再次调用 Gemini）：

```bash
OMNIGIBSON_HEADLESS=1 bash self_improving/sim_adapters/simfoundry/run.sh smoke \
  --scene-name fruits_da3_20260907 --root-dir "$PWD/data/simfoundry" \
  --mode smoke-random -- application_smoke.n_steps=120 application_smoke.video_fps=15
```

该命令会更新此场景的 smoke 视频和顶层运行报告；保留本次证据时先另存产物。

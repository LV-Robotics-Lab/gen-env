# 2_sim_migration — 阶段 2 · 仿真环境迁移

> 让场景能在 **SAPIEN 之外的仿真后端**运行。本阶段的**迁移引擎 = openxsim**（Open-X-Sim）：一套**后端中立 IR**（`EnvironmentPackage`）+ 多后端编译 + 跨模拟器一致性测试，代码在 `../shared/openxsim/`（2026-08-03 自本目录移入 `shared/`，因阶段 1 也消费它）并跑通自带测试。**与 env-gen 的桥接已实现**：env_gen importer 落 `../shared/openxsim/source/agenticsim/agenticsim/openxsim/env_gen.py`（`import_env_gen` + `import_environment` 自动分派，`resolved_scene` → OpenXSim IR）。Genesis 首版也必须经过这层 IR，再由 `GenesisCompiler` 生成固定姿态、无物理步进的多视角渲染；它不是绕过 OpenXSim 的单独 loader。

## 1. 这是什么 / 目标
- **输入**：文本 / 已有仿真环境 / 资产 →（编译进）后端中立 IR。
- **输出**：多个仿真后端的可运行环境 + L0-L4 一致性评估。
- **目标**：一份中立图纸、多后端加载；env-gen 的场景经此迁到 Genesis、Isaac 等后端。

## 2. openxsim 提供的 5 个工作流（`../shared/openxsim/scripts/openxsim.py`）
| 子命令 | 作用 |
|---|---|
| `text2env` | 纯文本编译进 IR |
| `anchor2env` | 文本 + 图像/视频证据编译 |
| `asset-scout` | 搜索/下载/转换/注册公开资产 |
| **`transfer`** | **导入一个已有环境，编译到另一个后端**（迁移核心） |
| `robotwin-evidence` | 校验 RoboTwin rollout 与其包/任务程序 |

## 3. 里面有什么
| 路径 | 作用 |
|---|---|
| `../shared/openxsim/source/agenticsim/agenticsim/openxsim/` | 引擎源码：`ir.py`(中立 IR) `backends.py`(多后端编译) `robotwin.py`(RoboTwin 适配) `conformance.py`(L0-L4 一致性) `importers.py` `text2env.py` `pipeline.py` `assets.py` |
| `../shared/openxsim/scripts/openxsim.py` | CLI 入口（自定位 `source/agenticsim`） |
| `../shared/openxsim/configs/` `../shared/openxsim/tests/` | 配置 / 自包含回归测试 |
| `../shared/openxsim/third_party/MetaSim/` | vendored 多后端仿真框架（含 IsaacSim 工具）。**不入 git**，见 `../shared/openxsim/UPSTREAM.md` |
| `../shared/openxsim/deps/metasim_core/` | vendored 依赖。**不入 git** |
| `lib/` | 空：桥接 adapter 未落于此，已直接落 openxsim 包内的 `env_gen.py`（见上） |

Genesis 的外部运行依赖不是 vendor copy：官方仓库固定在
`../../../../external/genesis-world` 子模块，commit
`0e74bf392781884ccad765c3f344419c86b872ca`。核心 `scene_gen/` 不导入
Genesis；编译器与 runner 只属于本平台适配层。

## 4. 怎么用
```bash
conda activate env-gen-yuxin
OX=/home/jingxiang/yuxin/env-gen-dev/shared/openxsim
export PYTHONPATH="$OX/source/agenticsim:$OX/deps/metasim_core:$OX/third_party/MetaSim:$PYTHONPATH"
python $OX/scripts/openxsim.py --help              # 5 个工作流
python -m pytest $OX/tests -q                       # OpenXSim 自包含回归
```

### env-gen → OpenXSim → Genesis Rasterizer

先用 OpenXSim `transfer` 编译，不直接把 `resolved_scene.json` 交给
Genesis：

```bash
python self_improving/asset_pipeline/active/shared/openxsim/scripts/openxsim.py \
  --output data/openxsim \
  transfer \
  --source data/generated_scenes/<scene-id>/resolved_scene.json \
  --source-backend env_gen \
  --backends genesis \
  --strict
```

`transfer` 会先把 IR 的 `target_backends` 规范成这次请求的 `genesis`，所以
后续 `scene.json`、compile manifest 与渲染证据使用同一个 package digest。

输出 JSON 中的 `compile_results.genesis.runtime_command` 是第二步的唯一
权威命令；把其中的 argv 按原顺序执行，不根据目录名另猜 runner 路径。
runner 的公共参数为 `--scene`、`--output-dir`、`--width`、`--height`、
`--frames`、`--fps`、`--compute-backend`（可选 `cpu`、`gpu` 或 `cuda`），
默认 CPU compute、640×480、120 帧、12 FPS。编译目录同时保留声明式
`scene.json`、薄启动脚本、canonical `environment_package.json` 与
`compile_manifest.json`。

成功的渲染目录固定包含：

```text
preview_head.png
preview_world_left.png
preview_world_right.png
preview_segmentation.png
observer_start.png
observer_mid.png
observer_end.png
observer_runtime.mp4
genesis_render_evidence.json
render_manifest.json
```

三张 preview 是 front-high、world-left、world-right；视频让相机绕场景
一周而不移动物体。`genesis_render_evidence.json` 的 schema 是
`agenticsim.genesis_render_evidence.v1`，以 OpenXSim package digest 绑定
Genesis commit/version、renderer、相机、对象加载与 resolved/final 姿态、
关节 requested/applied qpos、可见像素、视频总帧/互异帧和产物哈希。package
digest 同时包含所选主资产与递归发现的本地 sidecar 指纹；runner 在导入
Genesis 前验证这些字节，并从 canonical package 确定性重编译、要求
`scene.json` 精确一致。`render_manifest.json` 使用
`agenticsim.genesis_render_manifest.v1`，列出实际产物大小与 SHA-256，并
明确省略自己的 self-hash。MP4 会在编码完成后重新解码；帧数必须严格等于
`frames`，互异解码帧少于 `min(frames, 30)`、任一对象三视角均不可见，或
applied qpos 与 requested qpos 相差超过 `1e-6` 都会失败。runner 返回非零、
写失败 evidence，并删除已知的陈旧/部分产物和成功 manifest。

这条首版通道只做 **Genesis Rasterizer（Pyrender）渲染**：对象固定在
resolved pose，不调用 `scene.step()`，也不渲染机器人。它只能为
OpenXSim conformance 提供 L0/L1；L2-L4 必须保持 `not_evaluated`。这些
图片、分割和视频不能叫 `runtime_evidence.json`，也不能替代 RoboTwin /
SAPIEN 对 contact、support、containment、settling 与 articulation 的物理
验收。

## 5. 关键概念 / 术语
- **openxsim / Open-X-Sim**（跨仿真器迁移引擎；核心是后端中立 IR + 多后端编译 + 一致性测试）。
- **EnvironmentPackage（IR）**（typed、simulator-neutral 的中间表示；把任务语义与后端资产格式 USD/MJCF/URDF/SAPIEN 分离）。
- **MetaSim**（vendored 多后端仿真框架，openxsim 靠它对接 Isaac 等；见 `../shared/openxsim/UPSTREAM.md`）。
- **conformance L0-L4**（跨模拟器一致性的分级评估）。

## 6. 坑与注意
- 跑 openxsim 需把 `source/agenticsim`、`deps/metasim_core`、`third_party/MetaSim` 加进 `PYTHONPATH`（见上）。
- `third_party/` 与 `deps/` **不入 git**（本地已存在、可跑）；**fresh clone 需按 `../shared/openxsim/UPSTREAM.md` 重新填充**才能跑。
- **openxsim 已能读 env-gen**：`import_env_gen`（`env_gen.py`）解析 env-gen 的 `resolved_scene.json` 并编译进 openxsim IR；Isaac 后端对缺 USD 表示的网格资产会如实报 blocker（不静默降级）。
- **Genesis 资产不会静默降级**：首版只接受 OBJ、STL、DAE、GLB/GLTF
  和 URDF；缺文件、PLY/USD、仅有 collision mesh 而没有可渲染 mesh、
  各向异性 URDF scale 会成为 strict compile blocker；缺 joint、非单自由度
  或歧义 q index 则在 runner build 后明确失败。rigid asset 优先用
  `/visual/` mesh，普通 fixture mesh 视作 visual-and-collision；GLB/GLTF
  从 Y-up 转为 Z-up，各向异性 scale 会同步交换 Y/Z。OBJ/MTL/纹理、
  GLTF/GLB buffer/image、DAE image 与 URDF mesh/texture 的本地依赖会递归
  纳入 package digest；缺失、损坏或运行前被替换都会成为硬失败。

## 7. 来历 / 依赖

- **引擎来历**：从 `/home/jingxiang/workspace/openxsim-validation` 搬入（2026-08-02），详见 `../shared/openxsim/UPSTREAM.md`。
- **只读依赖**：`../external/env-gen-github`（上游 env-gen；桥接时作源，引用不改）。

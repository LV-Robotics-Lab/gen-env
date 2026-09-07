# Self-Improving 平台边界

`scene_gen/` 是稳定信任边界，负责把受限文本编译成可验证、可回放、哈希绑定的场景包。`self_improving/` 是消费者和编排层：它可以选择环境、组织采集与训练、评估失败、写诊断和记忆、决定是否晋升，也可以调用资产与仿真适配器；它不能伪造或跳过 `/gen-env` 的物理门控。

| 层 | 目录 | 写什么 | 不写什么 |
| --- | --- | --- | --- |
| 稳定核心 | `scene_gen/` | schema、parser、grounding、solver、builder、validator | 策略、训练循环、仿真器特定编排 |
| Harness 契约 | `self_improving/harness/` | 严格审计记录、Text2Env Skill 输入输出、权威载荷引用、公开 schema 快照 | 复制 `scene_gen` 载荷、Registry 业务分支、MCP 自有类型或发布决策 |
| 场景编排 | `self_improving/stage5/` | designer/critic/grounding agent、prompt、MCP-lite | 核心物理判定的替代实现 |
| 闭环 | `self_improving/alchedata/` | collect/train/evaluate/diagnose/transfer、失败记忆、promotion gate | 大规模 runs、checkpoint、下载缓存 |
| 资产 | `self_improving/asset_pipeline/` | 发现、ingest、ledger、catalog 对接、迁移 adapter | 第三方 mesh 与渲染产物 |
| 同学交接材料 | `self_improving/contributor_notes/` | 历史设计、运行说明、任务交接及来源哈希 | 当前运行命令或新的能力承诺 |
| 仿真适配 | `self_improving/sim_adapters/` | 薄脚本、schema、可隔离测试 | 完整复制 IsaacLab 或候选仓库 |
| 历史原型 | `self_improving/legacy/robotwin_text2env_alt/` | `text2env.tabletop.v0` 来源快照、修复工具、有限 smoke evidence | 覆盖当前 Stage 5 或成为新功能入口 |
| 被忽略的工作台 | `self_improving/asset_pipeline/workbench_snapshots/` | Yuxin 的 asset-spike、nightwatch、one-off 源码与笔记快照 | 直接作为当前运行入口 |
| 验收归档 | `self_improving/validation_evidence/`、`workspace_archives/` | 小型结构化证据、复现脚本、完整文件哈希及 Release 指针 | 把 cache、嵌套 Git 元数据或第三方 mesh 直接塞进主树 |
| 呈现层 | `apps/pearl_evidence_portal/` | PEARL 门户、浏览器报告子集、构建测试 | 生成验收结论或把页面文案当运行证据 |
| 外部项目 | `external/` | 钉住子模块 commit | vendor copy |
| 历史 | `self_improving/legacy/` | 只读来源快照 | 新功能 |

`python -m self_improving --json` 只检查这些源码是否到位以及子模块是否初始化，不导入 GPU 框架、不启动仿真器。来源工作区、提交、归档分支和排除项在 `self_improving/source_inventory.json`，它是清理旧副本前的审计入口。

Harness schema tranche 当前公开 14 个以 `$id` 标识的 JSON Schema：六个通用运行/审计记录、Qualification、EnvironmentPackage 和 compile/replay/validate 六个输入输出。`ArtifactRef.schema_version` 指向既有 `robotwin.*` 权威载荷，Harness 不重新定义其内部格式。`python script/export_harness_schemas.py --check` 锁住 committed snapshot；统一测试入口对 `self_improving.harness` 同时强制 100% 语句与分支覆盖。Registry、Text2Env handler 与 MCP adapter 不在这一批实现内，`docs/contracts/HARNESS_MVP_CONTRACT_V1.md` 仍保持 `Status: Proposed`。

字段边界、状态机、快照与未实现范围见 [Harness Schema Tranche](harness-schema-tranche.md)；
逐项实现和验证证据见 [PR1 实现报告](../../docs/contracts/HARNESS_MVP_PR1_IMPLEMENTATION_REPORT.zh-CN.md)。

当前离线、自包含回归基线是 564 passed、6 skipped。skip 仅对应未纳入 Git 的 Isaac/SceneAgent/媒体/报告原始包或本机未安装的 SAPIEN 物理运行时；源码、schema、ledger、fixture、Web Studio 和 OpenXSim IR/adapter 都有仓库内测试覆盖。完整命令见 `self_improving/README.md`。

2026-08-14 的同学工作区收口把 Yeyuxuan 的完整 RoboLab 分支历史与 20 份来源记录、Yuxin 当前 main/Web/未提交断点续测状态，以及 Bingsheng/Gujie/Yuxin 的独有说明归入同仓库。Yuxin 的第三方资产本体没有进入 Git；`asset_pipeline/receipts/asset_library_301_361.sha256` 只记录 12,047 个文件、约 27.64 GB 内容的精确摘要，`storage_uri: null` 表示它仍不是远端备份。

Jingxiang 上原先并列的 Stage04/Stage05/OpenXSim/AgenticSim 验证工作区已收口到单仓库：可审阅的 JSON、日志和运行脚本进入 `validation_evidence/openxsim_20260716/`，六个工作区的 cache-filtered 完整包进入同仓库 `workspace-consolidation-20260813` Release，逐文件 SHA-256 清单在 `workspace_archives/20260716/MANIFEST.sha256`。MetaSim 不再保留第二份 checkout，而是固定为 `external/MetaSim` 子模块 commit `6947e35`。

Genesis 渲染也沿用这条平台边界，而不是给 `scene_gen/` 增加仿真器依赖。读者可以把它理解成四段：`resolved_scene.json` 先由 env-gen importer 转成后端中立的 OpenXSim `EnvironmentPackage`，并把所选主资产与 OBJ/MTL、GLTF/GLB、DAE、URDF 的本地依赖闭包写入 SHA-256/大小；`GenesisCompiler` 再生成声明式 `scene.json` 与可复现启动命令；runner 验证资产字节并从 canonical package 精确重编译 scene；最后由 Genesis Rasterizer（Pyrender）输出三视角图片、实体级分割和环绕视频。Genesis 官方源码固定为 `external/genesis-world` 子模块 commit `0e74bf392781884ccad765c3f344419c86b872ca`，不是拷进平台的 vendor tree。

这条首版 Genesis 通道刻意只有渲染语义：对象保持 `ResolvedSceneSpec` 的确定姿态，视频移动的是相机，runner 不调用物理 `scene.step()`，也不渲染机器人。它生成 `agenticsim.genesis_render_evidence.v1` evidence 和 `agenticsim.genesis_render_manifest.v1` manifest；成功前会重新解码 MP4，严格核对帧数、互异帧、分辨率，并以 `1e-6` 门控 resolved/final pose 与关节 requested/applied qpos。它可证明资产加载、相机参数、对象可见性、渲染帧与产物哈希，却不能证明接触、支撑、包含、稳定或关节动力学。因而 OpenXSim 的 L0/L1 可以通过，L2-L4 被硬锁为 `not_evaluated`，伪造 runtime/policy evidence 也不能提升；RoboTwin/SAPIEN 的哈希绑定物理证据继续是权威门控。

操作上是明确的两步：先运行 `openxsim.py ... transfer --source-backend env_gen --backends genesis --strict`，再原样执行输出 JSON 里的 `compile_results.genesis.runtime_command`。不要绕过第一步直接加载 resolved scene，也不要自行拼接第二步路径；完整命令、runner 参数与十个固定产物名见 `self_improving/asset_pipeline/active/2_sim_migration/README.md`。

需要物理检查时，新增的独立入口是 `self_improving/sim_adapters/genesis/validate_physics.py --compile-manifest ... --output-dir <新目录>`：放在上述编译后、渲染前。它先复用 package/scene/资产完整性检查，再运行真实 CPU 刚体物理；失败保存轨迹与原因且不渲染，通过才复制 package、仅更新动态对象位姿，调用原有 `GenesisCompiler` 重编译终态并执行其渲染命令。物理与渲染成功分别记录，终态环绕视频不是物理过程视频。这是独立的 Genesis 场景验证，不提升 OpenXSim L2–L4，也不改动原有 SAPIEN 门控。

首版测试仅用 Genesis Box 和官方 mug，官方 MJCF 的单刚体、visual 与 32 个 collision 部件由平台辅助脚本转成 URDF，保留变换与哈希记录。支撑目标、保守杯内区域及阈值必须随输入 package 预先声明；未声明的旧渲染包会被拒绝。运行与真实验收命令、已知限制见 [Genesis 物理入口](../../self_improving/sim_adapters/genesis/README.md)。

这里的 Box/单 link URDF 和 compile_manifest 要求属于旧平台物理入口，不是 Genesis 本身的
格式或运行要求。固定官方版本已提供 `gs.morphs.MJCF`、`gs.morphs.Mesh`（含 GLB）、
`scene.step()`、`detect_collision()`、`get_contacts()` 和速度/角速度读取；官方
[资产加载](https://genesis-world.readthedocs.io/en/latest/user_guide/assets/loading_assets.html)、
[碰撞与接触](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/collision_detection.html)
说明了这些接口。它们可以承接新布局的物理运行，但“接触哪一个声明目标、持续多久才通过”
仍由平台定义。官方子模块 `tests/rigid/test_collision_nonconvex.py:test_concave_slanted_wall`
还有 orange_plastic_bowl.glb 的多碗堆叠测试；其姿态、碰撞和判据仅是参考，不能当作当前
四资产布局已物理通过的证据。

已有资产场景现在另有 `validate_asset_scene.py --scene-dir ... --clip-index ...
--fixed-object table_1 --profile baseline`，直接读取已完成 02，原生加载 MJCF/GLB。
它不调用模型、不重求布局，也不要求 OpenXSim 编译包；独立完成 03 后，04 始终 not_run。
只有地面自动固定，其他固定对象必须声明，on 源对象仍须动态。重跑保留 01/02，清理旧
03/04；总状态 physics_passed / physics_failed 与 simulation_executed、实际步数分别记录。

读者检查物理证据时，从冻结的 physics_input 经加载报告、初态和连续 trace 走到
physics_result：原生视觉参考零步进复现 02 摘要，动态加载逐顶点误差 ≤1e-5 m；
初态碰撞检测与步进后的受力分开。基线运行 4 秒，保存 1001 行，终末 0.5 秒的稳定性、
声明支撑、完整投影、2 cm 余量及明确相对关系分别门控，全程穿透上限 1 mm。
这些数值是平台物理验证计划冻结的验收要求，不是 Genesis 官方统一场景标准。
固定官方版本的 `test_many_objects_collision` 对非凸堆积用 100 步速度的 70% 分位数
（线速度 0.25 m/s、角速度 8 rad/s）及终态实测穿透 <5 mm；与本入口末段最大速度和
全程接触穿透的统计口径不同，不能直接移植其容差或据此宣布本场景通过。
多层对象可以同时接触父与子，但自己的向上支撑必须来自声明父对象。终末支撑面倾斜
超过 0.5° 时不猜测通过。GLB 原生非凸路径失败只诊断，不修复、分解或替换原模型。

四资产三次基线与半时间步均未通过；物理失败保留完整轨迹。物理验证视频和失败诊断
统一归属 `03_physics`，只有物理验证通过的结果可以进入 `04_final_render`。
用户请求的四视角终态诊断图保存在 `03_physics/diagnostics/final_state/`，
原速回放（201 帧/4.02 秒）、全部 1001 采样的 5 倍慢放（20.02 秒）以及失败终态
相机环绕（180 帧/6 秒）保存在 `03_physics/video/`，视频分辨率均为 1280×960。
它们使用保存的真实轨迹，不插值、不重新步进；每帧强制刷新 Genesis 可视化缓存，
编码前像素与完整解码后的互异帧分别核验，逐帧来源、视频哈希与上游快照一同归档。
原任务仍 physics_failed，04 为空且为 not_run，原物理证据和媒体字节保持不变。
后续根因诊断保存在 `03_physics/diagnostics/contact_analysis/`：基线复跑哈希一致，
末段三个物体的最大速度均发生在无接触步；碗的单步竖直速度变化约等于 -g×dt，
位置差分也复现峰值。初始左右物体几乎恰留 20 mm，没有运动缓冲。固定其它设置及
impratio=1 后，椭圆摩擦锥与 Signorini 两个对照仍失败，尚未隔离碰撞几何与响应刚度
各自影响。此诊断未改原场景、验收阈值或原结果。
TaskOutput 在物理结果完成、任务封存和读取时检查阶段门控：物理未通过时，04 不得有
产物或已执行状态；重跑 03 会同时清除旧诊断索引。独立物理入口默认仍不创建相机。
独立接触校准和生成 GLB 的动态中间层三层夹具通过；官方微波炉有旋转门关节，首版
仍不支持。视觉三层示例不能引用为物理通过。独立重复副本保留 01/02 字节，原 owner
快照单独归档并按摘要验证。具体命令、参数、误报测试和结果见
[Genesis 原生物理指南](../../self_improving/sim_adapters/genesis/README.md) 与
[真实验收记录](../../self_improving/sim_adapters/genesis/PHYSICS_VALIDATION_EVIDENCE.md)。

官方资产准备另有独立的 `build_official_index.py --output-dir assets/genesis/v1`：
仅获取固定官方 revision 的 mug、cup、apple、donut，保留原始文件，结构检查后先保存逐资产
记录，再直接用 Genesis 原生 MJCF 和零物理步 Rasterizer 生成六视图、拼图与总览。
四资产都保留 32 个碰撞部件，donut 的十个外观部件不能合并漏掉；外观几何与 MJCF 解析结果
以 1 µm 门槛核对。源文件、图片和报告纳入独立索引的文件哈希清单，`--verify-only` 可只读核验。
单资产失败保留结果并继续其余项，整体失败时不生成成功总览。
这是无语义描述的**文件与预览索引**，不是现有 `--asset-catalog` 的语义 catalog。
索引完成、预览通过、检索匹配与物理通过必须分开。

用户另行下载的官方非机器人原文件位于 `assets/genesis/non_robot_v1/sources/`，
revision 为 `5b01555c225977d2d3973710884adc5add269fe8`；下载清单记录 597 个文件、
1.664 GB，以及 251 个机器人相关文件的排除理由。此处文件数不是独立资产数。
`download_plan.json` / `download_manifest.json` / `scope_review.json` 保存范围与证据。
新增 `asset_library.py` / `build_library_previews.py` 从文件库发现 106 个候选入口，
74 个原生六视图预览通过、32 个依赖/材质/可见性失败，分别保留记录；只有通过项进入
`assets/genesis/clip_non_robot_v1/index.json` 的 444×512 向量。旧四资产包继续可用。
新库支持 USD、GLB、URDF、MJCF、OBJ、STL，按来源复用对应索引完整性检查；
源文件、预览和向量分目录保存，未生成假材质或运行物理。

外观选资产现在有独立的 `self_improving/sim_adapters/genesis/clip_select.py`：
`build-index` 复用官方包完整性检查，仅将预览成功资产的六张单视角图片编码成归一化
ChineseCLIP 向量；`select` 编码完整中文单物体描述，每资产取六视角最大余弦分数，
同分按资产 ID 排序，取 Top-K（默认 3、范围 1–5）。模型 revision 固定为
`36e679e65c2a2fead755ae21162091293ad37834`，文本超长直接报错，不截掉后半段要求。
每候选两个最佳视角放入一次 `gpt-4o` Chat Completions 请求，仅给描述、编号和图片，
不透露分数或语义文件名。平台 `vision_request.py` 独立于现有文本解析 transport。
先判断类别再比较外观，近似选择必须注明差异；无合理同类或证据不足允许拒绝，错误不回退。
严格校验候选编号后，才从可信索引取资产 ID、真实模型入口和源文件哈希，保存
`selected_asset.json`。新库绑定额外记录源根目录、模型格式和下载清单引用，不改对象 schema。此绑定不承诺全部描述、几何或物理要求满足，尚未接入求解或 OpenXSim。

每查询最多一次 VLM 请求；有效选择和拒绝按完整查询、K、索引/图片哈希、有效模型配置和
提示词版本缓存，错误不缓存，命中零请求。权重、向量、查询缓存和查询输出分目录保存，
拒绝覆盖已有输出或写入封存官方包。同进程复用模型；查询不启动 Genesis、不渲染或重编码
资产图。报告拆分模型加载、文本编码、相似度、完整性核验、VLM 和总耗时，密钥不进入日志/证据。
本机四条正例均命中 Top-3 并选中对应资产，锤子和米老鼠杯查询拒绝，真实验收 7 passed；
第三方六图接口已验证，但不构成新物理证据。扩展库上原六查询仍为 7 passed，新增芒果、
黄色塑料碗和绿色垃圾桶均正确命中/选择，含缓存验收 4 passed；32 个预览失败不隐藏。
[本次逐步证据](../../data/genesis_history/asset_selection/non_robot_acceptance_v2/README.md) 展示实际文本向量、
444 个视角分数、候选图片、请求体、响应及绑定。完整命令、CPU 耗时和回归入口见
[Genesis 平台指南](../../self_improving/sim_adapters/genesis/README.md)。

当前多物体入口为平台 `extract_assets.py`，`scene_preview.py` CLI/run 转到同一流程。
两阶段提示词提取所有显式独立物体与关系，再逐对象做 CLIP Top-3/单次 VLM 选择。默认
全部选择成功后自动构建初始场景；`--stop-after assets` 显式保留只选资产的行为。
不补桌子，不将把手和印花拆成对象，地面留作 Genesis 内置环境。平台 asset_request 格式
保留完整 description、mentions、attributes、数量与原文关系，不经过核心 SceneSpec。

第二阶段可用 `build_scene.py --scene-dir ... --clip-index ... --llm-config ...` 单独重跑，
默认 `--planner llm`；旧的单层规则排列需显式 `--planner rule`。先从已绑定官方资产测量
原生网格和支撑面，LLM 再原样保留对象/明确关系并提出区域和相对摆放偏好，平台程序求
坐标。模型不能增加或替换资产、输出 pose，不能推断新的 on/inside。原始比例和朝向保持。

`scene_planning.py` 负责模型、最多两次调用和成功缓存；`scene_layout.py` 负责 v2 场景图
契约与有界求解。支持多层 on、左右前后和远近；方向使用 X 右、Y 负向为前，远近使用完整
XY 投影边界距离。支撑必须覆盖完整源投影且有 2 cm 边缘余量，同层间隔 6 cm，近 ≤18 cm、
远 ≥30 cm。每次放置最多 96 个候选，全局 48 次回退。非支撑对象间保守 AABB 检查不能
代替真实碰撞。inside、内腔测量、数字距离和未知关系暂缓；缺少实测支撑面明确失败。

每次模型请求 60 秒超时，格式/语义/求解失败最多反馈修正一次；传输、完整性和不支持能力
错误不重试、不退回规则。模型配置自动沿用提取配置，可显式另传。缓存绑定请求、资产及
几何哈希、提示词、配置、seed 和求解版本，命中后重新校验。证据区分 HTTP 与离线注入，
不把离线响应称作真实模型结果。新图版本 genenv.asset_scene.v2，图和实测几何摘要绑定布局。

`01_obj` 保存提取和资产选择；`02_scene` 保存图、布局、实测几何、支撑面、规划证据、
求解验证报告和三个视角。每个对象至少在一个视角达到可见像素阈值，允许其他视角被堆叠
遮挡。自动链路释放资产锁后重获场景锁，核对交接 manifest；独立重跑只清空 02/03/04。
场景失败保留 objects passed 并标 scene failed，成功为 scene_built；03/04 均 not_run。
这些初始视觉产物不构成物理编译包，尚不能直接送入既有物理入口。

本轮真实 Genesis 本地测试使用固定离线规划响应，四资产和桌子→微波炉→苹果三层支撑
均通过几何检查和三视角渲染，物理步数为 0。用户授权后，原四资产任务另完成真实 HTTP
规划：1 次模型调用、缓存未命中、12.38 秒，苹果靠左/杯子居中/碗靠右；45 文件哈希通过，
01_obj 保留、02_scene 更新，03/04 仍 not_run。在线调用证据与离线响应验收分别记录。
命令和证据边界见
[Genesis 平台指南](../../self_improving/sim_adapters/genesis/README.md)。

SimFoundry 视频重建沿用独立 external 边界：上游子模块固定为
`NVlabs/SimFoundry@9e34ebefcd020583fbb755a8b57268dce78eca26`，
`self_improving/sim_adapters/simfoundry/run.sh` 只提供平台 CLI，原样调用上游
A 重建、B 增强和 C smoke。独立环境位于被忽略的 `.cache/simfoundry/`。
输入视频经深度、分割、网格和位姿估计输出 OmniGibson 场景；它尚未接到
`ResolvedSceneSpec` 或 OpenXSim，不构成稳定核心的物理通过证据。
`doctor` 仅检查本地存在性；抽帧、环境安装、完整重建与仿真 smoke 的验证状态
分别登记在 [复现记录](../../self_improving/sim_adapters/simfoundry/REPRODUCTION.md)，
agent 命令见 [调用指南](../../self_improving/sim_adapters/simfoundry/README.md)。
本机 Fruits 已完成 7 物体重建和 C 随机动作 120 步回放，视频 121 帧且 121 互异帧；
产物位于 `data/simfoundry/fruits_da3_20260907/`，哈希见复现目录的
`reconstruction_evidence.json`。本次文字模型使用代理 `gemini-2.5-flash`，
图片仍用 `gemini-3-pro-image`；没有测量重建精度或执行稳定核心物理验收。
SimFoundry 单物体已有独立迁移入口：导入自包含 URDF 包，等价规范化惯量坐标，
保留视觉、碰撞、质量和摩擦来源；六视图通过项进入新库，并与官方库联合检索。
联合库选中后执行通用落体验证，失败保留候选、返回错误并阻止自动场景构建。
这项单资产预检不代表原视频整场景已迁移，也不替代组合场景或稳定核心物理验收。
本机 7 个 Fruits 资产导入/预览/动态加载核验通过，7 项落体均完成但未通过既定物理阈值；
具体命令和证据见 [标准资产接入指南](../../self_improving/sim_adapters/genesis/SIMFOUNDRY_ASSETS.md)。
固定上游还实现了单图输入开关 `s1_video.single_image_input`，平台 CLI 可透传；
当前仅核实源码支持，单图端到端尚未实测，参数见调用指南。

AgenticSim 名称有两种历史含义：旧产品仓库已经证明是 TacHarness 的稀疏历史状态，其唯一文件归档进 TacHarness 后本机副本已删除；`sim_adapters/agenticsim_runtime/` 只保留后来非 Git 工作区里的 Isaac 编排脚本，二者不能再混用。

PEARL portal 与 alternate Text2Env 都通过有双亲的历史合并接到主线，来源 tip 分别仍可沿祖先链追溯；精确 source/merge commit 和被排除的本地缓存登记在 `self_improving/source_inventory.json`。散落的 can/basket video anchor 标注则作为小型结构化证据放在 `self_improving/alchedata/artifacts/openxsim/`。

### 已检索资产的文本构建与修复

Genesis 适配层的 `construct_asset_scene.py` 使用独立 `text_repair_v1`：从已有任务复制
原文及 `01_obj` 资产绑定，在新任务的 `02_scene` 测量可见几何、检查尺度、按需派生
碰撞，并按支撑图产生确定种子的 50 候选。透明碰撞显示网格不能作为视觉参考。
新流程不做图像恢复、不重新检索，也不修改稳定核心或旧原生验证的失败证据。

`03_physics` 每次重新加载全场景，采集 3 秒/1500 步；以最后一秒的质心速度、
真实父对象接触、全程穿透、完整目标局部投影及文本关系判定。失败时只重采样首个失败
对象的 XY/yaw，携带支撑子树，其余初态保留；所有动态对象重新仿真和判定。
每个尝试独立保留输入、轨迹、判定及 MP4。场景比例和初终态漂移是诊断/评分，
不能覆盖必需对象的失败。仅全部通过且指定 `--render` 才进入 `04_final_render`；
最终环绕视频明确只有相机运动。具体命令与限制见
[文本构建指南](../../self_improving/sim_adapters/genesis/TEXT_REPAIR_PLAN.md)，
实测状态见 [验收记录](../../self_improving/sim_adapters/genesis/TEXT_REPAIR_EVIDENCE.md)。

本轮文本修复的三种子四资产构建均在 02 的苹果/碗碰撞准备门控失败，03/04 未运行。
真实动态多层夹具与局部子树修复回归通过；独立真实杯子探测完成连续仿真，但速度和
预期支撑比例仍失败，其视频只保存在 03。这些结果不能相互替代。新增几何门控保留
非凸三角连接，避免整桌凸包虚构碰撞；运行时核对动态质量、质心及惯量与冻结资产一致。

2026-09-07 只读补查定位到苹果准备链的具体缺陷：99,994 面组件简化为 2,000 面时
引入两条非流形边，复刻数组与保存输入一致；它对 CoACD 超时的贡献尚未隔离。
杯子终末接触中断与超速样本重合，但高度波动仅 0.04417 mm，平均支撑力约等于自重。
这些现象和因果限制见适配器验收记录；不能把微小接触抖动描述成明显掉落。

用户另行确认的 `text_repair_gt75_v1` 将稳定速度达标比例与正确支撑比例改为
严格大于 75%；最后 500 个采样至少 376 个达标。其余门控不变，原 95% 配置和历史
失败证据保留。用新配置运行也必须逐对象全部通过才能接受，不能按场景总分替代。

### 输出、共享资源与缓存

`output/` 只保存按输入命名的场景任务目录，01/02/03/04 的职责不变。
共享原始资产、预览和检索索引位于 `assets/genesis/`；历史检索及布局验收归入
`data/genesis_history/`；SimFoundry 视频重建产物归入 `data/simfoundry/`。
缓存和任务锁统一位于 `.cache/genesis/`，CLIP 权重按需下载至其
`model_weights/chinese_clip/`。清缓存不会删除源资产或场景证据。

2026-09-07 整理保留原索引、资产和两个任务的文件字节及哈希。Genesis 读取旧
`output/genesis_assets/` 或 `ouput/genesis_assets/` 引用时，只对已迁移的明确资源根
做路径解析；不重写冻结 JSON，也不允许任意软链接替代依赖。旧根被重新创建时
不会静默重定向，资源内容仍须通过完整清单核验。

维护回执及旧目录说明位于 `data/storage_maintenance/`。整理时 SimFoundry 进程仍使用
旧参数，因此曾通过原子迁移暂留兼容链接；现有管线结束后该链接及残留空目录已清除。
新运行示例和本地重跑脚本直接使用 `data/simfoundry/`。
迁移清单和检查结果见 `data/storage_maintenance/output_reorganization_20260907/`。

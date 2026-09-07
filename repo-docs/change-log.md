# Change Log

- 2026-09-07：推送 gujie 分支前发现统一媒体依赖 SimFoundry 工作树修改；保存绑定上游版本的源码补丁和中文路径回归，补充新 checkout 应用说明，保持子模块指针和本机修改不变。

- 2026-09-07：推送前核对根 README 与当前 Genesis 源码，补齐文本 v2 修复、单图/视频统一重建、保留位姿场景导入及独立物理命令；按入口区分已完成与未通过的验收，移除依赖本机 test/ 媒体的示例路径。平台指南已覆盖对应行为，未修改代码或验收阈值。

- 2026-09-07：新增单图/视频统一 SimFoundry→Genesis 入口，保存完整媒体帧统计、stage 3 有限支撑观测、Genesis Top-3 支撑检索、自包含 fixed/collidable support_0 和四轮白名单物理修复；修复中文输出路径的 Hydra 引用。鼠标单图真实完成 stage 1b–4 并识别 desk，视频真实解码 124 帧/124 互异帧；外部 GPU 占用阻塞后续，且未获媒体发送 Gemini 的明确授权，故两个任务均为可验证 execution_failed、03/04 未运行，不宣称端到端或物理通过。


- 2026-09-07：测试 5 条已有资产自然语言组合，保存真实解析、固定资产复用来源、初态预览及杯子三秒物理视频；明确居中软偏好偏差、on 措辞限制、inside 未支持，以及碗加载惯量核验失败。仅诊断与文档更新，未放宽门槛或修改源资产。

- 2026-09-07：按用户授权扩展文本碰撞生成为 collision_repair_v3，保留全部原几何与 95% 门槛；区分 CoACD 300 秒调用预算与每资产 1200 秒预算，新增序列化代理哈希攻击测试。只读对照 SimFoundry，记录其每步清零/始末位移与连续自由动态验收的区别，不移植整体凸包免检兜底。完整四资产结果仍须六次真实验收。

- 2026-09-07：实施文本四资产修复 v2，正式验收恢复两项比例均 ≥95%；分离修复与数值配置，保留旧证据，增加实际接触参数审计、三种子/半时间步对照、准备失败分层和按 dt 命名的完整顺序回放。真实完整场景状态以文本修复验收记录为准，单项测试不替代物理验收。

- 2026-09-07：补充原 SimFoundry 支撑面选择与定义：类别顺序/SAM3 最大区域、参考帧评分、RANSAC 与相机到世界坐标；区分内置碰撞平面、可选桌子库资产和 visual_only 背景重建。本次记录为 15 帧中选择索引 11，未启用 GS。仅文档更新。

- 2026-09-07：进一步核实支撑面处理为分割深度点云的 RANSAC 平面拟合，仅输出类别、原点和法向，s4 用于坐标统一、s12 使用内置 plane.urdf；没有导出桌面资产网格。补充场景指南，未改代码或运行结果。

- 2026-09-07：核实 Fruits 无独立桌子资产：s3 将支撑面识别为 desk，s11 只有七个桌上物体，s14 实际 include_table=false 并使用内置平面；补充场景指南，澄清桌子未在 Genesis 转换中丢失。仅文档更新。

- 2026-09-07：为导入场景新增独立 Genesis 物理诊断与渲染入口，保留原始位姿和资产参数，记录通向地面的实际接触链而不推断声明关系。Fruits 1000 步完成，5 项稳定性通过、梨初态穿透及青色盘子角速度失败；101 帧/101 互异帧视频及终态三视角已生成，轨迹复判与哈希核验一致。仓库 363 passed，专项 33 passed。

- 2026-09-07：新增 SimFoundry 场景图到 Genesis v2 的独立转换与预览入口，保留完整位姿、资产哈希及未知关系语义；Fruits 七物体真实动态加载/惯量核验和三视角零步进预览通过。同步平台边界与命令指南，不宣称物理通过。根仓库 363 passed；Genesis 全套 408 passed / 44 skipped，最终新增源文件过期攻击后专项 21 passed；最终重转包哈希一致。

- 2026-09-07：按用户当前使用 Genesis 的要求，根 README 聚焦 Genesis 场景与 SimFoundry 资产流程，移除旧后端操作和验收介绍；同步指南阅读入口，保留来源归属及实际源码边界。仅修改文档。

- 2026-09-07：只读诊断持续失败，复刻确认苹果 quadric 简化引入两条非流形边；杯子127个无接触样本与超速完全重合，平均支撑力约等于自重。记录源摘要与复现脚本，区分已证实机制和未隔离的超时/接触主因；未修改运行行为或阈值。

- 2026-09-07：按用户明确确认新增 text_repair_gt75_v1，两项终末比例严格 >75%，其他检查与旧配置不变。桌子＋杯子真实复跑种子 0/42/87 仍为 74.6%/74.2%/74.0%，各 1500 步完整失败；完整四资产新任务在苹果/碗代理准备处拒绝，03/04 未运行。仓库 363 passed，Genesis 372 passed / 43 skipped，独立轨迹复判及原证据哈希检查通过。

- 2026-09-07：按用户要求将 output 收口为输入任务目录；共享 Genesis 资产迁至 assets/genesis，历史验收和维护回执迁至 data，缓存与任务锁迁至 .cache/genesis。保留冻结证据字节，增加旧资源路径读取兼容和篡改/锁回归；运行中的 SimFoundry 原子迁移，管线结束后临时链接已清理。仓库回归 363 passed，Genesis 365 passed / 43 skipped；真实四资产原生加载与原 02 网格哈希一致，CLI/lint 通过。

## 2026-09-07

- 实现 SimFoundry 单刚体 URDF 标准包、六视图及 Genesis 原生几何/惯量核验，修正惯量旋转的等价表示以兼容原生解析。新库与官方库联合索引共 81 项；先选中后验证，失败保留候选并阻止后续场景构建。Fruits 七项真实落体均完成 1000 步但未通过原阈值，保留诊断和连续采样视频；不迁移整场景、不修复资产。 根仓库回归 363 passed，Genesis 388 passed / 44 skipped，真实落体对照 1 passed；独立包在禁止原目录读取时完成真实加载/步进。

- 核实 SimFoundry 单图输入源码入口并补充调用说明：透传 single_image_input、单帧采样和参考图索引；明确只有视频路径完成实测，单图及模型推断的隐藏几何/物理参数不作测量保证。

- SimFoundry Fruits 完整重建完成：7 个物体，A 阶段全部 success（可选 stage 9 未启用），后半段 57 分 56.42 秒、退出码 0。产物定位到迁移后的 data/simfoundry，场景字节未变；从新路径运行 C 随机动作 120 步通过，视频 121 帧 / 121 互异帧。新增场景、阶段和视频哈希证据及可直接执行的 agent 命令；未测量重建精度或声明核心物理通过。

- SimFoundry DINOv3 授权与下载完成；用户明确授权 Fruits 示例图片发送到所提供代理，文字/图片输入及原生图片生成实测成功。针对代理流式文字的非标准 thoughtSignature，新增显式启用的进程级非流式文字适配，保留响应终止/安全元数据与图片原始流式路径；回归 363 passed。已启动示例 stage 5–14 的真实顺序重建，完成状态见复现记录。

- SimFoundry 授权后复验：SAM3 与 RMBG-2.0 权重已下载并记录哈希；SAM3 首帧离线推理得到 4 个非空掩码，Fruits stage 3 / 4 在显式 heuristic 选帧模式下通过。DINOv3 仍未授权；原 Gemini 配额仍为 0。另按用户指定的 google.generativeai 格式测试 gemini-2.5-flash，新候选密钥被 Google 官方接口以 API_KEY_INVALID 拒绝，等待服务来源/接入地址；完整场景与其 C smoke 仍未验证。

- 按 README 复现 SimFoundry，将已有干净 checkout 固定为 external 子模块，并增加独立 Miniforge 环境配置、薄 CLI、只读 doctor、来源清单和中文调用/复现指南。真实 Fruits 预处理通过：452 帧、452 互异帧、15 张采样图；随后在主环境已有 DA3 runtime 中离线完成 15 帧深度及相机估计，数值检查与产物摘要通过；CUDA 运算及原生 OmniGibson 立方体 120 步 smoke 通过，主仓库 356 passed，上游轻量 20 passed / 1 skipped。记录并恢复主环境 charset-normalizer 混装故障；完整视频重建与其 C smoke 尚未验证，四个独立 GPU 环境安装完成（恢复安装退出码 0），默认 DA3 环境离线视频深度复验通过，Hunyuan 形状/纹理模型离线加载通过；SAM3 授权及 Gemini 默认模型生成配额仍阻塞全流程；不改变稳定核心物理门控。

## 2026-09-06

- 补四资产失败的根因诊断：三次真实隔离对照、每次 1001 行；基线复跑哈希一致，换摩擦锥及接触模型仍失败。轨迹与位置差分确认无接触步重力速度峰值，初态边缘余量缺少缓冲；碰撞几何/响应刚度尚未隔离。证据归入 03 的 contact_analysis，原阈值、原结果和 04 not_run 保持不变。

- 澄清阈值来源：现有位移/速度/穿透、2 cm 余量与 80% 支撑要求属于平台计划；固定官方 Genesis 的多物体回归采用个案容差及分位数口径，不能当作通用验收标准。仅同步说明，未更改阈值或物理结果。

- 按用户纠正收紧阶段职责：物理验证视频与失败终态诊断全部迁入 03_physics，四资产任务 04 清空并恢复 not_run。原物理证据及全部 PNG/MP4 字节保留，路径迁移与新旧摘要归档。TaskOutput 在完成物理阶段、封存与读取时拒绝物理未通过却有 04 产物/已执行状态的任务，重跑清除旧诊断索引；同步指南及输出导航。以下此前将诊断写入 04 的记录属于已纠正的历史操作。 验证：仓库 pytest 356 passed；阶段/原生物理针对性回归 96 passed，场景构建回归 12 passed；相关 ruff、CLI、任务与视频摘要检查通过。

- 按用户请求为原四资产任务生成原速、5 倍慢放与终态环绕 MP4（1280×960）。全部 1001 个物理采样逐帧恢复；以 force_render 刷新零步进缓存，编码前后互异帧分别核验为 201/1001/180。视频绑定原轨迹与上游哈希，物理结论仍失败，默认验证入口不变。

- 用户显式要求查看四资产任务后，零步进渲染其真实 4 秒终态到 04_final_render：四视角 1600×1200，逐顶点/位姿及上游哈希核验通过；渲染 passed，任务仍 physics_failed，01/02/03 字节保留。未改物理入口默认流程，输出附可复现脚本和渲染报告。

- 实现原生 MJCF/GLB 的独立 `validate_asset_scene.py`：冻结已选资产及 02 输入，原生视觉参考与动态加载逐顶点核验，初态碰撞与逐步受力分开，连续轨迹、目标局部完整投影、多层父子接触及独立 physics_passed/physics_failed 状态；04 保持 not_run。
- TaskOutput 支持保留 01/02 字节的独立物理副本，旧 owner 快照单独按摘要核验。提取/构建默认行为与核心契约不变；无模型调用、无资产替换/缩放/网格修复或阈值放宽。
- 四资产三次基线及半时间步均真实失败并保留 1001/2001 行证据；三次基线轨迹哈希一致。校准、动态中间层 GLB 三层夹具及五类反例通过各自断言，真实测试 7 passed / 2 failed。官方微波炉含旋转门关节，明确不支持；其视觉三层例不能当物理通过。
- 同步根 README、Genesis 指南、原计划状态、中文平台指南及 PHYSICS_VALIDATION_EVIDENCE；全部正式试验产物哈希与副本上游内容摘要已读回核验。仓库回归 356 passed，Genesis 离线 295 passed / 33 skipped（新增 58 项），相关 lint、CLI、语法与 diff 检查通过。

- 新增已有资产场景物理验证实施方案 Markdown，记录原生 MJCF/GLB 接入、物理输入、坐标核验、运行判据、多层接触和测试要求；仅新增规划文档与导航，未实现或运行物理阶段。

- 核查 Genesis 官方资料与固定源码，澄清 Box/单 link URDF 及 compile_manifest 是旧平台入口限制；官方原生具备 MJCF/GLB 物理加载、接触/穿透与运动观测接口，并有橙色塑料碗堆叠回归。同步平台指南，未改动物理执行或验收规则。

- Genesis 已选资产场景构建新增 LLM 偏好规划、v2 图契约及有界几何求解，支持多层 on 和左右前后/远近。资产选择成功后默认自动执行第二阶段，--stop-after assets 保留旧入口边界；独立 build_scene 默认 llm，规则模式需显式选择。
- 模型不得改变资产或明确关系；每轮至多两次请求，成功缓存和脱敏证据绑定输入。内腔/inside 暂缓，03/04 仍 not_run；同步平台及输出指南。真实 Genesis 本地预览使用固定离线响应；后经用户明确授权，四资产任务在线规划 1 次 HTTP 请求、缓存未命中、12.38 秒通过，45 文件哈希核验通过。在线证据与离线预览分开，03/04 仍 not_run。

- 用户显式要求构建场景后，新增独立 build_scene 入口，使用四个已绑定官方资产和 Genesis 内置地面生成初始关系图、实测布局及三个视角。提取默认仍止于选资产。连续凸水平面覆盖完整物体投影，浅边条与上方遮挡有独立攻击测试；首次最高平面拒绝后，经原生网格测量确认中央桌面，重建实测 9.45 秒，43 文件及上游哈希通过。场景重跑保留对象记录、清空下游；物理步数与模型调用均为 0，未生成物理编译包。仓库 pytest 356 passed、Genesis 离线 202 passed / 22 skipped、OpenXSim 101 passed / 1 skipped，CLI 帮助、ruff、py_compile 与 diff 检查通过；同步平台、交接及输出指南。

- 新增平台完整物体提取与选择入口及独立版本格式/缓存，包含显式桌子、柜子等支撑物与容器，未提桌子时不补；原文数量、属性、完整查询和关系保留。旧 scene_preview CLI/run 转调用新流程，保留拒绝及错误对象，默认停止于资产选择，三个下游阶段 not_run。稳定核心 schema/提示词/求解器与 CLIP 选择算法不变，地面留待后续 Genesis 内置环境，物理和渲染不在本轮执行范围。真实四对象与无桌两对象最终通过，首轮关系证据改写失败保留；增加对应攻击测试。仓库 pytest 356 passed、Genesis 离线 190 passed / 22 skipped、OpenXSim 101 passed / 1 skipped；CLI、ruff、输出哈希与密钥检查通过。

- 按用户修正，仅任务顶层目录保留自然语言命名；内部目录统一为 `01_obj/`、`02_scene/`、`03_physics/`、`04_final_render/`，解析与选择子目录为 `parse_records/`、`asset_selection/`。现有三物体结果原地更新路径、导航和哈希清单，图片及源资产字节保留，后续运行使用相同英文结构。

- 目录调整后按用户要求真实运行苹果、黄色杯子和橙色塑料碗的多物体测试：新自然语言目录保存三对象、全部选择中间产物和三张真实初始场景图，32 文件清单核验通过；36.82 秒、三次选择缓存命中、新增 VLM 请求 0、物理步数 0。新结果入口更新到 `output/README.md`，物理与最终渲染保持 `not_run`。

- 按自然语言统一平台多物体任务输出到 `output/`，集中管理四阶段、原始请求、结果导航与 SHA-256 清单；增加安全命名、同请求覆盖、目录归属与排他锁。物理入口新增互斥 `--scene-dir`，仅清空第三、四阶段，绑定并复核上游哈希，终态编译和渲染路由到第四阶段。保持核心 schema、解析提示词、选资产逻辑与物理门控不变。
- 删除用户指定的旧三物体演示，不迁移或扩大删除范围；同步平台、交接和输出目录指南，保留官方包、权重、缓存及其他历史结果。离线夹具覆盖命名、覆盖、失败清理、路径保护和物理路由，本轮不新增真实物理实验。仓库 pytest 356 passed、Genesis 离线 173 passed / 22 skipped、OpenXSim 离线 101 passed / 1 skipped；CLI 帮助与 ruff 通过。

- 新增平台多物体初始场景预览入口：复用现有解析、对象记录和 CLIP/VLM 绑定，按实测边界原比例摆放独立桌面物体，保留完整中间产物和三张真实场景图；不改核心 schema，不步进物理。苹果、黄色杯子与橙色塑料碗实测成功，记录实际外观差异和哈希绑定初始位姿。同步平台指南、交接、输出目录和本指南，明确预览布局尚不是物理编译包，后续仍需接动态碰撞门控。

## 2026-09-05

- 将已下载的官方非机器人库接入独立 CLIP 选择器：发现 106 个候选入口，74 个通过六视图预览并编码为 444×512 向量，32 个失败保留原因；支持多格式可信入口及下载清单哈希，不改旧四资产契约、对象 schema 或物理流程。
- 新增来源/依赖/失败状态/路径/图片篡改回归；优化 OBJ 依赖分词扫描并单列完整性耗时，保留查询前后哈希核验。三种新资产真实测试与缓存 4 passed，原六查询在新库上 7 passed；同步平台指南、交接和输出目录指南，逐步产物位于 `ouput/asset_selection/non_robot_acceptance_v2/README.md`。

- 按用户要求下载固定官方 revision 的非机器人资产原文件到 `ouput/genesis_assets/non_robot_v1/sources/`：597 文件、1.664 GB，排除 251 个机器人相关文件，官方 Git/LFS 哈希和完成后读回检查通过。新增本地来源/排除/哈希清单，指南明确文件数不等于独立资产数，下载库尚未接入原四资产 CLIP 检索或物理通道。

- 按用户要求清理 `ouput/` 的旧运行/失败目录和重复资产预览，保留正式官方包、CLIP 索引及模型权重；小型清理记录保留历史六查询摘要，并将指南/交接中的旧证据目录标记为已删除。
- 用“我要一个黄色的、印有水果图案的杯子。”执行新的单次观测测试，选中 cup_2，记录文本向量、全部视角分数、真实六图请求、模型原始响应及最终绑定；查看入口为 `ouput/asset_selection/yellow_fruit_cup_01/README.md`。不改生产选择逻辑，不新增物理实验。

- 新增独立 ChineseCLIP 六视角 Top-K + 单次 gpt-4o 多图选择，固定模型 revision，输出可信索引绑定或明确拒绝；不改对象 schema、解析提示词/transport，不接求解、OpenXSim 或物理实验。补充哈希/超长文本/目录保护/缓存/非法响应攻击回归。
- 同步平台指南、Genesis 交接及本地 `ouput/README.md`：四条真实正例命中并选中对应资产，锤子与米老鼠杯拒绝，真实验收 7 passed；记录首次建索引、查询和缓存实测耗时，说明绑定不等于所有外观、几何或物理条件满足。

- 新增独立 Genesis 官方四资产文件索引和零物理步六视图入口，原生加载 `model.xml`，保留多外观及独立碰撞部件、固定来源版本、依赖与图片哈希；提供总览和只读完整性检查。明确它不是语义 catalog，不新增检索或改动场景/物理通道。
- 按用户指定，将自然语言 CLI 默认输出改为仓库 `ouput/`，保留 `--out-root` 覆盖并忽略生成产物；记录用 OpenXSim `--output ouput` 将已有编译和渲染产物集中在同一场景目录的方法。不迁移历史哈希绑定输出，也不新增物理/渲染自动编排。

- 调整自然语言 CLI 为“校验 SceneSpec → 提前保存逐对象 JSON/关系/解析证据 → 检索全部对象 → 全部匹配才求解”。新增 `asset_resolution.json` 区分 matched/missing/blocked 与整体 catalog 错误；保留失败中间产物，沿用现有 catalog，不接资产生成服务或官方资产索引。
- 对象及检索记录纳入既有包 manifest；只读复用同输入、同 catalog、同解析证据且静态验证完成的有效成功包，拒绝覆盖其他已有目录。显式旧代理生成、默认 rule、LLM 两阶段与缓存、求解算法以及下游 OpenXSim/Genesis 契约不变。

- 新增平台独立 Genesis 物理入口，串联“已有编译产物 → CPU 刚体检查 → 仅更新动态终态位姿 → 已有编译器和渲染命令”；不修改 OpenXSim 或稳定核心，物理失败不渲染，物理与渲染结果分别记录并哈希绑定。
- 新增官方 mug 单刚体 MJCF→URDF 测试准备，保留 visual/纹理及 32 个 collision 部件、几何测量与文件指纹；只覆盖 Box 落桌、mug 落桌、Box 入杯，不接入 RoboTwin 资产或 L2–L4。
- 增加离线攻击测试及显式开启的真实重复、半时间步、四类反例验收。初始阈值不保证三个正常场景全部通过；详细实测状态与复现入口见平台适配器 README 和 Genesis 交接说明。

## 2026-09-04

- 新增显式 opt-in 的两阶段 LLM 自然语言提取：objects/attributes 与 topology/lateral 分阶段生成，模型候选经严格 JSON、字段白名单、高置信语义一致性检查和同一 `SceneSpec` 契约后，才进入既有 catalog grounding 与确定性 solver；默认 rule、Demo 和批量 runner 不变，也不做自动 fallback。
- 增加环境变量及 `configs/llm.yaml` profile 配置；YAML profile 支持互斥的内联 `api_key` 或 `api_key_env`，本地配置保持 git-ignored，密钥不进入 repr、fingerprint、缓存、证据或失败报告。成功结果原子缓存并生成无密钥 `llm_parse_evidence.json`，由 package manifest 哈希绑定，配置、传输、语义或 grounding 失败均产生结构化失败报告。
- 加固 provider 信任边界：禁止代码、URI、POSIX/Windows/UNC/环境变量路径、backend id 与坐标进入模型；拒绝重复 YAML/JSON 键、超限输入、非有限或溢出数值、否定反转、未绑定关系和中文已知词嵌入未知复合名词；确定性歧义立即失败且不会重试。
- 新增 fake-transport、缓存、CLI 和攻击回归；测试不会访问真实模型服务或读取真实 API key，在线模型质量与首次未缓存调用仍需单独受控验收。

## 2026-09-03

- 新增 env-gen 经 OpenXSim 迁到 Genesis Rasterizer（Pyrender）的首版渲染通道说明：固定 `external/genesis-world` 子模块 commit，公开两步 `transfer` / `runtime_command` 用法、runner 参数与十个固定产物，并记录 `agenticsim.genesis_render_evidence.v1`、`agenticsim.genesis_render_manifest.v1` 的 package-digest 绑定。
- 明确 Genesis 首版保持 resolved pose、只移动相机且不做物理 step；渲染证据只能覆盖 OpenXSim L0/L1，L2-L4 仍为 `not_evaluated`，不能替代 RoboTwin/SAPIEN 的 contact、support、containment 与 stability 门控。
- 加固 Genesis 渲染证据：package digest 纳入主资产与递归本地 sidecar 指纹，runner 通过确定性重编译精确绑定 `scene.json`，并在成功前重新解码 MP4、严格门控帧数/互异帧/分辨率、对象 pose 与 articulation qpos；scene/资产篡改、Genesis backend 重标、伪造 L2-L4 evidence 和失败后陈旧产物均有攻击测试。
- 将官方 `Genesis-Embodied-AI/genesis-world` 登记进 `self_improving/source_inventory.json`，以 `external/genesis-world` gitlink 固定 commit `0e74bf392781884ccad765c3f344419c86b872ca`。
- 校正运行时门控指南中的静态旋转漂移阈值：`rotation_drift` 源码默认是 3°，`resolved_rotation_error` 才是 5°；未改变运行逻辑。

## 2026-08-18

- 为 Harness MVP PR1 增加正式中文实现报告，逐项记录 14 个公共 schema、Pydantic/JSON Schema 分工、状态机、Text2Env 边界、100% 覆盖率证据、兼容性风险和 PR2 前置清单。
- 增加 reader-facing Harness Schema Tranche 模块页，并同步平台总览、代码地图、术语表、三轮源码证据和质检残余风险；明确 schema tranche 尚不包含 Registry、handler 或 MCP，run `succeeded` 也不等于 validation pass 或 publishable。

## 2026-08-17

- 增加 Harness MVP PR1 schema tranche：14 个严格、不可变公开 schema，Text2Env compile/replay/validate 边界、committed JSON Schema 漂移检测与 100% 语句/分支覆盖门；Registry、handler、MCP 留待后续，RFC 仍为 Proposed。
- 校正回放指南中的 contact window 默认值：独立 `run_scene_runtime.py` CLI 当前默认 60；README、prompt matrix 与已验证配方显式传入 120。未改变源码或既有验收证据。

## 2026-08-14

- 统一仓库本地大数据根目录为 ignored `data/`，移除与其语义重复的旧目录约定；Jingxiang canonical checkout 的 7.1 GB payload 已同盘改名，9,576 条新路径 manifest 全量通过，历史 manifest 与清理 receipt 继续保留执行当时的原始路径作为不可改写的审计证据。
- 固化 Jingxiang 仿真组多人协作布局：canonical `workspace/robot-harness-gen-env` 只承载共享 `main`，Bingsheng、Gujie、Yeyuxuan、HYX 分别在个人名字目录使用 `worktree/<人名>`；个人产物不得越出个人 workspace，共享材料须经过明确提升。
- 将 `huyuxinn/env-gen-dev` 完整历史接入 `worktree/hyx`，迁入遗漏的 legacy 资产工具与 33 份个人 ledger，并在分支推送、manifest 复核后移除第二个本地 checkout；原事故记录中的“RoboTwin 全灭”继续以 canonical cleanup receipt 的现场复核为准。
- 迁入 Yeyuxuan 完整 RoboLab onboarding 分支历史、20 份资产来源记录、迁移 CLI 与运行时语义修复；大文件只保留 SHA-256 清单，未把第三方 payload 放入 Git。
- 调和 Yuxin 当前 `main`、`feat/web-studio-v2` 与未提交的断点续测修改，完整保留各历史 tip，并把资产流水线改为由 `runtime_config.py`/环境变量提供路径。
- 将 6 个旧归档入口共同保存、但不在 `main` 或任何个人 worktree 中的 38 个独有提交以 history-only merge 并入 `worktree/hyx`（`899c649`），前后 tree SHA 均为 `971cb34`；确认全部旧 tip 可从活动分支到达后，退役 15 个 `archive/*` 和 1 个 `integration/*` 远端分支。
- 保存 `301`–`361` 外部资产命名空间的 12,047 文件摘要、选择 manifest 与小型 ledger/model metadata；27,637,543,884 字节本体因 `storage_uri: null` 继续留在本地。
- 把 Bingsheng、Gujie、Yuxin 独有的设计/交接文档作为历史材料纳入 `self_improving/contributor_notes/`，不把旧绝对路径包装成当前命令。
- 补查 `.gitignore` 后保存 Yuxin 被所有分支遗漏的 42 份 `work/` 源码与笔记，包括中断的属性矩阵 driver；作为只读 workbench snapshot，不冒充正式入口。
- 把忽略的实验数据、checkpoints、RoboLab payload 与资产库移动到 canonical checkout；经 checksum-mode rsync 证明重复后删除 Bingsheng/Yuxin 的 16 GB RoboTwin 资产副本，并保存清理 receipt。Gujie 当前训练占用的 RoboTwin 仍明确留待训练退出后移动。
- 处理并发恢复任务：保留其 `ad28866` 与断点续测 dirty state 到组织归档分支，纠正“迁移即 530GB 丢失”的误判，并把 508 模型颜色、538 条原点校准、471 条顶面探针和 runtime revocation 四份小型实测元数据正式纳入 Git。
- 在最终整合 commit 上初始化所需顶层外部子模块后，自包含回归为 543 passed、5 skipped；Jingxiang 的真实 RoboTwin/SAPIEN 回放门也已完成。
- 原训练自然结束于 epoch 599/global step 25799；保存 `600.ckpt` 的 1,549,185,541 字节大小与 SHA-256 后，将 89 GB RoboTwin 树迁入 canonical `external/RoboTwin`，修复 227 个生成链接与六份 Curobo 配置，最终 broken symlink 为 0。
- 在 Jingxiang `robotwin-5090` 真环境完成 `place_a_can_on_the_table_acd20a6814` 的 900 步 SAPIEN 回放：`pass`、`fail_count=0`、`not_run_count=0`、120 帧（100 unique）；结构化 JSON 与 manifest 已进入 `validation_evidence/student_workspace_20260814/`。
- 对并发恢复的 Yuxin RoboTwin 再做 checksum-mode rsync，唯一六行差异正是 canonical 路径修复；日志和 48 份小文件转入现已统一命名为 `data/` 的 ignored 数据根后，删除所有四个同学的重复个人工作目录。`/home/jingxiang/workspace/` 项目层只剩 `lerobot` 与 `robot-harness-gen-env`。

## 2026-08-13

- 把仓库文档范围扩展为稳定 `/gen-env` 核心与 `self_improving/` 平台两层。
- 登记 Alchedata、stage-05、asset pipeline、sim adapter、onboarding、stage-04 历史和两个外部子模块的所有权边界。
- 明确 AgenticSim 历史仓库与后续 runtime adapter 不是同一个组件。
- 以完整祖先链迁入 PEARL evidence portal 与 alternate RoboTwin Text2Env，并把后者固定为只读 legacy；补回 can/basket video anchor 标注与精确 SHA-256 来源记录。
- 收口 Jingxiang workspace 的七个 sibling：OpenXSim 验收 JSON/日志/复现脚本进入 Git，六个 cache-filtered 完整工作区进入同仓库 Release，MetaSim 固定为子模块，重复 prompt-matrix clone 由既有历史合并覆盖。

| Timestamp | Request | Actions | Verification | Result |
| --- | --- | --- | --- | --- |
| 2026-07-29 10:55 +08:00 | 用 `repo-docs-zh` 为 `robot-harness-gen-env` 首次构建中文 repo-docs 包 | 读完父 `repo-docs` 三个核心规则文件（SKILL/REFERENCE/PAGE_RULES/WRITING/QUALITY_RULES/EXAMPLES）。在 `repo-docs/` 下新建：`README.md`（中文开场 + 阅读路径表）、`walkthroughs/one-real-run.md`（一条 prompt → resolved 包 → SAPIEN 回放 → 运行时门控，8 步）、`code-map.md`（`scene_gen/`/`script/`/`demo/`/`tests/` 四区目录职责 + 关键符号 + 主路径关系 + 排除项）、`modules/` 下 7 页（bounded-parser、scene-contract、target-local-geometry、solver、derived-proxy、replay-package、runtime-gates）、`references/source-evidence.md`（两轮 traversal + claim/evidence/confidence/caveat/used-by 表）、`references/quality-review.md`（Reader Simulation + 可理解性 review + 残余风险）、`glossary.md`（17 行术语）、本 change-log。在仓库根 `AGENTS.md` 末尾追加 `Repo docs` 路由句与中文 overlay 指明。 | `$env:PYTHONIOENCODING = "utf-8"; python "C:\Users\SatelluS\.agents\skills\repo-docs\scripts\validate_repo_docs.py" repo-docs --repo-root .` → 初轮 0 errors / 32 warnings（含 walkthrough 难点触发句、code-map 目录/Header/Coverage 形态、source locator 前缀、证伪检查、5 个高频术语缺 glossary 行），按 warning 逐项修了 5 轮后到 0 errors / 0 warnings。`pytest -q` 在本机未能跑——当前 Python 3.12 解释器没装 pytest、本地无 `.venv`，仓库要求 Python 3.11 + 装了 `dev` extra 才有 pytest；但本次改动只动了 `AGENTS.md` 文范畴路由段与新建 `repo-docs/*.md`，没动任何 Python 源码或 `tests/fixtures/`，故测试集状态不受影响。真机 SAPIEN/RoboTwin 回放另按根 `AGENTS.md` 在支持机器上验证。`git rev-parse HEAD` = `60a25971738e0cd4c64615e4455cc2b4098aaa43`，与 sync anchor 一致。 | build：通过；validator 0 errors / 0 warnings；测试集本机未跑（环境缺 pytest），仅改文档不影响契约层。 |

Synced through 9b720900ff1c3c1b5a6587f7bc5d78359d3af81b.

- 2026-09-06：新增 Genesis `text_repair_v1` 独立文本构建入口、按需碰撞派生、约束采样、
  质心速度物理判定、局部子树修复及按阶段保存的视频。同步平台指南，明确原任务只读、
  全场景重验与全部物理通过后才进入 04；实际回归及资产结果见适配器独立验收记录。
- 2026-09-07：完成文本修复入口的真实动态多层和子树修复回放，补充独立验收记录；
  明确流程回归通过不代表四资产构建通过，当前苹果/碗在资产准备门控被拒绝时 03/04 不运行。

- 2026-09-07：追加真实桌子/杯子 12 个摆放种子对照（含 42、87），全部完成连续采集和 03 MP4；0/12 通过，已核验固定变量、候选复现及证据哈希，结果链接归入 Genesis 验收记录。

- 2026-09-07：按用户要求将 ouput 的约 3.25 GB 有效共享资源迁入 output，保留历史字节与哈希，删除废弃的 scene_preview 占位说明；活动 CLI 改为 output，旧根只保留兼容链接。补充迁移别名、防篡改回归，真实核验两套 CLIP 索引、原场景绑定及离线模型编码。

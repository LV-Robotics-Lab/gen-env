# Genesis 平台适配器

已有场景图现通过既有物理入口完成位置求解、干预式稳定化和独立双时间步验收，支持多固定支撑、堆叠及方位关系，见 [场景图物理流程](POSITION_SOLVER.md)。

SimFoundry 重建物体现可导入为自包含 URDF 包，并与官方库联合检索；选中后执行单资产落体验证，失败报错且保留候选。见 [标准资产接入指南](SIMFOUNDRY_ASSETS.md)。

已有重建场景还可保留完整位姿导入 v2 场景图，并在 Genesis 中核验、预览及独立运行物理诊断，见 [场景图转换指南](SIMFOUNDRY_SCENES.md)。
## 单图/视频统一重建

`reconstruct_media` 默认采用 `--support-mode upstream`：导入前景资产，保留上游
物体位姿和平面的开关、可见性与变换，不检索桌子、不新增 `support_0`。
输出止于 `02_scene/preview` 的三视图和环绕预览；03/04 为 `not_run`，
退出码 0 表示转换与预览成功，不表示 Genesis 物理通过。
显式使用 `--support-mode retrieved --clip-index <index.json>` 才运行原有限桌面选择、
位置求解、稳定化、双时间步验收及通过后的最终渲染。
默认模式无需 CLIP 索引，以下命令中的索引参数仅供切换检索模式时使用：

```bash
.venv/bin/python -m self_improving.sim_adapters.genesis.reconstruct_media \
  --image test/鼠标.jpg --name 鼠标_单图 --output-root output \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --vlm-config configs/llm.yaml

.venv/bin/python -m self_improving.sim_adapters.genesis.reconstruct_media \
  --video test/鼠标视频.mp4 --name 鼠标_视频 --output-root output \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --vlm-config configs/llm.yaml
```

两种输入严格互斥。已有任务只有在 `--resume` 且任务名、输入 SHA-256、模式和有效配置
完全一致时才继续。检索模式退出码 0 表示物理及渲染通过，1 表示输入/依赖/模型/转换错误，
2 表示修复后物理仍失败，3 表示稳定但声明关系未验证。旧 SimFoundry
`run.sh reconstruct --video-fpath` 仍可用，但不会自动进入 Genesis 四阶段。
支撑观测、单图推断边界及本机鼠标媒体运行状态见
[统一重建验收](MEDIA_RECONSTRUCTION_EVIDENCE.md)。


已有重建可直接转换，不重复调用模型：

```bash
venv/genesis/bin/python -m self_improving.sim_adapters.genesis.reconstruct_media \
  --reconstruction-scene /path/to/reconstruction \
  --name '按SimFoundry原始支撑面转换重建场景' --output-root output
```

输出保留 `01_obj/source_outputs` 的上游 JSON、`01_obj/foreground_assets` 的标准 URDF
和 `02_scene` 自包含场景包。背景、高斯和机器人不在当前导入范围；
上游有额外桌子却缺少匹配资产包时明确报错，不自动替换。

`output/` 只保存按输入命名的任务目录。共享资产、预览和索引位于 `assets/genesis/`；
缓存、按需下载的 CLIP 权重及任务锁位于 `.cache/genesis/`；历史验收位于
`data/genesis_history/`。旧证据中的明确资产路径由适配层读取时映射，JSON 字节和哈希保持不变。
目录整理记录见 [存储维护](../../../data/storage_maintenance/README.md)。

已有资产场景可通过 `validate_asset_scene.py` 独立验证物理，见下文及 [真实验收记录](PHYSICS_VALIDATION_EVIDENCE.md)。
[原实施方案](PHYSICS_VALIDATION_PLAN.md) 保留设计背景；验证器已实现，原四资产场景未通过物理门控。

文本修复入口另支持用户确认的 `--profile text_repair_gt75_v1`：稳定速度达标比例与
正确支撑比例均须严格大于 75%（终末 500 个采样至少 376 个）。其他门槛不变；
原 `text_repair_v1` 的 95% 判据和旧原生验证配置保持原样。各配置分别冻结并保留证据。

## 自然语言到初始场景（当前入口）

`extract_assets.py` 默认执行两个阶段：完整物体/关系提取与逐对象 CLIP Top-3 + VLM
选择，然后自动调用 LLM 规划第二阶段的场景图和摆放偏好，程序求解坐标并生成初始预览。
`scene_preview.py` 的 CLI/run 是兼容入口。只需要资产时显式传 `--stop-after assets`。
物理验证与终态渲染仍是独立阶段，本入口不执行物理步进。

```bash
.venv/bin/python self_improving/sim_adapters/genesis/extract_assets.py \
  --request "桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。" \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --vlm-config configs/llm.yaml --output-root output
```

这句话提取 table_1、apple_1、cup_1、bowl_1；“一个苹果和一个黄色杯子。”只提取两对象，
不补桌子。把手、印花作为属性，不拆为独立资产。平台 `genenv.asset_request.v1` 保留完整
描述、原文 mentions/attributes 和明确关系；不使用核心桌面 SceneSpec，地面不参与检索。
每个对象独立选择，拒绝/错误保留对象且继续其他选择，不回退第一候选或生成代理。
全部选择成功才自动进入场景阶段；partial/error 时停止。提取成功缓存仍在
`.cache/genesis/asset_parse_cache/`，新提示词加入 far_from，缓存按提示词摘要隔离。

第二阶段也可独立重跑，复用 01_obj，不重新运行 CLIP 或选资产 VLM：

```bash
.venv/bin/python self_improving/sim_adapters/genesis/build_scene.py \
  --scene-dir 'output/桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。' \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --llm-config configs/llm.yaml --seed 42
```

`--planner llm` 是默认值；`--planner rule` 显式使用旧的单层 on / 无关系排列规则。
自动入口默认复用 `--vlm-config`，可用 `--llm-config` 单独指定场景规划配置。
两种入口都保持资产原始比例和朝向，只求平移。格式支持 MJCF、URDF、GLB/glTF、OBJ、STL；
USD、inside、其他未知关系及明确数字距离暂不支持，报告失败而不替换需求。

### 场景图、布局和几何边界

`build_scene.measure_geometry` 从 Genesis 原生视觉网格测量边界与需要的支撑面，之后
`scene_planning.ScenePlanner` 把原始描述、对象 ID/描述、实测边界和支撑面摘要交给模型。
不发送资产文件或绑定路径；模型只回传原始对象集合、原样的明确关系及可选布局偏好。
`scene_layout` 校验专用 v2 图契约，拒绝漏对象、增对象、改关系、支撑循环、多重直接支撑、
方向循环和远近矛盾。新增偏好单独标为 model_suggestion，不能推断 on 或 inside。
无明确支撑的对象落在 Genesis 内置地面，不在原文关系中补桌子或支撑边。

支持多层 on 和 left_of/right_of/in_front_of/behind/near/far_from。Z 向上、X 正向为右、
Y 负向为前；左右前后比较完整 XY 投影，近/远比较 XY 投影边界的欧氏距离，默认分别
≤18 cm / ≥30 cm。同层间隙至少 6 cm，支撑面边缘余量 2 cm。保留原文关系作为硬约束，
偏好只影响候选排序；每次放置最多 96 个候选，全局最多 48 次回退，失败留求解 trace。
多层支撑按拓扑顺序逐层转换目标局部坐标；非直接支撑对象间做保守 AABB 重叠检查。
该检查可能保守拒绝视觉可行的凹形组合，不能当作碰撞或稳定性证明。

支撑仍使用连续凸水平三角面覆盖完整源投影，并检查上方遮挡；浅边条下探不超过
min(5 mm, 资产高度 1%)。洞、断裂、重叠、深凹面不能冒充稳定面。缺少实测支撑面直接
失败，不让模型猜测。inside 和内腔测量延后；三层桌子→微波炉→苹果仅证明初始视觉几何。

每轮最多两次场景规划请求，每次 60 秒超时；格式/语义/求解失败反馈一次，第二次仍失败
就停止。传输、完整性和未支持能力错误直接停止，无规则 fallback。成功规划缓存位于
`.cache/genesis/scene_plan_cache/`，绑定请求、资产、实测几何、提示词、配置、seed 和求解版本，
命中后重新验证并求解。首次在线调用不保证位级确定性。证据记录脱敏请求响应、实际请求
次数、缓存命中及 `transport=http|injected`；注入测试调用不代表在线模型验收。

### 阶段产物与状态

```text
output/<原始自然语言>/
├── request.txt、README.md、run_report.json、manifest.json
├── 01_obj/           # 逐对象记录、asset_request、relations、解析证据和资产选择
├── 02_scene/
│   ├── input_manifest.json、scene_graph.json、scene_layout.json
│   ├── native_geometry.json、support_surfaces.json
│   ├── planning_evidence.json、layout_validation_report.json
│   ├── layout_attempt_<n>.json（失败尝试）
│   ├── build_report.json、render_report.json
│   └── overview.png、top.png、side.png
├── 03_physics/        # not_run
└── 04_final_render/   # not_run
```

场景图版本为 `genenv.asset_scene.v2`，布局绑定图和几何摘要。成功总状态为 scene_built；
只提取资产为 assets_selected；场景失败时 objects 仍 passed、scene 为 failed，不能误报
完整流程成功。三张图片均须生成，每个对象至少一个视角可见像素 ≥64，允许堆叠遮挡个别视角。

自动链路先释放资产阶段锁，再在场景锁内核对交接 manifest，避免嵌套锁及上游替换。
仅重建场景保留 01_obj、清空 02/03/04；完整重跑清空四阶段。绑定及源文件哈希在加载和
模型调用前后重新核对。根目录导航和全文件清单随状态更新。初始布局不是既有物理入口
所需的编译包，不能直接传给 validate_physics。核心 scene_gen 的信任边界和 SAPIEN 门控不变。

### 验收与历史证据

四资产任务已在用户授权后完成在线 LLM 构建：实际 HTTP 请求 1 次、缓存未命中、总耗时
12.38 秒。模型建议苹果靠左、杯子居中、碗靠右，原有三条 on 关系保持；程序求解零回退，
三视角中四个对象均达到可见阈值，45 文件哈希及图/几何摘要绑定通过。原目录 01_obj 保留，
02_scene 已更新，03/04 仍 not_run。此前规则构建为历史结果。
另有本地真实 Genesis 验收使用固定离线规划响应，覆盖同一四资产与
桌子→微波炉→苹果三层支撑，实测几何、三视角及零物理步进通过。产物位于
`output/scene_planning_acceptance_20260906/`，每组有 acceptance.json，明确 live_model_calls=0。
本轮回归：仓库 pytest 356 passed，Genesis 离线 237 passed / 24 skipped，真实本地预览
2 passed；ruff、CLI 帮助、py_compile、清单哈希检查通过。
在线调用证据位于任务目录 `02_scene/planning_evidence.json`，记录 transport=http、calls=1、
cache.hit=false；该记录与固定离线响应验收分开，不把初始预览当作物理通过。

```bash
# 离线回归（不访问模型服务）
.venv/bin/python -m pytest -q self_improving/sim_adapters/genesis/tests
# 真实 Genesis，仅使用固定离线规划响应；输出目录须为新目录
GENESIS_SCENE_REAL=1 GENESIS_SCENE_OUTPUT=/tmp/new_scene_acceptance \
  .venv/bin/python -m pytest -q \
  self_improving/sim_adapters/genesis/tests/test_real_scene_planning.py
```

## 官方非机器人资产下载（本地文件库）

2026-09-05 按用户要求，已将官方仓库 revision `5b01555c225977d2d3973710884adc5add269fe8`
的非机器人文件下载到 `assets/genesis/non_robot_v1/sources/`，保持远端目录和原始字节。
共 597 个文件、1,663,607,533 字节（1.664 GB / 1.549 GiB），排除 251 个机器人相关文件；
文件数包含纹理、碰撞部件和同物体多格式，不等于独立资产总数。
排除机械臂、灵巧手、人形/四足机器人、无人机、夹爪及专用零件；保留普通物体、场景、
布料/变形网格和配套材料。含糊条目检查见该目录 `scope_review.json`，不以 URDF 的
`<robot>` 标签把普通单物体也排除。下载逐文件验证固定官方 Git blob SHA-1 / LFS SHA-256
及大小，完成后重新读回 SHA-256 通过，失败 0；清单与报告分别为
`download_plan.json`、`download_manifest.json`、`download_report.json`。

该目录保留官方原文件与下载证据；本次另建预览和 CLIP 索引，接入结果见下一节。
原 `v1` 官方四资产包和 `clip_v1` 保持不变。下载没有额外完整缓存副本，资产二进制不进入 Git。

## 非机器人库接入 CLIP（当前可用）

`build_library_previews.py` 复用下载清单的固定官方版本和逐文件哈希核验，发现模型入口、
递归检查普通格式依赖，排除场景、碰撞件、依赖网格和字节相同的重复表示；USD 由原生
resolver 检查层与材料引用。597 个文件中有 388 个模型文件条目，排除 282 个，留下
106 个候选入口。此处按模型入口计数，不把每张纹理或碰撞网格算作独立资产。

本机 106 个候选中 **74 个六视图预览通过、32 个失败**，通过项为 USD 31、GLB 18、
URDF 16、MJCF XML 4、OBJ 4、STL 1。新索引位于
`assets/genesis/clip_non_robot_v1/index.json`，包含 **444×512** 归一化向量。
原四资产 `v1` / `clip_v1` 继续可用；使用新库时明确传入新索引路径。

```bash
# 本机已生成，无需再次执行。重建时两个输出目录都必须换成尚不存在的新目录。
.venv/bin/python -m pip install -r self_improving/sim_adapters/genesis/requirements-library-preview.txt
.venv/bin/python self_improving/sim_adapters/genesis/build_library_previews.py \
  --download-manifest assets/genesis/non_robot_v1/download_manifest.json \
  --output-dir assets/genesis/non_robot_previews_v1 --workers 2

.venv/bin/python self_improving/sim_adapters/genesis/clip_select.py build-index \
  --asset-index assets/genesis/non_robot_previews_v1/asset_index.json \
  --output-dir assets/genesis/clip_non_robot_v1

# 每次查询使用新输出目录；只输出资产绑定。
.venv/bin/python self_improving/sim_adapters/genesis/clip_select.py select \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --query "我要一个黄绿色的芒果。" --top-k 3 \
  --vlm-config configs/llm.yaml \
  --output-dir data/genesis_history/asset_selection/mango_new
```

预览入口需要已安装的固定 Genesis checkout，额外 USD 依赖钉为 `usd-core==26.8`。
每资产使用隔离进程、默认两个并发和 180 秒超时。仅调用 Genesis 原生格式解析器与
CPU Rasterizer，保持原始比例/坐标系，禁用碰撞构建和物理步进，不做材质烘焙。
薄片从某视角看不到、依赖缺失或材质不可解析时保留失败；不补假纹理、不用灰色代替
作者材料。32 个失败分为：16 个 MDL 引用缺失、3 个 USD 组合错误、1 个纹理缺失、
3 个 OBJ 的 MTL 缺失、9 个视角分割为空。完整逐项原因保存在 `assets/*.json`。
新库的几何证据限于原生解析、范围与非空留边检查，没有套用四资产的 1 µm 几何验收结论。

`non_robot_previews_v1` 保存独立资产记录、六图、拼图、逐进程日志、源清单引用及哈希；
总状态 `partial`、CLI 退出 2 表示有候选失败，后续 CLIP 只编码已通过项。
调试用 `--only` 子集不构成完整生产索引。`asset_library.verify_preview_index` 会重新检查
发现集合、依赖、源文件/图片哈希和六视图绑定；查询不启动 Genesis、不重编码资产图片。
新模型绑定增加 `source_root`、`model_format`、`source_inventory`，真实入口可为
GLB、URDF、USD 等，不再限于 `model.xml`。这些是独立绑定字段，不修改对象 schema。

实际验收及每一步中间产物见
[全库接入验收](../../../data/genesis_history/asset_selection/non_robot_acceptance_v2/README.md)。
“我要一个黄绿色的芒果。”、“一个黄色塑料碗”、“绿色垃圾桶”均命中 Top-3 并正确选择，
真实新资产/缓存测试 **4 passed**；黄色碗保留颜色不完全确定的差异。
原六查询在新库上 **7 passed**，四个正例正确，锤子和米老鼠杯拒绝。
本轮仓库 pytest **356 passed**，Genesis 离线 **132 passed / 22 skipped**，
OpenXSim 离线 **101 passed / 1 skipped**；CLI 帮助和 ruff 通过，没有新增物理实验。

本机 CPU 首次预览构建 396.57 秒；CLIP 建索引 126.10 秒，其中图片编码 83.67 秒。
查询依赖扫描优化后，芒果/黄碗/绿桶总耗时分别为 10.36 / 7.98 / 9.00 秒，
各一次 VLM 请求；同进程芒果缓存命中 4.10 秒、零请求。每查询前后两次完整核验合计
约 4 秒，另列为 `timings_s.integrity`。扫描只跳过 OBJ 几何行的引用分词，保留逐文件
哈希核验。优化前证据仍保存在 `non_robot_acceptance_v1` 和 `non_robot_original_cases_v1`。
新预览包约 46.4 MB，CLIP 索引/数组约 1.18 MB；权重复用原缓存。以上是本机实测，不承诺固定时延。

```bash
GENESIS_LIBRARY_CLIP_REAL=1 \
GENESIS_LIBRARY_CLIP_OUTPUT=data/genesis_history/asset_selection/non_robot_acceptance_new \
  .venv/bin/python -m pytest \
  self_improving/sim_adapters/genesis/tests/test_real_library_clip.py -q
```

真实验收入口额外捕获文本向量、全部 444 个视角分数、无鉴权头的实际请求体及候选 PNG；
普通 `select` 仍保存标准五类结果。预览、检索与绑定均不代表物理通过。

## 官方资产索引与预览（独立准备入口）

```bash
.venv/bin/python self_improving/sim_adapters/genesis/build_official_index.py \
  --output-dir assets/genesis/v1

.venv/bin/python self_improving/sim_adapters/genesis/build_official_index.py \
  --output-dir assets/genesis/v1 --verify-only
```

构建目录必须尚不存在；第二条命令只读检查已有索引，不下载、不渲染。
下载范围固定为 `Genesis-Intelligence/assets` 的 `mug_1`、`cup_2`、`apple_15`、
`donut_0`，revision 为 `4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc`。
依赖当前 Genesis 环境及 `huggingface_hub`；只下载四个子目录，每个文件对照固定版本
Hub 元数据核对大小及 Git blob SHA-1 / LFS SHA-256，随后记录本地 SHA-256。
原始文件位于 `sources/<asset_id>/`，不覆盖旧的 `data/` 资产或实验结果。

`assets/<asset_id>.json` 在结构检查后、渲染前保存，记录 `model.xml` 入口、依赖、
外观/碰撞部件、源文件哈希和状态。没有类别、语义描述或内腔能力推测。
`asset_index.json` 汇总记录和完整文件哈希清单；它**不是** `scene_gen` 的
`--asset-catalog` 输入，不解决自然语言运行缺少语义 catalog 的问题。

预览直接由 Genesis 原生 MJCF + CPU Rasterizer 生成，不使用 OpenXSim、不转换 URDF、
不步进物理、不沉降、不改资产比例。`model.xml` 的独立碰撞部件原样保留，不能改用
`output.xml`。四个资产均有 32 个碰撞部件；前三个各一个外观部件，donut 有十个。
Genesis 会另外创建碰撞可视副本，外观几何核对按 XML 的源 mesh 引用识别部件，
不使用 `mesh.color` 的 alpha 猜测角色。真实外观范围和双向最近顶点误差必须 ≤ 1 µm。

每个 `previews/<asset_id>/` 包含六张 `512×512` 的 `view_*.png`（四个 30° 斜视方位、
俯视、仰视）、`contact_sheet.png` 和 `preview_result.json`。固定白色背景、灯光和
35° 视场角，按原始外观包围球计算相机距离并留边，分割图用于非空和至少四像素边界检查。
`overview.png` 是四个资产首视角的总览，仅全部成功时生成；不生成视频或网页。

单资产失败继续检查其余资产，保存逐资产原因和 `build_report.json`，整体非零退出。
`--verify-only` 成功只代表文件和记录绑定完整，须另看 `build_status`，不能把完整的失败包
视为成功构建。文件清单不构成数字签名，不能抵抗攻击者同时重写所有记录和哈希。

状态含义必须分开：结构索引完成 ≠ 预览通过 ≠ 检索匹配 ≠ 物理通过。
本入口的 `physics_status` 始终为 `not_evaluated`。Genesis 构建可能提示求解器参数钳制，
这不产生物理证据；下方已有物理入口仍独立负责实际运行和验收。

离线测试包含 `tests/test_official_index.py`；四资产真实下载及预览验收独立开启，
测试会把 `Scene.step` 替换成直接失败的函数，保证未用物理步进生成预览：

```bash
GENESIS_INDEX_REAL=1 GENESIS_INDEX_OUTPUT=assets/genesis/acceptance_new \
  .venv/bin/python -m pytest \
  self_improving/sim_adapters/genesis/tests/test_real_official_index.py -q
```

2026-09-05 本机实测：正式产物位于 `assets/genesis/v1/`，四资产全部预览通过，
24 张视图、四张拼图与总览已生成并人工查看，源文件/图片完整性检查通过。
最大外观包围范围误差为 `4.81e-9 m`，最大双向顶点误差为 `9.43e-9 m`，均小于 `1e-6 m`。
四个模型各有 32 个碰撞部件，外观部件分别为 1、1、1、10。单独的新目录
`assets/genesis/acceptance_v2/` 完整真实 pytest 为 **1 passed**，覆盖四资产并禁止
`Scene.step`。上述 `acceptance_v2` 和早期 `probe_01` / `probe_02` 已按用户后续清理要求删除；
正式 `v1` 保留，历史验收结论不变，旧完整证据不再保留。
首次 v1 的构建已通过，但测试后置检查发现相对路径 bug；修复并增加回归后，v1 只读核验通过，
acceptance_v2 从头验收通过。固定配置不承诺不同进程的 Rasterizer 图片字节完全相同。
本轮仓库回归 **356 passed**，OpenXSim **101 passed / 1 skipped**，Genesis 适配器离线
回归 **57 passed / 11 skipped**（真实预览另行显式执行如上）；CLI 帮助、ruff 与 diff 检查通过。
本轮没有重新运行物理实验，下文物理验收结果仍是原先的独立证据。

## CLIP Top-K + 单次 VLM 选资产（独立入口）

`clip_select.py` 接收一句完整单物体描述，直接编码中文，不调用解析或翻译模型。
只输出本次候选绑定，不改对象 schema、解析提示词或文本 transport，也不接场景求解、
OpenXSim、Genesis 运行、物理验证或终态渲染。颜色、印花和形状参与选择；不生成或修改资产。

在已有仓库环境安装此入口的可选依赖：

```bash
.venv/bin/python -m pip install -r self_improving/sim_adapters/genesis/requirements-clip.txt

.venv/bin/python self_improving/sim_adapters/genesis/clip_select.py build-index \
  --asset-index assets/genesis/v1/asset_index.json \
  --output-dir assets/genesis/clip_v1

.venv/bin/python self_improving/sim_adapters/genesis/clip_select.py select \
  --clip-index assets/genesis/clip_v1/index.json \
  --query "一个印有米老鼠的杯子" --top-k 3 \
  --vlm-config configs/llm.yaml \
  --output-dir data/genesis_history/asset_selection/mickey_cup_01
```

例中的 `clip_v1` 已在本机生成，建索引复现时必须换新目录；查询目录也必须尚不存在。
CLIP 依赖独立于旧 `vlm` extra（后者面向 Qwen，限制 Transformers <5），不要把两套约束
当作同一环境安装配方。本机测试环境为 Python 3.12.3、Transformers 5.3.0、
Torch 2.14.0+cpu、NumPy 2.5.2、Pillow 11.3.0；未在 Python 3.11 或 CUDA 实测本入口。

固定模型为 `OFA-Sys/chinese-clip-vit-base-patch16`，revision
`36e679e65c2a2fead755ae21162091293ad37834`。使用 Transformers 原生
`ChineseCLIPModel` / `ChineseCLIPProcessor`、配套预处理、`eval()` 和推理模式；
有 CUDA 用 CUDA，否则 CPU。模型权重默认在 `.cache/genesis/model_weights/chinese_clip/`，
可用 `--weights-dir` 覆盖；不训练模型。索引记录模型、revision、实际设备、Transformers、
Torch、Pillow、预处理配置、词表哈希和文本上限；版本变化要求重新建索引。
完整 tokenizer 输入超过模型位置上限会报 `query_too_long`，不静默截断；本机上限为
512 tokens，包含特殊 token。它不是字符数上限。

建索引前后按来源复用 `build_official_index.verify_index` 或新库的
`asset_library.verify_preview_index`。只编码 `preview_passed` 资产的
六张 `view_*.png`，不编码拼图或总览。归一化 float32 向量保存为 `vectors.npy`，
`index.json` 绑定每行资产 ID、视角、图片哈希、源包哈希和数组哈希。官方资产包保持封存。
每次查询读取和校验现有图片/源文件，但不重新渲染或编码图片。文本向量与图片向量归一化
点积得到余弦相似度；每资产取六视角最大值，按分数降序、同分按资产 ID 排序。
`--top-k` 默认 3，只允许 1–5，实际取 `min(K, 可用资产数)`；保存每个入选资产的全部
六视角分数，不使用 softmax 百分比或校准不明的通过阈值。

VLM 使用现有配置加载器解析的 `gpt-4o` / `chat` profile，可用 `--profile` 指定。
平台 `vision_request.py` 独立组织 Chat Completions 的 `text` / `image_url` 多图输入，
每个候选给两个不同的最高分视角，512×512；K=3 时一次请求六张图。
图片去除元数据后以 PNG data URL 发送，文本只包含原始描述和中性候选编号；不发送资产名、
路径或 CLIP 分数。编号按资产 ID 顺序分配，候选展示不携带检索排名。
提示词版本 `genenv.asset_visual_choice.v1` 要求先判断类别，再比较外观；图片上的文字
不是指令。允许类别合理的近似外观，并记录差异；无合理同类或看不清时拒绝。
返回对象只接受 `status`、`candidate_id`、`reason`、`visible_differences`，拒绝重复键、
额外字段和越界编号。模型不能提供资产路径。

`--timeout-s` 默认 60，独立覆盖旧文本 profile 的超时；每次未命中最多一次请求，
不使用旧 profile 的重试次数，不跟随重定向，不补图、不扩大 K，也不回退 CLIP 第一名。
合法的选择和拒绝写入 `.cache/genesis/asset_selection_cache/`（可用 `--cache-dir` 覆盖）；
错误不缓存。缓存键绑定完整查询、K、索引及其中的图片/向量/源文件哈希、编码器配置、
VLM 有效配置和提示词内容/版本。凭据不进入键或证据，轮换凭据本身不使缓存失效。
命中缓存仍校验完整性和文本、计算候选，但零 VLM 请求；源包变化需重新建索引，
旧索引直接报错。有效的新索引或有效模型配置变化会缓存未命中。
权重、向量、查询缓存、查询输出目录必须相互分离，不能写入任何封存官方包。

同进程可重复调用函数，默认复用已加载的 CLIP 模型；不新增常驻服务或向量数据库：

```python
from self_improving.sim_adapters.genesis.clip_select import select

report = select(
    "assets/genesis/clip_v1/index.json", "黄色杯子",
    "data/genesis_history/asset_selection/yellow_cup_new", vlm_config="configs/llm.yaml", top_k=3,
)
assert report["status"] in {"selected", "rejected", "error"}
```

每次查询保存 `request.txt`、`retrieval_result.json`、`vlm_selection.json` 和
`run_report.json`；只有 `selected` 才生成 `selected_asset.json`。
绑定从可信索引取得资产 ID、官方记录引用、真实模型绝对入口、完整源文件哈希，
并保留查询哈希、选择键、理由和差异。它只表示“本次选中的候选”，不表示全部描述、
几何条件或物理要求满足。输出目录已存在或违反封存目录保护时，在写入前拒绝。
报告区分模型加载、文本编码、相似度、完整性核验、VLM 和总耗时，并记录实际设备、模型复用、
VLM 调用次数、缓存和产物哈希。CLI 退出码：选中 0、拒绝 2、错误 1。
哈希清单用于本地完整性检查，不是防止同时重写清单及全部数据的数字签名。

### 选资产实测（2026-09-05）

本机 `assets/genesis/clip_v1/` 为 4 资产、24×512 数组；CPU 图片编码
**1.82 秒**，首次建索引含模型加载/下载 **34.10 秒**。在线第三方 Chat Completions
六图输入已实测兼容。原证据目录 `data/genesis_history/asset_selection/acceptance_v1/` 已按用户后续
清理要求删除，小型结果摘要保留在 `output/cleanup_receipt.json` 的 `previous_selection_summary`；
以下表格是历史测量，没有修改预期答案或放宽断言。

| 查询 | Top-3（按 CLIP 分数） | VLM 结果 | VLM / 总耗时 |
| --- | --- | --- | --- |
| 苹果 | apple_15, donut_0, cup_2 | apple_15 | 5.02 / 11.34 秒 |
| 甜甜圈 | donut_0, cup_2, apple_15 | donut_0 | 5.14 / 5.56 秒 |
| 黄色杯子 | cup_2, mug_1, donut_0 | cup_2；注明水果图案 | 5.59 / 6.02 秒 |
| 带把手的杯子 | mug_1, cup_2, donut_0 | mug_1 | 4.38 / 4.81 秒 |
| 锤子 | mug_1, donut_0, apple_15 | 拒绝，无绑定 | 5.82 / 6.24 秒 |
| 印有米老鼠的杯子 | mug_1, cup_2, donut_0 | 拒绝，未声称确认印花 | 4.69 / 5.12 秒 |

第一条查询包含新进程模型加载 5.87 秒，后续查询复用模型；文本编码 0.015–0.025 秒，
相似度计算约 0.0002 秒。六条各一次 VLM 请求。苹果同进程缓存命中总耗时 **0.35 秒**、
VLM 调用 **0 次**。这些是这批数据上的单次测量，不是延迟保证或广泛模型质量结论。

离线测试覆盖数学、六视角去重、同分、K、超长文本、图片/数组/源包篡改、输出保护、
VLM 非首名选择/全部拒绝/差异/非法 JSON/越界/超时、缓存命中与失效、密钥保护。
真实验收默认跳过，显式开启以下命令会先执行全部六条，再检查答案，失败也保留：

```bash
GENESIS_CLIP_REAL=1 GENESIS_CLIP_OUTPUT=data/genesis_history/asset_selection/acceptance_new \
  .venv/bin/python -m pytest \
  self_improving/sim_adapters/genesis/tests/test_real_clip_select.py -q
```

可用 `GENESIS_CLIP_INDEX` / `GENESIS_CLIP_CONFIG` 覆盖索引和在线配置路径。
该真实测试还验证重复苹果零次请求；本次 **7 passed**。不调用 Genesis、不新增真实物理实验。
本轮最终离线回归：仓库 **356 passed**；OpenXSim **101 passed / 1 skipped**；
Genesis **107 passed / 18 skipped**（其中新增选择离线测试 50 项、在线选择默认跳过 7 项）。
两个子命令帮助、ruff、`git diff --check` 及日志/证据密钥扫描通过；官方 v1 只读完整性仍通过。
另用真实 CLI 复用上述苹果缓存，零 VLM、零资产图片编码，新进程总耗时 7.77 秒，
其中模型加载 7.40 秒；缓存命中不会消除新进程的 CLIP 模型加载成本。
接口格式依据 [OpenAI 图像输入文档](https://developers.openai.com/api/docs/guides/images-vision)，
编码器依据 [模型说明](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) 与
[Transformers ChineseCLIP 接口](https://huggingface.co/docs/transformers/model_doc/chinese_clip)。

### 清理后的单句观测测试

用户要求清理旧输出并查看每一步中间产物后，删除重复预览和旧运行目录，保留官方
`genesis_assets/v1`、`genesis_assets/clip_v1` 及权重。本次句子为“我要一个黄色的、印有水果图案的杯子。”，
Top-3 是 cup_2（0.503974）、mug_1（0.426066）、apple_15（0.365651），VLM 选中 cup_2，
可见差异为空；未命中缓存，一次 VLM 请求，CPU 总耗时 11.30 秒（模型加载 5.89 秒、VLM 4.96 秒）。
结果在 `data/genesis_history/asset_selection/yellow_fruit_cup_01/README.md`，按顺序展示输入、索引校验、
token/文本向量、全部 24 视角分数、Top-3、实际六图请求、原始响应、校验和可信绑定。
这些额外产物由临时观测脚本捕获既有函数和实际 HTTP 请求，不改生产代码，额外记录耗时计入总时长。
`trace_manifest.json` 绑定扩展产物，标准 CLI 的输出契约仍如上。未保存鉴权头，未启动仿真。

## 已有资产场景的独立物理入口

```bash
.venv/bin/python self_improving/sim_adapters/genesis/validate_asset_scene.py \
  --scene-dir 'output/桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。' \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --fixed-object table_1 --profile baseline
```

此入口直接读取已经完成的 01/02，不要求 OpenXSim 编译包，不调用模型、检索、布局求解
或渲染。`--fixed-object` 可重复；仅地面自动固定，声明 on 的源对象不能固定。
当前支持原生 MJCF、GLB 无关节刚体；原有提取/构建默认仍止于 02。
`baseline` 为 CPU、seed 0、4 ms × 1000 步，`half_dt` 为 2 ms × 2000 步；均为 4 秒。
Newton、50 次迭代、tolerance 1e-8、substeps 1、关闭休眠；实际求解参数另行记录。
GLB 使用原生非凸碰撞、无简化或修复，密度 600 kg/m³、摩擦 1.0 是固定版本默认仿真假设。
MJCF 保留原文件材料与惯量来源，记录实际质量、各 link 惯量、摩擦和自由度。

输入锁定资产/依赖、原文、图/布局、角色、物理参数与阈值。先在独立零步进、无相机的
参考场景复现 02 视觉网格摘要，再逐顶点核验动态原生加载（误差 ≤1e-5 m）；保持原生姿态，
只施加原布局平移。刚性附属 link 可无独立自由度和零惯量，自由主体必须有有效惯量；
不会通过固定悬空、锁关节、预沉降、缩放或替换资产制造成功。

初态检测不积分时间，接触力标为不可用。固定版本 `detect_collision()` 返回物理存储
槽的索引，未按接触裁剪后的排序重排；因此只用它触发检测并核对数量，全部接触字段由
官方 `collider.get_contacts()` 统一读取。每步的双方接触力与独立净接触力观测交叉核验。
初态及每次真实步进写一行，完整基线/半时间步分别为 1001/2001 行；异常保留部分证据。

终末 0.5 秒检查位移 ≤1 mm、转角 ≤0.5°，静止以位姿演化判定：净漂移速率 ≤2 mm/s、窗口内相对均值最大偏移 ≤1 mm、净转速 ≤1°/s；辅以扫掠半径加权的有效速度 `max(|v|, r·|ω|)`，阈值按每个 profile 从 `g·dt` 派生，且要求连续 5 步超限才算动。掉接触率是独立判据（≤0.05）；
声明父对象提供向上合力 >1e-6 N 的采样比例须 ≥0.8。初态及全程穿透 ≤1 mm。
使用目标当前局部坐标验证完整源视觉投影与实测支撑面，保持 2 cm 余量；支撑面倾斜
超过 0.5° 则首版模型不支持。直接父子接触合法，终末图外接触失败。左右前后、远近
沿用 02 定义，仅验证原文明示关系；模型区域偏好不作为硬条件。

`03_physics` 保存输入快照、physics_input、asset_physics_report、initial_state、trace、
final_state 和 physics_result。成功总状态为 `physics_passed`，失败为 `physics_failed`，
退出码分别为 0 / 2（物理失败）/ 1（输入、加载或采集错误）。`simulation_executed`、
实际步数和最后完整采样区分未运行、部分运行与完整失败；缺失阶段不伪造文件。
重跑保留 01/02，清空旧 03/04 及旧诊断索引；独立物理入口的终态渲染保持 `not_run`。
物理验证视频归入 `03_physics/video/`；失败终态图片归入
`03_physics/diagnostics/final_state/`。只有物理验证通过的结果才能进入 `04_final_render`。
物理未通过时，04 必须为空且状态为 `not_run`，即使用户请求查看失败场景，也只生成
03 内的诊断材料。`TaskOutput` 在完成物理阶段、封存和读取时拒绝违反这条规则的任务。

真实重复测试通过 `TaskOutput.copy_for_physics` 建独立任务副本，保持所有 01/02 字节；
原 owner 字节在 `.scene_source_owner.json` 中按旧输入快照哈希校验，新任务另有自身归属。

```bash
GENESIS_ASSET_PHYSICS_REAL=1 \
GENESIS_ASSET_PHYSICS_OUTPUT=data/genesis_asset_physics/acceptance_new \
.venv/bin/python -m pytest \
  self_improving/sim_adapters/genesis/tests/test_real_asset_physics.py -q
```

输出目录必须不存在。三个基线、半时间步、接触校准、动态中间层三层夹具及五种反例均
保留证据。正常四资产失败会使真实 pytest 失败，不使用 xfail 或放宽预期。
官方微波炉包含旋转门关节，当前不支持；三层物理正例是独立生成、实测的 GLB 刚体夹具，
不是桌子→微波炉→苹果的物理通过证明。具体结果与限制见 [验收记录](PHYSICS_VALIDATION_EVIDENCE.md)。

## 旧编译包物理入口

入口是本目录的 `validate_physics.py`，不修改 OpenXSim、`scene_gen/` 或现有渲染器。

```text
已有 Genesis compile_manifest.json
  → 完整性检查 → CPU 刚体运行与判定
  → 失败：保存轨迹和原因，停止
  → 通过：只更新动态对象位姿 → 已有编译器重编译 → 原样执行 runtime_command
```

## 运行

需要当前 `.venv` 中的 Genesis、NumPy、SciPy、trimesh、MuJoCo、Pillow，以及现有渲染器依赖；Genesis checkout 必须是 `0e74bf392781884ccad765c3f344419c86b872ca`。仅资产转换使用 MuJoCo 解析 MJCF，物理运行使用 Genesis。辅助脚本不会下载资产。

在仓库根目录准备三个官方资产测试包，再执行其中一个。以下输出目录必须尚不存在：

```bash
.venv/bin/python self_improving/sim_adapters/genesis/prepare_cases.py \
  --mug-dir data/genesis-official-assets/mug_1 \
  --output-dir data/genesis_physics/inputs

.venv/bin/python self_improving/sim_adapters/genesis/validate_physics.py \
  --compile-manifest data/genesis_physics/inputs/box_on_table/genesis/compile_manifest.json \
  --output-dir data/genesis_physics/box_run
```

另两个 case 为 `mug_on_table`、`box_in_mug`。只准备 Box 可加 `--case box_on_table`，不需要 mug 文件。敏感性测试用 `prepare_cases.py --dt 0.002` 创建另一套新输入，仍为 4 秒，不在验证器里覆盖 package 参数。

不是任何已有渲染包都能物理验证：package 必须事先声明 `metadata.genesis_physics.settings`、每个对象的 `bodies`、以及 `task.success` 中的 `support` / `inside` / `released` / `upright` 条件。缺失条件时失败，不按运行结果猜目标或内部区域。当前仅支持 Box、单 link URDF，无机器人和关节。

## 资产与判定

- 官方 `mug_1/model.xml` 转成单 link URDF，保留独立 visual、纹理和 32 个 collision 部件；MJCF 几何变换烘焙到各个 mesh。转换后逐点核对几何，误差上限 1 µm。`measurement.json` 保存原始文件和转换文件哈希；来源 revision 是官方示例声明，实际本地字节另行指纹绑定，不声称经过远端认证。
- 杯内区域在仿真前测量：向下射线探测杯底，逐个凸碰撞部件与内部棱柱做线性可行性检查，只有全部明确不相交才接受。该区域是保守子区域，不是完整杯腔；实际落点在杯内但越出该区域也会失败，不自动扩大区域。
- 默认 CPU、seed 0、`dt=0.004s`、1000 步、substeps 1、重力 `(0,0,-9.81)`；Newton solver、50 次迭代、tolerance `1e-8`、关闭休眠。约束 time constant 为 `0.05s`：低于 `2*dt` 会被 Genesis 静默钳到该下限并在最不稳定处求解，原先声明的 `0.001s` 正是如此，也是接触掉线极限环的来源。报告同时保存请求配置与实际各 geom 的 solver 参数。
- Box 密度 1000 kg/m³、摩擦 0.5；mug 摩擦和 solref/solimp 来自官方 MJCF，URDF 保留质量和惯量。实际加载质量、摩擦、碰撞数和自由度写入报告，固定体在 Genesis 中有效质量为 0。
- 从释放前开始记录全部步骤，不预沉降、不重置位姿、不在物理期间渲染。终末 0.5 秒最大位移 ≤ 1 mm、旋转 ≤ 0.5°，静止判据同上（位姿演化为主、有效速度为辅）。默认从最低碰撞顶点上方 10 mm 释放；`--at-rest` 改为贴面释放，用于源位姿本就静置的场景 —— 验收一字未改（穿透仍逐帧检查），只是不再包含一次该场景根本不存在的落地冲击。这类运行不是落体测试，`verify_evidence` 会拒绝它作为下游资产证据。
- 终末接触正确目标且有向上合力的比例 ≥ 0.8；完整对象包围角点投影须在桌面支撑区内。全程所有接触（含支撑对）的最大穿透 ≤ 1 mm。
- 入杯 Box 初始完整底面高于杯口，终末八个顶点全部在 mug 局部测量区域内，且全程不接触桌面。缺行、非有限数、接触读取异常均失败。

## 输出和成功含义

`trace.jsonl` 保存释放前和每一步的位姿、速度、接触位置/法向/穿透/力及 geom/link 对；`physics_result.json` 保存配置、测量和失败项，以及输入 manifest、scene、package 和轨迹的哈希关联。失败时可能只有部分轨迹，不能当完整运行证据。

仅物理通过后生成 `settled/genesis/`。新 package 保持资产、缩放、静态标记、任务条件及物理参数不变，只替换动态对象位姿。重编译后重新验证 package/scene 绑定与最终位姿，再执行编译器返回的渲染命令。

`physics_status` 和 `render_status` 分开；渲染失败不能抹掉已有物理结果，也不能报告整体成功。现有视频是“物理验证后的终态展示”，不是运动过程视频。现有 renderer 仍是零物理步的展示工具，OpenXSim L2–L4 仍未评估。

## 测试

离线判定和攻击测试（不需要 Genesis 启动）：

```bash
.venv/bin/python -m pytest self_improving/sim_adapters/genesis/tests -q
```

真实验收默认跳过，显式开启后会执行三个正常场景各三次、三个半时间步运行以及四个反例；完整输出目录必须新建。正常场景不通过会让 pytest 失败，不使用 xfail 或放宽阈值：

```bash
GENESIS_PHYSICS_REAL=1 \
GENESIS_PHYSICS_OUTPUT="$PWD/data/genesis_physics/acceptance_new" \
.venv/bin/python -m pytest \
  self_improving/sim_adapters/genesis/tests/test_real_physics.py -q
```

反例覆盖固定悬空、关闭碰撞、初始深穿透、用固定 Box 封住官方 mug 杯口；后两项必须真正运行物理，前两项在配置门控直接拒绝。所有反例都不得渲染。生成资产、轨迹、图片和视频仅留在被忽略的 `data/` 中，不进入源码仓库。

## 本机实测状态（2026-09-05）

代码入口已实现，但三个正常场景的验收**尚未全部通过**。最终真实运行留在本机 `data/genesis_physics/acceptance_final/`；这些大文件不随 Git 分发。

| 正常场景 | 4 ms，重复三次 | 2 ms，同为 4 秒 | 全程最大穿透（4 ms / 2 ms） |
| --- | --- | --- | --- |
| Box 落桌 | 3/3 物理与渲染通过 | 通过 | 0.354 / 0.158 mm |
| 官方 mug 落桌 | 3/3 物理与渲染通过 | 通过 | 0.732 / 0.339 mm |
| Box 入 mug | 3/3 被物理门控拒绝 | 拒绝 | 5.202 / 4.221 mm |

每种正常输入的三次轨迹 SHA-256 一致。入杯场景稳定、接触比例通过，但穿透超过 1 mm，且完整 Box 越出了预先测量的保守棱柱；这不等于已经证明 Box 掉出了实际杯腔。不得把它标成正常通过，也没有为它生成终态渲染。

四类反例全部被拒绝且未渲染：深穿透反例记录初始 10 mm 重叠；封口反例实际接触 `lid`，不是 mug 内底，支撑对象和包含检查均失败。真实 pytest 汇总为 **8 passed、2 failed**（重复三次在一个参数化测试项内完成），两个失败项均是正常入杯案例。

成功渲染保持 640×480、12 FPS、120 帧及 120 个互异帧，沉降姿态与渲染姿态在已有 `1e-6` 门槛内一致；已查看 mug 的终态图片。仓库回归 324 passed；OpenXSim 回归 101 passed、1 skipped（旧 can/cabinet 真实渲染未开启）；平台离线回归 29 passed、10 skipped（真实验收另行显式运行如上）。ruff 与 `git diff --check` 通过。未运行 RoboTwin/SAPIEN 回放，本次未改变该通道。

后续仍需单独解决入杯场景的数值穿透与保守区域适用性；本次不自动改时间步、扩大区域、放宽门槛或延长运行来制造成功。

## 文本构建与局部修复（text_repair_v1）

`construct_asset_scene.py` 读取已有 `01_obj` 绑定，创建新任务，完成按需碰撞预处理、
文本约束候选、真实沉降及局部子树修复。原任务和旧原生物理入口保持独立。
`03_physics` 保存全部失败和成功尝试及连续回放 MP4；只有所有必需物体通过且指定
`--render` 时，才生成 `04_final_render` 终态图片和相机环绕视频。
命令、冻结参数、原生碰撞资格检查和产物见 [文本构建指南](TEXT_REPAIR_PLAN.md)，
真实场景结果与流程回归分别见 [验收记录](TEXT_REPAIR_EVIDENCE.md)。

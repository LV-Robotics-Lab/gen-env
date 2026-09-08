> 2026-09-08 更新：`validate_imported_scene` 不带 `--profile` 时默认执行已有场景图的
> 位置求解、干预式稳定化与双时间步验收；要求显式有限支撑图。显式 `--profile`
> 保留单次自由回放诊断方式。多支撑、堆叠、恢复及本轮真实证据见
> [场景图物理流程](POSITION_SOLVER.md)。下文历史导入示例不代表满足新支撑图契约。

# SimFoundry 场景图接入 Genesis

`import_simfoundry_scene.py` 把已有重建场景与已转换的标准 URDF 库组合成
`genenv.asset_scene.v2` 的 `scene_graph.json` 和 `scene_layout.json`，保留物体身份、
位置和完整朝向。它不调用模型、不重新摆放物体，也不执行物理步进。

## 输入与坐标

| 输入 | 用途 |
| --- | --- |
| `s11_sim/scene_objects_info.json` | `iter_*` 与 category/model 的对应关系 |
| `s14_og/reconstructed_og_scene.json` | 默认位姿、固定状态、地面配置来源 |
| 已导入的 `library.json` | `iter_*` 到内容绑定 asset_id 和自包含 URDF 包的映射 |
| `s12_physics/pb_scene_poses.json` | 显式指定 `--pose-format pybullet` 时的较早阶段位姿 |

OmniGibson 保存的 `state.registry.object_registry[name].root_link.pos/ori` 是世界坐标，
采用米、Z-up、XYZW 四元数；导入保持世界坐标轴，仅把四元数重排为 Genesis 的 WXYZ。
`12_stabilize_physics.py` 在输出 PyBullet 位姿前已经消除了惯量坐标偏移；不能再次扣除
质心。s14 加载 s12 后还会调整和沉降，因此默认读取 s14，而不混用两个阶段。
世界包围盒由原始网格顶点完整旋转和平移后计算，不只平移旧包围盒。

资产库必须匹配源 metadata 的摘要；类别、模型、对象 ID、包摘要和原始资产文件摘要
也须一致。输出复制标准包的全部已绑定依赖，以场景包为相对路径根。复制或搬迁输出
后，`verify` 和 `preview` 无需访问原 SimFoundry 源码、重建目录或原资产库。

## 命令

在仓库根目录运行，输出目录必须不存在：

```bash
.venv/bin/python -m self_improving.sim_adapters.genesis.import_simfoundry_scene import \
  --scene-dir data/simfoundry/fruits_da3_20260907 \
  --library-path assets/genesis/simfoundry_fruits_v2/library.json \
  --output-dir data/simfoundry_genesis/my_scene

.venv/bin/python -m self_improving.sim_adapters.genesis.import_simfoundry_scene verify \
  --scene-package data/simfoundry_genesis/my_scene

OMP_NUM_THREADS=2 .venv/bin/python -m self_improving.sim_adapters.genesis.import_simfoundry_scene preview \
  --scene-package data/simfoundry_genesis/my_scene \
  --output-dir data/simfoundry_genesis/my_scene_preview
```

`--scene-file <文件>` 可指定兼容的编辑器保存状态；默认不会寻找或提升 `_latest.json`。
`--pose-format pybullet` 默认读取 s12 位姿，此格式没有地面配置和速度，采用 z=0 地面，
并记录零速度。不要将 `settled_poses.json` 和其附加机器人条目冒充 s12 输入。
所有命令成功返回 0，输入、绑定或加载错误返回 1，不用退出码表示物理通过。

产物包括：

- `scene_graph.json`：节点与资产映射、环境、坐标约定。
- `scene_layout.json`：完整位姿、固定/动态意图、物理参数、局部及世界包围盒。
- `native_geometry.json`、`conversion_report.json`：几何摘要、输入哈希、导入及排除清单。
- `source_scene.json`、`source_metadata.json`：保留源文件的结构化内容。
- `assets/<object_id>/`、`manifest.json`：自包含 URDF 包和逐文件哈希。
- 独立预览目录：`overview.png`、`top.png`、`side.png`、`preview_report.json`。

## 当前支持边界

当前转换的是已有单刚体资产及地面。机器人自动列入 `excluded_objects`；未匹配的
其他物体会失败，只有明确传入 `--exclude-object <名字>` 才允许排除，并保留原因。
缩放不是 `[1,1,1]`、关节状态、自定义 USD 路径、场景级质量/摩擦覆盖及 visual-only
物体会明确拒绝；需要先扩展相应资产转换，不能静默丢弃差异。地面位置、朝向、
可见性和启用状态来自源场景。相机、灯光及背景设置只保留在源快照中，未迁移到预览。

源场景没有明确提供 `on/inside` 等语义关系，所以 `edges=[]`，并标记
`relations_status=not_provided_by_source`。对象 `support=null` 表示未知；不会从盘子下方
有框架或包围盒重叠推断已验证支撑。睡眠状态不转换成固定物体。源速度保留在布局中，
零步进预览不恢复速度或执行动态回放。

现有文本 `build_scene.py` 使用平移求解且会重新摆位；导入结果应通过本模块的
`preview` 加载，它明确应用 WXYZ 旋转和场景包相对路径。这里的 v2 图/布局是平台
初始场景产物，尚不是 `validate_asset_scene.py` 的 TaskOutput 输入，也不构成稳定核心
编译包。独立 `validate_imported_scene.py` 现可直接运行导入包，测量稳定性与接触链；
完整语义验收仍需要明确支撑关系。

## Fruits 为什么没有桌子模型

本机 s11 清单只有七个桌上物体，没有独立桌子资产；s3 的
`image_11_floor_info.json` 将支撑面识别为 `desk`，桌面作为坐标基准和支撑平面处理。
具体做法是对分割区域的深度点云进行 RANSAC 平面拟合，只保存类别、`origin` 和
`z_dir`；s4 据此将点云转换到统一世界坐标。该步骤没有导出桌面网格、纹理、有限边界
或桌腿；s12 直接加载内置 `plane.urdf` 作为碰撞支撑。
s14 的实际 `stage_info.json` 记录 `include_table=false`，最终源场景使用内置 floor plane。
所以 Genesis 中的棋盘格平面承接的是这份桌面支撑抽象，转换没有丢弃桌子模型。
上游 s14 在启用 `include_table` 且未启用 GS 背景时可以另加载 BEHAVIOR 的
`conference_table/qzmjrj`；那是额外引入的库资产，不是本次从视频重建出的桌子。

### 支撑面如何选择与定义

这里选择的是场景的基础承托面，不是给每个物体确定一个支撑父节点。参考帧可以由
`s3_ground.img_idx` 指定；设为 auto 时，候选帧先分割支撑区域、拟合平面，再按物体
覆盖、分离程度（高于平面的连通区域数量）、清晰度、支撑区域覆盖和平面内点比例评分，
并惩罚画面边缘截断。本次 15 个候选中 heuristic 选中索引 11，无选帧 VLM 调用；
候选评价中桌面覆盖约 76.29%，平面内点比例约 99.71%。这些是选帧指标，不是重建精度。

区域分割按配置类别顺序尝试 SAM3，本次为 desk → table → counter，类别内最高置信度
达到 0.5 后停止尝试后续类别；选中类别有多个 mask 时取面积最大的一个。没有 mask
时存在 VLM 建议类别和人工点击回退。随后将 mask 内深度经相机内参反投影成点云，
正式 s3 使用 RANSAC（距离阈值 0.01 m、每次 3 点、1000 次迭代）拟合平面；
自动选帧的预拟合使用 2000 次迭代。0.01 m 是拟合内点阈值，不是实测误差保证。

平面由相机坐标系中的 origin 和 z_dir 定义；origin 来自平面内点均值的投影，
法向按背离相机正 Z 方向的约定定向。s4 先减 origin 再旋转，保存 cam2world；
场景以该支撑面为水平坐标基准。因此本例中的 floor 表示桌面参考层，而非房间地板。
mask 用于选择拟合点，不会自动成为有限桌面碰撞网格。

原仓库还提供独立 auto_bg_reconstruction：移除前景并补全背景帧 → DA3 种子点云 →
深度监督 Gaussian Splat → 与 OG 对齐 → 场景资产目录。其 s7 将背景对象设为
visual_only，恢复外观不等于恢复碰撞支撑；本次 include_gs=false，未启用该路径。
具体脚本见 `external/SimFoundry/scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/`。

## 本机验证

Fruits 结果：`data/simfoundry_genesis/fruits_scene_v1/`；三视角与加载回执：
`data/simfoundry_genesis/fruits_scene_preview_v1/`。
7 个物体完成转换，`robot0` 记录为排除；真实 Genesis CPU 原生动态加载及三视角渲染
通过，物理步数为 0。完整旋转后的世界网格与源位姿计算值的最大双向顶点距离约
`3.4431e-8 m`，阈值 `1e-5 m`；视觉/碰撞、质量、质心、惯量和摩擦核验通过。
这仅确认格式与加载一致性；原来的七项单资产落体失败结论不变。

测试覆盖非对称网格旋转、非零质心、失配/缺失物体、过期资产、未支持覆盖、非法四元数、
文件篡改、输出保护和删除源目录后的包验证。运行：

```bash
.venv/bin/python -m pytest -q self_improving/sim_adapters/genesis/tests/test_simfoundry_scene.py
```

## 导入场景的物理验证与渲染

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m self_improving.sim_adapters.genesis.validate_imported_scene \
  --scene-package data/simfoundry_genesis/fruits_scene_v1 \
  --output-dir data/simfoundry_genesis/my_scene_physics
```

该独立入口复用原生 URDF 核验和 baseline 阈值，要求单刚体对象均动态、尺度为 1、
源速度为零，保留导入地面。非零速度恢复和固定物体语义尚未支持，遇到会明确拒绝。
现有文本场景物理入口及 TaskOutput 不变。运行时直接读取原始位置与旋转，
不添加落体高度、不预沉降、不修复资产。CPU、seed 0、4 ms × 1000 步、重力 −9.81 m/s²。

全程（含初态）穿透须 ≤1 mm；末段 0.5 秒位移 ≤1 mm、转角 ≤0.5°、速度 ≤0.01 m/s、
角速度 ≤0.05 rad/s。每步按接触对合计向上力，检查通向地面的支撑链；终末有效支撑
比例须 ≥80%。静止但没有接触或悬空物体之间相互接触均不能通过。接触力与 Genesis
每个物体的 net contact force 独立核对。观测关系只写报告，不回填为声明关系。

退出码 1 表示输入/执行/证据错误，2 表示完整运行后稳定性检查失败；若稳定性全通过，
因尚无声明支撑关系，整体仍为 `incomplete`、退出 3，不报告完整物理通过。

输出 `physics_input.json`、`loaded_scene.json`、`trace.jsonl`、初末状态和
`physics_result.json`；每 10 步保存一张原始 PNG，初态到终态共 101 张，再编码
`simulation.mp4`。视频核验实际总帧数和互异帧数；`diagnostics/` 保存真实终态三视角，
不增加物理步进。`verify_evidence(directory)` 可核验源包、冻结输入、产物哈希并用
持久轨迹重算结论。

本机结果位于 `data/simfoundry_genesis/fruits_scene_physics_v1/`，详见其 `README.md`。
1000 步完整完成；5 个物体通过本次稳定性检查，2 个失败：黄色梨初态即有 2.8128 mm
地面穿透，青色盘子终末角速度 0.05671 rad/s 超限。源场景、阈值和物理属性未修改。
真实视频 101 帧 / 101 互异帧，终态总览/俯视/侧视已生成；独立轨迹复判一致。
原七项单资产落体测试使用不同初态与朝向，本次结果不覆盖原失败记录。

验证：仓库 363 passed；转换与物理专项 33 passed，包含悬空/无接触/间歇支撑、
初态穿透、固定物体、位姿/速度失配与不完整轨迹攻击测试。

## 上游“物理稳定化”与动态验收的区别

当前固定版 `12_stabilize_physics.py` 使用 PyBullet，加载 plane.urdf、重力
-9.81 m/s²，将物体按重建姿态上移 5 cm 后沉降。实际鼠标 s12 的配置为
load_at_once=true；每个仿真步后将全部对象线速度、角速度清零，再比较相邻步
位姿变化，阈值为 1e-4 m 和 1e-3 rad。最多 10000 步退出，但达到上限
不会单独判为失败；尾部再执行 50 步反复清零速度并保存 pb_scene_poses.json，
正常执行到尾部即 StageResult(success=True)。load_at_once=false 的另一分支
逐个沉降并暂用固定约束，最后移除约束；本次鼠标没有使用该分支。

这使用真实碰撞求解来整理初态，但没有本项目的连续自由动态速度窗口、
声明目标接触比例、有限桌面覆盖及全程穿透数值门控，不能据 s12.success
宣称这些门控通过。第 14 阶段转入 OmniGibson 后还会仿真沉降（默认
settle_steps=100），保存场景；当前脚本同样没有上述通过/失败指标检查。
因此上游 s12/s14 执行成功和下游 Genesis 动态验收失败并不矛盾。


## 统一媒体入口的默认支撑模式（2026-09-08）

`reconstruct_media` 现默认沿用本页的原始场景导入：`--support-mode upstream`。
支持 `--reconstruction-scene` 直接导入已有结果，不调用模型和资产检索。
原始平面转为 Genesis Plane，保留可见性、位姿和启用状态；
预览输出在 02，03/04 不运行。`--support-mode retrieved` 保留原有限桌面流程。

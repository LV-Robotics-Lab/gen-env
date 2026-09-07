# SimFoundry 标准资产接入 Genesis

整场景的位置、朝向和资产引用转换另见 [场景图接入指南](SIMFOUNDRY_SCENES.md)。

本入口只迁移单刚体资产，不重建视频、不迁移原场景布局或机器人。输出是自包含 URDF
资产包：URDF、视觉网格/纹理、碰撞网格、`physics.json` 和来源/文件哈希清单。
Genesis 原生加载这些文件，不需要安装 SimFoundry、OmniGibson 或调用重建模型。

## 转换与加载约定

导入使用 `s11_sim/scene_objects_info.json` 和每个模型的最终 `<model>.urdf`，忽略
`_original`、`_with_meta_links` 变体。保持网格、尺度、质量、质心和惯量，不重新生成网格、
凸分解或估计物理参数。摩擦从上游清单保存到 `physics.json` 并在加载时显式应用。
质量/摩擦注明上游模型估计；并非实测物理属性。

固定 Genesis 的混合 URDF/MuJoCo 解析路径会丢失本次原始惯量的坐标旋转。转换器将
惯量等价表达到 link 坐标系：`I_link = R * I_inertial * Rᵀ`，保留质心，惯量 origin 的
rpy 置零。转换前后独立解析核对张量和几何，不改变物体的物理含义或松动误差阈值。
最初未规范化的包保留在 `simfoundry_fruits_v1`，仅作诊断；正式索引使用 v2 包。

运行时使用原生 URDF，禁用自动凸分解、简化、修复、惯量重算及坐标对齐。独立 URDF
解析结果与实际视觉/碰撞顶点双向距离误差 ≤1e-5 m，质量和统一坐标惯量相对误差 ≤1e-4。
首版标准包支持单 link、无 joint 的网格/box URDF；关节、多刚体、缺失依赖和不支持的
几何明确报错，不降级为代理资产。资产 ID 绑定内容、摩擦、类别和转换版本。

## Agent 命令

从仓库根目录运行；新构建或验证输出目录必须不存在。`--only iter_1` 可以只导入指定
物体，可重复使用。以下目录为新机器示例，本机已有结果见最后一节。

```bash
# 1. 导入包；不会执行物理验证。
.venv/bin/python -m self_improving.sim_adapters.genesis.import_simfoundry_assets import \
  --scene-dir data/simfoundry/fruits_da3_20260907 \
  --output-dir assets/genesis/my_simfoundry_library

# 2. 独立进程生成六视图，零物理步进；失败项保留记录。
.venv/bin/python -m self_improving.sim_adapters.genesis.import_simfoundry_assets previews \
  --library-path assets/genesis/my_simfoundry_library/library.json \
  --output-dir assets/genesis/my_simfoundry_previews

# 3. 只编码预览通过项，不按物理结果筛选。
.venv/bin/python -m self_improving.sim_adapters.genesis.clip_select build-index \
  --asset-index assets/genesis/my_simfoundry_previews/asset_index.json \
  --output-dir assets/genesis/my_simfoundry_clip

# 4. 联合索引只引用原库，不修改官方文件。
.venv/bin/python -m self_improving.sim_adapters.genesis.clip_select build-union-index \
  --clip-index official=assets/genesis/clip_non_robot_v1/index.json \
  --clip-index simfoundry=assets/genesis/my_simfoundry_clip/index.json \
  --output-dir assets/genesis/my_union_clip

# 5. 联合检索 → VLM 选择 → 选中资产的独立物理测试。
.venv/bin/python -m self_improving.sim_adapters.genesis.clip_select select \
  --clip-index assets/genesis/my_union_clip/index.json --query '一个黄色的香蕉。' \
  --vlm-config configs/llm.yaml --output-dir data/simfoundry_genesis/my_selection

# 可脱离检索单独测试标准包；路径替换为 library.json 中的 package。
.venv/bin/python -m self_improving.sim_adapters.genesis.validate_single_asset \
  --package assets/genesis/my_simfoundry_library/packages/ASSET_ID/asset.json \
  --output-dir data/simfoundry_genesis/my_drop_test
```

选择仍使用现有 gpt-4o chat 配置，图片发往 `--vlm-config` 指定服务，与重建阶段的 Gemini
配置不同。联合成员须使用相同的 CLIP 模型、revision、预处理、维数和归一化方法。
Top-3 在全库统一计算，每个候选使用自己的图片根目录；成员索引变化会使旧联合索引
校验失败，需在新目录重建。HTTP 与注入测试的选择缓存分开绑定。

现有 `extract_assets.py --clip-index <联合索引>` 也会走上述选后验证；失败阻止自动
场景构建。`selected_asset.json` 保留失败候选，错误状态不会偷偷选择次优项。
单资产验证是选择阶段的预检，不等于组合场景的第三阶段物理验收。

## 物理验证与结果

格式分派不区分来源：新增单刚体 URDF，沿用原生单刚体 MJCF/GLB 路径；不支持的格式、
固定 MJCF 或关节物体直接报输入/加载错误，不为落体测试擅自加自由关节。
只平移使碰撞最低点位于地面上方 1 cm，原生朝向和尺度不变，物体动态、地面固定。
CPU、seed 0、4 ms ×1000 步、重力 −9.81 m/s²，使用 baseline 求解设置；实际选项及
接触参数另存 `loaded_asset.json`。原生求解器可能对接触时间常数做钳制，回执记录实际值。

保留初态和每一步，共 1001 行轨迹；初态接触力为不可用。终末 0.5 s 检查位移 ≤1 mm、
转角 ≤0.5°、速度 ≤0.01 m/s、角速度 ≤0.05 rad/s，向上接触力 >1e-6 N 的比例 ≥80%；
全程穿透 ≤1 mm。不重置速度、不预沉降、不修复碰撞或调整源参数。

- `0`：物理通过，`final_render/final.png` 保存终态。
- `2`：完整物理测试失败，保存 `diagnostics/final.png`；选中资产仍保留，调用返回错误。
- `1`：输入、格式、加载、进程或证据错误；不报告物理通过。

`video/drop.mp4` 保存初态及每 10 个真实步骤的观测，25 fps、完整运行 101 帧；报告实际
总帧数、解码互异帧数。稳定后帧可以重复，不补帧制造变化。视频、轨迹、资产和配置均有
哈希绑定；下游使用通过项时重新核验报告与轨迹。渲染预览不是物理证据。

## 本机 Fruits 验收

正式标准包：`assets/genesis/simfoundry_fruits_v2/library.json`。
六视图：`assets/genesis/simfoundry_previews_v2/asset_index.json`。
CLIP 子库：`assets/genesis/clip_simfoundry_v1/index.json`。
联合索引：`assets/genesis/clip_union_simfoundry_v1/index.json`，74 官方资产 + 7 新资产 = 81 项。

7 项导入及六视图均通过；动态加载的几何、质量、质心和惯量核验通过。
7 项落体测试均完成 1000 步，但均因全程穿透超过 1 mm 未通过；香蕉另有终末角速度超限。
保持原始失败结论，不以全部物理通过为迁移成功的条件。完整结果见
`data/simfoundry_genesis/fruits_acceptance_v2/` 与本目录 `SIMFOUNDRY_EVIDENCE.json`。

真实 CLIP 联合查询黄色香蕉将 SimFoundry 香蕉排在首位。选择联调使用注入候选决定，
没有发起 HTTP VLM 请求；后续 Genesis 物理测试是真实执行，退出 2 并保留选中香蕉。
通过路径另有独立 MJCF 落体对照：该夹具显式带 3 mm 接触 margin，在同一验收阈值下通过；
该对照不属于 Fruits，也不用于替换或修复失败资产。

可移植性另做了真实检查：将梨的包复制到独立目录，通过 Python 文件访问审计禁止打开
原 SimFoundry 源码、重建产物和原标准资产库，仍完成加载与 1000 步；原物理失败结论不变。
回执位于 `data/simfoundry_genesis/portability_v1/portability.json`。

最终检查：根仓库 pytest 363 passed；Genesis 适配器 388 passed / 44 skipped，
其中需显式开启的真实单资产对照另跑 1 passed。相关 ruff、CLI help、依赖/源文件哈希、
场景指南同步及 diff 检查通过。可开启真实对照：

```bash
GENESIS_SINGLE_ASSET_REAL=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  self_improving/sim_adapters/genesis/tests/test_real_single_asset.py
```

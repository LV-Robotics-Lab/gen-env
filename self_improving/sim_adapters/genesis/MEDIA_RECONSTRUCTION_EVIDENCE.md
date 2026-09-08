## 2026-09-08：按上游平面转换已有鼠标重建

当前默认 `--support-mode upstream` 保留上游场景平面，不再自动检索有限桌子。
本次复用归档的 `鼠标_单图_native_003/01_obj/reconstruction`，未重新调用模型。
输出：`output/将鼠标重建场景按SimFoundry原始支撑平面转换为Genesis/`。

原始 s14 场景启用平面，未另存 ground_plane_info；转换为 Genesis z=0 可见平面。
唯一动态物体为 iter_0，位置和朝向来自原始 s14 状态。没有 support_0 或新增桌子。
自包含场景校验、原始文件快照和任务清单校验通过，三视图与环绕预览已生成；
视频 120 帧、120 个不同帧、12 FPS、10 秒。棋盘格为 Genesis 默认平面外观。
结果为 scene_built，physics_status=not_run，03/04 为空；不构成物理通过证据。
此处以下的有限支撑验收记录属于显式 retrieved 模式的历史证据。

# 单图/视频统一重建验收

记录日期：2026-09-07。

统一入口是 `python -m self_improving.sim_adapters.genesis.reconstruct_media`。它只接受
`--image` 或 `--video` 之一，输出固定为 `output/<任务名>/` 的四阶段 TaskOutput。
SimFoundry 的旧 `run.sh reconstruct --video-fpath ...` 入口仍保留。

## 已验证契约

- 单图转为规范 PNG，并固定单图模式、单帧采样和索引 0；不启用 Gaussian splat。
- 视频记录 ffprobe 元数据、完整解码帧数、逐帧摘要、互异帧和 15 帧采样索引。
- stage 3 保存支撑 mask、深度、内参、RANSAC 内点和相机坐标点；平台结合 stage 4
  保存世界坐标点、有限可见范围、边界 censoring 和 measured/inferred 轴证据。
- 支撑资产从现有 Genesis 六视图索引检索，经过语义/外观排序、有限水平顶面、
  可加载格式和哈希门控；失败不回退到无限平面或程序化薄板。
- `support_0` 是 fixed、collidable 的自包含 URDF；物理验证要求真实向上接触，
  且完整前景投影位于有限顶面内。
- 最多执行基线和三次白名单修复。物理未通过时 `04_final_render` 必须为空。

## 本机输入与实际运行

| 输入 | SHA-256 | 解码帧 | 互异帧 | 采样 |
| --- | --- | ---: | ---: | ---: |
| `test/鼠标.jpg` | `3a3b65d558c9fc7b61d95071ad7beaf2fd90853b292d6c9cb10ecbbdbd5cb476` | 1 | 1 | 1 |
| `test/鼠标视频.mp4` | `5c7aa9c809aaff9c5f4d3076502d99cba0f72e46754102184b9c5c9db8d88634` | 124 | 124 | 15 |

单图严格离线运行已真实完成 stage 1b–4。stage 3 识别 `desk`，有限世界 XY 范围为
`[-0.26196,-0.18152]..[0.18587,0.37158]` 米；mask 触及左/下边界，两个轴均标为
inferred。流式 stage 5 因同机另一进程占用约 16.5 GiB 显存而未启动。

视频严格离线运行真实解码 124 帧并按顺序采样索引
`0,8,16,...,112`，15 个样本均互异；DA3 因同一外部 GPU 占用 OOM。

在线任务未启动：把用户媒体发送给已配置 Gemini 服务需要明确外发授权。当前两个任务
均为 `execution_failed`，manifest 可验证，物理和渲染均 `not_run`，
`04_final_render` 为空；它们不能被描述为完成重建或物理通过。

## 验证

```bash
.venv/bin/python -m pytest -q self_improving/sim_adapters/genesis/tests/test_media_reconstruction.py
.venv/bin/python -m pytest -q self_improving/sim_adapters/genesis/tests/test_imported_scene_physics.py self_improving/sim_adapters/genesis/tests/test_simfoundry_scene.py
```

新增媒体契约为 9 passed；导入场景/物理回归为 33 passed。真实任务报告位于
`output/鼠标_单图/` 和 `output/鼠标_视频/`。


## 当前工作区重新实现（2026-09-07）

此次使用同一鼠标原始图片与视频，输入哈希与上表相同。原生接口探测结果在
`data/media_acceptance/native_probe_001/report.json`：文字 1.61 秒、识图 11.94 秒、
图片编辑 26.57 秒，三项均通过，编辑结果为 896×1200 PNG。用户已授权本次媒体
使用其提供的 Gemini 原生服务；上文“在线未启动”描述的是之前的运行。

恢复执行新增阶段产物哈希检查；同媒体配置变化会将旧输出移入任务的
`attempt_history/` 并使下游失效，不允许更换输入媒体后冒充原任务续跑。
物理通过后的独立渲染读取并核验最终轨迹，输出参考视角、多视角、输入对照及
120 帧环绕视频。参考视角匹配相机变换和垂直视场角，主点使用渲染器居中模型，
不宣称完整相机标定一致。环绕视频是静态终态展示，不是输入视频动作恢复。

真实非对称夹具渲染测试通过：零物理步、终态 WXYZ 姿态几何误差低于既定门槛、
120 帧视频且多个互异帧，原场景包保持原字节。它不构成鼠标物理通过证据。
当前工作区已开始真实在线验收，结果与阻塞见下文；历史成功记录不能代替本次验收。

真实首轮 `output/鼠标_单图_native_001` 的深度、支撑平面和世界坐标阶段成功，
整图生成后唯一鼠标被低于支撑面的筛选剔除。已停止该任务并保留失败产物。
`data/media_acceptance/mask_depth_probe/report.json` 使用相同 SAM3 与真实
图像宽高比验证：生成图 mask 中 23.499% 点低于桌面 5 cm，原图为 0%。
因此禁用整图生成，仅对物体裁剪图补全；未更改筛选或物理通过阈值。
新增上游空分割结果失败门控；后续验收使用新目录。

### 本轮实际结果与恢复（2026-09-07 22:13）

`output/鼠标_单图_native_002` 的接口预检三项通过：文字 1.84 秒、识图
10.90 秒、图片编辑 99.18 秒（896×1200 PNG）。阶段 1–4 成功；阶段 5
使用 `gemini-2.5-flash` 的物体检测连续三次读取超时，每次超时配置 300 秒。
独立原生请求使用相同原图和上游检测提示词，在 181.11 秒后报
`gemini_transport_error`，记录在 `data/media_acceptance/detection_native_diagnostic.json`。
流水线失败后保存四个成功上游阶段的哈希检查点，未进入网格生成、物理和最终展示。
两次失败任务的完整 manifest 均已重新验证，原目录未覆盖。

当前实现还补齐资产源文件、检索向量与预览图的恢复指纹，以及渲染重试前的
物体/场景检查点验证。失败渲染部分产物保存在 `03_physics/render_failures/`。
模型拒绝保留 `gemini_incomplete_or_blocked` 原因；显存失败注入验证保留上游
断点，损坏断点后禁止 `--skip-successful`。这些测试不冒充真实 GPU OOM 验收。

验证结果：根目录回归 378 passed；Genesis 全量回归先前为 603 passed、
47 skipped；后续改动相关媒体/恢复/物理测试为 30 passed、1 skipped。
真实用户视频元数据测试已在当前输入路径通过：124 帧、124 互异帧，
15 个采样索引为 0,8,...,112。它不证明这些帧已通过视频深度重建。
新增上游空结果门控的两个分支已隔离执行验证。Ruff 与 `git diff --check` 通过。

环境准备证据：`data/media_acceptance/runtime_readiness.json`；主 SimFoundry、
Hunyuan 和 FoundationPose 导入成功，DA3 主环境权重加载成功，Hunyuan 形状与纹理
权重已下载。独立 `da3` 环境的上游安装器当时仍在编译 `torch_cluster`，正式
鼠标任务使用已验证的主环境 DA3（`--env-da3 simfoundry`）。
FoundationPose 权重从 NVIDIA 官方数据集镜像取得，版本及哈希见
`logs/simfoundry/weights-foundationpose-source.json`。

后续首先恢复可用的相同 Gemini 服务检测调用，再用新验收目录执行：

```bash
venv/genesis/bin/python -m self_improving.sim_adapters.genesis.reconstruct_media \
  --image ../鼠标.jpg --name 鼠标_单图_native_003 --output-root output \
  --clip-index assets/genesis/clip_readme_subset/index.json --vlm-config configs/llm.yaml \
  --simfoundry-arg=--env-da3 --simfoundry-arg=simfoundry
```

同一新目录中断后可加 `--resume`；本轮代码和配置已变化，恢复旧任务将归档旧
产物并重跑受影响阶段，不应手改历史检查点来强制复用。图片成功后再运行视频
入口、检查实际送入上游的采样帧、完成两者同时间长度 `half_dt` 物理复验，
并记录最终视觉对照。**图片和视频目前均未达到全流程完成标准，URDF 场景包、
鼠标物理轨迹和最终展示尚未交付。** Dashboard 三个状态地址仍为 HTTP 404，
未同步任务状态。

### 新密钥重跑：native_003（2026-09-07）

用户要求使用新密钥重跑。密钥通过无回显输入仅保存在进程内存，替换本轮
客户端的 credential，未写入配置、命令参数或报告。任务目录为
`output/鼠标_单图_native_003`，总耗时 714.92 秒。

接口预检通过：文字 1.62 秒、识图 11.16 秒、图片编辑 50.09 秒。
SimFoundry 执行的阶段 1–8、10–14 均成功（静态流程未执行 articulation 9）；
上游总耗时 10 分 37 秒。鼠标检测返回且剔除列表为空，实际保存一个分割物体。
原图分割及物体补全图已目视检查，保留粉色外壳、滚轮与侧键。
带纹理网格、自动尺度/位姿、碰撞几何、USD 和 OG 场景已生成。
网格非闭合体，体积和物理参数仍属估计，不宣称实测尺度精度。

前景自包含资产包已导出：
`01_obj/foreground_assets/packages/simfoundry_pink_mouse_7915827ae4309660/urdf/bnjzzp.urdf`。
任务阶段为 `objects=passed, scene=failed, physics=not_run, final_render=not_run`。
失败为 `SelectionError: invalid_json`，发生在桌面选择阶段，见任务根目录
`failure.json`。上游 `s12_physics` 成功不是 Genesis `03_physics` 验收通过。

完全本地复核记录在 `data/media_acceptance/native003_support_local_diagnostic/report.json`：
前景转换、索引验证、候选排序、请求构造均通过；候选为桌面与碗，请求包含
1 张观测裁剪图和 4 张候选预览图。复核用停止替身阻止任何额外图片外发。
当前 `NativeClient` 返回普通文本，选择器使用严格 JSON 解析；提示词中的
示例不是合法 JSON，且 rejected 示例省略了校验器要求的 `candidate_id: null`。
这些是格式契约缺口。原始失败响应未落盘，因此不能确认具体是哪种格式错误；
Markdown 包装是待验证假设，不能当作已取得的原始响应证据。

单独在线复现桌面选择被自动审批拒绝，理由是具体支撑图及候选预览图向
`api2.aigcbest.top` 外发授权不足。未执行被拒绝的请求，待明确授权后复现。
修复方向：有效的 JSON 提示词、原生 JSON 输出约束、解析前保存无凭据响应
和失败证据；继续严格验证候选 ID 与状态，不通过默认挑桌面绕过模型拒绝。
本轮证据 manifest 已验证；尚未运行 Genesis 物理和最终渲染。

### 桌面选择格式错误已在线复现（2026-09-07 22:43）

用户明确授权发送一张桌面裁剪图和四张候选预览图后，独立诊断使用相同
请求构造与新密钥，5.06 秒返回。证据：
`data/media_acceptance/native003_support_online_20260907_224351/report.json`
及同目录 `response.txt`。模型选中候选 1（`dex_table_d3996872`），并正确
指出原图木纹桌面与黑色候选桌面的颜色、纹理、边角及可见结构差异。

响应是被 Markdown `json` 代码块包装的合法 JSON；原样交给当前严格解析器
稳定复现 `invalid_json`。仅在离线诊断中移除准确的外层代码块后，原有
`validate_selection` 全部通过，未修改字段、候选或通过阈值。这确认了该
格式不兼容模式；native_003 原始失败响应仍未保存，不能声称恢复了原字节。

本次仅完成诊断，未修改生产解析器，也未重跑 Genesis 物理或渲染。后续应
在模型响应边界规范化完整代码块（拒绝额外解释文字），配合原生 JSON
输出约束和合法 JSON 提示词，并在解析前保存无凭据响应以保留失败证据。

### 真实 Genesis 物理验证（2026-09-07）

进入物理部分时复用 `native_003` 的已校验重建与授权取得的桌面选择响应，
仅在独立执行脚本中准确移除 JSON 代码块并使用原有严格校验。没有重新
调用模型或修改通用 JSON 解析器。两个原始重建/失败任务的 manifest 保持有效。

`output/鼠标_物理_native003_001` 在第 0 步加载门控失败：
`support_0 is not fixed and collidable`。已定位并修复
`validate_imported_scene.py` 遗漏参数：调用 `standard.morph` 时传入
`fixed=obj["fixed"]`，保持碰撞启用及所有验收阈值。此前默认 `fixed=False`
会把声明固定的桌面加载成动态对象。

修复后在 `output/鼠标_物理_native003_002` 完成真实仿真，最终判定
`physics_failed`，`04_final_render` 保持空。四次记录均通过证据重算与哈希验证：

- baseline：dt=0.004，1000 步，4 秒；接触比例 1.0，线速度门控值
  0.02348 m/s、角速度门控值 0.49206 rad/s，稳定性失败。
- normal clearance：按既定最多 10 mm 抬升规则计算，无需抬升（changes=[]）；
  保留重复基线记录，结果相同。没有合格的替代桌面，未使用碗替代。
- half_dt：dt=0.002，2000 步，仍为 4 秒；接触比例 0.98，
  速度 0.01698 m/s、角速度 0.26901 rad/s，稳定性失败。
- 最后一轮：满足既定 tangential_failure 触发条件，以 half_dt 和
  friction_multiplier=1.25 复验；接触比例 0.988，速度 0.01690 m/s、
  角速度 0.23508 rad/s，仍失败。最大穿透约 0.0311 mm，低于 1 mm 门槛。

所有试验支撑关系通过，末段位移与转角通过，速度/角速度未通过
0.01 m/s 和 0.05 rad/s 原有门槛。轨迹表现为接触处小幅振动，具体是
碰撞形状还是接触求解参数导致仍需进一步诊断，不宣称已定位到唯一物理原因。
同时间长度半时间步没有消除失败，不能以截图静止代替稳定性。
本輪已执行基线加三轮记录，不继续无界调参或放宽阈值。

汇总：`03_physics/verification_summary.json`；轨迹在各试验的 `trace.jsonl`；
末段诊断图 `03_physics/velocity_diagnostic.png`；可复查执行脚本也保存在
`03_physics/`。这些属于失败诊断，不是最终展示。
回归：核心 378 passed，媒体/导入物理 20 passed、1 skipped；Ruff 与
`git diff --check` 通过。

### 接触振动定位（2026-09-07）

九组真实 Genesis 对照使用相同鼠标质量、惯量、初态及 4 秒仿真。
基线末段最大速度/角速度为 0.023479 m/s、0.492058 rad/s；求解迭代
50→200 无变化。仅替换桌面碰撞体为闭合有限盒体网格，保留原鼠标，
两项降至约 8.74e-7 m/s、2.45e-5 rad/s，接触点稳定为 4 个。

已确认 `media_support.materialize` 对该桌面进行行列式 -1 的轴重排而不
反转面绕序：鼠标下方桌面顶面法线向下、底面向上。仅反转面绕序后
仍超限（0.021385 m/s、0.391424 rad/s），故不能把朝向错误说成唯一原因。
原桌面碰撞网格合并重复顶点后仍非闭合，局部桌板厚约 3.30678 mm；
Genesis 默认 SDF 目标间距 5 mm，且非闭合网格跳过薄壁精化。

盒体试验使用 20 mm 假设厚度及基线接触高度，并非正式几何修复或验收；
生产桌面生成代码和所有原始失败轨迹未修改。详细结果及后续精度对照状态
见 `data/media_acceptance/contact_diagnosis_001/diagnosis.md`。

最后的第十组精度对照已完成：保持原桌面网格顶点，反转面绕序并将桌面
SDF target=1.5 mm、max_res=384，末段速度 7.32e-7 m/s、角速度
1.93e-5 rad/s，接触点稳定为 3。该结果支持薄桌板距离场表示是关键因素，
不再只是未验证猜测；尚未测试精度单改、不修正绕序的组合。十组轨迹指标
重算一致。生产实现未改，仍须新任务完整物理验收及半时间步复验。

### 位置求解后真实鼠标验证通过

`output/鼠标_位置求解物理_002` 复用 native_003 的单图鼠标资产，应用
position_solver 的 +1.7507 mm Z 位移，修正桌面碰撞面绕序，并显式使用
1.5 mm / max_res 384 桌面 SDF。基线和同时间长度半时间步均实际物理通过，
末段接触比例 1.0，最大穿透分别 0.4644/0.2529 mm；全程完整足迹最小
边距 109.33 mm，通过 20 mm 门槛。旧失败场景及原资产仍保持不变。
终态参考视角、多视角、环绕视频已独立生成；桌面黑色与原图木纹差异明显，
渲染成功不代表视觉还原准确。详细步骤、脚本及限制见 POSITION_SOLVER.md。
本次是复用已有单图资产后的验收，不是新 checkout 全流程重建，也不覆盖视频。

### 2026-09-08 已有场景图物理流程

默认媒体物理阶段已改为位置求解、干预式稳定化和两次独立自由回放，不再
自动替换桌面或调整摩擦。当前新增 mouse/crowded/stack 三个四阶段任务，
均完成真实物理及终态展示；单鼠标和同桌双鼠标复用重建资产，双桌面/堆叠
为明确尺寸盒体夹具。稳定化不作为通过证据，两次自由回放均执行原门槛及
全程支撑足迹/声明关系检查。新任务在 `output/场景图物理_*_001`，详见
POSITION_SOLVER.md。未重新执行外部模型重建，未进行视频输入验收。

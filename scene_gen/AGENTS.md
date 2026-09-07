<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-29 | Updated: 2026-07-29 -->

# scene_gen

## Purpose（用途）
`/gen-env` 的核心库与心脏。默认以确定性规则解析器实现 `text -> 类型化 SceneSpec`，也提供显式 opt-in 的两阶段 LLM 语义提取；两条路径都必须经过同一受限 prompt 边界和 Pydantic 契约，之后再进入确定性的 RoboTwin 资产 grounding、目标局部 support/containment 求解、哈希绑定的 `ResolvedSceneSpec`、RoboTwin/SAPIEN 回放与 contact/stability/containment/visibility/video 门控。

## Key Files（关键文件）
| File | Description |
|------|-------------|
| `__init__.py` | 包初始化；导出 `SceneSpec`、`ResolvedSceneSpec`、`SceneSpecError`、`scene_spec_json_schema` |
| `schema.py` | 类型化 pydantic 契约：`SceneSpec`、`ResolvedSceneSpec`、`RelationType`、禁用键（约 502 行） |
| `parser.py` | 默认双语规则解析器及 provider 公共边界；绝不允许代码、路径、资产/模型 id 或 pose 进入 `SceneSpec` |
| `llm_provider.py` | 显式 opt-in 的两阶段 LLM 物体/关系提取、确定性清洗、重试、缓存与无密钥证据 |
| `semantic_checks.py` | 对 LLM 候选做无网络的高置信文本一致性检查 |
| `grounding.py` | RoboTwin 资产 grounding：类型化 `SceneSpec` -> 解析后的资产 |
| `object_records.py` | CLI 提前导出逐对象/关系记录、检索绑定核对与有效成功包只读复用 |
| `catalog.py` | 从 RoboTwin checkout 构建资产目录：尺寸、朝向、support 面、interiors、关节、可用性 |
| `asset_overrides.yml` | 实测 RoboTwin 资产 override（如 100 mm `003_plate` 面、8 mm 余量、12 mm `110_basket` 内底） |
| `asset_generator.py` | 为缺失/不稳定目录资产生成确定性几何代理 |
| `solver.py` | 目标局部 support 与 containment 求解器 |
| `support_geometry.py` | support 面 / 碰撞地面几何数学 |
| `builder.py` | 构建哈希绑定的 `ResolvedSceneSpec` 包 |
| `runtime_sampling.py` | 物理回放的运行时采样/接触窗口逻辑 |
| `validator.py` | 静态验证：contact、stability、containment、visibility、视频帧门控 |
| `acceptance.py` | 验收门逻辑：contact/support/containment/visibility/video 阈值 |
| `rendered_critic.py` | 可选 VLM 渲染评判，检查可见语义 |
| `colors.py` | 请求属性中的颜色名解析/映射 |

## Subdirectories（子目录）
| Directory | Purpose |
|-----------|---------|
| `envs/` | 生成场景的运行时环境（RoboTwin/SAPIEN 回放入口）（见 `envs/AGENTS.md`） |
| `prompts/` | `llm_provider.py` 使用的两阶段受限提取 prompt（见 `prompts/AGENTS.md`） |

## For AI Agents（给 AI agent 的提示）

### Working In This Directory（在本目录工作）
- `schema.py` 中的 schema 门控每个下游阶段。先有意识地改它；不要让后续阶段漂移契约。
- 规则解析器和 LLM provider 必须保持同一受限边界：绝不让代码、路径、资产/模型 id 或 pose 进入 `SceneSpec`；LLM 失败不得静默回退到规则解析。
- `asset_overrides.yml` 条目须有实测几何或文档化的仿真器探测——绝不编造尺寸。
- 渲染不是物理证据；`validator.py` / `acceptance.py` 中的运行时门控才是权威。
- 运行时证据与 resolved scene 保持哈希绑定（见 `builder.py`）。

### Testing Requirements（测试要求）
- `pytest -q` 通过 `tests/scene_gen/` 覆盖本目录（parser、LLM provider、schema、catalog、grounding、solver、builder+validator、asset generator、generated scene、acceptance、prompt matrix、rendered critic）。
- 对 support、containment、loader 或 validator 契约的改动，还须在支持 RoboTwin/SAPIEN 的机器上做真实回放。
- 在 `tests/scene_gen/` 中为每个已发现的误报模式保留一个攻击测试。

### Common Patterns（常见模式）
- `schema.py` 中为类型化 pydantic 模型；下游模块接受/返回这些类型，而非原始 dict。
- `llm_provider.py` 只负责语义候选；真实资产仍由 `grounding.py` 选择，pose 仍由 `solver.py` 计算。单元测试使用 fake transport，不访问真实模型服务。
- 目标局部几何（support 面、容器 interior）由声明的 target 计算，绝不来自外层 AABB。
- 确定性代理携带来源 lineage，可追溯到其替代的 catalog 缺失项。

## Dependencies（依赖）

### Internal（内部）
- 被 `script/`（CLI 入口）与 `demo/`（Flask API）消费。
- `envs/` 提供运行时脚本使用的回放入口。

### External（外部）
- pydantic >=2.9,<3 —— 类型化契约
- PyYAML >=6,<7 —— `asset_overrides.yml` 解析
- Pillow >=10,<12 —— `rendered_critic` 的图像处理
- transformers / accelerate / qwen-vl-utils（可选 extra `vlm`）—— 渲染评判

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->

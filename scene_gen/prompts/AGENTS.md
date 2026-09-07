<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-29 | Updated: 2026-07-29 -->

# prompts

## Purpose（用途）
显式 opt-in 的 `scene_gen/llm_provider.py` 使用的两阶段 prompt：先提取物体与属性，再只根据原句和已清洗的 object id 提取 topology/lateral 关系。模板通过 `[tool.setuptools.package-data]` 作为数据打包，而非代码。

## Key Files（关键文件）
| File | Description |
|------|-------------|
| `llm_objects.md` | 阶段 1：受限物体、属性、桌面区域与 articulation 提取 |
| `llm_relations.md` | 阶段 2：受限 support topology 与 lateral 关系提取 |
| `parse_scene.md` | 旧单阶段模板；当前运行时代码不加载 |

## Subdirectories（子目录）
无。

## For AI Agents（给 AI agent 的提示）

### Working In This Directory（在本目录工作）
- prompt 必须保持解析器受限：指示模型绝不产出代码、路径、资产/模型 id 或 pose。
- 这里改动会改变 LLM 行为并使 prompt-hash 缓存键自动变化；用 `tests/scene_gen/test_llm_provider.py` 的 fake transport、清洗与缓存攻击测试验证。

### Testing Requirements（测试要求）
- `pytest -q tests/scene_gen/test_llm_provider.py tests/scene_gen/test_parser.py` 覆盖两阶段 prompt 契约和共同 parser 边界，不调用真实模型服务。

### Common Patterns（常见模式）
- Markdown prompt 作为 package data 打包；`LLMSceneProvider` 初始化时加载两个 active prompt 并记录各自 SHA-256。

## Dependencies（依赖）

### Internal（内部）
- `scene_gen/llm_provider.py` 加载 `llm_objects.md` 与 `llm_relations.md`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
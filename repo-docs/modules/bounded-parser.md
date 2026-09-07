# 受限解析

## 为什么默认解析器不是「能听懂话」

`/gen-env` 默认仍用正则 + 双语词典解析；只有 CLI 显式传 `--provider llm` 才启用两阶段 LLM 语义提取。设计压力对两条路径相同：自然语言里说 `can`，机器人在 RoboTwin 里有十几个匹配项；如果允许 prompt 或模型输出直接带 `asset_id`、`qpos`、文件路径或 pose，编译就退化成「让调用方写机器人配置」。所以解析层的第一职责仍是「拒绝」——把输入压回一个语义子集。你看到这个职责起作用的位置在 [一条真实路径 Step 1](../walkthroughs/one-real-run.md)。

默认规则路径允许：`apple`/`basket`/`block`/`bottle`/`bowl`/`calculator`/`cabinet`/`cup`/`box`/`hammer`/`knife`/`laptop`/`oven`/`plate`/`tray`/`vegetable` 等词表内桌面物体名（中英同义）。LLM 路径可提出其他受限格式的英文单数语义类别，但能否继续仍由真实 catalog grounding 决定。两条路径共享颜色、材质、区域（left/right/front/back/center）、关系 `on_table`/`on_top_of`/`inside`/`left_of`/`right_of`/`front_of`/`behind`/`near`/`distance_at_least` 与 articulation 的 `SceneSpec` 契约。

调用任何 provider 前，`validate_prompt_boundary` 会拒绝可执行代码、URI、POSIX/Windows/UNC/环境变量路径、资产或模型 id、backend 字段、世界坐标和三元坐标；控制字符与双向文本控制符同样不能穿过边界。外加 schema 端的 `FORBIDDEN_SCENE_KEYS`——即使 provider 路径返回的 JSON 也过不去这道。provider 公共边界还会在本地重建 `scene_id`、`request`、`language`、`seed`、frame/unit/workspace，只接受模型提供的 objects/relations 语义。schema 的完整契约在 [类型化场景契约](scene-contract.md)。

## 默认规则路径怎么从一句话抽到 object + 关系

默认解析过程不是「整句翻译」，是「先找物体，再判断两两关系」。`extract_mentions` 把 sentence 里所有词典命中的 span 抽出来，长词优先；通用 `place/put/add a <object>` 模式补漏（按属性词剥离再拼成 category）。每个 mention 拿一个 `{category}_<counts>` 形式的 object_id，比如 `can_1`、`plate_1`。

然后 `_relation_between` 看两个 mention 之间的词和目标之后的窗口判定关系——`on top of` / `inside` / `to the left of` / `in front of` / `near` / `behind` 都匹配。嵌套 source（on_top_of / inside 的 source）去除 `on_table` 关系，只有放在桌上的才拿 `on_table`。最后 `parse_rule_based` 走 schema 构造 `SceneSpec`。

代表案（来自速测）的输入输出：

| Prompt | objects | relations |
| --- | --- | --- |
| `Place a can on top of a plate.` | can_1, plate_1 | on_table plate_1→table；on_top_of can_1→plate_1 |
| `Put a cup inside a basket.` | cup_1, basket_1 | on_table basket_1→table；inside cup_1→basket_1 |
| `Place a red can to the left of a plastic basket near the center.` | can_1, basket_1（color=red、material=plastic）| on_table basket_1→table；left_of can_1→basket_1；near can_1→basket_1(max 0.25 m) |
| `把杯子放进篮子里。` | cup_1, basket_1 | on_table basket_1→table；inside cup_1→basket_1 |
| `Place a half-open cabinet on the table.` | cabinet_1（articulation=partially_open 0.5）| on_table cabinet_1→table |

`half-open` / `打开一半` / `open 60%` 在 `_articulation_for` 里被识别成 `partially_open` + `open_fraction=0.5`（或显式百分比），比例必须在 `(0, 1)` 开区间内。

## 显式 LLM provider 怎么提取

`script/generate_scene.py --provider llm` 懒加载 `LLMSceneProvider`；默认 `rule`、Demo、prompt matrix、100-seed runner 和 `python -m scene_gen.parser` 都不调用模型。阶段 1 用 `llm_objects.md` 提取物体、颜色/材质、桌面区域和 articulation，并要求 `<category>_<n>` 的确定性 object id；阶段 2 用原句和阶段 1 的已清洗 id 提取每个物体唯一的 support topology，以及同一 immediate support 上的 lateral 关系。

安全顺序是 `validate_prompt_boundary` → 无法表达/有歧义语义预检 → 每阶段严格单 JSON + 字段白名单清洗 → 高置信 object/属性/直接关系一致性检查 → `parse_provider_payload`（内部执行 `SceneSpec.model_validate`）。

LLM 不会扩大 `SceneSpec` 的语义面：否定、选择或关系析取、未绑定引用、无法表达的尺寸/形状/材质 modifier、outside/touching/far-away 等未支持空间关系，以及把已知中文词嵌入未知复合名词的表达都会 fail-closed；明确数值的 `distance_at_least` 仍受支持。

模型报告的 ambiguity 或本地能确定的引用歧义会立即失败且不重试；传输、JSON 或可修正的候选语义错误默认最多尝试 3 次（可配置为 1–10），并把短失败原因作为下一次反馈。任何失败都不会自动退回规则解析。

配置既可由完整的一组 `GENENV_LLM_ENDPOINT`/`GENENV_LLM_API_KEY`/`GENENV_LLM_MODEL` 提供，也可由 `--llm-config`、`GENENV_LLM_CONFIG`、`X2ENV_LLM_CONFIG`、`SCENE_GEN_LLM_CONFIG` 或本地 `configs/llm.yaml` 选择 YAML；profile 可由 `--llm-profile` 或 `GENENV_LLM_PROFILE` 选择。当前选中的 YAML profile 必须且只能从 `api_key`（直接写在 ignored 本地配置中）或 `api_key_env`（命名环境变量）选择一种凭据来源；两者并存、`api_key_file` 和含糊的 key/token 别名都会被拒绝。完整的直接环境变量组优先于 YAML。两种配置都支持 OpenAI-compatible chat/completions 与 Responses API。

两个阶段都成功后才原子写缓存；配置项 `cache_dir` 的默认值是 `data/scene_gen/parse_cache`，直接环境配置可用 `GENENV_LLM_CACHE_DIR` 改写。缓存键覆盖 request、seed、provider/config fingerprint 与两个 prompt 的内容哈希，不包含密钥；命中时仍重验 payload、语义检查和证据结构。CLI 在 SceneSpec 校验成功后、加载 catalog 前，即保存 `scene_spec.json`、逐对象 `objects/<object_id>.json`、`relations.json` 和无密钥的 `llm_parse_evidence.json`；检索或求解失败不会丢弃这些解析记录。编译成功后再由既有 `package_manifest.json` 绑定其大小和 SHA-256。拆文件不增加 LLM 调用，也不改变提示词；规则路径同样导出对象与关系，但不产生 LLM 证据。外部模型对首次未缓存调用不承诺位级确定性；已验证 SceneSpec 下游和同键成功缓存重放才进入确定性边界。配置、提取或 grounding 失败都写结构化 failure report。

## 改动入口与验证

- 加新物体词：改 `OBJECT_TERMS`（也别忘了 `asset_overrides.yml`/catalog 端的真模型）。
- 加新关系：先改 `schema.RelationType`、`SceneSpec.semantic_consistency`；再改 `_relation_between` 与 `_pair_relation_reasons`。
- 加新禁用模式：改 `FORBIDDEN_PROMPT_PATTERNS` 或 schema 端 `FORBIDDEN_SCENE_KEYS`。
- 改 LLM 字段、prompt、配置、缓存或证据：同步 `llm_provider.py`、`semantic_checks.py` 与两个 active prompt。
- 改完跑 `pytest -q tests/scene_gen/test_parser.py tests/scene_gen/test_llm_provider.py`；该测试使用 fake transport 和预建缓存，不访问真实服务。
- CLI 表面另用 `python script/generate_scene.py --help` 核验。

要回到主路径，到 [一条真实路径](../walkthroughs/one-real-run.md)。要继续追「`SceneSpec` 里能带什么字段、为什么 frozen」，去 [类型化场景契约](scene-contract.md)。

证据状态：除特别标注外，本页基于当前源码已确认。

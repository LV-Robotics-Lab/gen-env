# 质检报告 For /gen-env 中文 Repo Docs

这是一份审计笔记，不是讲解页。它检查本指南是否真的把可用读者模型传下去，以及在哪里还留有残余风险。要理解仓库，从 [一条真实路径](../walkthroughs/one-real-run.md) 起；本页只在它后面做可理解性审计。

## Reader Simulation

下面的回答只从已生成的指南里推，假设读者此刻打不开源码。

| Reader question | Answer from the guide |
| --- | --- |
| What real path is followed? | 一条 prompt → `SceneSpec` → `ResolvedSceneSpec` 包 → SAPIEN 回放 → `runtime_validation_report.json`；`generate_scene.py` 默认走 rule，只有显式 `--provider llm` 才以两阶段 LLM 替换第一段提取。 |
| What is hard or non-trivial? | 文本到机器人的信任边界（prompt 和模型输出都不能带代码/路径/资产 id/pose）+「看起来稳」与「物理稳」之间的差距（外层 AABB 能落不等于 plate 内侧 100 mm 那块能落；渲染图能过不等于终末接触能过）。 |
| What changes at each phase? | 默认规则解析或显式两阶段 LLM 提取产出同一 `SceneSpec` → schema 锁契约 → grounding 选真实模型 → solver 在目标局部几何内 bounded 拒绝回退摆位 → derived proxy 补 catalog 不稳 → builder 写哈希绑定包（LLM 路径含解析证据）→ SAPIEN 回放采 contact + 视频 + drift → validator 把证据转 checks。 |
| Harness run `succeeded` 是否等于可以发布？ | 不等于。`succeeded` 只表示 handler 正常产出类型化 output；validate output 仍可为 `fail` 或 `incomplete`。只有 validate 明确给出 `validation_status=pass`、无 blockers，并且后续 handler 核对完整跨 run 发布条件后，才能得到 `publishable=true`。 |
| Where would I change this behavior? | 见 [代码地图](../code-map.md)：默认规则与公共边界在 `scene_gen/parser.py`；LLM 提取/语义检查/prompt 在 `llm_provider.py`、`semantic_checks.py`、`prompts/`；契约 `schema.py`；求解 `solver.py`；包 `builder.py`；门控 `validator.py`；回放 `generated_scene.py` 与 `run_scene_runtime.py`。 |
| Where do assumptions stop? | 真机 RoboTwin/SAPIEN 和真实 LLM endpoint 都不在 CI 中跑；LLM 测试用 fake transport/预建缓存，只证明适配器边界，首次未缓存模型调用也不承诺位级确定性。绘图评判仍不是物理证据。 |
| What would prove this explanation wrong? | 一份提示注解：`docs/evidence/prompt-matrix-20260719.md` 的 33/33 编译 + 10/10 SAPIEN 通过是 2026-07-19 在 RTX 5090 主机上一次性运行结果。后续若 `scene_gen/validator.py` 阈值或 `solver.py` 求解策略动过且未在真机回放，本指南里「真实路径」字面通过率不再对应源码。重跑 `python script/run_prompt_matrix.py --runtime` 应能再出 pass；出 fail 即证伪。 |
| What would a careful newcomer ask next? | 1) 「为什么不在 CI 里跑运行时？」→ 见本仓库根 `AGENTS.md` 的 Testing Requirements；2) 「digest 把哪些字段算进 canonical？」→ 见 [哈希绑定包](../modules/replay-package.md)；3) 「自动缩放什么时候允许？」→ 见 [确定性代理](../modules/derived-proxy.md) 的 `PRIMITIVE_PROXY_SOURCES` 白名单一段。 |
| How can I verify it? | `pytest -q`（无 RoboTwin）验主契约层；`pytest -q tests/scene_gen/test_parser.py tests/scene_gen/test_llm_provider.py` 验默认/LLM 边界、重试缓存与证据；builder/validator 攻击测试验物理误报，真机回放由 `run_scene_runtime.py` 跑。 |

## 可理解性 Review

| Review question | Result | Evidence | Follow-up |
| --- | --- | --- | --- |
| Can a reader state the hard part of the main path in one sentence? | Yes. | [一条真实路径](../walkthroughs/one-real-run.md) 开场与 Step 1 名把「文本到机器人的信任边界 + 看起来稳 vs 物理稳」两句压力直接给出。 | 下次 sync 时把这两句压力在 README 第二段开头也说明一次。 |
| Does the walkthrough still explain the flow if source links are hidden? | Yes. | 每步以「这一步做什么、为什么」起头，源码定位符排在句末做证据，不是主语。 | 维护时不退化成「函数名是主语」的叙述。 |
| Can a reader use `code-map.md` to locate the owning source area and a verification point without scanning the repository tree? | Yes. | [代码地图](../code-map.md) 每个目录给出关键符号 + 最近验证点（如 `tests/scene_gen/test_builder_validator.py`）+ 与主路径的关系。 | 如新增模块须保持 table 行扩展，避免遗漏验证点。 |
| Does each step start with observable behavior, pressure, or effect before source names? | Yes. | Step 1 先讲「prompt 不是被理解而是被收紧」，再点 `parser.py`；Step 4 先讲「外层 AABB 会翻」，再点 `support_geometry.py`。 | 维护时保留此序。 |
| Is there at least one real boundary, failure, retry, validation, or caveat path when evidence exists? | Yes. | Step 1 给 LLM ambiguity 立即拒绝、其余可修正失败的有界重试与 fail-closed/no-fallback；Step 4 给 `SceneSolveError`；Step 8 给具体物理误报攻击测试。 | 新增提取误报或物理假阳性攻击后，回到对应 module 补行。 |
| Are claims backed by `references/source-evidence.md`, tests, commands, configs, data, or artifacts? | Yes. | 每条主路径 claim 在 [证据底座](source-evidence.md) 表里有 evidence + confidence + caveat。 | 改动行为时同步回填证据表。 |
| Does the review name one falsifying check and one likely reader follow-up? | Yes. | falsifying check：`python script/run_prompt_matrix.py --runtime` 应再出 pass。follow-up：CI 不跑真机 + digest 字段 + 自动缩放白名单三条。 | 若真机回放产物 path 变更，更新本行。 |
| Does the evidence map prove at least two traversal passes and name adjacent paths checked but not traced? | Yes. | [证据底座](source-evidence.md#evidence-traversal-log) 有 Pass 1–4；第四轮覆盖 LLM provider、语义检查、active prompts、CLI、builder 与 fake-transport 测试。coverage note 仍列明批量 runner、rendered critic、demo 等相邻路径，并把尚不存在的 Registry/handler/MCP 标成缺口。 | 扩范围时把同步过的相邻路径提到追踪。 |
| What remains out of scope or partially verified? | 见 [残余风险](#残余风险) 表。 |

## 残余风险

| 残余项 | 风险 | 缓解 |
| --- | --- | --- |
| 真机 SAPIEN 回放不在 CI 中 | 「真实路径」里的运行时门控阈值在源码改动后可能不再有真机记录支撑 | 根 `AGENTS.md` 要求涉及 support / containment / loader / validator 契约的改动在支持 RoboTwin/SAPIEN 的机器上做真机回放；下一次再跑 `prompt-matrix-<新日期>.md` 时把本指南里的「已知通过」声明同步到新结果 |
| 外部 LLM 首次调用与服务漂移 | 相同输入的未缓存在线调用不保证位级相同，fake transport 测试也不能证明真实模型的语义质量 | LLM 仅显式 opt-in；严格清洗/语义检查、有界重试后 fail-closed、不 fallback；config/prompt 进入缓存指纹，成功解析证据由 package manifest 哈希绑定。真实 endpoint 升级后仍需单独做受控质量验收 |
| `demo/app.py` 队列层只做摘要 | demo 复用同一 `scene_gen` 流水线；本指南只说它是薄编排，未给独立 module | demo 模块若扩大其队列/产物注册职责到值得单说，再加一个 demo 控制面专属模块页 |
| `rendered_critic` 排除在详述之外 | VLM 判是可选 extra + 非物理证据；若它将来变成主流评判，会被误导性 | 在 [运行时门控](../modules/runtime-gates.md) 开场已显式说「渲染不是物理证据」；若 VLM 升为门控须立即改写本指南 |
| Threading 后端动态注入 | 解析器 / grounding / solver 都没并发；本指南沉默 | 未发现证据表明该方向有改动；如引入，新增一个并发与哈希专属模块页 |
| Harness 只有 PR1 schema tranche | 读者可能把 14 个 schema 误当成可调用的 compile/replay/validate，或把局部 `publishable` 自洽当成完整发布判定 | [Harness Schema Tranche](../modules/harness-schema-tranche.md) 明确列出 Registry、handler、invocation digest、artifact 内容校验、retry 和 MCP 尚未实现；PR2 落地时必须重走证据遍历并补执行路径 |

证据状态：除特别标注外，本页基于当前源码已确认。

# 哈希绑定包

## 为什么编译完的不只是一份 JSON

如果编译器的输出只是一份 `resolved_scene.json`，下游就没有办法在不信任生成环境时验证「这份 resolved 真是基于我当时那份 prompt + catalog 算出来的」。`scene_gen/builder.py` 将四个基础文件和 CLI 声明的附加记录纳入一份 manifest，每个文件的 SHA-256 与整个 resolved 的 canonical digest 都被记录；`verify_package` 重新算一遍对得上才 pass。这一步在 [一条真实路径 Step 6](../walkthroughs/one-real-run.md) 体现。

设计的信任线是这样的：上游 `SceneSpec.digest()` 是意图根；下游 `ResolvedSceneSpec.digest()` 是产物根；两者通过 `ResolvedSceneSpec.source_scene_spec_sha256` 连起来。catalog 也通过 `asset_catalog_sha256` 进根。任何一条改了，包哈希立刻不再匹配 manifest，就不再自证。

## 包里有什么

`script/generate_scene.py` 默认输出根目录为仓库的 `output/`，所以默认场景目录是 `output/<scene_id>/`；显式 `--out-root` 仍可覆盖。需要把现有 Genesis 编译和纯渲染产物也集中在这里时，OpenXSim transfer 使用 `--output output`，结果位于同一场景目录的 `transfer/compiled/genesis/`，执行其返回命令后图片位于 `render/`。这只是目录统一，不是自动接通自然语言到 Genesis 物理门控；历史 `data/` 输出不移动。

`build_scene_package` 在 `<out_root>/<scene_id>/` 下写五件：

| 文件 | 内容 | 哈希来源 |
| --- | --- | --- |
| `request.txt` | 用户的 prompt，结尾换行 | 文件 SHA-256 |
| `scene_spec.json` | `spec.canonical_dict()`（不含 None）pretty-printed JSON | 文件 SHA-256 |
| `resolved_scene.json` | `resolved.canonical_dict()` pretty-printed JSON | 文件 SHA-256 + `resolved.digest()` 两条 |
| `generated_scene.py` | `generated_module_source(resolved)` 生成的可导入模块 | 文件 SHA-256 |
| `package_manifest.json` | 上述四文件的 sha256 + bytes + resolved digest + scene_id/seed + 哈希根 | 自身不哈希 |

`generate_scene.py` 的逐对象流程另将 `objects/<object_id>.json`、`relations.json`、`asset_resolution.json` 作为 additional files 纳入同一 manifest；LLM 路径加 `llm_parse_evidence.json`，显式代理生成加 `asset_generation_report.json` 与 `effective_asset_catalog.json`。对象文件包含完整 SceneObjectSpec 字段（包括空值），关系文件是关系列表；两者统一从权威 SceneSpec 导出，不是新的可编辑输入接口。

解析记录在 catalog 加载前就保存，检索报告在求解前保存；失败目录只有中间记录和失败报告，没有成功包 manifest。CLI 只读复用输入、catalog、解析证据及生成选项相同、静态验证完成且哈希有效的成功包；其他已有目录要求更换 `--out-root`，不覆盖、清理或混入旧文件。单独调用 builder 的基础接口保持兼容。

`generated_scene.py` 提供两个入口给运行时回放：

- `generated_scene.py:load_scene(task, resolved_scene=None)` —— deploy 时用，把 resolved 直接以 string literal 嵌进模块，导入即拿。
- `scene_gen.envs.generated_scene:load_resolved_scene(task, payload)` —— resolved-only 入口。

manifest 看：

```text
schema_version: robotwin.generated_scene_package.v1
scene_id, seed, source_scene_spec_sha256, resolved_scene_sha256,
asset_catalog_sha256, compiler_version,
entrypoint: generated_scene.py:load_scene,
resolved_only_entrypoint: scene_gen.envs.generated_scene:load_resolved_scene,
files: [{path, sha256, bytes}, ...]
```

## 写盘前的拒绝条件

`build_scene_package` 落盘之前有两个强一致性检查：

```python
if resolved.source_scene_spec_sha256 != spec.digest():
    raise ValueError("resolved scene is not bound to the supplied SceneSpec")
if resolved.scene_id != spec.scene_id or resolved.seed != spec.seed:
    raise ValueError("resolved scene identity does not match SceneSpec")
```

也就是说，传入的 `resolved` 必须在哈希层证明自己是「基于这份 spec 算出来的」。`scene_id` 与 `seed` 也必须匹配。`solver.solve_scene` 在构造 `ResolvedSceneSpec` 时把 `source_scene_spec_sha256=spec.digest()`、`asset_catalog_sha256=catalog.digest()` 直接喂进去，所以正常路径必通过。

## `verify_package` 反向自证

`verify_package(root)` 给定一个包目录，反向逐个对：

1. 读 `package_manifest.json`。
2. 对 manifest 里每个文件记录：文件存在？SHA-256 算一遍对得上？记 `pass: exists and digest == record["sha256"]`。
3. 读 `resolved_scene.json` 重新 `ResolvedSceneSpec.model_validate_json` + canonical `digest()`；对 manifest 的 `resolved_scene_sha256`。

返回 `pass`/`fail` 的 schema 版本是 `robotwin.generated_scene_package_verification.v1`，含每条 check 的明细。`tests/scene_gen/test_builder_validator.py:test_package_verifier_detects_tampering` 锁住这条攻击——任何篡改立刻出 fail。

## builder / manifest 改动入口与验证

- 改包结构（加文件 / 改入口名）：改 `build_scene_package` 同步 manifest schema 版本号，并在 `verify_package` 同步读出新文件。
- 改 `ResolvedSceneSpec.digest` canonical 序列化（`sort_keys=True`、`ensure_ascii=False`、`separators=(",",":")`）：会让所有现存包停止自证，对历史经验是破坏性改动，前后明确。
- 调用方：`script/generate_scene.py` 直接调；`demo/app.py` 也走它产出每 job 的产物根。
- 改完跑 `pytest -q tests/scene_gen/test_builder_validator.py`。

要回到主路径，去 [一条真实路径](../walkthroughs/one-real-run.md)。要继续追「拿到包后怎么在 SAPIEN 里回放并采到物理证据、最后由哪些门控放过」，去 [运行时门控](runtime-gates.md)。

证据状态：除特别标注外，本页基于当前源码已确认。

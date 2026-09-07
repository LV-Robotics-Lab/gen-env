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

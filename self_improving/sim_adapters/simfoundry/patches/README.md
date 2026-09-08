# SimFoundry 媒体接入补丁

基于固定上游 `9e34ebefcd020583fbb755a8b57268dce78eca26`，保存本机媒体接入所需的三处源码改动：

- Hydra 参数正确引用中文及含空格的路径。
- stage 3 保存有限支撑 mask、深度、内参、平面内点及边界截断信息。
- 保留对应中文路径回归测试。

主仓库子模块指针保持上游版本。新 checkout 在安装与运行统一媒体入口前，从主仓库根目录执行：

```bash
git submodule update --init external/SimFoundry
git -C external/SimFoundry apply --unidiff-zero --check ../../self_improving/sim_adapters/simfoundry/patches/media-support.patch
git -C external/SimFoundry apply --unidiff-zero ../../self_improving/sim_adapters/simfoundry/patches/media-support.patch
```

已应用补丁的本机无需重复执行。可用 `git -C external/SimFoundry apply --unidiff-zero --reverse --check ../../self_improving/sim_adapters/simfoundry/patches/media-support.patch` 检查是否已应用；此命令只检查，不撤销。
补丁应用后子模块显示 dirty 是预期状态。不要覆盖子模块内其他未提交修改。

轻量验证（在根仓库开发环境运行；PYTHON_BIN 指向已安装上游依赖的 SimFoundry 环境）：

```bash
PYTHON_BIN=/absolute/path/to/simfoundry/python python -m pytest -q external/SimFoundry/tests/test_run_all_wrapper.py
```

该测试不代表模型服务、重建或 Genesis 物理端到端验收。

另需应用空结果门控补丁，防止分割物体全部被剔除后仍标记阶段成功：

```bash
git -C external/SimFoundry apply --check ../../self_improving/sim_adapters/simfoundry/patches/media-empty-decomposition.patch
git -C external/SimFoundry apply ../../self_improving/sim_adapters/simfoundry/patches/media-empty-decomposition.patch
```

统一媒体入口禁用整图上采样生成（`s5_scene.use_upsampled_source_image=false`），
确保分割 RGB 与深度同源；物体裁剪图补全继续使用固定默认图片模型。
静态场景不额外生成机器人（`s14_og.include_robot=false`）。

单图兼容视频还需应用奇数尺寸补丁（2026-09-08 真实输入 1280×1707 触发）：

```bash
git -C external/SimFoundry apply --check ../../self_improving/sim_adapters/simfoundry/patches/media-odd-image.patch
git -C external/SimFoundry apply ../../self_improving/sim_adapters/simfoundry/patches/media-odd-image.patch
```

仅兼容 H.264 视频补齐右侧／下侧最多一个像素；提供给深度和分割的 PNG 保持原尺寸和像素。
真实 FFmpeg 回归见 `genesis/tests/test_single_image_encoding.py`。

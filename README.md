# Gen-Env: Genesis Scene Generation and Asset Reuse

Build Genesis scenes from natural-language requests, reuse native and reconstructed
assets, and record explicit physics and rendering results. The current workflow
lives in `self_improving/sim_adapters/genesis/`; SimFoundry provides a separate
image/video reconstruction route. Both individual objects and reconstructed scene
poses can be imported into Genesis.

```text
Natural-language request
  -> object and relation extraction
  -> CLIP retrieval + VLM asset selection
  -> scene planning + deterministic layout solve
  -> initial Genesis previews
  -> independent Genesis physics validation
  -> final rendering after physical acceptance

Image/video -> SimFoundry -> portable URDF assets + reconstructed scene poses
  -> finite support asset selection -> Genesis physics -> final rendering
```

Asset selection, scene construction, physics validation, and final rendering have
separate results. A preview or a completed simulation does not imply that the
physical acceptance criteria passed.

## Start Here

| Goal | Entry point |
| --- | --- |
| Prepare the Genesis environment | [Install and test](#install-and-test) |
| Build a scene from text | [Native Genesis scene workflow](#build-a-native-genesis-asset-scene) |
| Check an existing scene's physics | [Genesis physics validation](#validate-an-existing-native-genesis-asset-scene) |
| Reconstruct an image or video into a Genesis task | [Unified media workflow](#reconstruct-image-or-video-into-genesis) |
| Construct and repair an existing text task | [Text construction and repair](#construct-and-repair-a-text-scene) |
| Preserve a reconstructed arrangement | [Scene import](#import-a-reconstructed-scene-with-its-poses) |
| Run upstream video reconstruction | [SimFoundry reconstruction](#reconstruct-video-with-simfoundry) |
| Add reconstructed assets to Genesis | [URDF import and acceptance](#reuse-simfoundry-assets-in-genesis) |
| Read detailed commands in Chinese | [Genesis adapter guide](self_improving/sim_adapters/genesis/README.md) |

## Repository Architecture

```text
self_improving/sim_adapters/genesis/     asset selection, layout, physics, rendering
self_improving/sim_adapters/simfoundry/  reconstruction CLI and environment setup
self_improving/asset_pipeline/          asset reuse and simulator migration
external/genesis-world/                pinned Genesis upstream submodule
external/SimFoundry/                    pinned reconstruction upstream submodule
configs/                               model configuration examples
assets/genesis/                        local assets, previews, and retrieval indexes
output/                                tasks named from natural-language input
data/                                  local reconstruction and acceptance artifacts
.cache/                                local model caches and task locks
repo-docs/                             Chinese architecture and behavior guides
```

Assets, caches, and bulk experiment outputs are local data and are not bundled in
a Git clone. Module ownership and source inventories are documented in
[self_improving/README.md](self_improving/README.md).

## Install And Test

Use Python 3.12 for the Genesis environment. Genesis is pinned to
`external/genesis-world@0e74bf392781884ccad765c3f344419c86b872ca`.
The following creates a separate environment with CPU PyTorch:

```bash
git submodule update --init external/genesis-world
python3.12 -m venv venv/genesis
source venv/genesis/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,platform]'
python -m pip install -e external/genesis-world
python -m pip install -r self_improving/sim_adapters/genesis/requirements-clip.txt

python -m pytest -q
```

Run the following examples from the repository root with the Genesis environment
activated. If you already have a working Genesis environment, activate that
instead. Asset preparation and preview dependencies are described in the
[adapter guide](self_improving/sim_adapters/genesis/README.md).
The root fixture tests do not establish real simulator acceptance; the Genesis
adapter has a separate suite:

```bash
python -m pytest -q self_improving/sim_adapters/genesis/tests
```

Some real simulator tests require explicit opt-in and local assets; see the
adapter's evidence notes for the corresponding commands.

Media reconstruction additionally requires the separate SimFoundry environment,
model checkpoints, service configuration, and FFmpeg/ffprobe. Installing Genesis
alone does not prepare the media pipeline. Apply the versioned
[SimFoundry media patch](self_improving/sim_adapters/simfoundry/patches/README.md)
to a fresh submodule checkout before using the unified media entry. See the
[SimFoundry setup guide](self_improving/sim_adapters/simfoundry/README.md).

## Model Configuration And Asset Prerequisites

Text extraction, VLM selection, and LLM scene planning require a configured model
service. Create the local configuration only if it does not already exist:

```bash
cp -n configs/llm.example.yaml configs/llm.yaml
chmod 600 configs/llm.yaml
```

Edit the active profile's endpoint, model, API mode, and credential source for
your service. The VLM selection service must support image inputs. Keep
`configs/llm.yaml` local; it is ignored by Git. SimFoundry uses its own separate
model and credential setup.

The text-to-scene example below expects the prepared index
`assets/genesis/clip_non_robot_v1/index.json` and its referenced asset files.
Follow the [asset preparation and CLIP instructions](self_improving/sim_adapters/genesis/README.md)
to build previews and an index before running it on a fresh machine. For
reconstructed assets, use the [SimFoundry asset guide](self_improving/sim_adapters/genesis/SIMFOUNDRY_ASSETS.md).

## Build A Native Genesis Asset Scene

This platform workflow uses native assets and its own scene contract. Prepare the
Genesis environment, asset previews, CLIP index, and local model configuration
using the [Genesis adapter guide](self_improving/sim_adapters/genesis/README.md).
Then run from the repository root with that environment's Python:

```bash
python self_improving/sim_adapters/genesis/extract_assets.py \
  --request "桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。" \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --vlm-config configs/llm.yaml --output-root output
```

The default entry point extracts objects and relations, selects each asset with
CLIP Top-3 plus VLM selection, and plans the initial scene. Code solves the
coordinates and renders initial views. `--stop-after assets` stops after selection.
Tasks retain `01_obj`, `02_scene`, `03_physics`, and `04_final_render`; a built
initial scene does not establish physical acceptance. Indexes and downloaded
assets are local prerequisites, not included in a fresh Git clone.

## Validate An Existing Native Genesis Asset Scene

An already selected and built `01_obj` / `02_scene` task can run an independent physics stage:

```bash
python self_improving/sim_adapters/genesis/validate_asset_scene.py \
  --scene-dir 'output/桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。' \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --fixed-object table_1 --profile baseline
```

This loads native MJCF/GLB, records four seconds of real CPU physics and checks stability,
declared support, penetration and explicit spatial relations. It preserves assets and initial
layout, replaces only `03_physics`, clears stale `04_final_render`, and never renders.
Exit codes: 0 passed, 2 physical failure, 1 input/loading/collection error. `half_dt` retains the
same four-second duration. The original four-asset scene currently fails the physical gates;
a completed validator does not imply the scene passes. See the
[adapter guide](self_improving/sim_adapters/genesis/README.md) and
[measured evidence](self_improving/sim_adapters/genesis/PHYSICS_VALIDATION_EVIDENCE.md).

## Construct And Repair A Text Scene

Use the explicit construction entry to copy an existing task's `01_obj` bindings
into a new task, prepare collision geometry, solve placement, and run physics
with bounded repairs. It preserves the source task and records each attempt:

```bash
python -m self_improving.sim_adapters.genesis.construct_asset_scene \
  --source-task 'output/桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。' \
  --output-dir data/genesis_text_repair/new_v2_seed_0 \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --fixed-object table_1 --seed 0 --profile text_repair_v1 \
  --repair-preset text_scene_v2 \
  --numerics-profile dt2ms_tau20ms_authored_v1 --render
```

The output directory must be new. This example explicitly selects the v2 repair
strategy and the recorded numerical configuration; it does not change CLI
defaults. The acceptance profile requires both final stability and correct
support ratios to reach 95%, together with the other geometry and physics gates.
`--render` produces final views only after all required objects pass. `inside`
is currently rejected by this text repair workflow.

The recorded table-and-cup controls passed three seeds at 2 ms and matching
1 ms replays. Full four-asset acceptance is still incomplete: the first v3
construction stopped at apple preparation with zero physics steps, and later
cup/bowl combination tests exposed a frozen-inertia mismatch during loading.
These are distinct failure stages, not a successful four-asset run. See the
[construction guide](self_improving/sim_adapters/genesis/TEXT_REPAIR_PLAN.md) and
[text repair evidence](self_improving/sim_adapters/genesis/TEXT_REPAIR_EVIDENCE.md).

## Reconstruct Image Or Video Into Genesis

The unified media entry accepts exactly one image or video and writes the same four-stage
task layout used by the text flow:

```bash
python -m self_improving.sim_adapters.genesis.reconstruct_media \
  --image /absolute/path/to/image.jpg --name image_trial_001 --output-root output \
  --clip-index assets/genesis/clip_non_robot_v1/index.json \
  --vlm-config configs/llm.yaml
```

For video, replace `--image /absolute/path/to/image.jpg` with
`--video /absolute/path/to/video.mp4` and use a new task name. Supply exactly one
input mode. Model-backed reconstruction and selection send images to the
configured services. `--resume` requires matching input bytes, mode, task name,
and effective configuration.

It preserves SimFoundry foreground poses, retrieves a finite Genesis desk/table/counter
asset, creates fixed collidable `support_0`, and runs contact-bound physical validation.
No final render is produced unless physics passes. Image depth and hidden geometry are
explicit inference, not measurement. The pipeline permits a baseline plus at
most three repair attempts. Exit codes are 0 for physics and rendering success,
1 for execution/input errors, 2 for physical failure, and 3 for stable results
whose declared relations remain unverified.

**Recorded local status:** the mouse image completed stages 1b–4; the video
decoded 124 distinct frames and sampled 15. GPU contention blocked later work,
and online execution was not started. Both tasks recorded `execution_failed`,
with physics and final rendering `not_run`. End-to-end media reconstruction
has not yet passed local acceptance. See the
[adapter guide](self_improving/sim_adapters/genesis/README.md) and
[local acceptance record](self_improving/sim_adapters/genesis/MEDIA_RECONSTRUCTION_EVIDENCE.md).

## Reconstruct Video With SimFoundry

`external/SimFoundry` is pinned to
`9e34ebefcd020583fbb755a8b57268dce78eca26`. The platform wrapper exposes
reconstruction, augmentation, and application smoke commands; the upstream
project owns the reconstruction algorithm. Follow the
[installation and model configuration guide](self_improving/sim_adapters/simfoundry/README.md)
to prepare its separate environment and model access before running:

```bash
bash self_improving/sim_adapters/simfoundry/run.sh doctor
bash self_improving/sim_adapters/simfoundry/run.sh reconstruct \
  --scene-name fruits_trial_001 \
  --video-fpath "$PWD/external/SimFoundry/docs/assets/example_videos/Fruits.mp4" \
  --root-dir "$PWD/data/simfoundry" \
  --dry-run -- s7_mesh.low_vram=true
```

The example requests an execution plan. Remove `--dry-run` to reconstruct after
configuring the models; use a new scene name for each input. `doctor` checks local
prerequisites, not service authorization or end-to-end readiness. The output
`s14_og/reconstructed_og_scene.json` is an OmniGibson scene. Importing its objects into Genesis is a separate step.

The recorded 2026-09-07 Fruits run reconstructed seven objects and completed a
120-step OmniGibson random-action smoke run. See the
[reproduction record](self_improving/sim_adapters/simfoundry/REPRODUCTION.md)
for evidence and limitations. This does not establish reconstruction accuracy or
Genesis physical acceptance. The newer single-image and video integration has
partial local evidence as described in the [unified media workflow](#reconstruct-image-or-video-into-genesis).

## Reuse SimFoundry Assets In Genesis

The independent importer converts reconstructed single rigid objects into
self-contained URDF packages with visual/collision meshes, physical metadata,
and source hashes. It preserves geometry and physical meaning while expressing
inertia in the link frame. It does not transfer the original scene layout or robot.
With an existing reconstruction, run in the Genesis environment:

```bash
python -m self_improving.sim_adapters.genesis.import_simfoundry_assets import \
  --scene-dir data/simfoundry/fruits_da3_20260907 \
  --output-dir assets/genesis/my_simfoundry_library
```

The output directory must be new. Follow the
[standard asset guide](self_improving/sim_adapters/genesis/SIMFOUNDRY_ASSETS.md)
for six-view previews, CLIP indexing, union retrieval, and independent drop tests.
Packages can be loaded in Genesis without a SimFoundry or OmniGibson installation.
Selection from the union index runs an asset physics precheck; failure retains
the selected candidate and blocks automatic scene construction.

**Recorded Fruits status (2026-09-07):** all seven packages, 42 preview images,
and dynamic geometry/mass/inertia checks passed. All seven 1,000-step drop tests
completed but failed the declared physical thresholds: penetration exceeded
1 mm for every asset, and the banana also exceeded the final angular-velocity
limit. No automatic repair was applied. These are asset import and loading
results, not seven physically accepted assets. Compact results are recorded in
[SIMFOUNDRY_EVIDENCE.json](self_improving/sim_adapters/genesis/SIMFOUNDRY_EVIDENCE.json);
bulk reports and videos remain local under
`data/simfoundry_genesis/fruits_acceptance_v2/`.

## Import A Reconstructed Scene With Its Poses

After importing the standard asset library, preserve the original object
positions and full rotations with the separate scene importer:

```bash
python -m self_improving.sim_adapters.genesis.import_simfoundry_scene import \
  --scene-dir data/simfoundry/fruits_da3_20260907 \
  --library-path assets/genesis/my_simfoundry_library/library.json \
  --output-dir data/simfoundry_genesis/my_scene

python -m self_improving.sim_adapters.genesis.import_simfoundry_scene verify \
  --scene-package data/simfoundry_genesis/my_scene

python -m self_improving.sim_adapters.genesis.import_simfoundry_scene preview \
  --scene-package data/simfoundry_genesis/my_scene \
  --output-dir data/simfoundry_genesis/my_scene_preview

python -m self_improving.sim_adapters.genesis.validate_imported_scene \
  --scene-package data/simfoundry_genesis/my_scene \
  --output-dir data/simfoundry_genesis/my_scene_physics
```

Use new output directories. The importer copies bound asset dependencies and
applies the source world poses; it does not rerun text layout planning. Imported
packages use this dedicated validator, not the text TaskOutput validator.
Missing source support relations remain unknown. Even if all stability checks
pass, that validator returns `incomplete` (exit 3) without declared relations;
physical failure returns 2 and input/execution errors return 1.

Details and supported source formats are in the
[scene graph importer](self_improving/sim_adapters/genesis/SIMFOUNDRY_SCENES.md).
It writes Genesis v2 graph/layout files with full object poses and portable assets.
The seven-object Fruits scene passed native loading and three-view rendering.
A separate 1,000-step physics run completed with five objects passing stability
checks and two failing: pear penetration and teal plate angular speed.
Sequential video and final diagnostic views are available; declared support
relationship acceptance remains unverified.

## Local Output Organization

`output/` contains task folders named from the input. Shared Genesis assets and indexes live in
`assets/genesis/`; historical checks in `data/genesis_history/`; video reconstruction outputs in
`data/simfoundry/`; caches and task locks in `.cache/genesis/`. Storage maintenance receipts live in
`data/storage_maintenance/`. See the [Chinese directory guide](repo-docs/modules/self-improving-platform.md#输出共享资源与缓存).

## Evidence And Acceptance

The recorded results below describe different inputs and profiles; they are not
interchangeable acceptance claims.

| Workflow | Recorded result | Remaining limitation |
| --- | --- | --- |
| Native four-asset text scene | Initial construction and previews available | Original physics gates failed |
| Text v2 table-and-cup controls | Three seeds and their half-step replays passed | Full four-asset acceptance incomplete |
| Unified mouse image/video tasks | Partial preprocessing completed | Execution failed; physics/rendering not run |
| Fruits single-asset import | 7 imports and 42 previews passed | All 7 drop tests failed |
| Fruits scene import | 7 objects loaded and rendered; physics completed | 2 objects failed; declared support acceptance unverified |


- Initial scene previews establish asset loading and visible layout. Run the
  independent physics stage to evaluate stability, support, penetration, and
  declared spatial relations.
- Single-asset drop tests are selection prechecks. A passing asset still needs
  validation in its assembled scene.
- Physics profiles define their own frozen thresholds. Record the selected
  profile and preserve failed runs, trajectories, and diagnostic artifacts.
- Images, videos, trajectories, and source assets are bound by manifests and
  hashes. Camera-orbit videos are distinct from sequential physics recordings.

Current commands and measured limitations are documented in the
[Genesis physics evidence](self_improving/sim_adapters/genesis/PHYSICS_VALIDATION_EVIDENCE.md)
and [SimFoundry asset evidence](self_improving/sim_adapters/genesis/SIMFOUNDRY_ASSETS.md).

## Provenance And License

External projects retain their independent Git histories and licenses as
submodules. Source provenance and retained-artifact policies are recorded in
[self_improving/source_inventory.json](self_improving/source_inventory.json).

This repository is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for licensing and attribution.

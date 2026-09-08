"""Unified image/video -> SimFoundry -> Genesis with source support preserved."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from scene_gen.llm_provider import load_llm_provider_config
from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import import_simfoundry_assets as asset_import
from self_improving.sim_adapters.genesis import import_simfoundry_scene as scene_import
from self_improving.sim_adapters.genesis import media_checkpoints as checkpoints
from self_improving.sim_adapters.genesis import media_support as support
from self_improving.sim_adapters.genesis import visual_support as visuals
from self_improving.sim_adapters.genesis import scene_physics_graph as graph_rules
from self_improving.sim_adapters.genesis import scene_physics_workflow as workflow
from self_improving.sim_adapters.genesis import validate_imported_scene as physics
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import DEFAULT_ROOT, TaskOutput, destination
from self_improving.sim_adapters.simfoundry import native_service

SCHEMA = "genenv.media_reconstruction.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SIMFOUNDRY_RUNNER = REPO_ROOT / "self_improving/sim_adapters/simfoundry/run.sh"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def probe_media(path, mode):
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError("input media must be a regular non-symlink file")
    suffix = path.suffix.lower()
    expected = IMAGE_EXTENSIONS if mode == "image" else VIDEO_EXTENSIONS
    if suffix not in expected:
        raise ValueError(f"unsupported {mode} extension")
    result = dict(
        schema_version=SCHEMA,
        mode=mode,
        original_path=str(path),
        original_name=path.name,
        sha256=file_hash(path),
        size_bytes=path.stat().st_size,
    )
    if mode == "image":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update(
                width=int(image.width), height=int(image.height),
                decoded_frame_count=1, unique_frame_count=1,
                sampled_frame_indices=[0], sampled_frame_count=1,
                sampled_unique_frame_count=1,
            )
        return result
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise ValueError("video input requires ffprobe and ffmpeg")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames:format=duration,format_name",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    if len(data.get("streams", [])) != 1:
        raise ValueError("input must contain exactly one video stream")
    stream = data["streams"][0]
    frames = int(stream["nb_read_frames"])
    if frames <= 0:
        raise ValueError("video has no decoded frames")
    hashes = []
    digest_output = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
        text=True,
    )
    for line in digest_output.splitlines():
        if line and not line.startswith("#"):
            hashes.append(line.rsplit(",", 1)[-1].strip())
    if len(hashes) != frames:
        raise ValueError("ffprobe and decoded frame counts differ")
    sample_count = min(15, frames)
    stride = max(1, frames // sample_count)
    sampled = list(range(0, frames, stride))[:sample_count]
    result.update(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_s=float(data["format"]["duration"]),
        average_frame_rate=stream["avg_frame_rate"],
        format_name=data["format"]["format_name"],
        decoded_frame_count=frames,
        unique_frame_count=len(set(hashes)),
        sampled_frame_indices=sampled,
        sampled_frame_count=len(sampled),
        sampled_unique_frame_count=len({hashes[index] for index in sampled}),
    )
    return result


def run_reconstruction(command, environment, log_path):
    """One reconstruction worker per host/user; redact credentials before persisting output."""
    import fcntl
    import signal
    from types import SimpleNamespace
    descriptor = os.open(f"/tmp/genenv-simfoundry-{os.getuid()}.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("SimFoundry GPU worker is busy; resume later") from None
        with Path(log_path).open("a") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True)
            try:
                for line in process.stdout:
                    for key in ("GEMINI_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"):
                        secret = environment.get(key)
                        if secret:
                            line = line.replace(secret, "[REDACTED]")
                    log.write(line)
                    log.flush()
                return SimpleNamespace(returncode=process.wait())
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    finally:
        os.close(descriptor)


def configuration_hash(path):
    # Include effective defaults and endpoint, never credentials.
    import yaml
    data = yaml.safe_load(Path(path).read_text())
    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()
                    if not any(word in k.lower()
                               for word in ("key", "token", "password", "secret"))}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    return clip.digest(scrub(data))


def render_validated(task, reconstruction=None):
    """Render the validated final state. `reconstruction` defaults to the task's own copy,
    and must be given when the reconstruction is referenced in place rather than staged."""
    import numpy as np
    validated = lib.read_json(task.stage("physics") / "validated_scene.json")
    if "workflow_report" in validated:
        official.verify_files(task.root, [validated["workflow_report"]])
        workflow.verify((task.root / validated["workflow_report"]["path"]).parent)
    source = Path(validated["scene_package"])
    if lib.sha256(source / "manifest.json") != validated["scene_manifest_sha256"]:
        raise ValueError("validated scene changed")
    official.verify_files(task.root, [validated["physics_result"]])
    evidence = (task.root / validated["physics_result"]["path"]).parent
    result = physics.verify_evidence(evidence)
    if result["physics_status"] != "passed":
        raise ValueError("final rendering requires passed physics")
    observation = lib.read_json(
        task.stage("objects") / "support_observation/support_observation.json")
    frame = observation["frame_index"]
    reconstruction = Path(reconstruction or task.stage("objects") / "reconstruction")
    transform = np.load(reconstruction / f"s4_frame/image_{frame}_cam2world.npy")
    arrays = np.load(task.stage("objects") / "support_observation/support_observation.npz")
    height, width = arrays["rgb"].shape[:2]
    out = task.stage("final_render") / "render"
    if out.exists():
        shutil.rmtree(out)
    final_state = lib.read_json(evidence / "final_state.json")
    with (evidence / "trace.jsonl").open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    if final_state != last:
        raise ValueError("final render state differs from verified trajectory")
    report = scene_import.preview(source, out,
        final_state=final_state, orbit=True,
        reference_camera={"cam2world": transform.tolist(), "resolution": [width, height],
                          "intrinsics": arrays["intrinsics"].tolist()})
    Image.fromarray(arrays["rgb"]).save(out / "input_reference.png")
    with Image.open(out / "reference.png") as rendered:
        comparison = Image.new("RGB", (width * 2, height), "white")
        comparison.paste(Image.fromarray(arrays["rgb"]), (0, 0))
        comparison.paste(rendered, (width, 0))
        comparison.save(out / "comparison.png")
    report.update(physics_status="passed", physics_result=validated["physics_result"])
    official.write_json(out / "render_report.json", report)
    return report


def verify_sampled_frames(scene_dir, media, mode):
    stage = Path(scene_dir) / "s1_video"
    frames = sorted((stage / "frames_all").glob("*.png"))
    count = 1 if mode == "image" else 15
    sampled = sorted((stage / f"frames_subsampled_{count}").glob("*.png"))
    if len(frames) != media["decoded_frame_count"]:
        raise ValueError("upstream decoded frame count differs from input")
    expected = [frames[i] for i in media["sampled_frame_indices"]]
    if [p.name for p in expected] != [p.name for p in sampled]:
        raise ValueError("upstream sampled frame indices differ from manifest")
    if any(file_hash(a) != file_hash(b) for a, b in zip(expected, sampled, strict=True)):
        raise ValueError("upstream sampled frame bytes differ from decoded frames")
    return dict(sampled_frame_indices=media["sampled_frame_indices"],
                files=[official.fingerprint(p, scene_dir) for p in sampled],
                decoded_frame_count=len(frames))


def asset_dependency_hash(index_path):
    index_path = local_path(index_path)
    index = lib.read_json(index_path)
    paths = {index_path}
    reference = index.get("official_index", {})
    if reference.get("path"):
        official_path = local_path(reference["path"])
        paths.add(official_path)
        for asset in index.get("assets", []):
            root = local_path(asset.get("source_root", official_path.parent))
            paths.update(root / row["path"] for row in asset.get("source_files", []))
            if asset.get("record"):
                paths.add(official_path.parent / asset["record"])
        paths.update(official_path.parent / row["image"]["path"]
                     for row in index.get("rows", []))
    if index.get("vectors", {}).get("path"):
        paths.add(index_path.parent / index["vectors"]["path"])
    return clip.digest({str(p): file_hash(p) if p.is_file() else None
                        for p in sorted(paths)})


PHYSICS_FILES = {'position_solver.py', 'scene_physics_graph.py', 'scene_physics_workflow.py',
                 'scene_stabilization.py', 'validate_imported_scene.py'}


def task_config(mode, source, clip_index, vlm_config, simfoundry_args,
                support_mode="upstream"):
    if support_mode not in {"upstream", "retrieved"}:
        raise ValueError("unsupported support mode")
    if support_mode == "retrieved" and clip_index is None:
        raise ValueError("retrieved support requires --clip-index")
    return dict(
        schema_version=SCHEMA,
        mode=mode,
        input_sha256=file_hash(source),
        support_mode=support_mode,
        clip_index=str(Path(clip_index).resolve()) if support_mode == "retrieved" else None,
        clip_index_sha256=file_hash(clip_index) if support_mode == "retrieved" else None,
        asset_dependencies_sha256=(asset_dependency_hash(clip_index)
                                   if support_mode == "retrieved" else None),
        vlm_config=str(Path(vlm_config).resolve()),
        configuration_sha256=configuration_hash(vlm_config),
        physics_implementation_sha256=clip.digest({
            name: file_hash(Path(__file__).parent/name) for name in sorted(PHYSICS_FILES)}),
        implementation_sha256=clip.digest({
            str(p.relative_to(REPO_ROOT)): file_hash(p)
            for directory in (Path(__file__).parent, SIMFOUNDRY_RUNNER.parent)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in PHYSICS_FILES
            and p.suffix in {".py", ".sh", ".patch", ".md"}
        }),
        upstream_sha256=clip.digest({
            str(p.relative_to(REPO_ROOT)): file_hash(p)
            for folder in ("scripts", "simfoundry")
            for p in sorted((REPO_ROOT / "external/SimFoundry" / folder).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".txt", ".md"}
        }),
        native_models=[native_service.TEXT_MODEL, native_service.IMAGE_MODEL],
        simfoundry_args=list(simfoundry_args),
        single_image_inference=mode == "image",
        background_splat=False,
        sampled_frames=1 if mode == "image" else 15,
        repair_rounds=0,
    )


def _normalize(source, mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    original = directory / ("original" + source.suffix.lower())
    shutil.copyfile(source, original)
    if mode == "image":
        canonical = directory / "source.png"
        with Image.open(source) as image:
            image.convert("RGB").save(canonical)
        # Upstream single-image mode historically derives source.png from source.MOV.
        runner_input = directory / "source.MOV"
    else:
        canonical = original
        runner_input = original
    return original, canonical, runner_input


def simfoundry_command(stage, runner_input, mode, extra):
    command = [
        "bash", str(SIMFOUNDRY_RUNNER), "reconstruct",
        "--scene-name", "reconstruction",
        "--root-dir", str(stage),
        "--video-fpath", str(runner_input),
        "--no-stream",
    ]
    command.extend(extra)
    command.append("--")
    command.extend(
        [
            f"s1_video.single_image_input={'true' if mode == 'image' else 'false'}",
            f"s1_video.n_subsampled_frames={1 if mode == 'image' else 15}",
            f"s3_ground.img_idx={0 if mode == 'image' else 'auto'}",
            "s1_video.splat_prep=false",
            "s3_ground.detection_model=gemini-2.5-flash",
            "s5_scene.detection_model=gemini-2.5-flash",
            "s5_scene.use_upsampled_source_image=false",
            "s14_og.include_robot=false",
            "s8_pose.front_pick_model=gemini-2.5-flash",
            "s11_sim.vlm_model=gemini-2.5-flash",
            "s7_mesh.low_vram=true",
        ]
    )
    return command


def _selection_document(query, candidates, rejected, evidence, selected):
    def clean(candidate):
        return {
            key: value for key, value in candidate.items()
            if key not in {"asset"}
        }
    return dict(
        schema_version=support.SELECTION_SCHEMA,
        query=query,
        weights=dict(semantic=0.4, observed_image=0.6),
        candidates=[clean(candidate) for candidate in candidates],
        rejected_candidates=rejected,
        vlm=evidence,
        selected_asset_id=selected["asset_id"],
        physics_status="not_run",
        meaning="retrieval and visual selection do not constitute physical acceptance",
    )


def _rewrite_manifest(scene):
    manifest = scene / "manifest.json"
    manifest.unlink(missing_ok=True)
    official.write_json(
        manifest,
        dict(
            schema_version=scene_import.VERSION,
            files=[
                official.fingerprint(path, scene)
                for path in sorted(scene.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            ],
        ),
    )
    scene_import.verify(scene)


def clearance_scene(source, destination_path):
    """Copy a scene and only lift penetrated foreground along +Z, capped at 10 mm."""
    source, target = Path(source).resolve(), Path(destination_path).resolve()
    shutil.copytree(source, target)
    layout = lib.read_json(target / "scene_layout.json")
    geometry = lib.read_json(target / "native_geometry.json")
    repairs = []
    for obj in layout["objects"]:
        if obj["fixed"]:
            continue
        minimum = float(obj["world_visual_bounds_m"][0][2])
        lift = min(0.01, max(0.0, -minimum + 1e-4))
        if lift:
            obj["translation_m"][2] += lift
            obj["world_visual_bounds_m"][0][2] += lift
            obj["world_visual_bounds_m"][1][2] += lift
            geometry[obj["object_id"]]["world_bounds"][0][2] += lift
            geometry[obj["object_id"]]["world_bounds"][1][2] += lift
            repairs.append(dict(object_id=obj["object_id"], lift_m=lift))
    layout["native_geometry_sha256"] = clip.digest(geometry)
    official.write_json(target / "scene_layout.json", layout)
    official.write_json(target / "native_geometry.json", geometry)
    _rewrite_manifest(target)
    return repairs


def tangential_failure(result):
    if not result or result.get("status") != "complete":
        return False
    return any(
        value.get("total_translation_m", 0) > 0.001
        and any(check["name"] == "support_fraction" and check["passed"]
                for check in value.get("checks", []))
        for value in result.get("objects", {}).values()
    )


def _physics_attempt(scene, directory, *, profile="baseline", friction_multiplier=1.0):
    directory.mkdir(parents=True, exist_ok=False)
    evidence_dir = directory / "evidence"
    result = physics.run(
        scene, evidence_dir, profile=profile, friction_multiplier=friction_multiplier
    )
    record = dict(
        directory=str(directory),
        scene_package=str(Path(scene).resolve()),
        scene_manifest_sha256=lib.sha256(Path(scene) / "manifest.json"),
        profile=profile,
        friction_multiplier=friction_multiplier,
        passed=result.get("physics_status") == "passed",
        exit_code=result["exit_code"],
        steps_executed=result["steps_executed"],
        result_sha256=lib.sha256(evidence_dir / "physics_result.json"),
    )
    official.write_json(directory / "attempt.json", record)
    return result, record


def run(
    source,
    mode,
    output_dir,
    clip_index,
    vlm_config,
    *,
    resume=False,
    simfoundry_args=(),
    process_runner=subprocess.run,
    selector=None,
    support_mode="upstream",
):
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    if any(any(word in arg.lower() for word in ("api_key", "token", "password", "secret"))
           for arg in simfoundry_args):
        raise ValueError("credentials must use private configuration, not command arguments")
    config = task_config(mode, source, clip_index, vlm_config, simfoundry_args, support_mode)
    request = f"{mode}:{source.name}"
    task = TaskOutput(output)
    config_changed = False
    if task.root.exists():
        if not resume:
            raise FileExistsError("output task exists; use --resume with identical inputs")
        prior = lib.read_json(task.stage("objects") / "input/media_manifest.json")
        previous_config = prior.get("effective_config", {})
        if any(previous_config.get(k) != config[k] for k in ("mode", "input_sha256")):
            raise ValueError("resume input mismatch; use a new task for different media")
        config_changed = previous_config != config
        task.owner(request)
        full_verified = True
        try:
            task.verify()
        except (OSError, ValueError):
            full_verified = False
            # A killed process may leave a stale full-task manifest. Stage checkpoints
            # are verified before any completed stage is reused.
            task.report = lib.read_json(task.root / "run_report.json")
        previous = lib.read_json(task.root / "run_report.json")
        if (full_verified and not config_changed
                and previous.get("physics", {}).get("render_status") == "passed"):
            return previous
    with task.lock():
        continuing = task.root.exists()
        if not continuing:
            task.start(request)
        else:
            if config_changed:
                upstream_equal = {k: v for k, v in previous_config.items()
                                  if k != "physics_implementation_sha256"} == {
                                      k: v for k, v in config.items()
                                      if k != "physics_implementation_sha256"}
                checkpoints.invalidate(task, "physics" if upstream_equal else "objects")
            else:
                task.owner(request)
                task.report = lib.read_json(task.root / "run_report.json")
        report = dict(
            schema_version=SCHEMA,
            status="running",
            exit_code=1,
            mode=mode,
            input_sha256=config["input_sha256"],
            output_dir=str(task.root),
            stages=task.report["stages"],
            physics_status="not_run",
            render_status="not_run",
            repair_budget=0,
        )
        phase = "media"
        try:
            if continuing and task.report.get("physics", {}).get("physics_status") == "passed":
                # Camera/depth observations and imported assets must remain bound too.
                for name in ("objects", "scene"):
                    if checkpoints.reusable(task, name, task.stage(name), config) is None:
                        if task.report.get("physics", {}).get("physics_status") == "passed":
                            checkpoints.invalidate(task, name)
                        break
            if continuing and task.report.get("physics", {}).get("physics_status") == "passed":
                phase = "render"
                report = dict(task.report["physics"])
                render_validated(task)
                report.update(status="physics_passed", render_status="passed", exit_code=0)
                task.report.update(report)
                (task.root / "failure.json").unlink(missing_ok=True)
                task.finish_physics(report)
                return lib.read_json(task.root / "run_report.json")
            cached_objects = checkpoints.reusable(task, "objects", task.stage("objects"), config)
            input_dir = task.stage("objects") / "input"
            if cached_objects is None:
                original, canonical, runner_input = _normalize(source, mode, input_dir)
                media = probe_media(original, mode)
                media.update(
                    canonical_input=official.fingerprint(canonical, task.root),
                    effective_config=config,
                    limitations=(
                        ["single-image depth, hidden geometry and physical properties are inferred"]
                        if mode == "image" else []
                    ),
                )
                official.write_json(input_dir / "media_manifest.json", media)
                if process_runner is subprocess.run:
                    phase = "service_preflight"
                    probe_image = canonical
                    if mode == "video":
                        probe_image = input_dir / "probe_frame.png"
                        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(canonical),
                                        "-frames:v", "1", str(probe_image)], check=True)
                    probe_dir = input_dir / "service_probe"
                    if not probe_dir.exists():
                        probe = native_service.probe(
                            vlm_config, probe_image,
                            probe_dir)
                    probe = lib.read_json(probe_dir / "report.json")
                    if probe["status"] != "passed":
                        raise ValueError("native service capability preflight failed")
                cache = REPO_ROOT / ".cache/simfoundry/model_calls" / clip.digest(config)
                command = simfoundry_command(
                    task.stage("objects"), runner_input, mode,
                    [*simfoundry_args, "--cache-mode", "--model-cache-dir", str(cache)])
                official.write_json(
                    input_dir / "simfoundry_invocation.json",
                    dict(command=command, background_splat=False, expected_scene="reconstruction"),
                )
                if file_hash(source) != config["input_sha256"]:
                    raise ValueError("input media changed before reconstruction")
                phase = "simfoundry"
                environment = os.environ.copy()
                if process_runner is subprocess.run:
                    provider = load_llm_provider_config(vlm_config)
                    environment.update(GEMINI_API_KEY=provider.api_key,
                        GOOGLE_GEMINI_BASE_URL=native_service.BASE_URL,
                        SIMFOUNDRY_GEMINI_BACKEND="api_key", SIMFOUNDRY_GEMINI_NONSTREAM_TEXT="1",
                        SIMFOUNDRY_GEMINI_NONSTREAM_IMAGES="1",
                        SIMFOUNDRY_SERVICE_ENV="/dev/null")
                scene_dir = task.stage("objects") / "reconstruction"
                try:
                    reconstructed = checkpoints.read(task, "reconstruction", scene_dir, config)
                except (OSError, ValueError, KeyError):
                    archive = task.root / "attempt_history" / str(time.time_ns())
                    archive.mkdir(parents=True)
                    if scene_dir.exists():
                        shutil.move(str(scene_dir), str(archive / "reconstruction"))
                    checkpoint = task.root / "checkpoints/reconstruction.json"
                    if checkpoint.exists():
                        shutil.move(str(checkpoint), str(archive / checkpoint.name))
                    reconstructed = None
                if reconstructed is None:
                    try:
                        reuse_upstream = checkpoints.reusable_upstream(task, scene_dir, config)
                    except (OSError, ValueError, KeyError):
                        archive = task.root / "attempt_history" / str(time.time_ns())
                        archive.mkdir(parents=True)
                        if scene_dir.exists():
                            shutil.move(str(scene_dir), str(archive / "reconstruction"))
                        reuse_upstream = False
                    if reuse_upstream:
                        command.insert(command.index("--"), "--skip-successful")
                    official.write_json(input_dir / "simfoundry_invocation.json",
                        dict(command=command, background_splat=False,
                             expected_scene="reconstruction", reused_upstream=reuse_upstream))
                    if process_runner is subprocess.run:
                        completed = run_reconstruction(command, environment,
                                                       input_dir / "reconstruction.log")
                    else:
                        completed = process_runner(command, cwd=REPO_ROOT, check=False,
                                                   env=environment)
                    checkpoints.save_upstream(task, scene_dir, config)
                    if completed.returncode:
                        raise RuntimeError(f"SimFoundry exited with {completed.returncode}")
                    if not (scene_dir / "s14_og/reconstructed_og_scene.json").is_file():
                        raise ValueError("SimFoundry did not produce the stage-14 scene")
                    sampled = verify_sampled_frames(scene_dir, media, mode)
                    official.write_json(input_dir / "actual_sampled_frames.json", sampled)
                    checkpoints.save(task, "reconstruction", scene_dir, config)
                for partial in ("foreground_assets", "support_observation"):
                    partial_dir = task.stage("objects") / partial
                    if partial_dir.exists():
                        shutil.rmtree(partial_dir)
                phase = "foreground_assets"
                foreground = task.stage("objects") / "foreground_assets"
                asset_import.import_assets(scene_dir, foreground)
                phase = "support_observation"
                # Extracted for both modes: it describes the surface the scene rests on,
                # which upstream needs for the reference camera and the visual stand-in.
                # Retrieval cannot proceed without it, so only there is failure fatal.
                try:
                    observation = support.observe(
                        scene_dir, task.stage("objects") / "support_observation")
                except (ValueError, OSError, KeyError):
                    if support_mode == "retrieved":
                        raise
                    observation = None
                if file_hash(source) != config["input_sha256"]:
                    raise ValueError("input media changed during reconstruction")
                task.finish_assets(
                    dict(
                        status="assets_selected",
                        reconstruction_scene=str(scene_dir),
                        foreground_library=str(foreground / "library.json"),
                        support_observation=observation,
                    )
                )

                checkpoints.save(task, "objects", task.stage("objects"), config)
                task.seal()
            else:
                scene_dir = task.stage("objects") / "reconstruction"
                foreground = task.stage("objects") / "foreground_assets"
                observation = (lib.read_json(task.stage("objects") /
                                            "support_observation/support_observation.json")
                               if support_mode == "retrieved" else None)

            task.seal()
            phase = "scene"
            if support_mode == "upstream":
                return finish_upstream_scene(task, scene_dir, foreground, config, report)
            cached_scene = checkpoints.reusable(task, "scene", task.stage("scene"), config)
            if cached_scene is None:
                task.start_scene()
                with tempfile.TemporaryDirectory(prefix="media-scene-") as temporary:
                    temporary = Path(temporary)
                    base = temporary / "foreground"
                    scene_import.convert(
                        scene_dir, foreground / "library.json", base, pose_format="og"
                    )
                    foreground_layout = scene_import.verify(base)
                    query, candidates, rejected = support.rank(
                        clip_index,
                        task.stage("objects") / "support_observation/support_crop.png",
                        observation["category"],
                    )
                    index, _ = clip.load_index(clip_index)
                    preview_root = local_path(index["official_index"]["path"]).resolve().parent
                    chosen, vlm_evidence = support.choose(
                        candidates,
                        task.stage("objects") / "support_observation/support_crop.png",
                        preview_root,
                        vlm_config,
                        vlm=selector or native_service.NativeClient(vlm_config),
                    )
                    selection_dir = task.stage("scene") / "support_selection"
                    selection_dir.mkdir()
                    official.write_json(
                        selection_dir / "selection.json",
                        _selection_document(query, candidates, rejected, vlm_evidence, chosen),
                    )
                    footprint = support.target_footprint(observation, foreground_layout)
                    support_root = temporary / "support"
                    support.materialize(
                        chosen,
                        footprint,
                        task.stage("objects") / "support_observation/support_crop.png",
                        preview_root,
                        support_root,
                    )
                    augmented = temporary / "augmented"
                    support.augment_scene(base, support_root, observation, augmented)
                    shutil.copytree(augmented, task.stage("scene"), dirs_exist_ok=True)
                    preview_dir = temporary / "preview"
                    scene_import.preview(task.stage("scene"), preview_dir)
                    shutil.copytree(preview_dir, task.stage("scene") / "preview")
                task.finish_scene(
                    dict(
                        status="scene_built",
                        support_asset_id=chosen["asset_id"],
                        finite_support=True,
                        infinite_plane=False,
                        physics_status="not_run",
                    )
                )

                checkpoints.save(task, "scene", task.stage("scene"), config,
                                 dict(chosen=chosen, candidates=candidates,
                                      preview_root=str(preview_root)))
                task.seal()
            else:
                saved = cached_scene["result"]
                chosen, candidates = saved["chosen"], saved["candidates"]
                preview_root = Path(saved["preview_root"])

            task.seal()
            phase = "physics"
            workflow_dir = task.stage("physics") / "workflow"
            if not (resume and workflow_dir.exists()):
                if any(task.stage("physics").iterdir()):
                    archive = task.root / "attempt_history" / str(time.time_ns())
                    shutil.copytree(task.stage("physics"), archive)
                    task.seal()
                task.start_physics()
            result = workflow.run(task.stage("scene"), workflow_dir,
                                  resume=resume and workflow_dir.exists())
            passed = result["physics_status"] == "passed"
            if passed:
                workflow.verify(workflow_dir)
                active_scene = Path(result["scene_package"])
                official.write_json(task.stage("physics") / "validated_scene.json", dict(
                    schema_version="genenv.validated_media_scene.v1",
                    scene_package=str(active_scene),
                    scene_manifest_sha256=lib.sha256(active_scene / "manifest.json"),
                    physics_result=official.fingerprint(
                        Path(result["physics_result"]), task.root),
                    workflow_report=official.fingerprint(
                        workflow_dir / "workflow_report.json", task.root),
                    input_sha256=config["input_sha256"], attempts=2))
            report.update(status="physics_passed" if passed else "physics_failed",
                          exit_code=result["exit_code"], physics_status=result["physics_status"],
                          render_status="not_run", attempts=result["attempts"], repairs=[],
                          workflow_stages=result["stages"], elapsed_s=time.perf_counter()-started)
            task.report.update(report)
            task.finish_physics(report)
            if passed:
                phase = "render"
                render_validated(task)
                report.update(render_status="passed", exit_code=0)
                task.report.update(report)
                task.finish_physics(report)
            task.verify()
            return lib.read_json(task.root / "run_report.json")
        except BaseException as exc:
            report.update(
                status="execution_failed",
                exit_code=1,
                physics_status="passed" if phase == "render" else (
                    "invalid" if phase == "physics" else "not_run"),
                render_status="not_run",
                failure_phase=phase,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.perf_counter() - started,
            )
            task.report.update(report)
            stages = task.report["stages"]
            if phase in {"media", "service_preflight", "simfoundry",
                         "foreground_assets", "support_observation"}:
                stages.update(objects="failed", scene="not_run", physics="not_run",
                              final_render="not_run")
            elif phase == "scene":
                stages.update(scene="failed", physics="not_run", final_render="not_run")
            elif phase == "render":
                report.update(render_status="failed")
                task.report.update(report)
                task.report["physics"] = dict(report)
                stages.update(physics="passed", final_render="failed")
            else:
                stages.update(physics="invalid", final_render="not_run")
            official.write_json(
                task.root / "failure.json",
                dict(schema_version=SCHEMA, phase=phase, error=report["error"]),
            )
            final_render = task.stage("final_render")
            if final_render.exists():
                if any(final_render.iterdir()):
                    archive = task.stage("physics") / "render_failures" / str(time.time_ns())
                    archive.mkdir(parents=True)
                    shutil.move(str(final_render), str(archive / "partial_render"))
                    official.write_json(archive / "failure.json",
                        dict(phase=phase, error=report["error"]))
                else:
                    final_render.rmdir()
                final_render.mkdir()
            task.seal()
            return lib.read_json(task.root / "run_report.json")



def add_visual_support(package, observation_dir):
    """Attach a render-only support surface to an already-validated scene package.

    Added after physics rather than before, so there is no path by which it could have
    influenced the result it is drawn alongside.
    """
    record = lib.read_json(Path(observation_dir) / "support_observation.json")
    root = Path(package)
    layout = lib.read_json(root / "scene_layout.json")
    surface = visuals.build(record, layout["environment"]["z_m"], root / "visual_support.obj")
    official.write_json(root / "visual_support.json", surface)
    manifest = lib.read_json(root / "manifest.json")
    known = {f["path"] for f in manifest["files"]}
    for name in ("visual_support.json", "visual_support.obj"):
        if name not in known:
            manifest["files"].append(official.fingerprint(root / name, root))
    manifest["files"].sort(key=lambda f: f["path"])
    official.write_json(root / "manifest.json", manifest)
    return surface


def finish_upstream_scene(task, scene_dir, foreground, config, report):
    """Preserve source poses/ground; publish a preview without physical acceptance claims."""
    cached = checkpoints.reusable(task, "scene", task.stage("scene"), config)
    if cached is None:
        task.start_scene()
        with tempfile.TemporaryDirectory(prefix="upstream-scene-") as temporary:
            temporary = Path(temporary)
            package = temporary / "package"
            scene_import.convert(scene_dir, foreground / "library.json", package, pose_format="og")
            scene_import.preview(package, temporary / "preview", orbit=True)
            shutil.copytree(package, task.stage("scene"), dirs_exist_ok=True)
            shutil.copytree(temporary / "preview", task.stage("scene") / "preview")
        # Before finish_scene, which seals the stage: a package cannot be edited afterwards,
        # and it must be complete when physics records the manifest hash it validated.
        surface = None
        observation = task.stage("objects") / "support_observation"
        try:
            surface = add_visual_support(task.stage("scene"), observation)
        except (ValueError, OSError, KeyError) as exc:
            # Renders against the bare plane instead. A decoration must never be able to
            # block validation of the scene it decorates.
            report["visual_support_skipped"] = f"{type(exc).__name__}: {exc}"
        layout = scene_import.verify(task.stage("scene"))
        task.finish_scene(dict(status="scene_built", support_mode="upstream",
                               environment=layout["environment"], support_asset_added=False,
                               visual_support=surface, physics_status="not_run"))
        checkpoints.save(task, "scene", task.stage("scene"), config)
    else:
        scene_import.verify(task.stage("scene"))
    report.update(status="scene_built", exit_code=0, support_mode="upstream",
                  stages=dict(task.report["stages"]), preview_status="passed",
                  preview_directory="02_scene/preview",
                  limitations=["Backgrounds, lighting and robots are not imported"])
    task.report.update(report)
    task.seal()
    # Free replay on the imported poses: the source placement is the claim under test, so
    # nothing is re-solved, settled or repaired. This used to be skipped because a scene
    # supported by the built-in plane could not be validated at all; it can now, so leaving
    # it unrun would report "not_run" for a stage that does work.
    phase = "physics"
    # A scene whose bodies rest on nothing has no support graph to validate against. That
    # is a fact about the scene, reported as such -- not an exception that kills the import,
    # and not a pass. Checked up front so a genuine error inside physics still propagates
    # instead of being absorbed into "not_run".
    try:
        graph_rules.topology(scene_import.verify(task.stage("scene")))
    except ValueError as exc:
        report.update(physics_status="not_run", render_status="not_run",
                      physics_not_run_reason=str(exc),
                      limitations=[*report["limitations"],
                                   f"Genesis physics not validated: {exc}"])
        task.report.update(report)
        task.seal()
        task.verify()
        return lib.read_json(task.root / "run_report.json")
    attempt = task.stage("physics") / "free_replay"
    if any(task.stage("physics").iterdir()):
        archive = task.root / "attempt_history" / str(time.time_ns())
        shutil.copytree(task.stage("physics"), archive)
        shutil.rmtree(attempt, ignore_errors=True)
        task.seal()
    task.start_physics()
    result, record = _physics_attempt(task.stage("scene"), attempt)
    passed = record["passed"]
    if passed:
        official.write_json(task.stage("physics") / "validated_scene.json", dict(
            schema_version="genenv.validated_media_scene.v1",
            scene_package=str(Path(task.stage("scene")).resolve()),
            scene_manifest_sha256=lib.sha256(task.stage("scene") / "manifest.json"),
            physics_result=official.fingerprint(
                attempt / "evidence/physics_result.json", task.root),
            # Absent when the entry point is an already-reconstructed scene rather
            # than source media; recorded as null instead of inventing an identity.
            input_sha256=config.get("input_sha256"), attempts=1))
    report.update(status="physics_passed" if passed else "physics_failed",
                  exit_code=result["exit_code"], physics_status=result["physics_status"],
                  render_status="not_run", attempts=[record],
                  stages=dict(task.report["stages"]))
    task.report.update(report)
    task.finish_physics(report)
    if passed:
        phase = "render"
        # The reference camera and the visual surface were both produced above, before
        # physics, so the package rendered here is byte-for-byte the validated one.
        render_validated(task, reconstruction=scene_dir)
        report.update(render_status="passed", exit_code=0)
        task.report.update(report)
        task.finish_physics(report)
    task.verify()
    return lib.read_json(task.root / "run_report.json")


def import_upstream_reconstruction(scene_dir, output_dir):
    """Create a four-stage task from existing upstream output, without asset retrieval."""
    scene_dir = Path(scene_dir).resolve()
    task = TaskOutput(output_dir)
    if task.root.exists():
        raise FileExistsError("use a new natural-language task directory")
    with task.lock():
        task.start(task.root.name)
        try:
            source_dir = task.stage("objects") / "source_outputs"
            source_dir.mkdir()
            inputs = []
            for relative in ("s11_sim/scene_objects_info.json",
                             "s12_physics/pb_scene_poses.json",
                             "s14_og/reconstructed_og_scene.json"):
                path = scene_dir / relative
                if relative.startswith("s12_physics/") and not path.is_file():
                    continue
                inputs.append(dict(path=str(path), sha256=lib.sha256(path)))
                target = source_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            foreground = task.stage("objects") / "foreground_assets"
            asset_import.import_assets(scene_dir, foreground)
            try:
                observation = support.observe(
                    scene_dir, task.stage("objects") / "support_observation")
            except (ValueError, OSError, KeyError):
                observation = None
            task.finish_assets(dict(status="assets_selected", reconstruction_scene=str(scene_dir),
                                    model_calls=0, retrieval_calls=0, source_files=inputs,
                                    support_observation=observation))
            config = dict(support_mode="upstream", source_files=inputs)
            return finish_upstream_scene(task, scene_dir, foreground, config,
                dict(schema_version=SCHEMA, mode="existing_reconstruction",
                     output_dir=str(task.root), stages=task.report["stages"]))
        except Exception as exc:
            for stage in ("objects", "scene"):
                if task.report["stages"][stage] == "running":
                    task.report["stages"][stage] = "failed"
            task.report.update(status="execution_failed", exit_code=1, error=str(exc),
                               physics_status="not_run", render_status="not_run")
            task.seal()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", type=Path)
    media.add_argument("--video", type=Path)
    media.add_argument("--reconstruction-scene", type=Path,
                       help="Convert an existing SimFoundry reconstruction without model calls")
    parser.add_argument("--name")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--clip-index", type=Path)
    parser.add_argument("--support-mode", choices=["upstream", "retrieved"],
                        default="upstream", help="Preserve source ground; stop at preview")
    parser.add_argument("--vlm-config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--simfoundry-arg", action="append", default=[],
        help="Repeat to forward an option before the Hydra override separator"
    )
    args = parser.parse_args(argv)
    if args.reconstruction_scene:
        if args.support_mode != "upstream" or args.resume:
            parser.error("existing reconstruction import requires upstream mode and a new output")
        name = args.name or f"将{args.reconstruction_scene.name}按原始支撑面转换为Genesis"
        output = destination(name, output_root=args.output_root)
        report = import_upstream_reconstruction(args.reconstruction_scene, output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(report["exit_code"])
    if args.vlm_config is None:
        parser.error("image/video reconstruction requires --vlm-config")
    mode, source = ("image", args.image) if args.image else ("video", args.video)
    name = args.name or f"{source.stem}_{mode}"
    output = destination(name, output_root=args.output_root)
    try:
        report = run(
            source, mode, output, args.clip_index, args.vlm_config,
            resume=args.resume, simfoundry_args=args.simfoundry_arg, support_mode=args.support_mode
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(report["exit_code"])
    except Exception as exc:
        print(json.dumps(
            dict(schema_version=SCHEMA, status="error", exit_code=1,
                 error=f"{type(exc).__name__}: {exc}"),
            ensure_ascii=False,
        ))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified image/video -> SimFoundry -> finite-support Genesis reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import import_simfoundry_assets as asset_import
from self_improving.sim_adapters.genesis import import_simfoundry_scene as scene_import
from self_improving.sim_adapters.genesis import media_support as support
from self_improving.sim_adapters.genesis import validate_imported_scene as physics
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import DEFAULT_ROOT, TaskOutput, destination

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


def task_config(mode, source, clip_index, vlm_config, simfoundry_args):
    return dict(
        schema_version=SCHEMA,
        mode=mode,
        input_sha256=file_hash(source),
        clip_index=str(Path(clip_index).resolve()),
        clip_index_sha256=file_hash(clip_index),
        vlm_config=str(Path(vlm_config).resolve()),
        simfoundry_args=list(simfoundry_args),
        single_image_inference=mode == "image",
        background_splat=False,
        sampled_frames=1 if mode == "image" else 15,
        repair_rounds=3,
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
    ]
    command.extend(extra)
    command.append("--")
    command.extend(
        [
            f"s1_video.single_image_input={'true' if mode == 'image' else 'false'}",
            f"s1_video.n_subsampled_frames={1 if mode == 'image' else 15}",
            f"s3_ground.img_idx={0 if mode == 'image' else 'auto'}",
            "s1_video.splat_prep=false",
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
):
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    config = task_config(mode, source, clip_index, vlm_config, simfoundry_args)
    request = f"{mode}:{source.name}"
    task = TaskOutput(output)
    if task.root.exists():
        if not resume:
            raise FileExistsError("output task exists; use --resume with identical inputs")
        prior = lib.read_json(task.stage("objects") / "input/media_manifest.json")
        if prior.get("effective_config") != config:
            raise ValueError("resume input or effective configuration mismatch")
        try:
            task.verify()
            report = lib.read_json(task.root / "run_report.json")
            if report["status"] in {"physics_passed", "physics_failed"}:
                return report
        except (OSError, ValueError):
            task.owner(request)
    with task.lock():
        task.start(request)
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
            repair_budget=3,
        )
        phase = "media"
        try:
            input_dir = task.stage("objects") / "input"
            original, canonical, runner_input = _normalize(source, mode, input_dir)
            media = probe_media(original, mode)
            media.update(
                canonical_input=official.fingerprint(canonical, task.root),
                effective_config=config,
                limitations=(
                    ["depth, hidden geometry and physical properties are inferred from one image"]
                    if mode == "image" else []
                ),
            )
            official.write_json(input_dir / "media_manifest.json", media)
            command = simfoundry_command(task.stage("objects"), runner_input, mode, simfoundry_args)
            official.write_json(
                input_dir / "simfoundry_invocation.json",
                dict(command=command, background_splat=False, expected_scene="reconstruction"),
            )
            if file_hash(source) != config["input_sha256"]:
                raise ValueError("input media changed before reconstruction")
            phase = "simfoundry"
            completed = process_runner(command, cwd=REPO_ROOT, check=False)
            if completed.returncode:
                raise RuntimeError(f"SimFoundry exited with {completed.returncode}")
            scene_dir = task.stage("objects") / "reconstruction"
            if not (scene_dir / "s14_og/reconstructed_og_scene.json").is_file():
                raise ValueError("SimFoundry did not produce the stage-14 scene")
            phase = "foreground_assets"
            foreground = task.stage("objects") / "foreground_assets"
            asset_import.import_assets(scene_dir, foreground)
            phase = "support_observation"
            observation = support.observe(
                scene_dir, task.stage("objects") / "support_observation"
            )
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

            phase = "scene"
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
                    vlm=selector,
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

            phase = "physics"
            task.start_physics()
            attempts, repairs = [], []
            active_scene = task.stage("scene")
            result, record = _physics_attempt(
                active_scene, task.stage("physics") / "attempts/000"
            )
            attempts.append(record)
            selected_order = [chosen] + [
                candidate for candidate in candidates
                if candidate["asset_id"] != chosen["asset_id"]
            ]
            if result.get("physics_status") != "passed" and len(selected_order) > 1:
                candidate = selected_order[1]
                directory = task.stage("physics") / "attempts/001"
                with tempfile.TemporaryDirectory(prefix="media-support-repair-") as temporary:
                    temporary = Path(temporary)
                    base = temporary / "foreground"
                    scene_import.convert(
                        scene_dir, foreground / "library.json", base, pose_format="og"
                    )
                    footprint = support.target_footprint(observation, scene_import.verify(base))
                    package = directory / "support_asset"
                    support.materialize(
                        candidate,
                        footprint,
                        task.stage("objects") / "support_observation/support_crop.png",
                        preview_root,
                        package,
                    )
                    repaired_scene = directory / "scene_package"
                    support.augment_scene(base, package, observation, repaired_scene)
                result, record = _physics_attempt(
                    repaired_scene, directory / "run"
                )
                attempts.append(record)
                active_scene = repaired_scene
                repairs.append(
                    dict(round=1, action="next_qualified_support_candidate",
                         asset_id=candidate["asset_id"])
                )
            if result.get("physics_status") != "passed":
                directory = task.stage("physics") / "attempts/002"
                repaired_scene = directory / "scene_package"
                repairs_for_scene = clearance_scene(active_scene, repaired_scene)
                repairs.append(
                    dict(round=2, action="normal_clearance", changes=repairs_for_scene,
                         maximum_lift_m=0.01)
                )
                result, record = _physics_attempt(repaired_scene, directory / "run")
                attempts.append(record)
                active_scene = repaired_scene
            if result.get("physics_status") != "passed":
                directory = task.stage("physics") / "attempts/003"
                multiplier = 1.25 if tangential_failure(result) else 1.0
                repairs.append(
                    dict(round=3, action="half_dt", friction_multiplier=multiplier,
                         physical_duration_preserved=True)
                )
                result, record = _physics_attempt(
                    active_scene, directory, profile="half_dt",
                    friction_multiplier=multiplier
                )
                attempts.append(record)
            official.write_json(task.stage("physics") / "attempts.json", attempts)
            official.write_json(task.stage("physics") / "repair_log.json", repairs)
            passed = result.get("physics_status") == "passed"
            render_status = "not_run"
            if passed:
                final_attempt = Path(attempts[-1]["directory"])
                evidence_dir = (
                    final_attempt / "evidence"
                    if (final_attempt / "evidence").is_dir()
                    else final_attempt / "run/evidence"
                )
                validated = dict(
                    schema_version="genenv.validated_media_scene.v1",
                    scene_package=str(active_scene),
                    scene_manifest_sha256=lib.sha256(active_scene / "manifest.json"),
                    physics_result=official.fingerprint(
                        evidence_dir / "physics_result.json", task.root
                    ),
                    input_sha256=config["input_sha256"],
                    attempts=len(attempts),
                )
                official.write_json(
                    task.stage("physics") / "validated_scene.json", validated
                )
                render = task.stage("final_render") / "render"
                render.mkdir(parents=True)
                for name in ("simulation.mp4", "frame_samples.json"):
                    shutil.copyfile(evidence_dir / name, render / name)
                shutil.copytree(evidence_dir / "diagnostics", render / "views")
                render_status = "passed"
            report.update(
                status="physics_passed" if passed else "physics_failed",
                exit_code=0 if passed else 2,
                physics_status="passed" if passed else "failed",
                render_status=render_status,
                attempts=attempts,
                repairs=repairs,
                elapsed_s=time.perf_counter() - started,
            )
            task.finish_physics(report)
            task.verify()
            return lib.read_json(task.root / "run_report.json")
        except BaseException as exc:
            report.update(
                status="execution_failed",
                exit_code=1,
                physics_status="invalid" if phase == "physics" else "not_run",
                render_status="not_run",
                failure_phase=phase,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.perf_counter() - started,
            )
            task.report.update(report)
            stages = task.report["stages"]
            if phase in {"media", "simfoundry", "foreground_assets", "support_observation"}:
                stages.update(objects="failed", scene="not_run", physics="not_run",
                              final_render="not_run")
            elif phase == "scene":
                stages.update(scene="failed", physics="not_run", final_render="not_run")
            else:
                stages.update(physics="invalid", final_render="not_run")
            official.write_json(
                task.root / "failure.json",
                dict(schema_version=SCHEMA, phase=phase, error=report["error"]),
            )
            final_render = task.stage("final_render")
            if final_render.exists():
                shutil.rmtree(final_render)
                final_render.mkdir()
            task.seal()
            return lib.read_json(task.root / "run_report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", type=Path)
    media.add_argument("--video", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--clip-index", type=Path, required=True)
    parser.add_argument("--vlm-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--simfoundry-arg", action="append", default=[],
        help="Repeat to forward an option before the Hydra override separator"
    )
    args = parser.parse_args(argv)
    mode, source = ("image", args.image) if args.image else ("video", args.video)
    name = args.name or f"{source.stem}_{mode}"
    output = destination(name, output_root=args.output_root)
    try:
        report = run(
            source, mode, output, args.clip_index, args.vlm_config,
            resume=args.resume, simfoundry_args=args.simfoundry_arg
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

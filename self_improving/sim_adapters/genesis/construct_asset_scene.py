"""Construct new, independently owned text scenes from already selected assets."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets as preparation
from self_improving.sim_adapters.genesis import repair_geometry as geometry
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video
from self_improving.sim_adapters.genesis.extract_assets import verified_binding
from self_improving.sim_adapters.genesis.task_output import TaskOutput

SCHEMA = "genenv.text_repair_run.v1"
REPAIR_PRESETS = ("legacy", "text_scene_v2")


def preferences(document):
    result = []
    words = {"偏左": "left", "偏右": "right", "靠前": "front", "靠后": "back", "中间": "center"}
    for obj in document["objects"]:
        description = obj["description"]
        for clause in re.split("[，。；,;]", document["request"]):
            if any(m in clause for m in obj.get("mentions", [])):
                description += " " + clause
        for word, region in words.items():
            if word in description:
                result.append(dict(object_id=obj["object_id"], region=region, source="text_soft"))
    return result


def bind_inputs(source, clip_index):
    document = lib.read_json(source.stage("objects") / "asset_request.json")
    index, _ = clip.load_index(clip_index)
    digest = lib.sha256(clip_index)
    bindings = {
        obj["object_id"]: verified_binding(
            obj, source.stage("objects") / "asset_selection" / obj["object_id"], index, digest
        )
        for obj in document["objects"]
    }
    return document, bindings


def make_initial(assets, graph, prefs, seed, out):
    poses, records = {}, []
    for n in graph["order"]:
        selected = None
        for attempt in range(11):
            selected, record = geometry.sample_object(
                assets, poses, n, graph["relations"], prefs, seed, attempt
            )
            records.append(record)
            if selected is not None:
                break
        clip.write_json(out / "candidate_sampling.json", records)
        if selected is None:
            raise PlacementFailure(f"{n}: no geometrically valid initial candidate")
        poses[n] = selected["pose"]
    return poses


class PlacementFailure(ValueError):
    pass


class PhysicsEvidenceChanged(ValueError):
    pass


def verify_artifacts(files):
    for path, digest in files.items():
        if not path.is_file() or lib.sha256(path) != digest:
            raise PhysicsEvidenceChanged(f"verified physics evidence changed: {path.name}")


def contact_dynamics_failed(result):
    """A new XY placement cannot isolate failures limited to contact dynamics."""
    failures = {reason for reasons in result["failures"].values() for reason in reasons}
    return bool(result["complete"] and failures and failures <= {
        "stable_velocity", "support_preserved"
    })


def repair_loop(assets, poses, graph, prefs, seed, out, check, *, simulator, recorder,
                profile=geometry.PROFILE, numerics_profile="legacy", repair_preset="legacy"):
    attempts, counts = [], {n: 0 for n in assets}
    dynamic = [n for n in graph["order"] if not assets[n]["fixed"]]
    budget = 1 + 10 * len(dynamic)
    repair_records = []
    current = copy.deepcopy(poses)
    result = None
    for trial in range(budget):
        directory = out / "attempts" / f"{trial:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        options = dict(profile=profile)
        if numerics_profile != "legacy":
            options["numerics_profile"] = numerics_profile
        if repair_preset != "legacy":
            options["repair_preset"] = repair_preset
        data = physics.frozen_input(assets, current, graph["relations"], seed, **options)
        input_path = directory / "physics_input.json"
        clip.write_json(input_path, data)
        input_hash = lib.sha256(input_path)
        verified_artifacts = {}

        def check_trial():
            check()
            verify_artifacts(verified_artifacts)
            if lib.sha256(input_path) != input_hash:
                raise ValueError("frozen physics input changed during trial")

        print(f"physics attempt {trial + 1}/{budget}", flush=True)
        try:
            result = simulator(data, directory, check_trial)
            check_trial()
            # Independently validate persisted evidence before accepting the verdict.
            rows = [json.loads(line) for line in (directory / "trace.jsonl").open()]
            reevaluated = physics.evaluate(data, rows, initial_rejected=not result["complete"])
            if reevaluated != result:
                raise ValueError("in-memory and persisted physical verdict mismatch")
        except BaseException as exc:
            last = None
            trace = directory / "trace.jsonl"
            if trace.exists():
                for line in trace.read_text().splitlines():
                    try:
                        last = json.loads(line)
                    except ValueError:
                        break
            aborted = dict(
                directory=str(directory),
                passed=False,
                execution_status="error",
                error=f"{type(exc).__name__}: {exc}",
                input_sha256=input_hash,
                steps_executed=(
                    lib.read_json(directory / "asset_physics_report.json").get(
                        "steps_executed", 0 if last is None else last["step"]
                    )
                    if (directory / "asset_physics_report.json").exists()
                    else 0
                    if last is None
                    else last["step"]
                ),
                trace_sha256=lib.sha256(trace) if trace.exists() else None,
            )
            clip.write_json(directory / "execution_error.json", aborted)
            attempts.append(aborted)
            clip.write_json(out / "attempts.json", attempts)
            raise
        result.update(input_sha256=input_hash, trace_sha256=lib.sha256(directory / "trace.jsonl"))
        clip.write_json(directory / "validation_result.json", result)
        verified_artifacts = {p: lib.sha256(p) for p in directory.rglob("*") if p.is_file()}
        attempts.append(
            dict(
                directory=str(directory),
                passed=result["passed"],
                result_sha256=lib.sha256(directory / "validation_result.json"),
                steps_executed=result["steps_executed"],
            )
        )
        clip.write_json(out / "attempts.json", attempts)
        if len(rows) > 1:
            try:
                recorder(
                    data,
                    directory / "trace.jsonl",
                    directory / "video",
                    check_trial,
                    physics_passed=result["passed"],
                )
            except BaseException as exc:
                attempts[-1].update(execution_status="media_error", error=str(exc))
                clip.write_json(out / "attempts.json", attempts)
                clip.write_json(directory / "media_error.json", attempts[-1])
                raise
        else:
            clip.write_json(
                directory / "video_not_generated.json",
                dict(reason="initial state rejected; no sequential physical motion"),
            )
        check_trial()
        durable = [json.loads(line) for line in (directory / "trace.jsonl").open()]
        final_verdict = physics.evaluate(data, durable, initial_rejected=not result["complete"])
        if any(result.get(k) != value for k, value in final_verdict.items()):
            raise PhysicsEvidenceChanged("physics verdict changed before sealing attempt")
        files = [
            official.fingerprint(p, directory) for p in sorted(directory.rglob("*")) if p.is_file()
        ]
        clip.write_json(directory / "manifest.json", dict(files=files))
        if result["passed"]:
            return result, attempts, current, data
        if repair_preset == "text_scene_v2" and contact_dynamics_failed(result):
            attempts[-1]["stop_reason"] = "CONTACT_DYNAMICS_FAILED"
            clip.write_json(out / "attempts.json", attempts)
            break
        failed = next((n for n in dynamic if result["failures"][n]), None)
        if failed is None or counts[failed] >= 10:
            break
        selected = None
        while counts[failed] < 10:
            counts[failed] += 1
            selected, record = geometry.sample_object(
                assets, current, failed, graph["relations"], prefs, seed, counts[failed]
            )
            record["moved_subtree"] = sorted(geometry.subtree(assets, failed))
            repair_records.append(record)
            clip.write_json(out / "repair_log.json", repair_records)
            if selected is not None:
                break
        if selected is None:
            break
        current = geometry.move_subtree(assets, current, failed, selected["pose"])
    return result, attempts, current, None


def run(
    source_task,
    output_dir,
    clip_index,
    *,
    fixed_objects=(),
    seed=0,
    profile=geometry.PROFILE,
    numerics_profile="legacy",
    repair_preset="legacy",
    render=False,
    asset_metadata=None,
    preparer=preparation.prepare,
    simulator=physics.simulate,
    recorder=video.render,
):
    if profile not in physics.PROFILES or seed < 0:
        raise ValueError("unsupported profile or negative seed")
    if repair_preset not in REPAIR_PRESETS:
        raise ValueError("unsupported repair preset")
    if repair_preset == "text_scene_v2" and profile != geometry.PROFILE:
        raise ValueError("text_scene_v2 requires the text_repair_v1 acceptance profile")
    physics.settings(profile, numerics_profile=numerics_profile)
    source, task = TaskOutput(source_task), TaskOutput(output_dir)
    clip.separate(source.root, task.root, Path(clip_index).resolve().parent, preparation.CACHE)
    started = time.perf_counter()
    with source.lock(), task.lock():
        original = source.verify()
        if source.report["stages"]["objects"] != "passed":
            raise ValueError("source task requires selected assets")
        if task.root.exists():
            raise FileExistsError("output must be a new task directory")
        document, bindings = bind_inputs(source, clip_index)
        fixed = set(fixed_objects)
        if len(fixed) != len(fixed_objects) or fixed - set(bindings):
            raise ValueError("duplicate or unknown fixed object")
        metadata = {} if asset_metadata is None else lib.read_json(asset_metadata)
        if set(metadata) - set(bindings):
            raise ValueError("metadata refers to unknown object")
        index_hash = lib.sha256(clip_index)
        metadata_hash = None if asset_metadata is None else lib.sha256(asset_metadata)

        def check_source():
            source.owner()
            official.verify_files(source.root, original["files"])
            if lib.sha256(clip_index) != index_hash:
                raise ValueError("asset index changed")
            if asset_metadata is not None and lib.sha256(asset_metadata) != metadata_hash:
                raise ValueError("asset metadata changed")
            for b in bindings.values():
                official.verify_files(Path(b["source_root"]), b["source_files"])

        task.start(document["request"])
        shutil.copytree(source.stage("objects"), task.stage("objects"), dirs_exist_ok=True)
        task.finish_assets(
            dict(
                status="assets_selected",
                source_task=str(source.root),
                model_calls=0,
                retrieval_calls=0,
            )
        )
        report = dict(
            schema_version=SCHEMA,
            profile=profile,
            status="running",
            exit_code=1,
            physics_status="not_run",
            render_status="not_run",
            random_seed=seed,
            simulation_executed=False,
            steps_executed=0,
            source_task=str(source.root),
            model_calls=0,
            retrieval_calls=0,
        )
        if repair_preset != "legacy" or numerics_profile != "legacy":
            report.update(repair_preset=repair_preset, numerics_profile=numerics_profile)
        stage = "scene"
        phase = "asset_preparation"
        try:
            task.start_scene()
            clip.write_json(
                task.stage("scene") / "source_manifest.json",
                dict(source_root=str(source.root), files=original["files"]),
            )
            preparation_options = (
                {} if repair_preset == "legacy" else dict(repair_preset=repair_preset)
            )
            assets = preparer(document, bindings, task.stage("scene"), fixed, metadata,
                              check_source, **preparation_options)
            phase = "scene"
            graph = geometry.graph(document, assets)
            prefs = preferences(document)
            poses = make_initial(assets, graph, prefs, seed, task.stage("scene"))
            scene = dict(
                schema_version="genenv.text_repair_scene.v1",
                profile=profile,
                assets=assets,
                poses=poses,
                graph=graph,
                preferences=prefs,
                random_seed=seed,
                metadata=metadata,
                metadata_sha256=metadata_hash,
            )
            if repair_preset != "legacy" or numerics_profile != "legacy":
                scene.update(repair_preset=repair_preset, numerics_profile=numerics_profile)
            clip.write_json(task.stage("scene") / "scene_v0.json", scene)
            task.finish_scene(dict(status="scene_built", profile=profile))
            stage = "physics"
            phase = "physics"
            task.start_physics()
            sealed_physics = {}

            def check():
                check_source()
                verify_artifacts(sealed_physics)
                task.verify_physics_inputs()
                for a in assets.values():
                    official.verify_files(Path(a["derived_root"]), a["derived_files"])

            result, attempts, last_poses, passing_data = repair_loop(
                assets,
                poses,
                graph,
                prefs,
                seed,
                task.stage("physics"),
                check,
                simulator=simulator,
                recorder=recorder,
                profile=profile,
                numerics_profile=numerics_profile,
                repair_preset=repair_preset,
            )
            passed = result is not None and result["passed"]
            report.update(
                status="physics_passed" if passed else "physics_failed",
                physics_status="passed" if passed else "failed",
                exit_code=0 if passed else 2,
                failure_kind=(None if passed else "CONTACT_DYNAMICS_FAILED"
                              if repair_preset == "text_scene_v2" and result is not None
                              and contact_dynamics_failed(result) else "PLACEMENT_FAILED"),
                attempts=attempts,
                final_metrics=result,
                simulation_executed=any(a["steps_executed"] for a in attempts),
                steps_executed=sum(a["steps_executed"] for a in attempts),
            )
            if passed:
                last = Path(attempts[-1]["directory"])
                sealed_physics = {p: lib.sha256(p) for p in last.rglob("*") if p.is_file()}
                state = lib.read_json(last / "final_state.json")["state"]
                validated = dict(
                    schema_version="genenv.validated_asset_scene.v1",
                    assets=assets,
                    transforms=state["objects"],
                    scene_graph=graph,
                    physics_properties=passing_data["settings"],
                    validation_metrics=result,
                    random_seed=seed,
                    source_trace_sha256=lib.sha256(last / "trace.jsonl"),
                    source_input_sha256=lib.sha256(last / "physics_input.json"),
                )
                if repair_preset != "legacy" or numerics_profile != "legacy":
                    validated.update(repair_preset=repair_preset,
                                     numerics_profile=numerics_profile, profile=profile)
                clip.write_json(task.stage("physics") / "validated_scene.json", validated)
                if render:
                    stage = "final_render"
                    phase = "final_render"
                    # Authorize 04 only after physical acceptance has been established.
                    task.report["stages"]["physics"] = "passed"
                    recorder(
                        passing_data,
                        last / "trace.jsonl",
                        task.stage("final_render") / "render",
                        check,
                        final=True,
                        physics_passed=True,
                    )
                    report["render_status"] = "passed"
            check()
            if passed:
                durable = [json.loads(line) for line in (last / "trace.jsonl").open()]
                final_verdict = physics.evaluate(passing_data, durable)
                if any(result.get(k) != value for k, value in final_verdict.items()):
                    raise PhysicsEvidenceChanged("physics verdict changed before sealing task")
        except PlacementFailure as exc:
            report.update(
                status=("scene_failed" if repair_preset == "text_scene_v2"
                        and stage == "scene" else "physics_failed"),
                exit_code=2,
                error=str(exc),
                failure_kind="PLACEMENT_FAILED",
            )
            if stage == "scene":
                task.report["stages"]["scene"] = "failed"
        except BaseException as exc:
            status = "physics_failed" if stage != "final_render" else "physics_passed"
            if repair_preset == "text_scene_v2":
                status = ("preparation_failed" if phase == "asset_preparation" else
                          "scene_failed" if stage == "scene" else "execution_failed")
            report.update(
                status=status,
                exit_code=1,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind=phase if repair_preset == "text_scene_v2" else stage,
            )
            if stage == "scene":
                task.report["stages"]["scene"] = "failed"
            elif stage == "physics":
                report["physics_status"] = (
                    "invalid" if repair_preset == "text_scene_v2" else "failed"
                )
                path = task.stage("physics") / "attempts.json"
                attempts = lib.read_json(path) if path.exists() else []
                report.update(
                    attempts=attempts,
                    steps_executed=sum(a["steps_executed"] for a in attempts),
                    simulation_executed=any(a["steps_executed"] for a in attempts),
                )
            else:
                report["render_status"] = "failed"
            if isinstance(exc, PhysicsEvidenceChanged):
                report.update(
                    status=("execution_failed" if repair_preset == "text_scene_v2"
                            else "physics_failed"),
                    physics_status="invalid" if repair_preset == "text_scene_v2" else "failed",
                    render_status="not_run", failure_kind="integrity",
                )
                diagnostics = task.stage("physics") / "diagnostics"
                if any(task.stage("final_render").iterdir()):
                    diagnostics.mkdir(parents=True, exist_ok=True)
                    shutil.move(
                        str(task.stage("final_render")), diagnostics / "rejected_final_render"
                    )
                    task.stage("final_render").mkdir()
                validated = task.stage("physics") / "validated_scene.json"
                if validated.exists():
                    diagnostics.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(validated), diagnostics / "rejected_validated_scene.json")
        finally:
            report["total_s"] = time.perf_counter() - started
            task.report.update(status=report["status"], construction=report)
            if stage != "scene":
                task.report["stages"].update(
                    physics=report["physics_status"], final_render=report["render_status"]
                )
                clip.write_json(task.stage("physics") / "physics_result.json", report)
            else:
                clip.write_json(task.stage("scene") / "construction_result.json", report)
            task.seal()
        task.verify()
        check_source()
        return report


class InputParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: {message}\n")


def main(argv=None):
    parser = InputParser(description=__doc__)
    parser.add_argument("--source-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-index", type=Path, required=True)
    parser.add_argument("--fixed-object", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=physics.PROFILES, default=geometry.PROFILE)
    parser.add_argument("--numerics-profile", choices=physics.NUMERICS_PROFILES, default="legacy")
    parser.add_argument("--repair-preset", choices=REPAIR_PRESETS, default="legacy")
    parser.add_argument("--asset-metadata", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.source_task,
            args.output_dir,
            args.clip_index,
            fixed_objects=args.fixed_object,
            seed=args.seed,
            profile=args.profile,
            numerics_profile=args.numerics_profile,
            repair_preset=args.repair_preset,
            render=args.render,
            asset_metadata=args.asset_metadata,
        )
        print(f"{report['status']}: {report.get('error', report.get('failure_kind'))}", flush=True)
        return report["exit_code"]
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay a sealed passing text_scene_v2 release with its registered half timestep."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import TaskOutput

SCHEMA = "genenv.text_half_dt_replay.v1"


class InputChanged(ValueError):
    pass


def half_input(baseline):
    """Change only registered numerical settings; retain the exact original release."""
    physics.validate_settings(baseline)
    if baseline.get("repair_preset") != "text_scene_v2" or baseline["profile"] != geo.PROFILE:
        raise ValueError(
            "half-dt replay requires text_scene_v2 with the 95 percent acceptance profile"
        )
    result = copy.deepcopy(baseline)
    name = numerics.half_dt_profile(baseline["numerics_profile"])
    result.update(numerics_profile=name, settings=physics.settings(baseline["profile"], name))
    physics.validate_settings(result)
    return result


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def baseline_evidence(source):
    manifest = source.verify()
    report = source.report
    construction = report.get("construction", {})
    if (
        report["status"] != "physics_passed"
        or report["stages"]["physics"] != "passed"
        or construction.get("repair_preset") != "text_scene_v2"
    ):
        raise ValueError("baseline task must be a sealed passing text_scene_v2 construction")
    attempts = construction.get("attempts", [])
    if not attempts or not attempts[-1]["passed"]:
        raise ValueError("baseline has no passing final construction attempt")
    directory = Path(attempts[-1]["directory"]).resolve()
    if not directory.is_relative_to(source.stage("physics")):
        raise ValueError("baseline passing attempt is outside its task")
    paths = {
        name: directory / filename
        for name, filename in (
            ("input", "physics_input.json"),
            ("trace", "trace.jsonl"),
            ("result", "validation_result.json"),
            ("attempt_manifest", "manifest.json"),
        )
    }
    input_data = lib.read_json(paths["input"])
    half_input(input_data)
    saved = lib.read_json(paths["result"])
    evaluated = physics.evaluate(input_data, read_rows(paths["trace"]))
    if not evaluated["passed"] or any(saved.get(k) != value for k, value in evaluated.items()):
        raise ValueError("baseline persisted trace does not independently pass")
    if (
        saved.get("input_sha256") != lib.sha256(paths["input"])
        or saved.get("trace_sha256") != lib.sha256(paths["trace"])
        or attempts[-1]["result_sha256"] != lib.sha256(paths["result"])
    ):
        raise ValueError("baseline input, trace or result binding mismatch")
    official.verify_files(directory, lib.read_json(paths["attempt_manifest"])["files"])
    provenance = dict(
        source_task=str(source.root),
        source_manifest_sha256=lib.sha256(source.root / "manifest.json"),
        baseline_numerics_profile=input_data["numerics_profile"],
        baseline_release_policy="original passing attempt physics_input; never final_state",
        files={name: dict(path=str(path), sha256=lib.sha256(path)) for name, path in paths.items()},
    )
    return input_data, manifest, provenance


def run(source_task, output_dir, *, render=False, simulator=None, recorder=None):
    simulator = physics.simulate if simulator is None else simulator
    recorder = video.render if recorder is None else recorder
    source = TaskOutput(source_task)
    with source.lock():
        original, source_manifest, provenance = baseline_evidence(source)
    data = half_input(original)
    target = source.copy_for_physics(output_dir)
    with target.lock():
        target.start_physics()
        # The copied construction described the source attempt, not this independent replay.
        target.report.pop("construction", None)
        out = target.stage("physics")
        attempt = out / "attempts" / "000"
        attempt.mkdir(parents=True)
        clip.write_json(out / "source_baseline.json", provenance)
        clip.write_json(attempt / "physics_input.json", data)
        input_hash = lib.sha256(attempt / "physics_input.json")
        report = dict(
            schema_version=SCHEMA,
            status="execution_failed",
            physics_status="invalid",
            render_status="not_run",
            exit_code=1,
            failure_kind="physics",
            profile=data["profile"],
            repair_preset=data["repair_preset"],
            numerics_profile=data["numerics_profile"],
            random_seed=data["random_seed"],
            source_baseline=provenance,
            release_unchanged=True,
            placement_calls=0,
            model_calls=0,
            retrieval_calls=0,
            simulation_executed=False,
            steps_executed=0,
            physics_input_sha256=input_hash,
        )

        verified_artifacts = {}
        verified_steps = None

        def check():
            try:
                if (
                    lib.sha256(source.root / "manifest.json")
                    != provenance["source_manifest_sha256"]
                ):
                    raise ValueError("baseline manifest changed")
                official.verify_files(source.root, source_manifest["files"])
                target.verify_physics_inputs()
                for path, expected_hash in verified_artifacts.items():
                    if lib.sha256(path) != expected_hash:
                        raise ValueError(f"verified half-dt evidence changed: {path.name}")
                for item in provenance["files"].values():
                    if (
                        "archived_path" in item
                        and lib.sha256(out / item["archived_path"]) != item["sha256"]
                    ):
                        raise ValueError("archived baseline evidence changed")
                if (
                    "archived_manifest" in provenance
                    and lib.sha256(out / provenance["archived_manifest"])
                    != provenance["source_manifest_sha256"]
                ):
                    raise ValueError("archived baseline manifest changed")
                if lib.sha256(attempt / "physics_input.json") != input_hash:
                    raise ValueError("half-dt frozen input changed")
                for a in data["assets"].values():
                    for root_key, records_key in (
                        ("source_root", "source_files"),
                        ("derived_root", "derived_files"),
                    ):
                        official.verify_files(local_path(a[root_key]), a[records_key])
            except Exception as exc:
                raise InputChanged(str(exc)) from exc

        rows, result = [], None
        phase = "physics"
        try:
            archive = out / "source_baseline"
            archive.mkdir()
            filenames = dict(
                input="physics_input.json",
                trace="trace.jsonl",
                result="validation_result.json",
                attempt_manifest="attempt_manifest.json",
            )
            for name, item in provenance["files"].items():
                destination = archive / filenames[name]
                shutil.copyfile(item["path"], destination)
                if lib.sha256(destination) != item["sha256"]:
                    raise InputChanged("archived baseline evidence changed during copy")
                item["archived_path"] = destination.relative_to(out).as_posix()
            shutil.copyfile(source.root / "manifest.json", archive / "task_manifest.json")
            if lib.sha256(archive / "task_manifest.json") != provenance["source_manifest_sha256"]:
                raise InputChanged("archived baseline manifest changed during copy")
            provenance["archived_manifest"] = "source_baseline/task_manifest.json"
            clip.write_json(out / "source_baseline.json", provenance)
            check()
            result = simulator(data, attempt, check)
            rows = read_rows(attempt / "trace.jsonl")
            evaluated = physics.evaluate(data, rows, initial_rejected=not result["complete"])
            if evaluated != result:
                raise InputChanged("half-dt in-memory and persisted verdict differ")
            check()
            result.update(input_sha256=input_hash, trace_sha256=lib.sha256(attempt / "trace.jsonl"))
            clip.write_json(attempt / "validation_result.json", result)
            verified_artifacts = {p: lib.sha256(p) for p in attempt.rglob("*") if p.is_file()}
            verified_steps = evaluated["steps_executed"]
            report.update(
                status="physics_passed" if result["passed"] else "physics_failed",
                physics_status="passed" if result["passed"] else "failed",
                exit_code=0 if result["passed"] else 2,
                failure_kind=None if result["passed"] else "physical",
                final_metrics=result,
                simulation_executed=result["simulation_executed"],
                steps_executed=result["steps_executed"],
            )
            phase = "physics_video"
            if len(rows) > 1:
                recorder(
                    data,
                    attempt / "trace.jsonl",
                    attempt / "video",
                    check,
                    physics_passed=result["passed"],
                )
            else:
                clip.write_json(
                    attempt / "video_not_generated.json",
                    dict(reason="initial state rejected; no sequential physical motion"),
                )
            check()
            if result["passed"]:
                graph = lib.read_json(target.stage("scene") / "scene_v0.json")["graph"]
                validated = dict(
                    schema_version="genenv.validated_asset_scene.v1",
                    assets=data["assets"],
                    transforms=rows[-1]["objects"],
                    scene_graph=graph,
                    physics_properties=data["settings"],
                    validation_metrics=result,
                    random_seed=data["random_seed"],
                    repair_preset=data["repair_preset"],
                    numerics_profile=data["numerics_profile"],
                    profile=data["profile"],
                    source_trace_sha256=lib.sha256(attempt / "trace.jsonl"),
                    source_input_sha256=input_hash,
                    source_baseline=provenance,
                )
                clip.write_json(out / "validated_scene.json", validated)
                if render:
                    phase = "final_render"
                    target.report["stages"]["physics"] = "passed"
                    recorder(
                        data,
                        attempt / "trace.jsonl",
                        target.stage("final_render") / "render",
                        check,
                        final=True,
                        physics_passed=True,
                    )
                    report["render_status"] = "passed"
            check()
            final_verdict = physics.evaluate(
                data, read_rows(attempt / "trace.jsonl"), initial_rejected=not result["complete"]
            )
            if any(result.get(k) != value for k, value in final_verdict.items()):
                raise InputChanged("half-dt verdict changed before sealing")
        except BaseException as exc:
            report.update(
                status="execution_failed",
                exit_code=1,
                failure_kind=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            if phase == "physics" or isinstance(exc, InputChanged):
                report["physics_status"] = "invalid"
            if phase == "final_render":
                report["render_status"] = "failed"
            if isinstance(exc, InputChanged):
                report["failure_kind"] = "integrity"
                if any(target.stage("final_render").iterdir()):
                    rejected = out / "diagnostics" / "rejected_final_render"
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target.stage("final_render")), rejected)
                    target.stage("final_render").mkdir()
                report["render_status"] = "not_run"
                validated = out / "validated_scene.json"
                if validated.exists():
                    rejected = out / "diagnostics" / "rejected_validated_scene.json"
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(validated), rejected)
            clip.write_json(
                attempt / "execution_error.json",
                dict(phase=phase, error=report["error"], failure_kind=report["failure_kind"]),
            )
        finally:
            trace = attempt / "trace.jsonl"
            if trace.exists():
                rows = []
                for line in trace.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        break
                if rows:
                    if not (attempt / "initial_state.json").exists():
                        clip.write_json(attempt / "initial_state.json", rows[0])
                    if not (attempt / "final_state.json").exists():
                        clip.write_json(
                            attempt / "final_state.json",
                            dict(
                                state=rows[-1],
                                complete=rows[-1]["step"] == data["settings"]["steps"],
                            ),
                        )
            load_report = attempt / "asset_physics_report.json"
            executed = (
                lib.read_json(load_report).get("steps_executed", 0)
                if load_report.exists()
                else rows[-1]["step"]
                if rows
                else 0
            )
            if verified_steps is not None:
                executed = verified_steps
            report.update(steps_executed=executed, simulation_executed=executed > 0)
            files = [
                official.fingerprint(p, attempt)
                for p in sorted(attempt.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            ]
            clip.write_json(attempt / "manifest.json", dict(files=files))
            item = dict(
                directory=str(attempt),
                passed=report["physics_status"] == "passed",
                execution_status="error" if report["exit_code"] == 1 else "complete",
                steps_executed=executed,
                input_sha256=input_hash,
                trace_sha256=lib.sha256(trace) if trace.exists() else None,
            )
            if (attempt / "validation_result.json").exists():
                item["result_sha256"] = lib.sha256(attempt / "validation_result.json")
            report["attempts"] = [item]
            clip.write_json(out / "attempts.json", [item])
            clip.write_json(out / "physics_result.json", report)
            target.finish_physics(report)
        target.verify()
    source.verify()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.source_task, args.output_dir, render=args.render)
        print(json.dumps(report, ensure_ascii=False))
        return report["exit_code"]
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

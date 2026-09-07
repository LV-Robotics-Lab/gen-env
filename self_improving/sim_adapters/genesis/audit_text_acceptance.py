"""Read-only, independent six-run acceptance audit for the complete text scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video
from self_improving.sim_adapters.genesis import replay_text_half_dt as half
from self_improving.sim_adapters.genesis import tune_text_physics as tune
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import TaskOutput, no_symlinks

SCHEMA = "genenv.complete_text_acceptance_audit.v1"
SCOPE = "complete_four_asset_acceptance"
SEEDS = (0, 42, 87)
OBJECTS = {f"{name}_1": name for name in ("table", "apple", "cup", "bowl")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    """Keep the original hashes throughout an audit, including external asset closures."""

    def __init__(self):
        self.files = {}

    def add(self, path):
        path = no_symlinks(local_path(str(path)))
        digest = lib.sha256(path)
        require(self.files.get(str(path), digest) == digest, f"evidence changed: {path}")
        self.files[str(path)] = digest
        return path

    def read(self, path):
        return lib.read_json(self.add(path))

    def records(self, root, records):
        root = no_symlinks(root)
        require(len({r["path"] for r in records}) == len(records), "duplicate manifest entry")
        official.verify_files(root, records)
        for record in records:
            self.add(root / record["path"])

    def check(self):
        for path, digest in self.files.items():
            require(lib.sha256(path) == digest, f"evidence changed during audit: {path}")


def inside(root, path):
    path = no_symlinks(path)
    require(path.is_relative_to(root), "evidence path escapes its owning directory")
    return path


def selected_profile(comparison, evidence):
    root = no_symlinks(comparison)
    if root.is_file():
        require(root.name == "comparison_result.json", "expected selected comparison result")
        root = root.parent
    report = evidence.read(root / "comparison_result.json")
    frozen = evidence.read(root / "comparison_input.json")
    require(all(report.get(k) == v for k, v in frozen.items()), "comparison freeze mismatch")
    profile = report["selected_profile"]
    require(
        report["status"] == "passed" and profile in numerics.CANDIDATE_PROFILES,
        "comparison has no registered passing selection",
    )
    require(
        report["seeds"] == list(SEEDS) and report["acceptance_profile"] == "text_repair_v1",
        "comparison seed or acceptance profile mismatch",
    )
    require(
        report["half_dt_profile"] == numerics.half_dt_profile(profile),
        "comparison half profile mismatch",
    )
    for mode, numerical in (("matrix", profile), ("sensitivity", report["half_dt_profile"])):
        records = report[mode][profile]
        require(set(records) == {str(s) for s in SEEDS}, "comparison requires all three seeds")
        for seed in SEEDS:
            record = records[str(seed)]
            trial = inside(root, Path(record["directory"]))
            require(
                record["passed"]
                and lib.sha256(evidence.add(trial / "trial_result.json"))
                == record["result_sha256"],
                "selected trial binding mismatch",
            )
            actual = tune.verify_trial(
                trial,
                numerics_profile=numerical,
                source_hash=frozen["source_input_sha256"][str(seed)],
                expected_implementation=frozen["implementation_sha256"],
            )
            require(actual["result"]["passed"], "selected comparison trial failed")
            data = evidence.read(trial / "physics_input.json")
            require(
                set(data["assets"]) == {"table_1", "cup_1"},
                "selected numerical comparison must be the table/cup isolation",
            )
            require(data["random_seed"] == seed, "comparison duplicate or wrong seed")
            evidence.records(trial, actual["artifacts"])
            source = evidence.read(trial / "source_manifest.json")
            evidence.add(source["source_input"])
            for asset in data["assets"].values():
                for prefix in ("source", "derived"):
                    evidence.records(local_path(asset[f"{prefix}_root"]), asset[f"{prefix}_files"])
    return profile


def asset_identity(asset, evidence):
    """Compare physical content, allowing task roots and diagnostic timings to differ."""
    bound = set()
    for prefix in ("source", "derived"):
        root = local_path(asset[f"{prefix}_root"])
        records = asset[f"{prefix}_files"]
        require(bool(records), "asset file closure is missing")
        evidence.records(root, records)
        bound.update(str((root / r["path"]).resolve()) for r in records)
    paths = {}
    for key in ("model_entrypoint", "physics_file", "geometry_file"):
        path = evidence.add(local_path(asset[key]))
        require(str(path) in bound, f"unbound asset physical file: {key}")
        paths[key] = lib.sha256(path)
    fields = (
        "asset_id",
        "category",
        "scale",
        "anchor_m",
        "native_pose",
        "fixed",
        "support",
        "hull",
        "collision_hulls",
        "mass_kg",
        "com_local_m",
        "inertia_local_kg_m2",
        "mass_source",
        "surface",
        "natural_up",
        "tip_limit_deg",
        "diagonal_m",
        "radius_m",
        "margin_m",
        "buffer_m",
        "bottom_offset_m",
        "visual_bottom_offset_m",
    )
    identity = {key: asset[key] for key in fields}
    identity.update(
        files=paths,
        source_files=asset["source_files"],
        collision_meshes=asset.get("collision_meshes"),
        collision={
            key: asset["collision"].get(key) for key in ("method", "policy", "key", "parts")
        },
    )
    require(asset["collision"]["quality"]["passed"], "asset collision qualification failed")
    require(
        "native" in asset["mass_source"] or "authored" in asset["mass_source"],
        "native mass properties were not preserved",
    )
    # The pre-candidate native freeze is required even for rejected historical candidates.
    native_path = local_path(asset["derived_root"]) / "native_mass_properties.json"
    require(str(native_path.resolve()) in bound, "native mass freeze is unbound")
    native = evidence.read(native_path)
    for key in ("mass_kg", "com_local_m", "inertia_local_kg_m2", "scale"):
        require(native[key] == asset[key], f"native mass freeze differs: {key}")
    return identity


def media(
    directory, data, rows, input_hash, trace_hash, evidence, *, final=False, physics_passed=True
):
    report = evidence.read(directory / "render_report.json")
    count, fps = (180, 30) if final else (len(rows), 50)
    filename = "orbit.mp4" if final else video.replay_timing(data["settings"]["dt"])["filename"]
    expected = dict(
        status="passed",
        physics_steps=0,
        source_input_sha256=input_hash,
        source_trace_sha256=trace_hash,
        source_input_canonical_sha256=clip.digest(data),
        source_sample_count=len(rows),
        physics_passed=physics_passed,
        camera_motion_only=final,
        purpose="final_orbit" if final else "physics_replay",
        frame_count=count,
        saved_frame_count=count,
        fps=fps,
        file=filename,
        frames_directory="frames",
        resolution=list(video.RES),
    )
    require(
        all(report.get(k) == v for k, v in expected.items()), "media source/count contract mismatch"
    )
    require(
        np.isfinite(report["maximum_visual_error_m"])
        and 0 <= report["maximum_visual_error_m"] <= 1e-5,
        "media visual transform error",
    )
    require(
        report["physics_duration_s"] == rows[-1]["time_s"] - rows[0]["time_s"],
        "media source duration differs",
    )
    if not final:
        timing = video.replay_timing(data["settings"]["dt"])
        require(
            all(report[k] == timing[k] for k in ("playback_speed", "slowdown_factor")),
            "media playback timing mismatch",
        )
    ledger_path = evidence.add(directory / "frames.jsonl")
    ledger = [json.loads(line) for line in ledger_path.open()]
    require(len(ledger) == count, "incomplete frame ledger")
    require(
        {p.name for p in (directory / "frames").iterdir()}
        == {f"frame_{i:04d}.png" for i in range(count)},
        "saved PNG set differs",
    )
    unique = set()
    for index, entry in enumerate(ledger):
        row = rows[-1] if final else rows[index]
        require(
            entry["frame"] == index
            and entry["source_step"] == row["step"]
            and entry["source_time_s"] == row["time_s"],
            "nonsequential frame ledger",
        )
        require(entry["png_path"] == f"frames/frame_{index:04d}.png", "aliased frame path")
        path = evidence.add(directory / entry["png_path"])
        require(lib.sha256(path) == entry["png_sha256"], "PNG hash mismatch")
        with Image.open(path) as picture:
            require(picture.size == video.RES and picture.mode == "RGB", "invalid PNG dimensions")
            digest = hashlib.sha256(np.asarray(picture).tobytes()).hexdigest()
        require(digest == entry["raw_rgb_sha256"], "PNG pixels differ from frame ledger")
        unique.add(digest)
    require(len(unique) == report["unique_raw_frames"], "raw frame uniqueness mismatch")
    actual = video.verify_video(evidence.add(directory / filename), count, fps)
    require(all(report.get(k) == v for k, v in actual.items()), "decoded video differs from report")
    if final:
        pictures = report["final_images"]
        require(
            {r["path"] for r in pictures} == {"overview.png", "top.png", "side.png"},
            "missing final views",
        )
        evidence.records(directory, pictures)
        for item in pictures:
            with Image.open(directory / item["path"]) as picture:
                require(picture.size == video.RES and picture.mode == "RGB", "invalid final view")
    return {
        k: report[k]
        for k in (
            "file",
            "frame_count",
            "unique_raw_frames",
            "unique_decoded_frames",
            "duration_s",
            "sha256",
        )
    }


def audit_task(root, seed, profile, evidence, *, baseline=None):
    task = TaskOutput(root)
    manifest = task.verify()
    evidence.add(task.root / "manifest.json")
    evidence.records(task.root, manifest["files"])
    task.verify_scene_inputs()
    task.verify_physics_inputs()
    require(
        task.report["status"] == "physics_passed"
        and all(
            task.report["stages"][k] == "passed"
            for k in ("objects", "scene", "physics", "final_render")
        ),
        "task is not fully passed",
    )
    source = evidence.read(task.stage("scene") / "source_manifest.json")
    evidence.records(local_path(source["source_root"]), source["files"])
    execution = task.report["physics"] if baseline is not None else task.report["construction"]
    attempts = execution["attempts"]
    require(
        0 < len(attempts) <= 31 and attempts[-1]["passed"],
        "missing passing last attempt or exceeded repair budget",
    )
    require(len({r["directory"] for r in attempts}) == len(attempts), "duplicate attempt")
    for item in attempts:
        directory = inside(task.stage("physics"), Path(item["directory"]))
        evidence.records(directory, evidence.read(directory / "manifest.json")["files"])
    attempt = inside(task.stage("physics"), Path(attempts[-1]["directory"]))
    data = evidence.read(attempt / "physics_input.json")
    require(set(data["assets"]) == set(OBJECTS), "complete four-object set is required")
    require(
        type(data["random_seed"]) is int and data["random_seed"] == seed, "wrong or duplicate seed"
    )
    require(
        data["profile"] == "text_repair_v1"
        and data["repair_preset"] == "text_scene_v2"
        and data["numerics_profile"] == profile,
        "acceptance or numerical profile differs",
    )
    physics.validate_settings(data)
    require(
        data["settings"]["stable_fraction"] == 0.95
        and data["settings"]["support_fraction"] == 0.95,
        "95 percent rules required",
    )
    identities = {}
    for name, category in OBJECTS.items():
        asset = data["assets"][name]
        require(
            asset["category"] == category and asset["fixed"] is (category == "table"),
            "wrong object category or fixed/dynamic role",
        )
        require(
            asset["support"] == ("ground" if category == "table" else "table_1"),
            "declared four-object table support differs",
        )
        identities[name] = asset_identity(asset, evidence)
    # Earlier failures belong to the evidence too: a final pass must not hide a
    # different asset set, an execution gap, or a resealed, unsupported old verdict.
    for item in attempts[:-1]:
        previous = inside(task.stage("physics"), Path(item["directory"]))
        old_data = evidence.read(previous / "physics_input.json")
        require(
            old_data["profile"] == data["profile"]
            and old_data["repair_preset"] == data["repair_preset"]
            and old_data["numerics_profile"] == profile
            and old_data["random_seed"] == seed,
            "earlier attempt configuration changed",
        )
        require(set(old_data["assets"]) == set(OBJECTS), "earlier attempt object set differs")
        require(
            {n: asset_identity(a, evidence) for n, a in old_data["assets"].items()} == identities,
            "earlier attempt asset content changed",
        )
        old_saved = evidence.read(previous / "validation_result.json")
        old_trace = evidence.add(previous / "trace.jsonl")
        old_rows = [json.loads(line) for line in old_trace.open()]
        require(
            old_rows and all("collision_capacity" in r for r in old_rows),
            "earlier attempt capacity telemetry is missing",
        )
        old_result = physics.evaluate(
            old_data, old_rows, initial_rejected=not old_saved["complete"]
        )
        require(
            old_saved
            == dict(
                old_result,
                input_sha256=lib.sha256(previous / "physics_input.json"),
                trace_sha256=lib.sha256(old_trace),
            ),
            "earlier attempt verdict is not supported by its trajectory",
        )
        require(
            item["passed"] == old_result["passed"]
            and item["result_sha256"] == lib.sha256(previous / "validation_result.json"),
            "earlier attempt summary binding differs",
        )
        old_load = evidence.read(previous / "asset_physics_report.json")
        require(
            old_load["status"] == "passed"
            and old_load["steps_executed"] == old_result["steps_executed"]
            and old_load["simulation_executed"] == old_result["simulation_executed"]
            and set(old_load["bodies"]) == set(OBJECTS),
            "earlier attempt execution evidence differs",
        )
        if len(old_rows) > 1:
            media(
                previous / "video",
                old_data,
                old_rows,
                lib.sha256(previous / "physics_input.json"),
                lib.sha256(old_trace),
                evidence,
                physics_passed=old_result["passed"],
            )
        else:
            evidence.read(previous / "video_not_generated.json")
    if baseline is None:
        half.baseline_evidence(task)
    else:
        require(data == half.half_input(baseline["data"]), "half release changed or resampled")
        require(
            execution["placement_calls"] == 0 and execution["release_unchanged"] is True,
            "half replay entered placement repair",
        )
        provenance = evidence.read(task.stage("physics") / "source_baseline.json")
        require(provenance == execution["source_baseline"], "half provenance report differs")
        require(
            Path(provenance["source_task"]).resolve() == baseline["root"]
            and provenance["source_manifest_sha256"] == baseline["manifest_sha256"],
            "half replay belongs to another baseline",
        )
        for name, filename in (
            ("input", "physics_input.json"),
            ("trace", "trace.jsonl"),
            ("result", "validation_result.json"),
            ("attempt_manifest", "manifest.json"),
        ):
            item = provenance["files"][name]
            original = baseline["attempt"] / filename
            require(
                Path(item["path"]).resolve() == original and item["sha256"] == lib.sha256(original),
                "archived baseline binding differs",
            )
            copied = evidence.add(
                inside(task.stage("physics"), task.stage("physics") / item["archived_path"])
            )
            require(lib.sha256(copied) == item["sha256"], "archived baseline bytes differ")
        archived = evidence.add(
            inside(task.stage("physics"), task.stage("physics") / provenance["archived_manifest"])
        )
        require(
            lib.sha256(archived) == baseline["manifest_sha256"], "baseline manifest archive differs"
        )
    trace = evidence.add(attempt / "trace.jsonl")
    rows = [json.loads(line) for line in trace.open()]
    require(
        rows and all("collision_capacity" in r for r in rows),
        "capacity telemetry is required on every row",
    )
    result = physics.evaluate(data, rows)
    require(result["passed"] and result["complete"], "independent physics evaluation failed")
    saved = evidence.read(attempt / "validation_result.json")
    input_hash, trace_hash = lib.sha256(attempt / "physics_input.json"), lib.sha256(trace)
    require(
        saved == dict(result, input_sha256=input_hash, trace_sha256=trace_hash),
        "saved verdict differs from independent physics",
    )
    require(
        attempts[-1]["result_sha256"] == lib.sha256(attempt / "validation_result.json"),
        "attempt result hash mismatch",
    )
    load = evidence.read(attempt / "asset_physics_report.json")
    require(
        load["status"] == "passed"
        and load["simulation_executed"] is True
        and load["steps_executed"] == data["settings"]["steps"]
        and set(load["bodies"]) == set(OBJECTS),
        "actual simulation/load report differs",
    )
    capacity = result["diagnostics"]["collision_capacity"]
    loaded_capacity = load["contact_capacity"]
    require(
        loaded_capacity["overflow"] is False
        and loaded_capacity["telemetry_schema"] == numerics.CAPACITY_SCHEMA
        and loaded_capacity["requested"] == capacity["requested"]
        and loaded_capacity["effective"] == capacity["effective"],
        "loaded capacity disagrees with trajectory",
    )
    for key, value in capacity["observed_peaks"].items():
        require(
            loaded_capacity["maximum_observed_" + key] == value,
            "loaded capacity peak differs from trajectory",
        )
    for name, asset in data["assets"].items():
        body = load["bodies"][name]
        require(
            body["fixed"] == asset["fixed"]
            and body["dofs"] == (0 if asset["fixed"] else 6)
            and body["collision_geoms"] > 0
            and body["collision_bounds_error_m"] <= 1e-5,
            "loaded body role or collision differs",
        )
        if not asset["fixed"]:
            mass = body["mass_properties_consistency"]
            require(
                mass["passed"]
                and np.isclose(body["mass_kg"], asset["mass_kg"], rtol=1e-5, atol=1e-9)
                and np.allclose(mass["canonical_com_m"], asset["com_local_m"], rtol=0, atol=1e-5)
                and np.allclose(
                    mass["canonical_inertia_kg_m2"],
                    asset["inertia_local_kg_m2"],
                    rtol=1e-4,
                    atol=1e-10,
                ),
                "loaded native mass properties differ",
            )
    require(evidence.read(attempt / "initial_state.json") == rows[0], "saved initial state differs")
    terminal = evidence.read(attempt / "final_state.json")
    require(
        terminal["complete"] is True and terminal["state"] == rows[-1],
        "saved terminal state differs",
    )
    validated = evidence.read(task.stage("physics") / "validated_scene.json")
    for key, expected in dict(
        assets=data["assets"],
        transforms=rows[-1]["objects"],
        physics_properties=data["settings"],
        validation_metrics=saved,
        random_seed=seed,
        source_input_sha256=input_hash,
        source_trace_sha256=trace_hash,
    ).items():
        require(validated[key] == expected, f"validated scene differs: {key}")
    replay = media(attempt / "video", data, rows, input_hash, trace_hash, evidence)
    orbit = media(
        task.stage("final_render") / "render",
        data,
        rows,
        input_hash,
        trace_hash,
        evidence,
        final=True,
    )
    task.verify()
    return dict(
        root=task.root,
        attempt=attempt,
        data=data,
        identities=identities,
        manifest_sha256=lib.sha256(task.root / "manifest.json"),
        metrics=result["objects"],
        capacity=result["diagnostics"]["collision_capacity"],
        replay=replay,
        orbit=orbit,
    )


def run(base_root, half_root, selected_comparison, output_dir):
    roots = [no_symlinks(p) for p in (base_root, half_root, selected_comparison, output_dir)]
    base_root, half_root, selected_comparison, out = roots
    for left in roots[:3]:
        require(
            not out.is_relative_to(left) and not left.is_relative_to(out),
            "audit output must be separate from inputs",
        )
    require(base_root != half_root, "base and half roots must differ")
    out.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    report = dict(
        schema_version=SCHEMA,
        scope=SCOPE,
        sixpass=False,
        status="failed",
        exit_code=1,
        tasks={},
        failures=[],
        simulation_steps_executed=0,
    )
    code = {str(p): lib.sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    try:
        profile = selected_profile(selected_comparison, evidence)
        report.update(selected_profile=profile, half_dt_profile=numerics.half_dt_profile(profile))
        reference = None
        for seed in SEEDS:
            base = None
            for mode, root, numerical in (
                ("base", base_root, profile),
                ("half", half_root, report["half_dt_profile"]),
            ):
                key = f"{mode}/seed_{seed}"
                try:
                    require(mode == "base" or base is not None, "baseline audit failed")
                    item = audit_task(
                        root / f"seed_{seed}",
                        seed,
                        numerical,
                        evidence,
                        baseline=base if mode == "half" else None,
                    )
                    if reference is None:
                        reference = item["identities"]
                    require(
                        item["identities"] == reference,
                        "assets or collision/native mass differ across runs",
                    )
                    if mode == "base":
                        base = item
                    report["tasks"][key] = dict(
                        passed=True,
                        task=str(item["root"]),
                        manifest_sha256=item["manifest_sha256"],
                        objects=item["metrics"],
                        capacity=item["capacity"],
                        replay=item["replay"],
                        orbit=item["orbit"],
                    )
                except Exception as exc:
                    report["tasks"][key] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
                    report["failures"].append(dict(task=key, error=str(exc)))
        evidence.check()
        require(all(lib.sha256(p) == h for p, h in code.items()), "audit implementation changed")
        report["sixpass"] = len(report["tasks"]) == 6 and all(
            r["passed"] for r in report["tasks"].values()
        )
        report.update(
            status="passed" if report["sixpass"] else "failed",
            exit_code=0 if report["sixpass"] else 1,
        )
    except Exception as exc:
        report["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
    report.update(code_sha256=code, evidence_sha256=evidence.files)
    official.write_json(out / "acceptance_result.json", report)
    official.write_json(
        out / "manifest.json",
        dict(
            schema_version=SCHEMA,
            scope=SCOPE,
            files=[official.fingerprint(out / "acceptance_result.json", out)],
        ),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-root", "half-root", "selected-comparison", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.base_root, args.half_root, args.selected_comparison, args.output_dir)
        print(json.dumps(report, ensure_ascii=False))
        return report["exit_code"]
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

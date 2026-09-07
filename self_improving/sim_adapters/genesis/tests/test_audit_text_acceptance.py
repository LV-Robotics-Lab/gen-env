"""Offline attacks against complete-scene, same-release and saved-media acceptance."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_text_repair import case

from self_improving.sim_adapters.genesis import audit_text_acceptance as audit
from self_improving.sim_adapters.genesis import replay_text_half_dt as replay

PROFILE = "dt2ms_tau20ms_authored_v1"


def write(path, data):
    audit.official.write_json(path, data)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def physical_case(root):
    original, old_rows = case()
    mapping = {"table": "table_1", "a": "apple_1", "b": "cup_1"}
    assets = {mapping[n]: copy.deepcopy(a) for n, a in original["assets"].items()}
    poses = {mapping[n]: p for n, p in original["poses"].items()}
    assets["bowl_1"] = copy.deepcopy(assets["cup_1"])
    poses["bowl_1"] = copy.deepcopy(poses["cup_1"])
    poses["bowl_1"]["position"][0] = -0.3
    for name, asset in assets.items():
        folder = root / "assets" / name
        folder.mkdir(parents=True)
        model = folder / "model.xml"
        model.write_text("offline physical asset fixture")
        asset.update(
            category=audit.OBJECTS[name],
            asset_id=name,
            anchor_m=[0, 0, 0],
            support="ground" if name == "table_1" else "table_1",
            scale=1.0,
            mass_kg=1.0,
            com_local_m=[0, 0, 0],
            inertia_local_kg_m2=np.eye(3).tolist(),
            mass_source="native_loaded_fixture",
            native_pose=dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0]),
            bottom_offset_m=0,
            visual_bottom_offset_m=0,
            collision=dict(method="native_collision", parts=1, quality=dict(passed=True)),
            model_entrypoint=str(model),
            physics_file=str(model),
            geometry_file=str(folder / "geometry.npz"),
        )
        np.savez(folder / "geometry.npz", vertices=asset["hull"])
        write(
            folder / "native_mass_properties.json",
            {k: asset[k] for k in ("mass_kg", "com_local_m", "inertia_local_kg_m2", "scale")},
        )
        files = [audit.official.fingerprint(p, folder) for p in sorted(folder.iterdir())]
        asset.update(
            source_root=str(folder),
            source_files=files,
            derived_root=str(folder),
            derived_files=files,
        )
    templates = []
    for row in (old_rows[0], old_rows[1]):
        row = copy.deepcopy(row)
        row["objects"] = {mapping.get(n, n): v for n, v in row["objects"].items()}
        row["objects"]["bowl_1"] = copy.deepcopy(row["objects"]["cup_1"])
        row["objects"]["bowl_1"]["position"][0] = -0.3
        row["objects"]["bowl_1"]["com_position"][0] = -0.3
        for c in row["contacts"]:
            c.update(a=mapping.get(c["a"], c["a"]), b=mapping.get(c["b"], c["b"]))
        extra = copy.deepcopy(row["contacts"][-1])
        extra.update(a="bowl_1", geom_a=3, link_a=3)
        row["contacts"].append(extra)
        row["collision_capacity"] = dict(
            schema_version=audit.numerics.CAPACITY_SCHEMA,
            limits=dict(
                collision_pairs=6,
                possible_geom_pairs=6,
                broadphase_pairs=6,
                broadphase_multiplier=1,
                candidate_contacts=40,
                postpruning_contacts=40,
            ),
            usage=dict(broadphase_pairs=3, postpruning_contacts=3, contacting_geom_pairs=3),
        )
        templates.append(row)
    return assets, poses, templates


def simulate(data, out, check, templates):
    rows = [
        copy.deepcopy(templates[0 if i == 0 else 1]) for i in range(data["settings"]["steps"] + 1)
    ]
    for i, row in enumerate(rows):
        row.update(step=i, time_s=i * data["settings"]["dt"])
    write_rows(out / "trace.jsonl", rows)
    bodies = {
        n: dict(
            fixed=a["fixed"],
            dofs=0 if a["fixed"] else 6,
            mass_kg=a["mass_kg"],
            collision_geoms=1,
            collision_bounds_error_m=0,
            mass_properties_consistency=dict(
                passed=True,
                canonical_com_m=a["com_local_m"],
                canonical_inertia_kg_m2=a["inertia_local_kg_m2"],
            ),
        )
        for n, a in data["assets"].items()
    }
    write(
        out / "asset_physics_report.json",
        dict(
            status="passed",
            simulation_executed=True,
            steps_executed=data["settings"]["steps"],
            bodies=bodies,
            contact_capacity=dict(
                overflow=False,
                telemetry_schema=audit.numerics.CAPACITY_SCHEMA,
                requested=dict(max_collision_pairs=1024, max_contacts=4096),
                effective=rows[0]["collision_capacity"]["limits"],
                maximum_observed_broadphase_pairs=3,
                maximum_observed_postpruning_contacts=3,
                maximum_observed_contacting_geom_pairs=3,
            ),
        ),
    )
    write(out / "initial_state.json", rows[0])
    write(out / "final_state.json", dict(complete=True, state=rows[-1]))
    check()
    return audit.physics.evaluate(data, rows)


def seal_attempt(task, attempt):
    write(
        attempt / "manifest.json",
        dict(
            files=[
                audit.official.fingerprint(p, attempt)
                for p in sorted(attempt.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            ]
        ),
    )
    task.seal()


@pytest.fixture(scope="module")
def scene_tasks(tmp_path_factory):
    root = tmp_path_factory.mktemp("six-scene-audit")
    assets, poses, templates = physical_case(root)
    source = audit.TaskOutput(root / "source")
    source.start("桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。")
    write(source.stage("objects") / "asset_request.json", dict(objects=list(audit.OBJECTS)))
    source.finish_assets(dict(status="assets_selected"))
    for seed in audit.SEEDS:
        task = audit.TaskOutput(root / "base" / f"seed_{seed}")
        task.start(source.owner()["request"])
        write(task.stage("objects") / "asset_request.json", dict(objects=list(audit.OBJECTS)))
        task.finish_assets(dict(status="assets_selected"))
        task.start_scene()
        write(
            task.stage("scene") / "source_manifest.json",
            dict(source_root=str(source.root), files=source.verify()["files"]),
        )
        write(
            task.stage("scene") / "scene_v0.json",
            dict(graph=dict(relations=[], order=list(assets))),
        )
        task.finish_scene(dict(status="scene_built"))
        task.start_physics()
        attempt = task.stage("physics") / "attempts" / "000"
        attempt.mkdir(parents=True)
        data = audit.physics.frozen_input(
            assets, poses, [], seed, numerics_profile=PROFILE, repair_preset="text_scene_v2"
        )
        write(attempt / "physics_input.json", data)
        result = simulate(data, attempt, lambda: None, templates)
        assert result["passed"]
        result.update(
            input_sha256=audit.lib.sha256(attempt / "physics_input.json"),
            trace_sha256=audit.lib.sha256(attempt / "trace.jsonl"),
        )
        write(attempt / "validation_result.json", result)
        terminal = audit.lib.read_json(attempt / "final_state.json")
        write(
            task.stage("physics") / "validated_scene.json",
            dict(
                assets=data["assets"],
                transforms=terminal["state"]["objects"],
                physics_properties=data["settings"],
                validation_metrics=result,
                random_seed=seed,
                source_input_sha256=result["input_sha256"],
                source_trace_sha256=result["trace_sha256"],
            ),
        )
        task.report["construction"] = dict(
            repair_preset="text_scene_v2",
            attempts=[
                dict(
                    directory=str(attempt),
                    passed=True,
                    result_sha256=audit.lib.sha256(attempt / "validation_result.json"),
                )
            ],
        )
        seal_attempt(task, attempt)
        task.finish_physics(
            dict(status="physics_passed", physics_status="passed", render_status="passed")
        )

        def recorder(data, trace, out, check, **kwargs):
            out.mkdir(parents=True)
            check()

        replay.run(
            task.root,
            root / "half" / f"seed_{seed}",
            render=True,
            simulator=lambda d, o, c: simulate(d, o, c, templates),
            recorder=recorder,
        )
    return root


@pytest.fixture
def no_media(monkeypatch):
    # Pixel/MP4 evidence has separate real-file attacks below; never mock physical evaluation.
    monkeypatch.setattr(audit, "media", lambda *a, **k: dict(offline_media_fixture=True))


def test_six_complete_real_evaluations_and_unchanged_half_releases(
    scene_tasks, tmp_path, monkeypatch, no_media
):
    monkeypatch.setattr(audit, "selected_profile", lambda *a: PROFILE)
    result = audit.run(
        scene_tasks / "base", scene_tasks / "half", tmp_path / "comparison", tmp_path / "audit"
    )
    assert result["sixpass"] and result["exit_code"] == 0
    assert result["scope"] == "complete_four_asset_acceptance"
    assert len(result["tasks"]) == 6
    assert all(set(r["objects"]) == set(audit.OBJECTS) for r in result["tasks"].values())
    assert result["code_sha256"] and result["evidence_sha256"]


@pytest.mark.parametrize(
    "attack",
    ["missing", "duplicate_seed", "small_scene", "fixed_cup", "gt75", "capacity", "half_pose"],
)
def test_resealed_task_attacks_are_rejected(scene_tasks, no_media, attack):
    mode = "half" if attack == "half_pose" else "base"
    task = audit.TaskOutput(scene_tasks / mode / "seed_0")
    task.verify()
    attempt = task.stage("physics") / "attempts" / "000"
    saved = {p: p.read_bytes() for p in task.root.rglob("*") if p.is_file()}
    try:
        path = attempt / "physics_input.json"
        data = audit.lib.read_json(path)
        if attack == "missing":
            (attempt / "trace.jsonl").unlink()
        elif attack == "duplicate_seed":
            data["random_seed"] = 42
        elif attack == "small_scene":
            del data["assets"]["bowl_1"]
            del data["poses"]["bowl_1"]
        elif attack == "fixed_cup":
            data["assets"]["cup_1"]["fixed"] = True
        elif attack == "gt75":
            data.update(
                profile="text_repair_gt75_v1",
                settings=audit.physics.settings("text_repair_gt75_v1", PROFILE),
            )
        elif attack == "half_pose":
            data["poses"]["cup_1"]["position"][0] += 0.005
        elif attack == "capacity":
            rows = replay.read_rows(attempt / "trace.jsonl")
            for row in rows:
                row.pop("collision_capacity")
            write_rows(attempt / "trace.jsonl", rows)
        write(path, data)
        if attack == "capacity":
            # Reseal a fully self-consistent legacy-style verdict: mandatory telemetry
            # must still reject it, even though the generic evaluator permits old rows.
            result = audit.physics.evaluate(data, rows)
            result.update(
                input_sha256=audit.lib.sha256(path),
                trace_sha256=audit.lib.sha256(attempt / "trace.jsonl"),
            )
            write(attempt / "validation_result.json", result)
            task.report["construction"]["attempts"][-1]["result_sha256"] = audit.lib.sha256(
                attempt / "validation_result.json"
            )
        seal_attempt(task, attempt)
        evidence = audit.Evidence()
        baseline = (
            audit.audit_task(scene_tasks / "base" / "seed_0", 0, PROFILE, evidence)
            if mode == "half"
            else None
        )
        with pytest.raises((ValueError, FileNotFoundError)):
            audit.audit_task(
                task.root,
                0,
                audit.numerics.half_dt_profile(PROFILE) if mode == "half" else PROFILE,
                evidence,
                baseline=baseline,
            )
    finally:
        for path in list(task.root.rglob("*")):
            if path.is_file() and path not in saved:
                path.unlink()
        for path, value in saved.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)


def test_different_collision_or_mass_across_passing_runs_fails(
    scene_tasks, tmp_path, monkeypatch, no_media
):
    monkeypatch.setattr(audit, "selected_profile", lambda *a: PROFILE)
    original = audit.audit_task

    def changed(root, seed, profile, evidence, **kwargs):
        result = original(root, seed, profile, evidence, **kwargs)
        if seed == 42:
            result["identities"]["cup_1"]["mass_kg"] += 0.01
        return result

    monkeypatch.setattr(audit, "audit_task", changed)
    result = audit.run(
        scene_tasks / "base", scene_tasks / "half", tmp_path / "comparison", tmp_path / "audit"
    )
    assert not result["sixpass"] and result["exit_code"] == 1
    assert not result["tasks"]["base/seed_42"]["passed"]


def media_fixture(tmp_path, monkeypatch, final=False):
    data, rows = case()
    rows = rows[:3]
    out = tmp_path / "media"
    (out / "frames").mkdir(parents=True)
    image = np.zeros((audit.video.RES[1], audit.video.RES[0], 3), dtype=np.uint8)
    raw_hash = hashlib.sha256(image.tobytes()).hexdigest()
    count, fps = (180, 30) if final else (3, 50)
    ledger = []
    for i in range(count):
        filename = f"frames/frame_{i:04d}.png"
        Image.fromarray(image).save(out / filename)
        row = rows[-1] if final else rows[i]
        ledger.append(
            dict(
                frame=i,
                source_step=row["step"],
                source_time_s=row["time_s"],
                png_path=filename,
                png_sha256=audit.lib.sha256(out / filename),
                raw_rgb_sha256=raw_hash,
            )
        )
    write_rows(out / "frames.jsonl", ledger)
    filename = "orbit.mp4" if final else "physics_replay_10x_slow.mp4"
    (out / filename).write_bytes(b"offline codec fixture")
    decoded = dict(
        file=filename,
        frame_count=count,
        unique_decoded_frames=1,
        fps=fps,
        duration_s=count / fps,
        resolution=list(audit.video.RES),
        sha256=audit.lib.sha256(out / filename),
    )
    monkeypatch.setattr(audit.video, "verify_video", lambda *a: decoded)
    report = dict(
        decoded,
        status="passed",
        physics_steps=0,
        source_input_sha256="input",
        source_trace_sha256="trace",
        source_input_canonical_sha256=audit.clip.digest(data),
        source_sample_count=len(rows),
        physics_passed=True,
        camera_motion_only=final,
        purpose="final_orbit" if final else "physics_replay",
        saved_frame_count=count,
        frames_directory="frames",
        maximum_visual_error_m=0,
        physics_duration_s=rows[-1]["time_s"],
        playback_speed=0.1,
        slowdown_factor=10,
        unique_raw_frames=1,
    )
    if final:
        report["final_images"] = []
        for name in ("overview", "top", "side"):
            Image.fromarray(image).save(out / f"{name}.png")
            report["final_images"].append(audit.official.fingerprint(out / f"{name}.png", out))
    write(out / "render_report.json", report)
    return out, data, rows, ledger, report


@pytest.mark.parametrize(
    "attack", [None, "png", "pixels", "ledger", "source", "duration", "video", "missing_final"]
)
def test_media_content_and_final_views_are_mandatory(tmp_path, monkeypatch, attack):
    final = attack == "missing_final"
    out, data, rows, ledger, report = media_fixture(tmp_path, monkeypatch, final=final)
    if attack == "png":
        (out / ledger[0]["png_path"]).write_bytes(b"tampered PNG")
    elif attack == "pixels":
        picture = np.ones((720, 960, 3), dtype=np.uint8)
        Image.fromarray(picture).save(out / ledger[0]["png_path"])
        ledger[0]["png_sha256"] = audit.lib.sha256(out / ledger[0]["png_path"])
    elif attack == "ledger":
        ledger[1]["source_step"] = 0
    elif attack == "source":
        report["source_trace_sha256"] = "wrong trace"
    elif attack == "duration":
        report["duration_s"] = 30.02
    elif attack == "video":
        report["sha256"] = "wrong encoded video hash"
    elif attack == "missing_final":
        report["final_images"].pop()
    write_rows(out / "frames.jsonl", ledger)
    write(out / "render_report.json", report)
    if attack is None:
        assert audit.media(out, data, rows, "input", "trace", audit.Evidence())["frame_count"] == 3
    else:
        with pytest.raises(ValueError):
            audit.media(out, data, rows, "input", "trace", audit.Evidence(), final=final)


def test_selected_comparison_cannot_be_just_a_passed_label(tmp_path):
    write(
        tmp_path / "comparison_input.json",
        dict(seeds=list(audit.SEEDS), acceptance_profile="text_repair_v1"),
    )
    write(
        tmp_path / "comparison_result.json",
        dict(
            seeds=list(audit.SEEDS),
            acceptance_profile="text_repair_v1",
            status="passed",
            selected_profile=PROFILE,
            half_dt_profile=audit.numerics.half_dt_profile(PROFILE),
            matrix={PROFILE: {}},
            sensitivity={PROFILE: {}},
        ),
    )
    with pytest.raises(ValueError, match="three seeds"):
        audit.selected_profile(tmp_path, audit.Evidence())


def test_missing_seed_directory_never_counts_as_six_passes(
    scene_tasks, tmp_path, monkeypatch, no_media
):
    monkeypatch.setattr(audit, "selected_profile", lambda *a: PROFILE)
    original = scene_tasks / "base" / "seed_87"
    hidden = scene_tasks / "temporarily_missing_seed_87"
    original.rename(hidden)
    try:
        result = audit.run(
            scene_tasks / "base", scene_tasks / "half", tmp_path / "comparison", tmp_path / "audit"
        )
        assert not result["sixpass"] and len(result["tasks"]) == 6
        assert not result["tasks"]["base/seed_87"]["passed"]
        assert not result["tasks"]["half/seed_87"]["passed"]
    finally:
        hidden.rename(original)


def test_same_physical_content_allows_different_asset_roots(scene_tasks, tmp_path):
    path = scene_tasks / "base/seed_0/03_physics/attempts/000/physics_input.json"
    original = audit.lib.read_json(path)["assets"]["cup_1"]
    moved = copy.deepcopy(original)
    source = Path(original["source_root"])
    destination = tmp_path / "relocated_asset"
    shutil.copytree(source, destination)
    for key in ("source_root", "derived_root"):
        moved[key] = str(destination)
    for key in ("model_entrypoint", "physics_file", "geometry_file"):
        moved[key] = str(destination / Path(original[key]).relative_to(source))
    assert audit.asset_identity(original, audit.Evidence()) == audit.asset_identity(
        moved, audit.Evidence()
    )
    moved["mass_kg"] += 0.001
    with pytest.raises(ValueError, match="native mass freeze"):
        audit.asset_identity(moved, audit.Evidence())


def test_last_pass_cannot_hide_resealed_earlier_trace(scene_tasks, no_media):
    task = audit.TaskOutput(scene_tasks / "base" / "seed_0")
    task.verify()
    attempt = task.stage("physics") / "attempts/000"
    last = task.stage("physics") / "attempts/001"
    saved = {p: p.read_bytes() for p in task.root.rglob("*") if p.is_file()}
    try:
        shutil.copytree(attempt, last)
        record = copy.deepcopy(task.report["construction"]["attempts"][-1])
        record["directory"] = str(last)
        task.report["construction"]["attempts"].append(record)
        rows = replay.read_rows(attempt / "trace.jsonl")
        for row in rows[-100:]:
            row["objects"]["cup_1"]["velocity"] = [0.02, 0, 0]
        write_rows(attempt / "trace.jsonl", rows)
        seal_attempt(task, attempt)
        with pytest.raises(ValueError, match="earlier attempt verdict"):
            audit.audit_task(task.root, 0, PROFILE, audit.Evidence())
    finally:
        shutil.rmtree(last)
        for path, value in saved.items():
            path.write_bytes(value)

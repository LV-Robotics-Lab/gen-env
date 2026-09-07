"""Validate compiled Genesis scenes, then render a separately compiled settled package.

This adapter deliberately lives outside OpenXSim. No rendering occurs during physics.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OPENXSIM = ROOT / "self_improving/asset_pipeline/active/shared/openxsim/source/agenticsim"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPENXSIM))
from agenticsim.openxsim import genesis_runtime as render
from agenticsim.openxsim.backends import CompileResult, GenesisCompiler
from agenticsim.openxsim.importers import import_compile_manifest
from agenticsim.openxsim.ir import Pose

from self_improving.sim_adapters.genesis.physics_math import angle, corners, rotation
from self_improving.sim_adapters.genesis.task_output import TaskOutput

GENESIS_COMMIT = "0e74bf392781884ccad765c3f344419c86b872ca"
SCHEMA = "genenv.genesis_physics_evidence.v1"
DEFAULTS = dict(
    dt=0.004,
    steps=1000,
    substeps=1,
    seed=0,
    window_s=0.5,
    translation_m=0.001,
    rotation_deg=0.5,
    speed_mps=0.01,
    angular_speed_radps=0.05,
    support_fraction=0.8,
    penetration_m=0.001,
    constraint_timeconst=0.001,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def config_for(package, scene):
    raw = package.metadata.get("genesis_physics")
    if not isinstance(raw, dict) or set(raw) != {"settings", "bodies"}:
        raise ValueError("package requires genesis_physics settings and bodies")
    if set(raw["settings"]) != set(DEFAULTS):
        raise ValueError("physics settings must explicitly declare every parameter")
    cfg = raw["settings"]
    for name, value in cfg.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid setting: {name}")
        if value < 0 or (name != "seed" and value == 0):
            raise ValueError(f"invalid setting: {name}")
    if any(type(cfg[k]) is not int for k in ("steps", "seed", "substeps")):
        raise ValueError("steps, seed, substeps must be integers")
    if cfg["substeps"] != 1 or cfg["window_s"] >= cfg["dt"] * cfg["steps"]:
        raise ValueError("v1 requires substeps=1 and a full terminal window")
    if cfg["support_fraction"] > 1:
        raise ValueError("invalid support fraction")
    ids = {o.instance_id for o in package.env.objects}
    if set(raw["bodies"]) != ids or "table" in ids:
        raise ValueError("body settings must match every object; table is reserved")
    if package.env.robots or package.env.up_axis != "Z":
        raise ValueError("v1 supports Z-up object-only scenes")
    if tuple(package.env.gravity_mps2) != (0.0, 0.0, -9.81):
        raise ValueError("v1 requires gravity (0, 0, -9.81)")
    by_id = {o.instance_id: o for o in package.env.objects}
    specs = {s["instance_id"]: s for s in scene["objects"]}
    for spec in scene["objects"]:
        name = spec["instance_id"]
        body = raw["bodies"][name]
        if spec["kind"] not in {"box", "urdf"} or spec.get("articulation"):
            raise ValueError("v1 supports only Box and single-link URDF")
        if body.get("collision") is not True:
            raise ValueError(f"{name}: collision must be enabled")
        for key in ("density", "friction"):
            if not math.isfinite(float(body[key])) or body[key] <= 0:
                raise ValueError(f"{name}: invalid {key}")
        corners(body["local_bounds"], spec["pose"])
        if by_id[name].static != spec["source_static"]:
            raise ValueError("compiled static flag differs")
        if spec["kind"] == "urdf":
            import xml.etree.ElementTree as ET

            root = ET.parse(spec["uri"]).getroot()
            if len(root.findall("link")) != 1 or root.findall("joint"):
                raise ValueError("v1 URDF must contain exactly one unarticulated link")
            if len(root.findall("./link/collision")) != body["collision_parts"]:
                raise ValueError("collision part count differs from package")
    support = {}
    for condition in package.task.success:
        kind, obj = condition.get("type"), condition.get("object")
        if kind not in {"support", "inside", "upright", "released"} or obj not in ids:
            raise ValueError("unsupported/unbound task success condition")
        if kind in {"upright", "released"}:
            key = "max_tilt_deg" if kind == "upright" else "minimum_drop_m"
            value = condition[key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid condition threshold: {key}")
        if kind in {"support", "inside"}:
            target = condition.get("target")
            if target not in ids | {"table"} or target == obj or by_id[obj].static:
                raise ValueError("support/inside requires a dynamic source and valid target")
            if obj in support:
                raise ValueError("each dynamic object needs exactly one support relation")
            support[obj] = target
            if kind == "inside":
                if target == "table" or specs[obj]["kind"] != "box":
                    raise ValueError("inside v1 requires a Box inside an object")
                interior = raw["bodies"][target]["interior_bounds"]
                corners(interior, dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0]))
                if not raw["bodies"][target].get("geometry_measurement_sha256"):
                    raise ValueError("inside requires measured geometry provenance")
                if specs[target]["kind"] == "urdf":
                    measurement = Path(specs[target]["uri"]).parent / "measurement.json"
                    body = raw["bodies"][target]
                    if render.sha256_file(measurement) != body["geometry_measurement_sha256"]:
                        raise ValueError("interior measurement hash differs")
                    measured = json.loads(measurement.read_text())
                    if (
                        measured["interior_bounds"] != body["interior_bounds"]
                        or measured["local_bounds"] != body["local_bounds"]
                        or measured["converted_urdf"]["sha256"]
                        != render.sha256_file(specs[target]["uri"])
                    ):
                        raise ValueError("interior measurement does not bind this geometry")
    if set(support) != {o.instance_id for o in package.env.objects if not o.static}:
        raise ValueError("every dynamic object must have a support/inside condition")
    if not support:
        raise ValueError("at least one dynamic test object is required")
    return raw


def evaluate(package, scene, rows):
    """Pure evaluator; no Genesis dependency and no missing-data defaults."""
    raw = config_for(package, scene)
    cfg, bodies = raw["settings"], raw["bodies"]
    ids = set(bodies)
    if len(rows) != cfg["steps"] + 1:
        raise ValueError("missing trajectory records")
    for i, row in enumerate(rows):
        if (
            row["step"] != i
            or not math.isfinite(row["time_s"])
            or abs(row["time_s"] - i * cfg["dt"]) > 1e-9
        ):
            raise ValueError("non-sequential trajectory")
        if set(row["objects"]) != ids:
            raise ValueError("trajectory object mismatch")
        for state in row["objects"].values():
            for key in ("position", "velocity", "angular_velocity", "net_contact_force"):
                value = np.asarray(state[key])
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid state {key}")
            rotation(state["orientation_wxyz"])
        for c in row["contacts"]:
            if c["a"] not in ids | {"table"} or c["b"] not in ids | {"table"}:
                raise ValueError("unknown contact entity")
            for key in ("position", "normal", "force_a", "force_b"):
                value = np.asarray(c[key])
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid contact {key}")
            if not math.isfinite(c["penetration"]):
                raise ValueError("nonfinite penetration")
    window = rows[-(math.ceil(cfg["window_s"] / cfg["dt"]) + 1) :]
    final = rows[-1]["objects"]
    checks = []

    def check(name, ok, **metrics):
        checks.append(dict(name=name, passed=bool(ok), **metrics))

    penetration = max((c["penetration"] for r in rows for c in r["contacts"]), default=0.0)
    check(
        "penetration",
        penetration <= cfg["penetration_m"],
        maximum_m=penetration,
        limit_m=cfg["penetration_m"],
    )
    for obj in package.env.objects:
        name = obj.instance_id
        states = [r["objects"][name] for r in window]
        distance = max(
            np.linalg.norm(np.array(s["position"]) - final[name]["position"]) for s in states
        )
        degrees = max(angle(s["orientation_wxyz"], final[name]["orientation_wxyz"]) for s in states)
        speed = max(np.linalg.norm(s["velocity"]) for s in states)
        angular = max(np.linalg.norm(s["angular_velocity"]) for s in states)
        check(
            f"{name}.settled",
            distance <= cfg["translation_m"]
            and degrees <= cfg["rotation_deg"]
            and speed <= cfg["speed_mps"]
            and angular <= cfg["angular_speed_radps"],
            displacement_m=float(distance),
            rotation_deg=degrees,
            speed_mps=float(speed),
            angular_speed_radps=float(angular),
        )
    table = scene["table"]
    for condition in package.task.success:
        kind, name = condition["type"], condition["object"]
        if kind == "released":
            drop = rows[0]["objects"][name]["position"][2] - min(
                r["objects"][name]["position"][2] for r in rows
            )
            check(f"{name}.released", drop >= condition["minimum_drop_m"], drop_m=drop)
            continue
        if kind == "upright":
            tilt = math.degrees(
                math.acos(np.clip(rotation(final[name]["orientation_wxyz"])[2, 2], -1, 1))
            )
            check(f"{name}.upright", tilt <= condition["max_tilt_deg"], tilt_deg=tilt)
            continue
        target = condition["target"]
        hits, unexpected = 0, set()
        for row in window:
            force = 0.0
            for c in row["contacts"]:
                if name not in (c["a"], c["b"]):
                    continue
                other = c["b"] if c["a"] == name else c["a"]
                f = c["force_a"] if c["a"] == name else c["force_b"]
                if other == target:
                    force += f[2]
                elif other != name:
                    unexpected.add(other)
            hits += force > 1e-6
        fraction = hits / len(window)
        check(
            f"{name}.support",
            fraction >= cfg["support_fraction"],
            target=target,
            fraction=fraction,
            limit=cfg["support_fraction"],
        )
        check(f"{name}.unexpected_support", not unexpected, targets=sorted(unexpected))
        if kind == "inside":
            touched_table = any(
                {c["a"], c["b"]} == {name, "table"} for r in rows for c in r["contacts"]
            )
            initial = rows[0]["objects"]
            local = (
                corners(bodies[name]["local_bounds"], initial[name]) - initial[target]["position"]
            ) @ rotation(initial[target]["orientation_wxyz"])
            check(
                f"{name}.entered_from_above",
                local[:, 2].min() > bodies[target]["local_bounds"][1][2],
                initial_bottom_local_z=float(local[:, 2].min()),
            )
            check(f"{name}.never_touched_table", not touched_table)
        fits = True
        minimum_margin = float("inf")
        for row in window:
            points = corners(bodies[name]["local_bounds"], row["objects"][name])
            if target == "table":
                lo = np.array(table["center"]) - np.array(table["size"]) / 2
                hi = np.array(table["center"]) + np.array(table["size"]) / 2
                fits &= bool((points[:, :2] >= lo[:2]).all() and (points[:, :2] <= hi[:2]).all())
            else:
                target_pose = row["objects"][target]
                local = (points - target_pose["position"]) @ rotation(
                    target_pose["orientation_wxyz"]
                )
                bounds = bodies[target]["interior_bounds" if kind == "inside" else "support_bounds"]
                axes = 3 if kind == "inside" else 2
                minimum_margin = min(
                    minimum_margin,
                    float((local[:, :axes] - np.array(bounds[0])[:axes]).min()),
                    float((np.array(bounds[1])[:axes] - local[:, :axes]).min()),
                )
                fits &= bool(
                    (local[:, :axes] >= np.array(bounds[0])[:axes]).all()
                    and (local[:, :axes] <= np.array(bounds[1])[:axes]).all()
                )
        check(
            f"{name}.{kind}_geometry",
            fits,
            minimum_margin_m=minimum_margin if math.isfinite(minimum_margin) else None,
        )
    return checks


def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def simulate(package, scene_spec, out, raw):
    os.environ["GS_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "1"
    import genesis as gs

    cfg = raw["settings"]
    entities, owners = {}, {}
    rows = []
    actual = {}
    gs.init(backend=gs.cpu, precision="32", seed=cfg["seed"], logging_level="warning")
    try:
        if render._discover_genesis_commit(gs) != GENESIS_COMMIT:
            raise ValueError("Genesis checkout does not match the pinned commit")
        sim_options = gs.options.SimOptions(dt=cfg["dt"], substeps=1, gravity=(0, 0, -9.81))
        rigid_options = gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=50,
            tolerance=1e-8,
            ls_iterations=50,
            use_hibernation=False,
            constraint_timeconst=cfg["constraint_timeconst"],
        )
        scene = gs.Scene(
            sim_options=sim_options, rigid_options=rigid_options, show_viewer=False, show_FPS=False
        )
        table = scene_spec["table"]
        entities["table"] = scene.add_entity(
            gs.morphs.Box(
                pos=tuple(table["center"]), size=tuple(table["size"]), fixed=True, collision=True
            ),
            material=gs.materials.Rigid(rho=1000, friction=0.5),
            name="table",
        )
        for spec in scene_spec["objects"]:
            name = spec["instance_id"]
            body = raw["bodies"][name]
            common = dict(
                pos=tuple(spec["pose"]["position"]),
                quat=tuple(spec["pose"]["orientation_wxyz"]),
                fixed=spec["source_static"],
                collision=True,
            )
            if spec["kind"] == "box":
                morph = gs.morphs.Box(size=tuple(spec["size_m"]), **common)
            else:
                morph = gs.morphs.URDF(
                    file=spec["uri"],
                    scale=spec["uniform_scale"],
                    align=False,
                    merge_fixed_links=False,
                    convexify=True,
                    decimate=False,
                    file_meshes_are_zup=True,
                    **common,
                )
            entities[name] = scene.add_entity(
                morph,
                material=gs.materials.Rigid(rho=body["density"], friction=body["friction"]),
                name=name,
            )
            if "sol_params" in body:
                for geom in entities[name].geoms:
                    geom.set_sol_params(np.asarray(body["sol_params"], dtype=float))
        scene.build()
        for name, entity in entities.items():
            expected_fixed = (
                name == "table"
                or next(o for o in package.env.objects if o.instance_id == name).static
            )
            if bool(entity.base_link.is_fixed) != expected_fixed:
                raise ValueError(f"{name}: actual fixed state differs")
            if not expected_fixed and entity.n_dofs != 6:
                raise ValueError(f"{name}: dynamic object must have six free DOFs")
            if not entity.geoms:
                raise ValueError(f"{name}: collision geometry missing")
            if name != "table":
                obj_spec = next(s for s in scene_spec["objects"] if s["instance_id"] == name)
                loaded_pose = render._entity_pose(entity)
                if max(render.pose_error(obj_spec["pose"], loaded_pose).values()) > 1e-6:
                    raise ValueError(f"{name}: initial pose differs from compiled scene")
                verts = np.vstack(
                    (
                        as_array(entity.get_verts()).reshape(-1, 3),
                        as_array(entity.get_vverts()).reshape(-1, 3),
                    )
                )
                local = (verts - loaded_pose["position"]) @ rotation(
                    loaded_pose["orientation_wxyz"]
                )
                actual_bounds = np.array([local.min(axis=0), local.max(axis=0)])
                if not np.isfinite(actual_bounds).all() or not np.allclose(
                    actual_bounds, raw["bodies"][name]["local_bounds"], atol=1e-5, rtol=0
                ):
                    raise ValueError(f"{name}: actual visual/collision bounds differ from package")
                if (
                    obj_spec["kind"] == "urdf"
                    and len(entity.geoms) != raw["bodies"][name]["collision_parts"]
                ):
                    raise ValueError(f"{name}: collision decomposition changed")
            for geom in entity.geoms:
                owners[geom.idx] = name
            mass = float(as_array(entity.get_mass()).reshape(-1)[0])
            if not expected_fixed and (not math.isfinite(mass) or mass <= 0):
                raise ValueError(f"{name}: invalid loaded mass")
            actual[name] = dict(
                mass_kg=mass if math.isfinite(mass) else None,
                collision_geoms=len(entity.geoms),
                fixed=expected_fixed,
                dofs=entity.n_dofs,
                friction=[float(as_array(g.get_friction()).reshape(-1)[0]) for g in entity.geoms],
                sol_params=[
                    as_array(g.get_sol_params()).reshape(-1).tolist() for g in entity.geoms
                ],
            )
        scene.rigid_solver.collider.detection()

        def snapshot(step):
            objects = {}
            for name, entity in entities.items():
                if name == "table":
                    continue
                pose = render._entity_pose(entity)
                objects[name] = {
                    **pose,
                    "velocity": as_array(entity.get_vel()).reshape(3).tolist(),
                    "angular_velocity": as_array(entity.get_ang()).reshape(3).tolist(),
                    "net_contact_force": as_array(entity.get_links_net_contact_force())
                    .reshape(-1, 3)
                    .sum(axis=0)
                    .tolist(),
                }
            contact_data = scene.rigid_solver.collider.get_contacts(to_torch=False)
            contacts = []
            for i in range(len(contact_data["geom_a"])):
                ga, gb = int(contact_data["geom_a"][i]), int(contact_data["geom_b"][i])
                force = np.asarray(contact_data["force"][i])
                contacts.append(
                    dict(
                        a=owners[ga],
                        b=owners[gb],
                        geom_a=ga,
                        geom_b=gb,
                        link_a=int(contact_data["link_a"][i]),
                        link_b=int(contact_data["link_b"][i]),
                        position=contact_data["position"][i].tolist(),
                        normal=contact_data["normal"][i].tolist(),
                        penetration=float(contact_data["penetration"][i]),
                        force_a=(-force).tolist(),
                        force_b=force.tolist(),
                    )
                )
            return dict(step=step, time_s=step * cfg["dt"], objects=objects, contacts=contacts)

        with (out / "trace.jsonl").open("x") as stream:
            for step in range(cfg["steps"] + 1):
                if step:
                    scene.step()
                row = snapshot(step)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
        return rows, dict(
            genesis_commit=GENESIS_COMMIT,
            actual_bodies=actual,
            compute_backend="cpu",
            precision="32",
            scene_step_calls=cfg["steps"],
            sim_options=sim_options.model_dump(mode="json"),
            rigid_options=rigid_options.model_dump(mode="json"),
        )
    finally:
        gs.destroy()


def settled_package(package, states):
    if set(states) != {o.instance_id for o in package.env.objects}:
        raise ValueError("terminal object set differs")
    objects = tuple(
        o
        if o.static
        else replace(
            o,
            pose=Pose(
                position=tuple(float(v) for v in states[o.instance_id]["position"]),
                orientation_wxyz=tuple(float(v) for v in states[o.instance_id]["orientation_wxyz"]),
            ),
        )
        for o in package.env.objects
    )
    result = replace(package, env=replace(package.env, objects=objects))
    # Explicitly prove that all non-pose content is preserved.
    original, updated = package.to_dict(), result.to_dict()
    for before, after in zip(original["env"]["objects"], updated["env"]["objects"]):
        after["pose"] = before["pose"]
    if original != updated:
        raise ValueError("settled package modified non-pose content")
    return result


def run(manifest, out=None, *, scene_dir=None):
    if (out is None) == (scene_dir is None):
        raise ValueError('specify exactly one of output-dir or scene-dir')
    if scene_dir is None:
        return _run(manifest, out)
    task = TaskOutput(scene_dir)
    with task.lock():
        # Inspect input locations before clearing any prior physics or final render.
        inputs = [manifest]
        try:
            compiled = CompileResult.read(Path(manifest).resolve())
            inputs.append(compiled.artifact_path)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # The normal pipeline records invalid inputs after stale results are cleared.
        task.start_physics(protected_inputs=inputs)
        try:
            return _run(manifest, task.stage('physics'), final_dir=task.stage('final_render'),
                        prepared=True, input_check=task.verify_physics_inputs)
        finally:
            result_path = task.stage('physics') / 'physics_result.json'
            result = (json.loads(result_path.read_text()) if result_path.exists() else
                      dict(status='failed', physics_status='failed', render_status='not_run',
                           error='physics_interrupted'))
            task.finish_physics(result)


def _run(manifest, out, *, final_dir=None, prepared=False, input_check=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=prepared)
    terminal_root = Path(final_dir).resolve() if final_dir is not None else out / 'settled'
    render_log = (Path(final_dir) if final_dir is not None else out) / 'render.log'
    report = dict(
        schema=SCHEMA,
        status="failed",
        physics_status="not_run",
        render_status="not_run",
        physical_runtime_evidence=False,
    )
    try:
        manifest = Path(manifest).resolve()
        report["compile_manifest_sha256"] = render.sha256_file(manifest)
        compiled = CompileResult.read(manifest)
        if compiled.backend != "genesis" or compiled.status != "compiled" or compiled.blockers:
            raise ValueError("requires a successful Genesis compile manifest")
        package = import_compile_manifest(manifest)
        scene_path = Path(compiled.artifact_path)
        report["input_package_digest"] = package.digest()
        report["input_scene_sha256"] = render.sha256_file(scene_path)
        spec = json.loads(scene_path.read_text())
        render.validate_scene_config(spec)
        binding = render.verify_package_binding(spec, scene_path)
        render.verify_asset_integrity(spec, scene_path)
        if package.digest() != spec["package_digest"]:
            raise ValueError("manifest and scene package disagree")
        raw = config_for(package, spec)
        report.update(
            package_binding=binding,
            configuration=raw,
        )
        if input_check is not None:
            input_check()
        rows, actual = simulate(package, spec, out, raw)
        if input_check is not None:
            input_check()
        report["runtime"] = actual
        report["checks"] = evaluate(package, spec, rows)
        report["physics_status"] = (
            "passed" if all(c["passed"] for c in report["checks"]) else "failed"
        )
        report["physical_runtime_evidence"] = True
        if report["physics_status"] != "passed":
            report["failure_reasons"] = [c["name"] for c in report["checks"] if not c["passed"]]
            return 1
        # Recheck bytes after the run, before producing any terminal render package.
        if (
            render.sha256_file(scene_path) != report["input_scene_sha256"]
            or render.sha256_file(manifest) != report["compile_manifest_sha256"]
        ):
            raise ValueError("compile input changed during physics")
        render.verify_package_binding(spec, scene_path)
        render.verify_asset_integrity(spec, scene_path)
        settled = settled_package(package, rows[-1]["objects"])
        terminal = GenesisCompiler().compile(settled, terminal_root, strict=True)
        terminal_spec = json.loads(Path(terminal.artifact_path).read_text())
        render.verify_package_binding(terminal_spec, terminal.artifact_path)
        for obj in terminal_spec["objects"]:
            expected = next(o for o in settled.env.objects if o.instance_id == obj["instance_id"])
            if obj["pose"] != dict(
                position=list(expected.pose.position),
                orientation_wxyz=list(expected.pose.orientation_wxyz),
            ):
                raise ValueError("terminal compiler changed settled pose")
        report.update(
            settled_package_digest=settled.digest(),
            settled_compile_manifest=terminal.manifest_path,
            trace_sha256=render.sha256_file(out / "trace.jsonl"),
        )
        write_json(out / "physics_result.json", report)
        report["render_status"] = "running"
        with render_log.open("x") as log:
            completed = subprocess.run(
                terminal.runtime_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        report["render_status"] = "passed" if completed.returncode == 0 else "failed"
        if completed.returncode:
            raise ValueError(f"terminal rendering failed; see {render_log}")
        evidence_path = Path(terminal.runtime_command[-1]) / render.EVIDENCE_OUTPUT
        evidence = json.loads(evidence_path.read_text())
        if evidence["status"] != "success" or evidence["package_digest"] != settled.digest():
            raise ValueError("terminal render evidence does not match settled package")
        if input_check is not None:
            input_check()
        report.update(
            status="success",
            render_evidence=str(evidence_path),
            render_evidence_sha256=render.sha256_file(evidence_path),
            video_description=(
                "Orbit of the physically validated settled state; not a physics rollout"
            ),
        )
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["physics_status"] == "not_run":
            report["physics_status"] = "failed"
        if report["render_status"] in {"running", "passed"}:
            report["render_status"] = "failed"
        return 1
    finally:
        if report["status"] != "success":
            if report["physics_status"] == "not_run":
                report["physics_status"] = "failed"
            if report["render_status"] == "running":
                report["render_status"] = "failed"
        if (out / "trace.jsonl").exists():
            report["trace_sha256"] = render.sha256_file(out / "trace.jsonl")
        write_json(out / "physics_result.json", report)
        print(json.dumps(dict(output=str(out), status=report["status"], error=report.get("error"))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-manifest", required=True)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output-dir", help="旧独立输出目录")
    outputs.add_argument("--scene-dir", help="已有自然语言任务目录，重跑覆盖物理与终态阶段")
    args = parser.parse_args(argv)
    return run(args.compile_manifest, args.output_dir, scene_dir=args.scene_dir)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run native imported rigid scenes from authored poses, with durable physics/media evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_physics as evidence
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
from self_improving.sim_adapters.genesis import scene_physics_graph as graph_rules
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis import physics_criteria as criteria
from self_improving.sim_adapters.genesis import validate_single_asset as single
from self_improving.sim_adapters.genesis.physics_math import angle, rotation, sweep_radius
from self_improving.sim_adapters.genesis.validate_asset_scene import contacts

SCHEMA = "genenv.imported_scene_physics.v1"
array = standard.array


def grounded_forces(row, minimum, fixed=()):
    """Observed upward force paths to a fixed body; floating cycles are not support."""
    pairs = {}
    if row["step"] == 0:
        return {}, {}
    for c in row["contacts"]:
        for side, other in (("a", "b"), ("b", "a")):
            key = c[side], c[other]
            pairs[key] = pairs.get(key, 0.0) + c["force_" + side][2]
    grounded = set(fixed)
    while True:
        newly = {a for (a, b), force in pairs.items() if b in grounded and force > minimum}
        if newly <= grounded:
            break
        grounded |= newly
    return {
        n: sum(max(0.0, f) for (a, b), f in pairs.items() if a == n and b in grounded)
        for n in row["objects"]
    }, pairs


# A body with no measured geometry gets the lever arm that reproduces the limit pair this
# entrance used before: 0.01 m/s of slide against 0.05 rad/s of spin is a 0.2 m arm. The
# single weighted speed then says the same thing the two separate limits used to say.
LEGACY_SWEEP_RADIUS_M = 0.2


def radius(name, geometry):
    """Sweep radius from the measured hull; never a declared or assumed body size."""
    if geometry is None or name not in geometry:
        return LEGACY_SWEEP_RADIUS_M
    return sweep_radius(geometry[name])


def validate_scene_row(row, step, names, dt):
    if row["step"] != step or abs(row["time_s"] - step * dt) > 1e-9:
        raise ValueError("non-sequential trajectory")
    if set(row["objects"]) != set(names):
        raise ValueError("trajectory object mismatch")
    if row["contact_phase"] != ("initial_detection" if step == 0 else "solved_step"):
        raise ValueError("invalid contact phase")
    for state in row["objects"].values():
        for key in ("position", "velocity", "angular_velocity"):
            evidence.finite(state[key], (3,), key)
        rotation(state["orientation_wxyz"])
    for contact in row["contacts"]:
        if contact["a"] not in names or contact["b"] not in names:
            raise ValueError("unknown contact object")
        if contact["a"] == contact["b"]:
            raise ValueError("self contact unsupported")
        evidence.finite(contact["position"], (3,), "contact position")
        normal = evidence.finite(contact["normal"], (3,), "contact normal")
        if abs(np.linalg.norm(normal) - 1) > 1e-4:
            raise ValueError("invalid contact normal")
        if contact["penetration"] < 0 or not np.isfinite(contact["penetration"]):
            raise ValueError("invalid penetration")
        if step == 0:
            if contact["force_a"] is not None or contact["force_b"] is not None:
                raise ValueError("initial contact force must be unavailable")
        else:
            a = evidence.finite(contact["force_a"], (3,), "force_a")
            b = evidence.finite(contact["force_b"], (3,), "force_b")
            if not np.allclose(a, -b, atol=1e-7, rtol=1e-6):
                raise ValueError("inconsistent contact force pair")


def evaluate(rows, layout, cfg, geometry=None):
    bodies = {o["object_id"]: o for o in layout["objects"]}
    dynamic = {name: obj for name, obj in bodies.items() if not obj["fixed"]}
    fixed_bodies = {name for name, obj in bodies.items() if obj["fixed"]}
    if geometry is None and fixed_bodies and fixed_bodies != {"support_0"}:
        raise ValueError("legacy fixed nested bodies must remain dynamic")
    if len(rows) != cfg["steps"] + 1:
        raise ValueError("incomplete trajectory")
    if not dynamic:
        raise ValueError("finite support validation requires dynamic objects")
    support_roots = fixed_bodies or {"ground"}
    trajectory_names = set(bodies) | ({"ground"} if not fixed_bodies else set())
    for i, row in enumerate(rows):
        validate_scene_row(row, i, trajectory_names, cfg["dt"])
    force_rows = [
        grounded_forces(row, cfg["support_force_n"], support_roots) for row in rows
    ]
    results = {}
    for name, obj in dynamic.items():
        initial = rows[0]["objects"][name]
        if (not np.allclose(initial["position"], obj["translation_m"], atol=1e-6, rtol=0)
                or angle(initial["orientation_wxyz"], obj["orientation_wxyz"]) > 1e-4):
            raise ValueError("initial pose differs from imported scene")
        for field, source in (("velocity", "source_velocity_mps"),
                              ("angular_velocity", "source_angular_velocity_radps")):
            if not np.allclose(initial[field], obj[source], atol=1e-6, rtol=0):
                raise ValueError("initial velocity differs from imported scene")
        local_rows = [
            dict(row["objects"][name], step=row["step"], time_s=row["time_s"],
                 contacts=[c for c in row["contacts"] if name in (c["a"], c["b"])],
                 ground_up_force_n=None if i == 0 else force_rows[i][0][name])
            for i, row in enumerate(rows)
        ]
        checks, budget = single.evaluate(local_rows, cfg, radius(name, geometry))
        final = rows[-1]["objects"][name]
        failed = [c["name"] for c in checks if not c["passed"]]
        results[name] = dict(
            category=obj["category"], checks=checks,
            passed=all(c["passed"] for c in checks),
            # Whether a different numerical configuration could legitimately be tried, or
            # whether this body really moved and retuning would only hide it.
            failure_categories=criteria.classify(failed),
            numerics_tunable=criteria.tunable(failed),
            **budget,
            total_translation_m=float(np.linalg.norm(
                np.array(final["position"]) - initial["position"])),
            total_rotation_deg=angle(final["orientation_wxyz"], initial["orientation_wxyz"]))
    for name in fixed_bodies:
        initial, final = rows[0]["objects"][name], rows[-1]["objects"][name]
        if (np.linalg.norm(np.asarray(final["position"]) - initial["position"]) > 1e-7
                or angle(final["orientation_wxyz"], initial["orientation_wxyz"]) > 1e-5):
            raise ValueError("fixed support moved")
    window = force_rows[-round(cfg["window_s"] / cfg["dt"]):]
    observed = []
    for a, b in sorted({key for _, pairs in window for key in pairs}):
        if a == b:
            continue
        forces = [pairs.get((a, b), 0.0) for _, pairs in window]
        # Denominator is the samples where the pair is in contact at all. Contact-detection
        # dropout is already bounded per object above; charging it here as well would count
        # one artefact as both a lost contact and a missing support.
        touching = sum((a, b) in pairs for _, pairs in window)
        fraction = (
            sum(force > cfg["support_force_n"] for force in forces) / touching
            if touching else 0.0
        )
        if fraction:
            observed.append(dict(
                source=a, target=b, upward_force_fraction=fraction,
                mean_upward_force_n=float(np.mean(np.maximum(0, forces)))))
    relation_results = []
    for relation in layout.get("relations", []):
        if relation["relation"] != "on":
            continue
        fraction = max(
            [row["upward_force_fraction"] for row in observed
             if row["source"] == relation["source"] and row["target"] == relation["target"]],
            default=0.0)
        relation_results.append(dict(
            **relation, observed_support_fraction=fraction,
            limit=cfg["support_fraction"], passed=fraction >= cfg["support_fraction"]))
    stability = all(value["passed"] for value in results.values())
    declared = bool(relation_results)
    relations_passed = declared and all(r["passed"] for r in relation_results)
    graph_checks = graph_rules.evaluate(rows, layout, geometry, cfg) if geometry else None
    passed = stability and relations_passed and (graph_checks is None or graph_checks["passed"])
    physics_status = (
        "passed" if passed else "incomplete" if stability and not declared else "failed"
    )
    exit_code = 0 if passed else 3 if physics_status == "incomplete" else 2
    answer = dict(
        stability_status="passed" if stability else "failed",
        physics_status=physics_status, exit_code=exit_code,
        objects=results,
        relation_validation=(
            "passed" if relations_passed else
            "not_evaluated_no_declared_support_relations" if not declared else "failed"
        ),
        relation_results=relation_results, observed_support_contacts=observed,
        meaning="declared finite support is accepted only from solved upward contact force")
    if graph_checks is not None:
        answer["graph_checks"] = graph_checks
    return answer


def support_sdf(cell_size=None, max_res=None):
    """Explicit finite-support distance-field override; no acceptance threshold changes."""
    if cell_size is None and max_res is None:
        return {}
    if (isinstance(cell_size, bool) or not isinstance(cell_size, (int, float))
            or not np.isfinite(cell_size) or not 0.0005 <= cell_size <= 0.005
            or isinstance(max_res, bool) or not isinstance(max_res, int)
            or not 32 <= max_res <= 384):
        raise ValueError("support SDF requires cell size 0.0005..0.005 m and max res 32..384")
    return dict(sdf_cell_size=float(cell_size), sdf_max_res=max_res)


def run(scene_package, output_dir, *, profile="baseline", numerics=None, friction_multiplier=1.0,
        support_sdf_cell_size=None, support_sdf_max_res=None):
    sdf = support_sdf(support_sdf_cell_size, support_sdf_max_res)
    root, out = Path(scene_package).resolve(), Path(output_dir).resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("physics output and imported package must be separate")
    layout = imported.verify(root)
    # Whether a ground plane is acceptable is topology()'s call now: it is a fixed root
    # when a body explicitly declares it as support, and still no excuse for an undeclared
    # one. Rejecting every scene with a plane here made "on the floor" unrepresentable.
    graph_rules.topology(layout)
    if any(o["scale"] != 1 for o in layout["objects"]):
        raise ValueError("requires baked unit-scale assets")
    measured_geometry = graph_rules.geometry(root, layout)
    if not (0 < friction_multiplier <= 1.25):
        raise ValueError("invalid bounded friction multiplier")
    if any(
        np.any(imported.vector(o[k], 3))
        for o in layout["objects"]
        for k in ("source_velocity_mps", "source_angular_velocity_radps")
    ):
        raise ValueError("nonzero saved velocity restoration is not implemented")
    out.mkdir(parents=True, exist_ok=False)
    cfg = evidence.settings(profile, numerics)
    frozen = dict(
        schema_version=SCHEMA,
        layout=layout,
        settings=cfg,
        scene_package=str(root),
        scene_manifest_sha256=standard.library.sha256(root / "manifest.json"),
        genesis_commit=official.GENESIS_COMMIT,
        pose_policy="exact imported poses; no clearance, settling, or repair",
        support_policy="declared upward contact force paths reaching fixed roots",
        physics_profile=profile,
        # Frozen so a stored run states the footing it was measured on, and verify_evidence
        # can refuse a configuration a search invented for this one scene.
        numerics=numerics,
        numerics_profile=(numerics or {}).get("numerics_profile", "asset_physics_baseline"),
        numerics_origin=(numerics or {}).get("numerics_origin", "default"),
        friction_multiplier=friction_multiplier,
        support_sdf=sdf,
        validation_geometry=measured_geometry,
    )
    official.write_json(out / "physics_input.json", frozen)
    input_hash = standard.library.sha256(out / "physics_input.json")

    def verify():
        imported.verify(root)
        if (
            standard.library.sha256(root / "manifest.json") != frozen["scene_manifest_sha256"]
            or standard.library.sha256(out / "physics_input.json") != input_hash
        ):
            raise ValueError("physics input changed during simulation")

    result = dict(
        schema_version=SCHEMA,
        status="error",
        exit_code=1,
        physics_status="invalid",
        steps_executed=0,
        physics_input_sha256=input_hash,
        model_calls=0,
    )
    rows, sampled, initialized, writer = [], [], False, None
    try:
        os.environ["GS_HEADLESS"] = "1"
        os.environ["PYGLET_HEADLESS"] = "1"
        import genesis as gs
        import imageio.v2 as imageio
        from PIL import Image

        commit = subprocess.check_output(
            ["git", "-C", str(Path(gs.__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if commit != official.GENESIS_COMMIT:
            raise ValueError("Genesis revision mismatch")
        gs.init(backend=gs.cpu, seed=cfg["seed"], precision="32", logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
            renderer=gs.renderers.Rasterizer(),
            sim_options=gs.options.SimOptions(
                dt=cfg["dt"], substeps=1, gravity=tuple(cfg["gravity"])
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                iterations=cfg["iterations"],
                ls_iterations=cfg["ls_iterations"],
                tolerance=cfg["tolerance"],
                constraint_timeconst=cfg["constraint_timeconst"],
                use_hibernation=False,
            ),
            vis_options=gs.options.VisOptions(
                background_color=(1, 1, 1),
                ambient_light=(0.4, 0.4, 0.4),
                lights=official.LIGHTS,
                shadow=False,
            ),
        )
        entities, expected = {}, {}
        for obj in layout["objects"]:
            name = obj["object_id"]
            _, entry, physics = standard.verify_package(root / obj["standard_package"])
            expected[name] = standard.inspect(entry)
            morph = standard.morph(gs, entry, fixed=obj["fixed"])
            morph.pos, morph.quat = tuple(obj["translation_m"]), tuple(obj["orientation_wxyz"])
            entities[name] = scene.add_entity(
                morph,
                name=name,
                vis_mode="visual",
                material=gs.materials.Rigid(
                    friction=physics["friction"] * friction_multiplier,
                    **(sdf if obj["fixed"] else {}),
                ),
            )
        # The environment plane is a real entity, not just a contract root. Without it a
        # scene whose only support is the floor has nothing to rest on and every body falls
        # forever -- which the trace layer already anticipated (it expects a "ground" name
        # when no fixed body exists) but the build never provided.
        if layout["environment"]["ground"] is not None:
            # gs.morphs.Plane() sits at z=0; a scene declaring any other height would be
            # simulated against a floor it did not describe, so refuse rather than shift.
            if layout["environment"]["z_m"] != 0:
                raise ValueError("only a ground plane at z=0 is supported")
            entities["ground"] = scene.add_entity(gs.morphs.Plane(), name="ground")
        camera = scene.add_camera(
            res=(960, 720), GUI=False, pos=(1, -1, 1), lookat=(0, 0, 0), fov=35
        )
        scene.build()
        verify()
        loaded = dict(
            objects={},
            sim_options=scene.options.sim.model_dump(mode="json"),
            rigid_options=scene.options.rigid.model_dump(mode="json"),
        )
        for obj in layout["objects"]:
            name, entity = obj["object_id"], entities[obj["object_id"]]
            loaded["objects"][name] = standard.audit(
                entity, expected[name], collision=not obj["fixed"],
                friction=None if obj["fixed"] else obj["friction"] * friction_multiplier,
            )
            if obj["fixed"] and (not entity.base_link.is_fixed or not entity.geoms):
                raise ValueError("declared root is not fixed and collidable")
            actual = np.concatenate(
                [array(g.get_vverts()).reshape(-1, 3) for link in entity.links for g in link.vgeoms]
            )
            target = expected[name]["visual"] @ rotation(obj["orientation_wxyz"]).T
            error = standard.distance(actual, target + obj["translation_m"])
            if error > 1e-5:
                raise ValueError("loaded world geometry differs from imported layout")
            loaded["objects"][name].update(
                world_vertex_error_m=error,
                loaded_sdf=[dict(cell_size_m=np.asarray(g.sdf_cell_size).tolist(),
                                 resolution=np.asarray(g._sdf_res).tolist())
                            for g in entity.geoms],
                effective_contact_parameters=[
                    array(g.get_sol_params()).tolist() for g in entity.geoms
                ],
            )
        official.write_json(out / "loaded_scene.json", loaded)
        boxes = np.array([o["world_visual_bounds_m"] for o in layout["objects"]])
        low, high = boxes[:, 0].min(0), boxes[:, 1].max(0)
        center = (low + high) / 2
        distance = float(np.linalg.norm(high - low) / 2 / np.sin(np.deg2rad(35 / 2)) * 1.2)

        def view(direction, up=(0, 0, 1)):
            direction = np.asarray(direction, float)
            camera.set_pose(
                pos=(center + distance * direction / np.linalg.norm(direction)).tolist(),
                lookat=center.tolist(),
                up=up,
            )

        view([0, -1, 0.7])
        owners, links = {}, {}
        all_entities = dict(entities)
        for name, entity in all_entities.items():
            for link in entity.links:
                for geom in link.geoms:
                    owners[geom.idx], links[geom.idx] = name, link.idx
        (out / "frames").mkdir()
        writer = imageio.get_writer(out / "simulation.mp4", fps=25, codec="libx264")
        scene.rigid_solver.detect_collision()
        with (out / "trace.jsonl").open("x") as stream:
            for step in range(cfg["steps"] + 1):
                if step:
                    scene.step()
                    result["steps_executed"] = step
                cs = contacts(
                    scene.rigid_solver.collider.get_contacts(to_torch=False),
                    owners,
                    links,
                    initial=step == 0,
                )
                objects = {}
                for name, entity in all_entities.items():
                    objects[name] = dict(
                        position=array(entity.get_pos()).reshape(3).tolist(),
                        orientation_wxyz=array(entity.get_quat()).reshape(4).tolist(),
                        velocity=array(entity.get_vel()).reshape(3).tolist(),
                        angular_velocity=array(entity.get_ang()).reshape(3).tolist(),
                    )
                    if step and name != "ground":
                        total = np.zeros(3)
                        for c in cs:
                            if c["a"] == name:
                                total += c["force_a"]
                            elif c["b"] == name:
                                total += c["force_b"]
                        measured = array(entity.get_links_net_contact_force()).reshape(-1, 3).sum(0)
                        if not np.allclose(total, measured, atol=1e-5, rtol=1e-4):
                            raise ValueError(f"{name}: contact force observation mismatch")
                row = dict(
                    step=step,
                    time_s=step * cfg["dt"],
                    objects=objects,
                    contacts=cs,
                    contact_phase="initial_detection" if step == 0 else "solved_step",
                )
                validate_scene_row(row, step, entities, cfg["dt"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if step % 10 == 0:
                    rgb = np.asarray(camera.render(rgb=True, force_render=True)[0], dtype=np.uint8)
                    Image.fromarray(rgb).save(out / "frames" / f"step_{step:04d}.png")
                    writer.append_data(rgb)
                    sampled.append(
                        dict(step=step, sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    )
                if step % 100 == 0:
                    print(f"scene physics: {step}/{cfg['steps']}", flush=True)
        writer.close()
        writer = None
        durable = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
        if durable != rows:
            raise ValueError("durable trajectory mismatch")
        result.update(evaluate(durable, layout, cfg, measured_geometry), status="complete")
        # Even a stable trace lacks declared support semantics; keep renders diagnostic.
        (out / "diagnostics").mkdir()
        for name, direction, up in [
            ("overview", [0, -1, 0.7], [0, 0, 1]),
            ("top", [0, 0, 1], [0, 1, 0]),
            ("side", [1, -1, 0.7], [0, 0, 1]),
        ]:
            view(direction, up)
            rgb = np.asarray(camera.render(rgb=True, force_render=True)[0], dtype=np.uint8)
            Image.fromarray(rgb).save(out / "diagnostics" / f"final_{name}.png")
        decoded = [
            hashlib.sha256(f.tobytes()).hexdigest()
            for f in imageio.get_reader(out / "simulation.mp4")
        ]
        if len(decoded) != len(sampled):
            raise ValueError("encoded frame count differs from real samples")
        result["video"] = dict(
            total_frames=len(decoded),
            unique_frames=len(set(decoded)),
            unique_input_frames=len({r["sha256"] for r in sampled}),
            fps=25,
            sampled_every_steps=10,
            physics_duration_s=cfg["steps"] * cfg["dt"],
        )
        official.write_json(out / "frame_samples.json", sampled)
        verify()
    except Exception as exc:
        result.update(
            status="error",
            physics_status="invalid",
            exit_code=1,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if writer is not None:
            writer.close()
        if initialized:
            gs.destroy()
        if rows:
            official.write_json(out / "initial_state.json", rows[0])
            official.write_json(out / "final_state.json", rows[-1])
        try:
            verify()
        except Exception as exc:
            result.update(status="error", physics_status="invalid", exit_code=1, error=str(exc))
        result["artifacts"] = [
            official.fingerprint(p, out)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "physics_result.json"
        ]
        official.write_json(out / "physics_result.json", result)
    return result


def verify_evidence(directory):
    out = Path(directory).resolve()
    report = standard.library.read_json(out / "physics_result.json")
    official.verify_files(out, report["artifacts"])
    if report["status"] != "complete":
        raise ValueError("incomplete execution")
    if standard.library.sha256(out / "physics_input.json") != report["physics_input_sha256"]:
        raise ValueError("physics input binding mismatch")
    frozen = standard.library.read_json(out / "physics_input.json")
    if frozen.get("numerics_origin") not in (None, "default", "registered"):
        raise ValueError("evidence used an agent-proposed numerical configuration")
    if frozen["settings"] != evidence.settings(
            frozen["physics_profile"], frozen.get("numerics")):
        raise ValueError("unexpected physics thresholds")
    if (
        standard.library.sha256(Path(frozen["scene_package"]) / "manifest.json")
        != frozen["scene_manifest_sha256"]
    ):
        raise ValueError("source scene changed")
    if imported.verify(frozen["scene_package"]) != frozen["layout"]:
        raise ValueError("frozen layout differs from source scene")
    required = {
        "physics_input.json",
        "trace.jsonl",
        "loaded_scene.json",
        "initial_state.json",
        "final_state.json",
        "frame_samples.json",
        "simulation.mp4",
    }
    if not required <= {r["path"] for r in report["artifacts"]}:
        raise ValueError("missing artifact bindings")
    if report["steps_executed"] != frozen["settings"]["steps"]:
        raise ValueError("incomplete executed steps")
    rows = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    geometry = frozen.get("validation_geometry")
    if geometry is not None and geometry != graph_rules.geometry(
            Path(frozen["scene_package"]), frozen["layout"]):
        raise ValueError("validation geometry differs from source assets")
    result = evaluate(rows, frozen["layout"], frozen["settings"], geometry)
    if any(report[k] != v for k, v in result.items()):
        raise ValueError("persisted trajectory does not match verdict")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("baseline", "half_dt"), default=None)
    parser.add_argument("--friction-multiplier", type=float, default=1.0)
    parser.add_argument("--support-sdf-cell-size", type=float)
    parser.add_argument("--support-sdf-max-res", type=int)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    if args['profile'] is not None:
        args.pop('resume')
        result = run(**args)
    else:
        from self_improving.sim_adapters.genesis import scene_physics_workflow as workflow
        args.pop('profile')
        if args.pop('friction_multiplier') != 1.0:
            raise ValueError("scene workflow preserves authored friction")
        result = workflow.run(**args)
    print(json.dumps(result, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate an existing 02_scene using native Genesis physics, without rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import asset_physics as evidence
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.extract_assets import verified_binding
from self_improving.sim_adapters.genesis.physics_math import angle, rotation
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import TaskOutput

GROUND = "ground"
ARTIFACTS = (
    "physics_input.json",
    "asset_physics_report.json",
    "initial_state.json",
    "trace.jsonl",
    "final_state.json",
)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        + "\n"
    )
    temporary.replace(path)


def material_declarations(binding):
    """Record authored densities; post-build Genesis geometry does not expose these."""
    declarations = []
    for source in binding["source_files"]:
        if Path(source["path"]).suffix.lower() != ".xml":
            continue
        path = official.safe_file(Path(binding["source_root"]), source["path"])
        root = ET.parse(path).getroot()
        for geom in root.iter("geom"):
            if "density" in geom.attrib:
                value = float(geom.attrib["density"])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("invalid source density declaration")
                declarations.append(
                    dict(
                        source=source["path"],
                        geom=geom.get("name"),
                        mesh=geom.get("mesh"),
                        density_kg_m3=value,
                        meaning="authored XML declaration; native parser resolves inheritance",
                    )
                )
    return declarations


def prepare(task, clip_index, fixed_objects, profile):
    index, _ = clip.load_index(clip_index)
    index_hash = library.sha256(clip_index)
    document = library.read_json(task.stage("objects") / "asset_request.json")
    if document["request"] != (task.root / "request.txt").read_text():
        raise ValueError("original request mismatch")
    parents = spatial.relations(document)
    graph = library.read_json(task.stage("scene") / "scene_graph.json")
    layout = library.read_json(task.stage("scene") / "scene_layout.json")
    geometry = library.read_json(task.stage("scene") / "native_geometry.json")
    if (
        layout["scene_graph_sha256"] != clip.digest(graph)
        or layout["native_geometry_sha256"] != clip.digest(geometry)
        or graph["edges"] != document["relations"]
        or layout["relations"] != document["relations"]
    ):
        raise ValueError("graph/layout/upstream digest or relation mismatch")
    if graph["frame"] != dict(up="+Z", right="+X", front="-Y", units="m"):
        raise ValueError("unsupported scene frame")
    if layout["environment"] != dict(ground="genesis_builtin_plane", z_m=0):
        raise ValueError("unsupported ground environment")
    ids = {o["object_id"] for o in document["objects"]}
    if (
        GROUND in ids
        or len(layout["objects"]) != len(ids)
        or {o["object_id"] for o in layout["objects"]} != ids
        or {o["object_id"] for o in graph["nodes"]} != ids
    ):
        raise ValueError("scene object identity mismatch")
    fixed = set(fixed_objects)
    if fixed - ids or len(fixed) != len(fixed_objects):
        raise ValueError("unknown or duplicate fixed object")
    if fixed & set(parents):
        raise ValueError("a declared on-source cannot be fixed (fixed suspension)")
    bodies = {}
    for obj in document["objects"]:
        name = obj["object_id"]
        binding = verified_binding(
            obj, task.stage("objects") / "asset_selection" / name, index, index_hash
        )
        placed = next(o for o in layout["objects"] if o["object_id"] == name)
        for key in (
            "asset_id",
            "selection_sha256",
            "model_entrypoint",
            "source_root",
            "source_files",
            "official_index",
        ):
            if placed[key] != binding[key]:
                raise ValueError(f"{name}: layout asset binding mismatch: {key}")
        if (
            next(n["asset_id"] for n in graph["nodes"] if n["object_id"] == name)
            != binding["asset_id"]
        ):
            raise ValueError("graph asset binding mismatch")
        if (
            placed["scale"] != 1.0
            or placed["orientation_policy"] != "preserve_native_orientation"
            or placed["support"] != parents.get(name, GROUND)
        ):
            raise ValueError(f"{name}: unsupported scale/orientation/support")
        translation = evidence.finite(placed["translation_m"], (3,), "translation")
        native_bounds = evidence.finite(geometry[name]["bounds"], (2, 3), "native bounds")
        if not np.allclose(
            native_bounds, placed["local_visual_bounds_m"], atol=1e-8, rtol=0
        ) or not np.allclose(
            native_bounds + translation, placed["world_visual_bounds_m"], atol=1e-8, rtol=0
        ):
            raise ValueError(f"{name}: invalid placed bounds")
        source = local_path(binding["model_entrypoint"])
        if source.suffix.lower() not in (".xml", ".glb"):
            raise ValueError(f"{name}: first version supports native MJCF and GLB only")
        if source.suffix == ".xml" and name in fixed:
            raise ValueError("fixing native MJCF is unsupported; original joints are preserved")
        surface = geometry[name].get("surface")
        if name in parents.values() and surface is None:
            raise ValueError(f"{name}: missing measured support surface")
        bodies[name] = dict(
            binding,
            translation_m=translation.tolist(),
            native_geometry=geometry[name],
            world_visual_bounds_m=placed["world_visual_bounds_m"],
            support=parents.get(name, GROUND),
            fixed=name in fixed,
            surface=surface,
            source_density_declarations=material_declarations(binding),
            material_policy=(
                "preserve_MJCF" if source.suffix == ".xml" else "pinned_Rigid_defaults"
            ),
            material_defaults=dict(
                density_kg_m3=600.0,
                friction=1.0,
                source="pinned Genesis RHO_OBJECT / default_friction; assumptions",
            ),
            morph_options=dict(
                scale=1.0, convexify=False, decimate=False, watertighten=None, collision=True
            ),
        )
    data = dict(
        schema_version=evidence.SCHEMA,
        genesis_commit=official.GENESIS_COMMIT,
        request=document["request"],
        request_sha256=task.report["request_sha256"],
        input_snapshot_sha256=library.sha256(task.stage("physics") / "scene_input_manifest.json"),
        clip_index=dict(path=str(Path(clip_index).resolve()), sha256=index_hash),
        scene_graph_sha256=clip.digest(graph),
        layout_sha256=clip.digest(layout),
        settings=evidence.settings(profile),
        profile=profile,
        bodies=bodies,
        relations=document["relations"],
        fixed_objects=sorted(fixed),
        environment=dict(ground="genesis_builtin_plane", z_m=0, fixed=True, collision=True),
        root_support_policy="undeclared dynamic roots are supported by ground",
        soft_preferences_evaluated=False,
        model_calls=0,
        render_status="not_run",
    )

    def check_inputs():
        task.owner()
        task.verify_physics_inputs()
        task.verify_scene_inputs()
        if library.sha256(clip_index) != index_hash:
            raise ValueError("CLIP index changed during physics")
        for obj in document["objects"]:
            verified_binding(
                obj, task.stage("objects") / "asset_selection" / obj["object_id"], index, index_hash
            )

    check_inputs()
    return data, check_inputs


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pose(entity):
    return dict(
        position=array(entity.get_pos()).reshape(3).tolist(),
        orientation_wxyz=array(entity.get_quat()).reshape(4).tolist(),
    )


def contacts(raw, owners, links, *, initial=False):
    """Collider 'force' acts on B; export both sides without double counting entities."""
    required = ("geom_a", "geom_b", "link_a", "link_b", "position", "normal", "penetration")
    if not initial:
        required += ("force",)
    count = len(raw["geom_a"])
    if any(len(raw[k]) != count for k in required):
        raise ValueError("inconsistent contact arrays")
    mask = np.asarray(raw.get("valid_mask", np.ones(count, dtype=bool)))
    if mask.shape != (count,):
        raise ValueError("invalid contact mask shape")
    output = []
    for i in range(count):
        if not mask[i]:
            continue
        ga, gb, la, lb = (int(raw[k][i]) for k in required[:4])
        if ga not in owners or gb not in owners or links.get(ga) != la or links.get(gb) != lb:
            raise ValueError("unknown geom/link contact mapping")
        force = None if initial else array(raw["force"][i]).reshape(3)
        output.append(
            dict(
                a=owners[ga],
                b=owners[gb],
                geom_a=ga,
                geom_b=gb,
                link_a=la,
                link_b=lb,
                position=array(raw["position"][i]).tolist(),
                normal=array(raw["normal"][i]).tolist(),
                penetration=float(raw["penetration"][i]),
                force_a=None if initial else (-force).tolist(),
                force_b=None if initial else force.tolist(),
            )
        )
    return output


def validate_loaded(name, spec, actual):
    if actual["fixed"] != spec["fixed"]:
        raise ValueError(f"{name}: actual fixed state differs from declared role")
    if not spec["fixed"] and actual["dofs"] != 6:
        raise ValueError(f"{name}: dynamic rigid object requires six free DOFs")
    if not actual["collision_geoms"] or not actual["collision_enabled"]:
        raise ValueError(f"{name}: collision disabled or absent")
    if not spec["fixed"]:
        if not math_is_positive(actual["mass_kg"]):
            raise ValueError(f"{name}: invalid dynamic mass")
        moving = [link for link in actual["links"] if not link["fixed"]]
        if not moving:
            raise ValueError(f"{name}: no dynamic link")
        if sorted(link["dofs"] for link in moving if link["dofs"]) != [6]:
            raise ValueError(f"{name}: articulated or multiple floating roots unsupported")
        for link in moving:
            inertia = evidence.finite(link["inertia_kg_m2"], (3, 3), "inertia")
            if (
                not np.allclose(inertia, inertia.T, atol=1e-8)
                or np.linalg.eigvalsh(inertia).min() < 0
                or (link["dofs"] > 0 and np.linalg.eigvalsh(inertia).min() <= 0)
            ):
                raise ValueError(f"{name}: invalid inertia")
    if any(
        v is not None and not math_is_positive(v) for v in actual.get("authored_density_kg_m3", [])
    ):
        raise ValueError(f"{name}: invalid authored density")
    if any(not math_is_positive(v) for v in actual["friction"]):
        raise ValueError(f"{name}: invalid friction")
    if actual["max_bounds_error_m"] > 1e-5 or actual["native_mesh_matches"] is not True:
        raise ValueError(f"{name}: native/placed visual geometry mismatch")
    if actual["orientation_error_deg"] > 1e-4:
        raise ValueError(f"{name}: native orientation changed")


def math_is_positive(value):
    return bool(np.isfinite(value) and value > 0)


def reference_geometry(gs, data):
    """Reproduce 02 native visual loading in a separate zero-step, camera-free scene."""
    scene = gs.Scene(show_viewer=False)
    entities = {}
    for name, spec in data["bodies"].items():
        options = dict(spec["morph_options"],
                       file=str(local_path(spec["model_entrypoint"])), collision=False)
        morph = (
            gs.morphs.MJCF(**options)
            if Path(spec["model_entrypoint"]).suffix == ".xml"
            else gs.morphs.Mesh(**options, fixed=True)
        )
        entities[name] = scene.add_entity(morph, material=gs.materials.Rigid(), vis_mode="visual")
    scene.build()
    references = {}
    for name, entity in entities.items():
        vertices, faces = builder.entity_mesh(entity)
        digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
        if digest != data["bodies"][name]["native_geometry"]["mesh_sha256"]:
            raise ValueError(f"{name}: original 02 visual mesh cannot be reproduced")
        references[name] = vertices, faces
    return references


def simulate(data, out, check_inputs, progress):
    os.environ["GS_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "1"
    import genesis as gs

    cfg = data["settings"]
    commit = subprocess.check_output(
        ["git", "-C", str(Path(gs.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != data["genesis_commit"]:
        raise ValueError("Genesis revision mismatch")
    gs.init(
        backend=gs.cpu, seed=cfg["seed"], precision=cfg["precision"], logging_level=logging.WARNING
    )
    loaded, rows = {}, []
    load_report = dict(
        schema_version=evidence.SCHEMA,
        status="loading",
        genesis_commit=commit,
        bodies=loaded,
        cameras_created=0,
        render_calls=0,
    )
    try:
        references = reference_geometry(gs, data)
        check_inputs()
        sim_options = gs.options.SimOptions(
            dt=cfg["dt"], substeps=cfg["substeps"], gravity=tuple(cfg["gravity"])
        )
        rigid_options = gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=cfg["iterations"],
            ls_iterations=cfg["ls_iterations"],
            tolerance=cfg["tolerance"],
            use_hibernation=cfg["use_hibernation"],
            constraint_timeconst=cfg["constraint_timeconst"],
        )
        scene = gs.Scene(
            sim_options=sim_options, rigid_options=rigid_options, show_viewer=False, show_FPS=False
        )
        load_report.update(
            sim_options=sim_options.model_dump(mode="json"),
            rigid_options=rigid_options.model_dump(mode="json"),
        )
        entities = {GROUND: scene.add_entity(gs.morphs.Plane(collision=True), name=GROUND)}
        for name, spec in data["bodies"].items():
            source = local_path(spec["model_entrypoint"])
            options = dict(spec["morph_options"], file=str(source))
            if source.suffix == ".xml":
                morph = gs.morphs.MJCF(**options)
                material = gs.materials.Rigid()
            else:
                morph = gs.morphs.Mesh(**options, fixed=spec["fixed"])
                defaults = spec["material_defaults"]
                material = gs.materials.Rigid(
                    rho=defaults["density_kg_m3"], friction=defaults["friction"]
                )
            entities[name] = scene.add_entity(
                morph, material=material, name=name, vis_mode="visual"
            )
        scene.build()
        check_inputs()
        owners, links = {}, {}
        for name, entity in entities.items():
            for link in entity.links:
                for geom in link.geoms:
                    owners[geom.idx], links[geom.idx] = name, link.idx
            if name == GROUND:
                load_report["ground"] = dict(
                    fixed=bool(entity.base_link.is_fixed),
                    collision_geoms=len(entity.geoms),
                    friction=[float(array(g.get_friction())) for g in entity.geoms],
                )
                if not entity.base_link.is_fixed or not entity.geoms:
                    raise ValueError("invalid built-in ground")
                continue
            spec = data["bodies"][name]
            if not entity.geoms or not entity.morph.collision:
                loaded[name] = dict(
                    collision_geoms=len(entity.geoms),
                    collision_enabled=bool(entity.morph.collision),
                    status="failed",
                    error="collision disabled or absent",
                )
                raise ValueError(f"{name}: collision disabled or absent")
            native = pose(entity)
            vertices, faces = builder.entity_mesh(entity)
            native_mesh_hash = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
            local_vertices = (vertices - native["position"]) @ rotation(native["orientation_wxyz"])
            hull = local_vertices[ConvexHull(local_vertices).vertices]
            entity.set_pos(np.array(native["position"]) + spec["translation_m"])
            initial = pose(entity)
            placed_vertices, _ = builder.entity_mesh(entity)
            error = float(
                np.max(
                    np.abs(
                        np.asarray(official.bounds(placed_vertices)) - spec["world_visual_bounds_m"]
                    )
                )
            )
            inertias = array(entity.get_links_inertia()).reshape(-1, 3, 3)
            actual = dict(
                native_pose=native,
                initial_pose=initial,
                visual_hull_local_m=hull.tolist(),
                max_bounds_error_m=error,
                native_mesh_sha256=native_mesh_hash,
                native_mesh_matches=(
                    vertices.shape == references[name][0].shape
                    and np.array_equal(faces, references[name][1])
                    and np.allclose(vertices, references[name][0], atol=1e-5, rtol=0)
                ),
                native_vertex_error_m=float(np.max(np.abs(vertices - references[name][0]))),
                orientation_error_deg=angle(
                    native["orientation_wxyz"], initial["orientation_wxyz"]
                ),
                fixed=bool(entity.base_link.is_fixed),
                dofs=int(entity.n_dofs),
                mass_kg=float(array(entity.get_mass()).reshape(-1)[0]),
                collision_geoms=len(entity.geoms),
                collision_meshes=[
                    dict(
                        geom_index=g.idx,
                        watertight=bool(g.mesh.is_watertight),
                        collision_processing="native_nonconvex_no_repair",
                        inertia_estimation_note=(
                            "native loader may use convex hull when not watertight"
                        ),
                    )
                    for g in entity.geoms
                    if g.mesh is not None
                ],
                collision_enabled=bool(entity.morph.collision),
                collision_bounds_m=official.bounds(array(entity.get_verts()).reshape(-1, 3)),
                source_density_declarations=spec.get("source_density_declarations", []),
                density_observation=("authored XML plus material policy; "
                                     "no per-geom postbuild accessor"),
                fallback_density_kg_m3=spec["material_defaults"]["density_kg_m3"],
                inertia_source="native loader preserves authored inertial data where present",
                friction=[float(array(g.get_friction()).reshape(-1)[0]) for g in entity.geoms],
                sol_params=[array(g.get_sol_params()).reshape(-1).tolist() for g in entity.geoms],
                material=entity.material.model_dump(mode="json"),
                links=[
                    dict(
                        index=link.idx,
                        dofs=int(link.n_dofs),
                        fixed=bool(link.is_fixed),
                        mass_kg=float(array(link.get_mass())),
                        inertia_kg_m2=inertias[i].tolist(),
                    )
                    for i, link in enumerate(entity.links)
                ],
            )
            loaded[name] = actual
            validate_loaded(name, spec, actual)
        load_report.update(status="passed", geom_owners=owners, geom_links=links)
        write_json(out / "asset_physics_report.json", load_report)
        check_inputs()
        progress["phase"] = "initial"
        # detect_collision clears contact caches and performs detection without integrating time.
        pairs = scene.rigid_solver.detect_collision()
        raw = scene.rigid_solver.collider.get_contacts(to_torch=False)
        # In this pinned revision detect_collision() reads the first N physical slots,
        # whereas get_contacts() applies contact_sort_idx after pruning. Its returned
        # pairs are diagnostic only: use the canonical accessor for all contact fields.
        load_report["initial_detector"] = dict(
            detected_count=len(pairs),
            canonical_count=len(raw["geom_a"]),
            penetration_source="detect_collision then collider.get_contacts (sorted/pruned)",
            force_available=False,
        )
        if len(pairs) != len(raw["geom_a"]):
            raise ValueError("initial detector/contact count mismatch")

        def snapshot(step, contact_data):
            row = dict(
                step=step,
                time_s=step * cfg["dt"],
                contact_phase="initial_detection" if step == 0 else "solved_step",
                objects={
                    n: dict(
                        pose(e),
                        velocity=array(e.get_vel()).reshape(3).tolist(),
                        angular_velocity=array(e.get_ang()).reshape(3).tolist(),
                    )
                    for n, e in entities.items()
                },
                contacts=contacts(contact_data, owners, links, initial=step == 0),
            )
            if step:
                for name, spec in data["bodies"].items():
                    if spec["fixed"]:
                        continue
                    observed = (
                        array(entities[name].get_links_net_contact_force()).reshape(-1, 3).sum(0)
                    )
                    summed = np.zeros(3)
                    for contact in row["contacts"]:
                        if contact["a"] == name:
                            summed += contact["force_a"]
                        elif contact["b"] == name:
                            summed += contact["force_b"]
                    if not np.allclose(observed, summed, atol=1e-5, rtol=1e-4):
                        raise ValueError(f"{name}: contact force direction/cache mismatch")
                    row["objects"][name]["net_contact_force"] = observed.tolist()
            evidence.validate_row(row, step, data)
            return row

        initial = snapshot(0, raw)
        checks = evidence.initial_checks(data, loaded, initial)
        write_json(
            out / "initial_state.json",
            dict(state=initial, checks=checks, force_available=False, collision_verified=True),
        )
        with (out / "trace.jsonl").open("x") as stream:

            def append(row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                progress["last_complete_step"] = row["step"]

            append(initial)
            if not all(c["passed"] for c in checks):
                raise evidence.PhysicsFailure("initial penetration or ground check failed")
            progress["phase"] = "simulation"
            for step in range(1, cfg["steps"] + 1):
                progress["simulation_executed"] = True
                scene.step()
                progress["steps_executed"] = step
                append(snapshot(step, scene.rigid_solver.collider.get_contacts(to_torch=False)))
        return loaded, rows
    except BaseException as exc:
        load_report.update(
            status="passed" if load_report["status"] == "passed" else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        write_json(out / "asset_physics_report.json", load_report)
        if rows:
            write_json(
                out / "final_state.json",
                dict(state=rows[-1], passed=False, complete=rows[-1]["step"] == cfg["steps"]),
            )
        gs.destroy()


def run(scene_dir, clip_index, *, fixed_objects=(), profile="baseline", simulator=None):
    task = TaskOutput(scene_dir)
    clip_index = Path(clip_index).resolve()
    clip.separate(task.root, clip_index.parent)
    with task.lock():
        task.start_physics(protected_inputs=(clip_index,))
        out = task.stage("physics")
        started = time.perf_counter()
        report = dict(
            schema_version=evidence.SCHEMA,
            status="physics_failed",
            physics_status="failed",
            render_status="not_run",
            simulation_executed=False,
            steps_executed=0,
            last_complete_step=None,
            phase="input",
            checks=[],
            exit_code=1,
            model_calls=0,
            cameras_created=0,
            render_calls=0,
        )
        check_inputs = None
        input_hash = None
        try:
            data, check_inputs = prepare(task, clip_index, fixed_objects, profile)
            write_json(out / "physics_input.json", data)
            input_hash = library.sha256(out / "physics_input.json")
            report["phase"] = "loading"
            loaded, rows = (simulator or simulate)(data, out, check_inputs, report)
            report["phase"] = "evaluation"
            # Read back the durable trajectory; in-memory rows alone are not evidence.
            durable = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
            if durable != rows:
                raise ValueError("durable trajectory differs from sampled states")
            report["checks"] = evidence.evaluate(data, loaded, durable)
            passed = all(c["passed"] for c in report["checks"])
            report.update(
                status="physics_passed" if passed else "physics_failed",
                physics_status="passed" if passed else "failed",
                exit_code=0 if passed else 2,
                failure_kind=None if passed else "physical",
                phase="complete",
            )
        except (Exception, KeyboardInterrupt) as exc:
            physical = isinstance(exc, evidence.PhysicsFailure)
            report.update(
                error=f"{type(exc).__name__}: {exc}",
                failure_kind="physical" if physical else report["phase"],
                exit_code=2 if physical else 1,
            )
            initial_path = out / "initial_state.json"
            if initial_path.exists():
                report["checks"] = library.read_json(initial_path)["checks"]
        finally:
            try:
                task.verify_physics_inputs()
                if check_inputs is not None:
                    check_inputs()
                if (
                    input_hash is not None
                    and library.sha256(out / "physics_input.json") != input_hash
                ):
                    raise ValueError("frozen physics input changed")
            except Exception as exc:
                report.update(
                    status="physics_failed",
                    physics_status="failed",
                    exit_code=1,
                    failure_kind="integrity",
                    integrity_error=str(exc),
                )
            final_path = out / "final_state.json"
            if final_path.exists():
                terminal = library.read_json(final_path)
                terminal["passed"] = report["physics_status"] == "passed"
                write_json(final_path, terminal)
            report.update(
                total_s=time.perf_counter() - started,
                physics_input_sha256=input_hash,
                artifacts=[
                    official.fingerprint(p, out)
                    for p in sorted(out.iterdir())
                    if p.is_file() and p.name != "physics_result.json"
                ],
                not_generated={
                    name: f"stopped at {report['phase']}"
                    for name in ARTIFACTS
                    if not (out / name).exists()
                },
            )
            write_json(out / "physics_result.json", report)
            task.finish_physics(report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--clip-index", type=Path, required=True)
    parser.add_argument("--fixed-object", action="append", default=[])
    parser.add_argument("--profile", choices=["baseline", "half_dt"], default="baseline")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.scene_dir, args.clip_index, fixed_objects=args.fixed_object, profile=args.profile
        )
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(dict(status="physics_failed", error=f"{type(exc).__name__}: {exc}")))
        return 1
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "exit_code", "simulation_executed", "steps_executed", "total_s")
            },
            ensure_ascii=False,
        )
    )
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

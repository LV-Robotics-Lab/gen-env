"""Diagnostic cup replay replacing its fixed table with an explicitly declared analytic plane."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets as preparation
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import validate_asset_scene as native
from self_improving.sim_adapters.genesis.physics_math import angle, rotation
from self_improving.sim_adapters.genesis.storage_paths import local_path

SCHEMA = "genenv.text_plane_control.v1"
PLANE = "analytic_plane"


def source_pair(data):
    """Require an explicit cup/table pair and a horizontal measured support plane."""
    physics.validate_settings(data)
    assets, poses = data["assets"], data["poses"]
    cups = [n for n, a in assets.items() if a["category"] == "cup" and not a["fixed"]]
    if len(cups) != 1:
        raise ValueError("plane control requires exactly one dynamic cup")
    cup = cups[0]
    parent = assets[cup]["support"]
    if parent not in assets or set(assets) != {cup, parent} or set(poses) != set(assets):
        raise ValueError("plane control requires only the cup and its declared table")
    table = assets[parent]
    if not table["fixed"] or table["category"] != "table" or not table.get("surface"):
        raise ValueError("plane control requires a fixed table with a measured support surface")
    normal = rotation(poses[parent]["orientation_wxyz"]) @ np.array([0.0, 0.0, 1.0])
    if not np.allclose(normal, [0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("plane control only supports horizontal table surfaces")
    point = geo.transform([[0, 0, table["surface"]["z_m"]]], poses[parent])[0]
    if not np.isfinite(point).all():
        raise ValueError("nonfinite measured support plane")
    return cup, parent, float(point[2])


def summarize(rows, cfg, cup, radius):
    if len(rows) != cfg["steps"] + 1:
        raise ValueError("incomplete plane-control trajectory")
    for step, row in enumerate(rows):
        if row["step"] != step or abs(row["time_s"] - step * cfg["dt"]) > 1e-9:
            raise ValueError("nonsequential plane-control trajectory")
        if set(row["objects"]) != {cup}:
            raise ValueError("unexpected plane-control object")
        state = row["objects"][cup]
        rotation(state["orientation_wxyz"])
        for key in ("position", "orientation_wxyz", "com_position", "velocity", "angular_velocity"):
            if not np.isfinite(state[key]).all():
                raise ValueError("nonfinite plane-control state")
        if not np.isfinite([state["collision_bottom_z_m"], state["visual_bottom_z_m"]]).all():
            raise ValueError("nonfinite plane-control bottom geometry")
        total = np.zeros(3)
        for contact in row["contacts"]:
            if {contact["a"], contact["b"]} != {cup, PLANE}:
                raise ValueError("unexpected plane-control contact pair")
            if not np.isfinite(contact["penetration"]) or contact["penetration"] < 0:
                raise ValueError("invalid plane-control penetration")
            if step:
                a, b = np.asarray(contact["force_a"]), np.asarray(contact["force_b"])
                if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
                    raise ValueError("invalid plane-control contact force")
                if not np.allclose(a, -b, atol=1e-7, rtol=1e-6):
                    raise ValueError("inconsistent plane-control contact force")
                total += a if contact["a"] == cup else b
            elif contact["force_a"] is not None or contact["force_b"] is not None:
                raise ValueError("initial plane-control contact force is unavailable")
        if step and not np.isclose(state["plane_up_force_n"], total[2], atol=1e-7, rtol=1e-6):
            raise ValueError("plane-control support force differs from contacts")
    window = [r for r in rows if r["time_s"] > cfg["window_start_s"] + 1e-9]
    if len(window) != cfg["window_samples"]:
        raise ValueError("invalid plane-control measurement window")
    speeds = [float(np.linalg.norm(r["objects"][cup]["velocity"])) for r in window]
    angular = [float(np.linalg.norm(r["objects"][cup]["angular_velocity"])) for r in window]
    effective = np.maximum(speeds, radius * np.array(angular))
    forces = [r["objects"][cup]["plane_up_force_n"] for r in window]
    bottoms = [r["objects"][cup]["collision_bottom_z_m"] for r in window]
    return dict(
        diagnostic_only=True,
        window_samples=len(window),
        stable_fraction=float(np.mean(effective < cfg["effective_speed_mps"])),
        support_fraction=float(np.mean(np.array(forces) > cfg["support_force_n"])),
        effective_velocity_max_mps=float(effective.max()),
        com_speed_max_mps=max(speeds),
        angular_speed_max_radps=max(angular),
        maximum_penetration_m=max(
            (c["penetration"] for r in rows for c in r["contacts"]), default=0.0
        ),
        no_contact_samples=sum(not r["contacts"] for r in window),
        mean_plane_up_force_n=float(np.mean(forces)),
        collision_bottom_range_m=float(np.ptp(bottoms)),
        collision_bottom_minimum_z_m=min(bottoms),
        source_scene_acceptance="not_evaluated",
    )


def run(source_input, output_dir, numerics_profile):
    source, out = Path(source_input).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("plane-control output must be a new directory")
    data = lib.read_json(source)
    cup, table, z = source_pair(data)
    if numerics_profile == "legacy":
        raise ValueError("plane control requires a registered numerical candidate")
    cfg = physics.settings(geo.PROFILE, numerics_profile)
    asset = data["assets"][cup]
    sources = []
    for a in data["assets"].values():
        for root_key, records_key in (
            ("source_root", "source_files"),
            ("derived_root", "derived_files"),
        ):
            if root_key not in a or records_key not in a:
                raise ValueError("plane-control source lacks asset file bindings")
            root = local_path(a[root_key]).resolve()
            if out.is_relative_to(root) or root.is_relative_to(out):
                raise ValueError("plane-control output overlaps source assets")
            sources.append((root, a[records_key]))
    bound_paths = {root / record["path"] for root, records in sources for record in records}
    if local_path(asset["physics_file"]).resolve() not in bound_paths:
        raise ValueError("plane-control collision entrypoint is not hash-bound")
    source_hash = lib.sha256(source)

    def check():
        if lib.sha256(source) != source_hash:
            raise ValueError("plane-control source input changed")
        for root, files in sources:
            official.verify_files(root, files)
        if lib.sha256(out / "diagnostic_input.json") != input_hash:
            raise ValueError("frozen plane-control input changed")

    out.mkdir(parents=True)
    frozen = dict(
        schema_version=SCHEMA,
        diagnostic_only=True,
        source_input=str(source),
        source_input_sha256=source_hash,
        source=data,
        cup=cup,
        replaced_object=table,
        replacement=dict(
            type="analytic_plane",
            name=PLANE,
            z_m=z,
            friction=1.0,
            meaning="infinite plane; not the source table collision geometry",
        ),
        settings=cfg,
        numerics_profile=numerics_profile,
        source_scene_acceptance="not_evaluated",
    )
    clip.write_json(out / "diagnostic_input.json", frozen)
    input_hash = lib.sha256(out / "diagnostic_input.json")
    report = dict(
        schema_version=SCHEMA,
        diagnostic_only=True,
        status="error",
        exit_code=1,
        physics_status="not_evaluated",
        source_scene_acceptance="not_evaluated",
        diagnostic_input_sha256=input_hash,
        source_input_sha256=source_hash,
        numerics_profile=numerics_profile,
        steps_executed=0,
        simulation_executed=False,
        replaced_object=table,
        cup=cup,
        analytic_plane_z_m=z,
    )
    gs, rows = None, []
    try:
        check()
        gs = preparation.init_genesis()
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
            sim_options=gs.options.SimOptions(
                dt=cfg["dt"], substeps=cfg["substeps"], gravity=tuple(cfg["gravity"])
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                iterations=cfg["iterations"],
                tolerance=cfg["tolerance"],
                use_hibernation=False,
                constraint_timeconst=cfg["constraint_timeconst"],
                friction_cone=gs.friction_cone.pyramidal,
                contact_resolution=gs.contact_resolution.convex,
                impratio=1.0,
                max_collision_pairs=cfg["max_collision_pairs"],
                max_contacts=cfg["max_contacts"],
            ),
        )
        plane = scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, z), collision=True),
            material=gs.materials.Rigid(friction=1.0),
            name=PLANE,
        )
        options = dict(
            file=str(local_path(asset["physics_file"])),
            scale=asset["scale"] if asset["native_collision"] else 1.0,
            convexify=False,
            decimate=False,
            watertighten=None,
            collision=True,
        )
        morph = (
            gs.morphs.MJCF(**options)
            if Path(asset["physics_file"]).suffix == ".xml"
            else gs.morphs.Mesh(**options, fixed=False)
        )
        entity = scene.add_entity(
            morph, material=gs.materials.Rigid(), name=cup, vis_mode="collision"
        )
        scene.build()
        reference = native.pose(entity)
        if entity.n_dofs != 6 or entity.base_link.is_fixed or not entity.geoms:
            raise ValueError("plane-control cup is not a colliding free body")
        physics.apply_pose(entity, asset, data["poses"][cup], reference)
        if len(entity.geoms) != len(asset["collision_hulls"]):
            raise ValueError("plane-control collision part count changed")
        error = max(
            physics.vertex_set_error(
                native.array(g.get_verts()).reshape(-1, 3), geo.transform(h, data["poses"][cup])
            )
            for g, h in zip(entity.geoms, asset["collision_hulls"])
        )
        if error > 1e-5:
            raise ValueError("plane-control initial collision geometry changed")
        mass, center, inertia = preparation.aggregate_inertia(entity, 1, np.zeros(3), gs)
        axes = rotation(data["poses"][cup]["orientation_wxyz"])
        canonical_com = geo.inverse([center], data["poses"][cup])[0]
        canonical_inertia = axes.T @ inertia @ axes
        if not np.isclose(mass, asset["mass_kg"], rtol=1e-5, atol=1e-9):
            raise ValueError("plane-control cup mass changed")
        if not np.allclose(canonical_com, asset["com_local_m"], atol=1e-5, rtol=0):
            raise ValueError("plane-control cup COM changed")
        if not np.allclose(canonical_inertia, asset["inertia_local_kg_m2"], atol=1e-10, rtol=1e-4):
            raise ValueError("plane-control cup inertia changed")
        audit = numerics.apply_contact_parameters({cup: entity, PLANE: plane}, cfg)
        clip.write_json(
            out / "loaded_scene.json",
            dict(
                initial_collision_vertex_error_m=error,
                mass_kg=mass,
                com_local_m=canonical_com.tolist(),
                inertia_local_kg_m2=canonical_inertia.tolist(),
                contact_parameter_audit=audit,
                rigid_options=scene.rigid_solver._options.model_dump(mode="json"),
                source_world_pose=data["poses"][cup],
                diagnostic_only=True,
            ),
        )
        owners, links = {}, {}
        for name, body in ((cup, entity), (PLANE, plane)):
            for link in body.links:
                for g in link.geoms:
                    owners[g.idx], links[g.idx] = name, link.idx
        ids = [link.idx for link in entity.links]
        masses = native.array(entity.get_links_mass()).reshape(-1)
        weights = masses / masses.sum()
        scene.rigid_solver.detect_collision()
        with (out / "trace.jsonl").open("x") as stream:
            for step in range(cfg["steps"] + 1):
                if step:
                    scene.step()
                    report.update(steps_executed=step, simulation_executed=True)
                contacts = native.contacts(
                    scene.rigid_solver.collider.get_contacts(to_torch=False),
                    owners,
                    links,
                    initial=step == 0,
                )
                numerics.check_capacity(scene.rigid_solver, cfg, len(contacts))
                raw = native.pose(entity)
                delta = (
                    rotation(raw["orientation_wxyz"]) @ rotation(reference["orientation_wxyz"]).T
                )
                offset = np.asarray(reference["position"]) - (
                    np.asarray(asset["anchor_m"]) if asset["native_collision"] else 0.0
                )
                pose = dict(
                    position=(np.asarray(raw["position"]) - delta @ offset).tolist(),
                    orientation_wxyz=geo.quat(delta),
                )
                if step == 0 and (
                    not np.allclose(
                        pose["position"], data["poses"][cup]["position"], atol=1e-5, rtol=0
                    )
                    or angle(pose["orientation_wxyz"], data["poses"][cup]["orientation_wxyz"])
                    > 1e-4
                ):
                    raise ValueError("plane-control cup initial world pose changed")
                com = native.array(
                    scene.rigid_solver.get_links_pos(ids, ref=gs.link_ref_frame.link_COM)
                ).reshape(-1, 3)
                velocity = native.array(
                    scene.rigid_solver.get_links_vel(ids, ref=gs.link_ref_frame.link_COM)
                ).reshape(-1, 3)
                force = (
                    None
                    if not step
                    else sum(
                        (
                            np.asarray(c["force_a" if c["a"] == cup else "force_b"])
                            for c in contacts
                        ),
                        np.zeros(3),
                    )
                )
                if step and not np.allclose(
                    force,
                    native.array(entity.get_links_net_contact_force()).reshape(-1, 3).sum(0),
                    atol=1e-5,
                    rtol=1e-4,
                ):
                    raise ValueError("plane-control contact force mapping mismatch")
                state = dict(
                    pose,
                    com_position=(weights @ com).tolist(),
                    velocity=(weights @ velocity).tolist(),
                    angular_velocity=native.array(entity.get_ang()).reshape(3).tolist(),
                    net_contact_force=None if force is None else force.tolist(),
                    plane_up_force_n=None if force is None else float(force[2]),
                    collision_bottom_z_m=float(
                        native.array(entity.get_verts()).reshape(-1, 3)[:, 2].min()
                    ),
                    visual_bottom_z_m=float(geo.transform(asset["hull"], pose)[:, 2].min()),
                )
                row = dict(
                    step=step,
                    time_s=step * cfg["dt"],
                    objects={cup: state},
                    contacts=contacts,
                    contact_phase="solved_step" if step else "initial_detection",
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
        check()
        durable = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
        if durable != rows:
            raise ValueError("plane-control durable trajectory mismatch")
        report.update(
            status="complete", exit_code=0, metrics=summarize(durable, cfg, cup, asset["radius_m"])
        )
    except BaseException as exc:
        report.update(status="error", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        if gs is not None:
            gs.destroy()
        if rows:
            clip.write_json(out / "initial_state.json", rows[0])
            clip.write_json(out / "final_state.json", rows[-1])
        try:
            check()
        except Exception as exc:
            report.update(status="error", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        report["artifacts"] = [
            official.fingerprint(p, out)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "diagnostic_result.json"
        ]
        clip.write_json(out / "diagnostic_result.json", report)
    return report

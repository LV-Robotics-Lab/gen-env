"""Explicit COM-based text_repair_v1 physics, plus simulator-independent evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from self_improving.sim_adapters.genesis import asset_physics as original
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis import validate_asset_scene as native
from self_improving.sim_adapters.genesis.physics_math import angle, rotation
from self_improving.sim_adapters.genesis.storage_paths import local_path

SETTINGS = dict(
    profile=geo.PROFILE,
    genesis_commit=official.GENESIS_COMMIT,
    dt=0.002,
    steps=1500,
    substeps=1,
    window_start_s=2.0,
    window_samples=500,
    seed=0,
    gravity=[0.0, 0.0, -9.81],
    iterations=50,
    tolerance=1e-8,
    constraint_timeconst=0.01,
    friction_cone="pyramidal",
    contact_resolution="convex",
    impratio=1.0,
    use_hibernation=False,
    precision="32",
    backend="cpu",
    effective_speed_mps=0.01,
    stable_fraction=0.95,
    support_fraction=0.95,
    support_force_n=1e-6,
    penetration_m=0.001,
    penetration_relative=0.01,
    surface_tilt_deg=0.5,
    scene_extent_m=4.0,
)


GT75_PROFILE = "text_repair_gt75_v1"
PROFILES = (geo.PROFILE, GT75_PROFILE)


def settings(profile=geo.PROFILE):
    """Separate user-requested acceptance criteria from the original frozen baseline."""
    if profile not in PROFILES:
        raise ValueError("unknown physics profile")
    cfg = copy.deepcopy(SETTINGS)
    if profile == GT75_PROFILE:
        cfg.update(profile=profile, stable_fraction=0.75, support_fraction=0.75,
                   fraction_comparison="strictly_greater")
    return cfg


def fraction_passes(value, cfg, key):
    if cfg.get("fraction_comparison") == "strictly_greater":
        return value > cfg[key]
    return value >= cfg[key]


def vertex_set_error(actual, expected):
    return float(max(cKDTree(actual).query(expected)[0].max(),
                     cKDTree(expected).query(actual)[0].max()))


def penetration_limit(a, b, assets):
    sizes = [assets[n]["diagonal_m"] for n in (a, b) if n != "ground"]
    return min(SETTINGS["penetration_m"], SETTINGS["penetration_relative"] * min(sizes))


def penetration_ok(contact, assets):
    p = contact["penetration"]
    return p <= SETTINGS["penetration_m"] and all(
        p / assets[n]["diagonal_m"] < SETTINGS["penetration_relative"]
        for n in (contact["a"], contact["b"])
        if n != "ground"
    )


def evaluate(data, rows, *, initial_rejected=False):
    assets, cfg = data["assets"], data["settings"]
    if cfg != settings(data.get("profile")):
        raise ValueError("frozen profile settings mismatch")
    if not assets or set(data["poses"]) != set(assets):
        raise ValueError("invalid frozen object set")
    geo.topology(assets)
    expected = 1 if initial_rejected else cfg["steps"] + 1
    if len(rows) != expected:
        raise ValueError("incomplete trajectory")
    for i, row in enumerate(rows):
        original.validate_row(row, i, dict(bodies=assets, settings=cfg))
        for n, state in row["objects"].items():
            original.finite(state["com_position"], (3,), "COM position")
            if n in assets:
                if i == 0 and (
                    np.max(np.abs(np.asarray(state["position"]) - data["poses"][n]["position"]))
                    > 1e-5
                    or angle(state["orientation_wxyz"], data["poses"][n]["orientation_wxyz"]) > 1e-4
                ):
                    raise ValueError("initial pose differs from frozen input")
    if initial_rejected and all(penetration_ok(c, assets) for c in rows[0]["contacts"]):
        raise ValueError("initial rejection without excessive penetration")
    window = [r for r in rows if r["time_s"] > cfg["window_start_s"] + 1e-9]
    if not initial_rejected and len(window) != cfg["window_samples"]:
        raise ValueError("invalid evaluation window")
    failures = {n: [] for n in assets}
    metrics = {n: {} for n in assets}
    allowed = {frozenset((n, a["support"])) for n, a in assets.items()}
    for row in rows:
        for c in row["contacts"]:
            if not penetration_ok(c, assets):
                for n in (c["a"], c["b"]):
                    if n in assets and "penetration" not in failures[n]:
                        failures[n].append("penetration")
        for n, a in assets.items():
            world = geo.transform(a["hull"], row["objects"][n])
            if np.abs(world[:, :2]).max() > cfg["scene_extent_m"] / 2 + 1e-6:
                if "scene_bounds" not in failures[n]:
                    failures[n].append("scene_bounds")
            if world[:, 2].min() < -0.001:
                if "below_ground" not in failures[n]:
                    failures[n].append("below_ground")
    for row in window:
        for c in row["contacts"]:
            if frozenset((c["a"], c["b"])) not in allowed:
                for n in (c["a"], c["b"]):
                    if n in assets and "unexpected_contact" not in failures[n]:
                        failures[n].append("unexpected_contact")
    for n, a in assets.items():
        if initial_rejected:
            metrics[n]["evaluated_window"] = False
            continue
        states = [r["objects"][n] for r in window]
        effective = [
            max(
                np.linalg.norm(s["velocity"]), a["radius_m"] * np.linalg.norm(s["angular_velocity"])
            )
            for s in states
        ]
        stable = sum(v < cfg["effective_speed_mps"] for v in effective) / len(states)
        hits, min_margin, min_ratio, max_tilt = 0, None, 1.0, 0.0
        for row in window:
            force = sum(
                c["force_a" if c["a"] == n else "force_b"][2]
                for c in row["contacts"]
                if {c["a"], c["b"]} == {n, a["support"]}
            )
            hits += force > cfg["support_force_n"]
            state = row["objects"][n]
            up = rotation(state["orientation_wxyz"]) @ np.array(a["natural_up"])
            max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))))
            parent = a["support"]
            if parent != "ground":
                ps = row["objects"][parent]
                tilt = float(
                    np.degrees(np.arccos(np.clip(rotation(ps["orientation_wxyz"])[2, 2], -1, 1)))
                )
                if tilt > cfg["surface_tilt_deg"] and "unsupported_support_tilt" not in failures[n]:
                    failures[n].append("unsupported_support_tilt")
                margin, ratio = geo.support_metrics(a, state, assets[parent], ps)
                min_margin = margin if min_margin is None else min(margin, min_margin)
                min_ratio = min(min_ratio, ratio)
        support = hits / len(states)
        if not a["fixed"]:
            if not fraction_passes(stable, cfg, "stable_fraction"):
                failures[n].append("stable_velocity")
            if not fraction_passes(support, cfg, "support_fraction"):
                failures[n].append("support_preserved")
            if max_tilt > a["tip_limit_deg"]:
                failures[n].append("tipped")
        if min_margin is not None and (min_margin < a["margin_m"] - 1e-8 or min_ratio < 1 - 1e-7):
            failures[n].append("support_geometry")
        first, last = rows[0]["objects"][n], rows[-1]["objects"][n]
        drift = float(
            np.linalg.norm(np.asarray(last["position"]) - first["position"]) / a["diagonal_m"]
        )
        turn = angle(last["orientation_wxyz"], first["orientation_wxyz"])
        object_contacts = [c for row in rows for c in row["contacts"] if n in (c["a"], c["b"])]
        maximum_penetration = max((c["penetration"] for c in object_contacts), default=0.0)
        metrics[n] = dict(
            maximum_penetration_m=maximum_penetration,
            maximum_relative_penetration=maximum_penetration / a["diagonal_m"],
            all_contact_objects=sorted(
                {c["b"] if c["a"] == n else c["a"] for c in object_contacts}
            ),
            effective_velocity_max_mps=max(effective),
            stable_fraction=stable,
            support_fraction=support,
            minimum_margin_m=min_margin,
            minimum_support_ratio=min_ratio,
            maximum_tilt_deg=max_tilt,
            normalized_drift=drift,
            rotation_drift_deg=turn,
            drift_score=drift / 0.01 + turn / 3,
            drift_is_hard=False,
        )
    for relation in data["relations"]:
        if relation["relation"] == "on":
            continue
        a, b = relation["source"], relation["target"]
        for row in window:
            boxes = [
                geo.bounds(geo.transform(assets[n]["hull"], row["objects"][n])) for n in (a, b)
            ]
            if not spatial.relation_ok(relation["relation"], *boxes):
                for n in (a, b):
                    if "text_relation" not in failures[n]:
                        failures[n].append("text_relation")
                break
    dynamic = [n for n, a in assets.items() if not a["fixed"]]
    passed = not any(failures.values()) and not initial_rejected
    return dict(
        profile=data["profile"],
        passed=passed,
        failures=failures,
        objects=metrics,
        complete=not initial_rejected,
        simulation_executed=len(rows) > 1,
        steps_executed=len(rows) - 1,
        stable_ratio=(sum(not failures[n] for n in dynamic) / len(dynamic) if dynamic else 1.0),
        support_preservation_ratio=(
            sum(fraction_passes(metrics[n].get("support_fraction", 0.0), cfg,
                                "support_fraction") for n in dynamic) / len(dynamic)
            if dynamic
            else 1.0
        ),
        penetration_ratio=sum("penetration" in f for f in failures.values()) / len(assets),
    )


def apply_pose(entity, asset, pose, reference):
    delta = rotation(pose["orientation_wxyz"])
    offset = np.asarray(reference["position"]) - (
        np.asarray(asset["anchor_m"]) if asset["native_collision"] else 0.0
    )
    entity.set_quat(np.asarray(geo.quat(delta @ rotation(reference["orientation_wxyz"]))))
    entity.set_pos(np.asarray(pose["position"]) + delta @ offset)


def simulate(data, out, check):
    if data["settings"] != settings(data.get("profile")):
        raise ValueError("frozen profile settings mismatch")
    clip.write_json(out / "visual_initial_report.json", repair_assets.verify_visual_input(data))
    check()
    gs = repair_assets.init_genesis()
    cfg, assets = data["settings"], data["assets"]
    rows, loaded = [], {}
    executed = 0
    report = dict(status="loading", profile=data["profile"], bodies=loaded, cameras_created=0)
    try:
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
            ),
        )
        entities = {"ground": scene.add_entity(gs.morphs.Plane(collision=True))}
        for n, a in assets.items():
            options = dict(
                file=str(local_path(a["physics_file"])),
                scale=a["scale"] if a["native_collision"] else 1.0,
                convexify=False,
                decimate=False,
                watertighten=None,
                collision=True,
            )
            morph = (
                gs.morphs.MJCF(**options)
                if Path(a["physics_file"]).suffix == ".xml"
                else gs.morphs.Mesh(**options, fixed=a["fixed"])
            )
            entities[n] = scene.add_entity(
                morph, material=gs.materials.Rigid(), vis_mode="collision"
            )
        scene.build()
        report["rigid_options"] = scene.rigid_solver._options.model_dump(mode="json")
        check()
        owners, links, references, masses = {}, {}, {}, {}
        for n, e in entities.items():
            for link in e.links:
                for g in link.geoms:
                    owners[g.idx], links[g.idx] = n, link.idx
            if n == "ground":
                continue
            a = assets[n]
            references[n] = native.pose(e)
            if (
                e.n_dofs != (0 if a["fixed"] else 6)
                or not e.geoms
                or bool(e.base_link.is_fixed) != a["fixed"]
            ):
                raise ValueError(f"{n}: collision or dynamic role mismatch")
            masses[n] = native.array(e.get_links_mass()).reshape(-1)
            inertia = native.array(e.get_links_inertia()).reshape(-1, 3, 3)
            if not a["fixed"] and (
                not np.isfinite(masses[n]).all()
                or masses[n].sum() <= 0
                or not np.isfinite(inertia).all()
                or np.linalg.eigvalsh(inertia).min() < -1e-10
            ):
                raise ValueError(f"{n}: invalid mass/inertia")
            apply_pose(e, a, data["poses"][n], references[n])
            collision_error = 0.0
            if len(e.geoms) != len(a["collision_hulls"]):
                raise ValueError(f"{n}: collision part count changed")
            for g, hull in zip(e.geoms, a["collision_hulls"]):
                # Compare the frozen vertices without selecting a numerically unstable hull.
                # Near-coplanar hull vertices may change under float32 rotations.
                actual = native.array(g.get_verts()).reshape(-1, 3)
                expected = geo.transform(hull, data["poses"][n])
                error = vertex_set_error(actual, expected)
                collision_error = max(collision_error, error)
            if collision_error > 1e-5:
                raise ValueError(f"{n}: collision origin/scale mismatch: {collision_error}")
            consistency = {}
            if not a["fixed"]:
                mass, center, tensor = repair_assets.aggregate_inertia(e, 1, np.zeros(3), gs)
                canonical_com = geo.inverse([center], data["poses"][n])[0]
                axes = rotation(data["poses"][n]["orientation_wxyz"])
                canonical_inertia = axes.T @ tensor @ axes
                if not np.isclose(mass, a["mass_kg"], rtol=1e-5, atol=1e-9):
                    raise ValueError(f"{n}: frozen mass mismatch")
                if "com_local_m" in a and not np.allclose(
                    canonical_com, a["com_local_m"], rtol=0, atol=1e-5
                ):
                    raise ValueError(f"{n}: frozen COM mismatch")
                if "inertia_local_kg_m2" in a and not np.allclose(
                    canonical_inertia, a["inertia_local_kg_m2"], rtol=1e-4, atol=1e-10
                ):
                    raise ValueError(f"{n}: frozen inertia mismatch")
                consistency = dict(
                    passed=True, canonical_com_m=canonical_com.tolist(),
                    canonical_inertia_kg_m2=canonical_inertia.tolist(),
                )
            loaded[n] = dict(
                mass_properties_consistency=consistency,
                dofs=e.n_dofs,
                fixed=a["fixed"],
                mass_kg=float(masses[n].sum()),
                inertia_kg_m2=inertia.tolist(),
                collision_geoms=len(e.geoms),
                collision_bounds_error_m=collision_error,
                friction=[float(native.array(g.get_friction())) for g in e.geoms],
                sol_params=[native.array(g.get_sol_params()).tolist() for g in e.geoms],
            )
        for n, a in assets.items():
            if a["fixed"] and (
                a["support"] != "ground"
                or abs(geo.transform(a["hull"], data["poses"][n])[:, 2].min()) > 0.001
            ):
                raise ValueError(f"{n}: fixed suspended object")
        report["status"] = "passed"
        clip.write_json(out / "asset_physics_report.json", report)

        def snapshot(step):
            raw = scene.rigid_solver.collider.get_contacts(to_torch=False)
            contacts = native.contacts(raw, owners, links, initial=step == 0)
            objects = {}
            for n, e in entities.items():
                state = native.pose(e)
                if n == "ground":
                    state.update(
                        velocity=[0.0, 0.0, 0.0],
                        angular_velocity=[0.0, 0.0, 0.0],
                        com_position=state["position"],
                    )
                else:
                    a, reference = assets[n], references[n]
                    delta = (
                        rotation(state["orientation_wxyz"])
                        @ rotation(reference["orientation_wxyz"]).T
                    )
                    offset = np.asarray(reference["position"]) - (
                        np.asarray(a["anchor_m"]) if a["native_collision"] else 0.0
                    )
                    position = np.asarray(state["position"]) - delta @ offset
                    ids = [link.idx for link in e.links]
                    com = native.array(
                        scene.rigid_solver.get_links_pos(ids, ref=gs.link_ref_frame.link_COM)
                    ).reshape(-1, 3)
                    vel = native.array(
                        scene.rigid_solver.get_links_vel(ids, ref=gs.link_ref_frame.link_COM)
                    ).reshape(-1, 3)
                    weights = (
                        masses[n] / masses[n].sum()
                        if masses[n].sum() > 0
                        else np.ones(len(masses[n])) / len(masses[n])
                    )
                    state = dict(
                        position=position.tolist(),
                        orientation_wxyz=geo.quat(delta),
                        com_position=(weights @ com).tolist(),
                        velocity=(weights @ vel).tolist(),
                        angular_velocity=native.array(e.get_ang()).reshape(3).tolist(),
                    )
                    summed = (
                        sum(
                            (
                                np.asarray(c["force_a" if c["a"] == n else "force_b"])
                                for c in contacts
                                if n in (c["a"], c["b"])
                            ),
                            start=np.zeros(3),
                        )
                        if step
                        else None
                    )
                    if step and not a["fixed"]:
                        observed = (
                            native.array(e.get_links_net_contact_force()).reshape(-1, 3).sum(0)
                        )
                        if not np.allclose(observed, summed, atol=1e-5, rtol=1e-4):
                            raise ValueError("contact force mapping/cache mismatch")
                    state["net_contact_force"] = None if summed is None else summed.tolist()
                objects[n] = state
            row = dict(
                step=step,
                time_s=step * cfg["dt"],
                objects=objects,
                contacts=contacts,
                contact_phase="solved_step" if step else "initial_detection",
            )
            original.validate_row(row, step, dict(bodies=assets, settings=cfg))
            return row

        scene.rigid_solver.detect_collision()
        with (out / "trace.jsonl").open("x") as stream:

            def append(row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)

            append(snapshot(0))
            clip.write_json(out / "initial_state.json", rows[0])
            initial_rejected = not all(penetration_ok(c, assets) for c in rows[0]["contacts"])
            if not initial_rejected:
                for step in range(1, cfg["steps"] + 1):
                    scene.step()
                    executed = step
                    append(snapshot(step))
        check()
        return evaluate(data, rows, initial_rejected=initial_rejected)
    except BaseException as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(
            steps_executed=executed,
            simulation_executed=executed > 0,
            last_complete_step=None if not rows else rows[-1]["step"],
        )
        clip.write_json(out / "asset_physics_report.json", report)
        if rows:
            clip.write_json(
                out / "final_state.json",
                dict(state=rows[-1], complete=rows[-1]["step"] == cfg["steps"]),
            )
        gs.destroy()


def frozen_input(assets, poses, relations, seed, *, profile=geo.PROFILE):
    return dict(
        schema_version="genenv.text_repair_input.v1",
        profile=profile,
        settings=settings(profile),
        assets=copy.deepcopy(assets),
        poses=copy.deepcopy(poses),
        relations=copy.deepcopy(relations),
        random_seed=seed,
    )

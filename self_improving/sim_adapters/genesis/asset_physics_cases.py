"""Isolated real acceptance fixtures, never substitutions for the user's scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_physics as evidence
from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis import validate_asset_scene as runtime

CASES = (
    "calibration",
    "mesh_calibration",
    "margin_slack",
    "deep_penetration",
    "fixed_suspension",
    "disabled_collision",
    "wrong_target",
    "moving",
    "three_levels",
    "microwave_probe",
)


def prepare_case(case, out, index_path):
    """Measure native fixture geometry without cameras, model calls or physics steps."""
    import genesis as gs
    import trimesh

    index, _ = runtime.clip.load_index(index_path)
    if case in ("microwave_probe", "mesh_calibration", "margin_slack"):
        # mesh_calibration is the positive control for the regime production actually runs:
        # real non-convex mesh contact, where a resting body loses its contact set for
        # single steps. Primitive-box fixtures never enter that regime, so on their own
        # they cannot show that the acceptance limits are reachable at all.
        chosen = (
            [("table", "dex_table_d3996872"), ("a", "cup_2"), ("b", "apple_15")]
            if case == "mesh_calibration"
            else [("table", "dex_table_d3996872"), ("a", "apple_15")]
            if case == "margin_slack"
            else [
                ("table", "dex_table_d3996872"),
                ("middle", "microwave_microwave_59704527"),
                ("a", "apple_15"),
            ]
        )
        bindings = {}
        for name, asset_id in chosen:
            asset = next(a for a in index["assets"] if a["asset_id"] == asset_id)
            source_root = Path(asset["source_root"])
            runtime.official.verify_files(source_root, asset["source_files"])
            bindings[name] = dict(
                asset_id=asset_id,
                source_root=str(source_root),
                model_entrypoint=str(runtime.official.safe_file(source_root, asset["entrypoint"])),
                source_files=asset["source_files"],
            )
    else:
        bindings = {}
        source_root = out / "fixture_assets"
        source_root.mkdir()
        shapes = [("table", [1.2, 0.8, 0.7]), ("a", [0.06, 0.06, 0.06])]
        if case == "three_levels":
            shapes.insert(1, ("middle", [0.3, 0.3, 0.15]))
        for name, extents in shapes:
            path = source_root / f"{name}.glb"
            trimesh.creation.box(extents=extents).export(path)
            bindings[name] = dict(
                asset_id=f"generated_test_{name}",
                source_root=str(source_root),
                model_entrypoint=str(path),
                source_files=[runtime.official.fingerprint(path, source_root)],
            )
    doc = dict(
        request=f"isolated physics acceptance fixture: {case}",
        objects=[dict(object_id=n, category=n, description=n) for n in bindings],
        relations=[
            dict(
                relation="on",
                source=n,
                target="middle"
                if n == "a" and case in ("three_levels", "microwave_probe")
                else "table",
                evidence="explicit test fixture",
            )
            for n in bindings
            if n != "table"
        ],
    )
    geometry = {}
    for binding in bindings.values():
        if Path(binding["model_entrypoint"]).suffix not in (".xml", ".glb"):
            runtime.write_json(
                out / "unsupported_asset.json",
                dict(
                    binding,
                    reason=(
                        "native URDF microwave has a revolute door; joint tasks are out of scope"
                    ),
                ),
            )
            raise ValueError("native URDF microwave with revolute door is unsupported")
    gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="warning")
    try:
        scene = gs.Scene(show_viewer=False)
        entities = {}
        for name, binding in bindings.items():
            options = dict(
                file=binding["model_entrypoint"],
                scale=1.0,
                convexify=False,
                decimate=False,
                watertighten=None,
                collision=False,
            )
            morph = (
                gs.morphs.MJCF(**options)
                if Path(options["file"]).suffix == ".xml"
                else gs.morphs.Mesh(**options, fixed=True)
            )
            entities[name] = scene.add_entity(
                morph, material=gs.materials.Rigid(), vis_mode="visual"
            )
        scene.build()
        parents = spatial.relations(doc)
        for name, entity in entities.items():
            vertices, faces = builder.entity_mesh(entity)
            geometry[name] = dict(
                bounds=runtime.official.bounds(vertices).tolist(),
                mesh_sha256=hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
            )
            if name in parents.values():
                geometry[name]["surface"] = builder.support_surface(vertices, faces)
    finally:
        gs.destroy()
    graph = spatial.graph_for(
        doc, bindings, dict(object_ids=list(bindings), relations=doc["relations"], preferences=[])
    )
    layout, validation = spatial.solve(doc, bindings, geometry, graph)
    bodies = {}
    for placed in layout["objects"]:
        name = placed["object_id"]
        bodies[name] = dict(
            bindings[name],
            translation_m=placed["translation_m"],
            world_visual_bounds_m=placed["world_visual_bounds_m"],
            native_geometry=geometry[name],
            support=placed["support"],
            fixed=name == "table",
            surface=geometry[name].get("surface"),
            material_defaults=dict(density_kg_m3=600.0, friction=1.0),
            morph_options=dict(
                scale=1.0, convexify=False, decimate=False, watertighten=None, collision=True
            ),
        )
    # Mutations describe separate counterexamples and are frozen before running.
    if case in ("deep_penetration", "wrong_target", "fixed_suspension"):
        delta = {
            "deep_penetration": [0, 0, -0.02],
            "wrong_target": [2, 0, 0],
            "fixed_suspension": [0, 0, 0.2],
        }[case]
        body = bodies["a"]
        body["translation_m"] = (np.array(body["translation_m"]) + delta).tolist()
        body["world_visual_bounds_m"] = (np.array(body["world_visual_bounds_m"]) + delta).tolist()
    if case == "margin_slack":
        # Put the body exactly on the planner's line, then require that after real
        # settling drift it is still inside the acceptance margin. This is what the slack
        # between PLANNING_MARGIN and MARGIN has to buy; without it the planner emits
        # layouts the validator is guaranteed to reject.
        body = bodies["a"]
        polygon = (
            np.asarray(bodies["table"]["surface"]["polygon_xy_m"])
            + np.asarray(bodies["table"]["translation_m"])[:2]
        )
        box = np.asarray(body["world_visual_bounds_m"], float)
        delta = [float(polygon[:, 0].min() + spatial.PLANNING_MARGIN - box[0, 0]), 0.0, 0.0]
        body["translation_m"] = (np.array(body["translation_m"]) + delta).tolist()
        body["world_visual_bounds_m"] = (np.array(body["world_visual_bounds_m"]) + delta).tolist()
    if case == "fixed_suspension":
        bodies["a"]["fixed"] = True
    if case == "disabled_collision":
        bodies["a"]["morph_options"]["collision"] = False
    settings = evidence.settings("baseline")
    if case == "moving":
        # A declared horizontal gravity component keeps the free test body moving.
        # This is an attack-only input, never a production CLI profile override.
        settings["gravity"] = [10.0, 0.0, -9.81]
    data = dict(
        schema_version=evidence.SCHEMA,
        genesis_commit=runtime.official.GENESIS_COMMIT,
        settings=settings,
        bodies=bodies,
        relations=doc["relations"],
        fixture=case,
        source="explicit isolated fixture, not an original-scene acceptance",
        request=doc["request"],
    )
    for name, value in [
        ("scene_graph.json", graph),
        ("scene_layout.json", layout),
        ("native_geometry.json", geometry),
        ("layout_validation_report.json", validation),
        ("physics_input.json", data),
    ]:
        runtime.write_json(out / name, value)
    return data


def run_case(case, output, index_path):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        data = prepare_case(case, out, index_path)
    except (Exception, KeyboardInterrupt) as exc:
        report = dict(
            status="physics_failed",
            render_status="not_run",
            checks=[],
            case=case,
            phase="preparation",
            simulation_executed=False,
            steps_executed=0,
            error=f"{type(exc).__name__}: {exc}",
            files=[
                runtime.official.fingerprint(p, out) for p in sorted(out.rglob("*")) if p.is_file()
            ],
        )
        runtime.write_json(out / "physics_result.json", report)
        return report
    input_hash = runtime.library.sha256(out / "physics_input.json")

    def check():
        if runtime.library.sha256(out / "physics_input.json") != input_hash:
            raise ValueError("fixture input changed")
        for body in data["bodies"].values():
            runtime.official.verify_files(Path(body["source_root"]), body["source_files"])

    progress = dict(
        simulation_executed=False, steps_executed=0, phase="loading", last_complete_step=None
    )
    report = dict(
        status="physics_failed",
        render_status="not_run",
        checks=[],
        input_sha256=input_hash,
        case=case,
        cameras_created=0,
        render_calls=0,
    )
    try:
        # Same declaration rule as production prepare(); no frozen on-source can pass.
        if any(b["fixed"] and b["support"] != "ground" for b in data["bodies"].values()):
            raise ValueError("fixed suspension rejected before simulation")
        loaded, rows = runtime.simulate(data, out, check, progress)
        report["checks"] = evidence.evaluate(data, loaded, rows)
        report["status"] = (
            "physics_passed" if all(c["passed"] for c in report["checks"]) else "physics_failed"
        )
        if case == "calibration":
            # Independent observation API verifies sign and same-step cache freshness.
            # Ground support fraction and a stable 4-second trace are also required.
            report["contact_sample_counts"] = [len(row["contacts"]) for row in rows]
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        initial = out / "initial_state.json"
        if initial.exists():
            report["checks"] = runtime.library.read_json(initial)["checks"]
    finally:
        check()
        report.update(progress)
        terminal = out / "final_state.json"
        if terminal.exists():
            value = runtime.library.read_json(terminal)
            value["passed"] = report["status"] == "physics_passed"
            runtime.write_json(terminal, value)
        report["files"] = [
            runtime.official.fingerprint(p, out) for p in sorted(out.rglob("*")) if p.is_file()
        ]
        runtime.write_json(out / "physics_result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-index", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["GS_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "1"
    report = run_case(args.case, args.output, args.clip_index)
    print(json.dumps({k: report.get(k) for k in ("case", "status", "steps_executed", "error")}))
    return 0 if report["status"] == "physics_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

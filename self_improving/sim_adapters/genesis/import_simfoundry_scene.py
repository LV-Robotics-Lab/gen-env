"""Import reconstructed rigid scenes into the Genesis v2 graph/layout contract.

This preserves authored placement, never invokes the text layout solver, and never
infers physical support from bounds. Simulator imports are confined to preview().
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis.physics_math import rotation

VERSION = "genenv.simfoundry_scene_import.v1"


def vector(value, size):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != size
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise ValueError("invalid numeric vector")
    result = np.asarray(value, float)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite numeric vector")
    return result


def quaternion(value):
    q = vector(value, 4)[[3, 0, 1, 2]]
    rotation(q)  # Reject invalid quaternions; tolerate only serialization roundoff.
    return (q / np.linalg.norm(q)).tolist()


def environment(source):
    args = source.get("init_info", {}).get("args", {})
    block = source.get("ground_plane_info", {})
    enabled = args.get("use_floor_plane", True)
    visible = block.get("visible", args.get("floor_plane_visible", True))
    if visible is None:
        visible = args.get("floor_plane_visible", True)
    if not isinstance(enabled, bool) or not isinstance(visible, bool):
        raise ValueError("invalid ground plane flags")
    pos = vector(block.get("position", [0, 0, 0]), 3).tolist()
    return dict(
        ground="genesis_builtin_plane" if enabled else None,
        position_m=pos,
        z_m=pos[2],
        visible=visible,
        orientation_wxyz=quaternion(block.get("orientation", [0, 0, 0, 1])),
    )


def read_objects(source, metadata, *, pose_format, exclude):
    by_name = {v["name"]: v for v in metadata.values()}
    if not by_name or len(by_name) != len(metadata):
        raise ValueError("empty or duplicate source object IDs")
    if pose_format == "og":
        initial = source["objects_info"]["init_info"]
        states = source["state"]["registry"]["object_registry"]
        if set(initial) != set(states):
            raise ValueError("scene object declarations and states differ")
    else:
        initial = {n: dict(args=by_name.get(n, {})) for n in source}
        states = {n: dict(root_link=dict(pos=p[0], ori=p[1])) for n, p in source.items()}
    if set(exclude) - set(initial):
        raise ValueError("unknown excluded object")
    objects, excluded = [], []
    for name, init in initial.items():
        robot = init.get("class_module", "").startswith("omnigibson.robots.")
        if name in exclude or robot:
            excluded.append(
                dict(
                    object_id=name,
                    reason="explicit exclusion"
                    if name in exclude
                    else "robot has no converted rigid asset",
                )
            )
            continue
        if name not in by_name:
            raise ValueError(f"{name}: no source metadata; convert asset or explicitly exclude")
        args, meta = init["args"], by_name[name]
        if any(args.get(k) != meta[k] for k in ("name", "category", "model")):
            raise ValueError(f"{name}: saved scene differs from source asset identity")
        if args.get("usd_path"):
            raise ValueError(f"{name}: custom USD path cannot be bound to the converted URDF")
        if not np.array_equal(vector(args.get("scale", [1, 1, 1]), 3), [1, 1, 1]):
            raise ValueError(f"{name}: resized scene asset requires a separate asset conversion")
        if any(k in args for k in ("mass", "link_physics_materials", "joint_limits")):
            raise ValueError(f"{name}: scene physics overrides require a separate asset conversion")
        state = states[name]
        if len(state.get("joint_pos", [])) or len(state.get("joint_vel", [])):
            raise ValueError(f"{name}: articulated scene state is unsupported")
        if args.get("visual_only", False):
            raise ValueError(f"{name}: visual-only scene object is not a rigid asset")
        fixed = args.get("fixed_base", False)
        if not isinstance(fixed, bool):
            raise ValueError("invalid fixed_base flag")
        root = state["root_link"]
        objects.append(
            dict(
                object_id=name,
                category=meta["category"],
                translation_m=vector(root["pos"], 3).tolist(),
                orientation_wxyz=quaternion(root["ori"]),
                scale=1.0,
                orientation_policy="preserve_source_root_link_pose",
                intended_dynamic=not fixed,
                fixed=fixed,
                support=None,
                source_velocity_mps=vector(root.get("lin_vel", [0, 0, 0]), 3).tolist(),
                source_angular_velocity_radps=vector(root.get("ang_vel", [0, 0, 0]), 3).tolist(),
            )
        )
    missing = set(by_name) - set(initial) - set(exclude)
    if missing:
        raise ValueError(f"missing source objects in scene: {sorted(missing)}")
    if not objects:
        raise ValueError("no convertible objects")
    return objects, excluded


def convert(scene_dir, library_path, output_dir, *, scene_file=None, pose_format="og", exclude=()):
    scene, library_path, out = map(
        lambda p: Path(p).resolve(), (scene_dir, library_path, output_dir)
    )
    if pose_format not in {"og", "pybullet"}:
        raise ValueError("unsupported pose format")
    source_path = (
        Path(scene_file).resolve()
        if scene_file
        else scene
        / (
            "s14_og/reconstructed_og_scene.json"
            if pose_format == "og"
            else "s12_physics/pb_scene_poses.json"
        )
    )
    for root in (scene, library_path.parent, source_path.parent):
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError("source and output must be separate")
    if out.exists():
        raise FileExistsError(out)
    meta_path = scene / "s11_sim/scene_objects_info.json"
    inputs = {
        k: dict(path=str(p), sha256=lib.sha256(p))
        for k, p in [("scene", source_path), ("metadata", meta_path), ("library", library_path)]
    }

    def check_inputs():
        for ref in inputs.values():
            if lib.sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("source changed during conversion")

    source, metadata = lib.read_json(source_path), lib.read_json(meta_path)
    inventory = lib.read_json(library_path)
    if (
        inventory["schema_version"] != standard.SCHEMA
        or inventory["source_metadata"]["sha256"] != inputs["metadata"]["sha256"]
    ):
        raise ValueError("asset library source metadata mismatch")
    rows = inventory["assets"]
    assets = {a["object_id"]: a for a in rows}
    if len(assets) != len(rows):
        raise ValueError("duplicate library object IDs")
    objects, excluded = read_objects(source, metadata, pose_format=pose_format, exclude=exclude)
    env = environment(source if pose_format == "og" else {})
    prepared, geometry, source_assets = [], {}, []
    for obj in objects:
        name = obj["object_id"]
        item = assets.get(name, {})
        if item.get("status") != "imported":
            raise ValueError(f"{name}: missing successful converted asset")
        package = official.safe_file(library_path.parent, item["package"])
        if lib.sha256(package) != item["package_sha256"]:
            raise ValueError(f"{name}: package manifest changed")
        pkg, entry, physics = standard.verify_package(package)
        if (
            pkg["source"]["object_id"] != name
            or pkg["category"] != obj["category"]
            or pkg["asset_id"] != item["asset_id"]
            or pkg["source"]["metadata"]["sha256"] != inputs["metadata"]["sha256"]
        ):
            raise ValueError(f"{name}: converted asset identity mismatch")
        meta = next(v for v in metadata.values() if v["name"] == name)
        source_root = (scene / "s11_sim/objects" / meta["category"] / meta["model"]).resolve()
        if not source_root.is_relative_to(scene / "s11_sim/objects"):
            raise ValueError("unsafe source asset path")
        source_records = [m["source"] for m in pkg["path_mapping"]]
        official.verify_files(source_root, source_records)
        source_assets.append((source_root, source_records))
        # Geometry and inertial coordinates were retained by the asset converter.
        measured = standard.inspect(entry)
        r, p = rotation(obj["orientation_wxyz"]), np.asarray(obj["translation_m"])
        local = official.bounds(measured["visual"])
        world = official.bounds(measured["visual"] @ r.T + p)
        relative = Path("assets") / name
        obj.update(
            asset_id=pkg["asset_id"],
            model_format="urdf",
            source_root=relative.as_posix(),
            source_files=pkg["files"],
            model_entrypoint=(relative / pkg["entrypoint"]).as_posix(),
            standard_package=(relative / "asset.json").as_posix(),
            standard_package_sha256=item["package_sha256"],
            local_visual_bounds_m=local.tolist(),
            world_visual_bounds_m=world.tolist(),
            mass_kg=measured["mass"],
            friction=physics["friction"],
        )
        # Check IDs as paths before any output is created.
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe source object ID")
        geometry[name] = dict(
            bounds=local.tolist(),
            world_bounds=world.tolist(),
            visual_vertex_count=len(measured["visual"]),
            collision_vertex_count=len(measured["collision"]),
        )
        prepared.append((package, pkg, relative))
    graph = dict(
        schema_version=spatial.SCHEMA,
        nodes=[{k: o[k] for k in ("object_id", "category", "asset_id")} for o in objects],
        edges=[],
        preferences=[],
        preference_source="none",
        relations_status="not_provided_by_source",
        environment=env,
        frame=dict(up="+Z", right="+X", front="-Y", units="m"),
        source_frame_policy="preserve_world_axes_without_reorientation",
    )
    layout = dict(
        schema_version=spatial.SCHEMA,
        scene_graph_sha256=clip.digest(graph),
        native_geometry_sha256=clip.digest(geometry),
        solver_version=VERSION,
        layout_method="import_source_poses",
        objects=objects,
        relations=[],
        support_surfaces={},
        environment=env,
        physics_steps=0,
        physics_status="not_run",
        meaning="source placement only; support, contact and stability unverified",
        path_base="scene_package",
        velocity_policy="recorded_only_in_zero_step_preview",
    )
    check_inputs()
    out.mkdir(parents=True)
    try:
        for package, pkg, relative in prepared:
            dest = out / relative
            dest.mkdir(parents=True)
            for record in pkg["files"]:
                src = official.safe_file(package.parent, record["path"])
                target = dest / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
            shutil.copyfile(package, dest / "asset.json")
            standard.verify_package(dest / "asset.json")
        for filename, doc in [
            ("scene_graph.json", graph),
            ("scene_layout.json", layout),
            ("native_geometry.json", geometry),
            ("source_scene.json", source),
            ("source_metadata.json", metadata),
        ]:
            official.write_json(out / filename, doc)
        check_inputs()
        for source_root, records in source_assets:
            official.verify_files(source_root, records)
        report = dict(
            schema_version=VERSION,
            status="converted",
            pose_format=pose_format,
            inputs=inputs,
            imported_objects=len(objects),
            excluded_objects=excluded,
            scope="converted rigid objects and ground plane",
            source_extras_status="cameras, lighting, backgrounds retained in source snapshot",
            physics_steps=0,
            physics_status="not_run",
        )
        official.write_json(out / "conversion_report.json", report)
        official.write_json(
            out / "manifest.json",
            dict(
                schema_version=VERSION,
                files=[official.fingerprint(p, out) for p in sorted(out.rglob("*")) if p.is_file()],
            ),
        )
        verify(out)
        return report
    except BaseException as exc:
        official.write_json(out / "conversion_error.json", dict(status="error", error=str(exc)))
        raise


def verify(directory):
    root = Path(directory).resolve()
    manifest = lib.read_json(root / "manifest.json")
    if manifest["schema_version"] != VERSION:
        raise ValueError("unsupported imported scene manifest")
    official.verify_files(root, manifest["files"])
    bound = {r["path"] for r in manifest["files"]}
    if not {"scene_graph.json", "scene_layout.json", "native_geometry.json"} <= bound:
        raise ValueError("missing scene manifest bindings")
    graph, layout, geometry = [
        lib.read_json(root / f"{n}.json")
        for n in ("scene_graph", "scene_layout", "native_geometry")
    ]
    if (
        layout["schema_version"] != spatial.SCHEMA
        or graph["schema_version"] != spatial.SCHEMA
        or layout["scene_graph_sha256"] != clip.digest(graph)
        or layout["native_geometry_sha256"] != clip.digest(geometry)
    ):
        raise ValueError("scene graph or geometry binding mismatch")
    if [n["object_id"] for n in graph["nodes"]] != [o["object_id"] for o in layout["objects"]]:
        raise ValueError("scene graph object set mismatch")
    for obj in layout["objects"]:
        package = official.safe_file(root, obj["standard_package"])
        if obj["standard_package"] not in bound:
            raise ValueError("unbound scene asset")
        if lib.sha256(package) != obj["standard_package_sha256"]:
            raise ValueError("scene package binding mismatch")
        pkg, entry, _ = standard.verify_package(package)
        if (
            entry != official.safe_file(root, obj["model_entrypoint"])
            or pkg["asset_id"] != obj["asset_id"]
            or pkg["files"] != obj["source_files"]
        ):
            raise ValueError("scene asset binding mismatch")
    return layout


def preview(directory, output_dir):
    """Load dynamic URDFs at the imported world poses, audit, render with zero steps."""
    from PIL import Image

    root, out = Path(directory).resolve(), Path(output_dir).resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("preview and scene package must be separate")
    layout = verify(root)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["GS_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "1"
    import genesis as gs

    commit = subprocess.check_output(
        ["git", "-C", str(Path(gs.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != official.GENESIS_COMMIT:
        raise ValueError("Genesis revision mismatch")
    initialized = False
    original_step = gs.Scene.step

    def no_step(*args, **kwargs):
        raise RuntimeError("physics step forbidden during imported scene preview")

    report = dict(
        status="error",
        genesis_commit=commit,
        physics_steps=0,
        physics_status="not_run",
        scene_manifest_sha256=lib.sha256(root / "manifest.json"),
        geometry=[],
        views=[],
    )
    gs.Scene.step = no_step
    try:
        gs.init(backend=gs.cpu, seed=0, logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
            renderer=gs.renderers.Rasterizer(),
            vis_options=gs.options.VisOptions(
                background_color=(1, 1, 1),
                ambient_light=(0.4, 0.4, 0.4),
                shadow=False,
                lights=official.LIGHTS,
            ),
        )
        env = layout["environment"]
        if env["ground"]:
            scene.add_entity(
                gs.morphs.Plane(
                    pos=env["position_m"],
                    quat=env["orientation_wxyz"],
                    visualization=env["visible"],
                )
            )
        entities = {}
        for obj in layout["objects"]:
            _, entry, physics = standard.verify_package(root / obj["standard_package"])
            morph = standard.morph(gs, entry, fixed=obj["fixed"])
            morph.pos = tuple(obj["translation_m"])
            morph.quat = tuple(obj["orientation_wxyz"])
            entities[obj["object_id"]] = scene.add_entity(
                morph,
                name=obj["object_id"],
                material=gs.materials.Rigid(friction=physics["friction"]),
                vis_mode="visual",
            )
        camera = scene.add_camera(
            res=(960, 720), GUI=False, pos=(1, -1, 1), lookat=(0, 0, 0), fov=35
        )
        scene.build()
        for obj in layout["objects"]:
            entity = entities[obj["object_id"]]
            measured = standard.inspect(root / obj["model_entrypoint"])
            audit = standard.audit(
                entity, measured, collision=not obj["fixed"], friction=obj["friction"]
            )
            actual = np.concatenate(
                [
                    standard.array(g.get_vverts()).reshape(-1, 3)
                    for link in entity.links
                    for g in link.vgeoms
                ]
            )
            expected = measured["visual"] @ rotation(obj["orientation_wxyz"]).T
            expected += obj["translation_m"]
            error = standard.distance(actual, expected)
            if error > 1e-5:
                raise ValueError("imported world pose differs from source placement")
            report["geometry"].append(
                dict(object_id=obj["object_id"], world_vertex_error_m=error, **audit)
            )
        boxes = np.array([o["world_visual_bounds_m"] for o in layout["objects"]])
        low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        center = (low + high) / 2
        distance = float(np.linalg.norm(high - low) / 2 / np.sin(np.deg2rad(35 / 2)) * 1.15)
        for name, direction, up in [
            ("overview", [0, -1, 0.7], [0, 0, 1]),
            ("top", [0, 0, 1], [0, 1, 0]),
            ("side", [1, -1, 0.7], [0, 0, 1]),
        ]:
            direction = np.asarray(direction, float)
            camera.set_pose(
                pos=(center + distance * direction / np.linalg.norm(direction)).tolist(),
                lookat=center.tolist(),
                up=up,
            )
            rgb, _, _, _ = camera.render(rgb=True)
            path = out / f"{name}.png"
            Image.fromarray(np.asarray(rgb, np.uint8)).save(path)
            report["views"].append(official.fingerprint(path, out))
        verify(root)
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        official.write_json(out / "preview_report.json", report)
        gs.Scene.step = original_step
        if initialized:
            gs.destroy()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--scene-dir", required=True)
    imp.add_argument("--library-path", required=True)
    imp.add_argument("--output-dir", required=True)
    imp.add_argument("--scene-file")
    imp.add_argument("--pose-format", choices=["og", "pybullet"], default="og")
    imp.add_argument("--exclude-object", action="append", default=[])
    for cmd in ("verify", "preview"):
        p = sub.add_parser(cmd)
        p.add_argument("--scene-package", required=True)
        if cmd == "preview":
            p.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            result = convert(
                args.scene_dir,
                args.library_path,
                args.output_dir,
                scene_file=args.scene_file,
                pose_format=args.pose_format,
                exclude=args.exclude_object,
            )
        elif args.command == "preview":
            result = preview(args.scene_package, args.output_dir)
        else:
            layout = verify(args.scene_package)
            result = dict(
                status="verified", objects=len(layout["objects"]), physics_status="not_run"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="error", error=f"{type(exc).__name__}: {exc}")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

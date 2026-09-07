"""Native single-asset drop validation; no settling, repair, retrieval or VLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_physics
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis.physics_math import angle
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.validate_asset_scene import contacts

array = standard.array


def evaluate(rows, cfg):
    if len(rows) != cfg["steps"] + 1:
        raise ValueError("incomplete trajectory")
    for i, row in enumerate(rows):
        if row["step"] != i or abs(row["time_s"] - i * cfg["dt"]) > 1e-9:
            raise ValueError("nonsequential trajectory")
        for k in ("position", "orientation_wxyz", "velocity", "angular_velocity"):
            if not np.isfinite(row[k]).all():
                raise ValueError("nonfinite state")
        angle(row["orientation_wxyz"], row["orientation_wxyz"])
    window = rows[-round(cfg["window_s"] / cfg["dt"]) :]
    positions = np.array([r["position"] for r in window])
    measures = dict(
        translation_m=float(np.linalg.norm(positions - positions[0], axis=1).max()),
        rotation_deg=max(
            angle(r["orientation_wxyz"], window[0]["orientation_wxyz"]) for r in window
        ),
        speed_mps=max(float(np.linalg.norm(r["velocity"])) for r in window),
        angular_speed_radps=max(float(np.linalg.norm(r["angular_velocity"])) for r in window),
        penetration_m=max([c["penetration"] for r in rows for c in r["contacts"]] + [0]),
        support_fraction=sum(r["ground_up_force_n"] > cfg["support_force_n"] for r in window)
        / len(window),
    )
    checks = [
        dict(
            name=k,
            observed=v,
            limit=cfg[k],
            passed=bool(v >= cfg[k] if k == "support_fraction" else v <= cfg[k]),
        )
        for k, v in measures.items()
    ]
    return checks


def resolve(package=None, binding=None):
    if bool(package) == bool(binding):
        raise ValueError("specify exactly one package or selected binding")
    if package:
        package = Path(package).resolve()
        data, source, physics = standard.verify_package(package)
        binding = dict(
            source_root=str(package.parent),
            model_entrypoint=str(source),
            source_files=data["files"],
            standard_package=str(package),
            standard_package_sha256=standard.library.sha256(package),
        )
    else:
        binding = json.loads(Path(binding).read_text())
        source = local_path(binding["model_entrypoint"]).resolve()
        if binding.get("standard_package"):
            package = local_path(binding["standard_package"])
            if standard.library.sha256(package) != binding["standard_package_sha256"]:
                raise ValueError("package fingerprint changed")
            data, entry, physics = standard.verify_package(package)
            if source != entry or data["files"] != binding["source_files"]:
                raise ValueError("selected package binding changed")
        else:
            physics = dict(friction=None)
    root = local_path(
        binding.get(
            "source_root",
            Path(binding["official_index"]["path"]).parent
            if "official_index" in binding
            else source.parent,
        )
    ).resolve()

    def verify():
        official.verify_files(root, binding["source_files"])
        if not source.is_relative_to(root) or str(source.relative_to(root)) not in {
            f["path"] for f in binding["source_files"]
        }:
            raise ValueError("unbound selected entrypoint")
        if binding.get("standard_package"):
            if (
                standard.library.sha256(binding["standard_package"])
                != binding["standard_package_sha256"]
            ):
                raise ValueError("package fingerprint changed")
            standard.verify_package(binding["standard_package"])

    verify()
    return source, binding, physics, verify


def run(output_dir, *, package=None, binding=None):
    out = Path(output_dir).resolve()
    from self_improving.sim_adapters.genesis.clip_select import writable_storage

    writable_storage(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(
        status="error",
        physics_status="not_evaluated",
        exit_code=1,
        steps_executed=0,
        simulation_executed=False,
        model_calls=0,
        scope="single rigid asset drop only; no scene support/containment acceptance",
    )
    initialized = False
    video = None
    sampled_hashes = []
    rows = []
    cfg = asset_physics.settings("baseline")
    verify = None
    try:
        source, bound, physics, verify = resolve(package, binding)
        if out.is_relative_to(local_path(bound["source_root"])):
            raise ValueError("output inside source assets")
        suffix = source.suffix.lower()
        if suffix not in (".urdf", ".xml", ".glb", ".gltf"):
            raise ValueError(f"unsupported_physics_format:{suffix}")
        expected = standard.inspect(source) if suffix == ".urdf" else None
        frozen = dict(
            binding=bound,
            settings=cfg,
            clearance_m=0.01,
            friction=physics,
            genesis_commit=official.GENESIS_COMMIT,
        )
        official.write_json(out / "physics_input.json", frozen)
        result["physics_input_sha256"] = standard.library.sha256(out / "physics_input.json")
        os.environ["GS_HEADLESS"] = "1"
        os.environ["PYGLET_HEADLESS"] = "1"
        import genesis as gs

        actual_commit = subprocess.check_output(
            ["git", "-C", str(Path(gs.__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual_commit != official.GENESIS_COMMIT:
            raise ValueError("Genesis revision mismatch")
        gs.init(backend=gs.cpu, seed=0, precision="32", logging_level=logging.WARNING)
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
                background_color=(1, 1, 1), ambient_light=(0.4, 0.4, 0.4), shadow=False
            ),
        )
        ground = scene.add_entity(gs.morphs.Plane(), name="ground")
        friction = physics.get("friction")
        if suffix == ".urdf":
            morph = standard.morph(gs, source)
            material = gs.materials.Rigid(
                **({"friction": friction} if friction is not None else {})
            )
        else:
            opts = dict(
                file=str(source),
                convexify=False,
                decimate=False,
                watertighten=None,
                recompute_inertia=False,
                align=False,
            )
            morph = gs.morphs.MJCF(**opts) if suffix == ".xml" else gs.morphs.Mesh(**opts)
            material = (
                gs.materials.Rigid()
                if suffix == ".xml"
                else gs.materials.Rigid(rho=600, friction=1)
            )
        entity = scene.add_entity(morph, material=material, name="asset", vis_mode="visual")
        camera = scene.add_camera(
            res=(512, 512), GUI=False, pos=(1, -1, 1), lookat=(0, 0, 0), fov=35
        )
        scene.build()
        verify()
        if entity.n_dofs != 6 or entity.base_link.is_fixed or not entity.geoms:
            raise ValueError(
                f"expected dynamic single rigid body: dofs={entity.n_dofs}, "
                f"fixed={entity.base_link.is_fixed}, collisions={len(entity.geoms)}"
            )
        if suffix == ".xml" and any(j.n_dofs for j in entity.joints if j.n_dofs != 6):
            raise ValueError("articulated MJCF unsupported")
        mass = float(array(entity.get_mass()).reshape(-1)[0])
        inertias = array(entity.get_links_inertia()).reshape(-1, 3, 3)
        if not np.isfinite(mass) or mass <= 0 or not np.isfinite(inertias).all():
            raise ValueError("invalid loaded mass or inertia")
        if not any(np.linalg.eigvalsh(i).min() > 0 for i in inertias):
            raise ValueError("no positive definite body inertia")
        loaded = (
            standard.audit(entity, expected, friction=friction)
            if expected
            else dict(
                mass_kg=mass,
                inertia=inertias.tolist(),
                dofs=entity.n_dofs,
                friction=[float(array(g.get_friction())) for g in entity.geoms],
                geometry_policy="native MJCF/GLB, no mesh repair or decomposition",
            )
        )
        loaded["sim_options"] = scene.options.sim.model_dump(mode="json")
        loaded["rigid_options"] = scene.options.rigid.model_dump(mode="json")
        loaded["effective_contact_parameters"] = [
            array(g.get_sol_params()).tolist() for g in entity.geoms
        ]
        official.write_json(out / "loaded_asset.json", loaded)
        points = array(entity.get_verts()).reshape(-1, 3)
        pos = array(entity.get_pos()).reshape(3)
        translation = np.array([0, 0, 0.01 - points[:, 2].min()])
        entity.set_pos(pos + translation)
        result["test_translation_m"] = translation.tolist()
        visual = np.concatenate(
            [array(g.get_vverts()).reshape(-1, 3) for link in entity.links for g in link.vgeoms]
        )
        box = np.array(official.bounds(visual))
        box[0, 2] = min(box[0, 2], 0)
        view = official.camera_views(box)[0]
        camera.set_pose(pos=view["pos"], lookat=view["lookat"], up=view["up"])
        nc = camera._rasterizer._camera_nodes[camera.uid].camera
        nc.znear, nc.zfar = view["near"], view["far"]
        owners, links = {}, {}
        for name, e in [("asset", entity), ("ground", ground)]:
            for link in e.links:
                for geom in link.geoms:
                    owners[geom.idx], links[geom.idx] = name, link.idx
        import imageio.v2 as imageio
        from PIL import Image

        (out / "video").mkdir()
        video = imageio.get_writer(out / "video/drop.mp4", fps=25, codec="libx264")
        scene.rigid_solver.detect_collision()
        with (out / "trace.jsonl").open("x") as stream:
            for step in range(cfg["steps"] + 1):
                if step:
                    scene.step()
                    result.update(steps_executed=step, simulation_executed=True)
                raw = scene.rigid_solver.collider.get_contacts(to_torch=False)
                cs = contacts(raw, owners, links, initial=step == 0)
                force = np.zeros(3)
                if step:
                    for c in cs:
                        force += c["force_a"] if c["a"] == "asset" else c["force_b"]
                    observed = array(entity.get_links_net_contact_force()).reshape(-1, 3).sum(0)
                    if not np.allclose(force, observed, atol=1e-5, rtol=1e-4):
                        raise ValueError("contact force observation mismatch")
                row = dict(
                    step=step,
                    time_s=step * cfg["dt"],
                    position=array(entity.get_pos()).reshape(3).tolist(),
                    orientation_wxyz=array(entity.get_quat()).reshape(4).tolist(),
                    velocity=array(entity.get_vel()).reshape(3).tolist(),
                    angular_velocity=array(entity.get_ang()).reshape(3).tolist(),
                    contacts=cs,
                    ground_up_force_n=None if step == 0 else float(force[2]),
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if step % 10 == 0:
                    rgb = np.asarray(camera.render(rgb=True)[0], dtype=np.uint8)
                    video.append_data(rgb)
                    sampled_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    if step == 0:
                        Image.fromarray(rgb).save(out / "initial.png")
        durable = [json.loads(s) for s in (out / "trace.jsonl").read_text().splitlines()]
        if rows != durable:
            raise ValueError("durable trajectory mismatch")
        checks = evaluate(durable, cfg)
        passed = all(c["passed"] for c in checks)
        result.update(
            status="physics_passed" if passed else "physics_failed",
            physics_status="passed" if passed else "failed",
            exit_code=0 if passed else 2,
            checks=checks,
        )
        dest = out / ("final_render" if passed else "diagnostics")
        dest.mkdir()
        Image.fromarray(rgb).save(dest / "final.png")
    except Exception as exc:
        result.update(status="error", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        if video is not None:
            video.close()
        if initialized:
            gs.destroy()
        if verify is not None:
            try:
                verify()
                if result.get("physics_input_sha256") != standard.library.sha256(
                    out / "physics_input.json"
                ):
                    raise ValueError("physics input changed")
            except Exception as exc:
                result.update(status="error", exit_code=1, physics_status="invalid", error=str(exc))
        if rows:
            official.write_json(out / "initial_state.json", rows[0])
            official.write_json(out / "final_state.json", rows[-1])
        video_path = out / "video/drop.mp4"
        if video_path.exists():
            import imageio.v2 as imageio

            hashes = [
                hashlib.sha256(f.tobytes()).hexdigest() for f in imageio.get_reader(video_path)
            ]
            result["video"] = dict(
                total_frames=len(hashes),
                unique_frames=len(set(hashes)),
                fps=25,
                sampled_every_steps=10,
                encoded_input_unique=len(set(sampled_hashes)),
            )
        result.update(
            total_s=time.perf_counter() - started,
            artifacts=[
                official.fingerprint(p, out)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "physics_result.json"
            ],
        )
        official.write_json(out / "physics_result.json", result)
    return result


def verify_evidence(directory, binding):
    """Verify a selected asset's durable successful test before downstream use."""
    directory = Path(directory)
    evidence = binding.get("physics_evidence")
    if not evidence or binding.get("physics_status") != "passed":
        raise ValueError("selected asset has no passing physics evidence")
    official.verify_files(directory, [evidence])
    path = official.safe_file(directory, evidence["path"])
    report = json.loads(path.read_text())
    if (
        report["exit_code"] != 0
        or report["physics_status"] != "passed"
        or report["steps_executed"] != 1000
        or report["simulation_executed"] is not True
    ):
        raise ValueError("selected asset physics did not complete successfully")
    official.verify_files(path.parent, report["artifacts"])
    input_path = path.parent / "physics_input.json"
    if standard.library.sha256(input_path) != report["physics_input_sha256"]:
        raise ValueError("physics input fingerprint changed")
    frozen = json.loads(input_path.read_text())
    for key in ("model_entrypoint", "source_files", "source_root"):
        if frozen["binding"][key] != binding[key]:
            raise ValueError("physics evidence belongs to a different asset")
    if frozen["settings"] != asset_physics.settings("baseline"):
        raise ValueError("unexpected single-asset physics thresholds")
    rows = [json.loads(line) for line in (path.parent / "trace.jsonl").read_text().splitlines()]
    checks = evaluate(rows, frozen["settings"])
    if checks != report["checks"] or not all(c["passed"] for c in checks):
        raise ValueError("trajectory does not establish physics success")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--package", type=Path)
    inputs.add_argument("--binding", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = vars(parser.parse_args())
    report = run(**args)
    print(json.dumps(report, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

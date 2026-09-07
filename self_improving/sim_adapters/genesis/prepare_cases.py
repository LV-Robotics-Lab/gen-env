"""Prepare three small Genesis official-asset cases; never uses a RoboTwin catalog."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    from . import validate_physics as physics
except ImportError:
    import validate_physics as physics

from agenticsim.openxsim.backends import GenesisCompiler
from agenticsim.openxsim.env_gen import _dependency_metadata
from agenticsim.openxsim.ir import (
    AssetBundle,
    AssetRepresentation,
    EnvironmentPackage,
    EnvSpec,
    Pose,
    SceneObject,
    TaskSpec,
)


def fingerprint(path):
    path = Path(path).resolve()
    return dict(
        uri=str(path), sha256=physics.render.sha256_file(path), size_bytes=path.stat().st_size
    )


def vector(values):
    return " ".join(format(float(x), ".17g") for x in values)


def vertical_top(hulls, x, y, ceiling):
    """First downward ray intersection with the union of collision convex parts."""
    hits = []
    for hull in hulls:
        low, high = -np.inf, np.inf
        possible = True
        for nx, ny, nz, offset in hull.equations:
            rhs = -(nx * x + ny * y + offset)
            if abs(nz) < 1e-12:
                if rhs < -1e-10:
                    possible = False
                    break
            elif nz > 0:
                high = min(high, rhs / nz)
            else:
                low = max(low, rhs / nz)
        if possible and low <= high and high < ceiling:
            hits.append(high)
    return max(hits, default=-np.inf)


def measure_interior(collision_meshes, visual_bounds):
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull

    hulls = [ConvexHull(m.vertices) for m in collision_meshes]
    ceiling = float(visual_bounds[1, 2] + 0.01)
    # Only this mug is supported. Scan a small central square, retaining the deepest
    # unobstructed candidate; this avoids assumptions about the handle's direction.
    candidates = []
    for y in np.linspace(-0.02, 0.03, 11):
        for x in np.linspace(-0.01, 0.01, 5):
            ray_xy = [
                (xx, yy)
                for xx in np.linspace(x - 0.008, x + 0.008, 5)
                for yy in np.linspace(y - 0.008, y + 0.008, 5)
            ]
            heights = [vertical_top(hulls, xx, yy, ceiling) for xx, yy in ray_xy]
            floor = max(heights)
            if not np.isfinite(heights).all() or ceiling - floor < 0.04:
                continue
            low = np.array([x - 0.008, y - 0.008, floor + 1e-6])
            high = np.array([x + 0.008, y + 0.008, visual_bounds[1, 2] - 0.003])
            if high[2] <= low[2]:
                continue
            # A convex-part/box intersection is a linear feasibility problem.
            # All parts must be disjoint from the open cavity prism.
            clear = all(
                linprog(
                    np.zeros(3),
                    A_ub=h.equations[:, :3],
                    b_ub=-h.equations[:, 3],
                    bounds=list(zip(low, high)),
                    method="highs",
                ).status
                == 2
                for h in hulls
            )
            if clear:
                candidates.append((floor, abs(x) + abs(y), low, high, ray_xy, heights))
    if not candidates:
        raise ValueError("no verified open interior prism found in official mug")
    _, _, low, high, probes, heights = min(candidates, key=lambda c: (c[0], c[1]))
    return dict(
        interior_bounds=[low.tolist(), high.tolist()],
        method="downward ray grid + convex halfspace/prism separation",
        downward_probe_xy=probes,
        downward_surface_z=heights,
        collision_parts=len(hulls),
        clearance_test_passed=True,
    )


def convert_mug(source, out):
    """Bake the official MJCF geom transforms exactly, preserving all 32 parts."""
    import mujoco
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree

    source = Path(source).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    xml_path = source / "model.xml"
    root = ET.parse(xml_path).getroot()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.njnt or model.ngeom != 33 or list(model.geom_group).count(1) != 1:
        raise ValueError("expected official mug single-body model with 32 collision parts")
    geoms = list(root.findall(".//geom"))
    assets = {m.attrib["name"]: m for m in root.findall("./asset/mesh")}
    body_ids = set(int(i) for i in model.geom_bodyid)
    if len(body_ids) != 1:
        raise ValueError("official mug geoms must share one body")
    body_id = body_ids.pop()
    robot = ET.Element("robot", name="genesis_official_mug")
    link = ET.SubElement(robot, "link", name="mug")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=vector(data.xipos[body_id]))
    ET.SubElement(inertial, "mass", value=str(model.body_mass[body_id]))
    ri = data.ximat[body_id].reshape(3, 3)
    inertia = ri @ np.diag(model.body_inertia[body_id]) @ ri.T
    ET.SubElement(
        inertial,
        "inertia",
        **{
            key: str(inertia[i, j])
            for key, i, j in (
                ("ixx", 0, 0),
                ("ixy", 0, 1),
                ("ixz", 0, 2),
                ("iyy", 1, 1),
                ("iyz", 1, 2),
                ("izz", 2, 2),
            )
        },
    )
    meshes, collisions, errors = [], [], []
    sources = {str(xml_path): fingerprint(xml_path)}
    for i, geom in enumerate(geoms):
        mesh_id = int(model.geom_dataid[i])
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        verts = (
            model.mesh_vert[start : start + count] @ data.geom_xmat[i].reshape(3, 3).T
            + data.geom_xpos[i]
        )
        start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        faces = model.mesh_face[start : start + count]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        asset = assets[geom.attrib["mesh"]]
        raw_path = source / asset.attrib["file"]
        sources[str(raw_path)] = fingerprint(raw_path)
        if model.geom_group[i] == 1:
            # Preserve OBJ UVs and texture, and verify its independently transformed
            # geometry against the MJCF parser's resolved vertices.
            visual = trimesh.load(raw_path, force="mesh", process=False)
            scale = np.fromstring(asset.attrib["scale"], sep=" ")
            refquat = np.fromstring(asset.attrib["refquat"], sep=" ")
            visual.vertices = np.asarray(visual.vertices) * scale @ physics.rotation(refquat)
            error = max(
                cKDTree(verts).query(visual.vertices)[0].max(),
                cKDTree(visual.vertices).query(verts)[0].max(),
            )
            if error > 1e-6:
                raise ValueError(f"visual/MJCF transform disagreement: {error}")
            texture = source / "visual/image0.png"
            sources[str(texture)] = fingerprint(texture)
            visual.visual = trimesh.visual.TextureVisuals(
                uv=visual.visual.uv, image=Image.open(texture).copy()
            )
            mesh = visual
            name, role = "visual.obj", "visual"
            errors.append(float(error))
        else:
            name, role = f"collision_{i - 1:02d}.obj", "collision"
            collisions.append(mesh)
        mesh.export(out / name)
        reloaded = trimesh.load(out / name, force="mesh", process=False)
        error = max(
            cKDTree(mesh.vertices).query(reloaded.vertices)[0].max(),
            cKDTree(reloaded.vertices).query(mesh.vertices)[0].max(),
        )
        if error > 1e-6:
            raise ValueError("exported geometry differs from MJCF")
        errors.append(float(error))
        meshes.append(mesh)
        node = ET.SubElement(link, role)
        ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=name)
    urdf = out / "mug.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, encoding="unicode")
    bounds = np.array(
        [
            np.vstack([m.vertices for m in meshes]).min(axis=0),
            np.vstack([m.vertices for m in meshes]).max(axis=0),
        ]
    )
    measured = measure_interior(collisions, bounds)
    measured.update(
        local_bounds=bounds.tolist(),
        max_alignment_error_m=max(errors),
        source_files=list(sources.values()),
        converted_urdf=fingerprint(urdf),
        source="Genesis-Intelligence/assets",
        source_revision="4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc",
        mass_kg=float(model.body_mass[body_id]),
        friction=float(model.geom_friction[0, 0]),
        sol_params=np.concatenate((model.geom_solref[0], model.geom_solimp[0])).tolist(),
    )
    physics.write_json(out / "measurement.json", measured)
    return urdf, measured


def make_package(case, mug=None, *, dt=0.004):
    box_size = 0.008 if case == "box_in_mug" else 0.03
    h = box_size / 2
    primitive = AssetBundle(
        "official_box",
        "box",
        (
            AssetRepresentation(
                format="primitive_box",
                uri="primitive://box",
                metadata=dict(half_size_m=[h] * 3, color_rgb=[0.8, 0.2, 0.1]),
            ),
        ),
        source=dict(kind="Genesis primitive"),
        physical=dict(dimensions_m=[box_size] * 3),
    )
    assets, objects, bodies, success = [], [], {}, []
    if case != "box_on_table":
        if mug is None:
            raise ValueError("mug cases require the converted official mug")
        urdf, measured = mug
        rep = AssetRepresentation(
            format="urdf",
            backend="genesis",
            role="visual_and_collision",
            **fingerprint(urdf),
            metadata=_dependency_metadata(urdf),
        )
        # Include source and measurement bytes in the same dependency closure.
        dependencies = rep.metadata["dependencies"]
        dependencies.extend(measured["source_files"])
        dependencies.append(fingerprint(urdf.parent / "measurement.json"))
        bounds = np.array(measured["local_bounds"])
        assets.append(
            AssetBundle(
                "official_mug",
                "mug",
                (rep,),
                source=dict(
                    provider="Genesis-Intelligence/assets", revision=measured["source_revision"]
                ),
                physical=dict(dimensions_m=(bounds[1] - bounds[0]).tolist()),
            )
        )
        mug_z = -bounds[0, 2] + (0.002 if case == "mug_on_table" else 0)
        objects.append(
            SceneObject(
                "mug", "official_mug", Pose(position=(0, 0, mug_z)), static=case == "box_in_mug"
            )
        )
        bodies["mug"] = dict(
            collision=True,
            collision_parts=32,
            density=100,
            friction=measured["friction"],
            sol_params=measured["sol_params"],
            local_bounds=measured["local_bounds"],
            interior_bounds=measured["interior_bounds"],
            geometry_measurement_sha256=physics.render.sha256_file(
                urdf.parent / "measurement.json"
            ),
        )
        if case == "mug_on_table":
            success.extend(
                [
                    dict(type="support", object="mug", target="table"),
                    dict(type="released", object="mug", minimum_drop_m=0.001),
                    dict(type="upright", object="mug", max_tilt_deg=5),
                ]
            )
    if case != "mug_on_table":
        assets.append(primitive)
        pos = (0, 0, h + 0.002)
        target = "table"
        if case == "box_in_mug":
            interior = np.array(measured["interior_bounds"])
            center = interior.mean(axis=0)
            pos = (float(center[0]), float(center[1]), float(mug_z + bounds[1, 2] + h + 0.002))
            target = "mug"
        objects.append(SceneObject("box", "official_box", Pose(position=pos)))
        bodies["box"] = dict(
            collision=True, density=1000, friction=0.5, local_bounds=[[-h] * 3, [h] * 3]
        )
        success.extend(
            [
                dict(type="inside" if target == "mug" else "support", object="box", target=target),
                dict(type="released", object="box", minimum_drop_m=0.001),
            ]
        )
    package = EnvironmentPackage(
        package_id=f"genesis_official_{case}",
        env=EnvSpec(
            name=case, objects=tuple(objects), workspace_bounds_m=(-0.3, -0.3, 0, 0.3, 0.3, 0.5)
        ),
        assets=tuple(assets),
        task=TaskSpec(
            instruction=case,
            intent="official_asset_physics_validation",
            reset=dict(object_poses="from_env_spec"),
            action=dict(interface="zero_action"),
            observation=dict(state=["object_pose", "contact"]),
            plan=(),
            success=tuple(success),
        ),
        source=dict(mode="genesis_official_cases"),
        target_backends=("genesis",),
        metadata=dict(
            genesis_physics=dict(
                settings={**physics.DEFAULTS, "dt": dt, "steps": round(4 / dt)}, bodies=bodies
            )
        ),
    )
    package.validate()
    # Canonicalise numeric fields exactly as the existing JSON importer does.
    return EnvironmentPackage.from_dict(package.to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mug-dir", default=str(physics.ROOT / "data/genesis-official-assets/mug_1")
    )
    parser.add_argument("--dt", type=float, choices=(0.004, 0.002), default=0.004)
    parser.add_argument(
        "--case", choices=("all", "box_on_table", "mug_on_table", "box_in_mug"), default="all"
    )
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mug = None if args.case == "box_on_table" else convert_mug(args.mug_dir, out / "mug")
    cases = ("box_on_table", "mug_on_table", "box_in_mug") if args.case == "all" else (args.case,)
    for case in cases:
        package = make_package(case, mug, dt=args.dt)
        compiled = GenesisCompiler().compile(package, out / case, strict=True)
        print(json.dumps(dict(case=case, compile_manifest=compiled.manifest_path)))


if __name__ == "__main__":
    main()

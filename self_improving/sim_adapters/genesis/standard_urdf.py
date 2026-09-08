"""Portable single-body URDF contract and independent geometry / inertia audit."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official

SCHEMA = "genenv.standard_urdf_library.v1"
PREVIEW_SCHEMA = "genenv.standard_urdf_previews.v1"


def vector(text, default, n=3):
    a = np.array([float(v) for v in (text or default).split()])
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("invalid URDF numeric vector")
    return a


def transform(node):
    origin = node.find("origin")
    attrs = {} if origin is None else origin.attrib
    return (
        Rotation.from_euler("xyz", vector(attrs.get("rpy"), "0 0 0")).as_matrix(),
        vector(attrs.get("xyz"), "0 0 0"),
    )


def inspect(path):
    """Parse authored geometry in the link frame, independently of Genesis."""
    import trimesh

    path = Path(path)
    text = path.read_text()
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("XML entities forbidden")
    tree = ET.fromstring(text)
    links = tree.findall("link")
    if len(links) != 1 or tree.findall("joint"):
        raise ValueError("only a single rigid URDF link without joints is supported")
    link = links[0]
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError("missing authored inertial")
    mass = float(inertial.find("mass").get("value"))
    i = inertial.find("inertia").attrib
    tensor = np.array(
        [
            [float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
            [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
            [float(i["ixz"]), float(i["iyz"]), float(i["izz"])],
        ]
    )
    eigen = np.linalg.eigvalsh(tensor)
    if (
        not np.isfinite(mass)
        or mass <= 0
        or not np.isfinite(tensor).all()
        or eigen.min() <= 0
        or eigen[-1] > eigen[:2].sum() + 1e-10
    ):
        raise ValueError("invalid mass or physically invalid inertia")
    ir, com = transform(inertial)
    result = dict(mass=mass, com=com, inertia=ir @ tensor @ ir.T)
    for kind in ("visual", "collision"):
        parts, cells, offset = [], [], 0
        for node in link.findall(kind):
            geometry = node.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is not None:
                ref = library.local_reference(path, mesh.get("filename"), path.parent)
                scene = trimesh.load(ref, force="scene", process=False)
                chunks = [
                    (np.asarray(g.vertices) @ t[:3, :3].T + t[:3, 3], np.asarray(g.faces, int))
                    for name in scene.graph.nodes_geometry
                    for t, key in [scene.graph[name]]
                    for g in [scene.geometry[key]]
                ]
                points = np.concatenate([c[0] for c in chunks])
                # Reindex each chunk's faces onto the concatenated vertex array.
                base, indices = 0, []
                for chunk, faces in chunks:
                    indices.append(faces + base)
                    base += len(chunk)
                triangles = np.concatenate(indices) if indices else np.empty((0, 3), int)
                scale = vector(mesh.get("scale"), "1 1 1")
                if np.any(scale <= 0):
                    raise ValueError("nonpositive mesh scale")
                points *= scale
            elif geometry.find("box") is not None:
                box = trimesh.creation.box(vector(geometry.find("box").get("size"), ""))
                points, triangles = box.vertices, np.asarray(box.faces, int)
            else:
                raise ValueError("unsupported geometry for independent URDF audit")
            r, p = transform(node)
            points = np.asarray(points) @ r.T + p
            if not len(points) or not np.isfinite(points).all():
                raise ValueError("empty or nonfinite geometry")
            parts.append(points)
            cells.append(np.asarray(triangles, int) + offset)
            offset += len(points)
        if not parts:
            raise ValueError(f"missing {kind} geometry")
        result[kind] = np.concatenate(parts)
        result[kind + "_faces"] = (
            np.concatenate(cells) if cells else np.empty((0, 3), int)
        )
        result[kind + "_count"] = len(parts)
    return result


def verify_package(path):
    path = Path(path).resolve()
    data = library.read_json(path)
    if data.get("schema_version") != "genenv.standard_urdf_asset.v1":
        raise ValueError("invalid standard URDF package")
    official.verify_files(path.parent, data["files"])
    entry = official.safe_file(path.parent, data["entrypoint"])
    records = {r["path"]: r for r in data["files"]}
    closure = library.dependencies(entry, path.parent, records)
    if set(closure) - records.keys():
        raise ValueError("unbound dependencies")
    physics = library.read_json(official.safe_file(path.parent, data["physics_file"]))
    friction = physics["friction"]
    if not isinstance(friction, (float, int)) or not np.isfinite(friction) or friction < 0:
        raise ValueError("invalid friction")
    return data, entry, physics


def morph(gs, source, *, collision=True, fixed=False):
    return gs.morphs.URDF(
        file=str(source),
        fixed=fixed,
        collision=collision,
        scale=1.0,
        convexify=False,
        decimate=False,
        watertighten=None,
        recompute_inertia=False,
        align=False,
        merge_fixed_links=False,
    )


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def distance(a, b):
    if not len(a) or not len(b):
        raise ValueError("empty geometry in audit")
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def audit(entity, expected, *, collision=True, friction=None):
    from self_improving.sim_adapters.genesis.physics_math import rotation

    r = rotation(array(entity.get_quat()).reshape(4))
    p = array(entity.get_pos()).reshape(3)
    visual = np.concatenate(
        [array(g.get_vverts()).reshape(-1, 3) for link in entity.links for g in link.vgeoms]
    )
    result = dict(visual_error_m=distance((visual - p) @ r, expected["visual"]))
    if collision:
        points = array(entity.get_verts()).reshape(-1, 3)
        result.update(
            collision_error_m=distance((points - p) @ r, expected["collision"]),
            collision_count=len(entity.geoms),
            dofs=int(entity.n_dofs),
            fixed=bool(entity.base_link.is_fixed),
        )
        if len(entity.geoms) != expected["collision_count"] or entity.n_dofs != 6:
            raise ValueError("collision count or floating degrees of freedom changed")
        if entity.base_link.is_fixed:
            raise ValueError("dynamic test body was fixed")
        mass = float(array(entity.get_mass()).reshape(-1)[0])
        inertia = array(entity.get_links_inertia()).reshape(-1, 3, 3)[0]
        # Genesis stores inertia in the inertial principal frame; rotate back to link frame.
        link = entity.base_link
        iq = np.asarray(link.desc.inertial_quat)
        ip = np.asarray(link.desc.inertial_pos)
        ir = rotation(iq)
        actual_i = ir @ inertia @ ir.T
        result.update(
            mass_kg=mass,
            inertia_link_kg_m2=actual_i.tolist(),
            com_link_m=ip.tolist(),
            mass_relative_error=abs(mass - expected["mass"]) / expected["mass"],
            inertia_relative_error=float(
                np.linalg.norm(actual_i - expected["inertia"]) / np.linalg.norm(expected["inertia"])
            ),
            com_error_m=float(np.max(np.abs(ip - expected["com"]))),
            friction=[float(array(g.get_friction())) for g in entity.geoms],
        )
        if (
            result["mass_relative_error"] > 1e-4
            or result["inertia_relative_error"] > 1e-4
            or result["com_error_m"] > 1e-5
        ):
            raise ValueError(
                "authored mass, COM or inertia changed during loading: " + json.dumps(result)
            )
        if friction is not None and not np.allclose(result["friction"], friction, atol=1e-6):
            raise ValueError("authored friction was not applied")
    if max(result.get("collision_error_m", 0), result["visual_error_m"]) > 1e-5:
        raise ValueError("native geometry differs from independently parsed URDF")
    return result


def verify_preview_index(path):
    path = Path(path).resolve()
    data = library.read_json(path)
    if data["schema_version"] != PREVIEW_SCHEMA:
        raise ValueError("invalid standard asset preview schema")
    official.verify_files(path.parent, data["files"])
    reference = data["source_inventory"]
    if library.sha256(reference["path"]) != reference["sha256"]:
        raise ValueError("standard library changed")
    inventory = library.read_json(reference["path"])
    if inventory["schema_version"] != SCHEMA:
        raise ValueError("invalid standard library")
    by_id = {a["asset_id"]: a for a in inventory["assets"] if a["status"] == "imported"}
    if set(by_id) != {a["asset_id"] for a in data["assets"]}:
        raise ValueError("preview asset set mismatch")
    if len(data["assets"]) != len(by_id):
        raise ValueError("duplicate preview IDs")
    for item in data["assets"]:
        record = library.read_json(official.safe_file(path.parent, item["record"]))
        package = Path(reference["path"]).parent / by_id[item["asset_id"]]["package"]
        pkg, entry, _ = verify_package(package)
        if (
            library.sha256(package) != by_id[item["asset_id"]]["package_sha256"]
            or record["source_files"] != pkg["files"]
            or Path(record["source_root"]) != package.parent
            or record["entrypoint"] != pkg["entrypoint"]
        ):
            raise ValueError("preview package binding mismatch")
        if item["status"] == "preview_passed":
            preview = library.read_json(official.safe_file(path.parent, record["preview_result"]))
            if (
                preview["status"] != "passed"
                or preview["physics_steps"] != 0
                or len(preview["views"]) != 6
                or not preview.get("geometry_audit")
            ):
                raise ValueError("invalid preview evidence")
    return data

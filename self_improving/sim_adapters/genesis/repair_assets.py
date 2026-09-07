"""Measured canonical assets and traceable, quality-gated collision derivatives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import validate_asset_scene as native
from self_improving.sim_adapters.genesis.storage_paths import local_path

COACD_VERSION = "1.0.14"
COACD = dict(
    threshold=0.005,
    max_convex_hull=64,
    preprocess_mode="auto",
    preprocess_resolution=300,
    resolution=2000,
    mcts_nodes=20,
    mcts_iterations=100,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    decimate=False,
    max_ch_vertex=256,
    extrude=False,
    apx_mode="ch",
    seed=0,
)
COACD_PREPARATION = dict(
    max_input_faces=2000,
    method="quadric_decimation",
    timeout_s=120,
    closed_mesh_preprocess="off",
    zero_volume_rank_tolerance_m=1e-7,
)
CACHE = Path(__file__).resolve().parents[3] / ".cache/genesis/repair_collision_cache_v1"


def init_genesis():
    os.environ["GS_HEADLESS"] = "1"
    os.environ["PYGLET_HEADLESS"] = "1"
    import subprocess

    import genesis as gs

    commit = subprocess.check_output(
        ["git", "-C", str(Path(gs.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != official.GENESIS_COMMIT:
        raise ValueError("Genesis revision mismatch")
    gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="warning")
    return gs


def visible_geometry(entity, source):
    """Exclude transparent MJCF collision display meshes from the visual reference."""
    source = local_path(source)
    if Path(source).suffix != ".xml":
        return builder.entity_mesh(entity)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(local_path(source)))
    if len(entity.vgeoms) != model.ngeom:
        raise ValueError("native visual/material mapping mismatch")
    vertices, faces, offset = [], [], 0
    for i, g in enumerate(entity.vgeoms):
        material = model.geom_matid[i]
        alpha = model.mat_rgba[material, 3] if material >= 0 else model.geom_rgba[i, 3]
        if alpha <= 0:
            continue
        points = native.array(g.get_vverts()).reshape(-1, 3)
        vertices.append(points)
        faces.append(np.asarray(g.init_vfaces) + offset)
        offset += len(points)
    if not vertices:
        raise ValueError("asset has no visible geometry")
    return np.concatenate(vertices), np.concatenate(faces)


def scale_for(category, size, description="", metadata=None):
    metadata = metadata or {}
    size = np.asarray(size, float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("invalid measured asset dimensions")
    prior = geo.PRIORS.get(category)
    if prior is None and not {"unit_scale", "natural_up", "tip_limit_deg"} <= metadata.keys():
        raise ValueError(f"NEEDS_ASSET_METADATA: {category}")
    unit = float(metadata.get("unit_scale", 1.0))
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("invalid unit scale")
    metric = prior[0] if prior else metadata.get("size_metric", "max")
    measured = float(
        size[2] if metric == "height" else max(size[:2]) if metric == "diameter" else max(size)
    )
    # Dimensions attached to this object's description, not an unrelated object in the text.
    dimensions = re.findall(
        r"(?:高(?:度)?|直径|最大边长|height|diameter)\s*(?:为|是|=|of)?\s*"
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|毫米|厘米|米)",
        description,
        re.I,
    )
    targets = [
        float(v) * ({"mm": 0.001, "毫米": 0.001, "cm": 0.01, "厘米": 0.01}.get(u, 1.0))
        for v, u in dimensions
    ]
    if targets and (min(targets) <= 0 or max(targets) - min(targets) > 1e-6):
        raise ValueError("conflicting text dimensions")
    if targets:
        if "unit_scale" in metadata and abs(measured * unit - targets[0]) > 1e-5:
            raise ValueError("text size conflicts with trusted asset metadata")
        return targets[0] / measured, "explicit_text_dimension"
    if prior and not prior[1] <= measured * unit <= prior[2]:
        if "unit_scale" in metadata:
            raise ValueError("trusted scale conflicts with category prior")
        return prior[3] / measured, "category_prior_outlier_normalization"
    return unit, "trusted_metadata" if metadata else "native_meter_scale_with_category_check"


def sample_surface(mesh, max_points=15000):
    # Vertices, centroids and interior samples detect caps bridging a cavity or leg gap.
    triangles = mesh.triangles
    values = [mesh.vertices]
    for u, v in [(1 / 3, 1 / 3), (0.1, 0.1), (0.8, 0.1), (0.1, 0.8), (0.5, 0.25), (0.25, 0.5)]:
        values.append(triangles[:, 0] * (1 - u - v) + triangles[:, 1] * u + triangles[:, 2] * v)
    points = np.concatenate(values)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points


def closed_convex(mesh):
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        return False
    envelope = ConvexHull(mesh.vertices).volume
    return abs(mesh.volume - envelope) <= max(1e-14, envelope * 1e-6)


def proxy_quality(visual, parts, surface=None, *, fixed_native=False, native_semantics=False):
    diagonal = float(np.linalg.norm(visual.extents))
    limit = min(0.002, 0.01 * diagonal)
    if not parts or (not fixed_native and any(not closed_convex(m) for m in parts)):
        return dict(passed=False, reason="nonconvex_or_open_proxy")
    joined = trimesh.util.concatenate(parts)
    if fixed_native:
        # Static concave geometry is acceptable only when its measured triangle surface
        # is preserved. Dynamic meshes still require closed convex collision parts.
        solids = trimesh.Trimesh(joined.vertices, joined.faces, process=True).split(
            only_watertight=False
        )
        if not solids or any(not m.is_watertight for m in solids):
            return dict(passed=False, reason="open_native_fixed_collision")

    def distances(mesh, points):
        values = []
        for block in np.array_split(points, max(1, math_ceil(len(points) / 256))):
            if len(block):
                _, ds, _ = trimesh.proximity.closest_point(mesh, block)
                values.append(ds)
        return np.concatenate(values) if values else np.array([0.0])

    def distance(mesh, points):
        return float(distances(mesh, points).max())

    # Internal decomposition boundaries are allowed. A hull cap deep inside a visual cavity
    # must not be allowed: expose only samples on the union boundary, not inter-part seams.
    def union_contains(points):
        inside = np.zeros(len(points), dtype=bool)
        for eq, box in zip(envelopes, boxes):
            selected = np.flatnonzero(
                ~inside
                & np.all(points >= box[0] - 1e-8, axis=1)
                & np.all(points <= box[1] + 1e-8, axis=1)
            )
            if len(selected):
                inside[selected] = np.all(points[selected] @ eq[:, :3].T + eq[:, 3] <= 1e-8, axis=1)
        return inside

    exposed = []
    envelopes = [ConvexHull(p.vertices).equations for p in parts]
    boxes = [p.bounds for p in parts]
    for i, part in enumerate(parts):
        points = sample_surface(part, max(256, 15000 // len(parts)))
        mask = np.ones(len(points), dtype=bool)
        for j, other in enumerate(parts):
            if i != j:
                if np.any(
                    np.minimum(boxes[i][1], boxes[j][1]) - np.maximum(boxes[i][0], boxes[j][0])
                    < -1e-6
                ):
                    continue
                eq = envelopes[j]
                relevant = mask & np.all(points >= boxes[j][0] - 1e-6, axis=1)
                relevant &= np.all(points <= boxes[j][1] + 1e-6, axis=1)
                indices = np.flatnonzero(relevant)
                if len(indices):
                    inside = np.all(points[indices] @ eq[:, :3].T + eq[:, 3] < -1e-6, axis=1)
                    mask[indices[inside]] = False
        # Shared planar seams are not strictly inside either component. Test a tiny
        # neighborhood in 26 directions before treating such seams as external caps.
        boundary = points[mask]
        internal = np.ones(len(boundary), dtype=bool)
        from itertools import product

        for direction in product((-1, 0, 1), repeat=3):
            if direction == (0, 0, 0):
                continue
            indices = np.flatnonzero(internal)
            if not len(indices):
                break
            delta = np.asarray(direction, float)
            delta *= 1e-5 / np.linalg.norm(delta)
            internal[indices] &= union_contains(boundary[indices] + delta)
        exposed.append(boundary[~internal])
    boundary = sample_surface(joined) if fixed_native else np.concatenate(exposed)
    # Cut faces buried inside a measured closed visual solid are legitimate internal
    # decomposition faces, even when approximate pieces leave a small internal seam.
    # Empty bowl cavities and spaces between table legs are outside these solids.
    buried = np.zeros(len(boundary), dtype=bool)
    outward = distance(visual, boundary)
    if outward > limit:
        solid_parts = trimesh.Trimesh(visual.vertices, visual.faces, process=True).split(
            only_watertight=True
        )
        buried = np.zeros(len(boundary), dtype=bool)
        for solid in solid_parts:
            bb = solid.bounds
            indices = np.flatnonzero(
                ~buried
                & np.all(boundary > bb[0] + 1e-7, axis=1)
                & np.all(boundary < bb[1] - 1e-7, axis=1)
            )
            # Ray/triangle candidate expansion scales with the original face count.
            # Bound memory without dropping any boundary sample or changing the gate.
            contains_batch = min(512, max(1, 1_000_000 // max(1, len(solid.faces))))
            for block in np.array_split(indices, max(1, math_ceil(len(indices) / contains_batch))):
                if len(block):
                    buried[block] = solid.contains(boundary[block])
        outward = distance(visual, boundary[~buried])
    inward_values = distances(joined, sample_surface(visual))
    inward = float(inward_values.max())
    area_samples, _ = trimesh.sample.sample_surface(visual, 15000, seed=0)
    visual_coverage = float(np.mean(distances(joined, area_samples) <= limit))
    bottom_error = float(abs(joined.bounds[0, 2] - visual.bounds[0, 2]))
    # Authored collision semantics may omit a small visual detail (e.g. an apple stem).
    # Preserve them only with 99% sampled surface coverage, a 1 mm bottom match, and
    # the same strict outside-volume and measured-support checks as derived proxies.
    inward_ok = (
        (visual_coverage >= 0.99 and bottom_error <= 0.001) if native_semantics else inward <= limit
    )
    height_error = 0.0
    coverage = True
    if surface:
        polygon = np.asarray(surface["polygon_xy_m"])
        xy = np.array(
            [
                [x, y]
                for x in np.linspace(polygon[:, 0].min(), polygon[:, 0].max(), 11)
                for y in np.linspace(polygon[:, 1].min(), polygon[:, 1].max(), 11)
            ]
        )
        xy = np.array([p for p in xy if geo.clearance(polygon, [p]) > 0.001])
        origins = np.c_[xy, np.full(len(xy), surface["z_m"] + 0.01)]
        hits, rays, _ = joined.ray.intersects_location(
            origins, np.tile([0.0, 0.0, -1.0], (len(xy), 1)), multiple_hits=True
        )
        for i in range(len(xy)):
            zs = hits[rays == i, 2]
            if not len(zs):
                coverage = False
            else:
                height_error = max(height_error, float(abs(zs.max() - surface["z_m"])))
    return dict(
        passed=bool(outward <= limit and inward_ok and coverage and height_error <= 0.001),
        native_semantics=native_semantics,
        visual_coverage_fraction=visual_coverage,
        visual_coverage_sampling="15000_area_uniform_seed_0",
        native_visual_coverage_required=0.99 if native_semantics else 1.0,
        bottom_alignment_error_m=bottom_error,
        external_surface_error_m=outward,
        visual_surface_error_m=inward,
        approximation_limit_m=limit,
        support_height_error_m=height_error,
        support_coverage=coverage,
        method="sampled_external_boundary_and_support_rays",
        buried_cut_face_samples=int(buried.sum()),
    )


def math_ceil(value):
    return int(np.ceil(value))


def decompose(visual, surface, out):
    import fcntl

    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _decompose_locked(visual, surface, out)


def _decompose_locked(visual, surface, out):
    if version("coacd") != COACD_VERSION:
        raise ValueError("CoACD version mismatch")
    key = hashlib.sha256(
        visual.vertices.tobytes()
        + visual.faces.tobytes()
        + json.dumps(COACD, sort_keys=True).encode()
        + COACD_VERSION.encode()
        + b"connected_components_v3"
        + json.dumps(COACD_PREPARATION, sort_keys=True).encode()
    ).hexdigest()
    cache = CACHE / key
    attempts = []
    if (cache / "manifest.json").exists():
        manifest = lib.read_json(cache / "manifest.json")
        official.verify_files(cache, manifest["files"])
        parts = [
            trimesh.load(cache / r["path"], force="mesh", process=False) for r in manifest["files"]
        ]
        quality = proxy_quality(visual, parts, surface)
        if not quality["passed"]:
            raise ValueError("cached proxy no longer passes geometry checks")
        return parts, dict(
            method="coacd",
            cache_hit=True,
            key=key,
            quality=quality,
            options=COACD,
            preprocessing=COACD_PREPARATION,
            version=COACD_VERSION,
        )
    try:
        # Preserve disconnected solid components before decomposition. This avoids
        # voxelizing the empty space between table frames and changing the thin tabletop.
        components = trimesh.Trimesh(visual.vertices, visual.faces, process=True).split(
            only_watertight=False
        )
        parts = []
        for i, component in enumerate(components):
            print(f"collision component {i + 1}/{len(components)}", flush=True)
            singular = np.linalg.svd(
                component.vertices - component.vertices.mean(0), compute_uv=False
            )
            if abs(component.volume) <= 1e-12 and singular[-1] <= 1e-7:
                clip.write_json(
                    out / f"zero_volume_component_{i:03d}.json",
                    dict(
                        reason="zero-volume coplanar patch; full visual geometry gate retained",
                        volume_m3=float(component.volume),
                        singular_values_m=singular.tolist(),
                    ),
                )
                continue
            if component.is_watertight and component.is_convex:
                parts.append(component.convex_hull)
                continue
            result = run_component(component, i, out)
            parts.extend(trimesh.Trimesh(v, f, process=True).convex_hull for v, f in result)
        print(f"checking collision proxy: {len(parts)} parts", flush=True)
        candidate_dir = out / "coacd_candidate"
        candidate_dir.mkdir(exist_ok=True)
        for i, part in enumerate(parts):
            part.export(candidate_dir / f"part_{i:03d}.obj")
        quality = proxy_quality(visual, parts, surface)
        attempts.append(dict(method="coacd", quality=quality))
        if quality["passed"]:
            cache.mkdir(parents=True, exist_ok=True)
            for i, part in enumerate(parts):
                part.export(cache / f"part_{i:03d}.obj")
            files = [official.fingerprint(p, cache) for p in sorted(cache.glob("*.obj"))]
            clip.write_json(cache / "manifest.json", dict(files=files))
            return parts, dict(
                method="coacd",
                cache_hit=False,
                key=key,
                quality=quality,
                options=COACD,
                preprocessing=COACD_PREPARATION,
                version=COACD_VERSION,
            )
    except Exception as exc:
        attempts.append(dict(method="coacd", error=f"{type(exc).__name__}: {exc}"))
    for method, part in [
        ("convex_hull", visual.convex_hull),
        ("bounding_box", visual.bounding_box),
    ]:
        quality = proxy_quality(visual, [part], surface)
        attempts.append(dict(method=method, quality=quality))
        if quality["passed"]:
            return [part], dict(method=method, attempts=attempts, quality=quality)
    clip.write_json(out / "collision_failure.json", dict(attempts=attempts))
    raise ValueError("ASSET_PREPARATION_FAILED: no faithful collision proxy")


def run_component(component, index, out):
    work = out / "coacd_work"
    work.mkdir(exist_ok=True)
    original_faces = len(component.faces)
    if original_faces > COACD_PREPARATION["max_input_faces"]:
        component = component.simplify_quadric_decimation(
            face_count=COACD_PREPARATION["max_input_faces"]
        )
    source = work / f"component_{index:03d}.npz"
    destination = work / f"result_{index:03d}.npz"
    options = work / f"coacd_options_{index:03d}.json"
    np.savez_compressed(source, vertices=component.vertices, faces=component.faces)
    effective_options = dict(COACD)
    if component.is_watertight and component.is_winding_consistent:
        effective_options["preprocess_mode"] = "off"
    clip.write_json(options, effective_options)
    clip.write_json(
        work / f"preparation_{index:03d}.json",
        dict(
            original_faces=original_faces,
            processed_faces=len(component.faces),
            options=COACD_PREPARATION,
            effective_coacd_options=effective_options,
            geometry_gate="final proxy checked against the complete original visible mesh",
        ),
    )
    print(f"CoACD input faces: {original_faces} -> {len(component.faces)}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improving.sim_adapters.genesis.repair_coacd",
            str(source),
            str(options),
            str(destination),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        timeout=COACD_PREPARATION["timeout_s"],
    )
    with np.load(destination) as result:
        return [
            (result[f"vertices_{i}"], result[f"faces_{i}"]) for i in range(len(result.files) // 2)
        ]


def proxy_inertia(parts, diagonal, density=600.0):
    pitch = diagonal / 128
    occupied = set()
    for part in parts:
        points = part.voxelized(pitch).fill().points
        occupied.update(map(tuple, np.rint(points / pitch).astype(int)))
    points = np.asarray(sorted(occupied), float) * pitch
    com = points.mean(0)
    delta = points - com
    voxel_mass = density * pitch**3
    mass = voxel_mass * len(points)
    inertia = voxel_mass * (np.eye(3) * (delta**2).sum() - delta.T @ delta)
    inertia += np.eye(3) * mass * pitch**2 / 6
    return mass, com, inertia


def aggregate_inertia(entity, scale, anchor, gs):
    """Aggregate rigid links in world axes with the requested canonical origin."""
    ids = [link.idx for link in entity.links]
    weights = native.array(entity.get_links_mass()).reshape(-1) * scale**3
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("invalid authored masses")
    solver = entity.scene.rigid_solver
    centers = native.array(solver.get_links_pos(ids, ref=gs.link_ref_frame.link_COM)).reshape(-1, 3)
    centers = centers * scale - anchor
    orientations = native.array(solver.get_links_quat(ids, relative=False)).reshape(-1, 4)
    tensors = native.array(entity.get_links_inertia()).reshape(-1, 3, 3) * scale**5
    com = weights @ centers / weights.sum()
    inertia = np.zeros((3, 3))
    for link, mass, center, q, tensor in zip(entity.links, weights, centers, orientations, tensors):
        delta = center - com
        rotation = geo.rotation(q) @ geo.rotation(link.desc.inertial_quat)
        inertia += rotation @ tensor @ rotation.T + mass * (
            np.eye(3) * np.dot(delta, delta) - np.outer(delta, delta)
        )
    if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("invalid authored aggregate inertia")
    return float(weights.sum()), com, inertia


def authored_properties(entity, source, scale, anchor, gs):
    """Preserve original mass/inertia and authored contact parameters."""
    import mujoco

    mass, com, inertia = aggregate_inertia(entity, scale, anchor, gs)
    model = mujoco.MjModel.from_xml_path(str(local_path(source)))
    contacts = [
        dict(
            friction=model.geom_friction[i].tolist(),
            solref=model.geom_solref[i].tolist(),
            solimp=model.geom_solimp[i].tolist(),
        )
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    if len(contacts) != len(entity.geoms):
        raise ValueError("authored contact material mapping mismatch")
    return mass, com, inertia, contacts


def write_collision_model(parts, out, mass, com, inertia, *, fixed=False, contact_params=None):
    root = ET.Element("mujoco", model="derived_rigid_asset")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="object")
    if not fixed:
        ET.SubElement(body, "freejoint")
    # MuJoCo's fullinertia eigensolver may discard small off-diagonal terms.
    # Export the principal frame explicitly so the frozen tensor survives loading.
    principal, axes = np.linalg.eigh(np.asarray(inertia, dtype=float))
    if not np.isfinite(principal).all() or principal.min() <= 0:
        raise ValueError("invalid frozen inertia for export")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    ET.SubElement(
        body, "inertial", pos=" ".join(map(str, com)), mass=str(mass),
        diaginertia=" ".join(map(str, principal)),
        quat=" ".join(map(str, geo.quat(axes))),
    )
    for i, part in enumerate(parts):
        path = out / f"collision_{i:03d}.obj"
        part.export(path)
        ET.SubElement(assets, "mesh", name=f"part{i}", file=path.name)
        params = (
            contact_params[i]
            if contact_params
            else dict(
                friction=[1, 0.005, 0.0001], solref=[0.01, 1], solimp=[0.9, 0.95, 0.001, 0.5, 2]
            )
        )
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=f"part{i}",
            rgba="0.6 0.6 0.6 1",
            **{k: " ".join(map(str, v)) for k, v in params.items()},
        )
    path = out / "collision_model.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def select_mass_properties(source_is_mjcf, frozen, parts, diagonal, repair_preset):
    """Keep v2 mass/COM/inertia tied to native loading, never to a repaired proxy."""
    if source_is_mjcf or repair_preset == "text_scene_v2":
        if frozen is None:
            raise ValueError("native mass properties were not frozen before collision repair")
        mass, com, inertia = frozen
        source = (
            "authored_mass_and_inertia_scaled_s3_s5" if source_is_mjcf
            else "native_loaded_mass_and_inertia_scaled_s3_s5"
        )
        return mass, com, inertia, source
    mass, com, inertia = proxy_inertia(parts, diagonal)
    return mass, com, inertia, "collision_union_voxels_d_over_128_density_600_assumption"


def prepare(document, bindings, out, fixed, metadata, check, *, repair_preset="legacy"):
    if repair_preset not in {"legacy", "text_scene_v2"}:
        raise ValueError("unknown collision repair preset")

    asset_deadline = None

    def check_asset():
        if asset_deadline is not None and time.monotonic() >= asset_deadline:
            raise ValueError("ASSET_PREPARATION_FAILED: total asset deadline exhausted")
        check()
        if asset_deadline is not None and time.monotonic() >= asset_deadline:
            raise ValueError("ASSET_PREPARATION_FAILED: total asset deadline exhausted")

    def native_quality(visual, parts, surface, target, **flags):
        if repair_preset == "legacy":
            return proxy_quality(visual, parts, surface, **flags)
        from self_improving.sim_adapters.genesis import repair_collision_v2

        return repair_collision_v2.native_quality(
            visual, parts, surface, target, deadline=asset_deadline, check=check_asset, **flags
        )

    def choose_proxy(visual, surface, target, category):
        if repair_preset == "legacy":
            return decompose(visual, surface, target)
        from self_improving.sim_adapters.genesis import repair_collision_v3

        return repair_collision_v3.decompose(
            visual, surface, target, category=category, check=check_asset, deadline=asset_deadline
        )

    gs = init_genesis()
    output, failures = {}, {}
    try:
        scene = gs.Scene(show_viewer=False, show_FPS=False)
        entities = {}
        for obj in document["objects"]:
            n = obj["object_id"]
            source = local_path(bindings[n]["model_entrypoint"])
            if source.suffix not in {".xml", ".glb"}:
                raise ValueError("first release supports MJCF and GLB")
            options = dict(
                file=str(source),
                scale=1.0,
                convexify=False,
                decimate=False,
                watertighten=None,
                collision=True,
            )
            morph = (
                gs.morphs.MJCF(**options) if source.suffix == ".xml" else gs.morphs.Mesh(**options)
            )
            entities[n] = scene.add_entity(morph, material=gs.materials.Rigid(), vis_mode="visual")
        scene.build()
        check()
        for obj in document["objects"]:
            n = obj["object_id"]
            entity = entities[n]
            binding = bindings[n]
            target = out / "assets" / n
            target.mkdir(parents=True)
            asset_started = time.monotonic()
            if repair_preset == "text_scene_v2":
                from self_improving.sim_adapters.genesis import repair_collision_v2

                asset_deadline = asset_started + repair_collision_v2.ASSET_BUDGET_S
            check()
            try:
                check_asset()
                vertices, faces = visible_geometry(entity, binding["model_entrypoint"])
                if entity.n_dofs != 6:
                    raise ValueError(f"{n}: articulated or nonfree source unsupported")
                info = metadata.get(n, {})
                if info.get("natural_up", [0, 0, 1]) != [0, 0, 1]:
                    raise ValueError(
                        "NEEDS_ASSET_METADATA: first release requires verified native +Z up"
                    )
                tip = float(info.get("tip_limit_deg", 15))
                if not np.isfinite(tip) or not 0 < tip <= 90:
                    raise ValueError("invalid natural-pose tilt rule")
                if obj["category"] in geo.PRIORS and tip != 15:
                    raise ValueError("text_repair_v1 freezes current-category tilt at 15 degrees")
                scale, scale_source = scale_for(
                    obj["category"], np.ptp(vertices, axis=0), obj["description"], info
                )
                anchor = np.r_[vertices[:, :2].mean(0), vertices[:, 2].min()] * scale
                # Use bounds to avoid dependence on tessellation density.
                anchor[:2] = (vertices[:, :2].min(0) + vertices[:, :2].max(0)) * scale / 2
                canonical = vertices * scale - anchor
                visual = trimesh.Trimesh(canonical, faces, process=False)
                np.savez_compressed(target / "visual_geometry.npz", vertices=canonical, faces=faces)
                size = visual.extents
                diagonal = float(np.linalg.norm(size))
                surface = None
                try:
                    surface = builder.support_surface(canonical, faces)
                except ValueError:
                    pass
                parts = [
                    trimesh.Trimesh(
                        native.array(g.get_verts()).reshape(-1, 3) * scale - anchor,
                        g.get_trimesh().faces,
                        process=True,
                    )
                    for g in entity.geoms
                ]
                source_is_mjcf = Path(binding["model_entrypoint"]).suffix == ".xml"
                original_parts = parts
                authored = source_is_mjcf
                frozen_properties = None
                if source_is_mjcf:
                    authored_mass, authored_com, authored_inertia, authored_contacts = (
                        authored_properties(entity, binding["model_entrypoint"], scale, anchor, gs)
                    )
                    frozen_properties = (authored_mass, authored_com, authored_inertia)
                elif repair_preset == "text_scene_v2":
                    frozen_properties = aggregate_inertia(entity, scale, anchor, gs)
                if repair_preset == "text_scene_v2":
                    fm, fc, fi = frozen_properties
                    clip.write_json(target / "native_mass_properties.json", dict(
                        mass_kg=fm, com_local_m=fc.tolist(), inertia_local_kg_m2=fi.tolist(),
                        origin="native load before any collision candidate", scale=scale,
                    ))
                fixed_native = n in fixed and not authored
                print(f"checking native collision: {n}", flush=True)
                if fixed_native:
                    collision = dict(
                        method="native_fixed_triangle_mesh",
                        parts=len(parts),
                        quality=native_quality(visual, parts, surface, target, fixed_native=True),
                    )
                    clip.write_json(target / "native_collision_report.json", collision)
                    if collision["quality"]["passed"]:
                        authored = True  # Preserve the native coordinate frame and source file.
                    else:
                        parts, collision = choose_proxy(visual, surface, target, obj["category"])
                elif authored and all(closed_convex(p) for p in parts):
                    collision = dict(
                        method="native_collision",
                        parts=len(parts),
                        quality=native_quality(
                            visual, parts, surface, target, native_semantics=True
                        ),
                    )
                    clip.write_json(target / "native_collision_report.json", collision)
                    print(f"native quality: {n}: {collision['quality']}", flush=True)
                    # Apply the same faithful geometry gate to authored decompositions.
                    if not collision["quality"]["passed"]:
                        parts, collision = choose_proxy(visual, surface, target, obj["category"])
                        authored = False
                else:
                    parts, collision = choose_proxy(visual, surface, target, obj["category"])
                    authored = False
                original_pose = native.pose(entity)
                physics_file = binding["model_entrypoint"]
                contact_params = None
                mass, com, inertia, mass_source = select_mass_properties(
                    source_is_mjcf, frozen_properties, parts, diagonal, repair_preset
                )
                if not authored or (source_is_mjcf and n in fixed):
                    if source_is_mjcf:
                        centers = np.array([p.vertices.mean(0) for p in original_parts])
                        contact_params = [
                            authored_contacts[
                                int(np.argmin(np.linalg.norm(centers - p.vertices.mean(0), axis=1)))
                            ]
                            for p in parts
                        ]
                    physics_file = str(
                        write_collision_model(
                            parts,
                            target,
                            mass,
                            com,
                            inertia,
                            fixed=n in fixed,
                            contact_params=contact_params,
                        )
                    )
                    authored = False
                asset = dict(
                    binding,
                    category=obj["category"],
                    scale=scale,
                    scale_source=scale_source,
                    anchor_m=anchor.tolist(),
                    native_pose=original_pose,
                    bbox_size_m=size.tolist(),
                    diagonal_m=diagonal,
                    radius_m=diagonal / 2,
                    hull=canonical[ConvexHull(canonical).vertices].tolist(),
                    surface=surface,
                    collision_hulls=[p.vertices.tolist() for p in parts],
                    collision=collision,
                    collision_meshes=(
                        [dict(vertices=p.vertices.tolist(), faces=p.faces.tolist()) for p in parts]
                        if collision["method"] == "native_fixed_triangle_mesh" else None
                    ),
                    native_collision=authored,
                    physics_file=physics_file,
                    mass_kg=mass,
                    com_local_m=com.tolist(),
                    inertia_local_kg_m2=inertia.tolist(),
                    bottom_offset_m=float(
                        -min(canonical[:, 2].min(), min(p.vertices[:, 2].min() for p in parts))
                    ),
                    visual_bottom_offset_m=float(-canonical[:, 2].min()),
                    mass_source=mass_source,
                    natural_up=[0, 0, 1],
                    tip_limit_deg=float(info.get("tip_limit_deg", 15)),
                    fixed=n in fixed,
                    margin_m=max(0.01, 0.02 * diagonal),
                    buffer_m=max(0.005 if repair_preset == "text_scene_v2" else 0.002,
                                 0.005 * diagonal),
                    visual_mesh_sha256=hashlib.sha256(
                        vertices.tobytes() + faces.tobytes()
                    ).hexdigest(),
                )
                np.savez_compressed(target / "visual_geometry.npz", vertices=canonical, faces=faces)
                asset["geometry_file"] = str(target / "visual_geometry.npz")
                if repair_preset == "text_scene_v2":
                    check_asset()
                    clip.write_json(target / "asset_budget.json", dict(
                        elapsed_s=time.monotonic() - asset_started,
                        limit_s=repair_collision_v2.ASSET_BUDGET_S,
                        scope="native per-asset measurement through collision qualification",
                    ))
                asset["derived_files"] = [
                    official.fingerprint(p, target)
                    for p in sorted(target.rglob("*"))
                    if p.is_file()
                ]
                asset["derived_root"] = str(target)
                output[n] = asset
                clip.write_json(target / "asset_physics_info.json", asset)
                print(
                    f"asset prepared: {n}, scale={scale}, collision={collision['method']}",
                    flush=True,
                )
            except Exception as exc:
                if repair_preset == "text_scene_v2":
                    clip.write_json(target / "asset_budget.json", dict(
                        elapsed_s=time.monotonic() - asset_started,
                        limit_s=repair_collision_v2.ASSET_BUDGET_S,
                        scope="native per-asset measurement through collision qualification",
                        budget_exhausted=time.monotonic() >= asset_deadline,
                    ))
                failures[n] = dict(error=f"{type(exc).__name__}: {exc}", binding=binding)
                clip.write_json(target / "preparation_error.json", failures[n])
                print(f"asset rejected: {n}: {exc}", flush=True)
        clip.write_json(
            out / "asset_preparation_report.json",
            dict(
                passed=not failures,
                assets=output,
                failures=failures,
                coacd_version=COACD_VERSION,
                coacd_options=COACD,
                coacd_preparation=COACD_PREPARATION,
                preprocessing=COACD_PREPARATION,
                **(dict(repair_preset=repair_preset, collision_policy="collision_repair_v3")
                   if repair_preset == "text_scene_v2" else {}),
            ),
        )
        check()
        if failures:
            raise ValueError(f"ASSET_PREPARATION_FAILED: {', '.join(failures)}")
        check()
        return output
    finally:
        gs.destroy()


def verify_visual_input(data):
    """Probe actual native visual transforms before any formal physics step, without cameras."""
    gs = init_genesis()
    report = {}
    try:
        scene = gs.Scene(show_viewer=False, show_FPS=False)
        entities = {}
        for n, asset in data["assets"].items():
            options = dict(
                file=str(local_path(asset["model_entrypoint"])),
                scale=asset["scale"],
                convexify=False,
                decimate=False,
                watertighten=None,
                collision=False,
            )
            morph = (
                gs.morphs.MJCF(**options)
                if Path(asset["model_entrypoint"]).suffix == ".xml"
                else gs.morphs.Mesh(**options, fixed=True)
            )
            entities[n] = scene.add_entity(morph, material=gs.materials.Rigid(), vis_mode="visual")
        scene.build()
        for n, entity in entities.items():
            asset, pose = data["assets"][n], data["poses"][n]
            reference = native.pose(entity)
            delta = geo.rotation(pose["orientation_wxyz"])
            entity.set_quat(np.array(geo.quat(delta @ geo.rotation(reference["orientation_wxyz"]))))
            entity.set_pos(
                np.array(pose["position"])
                + delta @ (np.array(reference["position"]) - asset["anchor_m"])
            )
            actual, _ = visible_geometry(entity, asset["model_entrypoint"])
            expected = geo.transform(np.load(asset["geometry_file"])["vertices"], pose)
            if actual.shape != expected.shape:
                raise ValueError(f"{n}: initial visual vertex set mismatch")
            error = float(np.abs(actual - expected).max())
            report[n] = dict(maximum_visual_error_m=error, vertices=len(actual))
            if error > 1e-5:
                raise ValueError(f"{n}: initial visual origin/scale mismatch: {error}")
        return dict(passed=True, bodies=report, cameras_created=0, physics_steps=0)
    finally:
        gs.destroy()

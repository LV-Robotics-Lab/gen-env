"""Measured lathe-profile partitions for bowls; full original geometry remains the gate."""

from __future__ import annotations

import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_collision_v2 as v2

STRATEGY = "bowl_partition_v1"
PROFILE_TOLERANCE_M = 2e-7
MAX_PARTS = 512
SEED = 0
THREAD_POLICY = {"openmp": 8, "openblas": 1, "mkl": 1}


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def area(polygon):
    return sum(cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))) / 2


def partition_profile(polygon):
    """Ear clipping followed only by exact, adjacent convex-polygon merges."""
    polygon = np.asarray(polygon, dtype=float)
    if not np.isfinite(polygon).all() or area(polygon) <= 1e-14:
        raise ValueError("invalid positively oriented profile")
    for i, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        for j, (c, d) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
            if j <= i or j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            signs = [
                cross(b - a, c - a),
                cross(b - a, d - a),
                cross(d - c, a - c),
                cross(d - c, b - c),
            ]
            crossed = signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0
            touching = any(
                abs(s) <= 1e-14 and (p - u) @ (p - v) <= 0
                for s, p, u, v in zip(signs, [c, d, a, b], [a, a, c, c], [b, b, d, d])
            )
            if crossed or touching:
                raise ValueError("profile has self-intersecting edges")
    indices = list(range(len(polygon)))
    triangles = []
    while len(indices) > 3:
        ear = None
        for k, i in enumerate(indices):
            a, b, c = indices[k - 1], i, indices[(k + 1) % len(indices)]
            if cross(polygon[b] - polygon[a], polygon[c] - polygon[b]) <= 1e-14:
                continue
            inside = False
            for j in indices:
                if j in (a, b, c):
                    continue
                p = polygon[j]
                if (
                    min(
                        cross(polygon[b] - polygon[a], p - polygon[a]),
                        cross(polygon[c] - polygon[b], p - polygon[b]),
                        cross(polygon[a] - polygon[c], p - polygon[c]),
                    )
                    >= -1e-14
                ):
                    inside = True
                    break
            if not inside:
                ear = k
                triangles.append([a, b, c])
                break
        if ear is None:
            raise ValueError("profile is not a simple triangulable polygon")
        indices.pop(ear)
    triangles.append(indices)
    if any(area(polygon[p]) <= 1e-14 for p in triangles):
        raise ValueError("profile triangulation has a nonpositive triangle")
    parts = triangles
    while True:
        choices = []
        for i in range(len(parts)):
            for j in range(i):
                left = set(zip(parts[i], parts[i][1:] + parts[i][:1]))
                right = set(zip(parts[j], parts[j][1:] + parts[j][:1]))
                shared = [edge for edge in left if edge[::-1] in right]
                if len(shared) != 1:
                    continue
                boundary = (left | right) - set(shared) - {edge[::-1] for edge in shared}
                links = {a: b for a, b in boundary}
                if len(links) != len(boundary):
                    continue
                first = min(links)
                merged = [first]
                while links[merged[-1]] != first and len(merged) <= len(boundary):
                    merged.append(links[merged[-1]])
                if len(merged) != len(boundary):
                    continue
                p = polygon[merged]
                if all(
                    cross(p[k] - p[k - 1], p[(k + 1) % len(p)] - p[k]) >= -1e-14
                    for k in range(len(p))
                ):
                    length = np.linalg.norm(polygon[shared[0][0]] - polygon[shared[0][1]])
                    choices.append((-length, j, i, merged))
        if not choices:
            break
        _, j, i, merged = min(choices)
        parts = [p for k, p in enumerate(parts) if k not in (i, j)] + [merged]
    total = sum(area(polygon[p]) for p in parts)
    if abs(total - area(polygon)) > max(1e-14, area(polygon) * 1e-10):
        raise ValueError("profile partition did not preserve area")
    return parts


def measure_profile(visual):
    """Reject non-lathe sources; map every solid vertex and triangle to the measured grid."""
    if not np.isfinite(visual.vertices).all():
        raise ValueError("nonfinite original visual")
    components = trimesh.Trimesh(visual.vertices, visual.faces, process=True).split(
        only_watertight=False
    )
    solids, skipped = [], []
    for i, component in enumerate(components):
        singular = np.linalg.svd(component.vertices - component.vertices.mean(0), compute_uv=False)
        if abs(component.volume) <= 1e-12 and singular[-1] <= 1e-7:
            skipped.append(
                dict(
                    component=i,
                    faces=len(component.faces),
                    mesh_sha256=v2.mesh_hash(component),
                    reason="zero-volume patch",
                )
            )
        else:
            solids.append(component)
    if len(solids) != 1 or not v2.topology_report(solids[0])["passed"]:
        raise ValueError("profile strategy requires exactly one closed nonzero solid")
    solid = solids[0]
    vertices = np.asarray(solid.vertices)
    center = (vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2
    radial = np.linalg.norm(vertices[:, :2] - center, axis=1)
    on_axis = radial <= PROFILE_TOLERANCE_M
    if np.count_nonzero(on_axis) != 2:
        raise ValueError("profile requires two measured axis endpoints")
    axis = vertices[on_axis, :2].mean(0)
    xy = vertices[:, :2] - axis
    radial = np.linalg.norm(xy, axis=1)
    mask = (np.abs(xy[:, 1]) <= PROFILE_TOLERANCE_M) & (xy[:, 0] >= -PROFILE_TOLERANCE_M)
    edges = solid.edges_unique[np.all(mask[solid.edges_unique], axis=1)]
    adjacent = {int(i): [] for i in np.flatnonzero(mask)}
    for a, b in edges:
        adjacent[int(a)].append(int(b))
        adjacent[int(b)].append(int(a))
    endpoints = [i for i, neighbors in adjacent.items() if len(neighbors) == 1]
    if len(endpoints) != 2 or any(len(n) not in (1, 2) for n in adjacent.values()):
        raise ValueError("measured +X meridian is not one simple path")
    order = [min(endpoints, key=lambda i: vertices[i, 2])]
    previous = None
    while True:
        choices = [i for i in adjacent[order[-1]] if i != previous]
        if not choices:
            break
        previous = order[-1]
        order.append(choices[0])
        if len(order) > len(adjacent):
            raise ValueError("cyclic meridian")
    if len(order) != len(adjacent) or not np.all(on_axis[[order[0], order[-1]]]):
        raise ValueError("meridian does not cover the source path")
    profile = np.c_[radial[order], vertices[order, 2]]
    profile[[0, -1], 0] = 0.0
    if area(profile) < 0:
        profile, order = profile[::-1], order[::-1]
    distances = np.linalg.norm(
        np.c_[radial, vertices[:, 2]][:, None, :] - profile[None, :, :], axis=2
    )
    profile_indices = np.argmin(distances, axis=1)
    fit_error = float(distances[np.arange(len(vertices)), profile_indices].max())
    ring_count = len(profile) - 2
    angular_count, remainder = divmod(len(vertices) - 2, ring_count)
    if fit_error > PROFILE_TOLERANCE_M or remainder or angular_count < 8:
        raise ValueError("source is not a uniformly sampled measured lathe")
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    angular_indices = np.rint(angles * angular_count / (2 * np.pi)).astype(int) % angular_count
    reconstruction = np.c_[
        profile[profile_indices, 0] * np.cos(angular_indices * 2 * np.pi / angular_count) + axis[0],
        profile[profile_indices, 0] * np.sin(angular_indices * 2 * np.pi / angular_count) + axis[1],
        profile[profile_indices, 1],
    ]
    reconstruction_error = float(np.linalg.norm(vertices - reconstruction, axis=1).max())
    if reconstruction_error > PROFILE_TOLERANCE_M:
        raise ValueError("source departs from measured rotational grid")
    lookup = {}
    for i, (p, a) in enumerate(zip(profile_indices, angular_indices)):
        key = (int(p), -1 if on_axis[i] else int(a))
        if key in lookup:
            raise ValueError("duplicate source profile/angle grid point")
        lookup[key] = i
    for p in range(len(profile)):
        expected = [-1] if profile[p, 0] == 0 else range(angular_count)
        if any((p, a) not in lookup for a in expected):
            raise ValueError("source profile ring has missing vertices")
    for face in solid.faces:
        p, a = profile_indices[face], angular_indices[face]
        if np.ptp(p) != 1:
            raise ValueError("source triangles cross measured profile bands")
        active = a[~on_axis[face]]
        if len(active) > 1:
            steps = np.abs(active[:, None] - active[None, :])
            if np.minimum(steps, angular_count - steps).max() > 1:
                raise ValueError("source triangle crosses measured angular cells")
    if len(solid.faces) != 2 * ring_count * angular_count:
        raise ValueError("source triangle grid is incomplete")
    partitions = partition_profile(profile)
    measured = dict(
        axis_xy_m=axis.tolist(),
        profile_rz_m=profile.tolist(),
        meridian_vertex_indices=order,
        profile_partitions=partitions,
        angular_count=angular_count,
        source_vertex_profile_indices=profile_indices.tolist(),
        source_vertex_angle_indices=angular_indices.tolist(),
        source_solid_sha256=v2.mesh_hash(solid),
        source_solid_faces=len(solid.faces),
        source_solid_vertices=len(solid.vertices),
        fit_error_m=fit_error,
        reconstruction_error_m=reconstruction_error,
        fit_limit_m=PROFILE_TOLERANCE_M,
        profile_area_m2=area(profile),
        partition_area_m2=sum(area(profile[p]) for p in partitions),
        skipped_components=skipped,
    )
    return solid, lookup, measured


def convex_mesh(points):
    points = np.unique(np.asarray(points), axis=0)
    envelope = ConvexHull(points)
    faces = envelope.simplices.copy()
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    )
    reverse = np.sum(normals * envelope.equations[:, :3], axis=1) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    mesh = trimesh.Trimesh(points, faces, process=False)
    mesh.remove_unreferenced_vertices()
    if not prep.closed_convex(mesh):
        raise ValueError("generated partition is not a closed positive convex solid")
    return mesh


def candidates(original_faces):
    return [
        dict(
            id=f"bowl_partition_{i}",
            strategy=STRATEGY,
            max_angular_step=step,
            max_convex_hull=MAX_PARTS,
            original_faces=original_faces,
        )
        for i, step in enumerate((4, 3))
    ]


def generate(solid, lookup, measured, candidate):
    step = int(candidate["max_angular_step"])
    if step not in (3, 4):
        raise ValueError("unsupported angular partition step")
    angular_count = measured["angular_count"]
    boundaries = list(range(0, angular_count, step)) + [angular_count]
    profiles = measured["profile_partitions"]
    if len(profiles) * (len(boundaries) - 1) > min(MAX_PARTS, candidate["max_convex_hull"]):
        raise ValueError("profile/angular partition exceeds total part budget")
    parts, records = [], []
    for p, polygon in enumerate(profiles):
        for a, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            source_indices = []
            for i in polygon:
                if (i, -1) in lookup:
                    source_indices.append(lookup[i, -1])
                else:
                    source_indices.extend(
                        lookup[i, k % angular_count] for k in range(start, stop + 1)
                    )
            source_indices = sorted(set(source_indices))
            mesh = convex_mesh(solid.vertices[source_indices])
            records.append(
                dict(
                    profile_partition=p,
                    angular_cell=a,
                    angular_indices_inclusive=[start, stop],
                    source_vertex_indices=source_indices,
                    mesh_sha256=v2.mesh_hash(mesh),
                )
            )
            parts.append(mesh)
    return parts, records


def worker(request):
    started = time.monotonic()
    np.random.seed(SEED)
    source, out = Path(request["source"]), Path(request["output_dir"])
    if prep.lib.sha256(source) != request["source_sha256"]:
        raise ValueError("original visual file hash mismatch")
    with np.load(source) as data:
        visual = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    if v2.mesh_hash(visual) != request["identity"]["visible_mesh_sha256"]:
        raise ValueError("original visual array hash mismatch")
    candidate = request["candidate"]
    if candidate["strategy"] != STRATEGY:
        raise ValueError("unknown bowl partition strategy")
    inputs, part_dir = out / "inputs", out / "parts"
    inputs.mkdir()
    part_dir.mkdir()
    solid, lookup, measured = measure_profile(visual)
    profile_path = inputs / "measured_profile.json"
    prep.clip.write_json(profile_path, measured)
    input_hashes = {str(profile_path.relative_to(out)): prep.lib.sha256(profile_path)}
    parts, records = generate(solid, lookup, measured, candidate)
    names, part_hashes = [], {}
    for i, mesh in enumerate(parts):
        path = part_dir / f"part_{i:03d}.obj"
        mesh.export(path, digits=17)
        names.append(path.relative_to(out).as_posix())
        part_hashes[names[-1]] = prep.lib.sha256(path)
        reloaded = trimesh.load(path, force="mesh", process=False)
        if prep.lib.sha256(path) != part_hashes[names[-1]]:
            raise ValueError("serialized partition changed while loading")
        records[i]["serialized_mesh_sha256"] = v2.mesh_hash(reloaded)
        parts[i] = reloaded
        records[i].update(path=names[-1], file_sha256=part_hashes[names[-1]])
    partition_path = inputs / "generated_partitions.json"
    prep.clip.write_json(partition_path, records)
    input_hashes[str(partition_path.relative_to(out))] = prep.lib.sha256(partition_path)
    quality = v2.quality_gate(visual, parts, request["surface"])
    for name, expected in part_hashes.items():
        if prep.lib.sha256(out / name) != expected:
            raise ValueError("serialized partition changed during qualification")
    for name, expected in input_hashes.items():
        if prep.lib.sha256(out / name) != expected:
            raise ValueError("measured partition input changed during qualification")
    generation = dict(
        status="recorded",
        strategy=STRATEGY,
        seed=SEED,
        thread_policy=THREAD_POLICY,
        observed_thread_environment={
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        candidate=candidate,
        original_visual_sha256=v2.mesh_hash(visual),
        implementation_sha256=prep.lib.sha256(Path(__file__)),
        versions={n: version(n) for n in ("numpy", "scipy", "trimesh")},
        quality_version=v2.QUALITY_VERSION,
        source_profile=measured,
        partition_record_sha256=input_hashes[str(partition_path.relative_to(out))],
        part_sha256=part_hashes,
        coacd_calls=[],
        native_mass_properties_modified=False,
    )
    return dict(
        passed=bool(quality["passed"]),
        status="passed" if quality["passed"] else "quality_rejected",
        parts=names,
        parts_sha256=part_hashes,
        quality=quality,
        processed_input_sha256=input_hashes,
        generation_provenance=generation,
        operation="generated_candidate",
        configured_options=candidate,
        effective_options=candidate,
        coacd_executed=False,
        request=request["identity"],
        elapsed_s=time.monotonic() - started,
    )


def main():
    import resource
    import sys
    import traceback

    resource.setrlimit(resource.RLIMIT_AS, (v2.WORKER_MEMORY_BYTES, v2.WORKER_MEMORY_BYTES))
    request = prep.lib.read_json(Path(sys.argv[1]))
    started = time.monotonic()
    try:
        result = worker(request)
    except Exception as exc:
        result = dict(
            passed=False,
            status="worker_error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )
        traceback.print_exc()
    prep.clip.write_json(Path(request["output_dir"]) / "result.json", result)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()

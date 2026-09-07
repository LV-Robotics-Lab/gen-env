"""Canonical geometry and deterministic constraint sampling for text_repair_v1."""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from self_improving.sim_adapters.genesis import build_scene as geom
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.physics_math import rotation

PROFILE = "text_repair_v1"
PRIORS = {
    "apple": ("max", 0.04, 0.12, 0.07),
    "cup": ("height", 0.06, 0.25, 0.12),
    "bowl": ("diameter", 0.08, 0.40, 0.20),
    "table": ("height", 0.50, 1.20, 0.75),
}


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return np.roll(q, 1).tolist()


def transform(points, pose):
    return np.asarray(points) @ rotation(pose["orientation_wxyz"]).T + pose["position"]


def inverse(points, pose):
    return (np.asarray(points) - pose["position"]) @ rotation(pose["orientation_wxyz"])


def bounds(points):
    return np.stack([np.min(points, axis=0), np.max(points, axis=0)])


def footprint(points):
    xy = np.asarray(points)[:, :2]
    return xy[ConvexHull(xy).vertices]


def clearance(polygon, points):
    polygon, points = np.asarray(polygon), np.asarray(points)
    values = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        e = b - a
        values.append(
            (e[0] * (points[:, 1] - a[1]) - e[1] * (points[:, 0] - a[0])) / np.linalg.norm(e)
        )
    return float(np.min(values))


def support_metrics(asset, pose, parent, parent_pose):
    surface = parent["surface"]
    local = inverse(transform(asset["hull"], pose), parent_pose)
    hull = footprint(local)
    polygon = np.asarray(surface["polygon_xy_m"])
    area = geom.polygon_area(hull)
    ratio = geom.polygon_area(geom.clip_polygon(hull, polygon)) / area
    if not geom.fits_surface(surface, hull, margin=0.0):
        return -1.0, min(1.0, ratio)
    return clearance(polygon, hull), min(1.0, ratio)


def shrink_region(polygon, hull, margin):
    """Minkowski erosion for a convex measured support polygon and oriented footprint."""
    polygon, hull = np.asarray(polygon), np.asarray(hull)
    region = polygon.copy()
    # Clip against n.p >= n.a + margin - min(n.hull).
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        offset = float(n @ a + margin - np.min(hull @ n))
        source, region = region, []
        if len(source) < 3:
            return np.empty((0, 2))
        for p, q in zip(source, np.roll(source, -1, axis=0)):
            dp, dq = p @ n - offset, q @ n - offset
            if dp >= -1e-12:
                region.append(p)
            if (dp >= 0) != (dq >= 0):
                region.append(p + (q - p) * dp / (dp - dq))
        region = np.asarray(region)
    return region


def uniform_polygon(polygon, rng):
    polygon = np.asarray(polygon)
    triangles = [polygon[[0, i, i + 1]] for i in range(1, len(polygon) - 1)]
    areas = np.array([geom.polygon_area(t) for t in triangles])
    t = triangles[int(rng.choice(len(triangles), p=areas / areas.sum()))]
    u, v = rng.random(2)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    return t[0] + u * (t[1] - t[0]) + v * (t[2] - t[0])


def overlap_depth(a, b):
    """Positive interior radius of convex intersection; touching is not penetration."""
    ba, bb = bounds(a), bounds(b)
    overlap = np.minimum(ba[1], bb[1]) - np.maximum(ba[0], bb[0])
    if np.min(overlap) <= 1e-9:
        return 0.0
    eq = np.concatenate([ConvexHull(a).equations, ConvexHull(b).equations])
    result = linprog(
        [0, 0, 0, -1],
        A_ub=np.c_[eq[:, :3], np.ones(len(eq))],
        b_ub=-eq[:, 3],
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if result.status == 2:  # Proven disjoint convex interiors.
        return 0.0
    if not result.success:
        raise ValueError(f"convex intersection solver failed: {result.message}")
    return max(0.0, float(result.x[3]))


def mesh_intersects_convex(vertices, faces, convex, epsilon=1e-6):
    """Test actual closed mesh surfaces/volume against a convex interior, not its outer hull."""
    import trimesh

    vertices, convex = np.asarray(vertices), np.asarray(convex)
    triangles = vertices[np.asarray(faces)]
    lo, hi = np.min(convex, axis=0), np.max(convex, axis=0)
    if np.min(np.minimum(vertices.max(0), hi) - np.maximum(vertices.min(0), lo)) <= epsilon:
        return False
    equations = ConvexHull(convex).equations
    nearby = triangles[
        np.all(triangles.max(1) >= lo + epsilon, axis=1)
        & np.all(triangles.min(1) <= hi - epsilon, axis=1)
    ]
    for triangle in nearby:
        polygon = list(triangle)
        for plane in equations:
            clipped = []
            for i, point in enumerate(polygon):
                previous = polygon[i - 1]
                distance = point @ plane[:3] + plane[3] + epsilon
                before = previous @ plane[:3] + plane[3] + epsilon
                if (distance <= 0) != (before <= 0):
                    clipped.append(previous + (point - previous) * before / (before - distance))
                if distance <= 0:
                    clipped.append(point)
            polygon = clipped
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            area = sum(
                np.linalg.norm(np.cross(polygon[i] - polygon[0], polygon[i + 1] - polygon[0]))
                for i in range(1, len(polygon) - 1)
            )
            if area > 1e-16:
                return True
    # Full containment has no surface crossing. Check an interior point with a distance guard.
    center = convex.mean(0)
    if np.all(equations[:, :3] @ center + equations[:, 3] < -epsilon):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        if mesh.contains([center])[0]:
            return bool(trimesh.proximity.closest_point(mesh, [center])[1][0] > epsilon)
    return False


def collision_pair(a, pose_a, b, pose_b):
    native_a = a.get("collision", {}).get("method") == "native_fixed_triangle_mesh"
    native_b = b.get("collision", {}).get("method") == "native_fixed_triangle_mesh"
    if native_a and native_b:
        raise ValueError("overlapping nonconvex fixed bodies require a mesh pair probe")
    if native_a or native_b:
        mesh_asset, mesh_pose, convex_asset, convex_pose = (
            (a, pose_a, b, pose_b) if native_a else (b, pose_b, a, pose_a)
        )
        if not mesh_asset.get("collision_meshes"):
            raise ValueError("native nonconvex geometry is missing frozen triangle connectivity")
        for part in mesh_asset["collision_meshes"]:
            for hull in convex_asset["collision_hulls"]:
                local = inverse(transform(hull, convex_pose), mesh_pose)
                if mesh_intersects_convex(part["vertices"], part["faces"], local):
                    return True
        return False
    return any(
        overlap_depth(transform(pa, pose_a), transform(pb, pose_b)) > 1e-6
        for pa in a["collision_hulls"] for pb in b["collision_hulls"]
    )


def topology(assets):
    depth, active = {}, set()

    def visit(name):
        if name in active:
            raise ValueError("support cycle")
        if name in depth:
            return depth[name]
        active.add(name)
        parent = assets[name]["support"]
        if parent != "ground" and parent not in assets:
            raise ValueError("unknown support parent")
        depth[name] = 0 if parent == "ground" else 1 + visit(parent)
        active.remove(name)
        return depth[name]

    for name in assets:
        visit(name)
    return sorted(assets, key=lambda n: (depth[n], -assets[n]["diagonal_m"], n))


def graph(document, assets):
    # Numeric dimensions are parsed separately; numeric spatial distances remain unsupported.
    checked = copy.deepcopy(document)
    for row in checked["relations"]:
        if row["relation"] == "on":
            row["evidence"] = ""
    parents = spatial.relations(checked)
    for n, a in assets.items():
        if n in parents:
            a.update(support=parents[n], support_source="explicit_text")
        else:
            candidates = [
                p
                for p, b in assets.items()
                if p != n
                and b.get("surface")
                and b["diagonal_m"] > a["diagonal_m"] * 1.25
                and np.all(
                    np.sort(b["bbox_size_m"][:2])
                    > np.sort(a["bbox_size_m"][:2]) + 2 * a["margin_m"]
                )
            ]
            parent = (
                max(candidates, key=lambda p: (assets[p]["surface"]["area_m2"], p))
                if candidates and a["category"] != "table" and not a["fixed"]
                else "ground"
            )
            a.update(support=parent, support_source="category_geometry_prior")
        if a["fixed"] and a["support"] != "ground":
            raise ValueError("fixed support child is forbidden")
    order = topology(assets)
    for a in assets.values():
        if a["support"] != "ground" and assets[a["support"]].get("surface") is None:
            raise ValueError("declared parent has no measured support surface")
    return dict(
        profile=PROFILE,
        order=order,
        relations=document["relations"],
        support={n: a["support"] for n, a in assets.items()},
        frame=dict(up="+Z", right="+X", front="-Y", units="m"),
    )


def subtree(assets, root):
    names = {root}
    while True:
        expanded = names | {n for n, a in assets.items() if a["support"] in names}
        if expanded == names:
            return names
        names = expanded


def move_subtree(assets, poses, root, replacement):
    out = copy.deepcopy(poses)
    old = poses[root]
    delta = rotation(replacement["orientation_wxyz"]) @ rotation(old["orientation_wxyz"]).T
    for n in subtree(assets, root):
        if n in poses:
            out[n] = dict(
                position=(
                    delta @ (np.asarray(poses[n]["position"]) - old["position"])
                    + replacement["position"]
                ).tolist(),
                orientation_wxyz=quat(delta @ rotation(poses[n]["orientation_wxyz"])),
            )
    out[root] = replacement
    return out


def geometric_checks(assets, poses, relations, extent=4.0, initial_buffer=True):
    errors, boxes = [], {}
    for n, pose in poses.items():
        a = assets[n]
        vertices = transform(a["hull"], pose)
        boxes[n] = bounds(vertices)
        if np.any(np.abs(vertices[:, :2]) > extent / 2 + 1e-9) or vertices[:, 2].min() < -1e-6:
            errors.append(f"{n}:scene_bounds")
        parent = a["support"]
        if parent in poses:
            margin, ratio = support_metrics(a, pose, assets[parent], poses[parent])
            required = a["margin_m"] + (a["buffer_m"] if initial_buffer else 0.0)
            if margin < required - 1e-8 or ratio < 1 - 1e-7:
                errors.append(f"{n}:support_geometry")
    for relation in relations:
        x, y = relation["source"], relation["target"]
        if relation["relation"] != "on" and x in poses and y in poses:
            if not spatial.relation_ok(relation["relation"], boxes[x], boxes[y]):
                errors.append(f"{x}:{relation['relation']}:{y}")
    names = list(poses)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ba, bb = boxes[a], boxes[b]
            if np.min(np.minimum(ba[1], bb[1]) - np.maximum(ba[0], bb[0])) <= 1e-8:
                continue
            if collision_pair(assets[a], poses[a], assets[b], poses[b]):
                errors.append(f"{a}:collision:{b}")
    return errors


def sample_object(assets, poses, name, relations, preferences, seed, retry, extent=4.0):
    token = hashlib.sha256(f"{seed}:{name}:{retry}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
    a = assets[name]
    candidates = []
    for i in range(50):
        yaw = float(rng.uniform(0, 2 * math.pi))
        q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        oriented = np.asarray(a["hull"]) @ rotation(q).T
        hull = footprint(oriented)
        parent = a["support"]
        if parent == "ground":
            polygon = np.array(
                [
                    [-extent / 2, -extent / 2],
                    [extent / 2, -extent / 2],
                    [extent / 2, extent / 2],
                    [-extent / 2, extent / 2],
                ]
            )
            z = a.get("bottom_offset_m", -oriented[:, 2].min())
        else:
            surface = assets[parent]["surface"]
            polygon = np.asarray(surface["polygon_xy_m"])
            z = surface["z_m"] + a.get("bottom_offset_m", -oriented[:, 2].min())
        region = shrink_region(polygon, hull, a["margin_m"] + a["buffer_m"])
        entry = dict(candidate=i, yaw_rad=yaw)
        candidates.append(entry)
        if len(region) < 3 or geom.polygon_area(region) < 1e-12:
            entry.update(rejected=["empty_valid_region"])
            continue
        xy = uniform_polygon(region, rng)
        pose = dict(position=[*xy.tolist(), float(z)], orientation_wxyz=q)
        if parent != "ground":
            pose = dict(
                position=transform([pose["position"]], poses[parent])[0].tolist(),
                orientation_wxyz=quat(rotation(poses[parent]["orientation_wxyz"]) @ rotation(q)),
            )
        trial = (
            move_subtree(assets, poses, name, pose)
            if name in poses
            else dict(poses, **{name: pose})
        )
        errors = geometric_checks(assets, trial, relations, extent)
        entry.update(pose=pose, rejected=errors)
        if errors:
            continue
        low, high = region.min(0), region.max(0)
        desired = (low + high) / 2
        # Drop soft preferences after five retries, never hard relations or physical margin.
        for pref in preferences if retry < 5 else []:
            if pref.get("object_id") == name and pref.get("region") in spatial.REGIONS:
                region_name = pref["region"]
                if region_name != "center":
                    axis = 0 if region_name in {"left", "right"} else 1
                    fraction = 1 / 6 if region_name in {"left", "front"} else 5 / 6
                    desired[axis] = low[axis] + fraction * (high[axis] - low[axis])
        text_cost = float(np.linalg.norm((xy - desired) / np.maximum(high - low, 1e-6)))
        edge_cost = 1 / (1 + max(0.0, clearance(region, [xy])) / a["diagonal_m"])
        other = [
            p
            for n, p in trial.items()
            if n != name and n not in subtree(assets, name) and n != parent
        ]
        spacing = min(
            (np.linalg.norm(np.asarray(p["position"]) - pose["position"]) for p in other),
            default=extent,
        )
        entry["cost"] = 4 * text_cost + edge_cost + 1 / (1 + spacing / a["diagonal_m"])
    legal = sorted([c for c in candidates if not c["rejected"]], key=lambda c: c["cost"])[:5]
    selected = legal[int(rng.integers(len(legal)))] if legal else None
    return selected, dict(
        object_id=name,
        retry=retry,
        candidates=candidates,
        top_k=[c["candidate"] for c in legal],
        selected=None if selected is None else selected["candidate"],
    )

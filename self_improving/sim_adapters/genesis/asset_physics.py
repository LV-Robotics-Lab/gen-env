"""Pure evidence validation for native asset scenes; no Genesis or OpenXSim imports."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from self_improving.sim_adapters.genesis import build_scene as geometry
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.physics_math import angle, rotation

SCHEMA = "genenv.asset_physics.v1"
DEFAULTS = dict(
    dt=0.004,
    steps=1000,
    substeps=1,
    seed=0,
    window_s=0.5,
    translation_m=0.001,
    rotation_deg=0.5,
    speed_mps=0.01,
    angular_speed_radps=0.05,
    support_fraction=0.8,
    support_force_n=1e-6,
    penetration_m=0.001,
    margin_m=0.02,
    surface_tilt_deg=0.5,
    gravity=[0.0, 0.0, -9.81],
    constraint_timeconst=0.001,
    iterations=50,
    ls_iterations=50,
    tolerance=1e-8,
    use_hibernation=False,
    precision="32",
    backend="cpu",
    constraint_solver="Newton",
)


class PhysicsFailure(ValueError):
    """An observed physical violation, as opposed to missing or invalid evidence."""


def settings(profile):
    if profile not in ("baseline", "half_dt"):
        raise ValueError("unknown physics profile")
    return dict(DEFAULTS, **({"dt": 0.002, "steps": 2000} if profile == "half_dt" else {}))


def finite(value, shape, label):
    data = np.asarray(value, dtype=float)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"invalid {label}")
    return data


def world_vertices(body, state):
    points = np.asarray(body["visual_hull_local_m"])
    return points @ rotation(state["orientation_wxyz"]).T + state["position"]


def native_points(points, body, state):
    """World -> original native loading frame (not inertial or AABB center)."""
    native = body["native_pose"]
    return (points - state["position"]) @ rotation(state["orientation_wxyz"]) @ rotation(
        native["orientation_wxyz"]
    ).T + native["position"]


def validate_row(row, step, data):
    ids = set(data["bodies"]) | {"ground"}
    if (
        row["step"] != step
        or not math.isfinite(row["time_s"])
        or abs(row["time_s"] - step * data["settings"]["dt"]) > 1e-9
    ):
        raise ValueError("non-sequential trajectory")
    if set(row["objects"]) != ids:
        raise ValueError("trajectory object mismatch")
    if row["contact_phase"] != ("initial_detection" if step == 0 else "solved_step"):
        raise ValueError("invalid contact phase")
    for state in row["objects"].values():
        for key in ("position", "velocity", "angular_velocity"):
            finite(state[key], (3,), key)
        rotation(state["orientation_wxyz"])
    for c in row["contacts"]:
        if c["a"] not in ids or c["b"] not in ids:
            raise ValueError("unknown contact object")
        if c["a"] == c["b"]:
            raise ValueError("articulated or self contact unsupported")
        finite(c["position"], (3,), "contact position")
        normal = finite(c["normal"], (3,), "contact normal")
        if abs(np.linalg.norm(normal) - 1) > 1e-4:
            raise ValueError("invalid contact normal length")
        if not math.isfinite(c["penetration"]) or c["penetration"] < 0:
            raise ValueError("invalid penetration")
        for key in ("geom_a", "geom_b", "link_a", "link_b"):
            if not isinstance(c[key], int) or c[key] < 0:
                raise ValueError("invalid contact index")
        if step == 0:
            if c["force_a"] is not None or c["force_b"] is not None:
                raise ValueError("initial force must be unavailable")
        else:
            a = finite(c["force_a"], (3,), "force_a")
            b = finite(c["force_b"], (3,), "force_b")
            if not np.allclose(a, -b, atol=1e-7, rtol=1e-6):
                raise ValueError("contact force pair inconsistent")


def initial_checks(data, loaded, row):
    validate_row(row, 0, data)
    checks = []
    penetration = max((c["penetration"] for c in row["contacts"]), default=0.0)
    checks.append(
        dict(
            name="initial.penetration",
            passed=penetration <= 0.001,
            maximum_m=penetration,
            limit_m=0.001,
        )
    )
    for name, body in loaded.items():
        state = row["objects"][name]
        expected = np.array(body["native_pose"]["position"]) + data["bodies"][name]["translation_m"]
        if (
            np.max(np.abs(np.array(state["position"]) - expected)) > 1e-5
            or angle(state["orientation_wxyz"], body["native_pose"]["orientation_wxyz"]) > 1e-4
        ):
            raise ValueError(f"{name}: initial pose differs from frozen input")
        points = world_vertices(body, row["objects"][name])
        z = float(points[:, 2].min())
        checks.append(
            dict(name=f"{name}.initial_ground", passed=z >= -0.001, minimum_z_m=z, limit_m=-0.001)
        )
    return checks


def evaluate(data, loaded, rows):
    """Evaluate complete trace. A legitimate zero-contact sample is not missing data."""
    cfg = data["settings"]
    if len(rows) != cfg["steps"] + 1 or set(loaded) != set(data["bodies"]):
        raise ValueError("missing trajectory records or loaded bodies")
    for i, row in enumerate(rows):
        validate_row(row, i, data)
    checks = initial_checks(data, loaded, rows[0])

    def check(name, passed, **metrics):
        checks.append(dict(name=name, passed=bool(passed), **metrics))

    window = [r for r in rows if r["time_s"] >= cfg["steps"] * cfg["dt"] - cfg["window_s"] - 1e-9]
    maximum = max((c["penetration"] for r in rows for c in r["contacts"]), default=0.0)
    check(
        "penetration",
        maximum <= cfg["penetration_m"],
        maximum_m=maximum,
        limit_m=cfg["penetration_m"],
    )
    allowed = {frozenset((n, b["support"])) for n, b in data["bodies"].items()}
    unexpected = sorted(
        {
            tuple(sorted((c["a"], c["b"])))
            for r in window
            for c in r["contacts"]
            if frozenset((c["a"], c["b"])) not in allowed
        }
    )
    check("declared_contacts", not unexpected, unexpected_pairs=unexpected)
    cached = {}
    for row in window:
        cached[row["step"]] = {n: world_vertices(b, row["objects"][n]) for n, b in loaded.items()}
    for name, spec in data["bodies"].items():
        body = loaded[name]
        final = rows[-1]["objects"][name]
        states = [r["objects"][name] for r in window]
        displacement = max(
            float(np.linalg.norm(np.array(s["position"]) - final["position"])) for s in states
        )
        degrees = max(angle(s["orientation_wxyz"], final["orientation_wxyz"]) for s in states)
        speed = max(float(np.linalg.norm(s["velocity"])) for s in states)
        angular = max(float(np.linalg.norm(s["angular_velocity"])) for s in states)
        if not spec["fixed"]:
            check(
                f"{name}.settled",
                displacement <= cfg["translation_m"]
                and degrees <= cfg["rotation_deg"]
                and speed <= cfg["speed_mps"]
                and angular <= cfg["angular_speed_radps"],
                displacement_m=displacement,
                rotation_deg=degrees,
                speed_mps=speed,
                angular_speed_radps=angular,
            )
            hits = 0
            for row in window:
                force = 0.0
                for c in row["contacts"]:
                    if {c["a"], c["b"]} == {name, spec["support"]}:
                        force += c["force_a" if c["a"] == name else "force_b"][2]
                hits += force > cfg["support_force_n"]
            fraction = hits / len(window)
            check(
                f"{name}.support",
                fraction >= cfg["support_fraction"],
                target=spec["support"],
                fraction=fraction,
                limit=cfg["support_fraction"],
            )
        # All frames, including transient motion, must remain above the environment plane.
        minimum_z = min(float(world_vertices(body, r["objects"][name])[:, 2].min()) for r in rows)
        check(f"{name}.ground", minimum_z >= -cfg["penetration_m"], minimum_z_m=minimum_z)
        target = spec["support"]
        if target == "ground":
            continue
        surface = data["bodies"][target]["surface"]
        fits, max_tilt, minimum_margin = True, 0.0, float("inf")
        for row in window:
            target_state = row["objects"][target]
            parent = loaded[target]
            rdelta = (
                rotation(target_state["orientation_wxyz"])
                @ rotation(parent["native_pose"]["orientation_wxyz"]).T
            )
            tilt = math.degrees(math.acos(np.clip(rdelta[2, 2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
            local = native_points(cached[row["step"]][name], parent, target_state)
            xy = local[:, :2]
            hull = xy[ConvexHull(xy).vertices]
            fits &= geometry.fits_surface(surface, hull, margin=cfg["margin_m"])
            polygon = np.asarray(surface["polygon_xy_m"])
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = b - a
                distances = (
                    edge[0] * (hull[:, 1] - a[1]) - edge[1] * (hull[:, 0] - a[0])
                ) / np.linalg.norm(edge)
                minimum_margin = min(minimum_margin, float(distances.min()))
        check(
            f"{name}.support_model",
            max_tilt <= cfg["surface_tilt_deg"],
            maximum_tilt_deg=max_tilt,
            limit_deg=cfg["surface_tilt_deg"],
        )
        check(
            f"{name}.support_geometry",
            fits,
            margin_m=cfg["margin_m"],
            minimum_margin_m=minimum_margin,
            target=target,
        )
    for relation in data["relations"]:
        if relation["relation"] == "on":
            continue
        valid = True
        for row in window:
            bounds = []
            for name in (relation["source"], relation["target"]):
                points = cached[row["step"]][name]
                bounds.append(np.array([points.min(axis=0), points.max(axis=0)]))
            valid &= spatial.relation_ok(relation["relation"], *bounds)
        check(f"relation.{relation['source']}.{relation['relation']}.{relation['target']}", valid)
    return checks

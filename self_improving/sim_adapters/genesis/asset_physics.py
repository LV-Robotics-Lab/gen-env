"""Pure evidence validation for native asset scenes; no Genesis or OpenXSim imports."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from self_improving.sim_adapters.genesis import build_scene as geometry
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.physics_math import (
    angle,
    effective_speed,
    free_fall_step,
    longest_run,
    rotation,
    stiffness_floor,
    sweep_radius,
)

SCHEMA = "genenv.asset_physics.v2"
DEFAULTS = dict(
    dt=0.004,
    steps=1000,
    substeps=1,
    seed=0,
    window_s=0.5,
    # Rest is decided on pose evolution. These are the primary criteria: where the body
    # ends up over the window, not how fast a single sample says it was moving.
    translation_m=0.001,
    rotation_deg=0.5,
    drift_rate_mps=0.002,
    excursion_m=0.001,
    rotation_rate_dps=1.0,
    # Effective speed max(|v|, r*|w|) is the auxiliary, weighted by sweep radius the way
    # Genesis weights its own rest test. Its limit is derived per profile in settings():
    # a shared constant is unsatisfiable at one step size and slack at the other. Motion
    # must persist for speed_run_steps consecutive samples, because an isolated sample
    # above the limit is the contact-dropout artefact rather than the body moving.
    speed_floor_multiple=1.5,
    speed_run_steps=5,
    # Dropout is a first-class criterion with its own bound, so it stops leaking into the
    # speed and support readings as if it were motion or a missing support.
    contact_dropout_max=0.05,
    # Support asks "while touching, is the declared parent holding it", so the denominator
    # is the touching samples only; the dropout bound above covers the rest.
    support_fraction=0.8,
    support_force_n=1e-6,
    penetration_m=0.001,
    margin_m=0.02,
    surface_tilt_deg=0.5,
    gravity=[0.0, 0.0, -9.81],
    # Genesis clamps constraint_timeconst up to twice the step and warns. The former
    # 0.001 was silently clamped to that floor -- the stiffest the step allows and the
    # least stable -- which is what drove the contact-dropout limit cycle.
    constraint_timeconst=0.05,
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


NUMERICAL_KEYS = frozenset(
    {"dt", "steps", "substeps", "constraint_timeconst", "iterations", "ls_iterations",
     "tolerance", "contact_solref", "contact_solimp"}
)


def settings(profile, numerics=None):
    """Acceptance thresholds for a profile, optionally on a different numerical footing.

    `numerics` may change how the trajectory is computed and never what counts as passing:
    only NUMERICAL_KEYS are accepted, so a caller searching for a workable solve cannot
    reach the thresholds it is being judged against. Everything derived from the step size
    is recomputed afterwards and re-checked by satisfiable().
    """
    if profile not in ("baseline", "half_dt"):
        raise ValueError("unknown physics profile")
    cfg = dict(DEFAULTS, **({"dt": 0.002, "steps": 2000} if profile == "half_dt" else {}))
    if numerics:
        offered = {k: v for k, v in numerics.items() if k in NUMERICAL_KEYS}
        thresholds = set(numerics) & set(DEFAULTS) - NUMERICAL_KEYS
        if thresholds:
            raise ValueError(
                f"numerics override may not set acceptance thresholds: {sorted(thresholds)}"
            )
        # Simulated duration is held fixed while the step size moves, so two runs remain
        # comparable: a shorter trace would reach the terminal window before the same
        # physical time and quietly change what "settled" is being asked about.
        seconds = cfg["steps"] * cfg["dt"]
        cfg.update(offered)
        if "dt" in offered and "steps" not in numerics:
            cfg["steps"] = round(seconds / cfg["dt"])
    # Each profile is calibrated against its own discretisation floor. Sharing one speed
    # constant across step sizes is what made the limit unsatisfiable at the coarse step.
    cfg["effective_speed_mps"] = round(cfg["speed_floor_multiple"] * free_fall_step(cfg), 6)
    return satisfiable(cfg)


def satisfiable(cfg):
    """Reject acceptance limits no resting body can meet, before any scene is blamed."""
    floor = free_fall_step(cfg)
    if cfg["effective_speed_mps"] <= floor:
        raise ValueError(
            f"speed limit {cfg['effective_speed_mps']} m/s is unsatisfiable: one free-fall "
            f"step at dt={cfg['dt']} is {floor:.5f} m/s"
        )
    if cfg["speed_run_steps"] < 2:
        raise ValueError("speed criterion must span consecutive steps, not a single sample")
    if cfg["drift_rate_mps"] * cfg["window_s"] < cfg["dt"] * floor:
        raise ValueError("drift rate limit is below one free-fall step of travel")
    if not 0.0 <= cfg["contact_dropout_max"] < 1.0:
        raise ValueError("contact dropout limit must be a fraction below one")
    stiffness = stiffness_floor(cfg)
    if cfg["constraint_timeconst"] < stiffness:
        raise ValueError(
            f"constraint_timeconst {cfg['constraint_timeconst']} s sits below the Genesis "
            f"stability floor 2*dt = {stiffness} s and would be silently clamped to it"
        )
    return cfg


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
        if not spec["fixed"]:
            radius = sweep_radius(body["visual_hull_local_m"])
            effective = [effective_speed(s, radius) for s in states]
            track = np.array([s["position"] for s in states], dtype=float)
            drift_rate = float(np.linalg.norm(track[-1] - track[0])) / cfg["window_s"]
            excursion = float(np.linalg.norm(track - track.mean(0), axis=1).max())
            rotation_rate = (
                angle(states[-1]["orientation_wxyz"], states[0]["orientation_wxyz"])
                / cfg["window_s"]
            )
            # Pose evolution decides rest; the speed run is the auxiliary that catches a
            # body creeping steadily enough to keep every pose delta inside its budget.
            run = longest_run(v >= cfg["effective_speed_mps"] for v in effective)
            check(
                f"{name}.settled",
                displacement <= cfg["translation_m"]
                and degrees <= cfg["rotation_deg"]
                and drift_rate <= cfg["drift_rate_mps"]
                and excursion <= cfg["excursion_m"]
                and rotation_rate <= cfg["rotation_rate_dps"]
                and run < cfg["speed_run_steps"],
                displacement_m=displacement,
                rotation_deg=degrees,
                drift_rate_mps=drift_rate,
                excursion_m=excursion,
                rotation_rate_dps=rotation_rate,
                effective_speed_max_mps=max(effective),
                effective_speed_limit_mps=cfg["effective_speed_mps"],
                overspeed_run_steps=run,
                overspeed_run_limit=cfg["speed_run_steps"],
                sweep_radius_m=radius,
            )
            # Naming the contact-detection dropout keeps the artefact bounded and visible
            # instead of letting it leak into the speed reading as if it were motion.
            absent = sum(
                not any(name in (c["a"], c["b"]) for c in row["contacts"]) for row in window
            )
            fraction = absent / len(window)
            check(
                f"{name}.contact_continuity",
                fraction <= cfg["contact_dropout_max"],
                dropout_fraction=fraction,
                limit=cfg["contact_dropout_max"],
            )
            # Ask whether the declared parent is the one holding the body up, over the
            # samples where it is touching anything at all. Counting the dropout samples
            # here too would charge the same artefact twice and hide which one failed.
            hits, touching = 0, 0
            for row in window:
                force, contact = 0.0, False
                for c in row["contacts"]:
                    if name not in (c["a"], c["b"]):
                        continue
                    contact = True
                    if {c["a"], c["b"]} == {name, spec["support"]}:
                        force += c["force_a" if c["a"] == name else "force_b"][2]
                touching += contact
                hits += contact and force > cfg["support_force_n"]
            fraction = hits / touching if touching else 0.0
            check(
                f"{name}.support",
                touching > 0 and fraction >= cfg["support_fraction"],
                target=spec["support"],
                fraction=fraction,
                limit=cfg["support_fraction"],
                touching_samples=touching,
                window_samples=len(window),
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

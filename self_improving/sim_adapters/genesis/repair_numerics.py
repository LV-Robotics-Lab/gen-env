"""Versioned numerical controls for text repair, independent of acceptance thresholds."""

from __future__ import annotations

import copy

import numpy as np

from self_improving.sim_adapters.genesis import repair_geometry as geo

STANDARD_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
REPAIR_PRESETS = ("legacy", "text_scene_v2")


def _name(dt_ms, tau_ms, mode):
    dt = format(dt_ms, "g").replace(".", "p")
    return f"dt{dt}ms_tau{tau_ms}ms_{mode}_v1"


_REGISTRY = {
    _name(dt, tau, mode): dict(
        numerics_profile=_name(dt, tau, mode),
        dt=dt / 1000,
        steps=round(3000 / dt),
        substeps=1,
        window_start_s=2.0,
        window_samples=round(1000 / dt),
        constraint_timeconst=tau / 1000,
        contact_solref=[tau / 1000, 1.0],
        contact_solimp=None if mode == "authored" else list(STANDARD_SOLIMP),
        max_collision_pairs=1024,
        max_contacts=4096,
    )
    for dt in (2.0, 1.0, 0.5, 0.25)
    for mode in ("authored", "standard")
    for tau in (10, 20)
}
PROFILES = ("legacy", *_REGISTRY)
CANDIDATE_PROFILES = tuple(_REGISTRY)[:12]


def configuration(profile="legacy"):
    if profile not in PROFILES:
        raise ValueError("unknown numerics profile")
    return {} if profile == "legacy" else copy.deepcopy(_REGISTRY[profile])


def half_dt_profile(profile):
    cfg = configuration(profile)
    if not cfg:
        raise ValueError("legacy numerics has no registered half-dt comparison")
    for name, candidate in _REGISTRY.items():
        if (
            candidate["dt"] == cfg["dt"] / 2
            and candidate["contact_solref"] == cfg["contact_solref"]
            and candidate["contact_solimp"] == cfg["contact_solimp"]
        ):
            return name
    raise ValueError("no registered half-dt comparison")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def apply_contact_parameters(entities, cfg):
    """Override and read back every geom, including ground, without changing material or mass."""
    audits = []
    for name, entity in entities.items():
        masses = _array(entity.get_links_mass()).copy()
        inertias = _array(entity.get_links_inertia()).copy()
        for geom in entity.geoms:
            before = _array(geom.get_sol_params()).reshape(7).copy()
            friction_before = float(_array(geom.get_friction()).reshape(()))
            requested = before.copy()
            requested[:2] = cfg["contact_solref"]
            if cfg["contact_solimp"] is not None:
                requested[2:] = cfg["contact_solimp"]
            geom.set_sol_params(requested)
            after = _array(geom.get_sol_params()).reshape(7)
            friction_after = float(_array(geom.get_friction()).reshape(()))
            if not np.allclose(after, requested, atol=1e-9, rtol=1e-6):
                raise ValueError(f"{name}: effective contact parameters differ from frozen request")
            if friction_after != friction_before:
                raise ValueError(f"{name}: numerical profile changed friction")
            audits.append(dict(
                object_id=name,
                geom_index=int(geom.idx),
                sol_params_before=before.tolist(),
                sol_params_requested=requested.tolist(),
                sol_params_after=after.tolist(),
                friction_before=friction_before,
                friction_after=friction_after,
            ))
        if not np.array_equal(masses, _array(entity.get_links_mass())) or not np.array_equal(
            inertias, _array(entity.get_links_inertia())
        ):
            raise ValueError(f"{name}: numerical profile changed mass or inertia")
    return audits


def check_capacity(solver, cfg, count):
    """Genesis owns broadphase/island overflow checks; also reject excessive exported contacts."""
    solver.check_errno()
    if count > cfg["max_contacts"]:
        raise ValueError("contact capacity overflow in exported trajectory")


def diagnostics(data, rows):
    """Explain support gaps and speed spikes using the same frozen terminal window."""
    cfg = data["settings"]
    window = [r for r in rows if r["time_s"] > cfg["window_start_s"] + 1e-9]
    result = {}
    for name, asset in data["assets"].items():
        touching, supported, fast, com_z, bottom_z = [], [], [], [], []
        for row in window:
            state = row["objects"][name]
            contacts = [c for c in row["contacts"] if name in (c["a"], c["b"])]
            force = sum(
                c["force_a" if c["a"] == name else "force_b"][2]
                for c in contacts if {c["a"], c["b"]} == {name, asset["support"]}
            )
            touching.append(bool(contacts))
            supported.append(force > cfg["support_force_n"])
            fast.append(max(
                np.linalg.norm(state["velocity"]),
                asset["radius_m"] * np.linalg.norm(state["angular_velocity"]),
            ) >= cfg["effective_speed_mps"])
            com_z.append(float(state["com_position"][2]))
            bottom_z.append(float(min(
                geo.transform(hull, state)[:, 2].min() for hull in asset["collision_hulls"]
            )))
        longest = current = 0
        for hit in supported:
            current = 0 if hit else current + 1
            longest = max(longest, current)
        result[name] = dict(
            window_samples=len(window),
            contact_samples=sum(touching),
            no_contact_samples=sum(not hit for hit in touching),
            no_support_samples=sum(not hit for hit in supported),
            overspeed_samples=sum(bool(hit) for hit in fast),
            overspeed_no_contact_samples=sum(bool(a and not b) for a, b in zip(fast, touching)),
            overspeed_no_support_samples=sum(bool(a and not b) for a, b in zip(fast, supported)),
            longest_no_support_s=longest * cfg["dt"],
            minimum_com_height_m=min(com_z, default=None),
            maximum_com_height_m=max(com_z, default=None),
            minimum_collision_height_m=min(bottom_z, default=None),
            maximum_collision_height_m=max(bottom_z, default=None),
        )
    return dict(window_start_s=cfg["window_start_s"], objects=result)


CAPACITY_SCHEMA = "genenv.text_contact_capacity_sample.v1"
_CAPACITY_LIMITS = {
    "collision_pairs", "broadphase_pairs", "candidate_contacts", "postpruning_contacts",
    "possible_geom_pairs", "broadphase_multiplier",
}
_CAPACITY_USAGE = {"broadphase_pairs", "postpruning_contacts", "contacting_geom_pairs"}


def _counter(value):
    values = np.asarray(value.to_numpy()).reshape(-1)
    if len(values) != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("expected one integer collision counter for the single environment")
    return int(values[0])


def sample_capacity(solver, cfg, contacts):
    """Observe broadphase candidates separately from post-pruning contact geometry pairs."""
    solver.check_errno()
    collider = solver.collider
    info, state = collider._collider_info, collider._collider_state
    sample = dict(
        schema_version=CAPACITY_SCHEMA,
        limits=dict(
            collision_pairs=_counter(info.max_collision_pairs),
            broadphase_pairs=_counter(info.max_collision_pairs_broad),
            candidate_contacts=_counter(info.max_candidate_contacts),
            postpruning_contacts=_counter(info.max_contacts),
            possible_geom_pairs=_counter(info.max_possible_pairs),
            broadphase_multiplier=int(solver._options.multiplier_collision_broad_phase),
        ),
        usage=dict(
            broadphase_pairs=_counter(state.n_broad_pairs),
            postpruning_contacts=_counter(state.n_contacts),
            contacting_geom_pairs=len({tuple(sorted((c["geom_a"], c["geom_b"])))
                                      for c in contacts}),
        ),
    )
    _validate_capacity_sample(cfg, sample, contacts)
    return sample


def _validate_capacity_sample(cfg, sample, contacts):
    if (
        set(sample) != {"schema_version", "limits", "usage"}
        or sample["schema_version"] != CAPACITY_SCHEMA
    ):
        raise ValueError("invalid collision capacity telemetry schema")
    limits, usage = sample["limits"], sample["usage"]
    if set(limits) != _CAPACITY_LIMITS or set(usage) != _CAPACITY_USAGE:
        raise ValueError("incomplete collision capacity telemetry")
    for value in (*limits.values(), *usage.values()):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError("collision capacity counters must be nonnegative integers")
    if limits["broadphase_multiplier"] < 1 or (
        limits["collision_pairs"] != min(cfg["max_collision_pairs"], limits["possible_geom_pairs"])
        or limits["broadphase_pairs"] != limits["collision_pairs"] * limits["broadphase_multiplier"]
        or limits["postpruning_contacts"] != min(cfg["max_contacts"], limits["candidate_contacts"])
    ):
        raise ValueError("effective collision capacities contradict requested capacities")
    if (
        usage["broadphase_pairs"] > limits["broadphase_pairs"]
        or usage["postpruning_contacts"] > limits["postpruning_contacts"]
        or usage["postpruning_contacts"] > limits["candidate_contacts"]
        or usage["contacting_geom_pairs"] > limits["collision_pairs"]
    ):
        raise ValueError("collision capacity overflow in telemetry")
    geom_pairs = {tuple(sorted((c["geom_a"], c["geom_b"]))) for c in contacts}
    if (
        usage["postpruning_contacts"] != len(contacts)
        or usage["contacting_geom_pairs"] != len(geom_pairs)
    ):
        raise ValueError("collision capacity telemetry disagrees with exported contacts")


def validate_capacity_rows(cfg, rows):
    """Missing telemetry is supported for existing v2 evidence; partial telemetry is invalid."""
    present = ["collision_capacity" in row for row in rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("incomplete collision capacity trajectory")
    effective = rows[0]["collision_capacity"]["limits"]
    peaks = {key: 0 for key in _CAPACITY_USAGE}
    for row in rows:
        sample = row["collision_capacity"]
        _validate_capacity_sample(cfg, sample, row["contacts"])
        if sample["limits"] != effective:
            raise ValueError("effective collision capacities changed during trajectory")
        for key in peaks:
            peaks[key] = max(peaks[key], sample["usage"][key])
    return dict(
        schema_version=CAPACITY_SCHEMA,
        requested=dict(max_collision_pairs=cfg["max_collision_pairs"],
                       max_contacts=cfg["max_contacts"]),
        effective=copy.deepcopy(effective),
        observed_peaks=peaks,
        samples=len(rows),
        semantics=dict(
            broadphase_pairs="AABB candidate geom pairs sent to narrowphase",
            postpruning_contacts="contact points retained for the constraint solver",
            contacting_geom_pairs="distinct geom pairs in retained contact points",
        ),
    )

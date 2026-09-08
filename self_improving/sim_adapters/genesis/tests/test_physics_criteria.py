"""Layer 2: verification of the acceptance criteria themselves.

Every physics entrance in this adapter has now been given the same three-layer shape, and
these tests check the layer that has no scene in it: that the limits are reachable at all,
that they are calibrated per step size instead of shared, and that the contact-detection
dropout which made the old limits unreachable is charged exactly once.
"""

import copy
import itertools

import numpy as np
import pytest
from test_scene_planning import surface

from self_improving.sim_adapters.genesis import asset_physics, repair_physics, validate_physics
from self_improving.sim_adapters.genesis import physics_criteria
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import scene_physics_workflow as workflow
from self_improving.sim_adapters.genesis import scene_stabilization as settling
from self_improving.sim_adapters.genesis.physics_math import (
    creep_step,
    effective_speed,
    free_fall_step,
    longest_run,
    stiffness_floor,
    sweep_radius,
)

ASSET_PROFILES = ("baseline", "half_dt")
REPAIR_PROFILES = [
    (p, n) for p in repair_physics.PROFILES for n in ("legacy", *numerics.CANDIDATE_PROFILES)
]


def every_configuration():
    for profile in ASSET_PROFILES:
        yield f"asset_physics/{profile}", asset_physics.settings(profile)
    for profile, numerical in REPAIR_PROFILES:
        yield f"repair_physics/{profile}/{numerical}", repair_physics.settings(profile, numerical)


# --------------------------------------------------------------------------------------
# Satisfiability: no limit may sit under the floor its own discretisation produces.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,cfg", list(every_configuration()), ids=lambda v: getattr(v, "", v))
def test_shipped_limits_are_reachable(name, cfg):
    """The limit a resting body is asked to meet must exceed one free-fall step."""
    floor = free_fall_step(cfg)
    assert cfg["effective_speed_mps"] > floor, name
    assert cfg["speed_run_steps"] >= 2, name


@pytest.mark.parametrize("name,cfg", list(every_configuration()), ids=lambda v: getattr(v, "", v))
def test_shipped_contact_stiffness_is_not_clamped(name, cfg):
    """Requesting a timeconst under 2*dt does not stiffen contact; Genesis clamps it there.

    The old 0.001 request at dt=0.004 was silently raised to 0.008, so every run was made
    at the stiffest, least stable setting the step allowed while the evidence recorded a
    value that was never used.
    """
    assert cfg["constraint_timeconst"] >= stiffness_floor(cfg), name


def test_validate_physics_defaults_are_reachable():
    cfg = validate_physics.satisfiable(dict(validate_physics.DEFAULTS))
    assert cfg["effective_speed_mps"] > free_fall_step(dict(cfg, gravity=[0.0, 0.0, -9.81]))
    assert cfg["constraint_timeconst"] >= stiffness_floor(cfg)


def test_stabilization_convergence_delta_still_discriminates():
    """The settling delta must stay under the per-step creep of an unsupported body."""
    cfg = asset_physics.settings("baseline")
    creep = settling.discriminating(cfg)
    assert creep == pytest.approx(creep_step(dict(cfg, dt=settling.SETTINGS["dt"])))
    assert settling.SETTINGS["position_delta_m"] < creep


# --------------------------------------------------------------------------------------
# The guards fire. A criterion nobody can meet must be an error here, not a scene failure.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.5, 1.0])
def test_asset_physics_rejects_unreachable_speed_limit(factor):
    cfg = asset_physics.settings("baseline")
    cfg["effective_speed_mps"] = factor * free_fall_step(cfg)
    with pytest.raises(ValueError, match="unsatisfiable"):
        asset_physics.satisfiable(cfg)


@pytest.mark.parametrize("factor", [0.5, 1.0])
def test_repair_physics_rejects_unreachable_speed_limit(factor):
    """The guard asset_physics carried and this entrance did not."""
    cfg = repair_physics.settings()
    cfg["effective_speed_mps"] = factor * free_fall_step(cfg)
    with pytest.raises(ValueError, match="unsatisfiable"):
        repair_physics.satisfiable(cfg)


def test_validate_physics_rejects_unreachable_speed_limit():
    cfg = dict(validate_physics.DEFAULTS, speed_floor_multiple=1.0)
    with pytest.raises(ValueError, match="unsatisfiable"):
        validate_physics.satisfiable(cfg)


@pytest.mark.parametrize(
    "module", [asset_physics, repair_physics, validate_physics], ids=lambda m: m.__name__
)
def test_every_entrance_rejects_clamped_contact_stiffness(module):
    cfg = (
        module.settings()
        if module is repair_physics
        else module.settings("baseline")
        if module is asset_physics
        else dict(module.DEFAULTS)
    )
    cfg["constraint_timeconst"] = stiffness_floor(cfg) / 2
    with pytest.raises(ValueError, match="clamped"):
        module.satisfiable(cfg)


@pytest.mark.parametrize(
    "module", [asset_physics, repair_physics], ids=lambda m: m.__name__
)
def test_single_sample_speed_criterion_rejected(module):
    cfg = module.settings() if module is repair_physics else module.settings("baseline")
    cfg["speed_run_steps"] = 1
    with pytest.raises(ValueError, match="consecutive"):
        module.satisfiable(cfg)


# --------------------------------------------------------------------------------------
# Per-profile calibration: one shared constant is what made the coarse step unreachable.
# --------------------------------------------------------------------------------------


def test_speed_limits_are_calibrated_per_step_size():
    baseline, half = (asset_physics.settings(p) for p in ASSET_PROFILES)
    assert baseline["dt"] != half["dt"]
    assert baseline["effective_speed_mps"] != half["effective_speed_mps"]
    for cfg in (baseline, half):
        assert cfg["effective_speed_mps"] > free_fall_step(cfg)
    # The coarse profile's own limit is what the old shared 0.01 m/s constant sat under.
    assert 0.01 < free_fall_step(baseline)


def test_numerics_registry_offers_a_rung_above_the_stability_floor():
    """A sweep confined to the stiff side cannot separate a bad scene from a bad solve."""
    taus = {numerics.configuration(n)["constraint_timeconst"] for n in numerics.CANDIDATE_PROFILES}
    assert max(taus) >= 0.05
    for name in numerics.CANDIDATE_PROFILES:
        cfg = numerics.configuration(name)
        assert cfg["constraint_timeconst"] >= stiffness_floor(cfg), name


# --------------------------------------------------------------------------------------
# Behaviour: the dropout artefact is charged once, and only to contact continuity.
# --------------------------------------------------------------------------------------


def trace(profile="baseline", steps=200, dropout_steps=(), speed=0.0, moving_steps=(),
          spike_multiple=1.0):
    """A resting stack, with contact for `a` optionally dropped on the named steps.

    `dropout_steps` reproduces the artefact: the contact pair vanishes for those steps and
    the body reads `spike_multiple` free-fall steps of speed. One is what a single lost
    step gives; the impulse doubles when the pair comes back, so two is also observed.
    """
    cfg = asset_physics.settings(profile)
    cfg.update(steps=steps)
    specs = dict(
        table=dict(fixed=True, support="ground", surface=surface(0.6, 0.4, 0.7)),
        a=dict(fixed=False, support="table", surface=None),
    )
    data = dict(settings=cfg, bodies=specs, relations=[])
    loaded, base = {}, {}
    for name, bounds, pos in [
        ("table", [[-0.6, -0.4, 0], [0.6, 0.4, 0.7]], [0, 0, 0]),
        ("a", [[-0.03, -0.03, 0], [0.03, 0.03, 0.06]], [0, 0, 0.7]),
    ]:
        specs[name]["translation_m"] = list(pos)
        base[name] = dict(position=list(pos), orientation_wxyz=[1, 0, 0, 0],
                          velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        loaded[name] = dict(
            native_pose=dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0]),
            visual_hull_local_m=list(itertools.product(*zip(*bounds))),
        )
    base["ground"] = dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0],
                          velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    rows = []
    for i in range(steps + 1):
        objects = copy.deepcopy(base)
        if i in dropout_steps:
            # No contact this step, so the body reads exactly one free-fall step.
            objects["a"]["velocity"] = [0, 0, -spike_multiple * free_fall_step(cfg)]
        if i in moving_steps:
            objects["a"]["velocity"] = [speed, 0, 0]
        contacts = [] if i in dropout_steps else [dict(
            a="a", b="table", geom_a=1, geom_b=0, link_a=1, link_b=0,
            position=[0, 0, 0.7], normal=[0, 0, -1], penetration=0.0001,
            force_a=None if i == 0 else [0, 0, 1],
            force_b=None if i == 0 else [0, 0, -1])]
        rows.append(dict(step=i, time_s=i * cfg["dt"],
                         contact_phase="initial_detection" if i == 0 else "solved_step",
                         objects=objects, contacts=contacts))
    return data, loaded, rows


def outcome(data, loaded, rows):
    return {c["name"]: c for c in asset_physics.evaluate(data, loaded, rows)}


def test_resting_stack_passes():
    assert all(c["passed"] for c in outcome(*trace()).values())


def test_one_lost_step_sits_under_the_calibrated_limit():
    """What the old shared 0.01 m/s constant could never express.

    A single lost contact step reads exactly g*dt. Calibrating the limit to 1.5*g*dt for
    this step size puts the artefact under it outright, which is the whole reason the
    limit has to be derived per profile rather than shared across step sizes.
    """
    data, loaded, rows = trace(dropout_steps=range(8, 201, 50))
    settled = outcome(data, loaded, rows)["a.settled"]
    assert settled["effective_speed_max_mps"] == pytest.approx(
        free_fall_step(data["settings"])
    )
    assert settled["effective_speed_max_mps"] < settled["effective_speed_limit_mps"]
    assert settled["passed"]


def test_isolated_spike_over_the_limit_is_still_not_motion():
    """Second line of defence: re-contact doubles the impulse, and one sample still passes."""
    data, loaded, rows = trace(dropout_steps=range(8, 201, 50), spike_multiple=2.0)
    settled = outcome(data, loaded, rows)["a.settled"]
    assert settled["effective_speed_max_mps"] > settled["effective_speed_limit_mps"]
    assert settled["overspeed_run_steps"] == 1
    assert settled["passed"]


def test_sustained_overspeed_is_motion():
    cfg = asset_physics.settings("baseline")
    fast = 2 * cfg["effective_speed_mps"]
    data, loaded, rows = trace(speed=fast, moving_steps=range(150, 201))
    settled = outcome(data, loaded, rows)["a.settled"]
    assert settled["overspeed_run_steps"] >= cfg["speed_run_steps"]
    assert not settled["passed"]


def test_dropout_is_charged_to_continuity_and_not_to_support():
    """Support asks whether the parent holds the body while it is touching anything.

    Counting the dropout samples in the support denominator too charged one artefact to
    two criteria, so neither number said what it appeared to say.
    """
    data, loaded, rows = trace(dropout_steps=range(8, 201, 50), spike_multiple=2.0)
    result = outcome(data, loaded, rows)
    assert result["a.support"]["fraction"] == 1.0
    assert result["a.support"]["touching_samples"] < result["a.support"]["window_samples"]
    assert 0 < result["a.contact_continuity"]["dropout_fraction"]
    assert result["a.contact_continuity"]["passed"]


def test_the_old_limit_cycle_now_fails_exactly_one_criterion():
    """An eight-step dropout cycle is the signature of contact solved at the stiffness floor.

    It used to surface as a speed failure and a support failure, blaming the scene twice
    for a solver setting. It is now one named, bounded failure and nothing else.
    """
    data, loaded, rows = trace(dropout_steps=range(8, 201, 8), spike_multiple=2.0)
    result = outcome(data, loaded, rows)
    assert not result["a.contact_continuity"]["passed"]
    assert result["a.contact_continuity"]["dropout_fraction"] > result[
        "a.contact_continuity"]["limit"]
    assert result["a.support"]["passed"]
    assert result["a.settled"]["passed"]


def test_body_that_never_touches_anything_fails_support():
    data, loaded, rows = trace(dropout_steps=range(0, 201))
    result = outcome(data, loaded, rows)
    assert not result["a.support"]["passed"]
    assert result["a.support"]["touching_samples"] == 0


# --------------------------------------------------------------------------------------
# Shared primitives.
# --------------------------------------------------------------------------------------


def test_effective_speed_weights_spin_by_sweep_radius():
    spinning = dict(velocity=[0, 0, 0], angular_velocity=[0, 0, 2.0])
    assert effective_speed(spinning, 0.5) == pytest.approx(1.0)
    # A pen and a marble spinning at the same rate are not equally at rest.
    assert effective_speed(spinning, 0.01) < effective_speed(spinning, 0.5)


def test_sweep_radius_is_the_largest_lever_arm():
    box = np.array(list(itertools.product([-1, 1], [-2, 2], [-3, 3])), dtype=float)
    assert sweep_radius(box) == pytest.approx(np.sqrt(1 + 4 + 9))
    with pytest.raises(ValueError):
        sweep_radius(np.zeros((0, 3)))


def test_longest_run_counts_consecutive_samples():
    assert longest_run([]) == 0
    assert longest_run([True, False, True, False, True]) == 1
    assert longest_run([False, True, True, True, False]) == 3


# --------------------------------------------------------------------------------------
# Cross-step-size agreement is a gate, not a note in the report.
# --------------------------------------------------------------------------------------


def write_states(root, name, objects):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    workflow.official.write_json(directory / "final_state.json", dict(objects=objects))


def resting(z=0.7, quat=(1, 0, 0, 0)):
    return dict(pen=dict(position=[0, 0, z], orientation_wxyz=list(quat)))


def test_agreeing_profiles_are_reported_as_agreed(tmp_path):
    write_states(tmp_path, "baseline", resting())
    write_states(tmp_path, "half_dt", resting(z=0.7001))
    result = workflow.agreement(tmp_path)
    assert result["status"] == "compared" and result["agreed"]
    assert result["disagreeing_objects"] == []


def test_rolled_body_makes_the_shared_verdict_inconclusive(tmp_path):
    """160 degrees apart at the two step sizes is not a scene either profile validated."""
    write_states(tmp_path, "baseline", resting())
    half = np.pi * 160 / 360
    write_states(tmp_path, "half_dt", resting(quat=(np.cos(half), np.sin(half), 0, 0)))
    result = workflow.agreement(tmp_path)
    assert result["status"] == "compared" and not result["agreed"]
    assert result["disagreeing_objects"] == ["pen"]
    assert result["maximum_rotation_delta_deg"] > workflow.AGREEMENT_ROTATION_DEG


def test_missing_final_state_is_unavailable_not_a_failure(tmp_path):
    write_states(tmp_path, "baseline", resting())
    assert workflow.agreement(tmp_path)["status"] == "unavailable"


# --------------------------------------------------------------------------------------
# Layer 2, applied to the taxonomy itself: an uncategorised criterion is a silent failure
# of the whole repair loop, because a caller cannot tell a scene defect from a bad solve.
# --------------------------------------------------------------------------------------


def test_every_criterion_the_single_asset_entrance_emits_is_categorised():
    """Derived from a real evaluation rather than a hand-kept list, so drift is caught."""
    from self_improving.sim_adapters.genesis import validate_single_asset as single

    cfg = asset_physics.settings("baseline")
    rows = [
        dict(step=i, time_s=i * cfg["dt"], position=[0.0, 0.0, 0.05],
             orientation_wxyz=[1.0, 0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0],
             angular_velocity=[0.0, 0.0, 0.0], ground_up_force_n=1.0,
             contacts=[dict(penetration=0.0002)])
        for i in range(cfg["steps"] + 1)
    ]
    checks, budget = single.evaluate(rows, cfg, 0.05)
    assert checks and all(c["passed"] for c in checks)
    for c in checks:
        assert c["category"] in (physics_criteria.SCENE_TRUTH,
                                 physics_criteria.NUMERICS_COUPLED,
                                 physics_criteria.DIAGNOSTIC), c["name"]
    assert budget["penetration_budget_used"] == pytest.approx(0.2)
    assert budget["numerics_saturated"] is False


def test_repair_physics_failure_names_are_categorised():
    """The other entrance names the same ideas differently; both must be covered."""
    for name in ("penetration", "scene_bounds", "below_ground", "unexpected_contact",
                 "unsupported_support_tilt", "pose_drift", "pose_rotation",
                 "stable_velocity", "contact_continuity", "support_preserved", "tipped",
                 "support_geometry", "text_relation"):
        assert physics_criteria.category(name)


def test_uncategorised_criterion_raises_rather_than_defaulting():
    """Defaulting an unknown name to either category would silently mis-route a failure."""
    with pytest.raises(KeyError, match="uncategorised"):
        physics_criteria.category("some_new_criterion")


def test_only_a_pure_numerics_failure_is_tunable():
    assert physics_criteria.tunable(["penetration_m"]) is True
    assert physics_criteria.tunable(["penetration_m", "contact_dropout_max"]) is True
    # Real motion alongside a solver artefact still means the scene is wrong.
    assert physics_criteria.tunable(["penetration_m", "drift_rate_mps"]) is False
    assert physics_criteria.tunable(["support_fraction"]) is False
    # Nothing failed: there is nothing to tune, and "tunable" must not read as "retry".
    assert physics_criteria.tunable([]) is False

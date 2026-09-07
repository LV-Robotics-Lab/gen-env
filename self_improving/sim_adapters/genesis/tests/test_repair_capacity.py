"""Actual pair usage must stay distinct from retained contacts, including offline replay."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_text_repair import case

from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics


class Field:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to_numpy(self):
        return self.value.copy()


def sample_case():
    contacts = [dict(geom_a=0, geom_b=1), dict(geom_a=0, geom_b=1), dict(geom_a=0, geom_b=2)]
    info = SimpleNamespace(
        **{
            name: Field(value)
            for name, value in dict(
                max_collision_pairs=6,
                max_collision_pairs_broad=48,
                max_candidate_contacts=30,
                max_contacts=30,
                max_possible_pairs=6,
            ).items()
        }
    )
    state = SimpleNamespace(n_broad_pairs=Field([5]), n_contacts=Field([3]))
    solver = SimpleNamespace(
        check_errno=lambda: None,
        collider=SimpleNamespace(_collider_info=info, _collider_state=state),
        _options=SimpleNamespace(multiplier_collision_broad_phase=8),
    )
    cfg = physics.settings(numerics_profile="dt2ms_tau10ms_authored_v1")
    sample = numerics.sample_capacity(solver, cfg, contacts)
    return solver, cfg, contacts, sample


def test_native_pairs_and_effective_capacities_are_separate_from_contacts():
    _, cfg, contacts, sample = sample_case()
    assert sample["usage"] == dict(
        broadphase_pairs=5, postpruning_contacts=3, contacting_geom_pairs=2
    )
    result = numerics.validate_capacity_rows(
        cfg, [dict(contacts=contacts, collision_capacity=sample)]
    )
    assert result["requested"] == dict(max_collision_pairs=1024, max_contacts=4096)
    assert result["effective"]["collision_pairs"] == 6
    assert result["effective"]["postpruning_contacts"] == 30
    assert result["observed_peaks"] == sample["usage"]


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("usage", "broadphase_pairs", -1),
        ("usage", "broadphase_pairs", float("nan")),
        ("usage", "broadphase_pairs", float("inf")),
        ("usage", "broadphase_pairs", 1.5),
        ("usage", "broadphase_pairs", True),
        ("usage", "broadphase_pairs", 49),
        ("usage", "postpruning_contacts", 31),
        ("usage", "contacting_geom_pairs", 7),
        ("usage", "postpruning_contacts", 2),
        ("usage", "contacting_geom_pairs", 1),
        ("limits", "collision_pairs", 7),
        ("limits", "postpruning_contacts", 31),
    ],
)
def test_invalid_usage_capacity_or_contact_correspondence_is_rejected(group, key, value):
    _, cfg, contacts, sample = sample_case()
    sample[group][key] = value
    with pytest.raises(ValueError):
        numerics.validate_capacity_rows(cfg, [dict(contacts=contacts, collision_capacity=sample)])


def test_missing_historical_telemetry_allowed_but_partial_new_telemetry_rejected():
    _, cfg, contacts, sample = sample_case()
    assert numerics.validate_capacity_rows(cfg, [dict(contacts=contacts)]) is None
    with pytest.raises(ValueError, match="incomplete"):
        numerics.validate_capacity_rows(
            cfg, [dict(contacts=contacts, collision_capacity=sample), dict(contacts=contacts)]
        )


def test_effective_capacity_cannot_change_between_steps():
    _, cfg, contacts, sample = sample_case()
    other = copy.deepcopy(sample)
    other["limits"].update(candidate_contacts=40, postpruning_contacts=40)
    with pytest.raises(ValueError, match="changed"):
        numerics.validate_capacity_rows(
            cfg,
            [
                dict(contacts=contacts, collision_capacity=sample),
                dict(contacts=contacts, collision_capacity=other),
            ],
        )


def test_native_overflow_and_native_export_mismatch_rejected_live():
    solver, cfg, contacts, _ = sample_case()
    solver.collider._collider_state.n_contacts = Field([4])
    with pytest.raises(ValueError, match="disagrees"):
        numerics.sample_capacity(solver, cfg, contacts)

    def overflow():
        raise RuntimeError("Contact island buffer overflow")

    solver.check_errno = overflow
    with pytest.raises(RuntimeError, match="overflow"):
        numerics.sample_capacity(solver, cfg, contacts)


def test_offline_physics_rejudges_new_telemetry_and_preserves_existing_v2():
    old, rows = case()
    data = physics.frozen_input(
        old["assets"], old["poses"], [], 0, numerics_profile="dt2ms_tau10ms_authored_v1"
    )
    historical = physics.evaluate(data, rows)
    assert historical["passed"]
    assert "collision_capacity" not in historical["diagnostics"]
    _, _, _, sample = sample_case()
    for row in rows:
        sample = copy.deepcopy(sample)
        sample["usage"].update(
            postpruning_contacts=len(row["contacts"]),
            contacting_geom_pairs=len(
                {tuple(sorted((c["geom_a"], c["geom_b"]))) for c in row["contacts"]}
            ),
        )
        row["collision_capacity"] = sample
    observed = physics.evaluate(data, rows)
    assert observed["passed"]
    assert observed["diagnostics"]["collision_capacity"]["samples"] == 1501
    observed["diagnostics"].pop("collision_capacity")
    assert observed == historical
    rows[-1]["collision_capacity"]["usage"]["postpruning_contacts"] += 1
    with pytest.raises(ValueError, match="disagrees"):
        physics.evaluate(data, rows)

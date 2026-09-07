"""Versioned numerical controls preserve acceptance and expose their actual application."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_text_repair import case

from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics


def test_legacy_input_and_settings_remain_exact():
    data, rows = case()
    expected = dict(
        schema_version="genenv.text_repair_input.v1",
        profile=physics.geo.PROFILE,
        settings=copy.deepcopy(physics.SETTINGS),
        assets=copy.deepcopy(data["assets"]),
        poses=copy.deepcopy(data["poses"]),
        relations=[],
        random_seed=0,
    )
    assert json.dumps(data) == json.dumps(expected)
    assert physics.settings() == physics.SETTINGS
    assert "diagnostics" not in physics.evaluate(data, rows)
    assert physics.frozen_input(data["assets"], data["poses"], [], 0,
                                numerics_profile="legacy", repair_preset="legacy") == data


def test_registered_candidates_and_half_dt_preserve_acceptance():
    assert len(numerics.CANDIDATE_PROFILES) == 12
    assert len(numerics.PROFILES) == 17
    order = []
    for name in numerics.CANDIDATE_PROFILES:
        cfg = physics.settings(numerics_profile=name)
        half = physics.settings(numerics_profile=numerics.half_dt_profile(name))
        assert cfg["dt"] * cfg["steps"] == pytest.approx(3)
        assert cfg["window_samples"] * cfg["dt"] == pytest.approx(1)
        assert cfg["window_start_s"] == 2
        assert cfg["substeps"] == 1
        assert cfg["stable_fraction"] == cfg["support_fraction"] == 0.95
        assert cfg["penetration_m"] == 0.001
        assert cfg["penetration_relative"] == 0.01
        assert cfg["max_collision_pairs"] == 1024 and cfg["max_contacts"] == 4096
        assert half["dt"] == cfg["dt"] / 2 and half["steps"] == cfg["steps"] * 2
        assert half["contact_solref"] == cfg["contact_solref"]
        assert half["contact_solimp"] == cfg["contact_solimp"]
        order.append((-cfg["dt"], cfg["contact_solimp"] is not None, cfg["constraint_timeconst"]))
    assert order == sorted(order)
    cfg["contact_solref"][0] = 999
    assert numerics.configuration(name)["contact_solref"][0] != 999


@pytest.mark.parametrize("profile", ["unknown", "", "dt1ms_tau1ms_authored_v1"])
def test_unregistered_profiles_rejected(profile):
    with pytest.raises(ValueError, match="numerics profile"):
        physics.settings(numerics_profile=profile)


def test_new_input_binds_both_profiles_and_rejects_mismatch():
    old, rows = case()
    data = physics.frozen_input(old["assets"], old["poses"], [], 0,
                                numerics_profile=numerics.CANDIDATE_PROFILES[0],
                                repair_preset="text_scene_v2")
    assert data["schema_version"] == "genenv.text_repair_input.v2"
    assert physics.evaluate(data, rows)["passed"]
    for field, value in [("repair_preset", "unknown"), ("numerics_profile", "legacy"),
                         ("schema_version", "genenv.text_repair_input.v1")]:
        bad = copy.deepcopy(data)
        bad[field] = value
        with pytest.raises(ValueError):
            physics.evaluate(bad, rows)
    bad = copy.deepcopy(data)
    bad["settings"]["support_fraction"] = .75
    with pytest.raises(ValueError, match="settings mismatch"):
        physics.evaluate(bad, rows)
    del data["repair_preset"]
    with pytest.raises(ValueError, match="requires frozen"):
        physics.evaluate(data, rows)
    with pytest.raises(ValueError, match="repair preset"):
        physics.frozen_input(old["assets"], old["poses"], [], 0, repair_preset="unknown")


@pytest.mark.parametrize("bad_count,passed", [(25, True), (26, False)])
@pytest.mark.parametrize("metric", ["support", "speed"])
def test_new_numerics_keeps_95_percent_boundary(bad_count, passed, metric):
    data, rows = case()
    data = physics.frozen_input(data["assets"], data["poses"], [], 0,
                                numerics_profile=numerics.CANDIDATE_PROFILES[0])
    for row in rows[-bad_count:]:
        if metric == "support":
            row["contacts"] = row["contacts"][1:]
        else:
            row["objects"]["a"]["velocity"] = [.01, 0, 0]
    result = physics.evaluate(data, rows)
    assert result["passed"] is passed
    assert result == physics.evaluate(data, json.loads(json.dumps(rows)))


class Geom:
    def __init__(self, index, *, ignore=False):
        self.idx = index
        self.params = np.array([.02, .7, .1, .8, .003, .6, 3.])
        self.friction = .4
        self.ignore = ignore

    def get_sol_params(self):
        return self.params

    def set_sol_params(self, value):
        if not self.ignore:
            self.params = np.asarray(value, dtype=np.float32)

    def get_friction(self):
        return self.friction


class Entity:
    def __init__(self, *geoms):
        self.geoms = geoms

    def get_links_mass(self):
        return np.array([.5])

    def get_links_inertia(self):
        return np.eye(3)[None]


@pytest.mark.parametrize("mode", ["authored", "standard"])
def test_actual_contact_override_includes_ground_and_preserves_materials(mode):
    cfg = physics.settings(numerics_profile=f"dt1ms_tau10ms_{mode}_v1")
    entities = dict(ground=Entity(Geom(0)), asset=Entity(Geom(1), Geom(2)))
    audit = numerics.apply_contact_parameters(entities, cfg)
    assert [a["object_id"] for a in audit] == ["ground", "asset", "asset"]
    assert [a["geom_index"] for a in audit] == [0, 1, 2]
    for row in audit:
        assert row["sol_params_before"][:2] == [.02, .7]
        assert row["sol_params_after"][:2] == pytest.approx([.01, 1.])
        expected = row["sol_params_before"][2:] if mode == "authored" else numerics.STANDARD_SOLIMP
        assert row["sol_params_after"][2:] == pytest.approx(expected)
        assert row["friction_before"] == row["friction_after"] == .4
    json.dumps(audit, allow_nan=False)


def test_ignored_contact_override_is_error():
    cfg = physics.settings(numerics_profile=numerics.CANDIDATE_PROFILES[0])
    with pytest.raises(ValueError, match="effective contact parameters"):
        numerics.apply_contact_parameters(dict(ground=Entity(Geom(0, ignore=True))), cfg)


def test_engine_and_exported_contact_overflow_are_errors():
    cfg = physics.settings(numerics_profile=numerics.CANDIDATE_PROFILES[0])
    calls = []
    solver = SimpleNamespace(check_errno=lambda: calls.append(True))
    numerics.check_capacity(solver, cfg, 4096)
    assert calls == [True]
    with pytest.raises(ValueError, match="capacity overflow"):
        numerics.check_capacity(solver, cfg, 4097)

    def overflow():
        raise RuntimeError("Contact island buffer overflow")

    solver.check_errno = overflow
    with pytest.raises(RuntimeError, match="island buffer overflow"):
        numerics.check_capacity(solver, cfg, 1)


def test_diagnostics_separate_no_contact_from_no_upward_support():
    data, rows = case()
    window = rows[-5:]
    # Three consecutive support gaps: two absent contacts and one downward contact.
    for row in window[:2]:
        row["contacts"] = [c for c in row["contacts"] if "a" not in (c["a"], c["b"])]
    c = window[2]["contacts"][0]
    c.update(force_a=[0, 0, -1], force_b=[0, 0, 1])
    for row in window[1:3]:
        row["objects"]["a"]["velocity"] = [.02, 0, 0]
    result = numerics.diagnostics(data, window)["objects"]["a"]
    assert result["window_samples"] == 5
    assert result["no_contact_samples"] == 2
    assert result["no_support_samples"] == 3
    assert result["overspeed_samples"] == 2
    assert result["overspeed_no_contact_samples"] == 1
    assert result["overspeed_no_support_samples"] == 2
    assert result["longest_no_support_s"] == pytest.approx(.006)
    assert result["minimum_com_height_m"] == window[0]["objects"]["a"]["com_position"][2]
    assert result["minimum_collision_height_m"] is not None
    json.dumps(result, allow_nan=False)

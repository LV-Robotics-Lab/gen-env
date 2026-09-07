"""Analytic support-plane controls cannot stand in for source-scene acceptance."""

import copy
import json
import math

import pytest
from test_text_repair import case

from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import text_plane_control as control


def pair():
    data, _ = case()
    data["assets"] = {k: data["assets"][k] for k in ("table", "a")}
    data["poses"] = {k: data["poses"][k] for k in data["assets"]}
    data["assets"]["a"]["category"] = "cup"
    return data


def test_horizontal_plane_uses_measured_world_height_without_moving_cup():
    data = pair()
    data["poses"]["table"]["position"] = [0.2, -0.4, 0.3]
    data["poses"]["table"]["orientation_wxyz"] = [math.cos(0.3), 0, 0, math.sin(0.3)]
    before = copy.deepcopy(data)
    cup, table, z = control.source_pair(data)
    assert (cup, table) == ("a", "table")
    assert z == pytest.approx(0.3 + data["assets"][table]["surface"]["z_m"])
    assert data == before


@pytest.mark.parametrize("attack", ["tilt", "dynamic_table", "extra_body", "no_surface"])
def test_plane_control_rejects_ambiguous_source_support(attack):
    data = pair()
    if attack == "tilt":
        data["poses"]["table"]["orientation_wxyz"] = [math.cos(0.1), math.sin(0.1), 0, 0]
    elif attack == "dynamic_table":
        data["assets"]["table"]["fixed"] = False
    elif attack == "extra_body":
        data["assets"]["extra"] = copy.deepcopy(data["assets"]["table"])
        data["poses"]["extra"] = copy.deepcopy(data["poses"]["table"])
    else:
        data["assets"]["table"]["surface"] = None
    with pytest.raises(ValueError):
        control.source_pair(data)


def trace():
    cfg = dict(
        steps=10,
        dt=0.1,
        window_start_s=0.5,
        window_samples=5,
        effective_speed_mps=0.01,
        support_force_n=1e-6,
    )
    rows = []
    for step in range(11):
        contact = dict(
            a="a",
            b=control.PLANE,
            penetration=0.0001,
            force_a=[0, 0, 1] if step else None,
            force_b=[0, 0, -1] if step else None,
        )
        state = dict(
            position=[0, 0, 1],
            orientation_wxyz=[1, 0, 0, 0],
            com_position=[0, 0, 1],
            velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0],
            collision_bottom_z_m=0.7,
            visual_bottom_z_m=0.7,
            plane_up_force_n=1 if step else None,
        )
        rows.append(dict(step=step, time_s=step * 0.1, objects=dict(a=state), contacts=[contact]))
    return cfg, rows


def test_plane_metrics_remain_diagnostic_even_if_all_samples_are_stable():
    cfg, rows = trace()
    result = control.summarize(rows, cfg, "a", 0.1)
    assert result["stable_fraction"] == result["support_fraction"] == 1
    assert result["diagnostic_only"]
    assert result["source_scene_acceptance"] == "not_evaluated"
    assert "passed" not in result and "physics_passed" not in result


@pytest.mark.parametrize("attack", ["truncated", "sequence", "nan", "wrong_parent", "force"])
def test_diagnostic_metrics_reject_incomplete_or_invalid_observations(attack):
    cfg, rows = trace()
    if attack == "truncated":
        rows.pop()
    elif attack == "sequence":
        rows[5]["step"] = 3
    elif attack == "nan":
        rows[-1]["objects"]["a"]["velocity"] = [float("nan"), 0, 0]
    elif attack == "wrong_parent":
        rows[-1]["contacts"][0]["b"] = "table"
    else:
        rows[-1]["objects"]["a"]["plane_up_force_n"] = 0.8
    with pytest.raises(ValueError):
        control.summarize(rows, cfg, "a", 0.1)


def test_execution_error_preserves_separate_diagnostic_report(tmp_path, monkeypatch):
    data = pair()
    assets = tmp_path / "assets"
    assets.mkdir()
    model = assets / "model.xml"
    model.write_text("fixture bound bytes; simulator initialization intentionally fails")
    files = [control.official.fingerprint(model, assets)]
    for a in data["assets"].values():
        a.update(
            source_root=str(assets),
            source_files=files,
            derived_root=str(assets),
            derived_files=files,
            physics_file=str(model),
        )
    source = tmp_path / "physics_input.json"
    source.write_text(json.dumps(data))
    before = source.read_bytes()

    def fail():
        raise RuntimeError("simulator unavailable")

    monkeypatch.setattr(control.preparation, "init_genesis", fail)
    out = tmp_path / "control"
    report = control.run(source, out, numerics.CANDIDATE_PROFILES[0])
    assert report["status"] == "error" and report["exit_code"] == 1
    assert report["physics_status"] == "not_evaluated" and report["diagnostic_only"]
    assert not report["simulation_executed"] and report["steps_executed"] == 0
    assert (out / "diagnostic_result.json").exists()
    assert not (out / "physics_result.json").exists()
    assert source.read_bytes() == before
    control.official.verify_files(out, report["artifacts"])

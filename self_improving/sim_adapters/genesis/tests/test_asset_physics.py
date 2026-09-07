"""Native-scene physics evidence attack tests, without starting Genesis."""

import copy
import itertools
import json

import numpy as np
import pytest
from test_scene_planning import surface
from test_task_output import make_task, snapshot

from self_improving.sim_adapters.genesis import asset_physics as checks
from self_improving.sim_adapters.genesis import validate_asset_scene as entry


def sample_case(stack=False):
    specs = dict(
        table=dict(fixed=True, support="ground", surface=surface(0.6, 0.4, 0.7)),
        a=dict(fixed=False, support="table", surface=surface(0.15, 0.15, 0.1)),
        b=dict(fixed=False, support="a" if stack else "table", surface=None),
    )
    data = dict(settings=checks.settings("baseline"), bodies=specs, relations=[])
    data["settings"].update(dt=0.1, steps=10)  # Small offline trace, not a runtime profile.
    loaded, states = (
        {},
        {
            "ground": dict(
                position=[0, 0, 0],
                orientation_wxyz=[1, 0, 0, 0],
                velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
            )
        },
    )
    for name, bounds, pos in [
        ("table", [[-0.6, -0.4, 0], [0.6, 0.4, 0.7]], [0, 0, 0]),
        ("a", [[-0.15, -0.15, 0], [0.15, 0.15, 0.1]], [0, 0, 0.7]),
        ("b", [[-0.03, -0.03, 0], [0.03, 0.03, 0.1]], [0, 0, 0.8] if stack else [0.3, 0, 0.7]),
    ]:
        data["bodies"][name]["translation_m"] = list(pos)
        states[name] = dict(
            position=pos,
            orientation_wxyz=[1, 0, 0, 0],
            velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0],
        )
        loaded[name] = dict(
            native_pose=dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0]),
            visual_hull_local_m=list(itertools.product(*zip(*bounds))),
        )
    rows = []
    for i in range(11):
        contacts = []
        for j, name in enumerate(("a", "b")):
            contacts.append(
                dict(
                    a=name,
                    b=specs[name]["support"],
                    geom_a=j + 1,
                    geom_b=0,
                    link_a=j + 1,
                    link_b=0,
                    position=[0, 0, 0.7],
                    normal=[0, 0, -1],
                    penetration=0.0001,
                    force_a=None if i == 0 else [0, 0, 1],
                    force_b=None if i == 0 else [0, 0, -1],
                )
            )
        rows.append(
            dict(
                step=i,
                time_s=i * 0.1,
                contact_phase="initial_detection" if i == 0 else "solved_step",
                objects=copy.deepcopy(states),
                contacts=contacts,
            )
        )
    return data, loaded, rows


def failures(data, loaded, rows):
    return {r["name"] for r in checks.evaluate(data, loaded, rows) if not r["passed"]}


def test_normal_and_multilevel_bidirectional_contacts():
    for stack in (False, True):
        assert not failures(*sample_case(stack))


@pytest.mark.parametrize(
    "attack",
    [
        "wrong_support",
        "ground_settled",
        "moving",
        "rotation",
        "penetration",
        "initial_penetration",
        "footprint",
        "tilt",
        "unexpected_child",
        "no_contact",
    ],
)
def test_physical_false_positives_rejected(attack):
    data, loaded, rows = sample_case()
    for row in rows:
        state = row["objects"]["a"]
        if attack in ("wrong_support", "ground_settled"):
            row["contacts"][0]["b"] = "ground"
        if attack == "ground_settled":
            state["position"][2] = 0
        if attack == "moving":
            state["velocity"][0] = 0.011
        if attack == "rotation" and row["step"] == 8:
            state["orientation_wxyz"] = [np.cos(0.01), 0, 0, np.sin(0.01)]
        if attack == "penetration" or (attack == "initial_penetration" and row["step"] == 0):
            row["contacts"][0]["penetration"] = 0.002
        if attack == "footprint":
            state["position"][0] = 0.5
        if attack == "tilt":
            row["objects"]["table"]["orientation_wxyz"] = [np.cos(0.01), np.sin(0.01), 0, 0]
        if attack == "unexpected_child":
            row["contacts"][0]["b"] = "b"
        if attack == "no_contact":
            row["contacts"] = []
    for name, spec in data["bodies"].items():
        spec["translation_m"] = rows[0]["objects"][name]["position"]
    if attack == "tilt":
        loaded["table"]["native_pose"]["orientation_wxyz"] = rows[0]["objects"]["table"][
            "orientation_wxyz"
        ]
        # Keep initial pose intact; tilt only during the actual observation window.
        loaded["table"]["native_pose"]["orientation_wxyz"] = [1, 0, 0, 0]
        rows[0]["objects"]["table"]["orientation_wxyz"] = [1, 0, 0, 0]
    assert failures(data, loaded, rows)


@pytest.mark.parametrize(
    "attack",
    [
        "missing_row",
        "repeated_step",
        "wrong_time",
        "missing_object",
        "nan",
        "inf_contact",
        "missing_force",
        "initial_fake_force",
        "unknown_object",
        "wrong_force_sign",
        "zero_quat",
        "missing_contact_field",
        "negative_penetration",
    ],
)
def test_incomplete_or_invalid_evidence_is_error(attack):
    data, loaded, rows = sample_case()
    row = rows[5]
    if attack == "missing_row":
        rows.pop()
    elif attack == "repeated_step":
        row["step"] = 4
    elif attack == "wrong_time":
        row["time_s"] += 0.001
    elif attack == "missing_object":
        row["objects"].pop("a")
    elif attack == "nan":
        row["objects"]["a"]["velocity"][0] = float("nan")
    elif attack == "inf_contact":
        row["contacts"][0]["penetration"] = float("inf")
    elif attack == "missing_force":
        row["contacts"][0]["force_a"] = None
    elif attack == "initial_fake_force":
        rows[0]["contacts"][0]["force_a"] = [0, 0, 0]
    elif attack == "unknown_object":
        row["contacts"][0]["a"] = "other"
    elif attack == "wrong_force_sign":
        row["contacts"][0]["force_a"] = row["contacts"][0]["force_b"]
    elif attack == "zero_quat":
        row["objects"]["a"]["orientation_wxyz"] = [0, 0, 0, 0]
    elif attack == "missing_contact_field":
        row.pop("contacts")
    else:
        row["contacts"][0]["penetration"] = -0.1
    with pytest.raises((ValueError, KeyError)):
        checks.evaluate(data, loaded, rows)


def test_target_native_frame_translation_and_rotation():
    data, loaded, rows = sample_case(stack=True)
    # Shift and yaw the complete world; the native support planes remain unchanged.
    yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for row in rows[1:]:
        for name, state in row["objects"].items():
            if name == "ground":
                continue
            state["position"] = (np.array(state["position"]) @ yaw.T + [1, 2, 0]).tolist()
            state["orientation_wxyz"] = [2**-0.5, 0, 0, 2**-0.5]
    assert not failures(data, loaded, rows)


def test_explicit_relation_rechecked_and_soft_preferences_ignored():
    data, loaded, rows = sample_case()
    data["relations"] = [dict(source="a", target="b", relation="left_of")]
    assert not failures(data, loaded, rows)
    data["relations"][0]["relation"] = "right_of"
    assert "relation.a.right_of.b" in failures(data, loaded, rows)


def test_contact_mapping_mask_and_direction():
    raw = dict(
        geom_a=[1, -1],
        geom_b=[2, -1],
        link_a=[3, -1],
        link_b=[4, -1],
        position=[[0, 0, 0], [float("nan")] * 3],
        normal=[[0, 0, 1], [0] * 3],
        penetration=[0.0001, float("nan")],
        force=[[0, 0, 2], [0] * 3],
        valid_mask=[True, False],
    )
    converted = entry.contacts(raw, {1: "a", 2: "table"}, {1: 3, 2: 4})
    assert len(converted) == 1
    assert converted[0]["force_a"] == [0, 0, -2]
    assert converted[0]["force_b"] == [0, 0, 2]
    assert (
        entry.contacts(raw, {1: "a", 2: "table"}, {1: 3, 2: 4}, initial=True)[0]["force_a"] is None
    )
    with pytest.raises(ValueError, match="mapping"):
        entry.contacts(raw, {1: "a", 2: "table"}, {1: 4, 2: 3})


@pytest.mark.parametrize(
    "attack",
    ["fixed", "dofs", "collision", "mass", "inertia", "friction", "bounds", "mesh", "orientation"],
)
def test_loaded_contract_gate(attack):
    actual = dict(
        fixed=False,
        dofs=6,
        collision_geoms=1,
        collision_enabled=True,
        mass_kg=1.0,
        links=[dict(fixed=False, dofs=6, inertia_kg_m2=np.eye(3).tolist())],
        friction=[1.0],
        max_bounds_error_m=0.0,
        native_mesh_matches=True,
        orientation_error_deg=0.0,
    )
    entry.validate_loaded("a", {"fixed": False}, actual)
    if attack == "fixed":
        actual["fixed"] = True
    elif attack == "dofs":
        actual["dofs"] = 0
    elif attack == "collision":
        actual["collision_enabled"] = False
    elif attack == "mass":
        actual["mass_kg"] = 0
    elif attack == "inertia":
        actual["links"][0]["inertia_kg_m2"][0][0] = -1
    elif attack == "friction":
        actual["friction"] = [float("nan")]
    elif attack == "bounds":
        actual["max_bounds_error_m"] = 0.01
    elif attack == "mesh":
        actual["native_mesh_matches"] = False
    else:
        actual["orientation_error_deg"] = 1
    with pytest.raises(ValueError):
        entry.validate_loaded("a", {"fixed": False}, actual)


def install_offline(monkeypatch, task, *, attack=None):
    data, loaded, rows = sample_case()

    def prepare(*args):
        return data, task.verify_physics_inputs

    def simulate(data, out, check, progress):
        check()
        if attack == "missing_row":
            rows.pop()
        if attack == "physical":
            rows[-1]["contacts"][0]["penetration"] = 0.01
        if attack == "tamper":
            (task.stage("objects") / "apple_1.json").write_text("tampered")
        entry.write_json(out / "asset_physics_report.json", dict(bodies=loaded))
        entry.write_json(out / "initial_state.json", dict(checks=[]))
        entry.write_json(out / "final_state.json", dict(state=rows[-1], passed=False))
        (out / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if attack == "disk":
            with (out / "trace.jsonl").open("a") as stream:
                stream.write("{}\n")
        if attack == "input":
            (out / "physics_input.json").write_text("{}")
        progress.update(
            simulation_executed=True, steps_executed=len(rows) - 1, last_complete_step=len(rows) - 1
        )
        return loaded, rows

    monkeypatch.setattr(entry, "prepare", prepare)
    return simulate


@pytest.mark.parametrize("attack", [None, "physical", "tamper", "missing_row", "disk", "input"])
def test_independent_lifecycle_and_failure_evidence(tmp_path, monkeypatch, attack):
    task = make_task(tmp_path)
    before = {s: snapshot(task.stage(s)) for s in ("objects", "scene")}
    for stage in ("physics", "final_render"):
        (task.stage(stage) / "stale.png").write_bytes(b"old")
    task.report["stages"].update(physics="passed", final_render="passed")
    task.seal()
    report = entry.run(
        task.root,
        tmp_path / "index" / "index.json",
        simulator=install_offline(monkeypatch, task, attack=attack),
    )
    task.verify()
    assert report["exit_code"] == (0 if attack is None else 2 if attack == "physical" else 1)
    assert task.report["status"] == ("physics_passed" if attack is None else "physics_failed")
    assert task.report["stages"]["final_render"] == "not_run"
    assert not list(task.stage("final_render").iterdir())
    assert not (task.stage("physics") / "stale.png").exists()
    assert (task.stage("physics") / "trace.jsonl").exists()
    if attack != "tamper":
        assert all(snapshot(task.stage(s)) == before[s] for s in before)
    assert json.loads((task.stage("physics") / "final_state.json").read_text())["passed"] == (
        attack is None
    )


def test_fixed_profiles_keep_duration_and_thresholds():
    a, b = checks.settings("baseline"), checks.settings("half_dt")
    assert a["dt"] * a["steps"] == b["dt"] * b["steps"] == 4
    assert {k: v for k, v in a.items() if k not in ("dt", "steps")} == {
        k: v for k, v in b.items() if k not in ("dt", "steps")
    }


def test_zero_inertia_fixed_child_is_allowed_but_free_child_is_not():
    actual = dict(
        fixed=False,
        dofs=6,
        collision_geoms=1,
        collision_enabled=True,
        mass_kg=1.0,
        links=[
            dict(fixed=False, dofs=6, inertia_kg_m2=np.eye(3).tolist()),
            dict(fixed=False, dofs=0, inertia_kg_m2=np.zeros((3, 3)).tolist()),
        ],
        friction=[1.0],
        max_bounds_error_m=0.0,
        native_mesh_matches=True,
        orientation_error_deg=0.0,
    )
    entry.validate_loaded("a", {"fixed": False}, actual)
    actual["links"][1]["dofs"] = 1
    with pytest.raises(ValueError):
        entry.validate_loaded("a", {"fixed": False}, actual)


# Use the existing complete selected-asset fixture, including all index/source hashes.
from test_build_scene import geometry, prepared  # noqa: E402
from test_clip_select import setup  # noqa: E402, F401


@pytest.fixture
def input_task(setup, monkeypatch):  # noqa: F811
    from self_improving.sim_adapters.genesis import build_scene as builder

    task = prepared(setup, monkeypatch)

    def renderer(doc, bindings, bounds, output, check, *, plan_layout):
        geom = geometry()
        for value in geom.values():
            value["mesh_sha256"] = "offline_geometry_hash"
        layout = plan_layout(geom)
        entry.write_json(output / "native_geometry.json", geom)
        entry.write_json(output / "scene_layout.json", layout)
        entry.write_json(output / "support_surfaces.json", layout["support_surfaces"])
        return dict(status="passed")

    result = builder.run(task.root, setup.index / "index.json", planner="rule", renderer=renderer)
    assert result["status"] == "scene_built"
    task.verify()
    return task, setup.index / "index.json"


@pytest.mark.parametrize(
    "attack",
    [
        None,
        "fixed_source",
        "unknown_fixed",
        "graph",
        "translation",
        "source",
        "asset_binding",
        "layout_digest",
        "upstream",
    ],
)
def test_input_binding_and_configuration_gate(input_task, attack):
    task, index = input_task
    if attack in ("graph", "translation", "asset_binding", "layout_digest"):
        p = task.stage("scene") / ("scene_graph.json" if attack == "graph" else "scene_layout.json")
        value = json.loads(p.read_text())
        if attack == "graph":
            value["edges"][0]["target"] = "cup_1"
        elif attack == "translation":
            value["objects"][0]["translation_m"][0] += 1
        elif attack == "asset_binding":
            value["objects"][0]["asset_id"] = "other"
        else:
            value["scene_graph_sha256"] = "tampered"
        entry.write_json(p, value)
        task.seal()  # Even a refreshed outer manifest cannot bypass semantic checks.
    with task.lock():
        task.start_physics()
        if attack == "source":
            layout = json.loads((task.stage("scene") / "scene_layout.json").read_text())
            from pathlib import Path

            Path(layout["objects"][0]["model_entrypoint"]).write_text("tampered")
        if attack == "upstream":
            (task.stage("objects") / "apple_1.json").write_text("tampered")
        fixed = (
            ["apple_1"]
            if attack == "fixed_source"
            else ["missing"]
            if (attack == "unknown_fixed")
            else []
        )
        if attack is None:
            data, check = entry.prepare(task, index, fixed, "baseline")
            check()
            assert len(data["bodies"]) == 4 and data["model_calls"] == 0
        else:
            with pytest.raises((ValueError, KeyError)):
                entry.prepare(task, index, fixed, "baseline")


def test_owned_physics_copies_preserve_all_upstream_bytes(input_task, tmp_path):
    task, _ = input_task
    before = {s: snapshot(task.stage(s)) for s in ("objects", "scene")}
    target = task.copy_for_physics(tmp_path / "copied")
    target.verify()
    target.verify_scene_inputs()
    assert all(snapshot(target.stage(s)) == before[s] for s in before)
    with pytest.raises(FileExistsError):
        task.copy_for_physics(target.root)
    second = target.copy_for_physics(tmp_path / "copied_again")
    second.verify_scene_inputs()
    (second.root / ".scene_source_owner.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        second.verify_scene_inputs()


def test_scene_rebuild_after_physics_copy_uses_current_owner(input_task, tmp_path):
    task, _ = input_task
    target = task.copy_for_physics(tmp_path / "rebuild_copy")
    with target.lock():
        target.start_scene()
        target.verify_scene_inputs()
        target.finish_scene(dict(status="scene_built"))
    target.verify()


def test_changed_initial_pose_cannot_be_validated_as_original():
    data, loaded, rows = sample_case()
    rows[0]["objects"]["a"]["position"][0] += 0.01
    with pytest.raises(ValueError, match="initial pose"):
        checks.evaluate(data, loaded, rows)


def test_preparation_failure_seals_failed_stage_without_stale_render(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    (task.stage("final_render") / "old.png").write_bytes(b"old")
    task.report["stages"].update(physics="passed", final_render="passed")
    task.seal()

    def fail(*args):
        raise ValueError("unsupported native format")

    monkeypatch.setattr(entry, "prepare", fail)
    report = entry.run(task.root, tmp_path / "index" / "index.json")
    task.verify()
    assert report["failure_kind"] == "input" and report["exit_code"] == 1
    assert not report["simulation_executed"]
    assert not list(task.stage("final_render").iterdir())
    assert "trace.jsonl" in report["not_generated"]


def test_interruption_retains_last_complete_sampling(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    data, loaded, rows = sample_case()
    monkeypatch.setattr(entry, "prepare", lambda *a: (data, task.verify_physics_inputs))

    def interrupted(data, out, check, progress):
        (out / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:3]) + "\n")
        entry.write_json(
            out / "final_state.json", dict(state=rows[2], passed=False, complete=False)
        )
        progress.update(
            simulation_executed=True, steps_executed=2, last_complete_step=2, phase="simulation"
        )
        raise KeyboardInterrupt()

    report = entry.run(task.root, tmp_path / "index" / "index.json", simulator=interrupted)
    task.verify()
    assert report["exit_code"] == 1 and report["last_complete_step"] == 2
    assert len((task.stage("physics") / "trace.jsonl").read_text().splitlines()) == 3
    assert not json.loads((task.stage("physics") / "final_state.json").read_text())["passed"]

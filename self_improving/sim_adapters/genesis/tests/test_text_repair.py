"""Text construction contract tests without a simulator or asset downloads."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from test_asset_physics import sample_case

from self_improving.sim_adapters.genesis import construct_asset_scene as entry
from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video


def case(stack=False):
    old, loaded, old_rows = sample_case(stack)
    assets, poses = {}, {}
    for n, a in old["bodies"].items():
        hull = np.array(loaded[n]["visual_hull_local_m"])
        d = float(np.linalg.norm(np.ptp(hull, axis=0)))
        assets[n] = dict(
            a,
            hull=hull.tolist(),
            collision_hulls=[hull.tolist()],
            diagonal_m=d,
            radius_m=d / 2,
            margin_m=max(0.01, 0.02 * d),
            buffer_m=max(0.002, 0.005 * d),
            natural_up=[0, 0, 1],
            tip_limit_deg=15,
            bbox_size_m=np.ptp(hull, axis=0).tolist(),
            category="table" if n == "table" else "apple",
        )
        poses[n] = {k: old_rows[0]["objects"][n][k] for k in ("position", "orientation_wxyz")}
    data = physics.frozen_input(assets, poses, [], 0)
    rows = []
    for i in range(1501):
        row = copy.deepcopy(old_rows[0 if not i else -1])
        row.update(step=i, time_s=i * 0.002)
        for state in row["objects"].values():
            state["com_position"] = list(state["position"])
        rows.append(row)
    return data, rows


def test_baseline_and_dynamic_stack():
    for stack in (False, True):
        data, rows = case(stack)
        assert physics.evaluate(data, rows)["passed"]
        assert not data["assets"]["a"]["fixed"]


@pytest.mark.parametrize("bad_count,passed", [(25, True), (26, False), (500, False)])
def test_exact_95_percent_speed(bad_count, passed):
    data, rows = case()
    for row in rows[-bad_count:]:
        row["objects"]["a"]["velocity"] = [0.01, 0, 0]
    result = physics.evaluate(data, rows)
    assert result["passed"] is passed
    assert result["objects"]["a"]["stable_fraction"] == (500 - bad_count) / 500


@pytest.mark.parametrize("bad_count,passed", [(25, True), (26, False)])
def test_support_contact_jitter_boundary(bad_count, passed):
    data, rows = case()
    for row in rows[-bad_count:]:
        row["contacts"] = row["contacts"][1:]
    assert physics.evaluate(data, rows)["passed"] is passed


def test_angular_speed_scales_with_radius():
    data, rows = case()
    for row in rows[1001:]:
        row["objects"]["a"]["angular_velocity"] = [0, 0, 0.011 / data["assets"]["a"]["radius_m"]]
    result = physics.evaluate(data, rows)
    assert "stable_velocity" in result["failures"]["a"]
    assert result["objects"]["a"]["effective_velocity_max_mps"] == pytest.approx(0.011)


@pytest.mark.parametrize(
    "attack",
    [
        "penetration",
        "initial_penetration",
        "wrong_support",
        "ground_settled",
        "tipped",
        "footprint",
        "unexpected_contact",
        "support_tilt",
        "wrong_force",
    ],
)
def test_physical_false_positives(attack):
    data, rows = case()
    for row in rows[1001:] if attack != "initial_penetration" else rows[:1]:
        state = row["objects"]["a"]
        contact = row["contacts"][0]
        if attack in ("penetration", "initial_penetration"):
            contact["penetration"] = 0.00101
        if attack in ("wrong_support", "ground_settled"):
            contact["b"] = "ground"
        if attack == "ground_settled":
            state["position"][2] = 0
        if attack == "tipped":
            state["orientation_wxyz"] = [np.cos(0.2), np.sin(0.2), 0, 0]
        if attack == "support_tilt":
            row["objects"]["table"]["orientation_wxyz"] = [np.cos(0.01), np.sin(0.01), 0, 0]
        if attack == "footprint":
            state["position"][0] = 0.5
        if attack == "unexpected_contact":
            contact["b"] = "b"
        if attack == "wrong_force":
            contact.update(force_a=[0, 0, -1], force_b=[0, 0, 1])
    assert not physics.evaluate(data, rows)["passed"]


def test_pose_drift_is_score_only():
    data, rows = case()
    for row in rows[1:]:
        row["objects"]["a"]["position"][0] += 0.015
    result = physics.evaluate(data, rows)
    assert result["passed"]
    assert result["objects"]["a"]["normalized_drift"] > 0.01


@pytest.mark.parametrize(
    "attack", ["missing_row", "nan", "inf", "missing_object", "origin", "initial_force", "settings"]
)
def test_evidence_integrity(attack):
    data, rows = case()
    if attack == "missing_row":
        rows.pop(100)
    if attack in ("nan", "inf"):
        rows[2]["objects"]["a"]["velocity"][0] = float(attack)
    if attack == "missing_object":
        del rows[3]["objects"]["a"]
    if attack == "origin":
        rows[0]["objects"]["a"]["position"][0] += 0.001
    if attack == "initial_force":
        rows[0]["contacts"][0]["force_a"] = [0, 0, 0]
    if attack == "settings":
        data["settings"]["stable_fraction"] = 0.5
    with pytest.raises((ValueError, KeyError)):
        physics.evaluate(data, rows)


def test_initial_rejection_preserves_no_solved_forces():
    data, rows = case()
    rows = rows[:1]
    rows[0]["contacts"][0]["penetration"] = 0.003
    result = physics.evaluate(data, rows, initial_rejected=True)
    assert not result["passed"] and not result["simulation_executed"]
    assert result["steps_executed"] == 0


def test_relative_penetration_uses_stricter_object():
    assets = {"a": {"diagonal_m": 0.05}, "b": {"diagonal_m": 1}}
    assert not physics.penetration_ok(dict(a="a", b="b", penetration=0.0005), assets)
    assert physics.penetration_ok(dict(a="a", b="b", penetration=0.000499), assets)


def test_scale_priors_conflicts_and_unknowns():
    assert prep.scale_for("cup", [0.1, 0.1, 0.12])[0] == 1
    assert prep.scale_for("cup", [1, 1, 1.5])[0] == pytest.approx(0.08)
    assert prep.scale_for("cup", [0.1, 0.1, 0.12], "高度20厘米")[0] == pytest.approx(5 / 3)
    with pytest.raises(ValueError, match="conflict"):
        prep.scale_for("cup", [0.1, 0.1, 0.12], "高度20厘米", {"unit_scale": 1})
    with pytest.raises(ValueError, match="NEEDS_ASSET_METADATA"):
        prep.scale_for("unrecognized", [1, 1, 1])


def test_oriented_erosion_and_reproducible_top5():
    poly = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]])
    rect = np.array([[-0.4, -0.1], [0.4, -0.1], [0.4, 0.1], [-0.4, 0.1]])
    a = geo.shrink_region(poly, rect, 0.02)
    b = geo.shrink_region(poly, rect[:, ::-1], 0.02)
    assert not np.allclose(np.ptp(a, axis=0), np.ptp(b, axis=0))
    data, _ = case()
    assets = data["assets"]
    poses = {"table": data["poses"]["table"]}
    selected, record = geo.sample_object(assets, poses, "a", [], [], 7, 0)
    assert (selected, record) == geo.sample_object(assets, poses, "a", [], [], 7, 0)
    assert len(record["candidates"]) == 50 and len(record["top_k"]) == 5
    assert record["selected"] in record["top_k"]
    assert not geo.geometric_checks(assets, dict(poses, a=selected["pose"]), [])
    assert record != geo.sample_object(assets, poses, "a", [], [], 8, 0)[1]


def test_subtree_repair_preserves_local_transform_and_unrelated_initial():
    data, _ = case(True)
    assets = data["assets"]
    poses = data["poses"]
    old = geo.inverse([poses["b"]["position"]], poses["a"])
    replacement = dict(position=[0.1, 0.1, 0.7], orientation_wxyz=[2**-0.5, 0, 0, 2**-0.5])
    moved = geo.move_subtree(assets, poses, "a", replacement)
    assert np.allclose(geo.inverse([moved["b"]["position"]], moved["a"]), old)
    assert moved["table"] == poses["table"] and poses["a"] != moved["a"]
    assets["a"]["support"] = "b"
    with pytest.raises(ValueError, match="cycle"):
        geo.topology(assets)


def test_obstacle_above_surface_rejected():
    data, _ = case()
    assets = data["assets"]
    assets["table"]["surface"]["above_triangles_xy_m"] = [[[-0.2, -0.2], [0.2, -0.2], [0, 0.2]]]
    assert "a:support_geometry" in geo.geometric_checks(assets, data["poses"], [])


def test_hull_filling_cavity_and_box_bridging_legs_rejected():
    parts = []
    for x in (-0.45, 0.45):
        part = trimesh.creation.box([0.1, 0.1, 1])
        part.apply_translation([x, 0, 0])
        parts.append(part)
    visual = trimesh.util.concatenate(parts)
    assert not prep.proxy_quality(visual, [visual.convex_hull])["passed"]
    assert not prep.proxy_quality(visual, [visual.bounding_box])["passed"]
    assert prep.proxy_quality(parts[0], [parts[0]])["passed"]


def test_final_video_requires_physical_pass(tmp_path):
    with pytest.raises(ValueError, match="physics"):
        video.render({}, Path("missing"), tmp_path, lambda: None, final=True)


def test_repair_rechecks_entire_scene_and_keeps_attempts(tmp_path, monkeypatch):
    data, template = case(True)
    calls = []

    def simulator(current, out, check):
        rows = copy.deepcopy(template)
        for row in rows:
            for n, pose in current["poses"].items():
                row["objects"][n].update(copy.deepcopy(pose))
        if not calls:
            for row in rows[1001:]:
                row["objects"]["a"]["velocity"] = [0.02, 0, 0]
        calls.append(copy.deepcopy(current))
        (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "final_state.json").write_text(json.dumps(dict(state=rows[-1])))
        check()
        return physics.evaluate(current, rows)

    selected = dict(pose=dict(position=[0.05, 0, 0.7], orientation_wxyz=[1, 0, 0, 0]))
    monkeypatch.setattr(geo, "sample_object", lambda *args: (selected, {}))
    recorded = []
    result, attempts, _, passing = entry.repair_loop(
        data["assets"],
        data["poses"],
        dict(order=geo.topology(data["assets"]), relations=[]),
        [],
        0,
        tmp_path,
        lambda: None,
        simulator=simulator,
        recorder=lambda *a, **k: recorded.append(a),
    )
    assert result["passed"] and passing and len(attempts) == 2 and len(recorded) == 2
    assert calls[1]["poses"]["b"]["position"][0] == pytest.approx(0.05)
    assert calls[1]["poses"]["table"] == calls[0]["poses"]["table"]
    assert (tmp_path / "attempts/000/validation_result.json").exists()
    assert len(calls[1]["assets"]) == 3


def test_partial_collection_error_retains_attempt(tmp_path):
    data, rows = case()

    def broken(current, out, check):
        (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))
        raise ValueError("collector failed")

    with pytest.raises(ValueError, match="collector failed"):
        entry.repair_loop(
            data["assets"],
            data["poses"],
            dict(order=geo.topology(data["assets"]), relations=[]),
            [],
            0,
            tmp_path,
            lambda: None,
            simulator=broken,
            recorder=lambda *a, **k: pytest.fail("must not render"),
        )
    record = json.loads((tmp_path / "attempts.json").read_text())[0]
    assert record["steps_executed"] == 3 and record["execution_status"] == "error"
    assert (tmp_path / "attempts/000/trace.jsonl").exists()


@pytest.mark.parametrize("mode", ["passed", "exhausted", "tampered", "prepare_error"])
def test_task_lifecycle_source_readonly_and_final_gate(tmp_path, monkeypatch, mode):
    from test_task_output import make_task, snapshot

    source = make_task(tmp_path / "source")
    before = snapshot(source.root)
    index = tmp_path / "index" / "index.json"
    index.parent.mkdir()
    index.write_text("{}")
    data, template = case(True)
    document = dict(
        request=source.root.name,
        objects=[
            dict(object_id=n, category=a["category"], description=n)
            for n, a in data["assets"].items()
        ],
        relations=[
            dict(relation="on", source="a", target="table", evidence=""),
            dict(relation="on", source="b", target="a", evidence=""),
        ],
    )
    bindings = {n: dict(source_root=str(index.parent), source_files=[]) for n in data["assets"]}
    monkeypatch.setattr(entry, "bind_inputs", lambda *a: (document, bindings))

    def preparer(doc, b, out, fixed, metadata, check):
        check()
        if mode == "prepare_error":
            raise ValueError("invalid collision")
        assets = copy.deepcopy(data["assets"])
        for a in assets.values():
            a.update(derived_root=str(out), derived_files=[])
        return assets

    monkeypatch.setattr(entry, "make_initial", lambda *a: copy.deepcopy(data["poses"]))
    monkeypatch.setattr(
        geo, "sample_object", lambda assets, poses, name, *a: (dict(pose=poses[name]), {})
    )

    def simulator(current, out, check):
        rows = copy.deepcopy(template)
        if mode == "exhausted":
            for row in rows[1001:]:
                row["objects"]["a"]["velocity"] = [0.02, 0, 0]
        (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "final_state.json").write_text(json.dumps(dict(state=rows[-1])))
        if mode == "tampered":
            (out / "physics_input.json").write_text("{}")
        check()
        return physics.evaluate(current, rows)

    rendered = []

    def recorder(data, trace, out, check, **kw):
        check()
        out.mkdir(parents=True)
        (out / "fixture.json").write_text(json.dumps(kw))
        rendered.append(kw)

    target = tmp_path / "result"
    report = entry.run(
        source.root,
        target,
        index,
        fixed_objects=["table"],
        render=True,
        preparer=preparer,
        simulator=simulator,
        recorder=recorder,
    )
    assert snapshot(source.root) == before
    assert (
        report["exit_code"]
        == {"passed": 0, "exhausted": 2, "tampered": 1, "prepare_error": 1}[mode]
    )
    task = entry.TaskOutput(target)
    task.verify()
    assert bool(list(task.stage("final_render").rglob("*.json"))) == (mode == "passed")
    if mode == "exhausted":
        assert len(report["attempts"]) == 11
        assert len(list(task.stage("physics").glob("attempts/*/validation_result.json"))) == 11
    if mode == "passed":
        assert rendered[-1]["final"] and rendered[-1]["physics_passed"]


def test_internal_decomposition_seam_is_not_a_false_cavity_cap():
    whole = trimesh.creation.box([0.1, 0.1, 0.1])
    parts = []
    for x in [-0.025, 0.025]:
        part = trimesh.creation.box([0.05, 0.1, 0.1])
        part.apply_translation([x, 0, 0])
        parts.append(part)
    assert prep.proxy_quality(whole, parts)["passed"]


def test_transparent_collision_does_not_define_visual_truth(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "asset.xml"
    source.write_text(
        '<mujoco><worldbody><geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/>'
        '<geom type="box" size="1 1 1" rgba="1 0 0 0"/></worldbody></mujoco>'
    )
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    hidden = trimesh.creation.box([2, 2, 2])
    entity = SimpleNamespace(
        vgeoms=[
            SimpleNamespace(get_vverts=lambda m=m: m.vertices, init_vfaces=m.faces)
            for m in [visual, hidden]
        ]
    )
    vertices, faces = prep.visible_geometry(entity, source)
    assert np.allclose(np.ptp(vertices, axis=0), [0.1, 0.1, 0.1])
    assert not prep.proxy_quality(
        trimesh.Trimesh(vertices, faces), [hidden], native_semantics=True
    )["passed"]


def test_topology_orders_each_depth_by_size():
    assets = {
        "tiny": dict(diagonal_m=0.1, support="ground"),
        "large": dict(diagonal_m=2, support="ground"),
        "tall": dict(diagonal_m=3, support="tiny"),
    }
    assert geo.topology(assets) == ["large", "tiny", "tall"]


def test_cli_input_errors_use_exit_one():
    with pytest.raises(SystemExit) as error:
        entry.main([])
    assert error.value.code == 1


def test_collision_bottom_offset_is_used_in_sampling():
    data, _ = case()
    assets = data["assets"]
    assets["a"]["bottom_offset_m"] = 0.0002
    selected, _ = geo.sample_object(assets, {"table": data["poses"]["table"]}, "a", [], [], 0, 0)
    assert selected["pose"]["position"][2] == pytest.approx(0.7002)


@pytest.mark.parametrize("corruption", ["input", "trace", "verdict"])
def test_post_simulation_integrity_error_retains_executed_steps(tmp_path, corruption):
    data, rows = case()

    def simulator(current, out, check):
        result = physics.evaluate(current, rows)
        check()
        persisted = rows[:-1] if corruption == "trace" else rows
        (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in persisted))
        (out / "asset_physics_report.json").write_text(json.dumps(dict(steps_executed=1500)))
        if corruption == "input":
            (out / "physics_input.json").write_text("{}")
        if corruption == "verdict":
            result["passed"] = False
        return result

    with pytest.raises(ValueError):
        entry.repair_loop(
            data["assets"], data["poses"], dict(order=geo.topology(data["assets"]), relations=[]),
            [], 0, tmp_path, lambda: None, simulator=simulator,
            recorder=lambda *a, **k: pytest.fail("corrupt evidence must not render"),
        )
    attempts = json.loads((tmp_path / "attempts.json").read_text())
    assert len(attempts) == 1
    assert attempts[0]["steps_executed"] == 1500
    assert attempts[0]["execution_status"] == "error"


@pytest.mark.parametrize("x,z,intersects", [(0, .15, False), (.45, .15, True), (0, .04, True)])
def test_nonconvex_raised_rim_does_not_fill_recess(x, z, intersects):
    base = trimesh.creation.box([1, 1, .1])
    rim = trimesh.creation.box([.1, 1, .2])
    rim.apply_translation([.45, 0, .1])
    parent = trimesh.util.concatenate([base, rim])
    child = trimesh.creation.box([.1, .1, .2])
    child.apply_translation([x, 0, z])
    # The outer hull overlaps even for a valid placement on the recessed surface.
    assert geo.overlap_depth(parent.vertices, child.vertices) > 1e-6
    assert geo.mesh_intersects_convex(parent.vertices, parent.faces, child.vertices) is intersects


def test_nonconvex_mesh_contains_small_body_without_surface_crossing():
    parent = trimesh.creation.box([1, 1, 1])
    child = trimesh.creation.box([.1, .1, .1])
    assert geo.mesh_intersects_convex(parent.vertices, parent.faces, child.vertices)



def test_near_coplanar_vertex_survives_float32_geometry_check():
    from scipy.spatial import ConvexHull, cKDTree

    box = trimesh.creation.box([1, 1, 1]).vertices
    expected = np.vstack([box, [0, 0, .50000001]])
    actual = expected.astype(np.float32).astype(float)
    old_error = cKDTree(actual[ConvexHull(actual).vertices]).query(
        expected[ConvexHull(expected).vertices]
    )[0].max()
    assert old_error > .1  # Hull vertex membership is unstable, although the mesh barely changed.
    assert physics.vertex_set_error(actual, expected) < 1e-6


@pytest.mark.parametrize("bad_count,passed", [(124, True), (125, False), (126, False)])
@pytest.mark.parametrize("metric", ["speed", "support"])
def test_strict_75_percent_profile(bad_count, passed, metric):
    data, rows = case()
    data = physics.frozen_input(data["assets"], data["poses"], [], 0,
                                profile=physics.GT75_PROFILE)
    for row in rows[-bad_count:]:
        if metric == "speed":
            row["objects"]["a"]["velocity"] = [0.01, 0, 0]
        else:
            row["contacts"] = row["contacts"][1:]
    result = physics.evaluate(data, rows)
    assert result["passed"] is passed
    assert result["profile"] == physics.GT75_PROFILE
    if metric == "support":
        # Only a loses contact; the second dynamic object b remains supported.
        assert result["support_preservation_ratio"] == (1.0 if passed else 0.5)


def test_75_profile_changes_only_the_two_acceptance_fractions():
    old = physics.settings()
    new = physics.settings(physics.GT75_PROFILE)
    assert {k for k in new if new[k] != old.get(k)} == {
        "profile", "stable_fraction", "support_fraction", "fraction_comparison"}
    assert old == physics.SETTINGS
    data, rows = case()
    data["settings"] = new
    with pytest.raises(ValueError, match="settings mismatch"):
        physics.evaluate(data, rows)
    data["profile"] = physics.GT75_PROFILE
    rows[0]["contacts"][0]["penetration"] = 0.003
    assert "penetration" in physics.evaluate(data, rows)["failures"]["a"]

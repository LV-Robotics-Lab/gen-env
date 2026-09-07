# ruff: noqa: E402
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTER))
import prepare_cases as cases
import validate_physics as physics


def example(tmp_path):
    package = cases.make_package("box_on_table")
    compiled = physics.GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)
    scene = json.loads(Path(compiled.artifact_path).read_text())
    return package, compiled, scene


def trace(package):
    cfg = package.metadata["genesis_physics"]["settings"]
    state = dict(
        position=[0, 0, 0.015],
        orientation_wxyz=[1, 0, 0, 0],
        velocity=[0, 0, 0],
        angular_velocity=[0, 0, 0],
        net_contact_force=[0, 0, 0.26487],
    )
    contact = dict(
        a="table",
        b="box",
        geom_a=0,
        geom_b=1,
        link_a=0,
        link_b=1,
        position=[0, 0, 0],
        normal=[0, 0, 1],
        penetration=0.0001,
        force_a=[0, 0, -0.26487],
        force_b=[0, 0, 0.26487],
    )
    rows = [
        dict(
            step=i,
            time_s=i * cfg["dt"],
            objects={"box": deepcopy(state)},
            contacts=[deepcopy(contact)] if i else [],
        )
        for i in range(cfg["steps"] + 1)
    ]
    rows[0]["objects"]["box"]["position"][2] = 0.017
    return rows


def test_pass_and_settled_package_preserves_contract(tmp_path):
    package, compiled, scene = example(tmp_path)
    assert physics.import_compile_manifest(compiled.manifest_path) == package
    rows = trace(package)
    assert all(c["passed"] for c in physics.evaluate(package, scene, rows))
    final = physics.settled_package(package, rows[-1]["objects"])
    assert final.digest() != package.digest()
    assert final.assets == package.assets and final.task == package.task
    assert final.metadata == package.metadata
    assert final.env.objects[0].pose.position == (0, 0, 0.015)
    assert final.env.objects[0].static is False


@pytest.mark.parametrize(
    "attack", ["initial_penetration", "no_contact", "side_force", "intermittent", "drift"]
)
def test_false_positives_fail(tmp_path, attack):
    package, _, scene = example(tmp_path)
    rows = trace(package)
    if attack == "initial_penetration":
        rows[0]["contacts"] = deepcopy(rows[1]["contacts"])
        rows[0]["contacts"][0]["penetration"] = 0.005
    elif attack == "no_contact":
        for r in rows:
            r["contacts"] = []
    elif attack == "side_force":
        for r in rows[1:]:
            r["contacts"][0]["force_b"] = [1, 0, 0]
    elif attack == "intermittent":
        for r in rows[-50:]:
            r["contacts"] = []
    else:
        rows[-50]["objects"]["box"]["position"][0] = 0.005
    assert not all(c["passed"] for c in physics.evaluate(package, scene, rows))


@pytest.mark.parametrize(
    "attack", ["missing_row", "missing_contact", "nan", "missing_object", "time", "nan_time"]
)
def test_incomplete_records_fail_closed(tmp_path, attack):
    package, _, scene = example(tmp_path)
    rows = trace(package)
    if attack == "missing_row":
        rows.pop(1)
    elif attack == "missing_contact":
        del rows[4]["contacts"]
    elif attack == "nan":
        rows[4]["objects"]["box"]["velocity"][0] = float("nan")
    elif attack == "missing_object":
        rows[4]["objects"] = {}
    elif attack == "nan_time":
        rows[4]["time_s"] = float("nan")
    else:
        rows[4]["time_s"] += 1
    with pytest.raises((ValueError, KeyError)):
        physics.evaluate(package, scene, rows)


@pytest.mark.parametrize("attack", ["fixed", "collision", "unknown_condition"])
def test_bad_configuration_rejected(tmp_path, attack):
    package = cases.make_package("box_on_table")
    if attack == "fixed":
        package = replace(
            package,
            env=replace(package.env, objects=(replace(package.env.objects[0], static=True),)),
        )
    elif attack == "collision":
        package.metadata["genesis_physics"]["bodies"]["box"]["collision"] = False
    else:
        package = replace(
            package, task=replace(package.task, success=({"type": "unbound", "object": "box"},))
        )
    compiled = physics.GenesisCompiler().compile(package, tmp_path, strict=True)
    with pytest.raises(ValueError):
        physics.config_for(package, json.loads(Path(compiled.artifact_path).read_text()))


def inside_example(tmp_path):
    # Dependency-free boxes serve as test geometry, not official runtime evidence.
    package = cases.make_package("box_on_table")
    obj = package.env.objects[0]
    target = replace(obj, instance_id="container", static=True)
    package = replace(
        package,
        env=replace(package.env, objects=(obj, target)),
        task=replace(
            package.task, success=({"type": "inside", "object": "box", "target": "container"},)
        ),
    )
    package.metadata["genesis_physics"]["bodies"]["container"] = dict(
        collision=True,
        density=1000,
        friction=0.5,
        local_bounds=[[-0.02] * 3, [0.02] * 3],
        interior_bounds=[[-0.016] * 3, [0.016] * 3],
        geometry_measurement_sha256="a" * 64,
    )
    compiled = physics.GenesisCompiler().compile(package, tmp_path, strict=True)
    rows = trace(package)
    for r in rows:
        r["objects"]["box"]["position"] = [0, 0, 0]
        r["objects"]["container"] = deepcopy(r["objects"]["box"])
        for c in r["contacts"]:
            c["a"] = "container"
    rows[0]["objects"]["box"]["position"][2] = 0.04
    return package, json.loads(Path(compiled.artifact_path).read_text()), rows


@pytest.mark.parametrize("attack", ["corner_outside", "sealed_opening", "table_contact"])
def test_containment_attacks(tmp_path, attack):
    package, scene, rows = inside_example(tmp_path)
    assert all(c["passed"] for c in physics.evaluate(package, scene, rows))
    if attack == "corner_outside":
        for r in rows:
            r["objects"]["box"]["position"][0] = 0.003
    elif attack == "sealed_opening":
        for r in rows:
            r["objects"]["box"]["position"][2] = 0.032
    else:
        for r in rows[1:]:
            r["contacts"][0]["a"] = "table"
    assert not all(c["passed"] for c in physics.evaluate(package, scene, rows))


def test_failure_never_calls_renderer_and_new_output_required(tmp_path, monkeypatch):
    package, compiled, _ = example(tmp_path)
    rows = trace(package)
    for r in rows:
        r["contacts"] = []
    monkeypatch.setattr(physics, "simulate", lambda *args: (rows, {}))
    monkeypatch.setattr(
        physics.subprocess, "run", lambda *a, **k: pytest.fail("renderer was called")
    )
    out = tmp_path / "failed"
    assert physics.run(compiled.manifest_path, out) == 1
    report = json.loads((out / "physics_result.json").read_text())
    assert report["physics_status"] == "failed" and report["render_status"] == "not_run"
    with pytest.raises(FileExistsError):
        physics.run(compiled.manifest_path, out)


def test_tampered_scene_never_simulates(tmp_path, monkeypatch):
    _, compiled, scene = example(tmp_path)
    scene["objects"][0]["pose"]["position"][0] += 0.1
    physics.write_json(compiled.artifact_path, scene)
    monkeypatch.setattr(physics, "simulate", lambda *a: pytest.fail("tampered input simulated"))
    assert physics.run(compiled.manifest_path, tmp_path / "failed") == 1


def test_render_failure_preserves_physical_success(tmp_path, monkeypatch):
    package, compiled, _ = example(tmp_path)
    rows = trace(package)

    def simulate(package, spec, out, raw):
        (out / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return rows, {}

    monkeypatch.setattr(physics, "simulate", simulate)
    monkeypatch.setattr(physics.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7))
    out = tmp_path / "render_fail"
    assert physics.run(compiled.manifest_path, out) == 1
    report = json.loads((out / "physics_result.json").read_text())
    assert report["physics_status"] == "passed" and report["render_status"] == "failed"
    terminal = physics.import_compile_manifest(report["settled_compile_manifest"])
    assert terminal.env.objects[0].pose.position == (0, 0, 0.015)


def test_terminal_object_mismatch_rejected(tmp_path):
    package, _, _ = example(tmp_path)
    with pytest.raises(ValueError, match="object set"):
        physics.settled_package(package, {})


def test_even_small_containment_overflow_fails(tmp_path):
    package, scene, rows = inside_example(tmp_path)
    for row in rows[1:]:
        row["objects"]["box"]["position"][0] = 0.00101
    assert not all(c["passed"] for c in physics.evaluate(package, scene, rows))


def test_early_table_contact_fails_containment(tmp_path):
    package, scene, rows = inside_example(tmp_path)
    rows[1]["contacts"][0]["a"] = "table"
    checks = physics.evaluate(package, scene, rows)
    assert not next(c for c in checks if c["name"] == "box.never_touched_table")["passed"]


def test_spawn_inside_is_not_entry(tmp_path):
    package, scene, rows = inside_example(tmp_path)
    rows[0]["objects"]["box"]["position"][2] = 0
    checks = physics.evaluate(package, scene, rows)
    assert not next(c for c in checks if c["name"] == "box.entered_from_above")["passed"]


def test_tampered_asset_never_simulates(tmp_path, monkeypatch):
    urdf = tmp_path / "fixture.urdf"
    urdf.write_text(
        '<robot name="fixture"><link name="body">'
        '<visual><geometry><box size="0.03 0.03 0.03"/></geometry></visual>'
        '<collision><geometry><box size="0.03 0.03 0.03"/></geometry></collision>'
        "</link></robot>"
    )
    package = cases.make_package("box_on_table")
    asset = replace(
        package.assets[0],
        representations=(
            cases.AssetRepresentation(
                format="urdf",
                backend="genesis",
                role="visual_and_collision",
                **cases.fingerprint(urdf),
                metadata=cases._dependency_metadata(urdf),
            ),
        ),
    )
    package = replace(package, assets=(asset,))
    package.metadata["genesis_physics"]["bodies"]["box"]["collision_parts"] = 1
    compiled = physics.GenesisCompiler().compile(package, tmp_path / "input", strict=True)
    urdf.write_text(urdf.read_text() + "\n<!-- changed -->")
    monkeypatch.setattr(physics, "simulate", lambda *a: pytest.fail("tampered asset simulated"))
    assert physics.run(compiled.manifest_path, tmp_path / "output") == 1


def test_contact_read_exception_is_recorded(tmp_path, monkeypatch):
    _, compiled, _ = example(tmp_path)

    def broken(*args):
        raise RuntimeError("contact read failed")

    monkeypatch.setattr(physics, "simulate", broken)
    monkeypatch.setattr(physics.subprocess, "run", lambda *a, **k: pytest.fail("render called"))
    out = tmp_path / "failed"
    assert physics.run(compiled.manifest_path, out) == 1
    report = json.loads((out / "physics_result.json").read_text())
    assert report["error"] == "RuntimeError: contact read failed"
    assert report["input_package_digest"]
    assert report["input_scene_sha256"]
    assert report["compile_manifest_sha256"]
    assert report["physics_status"] == "failed"
    assert report["render_status"] == "not_run"


def test_terminal_compiler_pose_corruption_stops_render(tmp_path, monkeypatch):
    package, compiled, _ = example(tmp_path)
    rows = trace(package)
    monkeypatch.setattr(physics, "simulate", lambda *a: (rows, {}))
    original = physics.GenesisCompiler.compile

    def corrupt(self, package, out, **kwargs):
        result = original(self, package, out, **kwargs)
        if Path(out).name != "settled":
            return result
        spec = json.loads(Path(result.artifact_path).read_text())
        spec["objects"][0]["pose"]["position"][0] += 0.01
        physics.write_json(result.artifact_path, spec)
        return result

    monkeypatch.setattr(physics.GenesisCompiler, "compile", corrupt)
    monkeypatch.setattr(physics.subprocess, "run", lambda *a, **k: pytest.fail("render called"))
    out = tmp_path / "failed"
    assert physics.run(compiled.manifest_path, out) == 1
    report = json.loads((out / "physics_result.json").read_text())
    assert report["render_status"] == "not_run"
    assert report["physics_status"] == "passed"


def test_input_changed_during_physics_stops_render(tmp_path, monkeypatch):
    package, compiled, _ = example(tmp_path)

    def simulate(*args):
        scene = Path(compiled.artifact_path)
        scene.write_text(scene.read_text() + "\n")
        return trace(package), {}

    monkeypatch.setattr(physics, "simulate", simulate)
    monkeypatch.setattr(physics.subprocess, "run", lambda *a, **k: pytest.fail("render called"))
    out = tmp_path / "failed"
    assert physics.run(compiled.manifest_path, out) == 1
    report = json.loads((out / "physics_result.json").read_text())
    assert "input changed" in report["error"]
    assert report["physics_status"] == "passed"
    assert report["render_status"] == "not_run"

"""Opt-in, real CPU acceptance. Known failures are failures, never xfails.

GENESIS_PHYSICS_REAL=1 GENESIS_PHYSICS_OUTPUT=/new/path pytest -q <this file>
"""

# ruff: noqa: E402
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_cases as cases
import validate_physics as physics

pytestmark = pytest.mark.skipif(
    os.environ.get("GENESIS_PHYSICS_REAL") != "1", reason="opt-in real Genesis acceptance"
)
CASES = ("box_on_table", "mug_on_table", "box_in_mug")


@pytest.fixture(scope="module")
def prepared():
    out = Path(os.environ["GENESIS_PHYSICS_OUTPUT"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mug = cases.convert_mug(
        os.environ.get("GENESIS_MUG_DIR", physics.ROOT / "data/genesis-official-assets/mug_1"),
        out / "mug",
    )
    inputs = {}
    for dt in (0.004, 0.002):
        for name in CASES:
            package = cases.make_package(name, mug, dt=dt)
            compiled = physics.GenesisCompiler().compile(
                package, out / "inputs" / f"{name}_{dt}", strict=True
            )
            inputs[name, dt] = (package, compiled)
    return out, inputs


def run_case(compiled, out):
    with out.with_suffix(".log").open("x") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(physics.__file__)),
                "--compile-manifest",
                compiled.manifest_path,
                "--output-dir",
                str(out),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
    report = json.loads((out / "physics_result.json").read_text())
    assert result.returncode == (0 if report["status"] == "success" else 1)
    if report["physics_status"] != "passed":
        assert report["render_status"] == "not_run"
        assert not (out / "render.log").exists()
        assert not (out / "settled").exists()
    return report


@pytest.mark.parametrize("name", CASES)
def test_three_repeats(prepared, name):
    out, inputs = prepared
    reports = [run_case(inputs[name, 0.004][1], out / f"{name}_repeat_{i}") for i in range(3)]
    # Run all three even if the first fails, retaining evidence of reproducibility.
    assert len({r["trace_sha256"] for r in reports}) == 1, "trajectory not deterministic"
    assert all(r["physics_status"] == "passed" for r in reports), [
        r.get("failure_reasons", r.get("error")) for r in reports
    ]
    assert all(r["render_status"] == "passed" for r in reports)
    assert len({r["settled_package_digest"] for r in reports}) == 1


@pytest.mark.parametrize("name", CASES)
def test_half_dt_same_duration(prepared, name):
    out, inputs = prepared
    report = run_case(inputs[name, 0.002][1], out / f"{name}_sensitivity")
    assert report["physics_status"] == "passed", report.get("failure_reasons", report.get("error"))
    assert report["render_status"] == "passed"


@pytest.mark.parametrize("attack", ["fixed_floating", "collision_off", "deep_penetration", "lid"])
def test_rejects_without_rendering(prepared, attack):
    out, inputs = prepared
    name = "box_in_mug" if attack == "lid" else "box_on_table"
    package = deepcopy(inputs[name, 0.004][0])
    objects = list(package.env.objects)
    if attack == "fixed_floating":
        objects[0] = replace(objects[0], static=True)
    elif attack == "collision_off":
        package.metadata["genesis_physics"]["bodies"]["box"]["collision"] = False
    elif attack == "deep_penetration":
        objects[0] = replace(objects[0], pose=cases.Pose(position=(0.0, 0.0, 0.005)))
    else:
        # A fixed Genesis Box physically seals the official mug opening.
        box = next(o for o in objects if o.instance_id == "box")
        mug = next(o for o in objects if o.instance_id == "mug")
        z = (
            mug.pose.position[2]
            + package.metadata["genesis_physics"]["bodies"]["mug"]["local_bounds"][1][2]
        )
        objects.append(
            replace(
                box,
                instance_id="lid",
                static=True,
                pose=cases.Pose(position=(0.0, 0.0, z - 0.001)),
                scale=(12.0, 12.0, 0.25),
            )
        )
        package.metadata["genesis_physics"]["bodies"]["lid"] = dict(
            collision=True,
            density=1000,
            friction=0.5,
            local_bounds=[[-0.048, -0.048, -0.001], [0.048, 0.048, 0.001]],
        )
    package = replace(package, env=replace(package.env, objects=tuple(objects)))
    package = cases.EnvironmentPackage.from_dict(package.to_dict())
    compiled = physics.GenesisCompiler().compile(package, out / f"attack_{attack}", strict=True)
    report = run_case(compiled, out / f"reject_{attack}")
    assert report["physics_status"] == "failed"
    if attack in {"deep_penetration", "lid"}:
        assert report["physical_runtime_evidence"], report.get("error")

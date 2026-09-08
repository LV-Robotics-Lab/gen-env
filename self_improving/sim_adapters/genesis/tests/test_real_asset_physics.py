"""Opt-in real native physics acceptance; normal-scene failures remain test failures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import validate_asset_scene as entry
from self_improving.sim_adapters.genesis.task_output import TaskOutput

pytestmark = pytest.mark.skipif(
    os.environ.get("GENESIS_ASSET_PHYSICS_REAL") != "1",
    reason="opt-in real native asset physics acceptance",
)
SCENE = Path("output/桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。")
INDEX = Path("assets/genesis/clip_non_robot_v1/index.json")
ADAPTER = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def acceptance_root(tmp_path_factory):
    root = Path(
        os.environ.get(
            "GENESIS_ASSET_PHYSICS_OUTPUT",
            str(tmp_path_factory.mktemp("native_physics") / "acceptance"),
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root


def original_trial(root, name, profile):
    source = TaskOutput(Path(os.environ.get("GENESIS_ASSET_PHYSICS_SCENE", str(SCENE))))
    task = source.copy_for_physics(root / name)
    before = {
        s: {
            p.relative_to(task.stage(s)).as_posix(): library.sha256(p)
            for p in task.stage(s).rglob("*")
            if p.is_file()
        }
        for s in ("objects", "scene")
    }
    with (root / f"{name}.log").open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(ADAPTER / "validate_asset_scene.py"),
                "--scene-dir",
                str(task.root),
                "--clip-index",
                str(INDEX),
                "--fixed-object",
                "table_1",
                "--profile",
                profile,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    task.verify()
    task.verify_scene_inputs()
    for stage, files in before.items():
        assert {
            p.relative_to(task.stage(stage)).as_posix(): library.sha256(p)
            for p in task.stage(stage).rglob("*")
            if p.is_file()
        } == files
    assert not list(task.stage("final_render").iterdir())
    report = library.read_json(task.stage("physics") / "physics_result.json")
    assert completed.returncode == report["exit_code"]
    official.verify_files(task.stage("physics"), report["artifacts"])
    assert report["simulation_executed"] and report["steps_executed"] == (
        1000 if profile == "baseline" else 2000
    )
    assert report["last_complete_step"] == report["steps_executed"]
    return report


def test_four_assets_three_repeats(acceptance_root):
    reports = [original_trial(acceptance_root, f"baseline_{i}", "baseline") for i in range(1, 4)]
    entry.write_json(acceptance_root / "repeat_summary.json", reports)
    assert all(r["physics_status"] == "passed" for r in reports), (
        "original physical failures retained"
    )


def test_four_assets_half_timestep(acceptance_root):
    report = original_trial(acceptance_root, "half_dt", "half_dt")
    assert report["physics_status"] == "passed", "half-dt physical failure retained"


@pytest.mark.parametrize(
    "case",
    [
        "calibration",
        "mesh_calibration",
        "margin_slack",
        "three_levels",
        "deep_penetration",
        "fixed_suspension",
        "disabled_collision",
        "wrong_target",
        "moving",
    ],
)
def test_real_fixtures(acceptance_root, case):
    out = acceptance_root / case
    with (acceptance_root / f"{case}.log").open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(ADAPTER / "asset_physics_cases.py"),
                "--case",
                case,
                "--output",
                str(out),
                "--clip-index",
                str(INDEX),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = library.read_json(out / "physics_result.json")
    official.verify_files(out, report["files"])
    assert report["render_status"] == "not_run"
    assert not list(out.rglob("*.png")) and not list(out.rglob("*.mp4"))
    if case in ("calibration", "mesh_calibration", "margin_slack", "three_levels"):
        assert completed.returncode == 0 and report["status"] == "physics_passed", report
        assert report["steps_executed"] == 1000
        if case == "three_levels":
            loaded = library.read_json(out / "asset_physics_report.json")["bodies"]
            assert loaded["middle"]["fixed"] is False and loaded["middle"]["dofs"] == 6
    else:
        assert completed.returncode == 2 and report["status"] == "physics_failed"
        if case in ("fixed_suspension", "disabled_collision", "deep_penetration"):
            assert not report["simulation_executed"]
            if case == "deep_penetration":
                assert any(
                    c["name"] == "initial.penetration" and not c["passed"] for c in report["checks"]
                )
            else:
                assert ("fixed suspension" if case == "fixed_suspension" else "collision") in (
                    report["error"]
                )
        else:
            assert report["simulation_executed"] and report["steps_executed"] == 1000
            suffix = ".support" if case == "wrong_target" else ".settled"
            assert any(c["name"].endswith(suffix) and not c["passed"] for c in report["checks"])

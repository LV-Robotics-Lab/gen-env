"""Scene import: asymmetric geometry, identity attacks and portable replay inputs."""

import copy
import json
import shutil

import numpy as np
import pytest

from self_improving.sim_adapters.genesis import import_simfoundry_assets as assets
from self_improving.sim_adapters.genesis import import_simfoundry_scene as scenes
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis.tests.test_standard_urdf import source


@pytest.fixture
def case(tmp_path):
    scene, entry = source(tmp_path)
    assets.import_assets(scene, tmp_path / "library")
    state = dict(
        state=dict(
            registry=dict(
                object_registry={
                    "iter_0": dict(
                        root_link=dict(
                            pos=[1, 2, 3],
                            ori=[0, 0, 2**-0.5, 2**-0.5],
                            lin_vel=[0.1, 0, 0],
                            ang_vel=[0, 0, 0.2],
                        )
                    ),
                    "robot0": dict(root_link=dict(pos=[0, 0, 0], ori=[0, 0, 0, 1])),
                }
            )
        ),
        objects_info=dict(
            init_info={
                "iter_0": dict(
                    args=dict(name="iter_0", category="box", model="model", scale=[1, 1, 1])
                ),
                "robot0": dict(class_module="omnigibson.robots.franka", args={}),
            }
        ),
        ground_plane_info=dict(position=[0, 0, 0.4], orientation=[0, 0, 0, 1], visible=False),
    )
    path = scene / "s14_og/reconstructed_og_scene.json"
    path.parent.mkdir()
    path.write_text(json.dumps(state))
    return scene, tmp_path / "library/library.json", tmp_path / "result", state, entry


def run(case, state=None, **kwargs):
    scene, library, out, original, _ = case
    (scene / "s14_og/reconstructed_og_scene.json").write_text(
        json.dumps(original if state is None else state)
    )
    return scenes.convert(scene, library, out, **kwargs)


def test_rotation_root_frame_portability_and_no_inferred_support(case, tmp_path):
    report = run(case)
    layout = scenes.verify(case[2])
    obj = layout["objects"][0]
    measured = standard.inspect(case[4])
    # Known +90 degree yaw: (x, y, z) -> (-y, x, z), then world translation.
    expected = measured["visual"][:, [1, 0, 2]] * [-1, 1, 1] + [1, 2, 3]
    np.testing.assert_allclose(obj["world_visual_bounds_m"], [expected.min(0), expected.max(0)])
    np.testing.assert_allclose(obj["orientation_wxyz"], [2**-0.5, 0, 0, 2**-0.5])
    assert obj["translation_m"] == [1, 2, 3]  # No second COM offset.
    assert obj["source_velocity_mps"] == [0.1, 0, 0]
    assert obj["intended_dynamic"] and not obj["fixed"]
    assert obj["support"] is None and layout["relations"] == []
    assert layout["environment"]["z_m"] == 0.4
    assert layout["environment"]["visible"] is False
    assert report["excluded_objects"][0]["object_id"] == "robot0"
    assert report["physics_status"] == "not_run"
    moved = tmp_path / "moved"
    shutil.copytree(case[2], moved)
    shutil.rmtree(case[0])
    shutil.rmtree(case[1].parent)
    shutil.rmtree(case[2])
    assert scenes.verify(moved) == layout


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_pose", "declarations and states"),
        ("missing_object", "missing source objects"),
        ("unknown_object", "no source metadata"),
        ("wrong_model", "identity"),
        ("scale", "resized"),
        ("mass", "physics overrides"),
        ("friction", "physics overrides"),
        ("joints", "articulated"),
        ("quaternion", "quaternion"),
        ("nan", "invalid_json"),
        ("bool", "numeric vector"),
        ("usd", "custom USD"),
    ],
)
def test_invalid_or_unrepresented_source_never_silently_accepted(case, change, match):
    state = copy.deepcopy(case[3])
    args = state["objects_info"]["init_info"]["iter_0"]["args"]
    states = state["state"]["registry"]["object_registry"]
    if change == "missing_pose":
        del states["iter_0"]
    elif change == "missing_object":
        del states["iter_0"]
        del state["objects_info"]["init_info"]["iter_0"]
    elif change == "unknown_object":
        states["unknown"] = copy.deepcopy(states["iter_0"])
        state["objects_info"]["init_info"]["unknown"] = dict(args={})
    elif change == "wrong_model":
        args["model"] = "another"
    elif change == "scale":
        args["scale"] = [1, 2, 1]
    elif change == "mass":
        args["mass"] = 0.7
    elif change == "friction":
        args["link_physics_materials"] = {}
    elif change == "joints":
        states["iter_0"]["joint_pos"] = [0.1]
    elif change == "quaternion":
        states["iter_0"]["root_link"]["ori"] = [0, 0, 0, 2]
    elif change == "nan":
        states["iter_0"]["root_link"]["pos"][0] = float("nan")
    elif change == "bool":
        states["iter_0"]["root_link"]["pos"][0] = True
    elif change == "usd":
        args["usd_path"] = "other.usd"
    with pytest.raises(ValueError, match=match):
        run(case, state)
    assert not case[2].exists()


@pytest.mark.parametrize("target", ["scene_layout.json", "scene_graph.json", "asset"])
def test_output_tampering_rejected(case, target):
    run(case)
    if target == "asset":
        target = scenes.verify(case[2])["objects"][0]["model_entrypoint"]
    with (case[2] / target).open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError):
        scenes.verify(case[2])


def test_wrong_library_and_overwrite_rejected(case):
    library = json.loads(case[1].read_text())
    library["source_metadata"]["sha256"] = "0" * 64
    case[1].write_text(json.dumps(library))
    with pytest.raises(ValueError, match="metadata mismatch"):
        run(case)
    case[2].mkdir()
    sentinel = case[2] / "user-file"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run(case)
    assert sentinel.read_text() == "preserve"


def test_explicit_pybullet_root_link_poses(case):
    source_file = case[0] / "pb.json"
    source_file.write_text(json.dumps({"iter_0": [[0.1, 0.2, 0.3], [0, 0, 0, 1]]}))
    run(case, scene_file=source_file, pose_format="pybullet")
    obj = scenes.verify(case[2])["objects"][0]
    assert obj["translation_m"] == [0.1, 0.2, 0.3]
    assert obj["orientation_wxyz"] == [1, 0, 0, 0]


def test_cli_verify_and_failed_import(case, capsys):
    run(case)
    assert scenes.main(["verify", "--scene-package", str(case[2])]) == 0
    assert '"physics_status": "not_run"' in capsys.readouterr().out
    assert (
        scenes.main(
            [
                "import",
                "--scene-dir",
                str(case[0]),
                "--library-path",
                str(case[1]),
                "--output-dir",
                str(case[2]),
            ]
        )
        == 1
    )


def test_explicit_unknown_exclusion_fails(case):
    with pytest.raises(ValueError, match="unknown excluded"):
        run(case, exclude=["typo"])


def test_original_geometry_changed_since_asset_conversion(case):
    with case[4].open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError):
        run(case)
    assert not case[2].exists()

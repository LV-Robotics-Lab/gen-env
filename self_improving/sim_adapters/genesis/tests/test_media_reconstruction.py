"""Media reconstruction contracts without running remote models or Genesis."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from self_improving.sim_adapters.genesis import asset_physics, clip_select, standard_urdf
from self_improving.sim_adapters.genesis import media_support as support
from self_improving.sim_adapters.genesis import reconstruct_media as media
from self_improving.sim_adapters.genesis import validate_imported_scene as physics

ROOT = Path(__file__).resolve().parents[4]


def test_requested_mouse_media_metadata():
    fixture_root = ROOT / "test"
    if not (fixture_root / "鼠标.jpg").is_file():
        fixture_root = ROOT.parent
    image = fixture_root / "鼠标.jpg"
    video = fixture_root / "鼠标视频.mp4"
    if not image.is_file() or not video.is_file():
        pytest.skip("user-provided media fixtures are local-only")
    image_report = media.probe_media(image, "image")
    video_report = media.probe_media(video, "video")
    assert image_report["sha256"] == (
        "3a3b65d558c9fc7b61d95071ad7beaf2fd90853b292d6c9cb10ecbbdbd5cb476"
    )
    assert image_report["decoded_frame_count"] == image_report["unique_frame_count"] == 1
    assert video_report["sha256"] == (
        "5c7aa9c809aaff9c5f4d3076502d99cba0f72e46754102184b9c5c9db8d88634"
    )
    assert video_report["decoded_frame_count"] == 124
    assert video_report["unique_frame_count"] == 124
    assert video_report["sampled_frame_count"] == 15
    assert video_report["sampled_unique_frame_count"] == 15
    assert video_report["sampled_frame_indices"] == list(range(0, 120, 8))


def test_image_normalization_and_simfoundry_overrides(tmp_path):
    source = tmp_path / "input.JPEG"
    Image.new("RGB", (12, 8), (1, 2, 3)).save(source)
    original, canonical, runner_input = media._normalize(source, "image", tmp_path / "input")
    assert original.name == "original.jpeg"
    assert canonical.name == "source.png"
    assert runner_input.name == "source.MOV"
    assert not runner_input.exists()
    assert Image.open(canonical).size == (12, 8)
    command = media.simfoundry_command(tmp_path, runner_input, "image", ())
    assert "s1_video.single_image_input=true" in command
    assert "s1_video.n_subsampled_frames=1" in command
    assert "s3_ground.img_idx=0" in command
    assert "--bg-splat" not in command


@pytest.mark.parametrize("mode,suffix", [("image", ".mp4"), ("video", ".jpg")])
def test_wrong_media_extension_rejected(tmp_path, mode, suffix):
    path = tmp_path / f"wrong{suffix}"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="extension"):
        media.probe_media(path, mode)


def test_failed_reconstruction_keeps_four_stage_contract_and_empty_final(tmp_path):
    source = tmp_path / "mouse.jpg"
    Image.new("RGB", (20, 10), (220, 180, 160)).save(source)
    index, config = tmp_path / "index.json", tmp_path / "config.yaml"
    index.write_text("{}")
    config.write_text("profiles: {}")
    output = tmp_path / "output/mouse_image"

    def failed_runner(*args, **kwargs):
        return SimpleNamespace(returncode=7)

    result = media.run(
        source, "image", output, index, config, process_runner=failed_runner
    )
    assert result["status"] == "execution_failed"
    assert result["failure_phase"] == "simfoundry"
    assert result["stages"] == {
        "objects": "failed", "scene": "not_run",
        "physics": "not_run", "final_render": "not_run",
    }
    assert list((output / "04_final_render").iterdir()) == []
    from self_improving.sim_adapters.genesis.task_output import TaskOutput

    TaskOutput(output).verify()
    with pytest.raises(FileExistsError):
        media.run(source, "image", output, index, config, process_runner=failed_runner)
    Image.new("RGB", (20, 10), (1, 2, 3)).save(source)
    with pytest.raises(ValueError, match="resume"):
        media.run(
            source, "image", output, index, config, resume=True,
            process_runner=failed_runner,
        )


def test_support_observation_world_binding_and_censored_edges(tmp_path):
    scene = tmp_path / "scene"
    stage3, stage4 = scene / "s3_ground", scene / "s4_frame"
    stage3.mkdir(parents=True)
    stage4.mkdir()
    mask = np.zeros((4, 5), np.uint8)
    mask[1:, :3] = 1
    rgb = np.full((4, 5, 3), [160, 120, 80], np.uint8)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], np.float32
    )
    Image.fromarray(mask * 255).save(stage3 / "support_mask.png")
    np.savez_compressed(
        stage3 / "support_observation.npz",
        rgb=rgb, depth=np.ones((4, 5), np.float32),
        intrinsics=np.eye(3), mask=mask, points_camera=points,
        plane_inlier_indices=np.arange(4),
    )
    (stage3 / "image_0_floor_info.json").write_text(json.dumps(
        dict(
            floor_category="desk", origin=[0, 0, 0], z_dir=[0, 0, 1],
            support_mask="support_mask.png",
            support_observation="support_observation.npz",
        )
    ))
    transform = np.eye(4)
    transform[:2, 3] = [3, 4]
    np.save(stage4 / "image_0_cam2world.npy", transform)
    report = support.observe(scene, tmp_path / "observation")
    assert report["visible_footprint_world_xy_m"] == [[3.0, 4.0], [4.0, 6.0]]
    assert report["boundary_censored"] == {
        "top": False, "bottom": True, "left": True, "right": False
    }
    assert report["extent_evidence"] == ["inferred", "inferred"]
    arrays = np.load(tmp_path / "observation/support_observation.npz")
    assert arrays["points_world"].shape == (4, 3)


def table_mesh(path):
    import trimesh

    mesh = trimesh.creation.box(extents=[2.0, 0.7, 1.0])
    mesh.export(path)
    return mesh


def test_finite_support_geometry_materialization_and_color_threshold(tmp_path):
    source = tmp_path / "table.glb"
    table_mesh(source)
    preview = tmp_path / "previews"
    preview.mkdir()
    Image.new("RGB", (32, 32), (80, 60, 40)).save(preview / "view.png")
    crop = tmp_path / "crop.png"
    Image.new("RGB", (32, 32), (200, 180, 150)).save(crop)
    gate = support.geometry_gate(source)
    assert gate["up_axis"] == 1
    assert gate["top_area_m2"] > 1.9
    candidate = dict(
        asset_id="table",
        source_root=str(tmp_path),
        model_entrypoint=str(source),
        geometry=gate,
        selected_views=[
            dict(image=clip_select.official.fingerprint(preview / "view.png", preview))
        ],
    )
    package, provenance = support.materialize(
        candidate,
        dict(extents_xy_m=[1.2, 0.8]),
        crop,
        preview,
        tmp_path / "package",
    )
    assert provenance["appearance"]["ciede2000"] > 12
    assert provenance["appearance"]["method"] == "deterministic_lab_target"
    verified, entry, physical = standard_urdf.verify_package(
        tmp_path / "package/asset.json"
    )
    assert verified == package
    assert entry.name == "support.urdf"
    assert physical["fixed"] is True
    measured = standard_urdf.inspect(entry)
    assert np.isclose(measured["visual"][:, 2].max(), 0)


def test_legacy_source_root_maps_to_current_genesis_library():
    index_path = ROOT / "assets/genesis/clip_non_robot_v1/index.json"
    if not index_path.is_file():
        pytest.skip("local Genesis asset library is not installed")
    index, _ = clip_select.load_index(index_path)
    asset = next(a for a in index["assets"] if a["asset_id"] == "work_table_b7d04d4a")
    root, source = support._source_root(asset)
    assert root == ROOT / "assets/genesis/non_robot_v1"
    assert source == root / "sources/work_table.glb"


def finite_rows(contact=True):
    state = dict(
        position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0],
        velocity=[0, 0, 0], angular_velocity=[0, 0, 0],
    )
    support_state = copy.deepcopy(state)
    rows = []
    for step in range(1001):
        contacts = []
        if contact:
            contacts.append(
                dict(
                    a="support_0", b="mouse", geom_a=0, geom_b=1,
                    link_a=0, link_b=1, position=[0, 0, 0],
                    normal=[0, 0, 1], penetration=0,
                    force_a=None if step == 0 else [0, 0, -1],
                    force_b=None if step == 0 else [0, 0, 1],
                )
            )
        rows.append(
            dict(
                step=step, time_s=step * 0.004,
                contact_phase="initial_detection" if step == 0 else "solved_step",
                objects={"mouse": copy.deepcopy(state), "support_0": copy.deepcopy(support_state)},
                contacts=contacts,
            )
        )
    layout = dict(
        objects=[
            dict(
                object_id="mouse", category="mouse", fixed=False,
                translation_m=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0],
                source_velocity_mps=[0, 0, 0],
                source_angular_velocity_radps=[0, 0, 0],
            ),
            dict(
                object_id="support_0", category="desk", fixed=True,
                translation_m=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0],
                source_velocity_mps=[0, 0, 0],
                source_angular_velocity_radps=[0, 0, 0],
            ),
        ],
        relations=[
            dict(relation="on", source="mouse", target="support_0", evidence="observed")
        ],
    )
    return rows, layout, asset_physics.settings("baseline")


def test_fixed_finite_support_requires_real_declared_contact():
    passed = physics.evaluate(*finite_rows())
    assert passed["physics_status"] == "passed"
    assert passed["exit_code"] == 0
    assert passed["relation_results"][0]["passed"]
    failed = physics.evaluate(*finite_rows(contact=False))
    assert failed["physics_status"] == "failed"
    assert failed["exit_code"] == 2
    assert not failed["relation_results"][0]["passed"]

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from agenticsim.openxsim import genesis_runtime
from agenticsim.openxsim.backends import BackendCompileError, GenesisCompiler
from agenticsim.openxsim.conformance import evaluate_conformance
from agenticsim.openxsim.importers import import_compile_manifest, import_environment
from agenticsim.openxsim.ir import AssetRepresentation
from agenticsim.openxsim.pipeline import OpenXSimPipeline

FIXTURES = Path(__file__).parent / "fixtures" / "env_gen"
CAN_ON_PLATE = FIXTURES / "can_on_plate.resolved_scene.json"
CABINET = FIXTURES / "cabinet_articulated.resolved_scene.json"


def test_env_gen_transfer_compiles_genesis_render_bundle(tmp_path: Path) -> None:
    pipeline = OpenXSimPipeline(tmp_path / "artifacts")
    package, results, reports = pipeline.transfer(
        CAN_ON_PLATE,
        source_backend="env_gen",
        target_backends=("genesis",),
        strict=True,
    )

    assert package.target_backends == ("genesis",)
    result = results["genesis"]
    assert result.status == "compiled"
    assert result.blockers == ()
    assert result.metadata["artifact_format"] == "agenticsim.genesis_render_scene.v1"
    assert result.metadata["render_only"] is True
    assert result.runtime_command[0] == sys.executable

    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    assert scene["schema"] == "agenticsim.genesis_render_scene.v1"
    assert scene["package_digest"] == package.digest()
    assert scene["package_path"] == "environment_package.json"
    assert scene["render"] == {
        "compute_backend": "cpu",
        "fps": 12,
        "frames": 120,
        "height": 480,
        "renderer": "genesis_rasterizer",
        "width": 640,
    }
    assert {item["instance_id"] for item in scene["objects"]} == {"can_1", "plate_1"}
    assert all(item["kind"] == "mesh" for item in scene["objects"])
    assert all(item["render_fixed"] is True for item in scene["objects"])
    assert all(item["z_policy"] == "origin_on_table" for item in scene["objects"])

    recovered = import_compile_manifest(result.manifest_path)
    assert recovered == package
    checks = {check.level: check.status for check in reports["genesis"].checks}
    assert checks == {
        "L0": "pass",
        "L1": "pass",
        "L2": "not_evaluated",
        "L3": "not_evaluated",
        "L4": "not_evaluated",
    }


def test_genesis_compile_is_deterministic(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))

    first = GenesisCompiler().compile(package, tmp_path / "first", strict=True)
    second = GenesisCompiler().compile(package, tmp_path / "second", strict=True)

    assert Path(first.artifact_path).read_bytes() == Path(second.artifact_path).read_bytes()
    assert first.package_digest == second.package_digest == package.digest()


def test_genesis_conformance_cannot_be_promoted_by_runtime_or_policy_evidence(
    tmp_path: Path,
) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    contract_hash = hashlib.sha256(
        json.dumps(
            package.task.semantic_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    runtime_evidence = {
        "action_interface_bound": True,
        "success_evaluator_bound": True,
        "task_contract_hash": contract_hash,
        "reset_ok": True,
        "step_ok": True,
        "observation_keys": list(package.task.observation.get("state") or []),
        "trajectory": [
            {
                "objects": {
                    obj.instance_id: list(obj.pose.position) for obj in package.env.objects
                },
                "contacts": [],
            }
        ],
    }
    policy_evidence = {"episodes": 100, "success_rate": 1.0}

    report = evaluate_conformance(
        package,
        result,
        source_backend="env_gen",
        source_runtime=runtime_evidence,
        target_runtime=runtime_evidence,
        source_policy=policy_evidence,
        target_policy=policy_evidence,
    )

    assert report.highest_consecutive_level == "L1"
    assert {check.level: check.status for check in report.checks} == {
        "L0": "pass",
        "L1": "pass",
        "L2": "not_evaluated",
        "L3": "not_evaluated",
        "L4": "not_evaluated",
    }

    # Backend identity is not trusted in isolation: a Genesis artifact relabeled
    # as generic SAPIEN JSON must fail L0 and remain render-only at L2-L4.
    relabeled = replace(result, backend="sapien", metadata={})
    relabeled_report = evaluate_conformance(
        package,
        relabeled,
        source_backend="env_gen",
        source_runtime=runtime_evidence,
        target_runtime=runtime_evidence,
        source_policy=policy_evidence,
        target_policy=policy_evidence,
    )
    assert relabeled_report.highest_consecutive_level is None
    assert {check.level: check.status for check in relabeled_report.checks} == {
        "L0": "fail",
        "L1": "pass",
        "L2": "not_evaluated",
        "L3": "not_evaluated",
        "L4": "not_evaluated",
    }
    assert "cannot be relabeled" in relabeled_report.checks[0].details


def test_genesis_strict_compile_rejects_collision_only_asset(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    asset = package.assets[0]
    source = Path(asset.representations[0].uri)
    collision_only = replace(
        asset,
        representations=(
            AssetRepresentation(
                format=source.suffix.lstrip("."),
                uri=str(source),
                backend="genesis",
                role="collision",
            ),
        ),
    )
    package = replace(
        package,
        assets=(collision_only, *package.assets[1:]),
    )

    with pytest.raises(BackendCompileError, match="no existing Genesis visual representation"):
        GenesisCompiler().compile(package, tmp_path, strict=True)


def test_genesis_strict_compile_rejects_incomplete_dependency_closure(
    tmp_path: Path,
) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    asset = package.assets[0]
    representations = []
    for representation in asset.representations:
        if representation.backend != "genesis":
            representations.append(representation)
            continue
        metadata = dict(representation.metadata)
        metadata.pop("dependency_discovery")
        representations.append(replace(representation, metadata=metadata))
    package = replace(
        package,
        assets=(replace(asset, representations=tuple(representations)), *package.assets[1:]),
    )

    with pytest.raises(BackendCompileError, match="missing dependency_discovery"):
        GenesisCompiler().compile(package, tmp_path, strict=True)



def test_genesis_strict_compile_rejects_unsupported_format(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    asset = package.assets[0]
    ply = tmp_path / "asset.ply"
    ply.write_text("ply\n", encoding="utf-8")
    unsupported = replace(
        asset,
        representations=(
            AssetRepresentation(
                format="ply",
                uri=str(ply),
                backend="genesis",
                role="visual",
            ),
        ),
    )
    package = replace(package, assets=(unsupported, *package.assets[1:]))

    with pytest.raises(BackendCompileError, match="no existing Genesis visual representation"):
        GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)


def test_genesis_strict_compile_rejects_anisotropic_urdf_scale(tmp_path: Path) -> None:
    package = replace(import_environment(CABINET), target_backends=("genesis",))
    cabinet = package.env.objects[0]
    package = replace(
        package,
        env=replace(package.env, objects=(replace(cabinet, scale=(1.0, 1.1, 1.0)),)),
    )

    with pytest.raises(BackendCompileError, match="requires uniform scale"):
        GenesisCompiler().compile(package, tmp_path, strict=True)


def test_genesis_primitive_compiles_without_runtime_dependency(tmp_path: Path) -> None:
    from agenticsim.openxsim.text2env import compile_text

    package = compile_text(
        "Move the red block onto the blue zone.",
        repo_root=tmp_path,
        target_backends=("genesis",),
    )
    first = package.env.objects[0]
    package = replace(
        package,
        env=replace(
            package.env,
            objects=(
                replace(first, metadata={**first.metadata, "color": "red"}),
                *package.env.objects[1:],
            ),
        ),
    )
    result = GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))

    assert result.status == "compiled"
    assert {item["kind"] for item in scene["objects"]} == {"box"}
    assert all(len(item["size_m"]) == 3 for item in scene["objects"])
    assert scene["objects"][0]["color"] == "red"
    assert scene["objects"][0]["color_rgb"] == [0.82, 0.10, 0.12]



def test_genesis_urdf_compile_records_named_articulation(tmp_path: Path) -> None:
    package = replace(import_environment(CABINET), target_backends=("genesis",))
    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    cabinet = scene["objects"][0]

    assert cabinet["kind"] == "urdf"
    assert cabinet["uniform_scale"] == pytest.approx(0.27)
    assert cabinet["articulation"]["joint_names"] == ["joint_1", "joint_2", "joint_3"]
    assert cabinet["articulation"]["qpos"] == pytest.approx([0.0675, 0.0675, 0.0675])
    runner = Path(result.metadata["render_runner"])
    compile(runner.read_text(encoding="utf-8"), str(runner), "exec")


@pytest.mark.parametrize(("fmt", "create_file"), [("obj", False), ("usd", True)])
def test_genesis_strict_compile_rejects_missing_or_usd_representation(
    tmp_path: Path,
    fmt: str,
    create_file: bool,
) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    path = tmp_path / f"asset.{fmt}"
    if create_file:
        path.write_text("# unsupported fixture\n", encoding="utf-8")
    replacement = replace(
        package.assets[0],
        representations=(
            AssetRepresentation(
                format=fmt,
                uri=str(path),
                backend="genesis",
                role="visual",
            ),
        ),
    )
    package = replace(package, assets=(replacement, *package.assets[1:]))

    with pytest.raises(BackendCompileError, match="no existing Genesis visual representation"):
        GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)


def test_genesis_strict_compile_rejects_missing_urdf_joint(tmp_path: Path) -> None:
    package = replace(import_environment(CABINET), target_backends=("genesis",))
    cabinet = package.env.objects[0]
    articulation = {
        "joint_names": ["missing_joint"],
        "joint_limits": [[0.0, 0.2]],
        "qpos": [0.1],
        "state": "partially_open",
    }
    package = replace(
        package,
        env=replace(
            package.env,
            objects=(
                replace(cabinet, metadata={**cabinet.metadata, "articulation": articulation}),
            ),
        ),
    )

    with pytest.raises(BackendCompileError, match="missing declared joints"):
        GenesisCompiler().compile(package, tmp_path, strict=True)


def test_genesis_strict_compile_rejects_non_dof_urdf_joint(tmp_path: Path) -> None:
    package = replace(import_environment(CABINET), target_backends=("genesis",))
    source = FIXTURES / "fixture_assets" / "cabinet.urdf"
    bad_urdf = tmp_path / "fixed_joint.urdf"
    bad_urdf.write_text(
        source.read_text(encoding="utf-8").replace(
            'name="joint_1" type="prismatic"',
            'name="joint_1" type="fixed"',
            1,
        ),
        encoding="utf-8",
    )
    replacement = replace(
        package.assets[0],
        representations=tuple(
            replace(
                rep,
                uri=str(bad_urdf),
                sha256=hashlib.sha256(bad_urdf.read_bytes()).hexdigest(),
                size_bytes=bad_urdf.stat().st_size,
            )
            if rep.backend == "genesis"
            else rep
            for rep in package.assets[0].representations
        ),
    )
    package = replace(package, assets=(replacement, *package.assets[1:]))

    with pytest.raises(BackendCompileError, match="one-DoF articulation joints"):
        GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)



def test_genesis_strict_compile_rejects_unknown_explicit_color(tmp_path: Path) -> None:
    from agenticsim.openxsim.text2env import compile_text

    package = compile_text(
        "Move the red block onto the blue zone.",
        repo_root=tmp_path,
        target_backends=("genesis",),
    )
    first = package.env.objects[0]
    package = replace(
        package,
        env=replace(
            package.env,
            objects=(
                replace(first, metadata={**first.metadata, "color": "chartreuse"}),
                *package.env.objects[1:],
            ),
        ),
    )

    with pytest.raises(BackendCompileError, match="unsupported explicit color"):
        GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)


def test_genesis_y_up_mesh_permutes_anisotropic_scale(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    mesh = tmp_path / "asset.glb"
    mesh.write_bytes(b"glTF-test")
    representation = AssetRepresentation(
        format="glb",
        uri=str(mesh),
        backend="genesis",
        role="visual",
        sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        size_bytes=mesh.stat().st_size,
        metadata={
            "file_meshes_are_zup": False,
            "dependency_discovery": "test.local_dependencies.v1",
            "dependencies": [],
            "dependency_errors": [],
        },
    )
    package = replace(
        package,
        assets=(
            replace(package.assets[0], representations=(representation,)),
            *package.assets[1:],
        ),
        env=replace(
            package.env,
            objects=(
                replace(package.env.objects[0], scale=(1.0, 2.0, 3.0)),
                *package.env.objects[1:],
            ),
        ),
    )

    result = GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    first = scene["objects"][0]
    assert first["scale"] == [1.0, 2.0, 3.0]
    assert first["genesis_scale"] == [1.0, 3.0, 2.0]
    assert first["file_meshes_are_zup"] is False


def test_genesis_per_instance_articulation_state_is_preserved(tmp_path: Path) -> None:
    package = replace(import_environment(CABINET), target_backends=("genesis",))
    first = package.env.objects[0]
    first_articulation = dict(
        first.metadata.get("articulation") or package.assets[0].articulation
    )
    second_articulation = {**first_articulation, "qpos": [0.01, 0.02, 0.03]}
    second = replace(
        first,
        instance_id="cabinet_2",
        metadata={**first.metadata, "articulation": second_articulation},
    )
    package = replace(package, env=replace(package.env, objects=(first, second)))

    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    articulations = {
        item["instance_id"]: item["articulation"]["qpos"] for item in scene["objects"]
    }
    assert articulations["cabinet_1"] == pytest.approx([0.0675] * 3)
    assert articulations["cabinet_2"] == pytest.approx([0.01, 0.02, 0.03])


@pytest.mark.parametrize("tamper", ["pose", "delete_object"])
def test_genesis_scene_tamper_fails_static_conformance(
    tmp_path: Path, tamper: str
) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    artifact = Path(result.artifact_path)
    scene = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "pose":
        scene["objects"][0]["pose"]["position"][0] += 0.25
    else:
        scene["objects"].pop()
    artifact.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = evaluate_conformance(package, result, source_backend="env_gen")

    assert report.checks[0].level == "L0"
    assert report.checks[0].status == "fail"
    assert report.highest_consecutive_level is None


def test_genesis_scene_requires_package_path(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    scene.pop("package_path")

    with pytest.raises(genesis_runtime.GenesisRenderError, match="package_path"):
        genesis_runtime.validate_scene_config(scene)


def test_genesis_asset_mutation_breaks_package_binding(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    mesh = tmp_path / "asset.glb"
    mesh.write_bytes(b"glTF-before")
    representation = AssetRepresentation(
        format="glb",
        uri=str(mesh),
        backend="genesis",
        role="visual",
        sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        size_bytes=mesh.stat().st_size,
        metadata={
            "file_meshes_are_zup": False,
            "dependency_discovery": "test.local_dependencies.v1",
            "dependencies": [],
            "dependency_errors": [],
        },
    )
    package = replace(
        package,
        assets=(
            replace(package.assets[0], representations=(representation,)),
            *package.assets[1:],
        ),
    )
    result = GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)
    artifact = Path(result.artifact_path)
    scene = json.loads(artifact.read_text(encoding="utf-8"))
    mesh.write_bytes(b"glTF-after!")

    with pytest.raises(genesis_runtime.GenesisRenderError, match="could not reproduce"):
        genesis_runtime.verify_package_binding(scene, artifact)


def test_genesis_dependency_mutation_breaks_package_binding(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    dependency = tmp_path / "albedo.png"
    dependency.write_bytes(b"texture-before")
    asset = package.assets[0]
    representations = tuple(
        replace(
            representation,
            metadata={
                **representation.metadata,
                "dependencies": [
                    {
                        "uri": str(dependency),
                        "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
                        "size_bytes": dependency.stat().st_size,
                    }
                ],
            },
        )
        if representation.backend == "genesis"
        else representation
        for representation in asset.representations
    )
    package = replace(
        package,
        assets=(replace(asset, representations=representations), *package.assets[1:]),
    )
    result = GenesisCompiler().compile(package, tmp_path / "compiled", strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    dependency.write_bytes(b"texture-after!")

    with pytest.raises(genesis_runtime.GenesisRenderError, match="could not reproduce"):
        genesis_runtime.verify_package_binding(scene, result.artifact_path)


def test_encoded_video_verification_rejects_dropped_frames(tmp_path: Path) -> None:
    import numpy as np

    video = tmp_path / "short.mp4"
    with genesis_runtime._VideoWriter(video, 16, 16, 12) as writer:
        writer.append(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.append(np.full((16, 16, 3), 255, dtype=np.uint8))

    verified = genesis_runtime.verify_encoded_video(
        video,
        expected_frames=2,
        width=16,
        height=16,
    )
    assert verified["decoded_frame_count"] == 2
    assert verified["decoded_unique_frame_count"] == 2
    with pytest.raises(genesis_runtime.GenesisRenderError, match="expected exactly 3"):
        genesis_runtime.verify_encoded_video(
            video,
            expected_frames=3,
            width=16,
            height=16,
        )


def test_articulation_qpos_mismatch_is_a_hard_failure() -> None:
    class Joint:
        n_dofs = 1
        n_qs = 1
        qs_idx_local = [0]

    class Entity:
        def get_joint(self, *, name: str) -> Joint:
            assert name == "drawer"
            return Joint()

        def set_qpos(self, values: list[float], *, qs_idx_local: list[int]) -> None:
            assert values == [0.1]
            assert qs_idx_local == [0]

        def get_qpos(self, *, qs_idx_local: list[int]) -> list[float]:
            assert qs_idx_local == [0]
            return [0.101]

    spec = {
        "instance_id": "cabinet",
        "articulation": {"joint_names": ["drawer"], "qpos": [0.1]},
    }
    with pytest.raises(genesis_runtime.GenesisRenderError, match="applied articulation qpos"):
        genesis_runtime._apply_articulation(Entity(), spec)

def test_genesis_runtime_helpers_are_dependency_free_and_deterministic(tmp_path: Path) -> None:
    package = replace(import_environment(CAN_ON_PLATE), target_backends=("genesis",))
    result = GenesisCompiler().compile(package, tmp_path, strict=True)
    scene = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))

    assert genesis_runtime.validate_scene_config(scene) is scene
    plan = genesis_runtime.build_camera_plan(scene, 120)
    assert len(plan["orbit"]) == 120
    assert plan["duplicates_endpoint"] is False
    assert plan["target"][2] == pytest.approx(0.79425)
    assert plan["orbit"][0]["position"] != plan["orbit"][-1]["position"]
    assert genesis_runtime.unique_frame_count([b"one", b"two", b"one"]) == 2
    assert genesis_runtime.pose_error(
        {"position": [1, 2, 3], "orientation_wxyz": [1, 0, 0, 0]},
        {"position": [1, 2, 3], "orientation_wxyz": [-1, 0, 0, 0]},
    ) == {
        "position_max_abs_m": 0.0,
        "quaternion_sign_invariant_max_abs": 0.0,
    }


def test_genesis_runtime_failure_writes_evidence_without_success_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene = tmp_path / "invalid_scene.json"
    scene.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "render"
    output.mkdir()
    (output / genesis_runtime.MANIFEST_OUTPUT).write_text("stale\n", encoding="utf-8")
    (output / genesis_runtime.VIDEO_OUTPUT).write_bytes(b"stale-video")
    (output / genesis_runtime.STATIC_OUTPUTS["front_high"]).write_bytes(b"stale-png")

    status = genesis_runtime.main(
        ["--scene", str(scene), "--output-dir", str(output), "--frames", "3"]
    )

    assert status == 1
    failed = json.loads(
        (output / genesis_runtime.EVIDENCE_OUTPUT).read_text(encoding="utf-8")
    )
    assert failed["schema"] == genesis_runtime.EVIDENCE_SCHEMA
    assert failed["status"] == "failed"
    assert failed["physical_runtime_evidence"] is False
    assert failed["conformance"]["L2"] == "not_evaluated"
    assert not (output / genesis_runtime.MANIFEST_OUTPUT).exists()
    assert not (output / genesis_runtime.VIDEO_OUTPUT).exists()
    assert not (output / genesis_runtime.STATIC_OUTPUTS["front_high"]).exists()
    assert "Genesis render failed" in capsys.readouterr().err


_RUN_GENESIS = os.environ.get("OPENXSIM_RUN_GENESIS") == "1"


@pytest.mark.skipif(not _RUN_GENESIS, reason="set OPENXSIM_RUN_GENESIS=1 for real rendering")
def test_real_genesis_can_and_cabinet_render_acceptance(tmp_path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    pipeline = OpenXSimPipeline(tmp_path / "artifacts")
    _, can_results, _ = pipeline.transfer(
        CAN_ON_PLATE,
        source_backend="env_gen",
        target_backends=("genesis",),
        strict=True,
    )
    can_result = can_results["genesis"]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        can_result.runtime_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    render_dir = Path(can_result.runtime_command[-1])
    assert {path.name for path in render_dir.iterdir()} == {
        *genesis_runtime.STATIC_OUTPUTS.values(),
        genesis_runtime.SEGMENTATION_OUTPUT,
        *genesis_runtime.OBSERVER_OUTPUTS.values(),
        genesis_runtime.VIDEO_OUTPUT,
        genesis_runtime.EVIDENCE_OUTPUT,
        genesis_runtime.MANIFEST_OUTPUT,
    }
    for name in genesis_runtime.STATIC_OUTPUTS.values():
        with Image.open(render_dir / name) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert image.size == (640, 480)
            assert int(pixels.max()) > int(pixels.min())

    evidence = json.loads(
        (render_dir / genesis_runtime.EVIDENCE_OUTPUT).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "success"
    assert evidence["package_binding"]["verified"] is True
    assert evidence["genesis"]["commit"] == "0e74bf392781884ccad765c3f344419c86b872ca"
    assert evidence["render"]["total_frame_count"] == 120
    assert evidence["render"]["unique_frame_count"] >= 30
    assert all(max(item["visibility_pixels"].values()) > 0 for item in evidence["objects"])
    assert all(
        item["pose_error"]["position_max_abs_m"] <= 1e-6
        and item["pose_error"]["quaternion_sign_invariant_max_abs"] <= 1e-6
        for item in evidence["objects"]
    )

    reader = imageio.get_reader(render_dir / genesis_runtime.VIDEO_OUTPUT)
    try:
        assert sum(1 for _ in reader) == 120
    finally:
        reader.close()

    manifest = json.loads(
        (render_dir / genesis_runtime.MANIFEST_OUTPUT).read_text(encoding="utf-8")
    )
    for name, record in manifest["artifacts"].items():
        path = render_dir / name
        assert path.stat().st_size == record["size_bytes"]
        assert genesis_runtime.sha256_file(path) == record["sha256"]

    _, cabinet_results, _ = pipeline.transfer(
        CABINET,
        source_backend="env_gen",
        target_backends=("genesis",),
        strict=True,
    )
    cabinet_result = cabinet_results["genesis"]
    cabinet_command = (
        *cabinet_result.runtime_command,
        "--width",
        "320",
        "--height",
        "240",
        "--frames",
        "3",
    )
    completed = subprocess.run(
        cabinet_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    cabinet_render_dir = Path(cabinet_result.runtime_command[-1])
    cabinet_evidence = json.loads(
        (cabinet_render_dir / genesis_runtime.EVIDENCE_OUTPUT).read_text(encoding="utf-8")
    )
    articulation = cabinet_evidence["objects"][0]["articulation"]
    assert articulation["joint_names"] == ["joint_1", "joint_2", "joint_3"]
    assert articulation["q_indices_local"] == [0, 1, 2]
    assert articulation["applied_qpos"] == pytest.approx([0.0675] * 3, abs=1e-6)

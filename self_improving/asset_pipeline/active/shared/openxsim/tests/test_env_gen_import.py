import hashlib
import json
import struct
from pathlib import Path

import pytest
from agenticsim.openxsim.env_gen import import_env_gen

FIX = Path(__file__).parent / "fixtures" / "env_gen" / "can_on_plate.resolved_scene.json"


def _single_object_scene(
    tmp_path: Path,
    source_files: list[Path],
    *,
    load_type: str = "rigid",
) -> Path:
    data = json.loads(FIX.read_text(encoding="utf-8"))
    data["objects"] = [data["objects"][0]]
    data["objects"][0]["load_type"] = load_type
    data["objects"][0]["source_files"] = [str(path) for path in source_files]
    data["relations"] = []
    scene = tmp_path / "scene.resolved_scene.json"
    scene.write_text(json.dumps(data), encoding="utf-8")
    return scene


def _write_glb(path: Path, document: dict[str, object]) -> None:
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    payload_size = 12 + 8 + len(json_chunk)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, payload_size)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
    )


def _write_parseable_asset(path: Path) -> None:
    if path.suffix == ".dae":
        path.write_text("<COLLADA/>", encoding="utf-8")
    elif path.suffix == ".gltf":
        path.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
    elif path.suffix == ".glb":
        _write_glb(path, {"asset": {"version": "2.0"}})
    else:
        path.write_text("visual", encoding="utf-8")


def test_rigid_scene_maps_to_valid_ir():
    pkg = import_env_gen(FIX)
    pkg.validate()
    assert pkg.package_id.startswith("place_a_can")
    assert len(pkg.env.objects) == 2
    ids = {o.instance_id for o in pkg.env.objects}
    assert ids == {"can_1", "plate_1"}
    can = next(o for o in pkg.env.objects if o.instance_id == "can_1")
    assert can.pose.position[0] == pytest.approx(-0.125948517)
    assert can.static is False
    assert can.metadata["z_policy"] == "origin_on_table"
    assert can.asset_id.startswith("asset_071_can")
    asset = next(a for a in pkg.assets if a.asset_id == can.asset_id)
    rep = asset.representations[0]
    # Keep the original SAPIEN representation first for compatibility.
    assert rep.backend == "sapien" and rep.format in {
        "glb",
        "obj",
        "dae",
        "stl",
        "urdf",
    }
    genesis = asset.representations[1]
    assert genesis.backend == "genesis"
    assert genesis.format == "obj"
    assert genesis.role == "visual_and_collision"
    assert genesis.uri == rep.uri
    assert genesis.metadata == {
        "dependencies": [],
        "dependency_discovery": "env_gen.local_asset_dependencies.v1",
        "dependency_errors": [],
        "file_meshes_are_zup": True,
    }
    for representation in asset.representations:
        payload = Path(representation.uri).read_bytes()
        assert representation.sha256 == hashlib.sha256(payload).hexdigest()
        assert representation.size_bytes == len(payload)
    assert pkg.task.instruction
    assert any(c.get("type") == "unbound" for c in pkg.task.success)


def test_fidelity_unknown_physics_and_provenance_and_relations():
    pkg = import_env_gen(FIX)
    asset = pkg.assets[0]
    # 未知物理显式标记，不编造
    assert asset.physical["mass_kg"] == {"status": "unknown"}
    assert asset.physical["inertia"] == {"status": "unknown"}
    assert asset.physical["dimensions_m"] is not None
    # 血缘入 IR
    assert asset.source["kind"] == "env_gen"
    assert asset.source["asset_provenance"] == "robotwin_catalog"
    # relations 作为数据带入 env.metadata（本任务不合成 task）
    rels = pkg.env.metadata["relations"]
    assert {r["relation"] for r in rels} == {"on_table", "on_top_of"}
    # env-gen 溯源哈希带入
    assert pkg.env.metadata["source_scene_spec_sha256"]
    assert pkg.env.metadata["compiler_version"].startswith("scene_gen")


def test_rigid_asset_has_empty_articulation():
    pkg = import_env_gen(FIX)
    assert all(a.articulation == {} for a in pkg.assets)  # can/plate 无关节


def test_missing_asset_file_raises():
    from agenticsim.openxsim.importers import EnvironmentImportError

    bad = FIX.with_name("missing_asset.resolved_scene.json")
    with pytest.raises(EnvironmentImportError):
        import_env_gen(bad)


def test_non_env_gen_input_raises(tmp_path):
    from agenticsim.openxsim.importers import EnvironmentImportError

    p = tmp_path / "other.json"
    p.write_text('{"compiler_version": "something_else", "objects": []}')
    with pytest.raises(EnvironmentImportError):
        import_env_gen(p)


def test_malformed_json_raises(tmp_path):
    from agenticsim.openxsim.importers import EnvironmentImportError

    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(EnvironmentImportError):
        import_env_gen(p)


def test_dispatcher_routes_env_gen():
    from agenticsim.openxsim.importers import import_environment

    pkg = import_environment(FIX)  # 自动识别
    assert pkg.source["backend"] == "env_gen"
    pkg2 = import_environment(FIX, source_backend="env_gen")  # 显式
    assert pkg2.package_id == pkg.package_id


def test_determinism_same_input_same_digest():
    assert import_env_gen(FIX).digest() == import_env_gen(FIX).digest()


def test_articulated_scene_carries_joints_and_state():
    fix = FIX.with_name("cabinet_articulated.resolved_scene.json")
    pkg = import_env_gen(fix)
    pkg.validate()
    art = [a.articulation for a in pkg.assets if a.articulation]
    assert art, "expected at least one asset with non-empty articulation"
    a0 = art[0]
    assert a0["joint_names"]  # 关节名带过来了
    assert "state" in a0  # articulation_state 字段存在（Fix 1）
    asset = next(asset for asset in pkg.assets if asset.articulation)
    genesis = next(rep for rep in asset.representations if rep.backend == "genesis")
    assert genesis.format == "urdf"
    assert genesis.role == "visual_and_collision"
    assert "file_meshes_are_zup" not in genesis.metadata
    obj = next(obj for obj in pkg.env.objects if obj.asset_id == asset.asset_id)
    assert obj.metadata["articulation"] == asset.articulation


@pytest.mark.parametrize(
    ("suffix", "file_meshes_are_zup"),
    [
        ("obj", True),
        ("stl", True),
        ("dae", True),
        ("glb", False),
        ("gltf", False),
    ],
)
def test_genesis_prefers_visual_mesh_and_records_up_axis(
    tmp_path: Path,
    suffix: str,
    file_meshes_are_zup: bool,
):
    collision = tmp_path / "collision" / "baseline.obj"
    collision.parent.mkdir()
    collision.write_text("collision", encoding="utf-8")
    visual = tmp_path / "visual" / f"asset.{suffix}"
    visual.parent.mkdir()
    _write_parseable_asset(visual)
    scene = _single_object_scene(tmp_path, [collision, visual])

    pkg = import_env_gen(scene)
    asset = pkg.assets[0]
    sapien, genesis = asset.representations

    assert sapien.backend == "sapien"
    assert sapien.uri == str(collision)
    assert genesis.backend == "genesis"
    assert genesis.uri == str(visual)
    assert genesis.format == suffix
    assert genesis.role == "visual"
    assert genesis.metadata == {
        "dependencies": [],
        "dependency_discovery": "env_gen.local_asset_dependencies.v1",
        "dependency_errors": [],
        "file_meshes_are_zup": file_meshes_are_zup,
    }


@pytest.mark.parametrize("case", ["collision_only", "missing_visual", "unsupported"])
def test_genesis_incompatible_sources_remain_compiler_blockers(
    tmp_path: Path,
    case: str,
):
    if case == "unsupported":
        sapien_source = tmp_path / "asset.ply"
        sapien_source.write_text("ply", encoding="utf-8")
        source_files = [sapien_source]
        expected_format = "ply"
    else:
        sapien_source = tmp_path / "collision" / "asset.obj"
        sapien_source.parent.mkdir()
        sapien_source.write_text("collision", encoding="utf-8")
        source_files = [sapien_source]
        expected_format = "obj"
        if case == "missing_visual":
            source_files.append(tmp_path / "visual" / "missing.glb")

    scene = _single_object_scene(tmp_path, source_files)
    pkg = import_env_gen(scene)
    asset = pkg.assets[0]

    assert len(asset.representations) == 1
    assert asset.representations[0].backend == "sapien"
    assert asset.representations[0].format == expected_format


def test_obj_dependency_closure_is_hash_bound_and_deterministic(tmp_path: Path):
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    texture = texture_dir / "albedo.png"
    texture.write_bytes(b"first-texture")
    material = tmp_path / "material.mtl"
    material.write_text("newmtl material\nmap_Kd textures/albedo.png\n", encoding="utf-8")
    mesh = tmp_path / "asset.obj"
    mesh.write_text(
        "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    scene = _single_object_scene(tmp_path, [mesh])

    first = import_env_gen(scene)
    first_rep = next(rep for rep in first.assets[0].representations if rep.backend == "genesis")
    dependencies = first_rep.metadata["dependencies"]
    assert [item["uri"] for item in dependencies] == sorted([str(material), str(texture)])
    for item in dependencies:
        payload = Path(item["uri"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["size_bytes"] == len(payload)
    first_digest = first.digest()
    first_primary_sha = first_rep.sha256
    first_dependency_sha = {item["uri"]: item["sha256"] for item in dependencies}

    texture.write_bytes(b"second-texture")
    second = import_env_gen(scene)
    second_rep = next(rep for rep in second.assets[0].representations if rep.backend == "genesis")
    second_dependency_sha = {
        item["uri"]: item["sha256"] for item in second_rep.metadata["dependencies"]
    }

    assert second.digest() != first_digest
    assert second_rep.sha256 == first_primary_sha
    assert second_dependency_sha[str(material)] == first_dependency_sha[str(material)]
    assert second_dependency_sha[str(texture)] != first_dependency_sha[str(texture)]
    assert second_rep.metadata["dependencies"] == sorted(
        second_rep.metadata["dependencies"], key=lambda item: item["uri"]
    )


@pytest.mark.parametrize("suffix", ["dae", "gltf", "glb"])
def test_unparseable_dependency_document_is_recorded(tmp_path: Path, suffix: str):
    mesh = tmp_path / f"asset.{suffix}"
    mesh.write_bytes(b"not-a-valid-document")
    scene = _single_object_scene(tmp_path, [mesh])

    pkg = import_env_gen(scene)
    genesis = next(rep for rep in pkg.assets[0].representations if rep.backend == "genesis")

    assert genesis.metadata["dependencies"] == []
    errors = genesis.metadata["dependency_errors"]
    assert len(errors) == 1
    assert errors[0].startswith(f"dependency parse failed for {mesh}:")


def test_missing_obj_sidecar_is_recorded_for_compiler_blocking(tmp_path: Path):
    mesh = tmp_path / "asset.obj"
    mesh.write_text(
        "mtllib missing.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    scene = _single_object_scene(tmp_path, [mesh])

    pkg = import_env_gen(scene)
    genesis = next(rep for rep in pkg.assets[0].representations if rep.backend == "genesis")

    assert genesis.metadata["dependencies"] == []
    assert genesis.metadata["dependency_errors"] == [
        f"missing dependency {tmp_path / 'missing.mtl'} referenced by {mesh}"
    ]


@pytest.mark.parametrize("suffix", ["gltf", "glb"])
def test_gltf_family_binds_external_buffers_and_images(
    tmp_path: Path,
    suffix: str,
):
    buffer = tmp_path / "mesh.bin"
    buffer.write_bytes(b"buffer")
    image = tmp_path / "albedo.png"
    image.write_bytes(b"image")
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin", "byteLength": len(buffer.read_bytes())}],
        "images": [{"uri": "albedo.png"}],
    }
    mesh = tmp_path / f"asset.{suffix}"
    if suffix == "glb":
        _write_glb(mesh, document)
    else:
        mesh.write_text(json.dumps(document), encoding="utf-8")
    scene = _single_object_scene(tmp_path, [mesh])

    pkg = import_env_gen(scene)
    genesis = next(rep for rep in pkg.assets[0].representations if rep.backend == "genesis")

    if suffix == "gltf":
        assert len(pkg.assets[0].representations) == 1
    assert genesis.metadata["dependency_errors"] == []
    assert [item["uri"] for item in genesis.metadata["dependencies"]] == sorted(
        [str(image), str(buffer)]
    )


def test_dae_binds_image_init_from_without_material_id_false_positive(
    tmp_path: Path,
):
    texture = tmp_path / "surface.png"
    texture.write_bytes(b"image")
    mesh = tmp_path / "asset.dae"
    mesh.write_text(
        """<COLLADA>
  <library_images><image><init_from>surface.png</init_from></image></library_images>
  <library_effects><effect><profile_COMMON><newparam><surface>
    <init_from>image-id-not-a-file</init_from>
  </surface></newparam></profile_COMMON></effect></library_effects>
</COLLADA>""",
        encoding="utf-8",
    )
    scene = _single_object_scene(tmp_path, [mesh])

    pkg = import_env_gen(scene)
    genesis = next(rep for rep in pkg.assets[0].representations if rep.backend == "genesis")

    assert genesis.metadata["dependency_errors"] == []
    assert [item["uri"] for item in genesis.metadata["dependencies"]] == [str(texture)]


def test_urdf_binds_mesh_texture_and_recursive_mesh_sidecars(tmp_path: Path):
    mesh_dir = tmp_path / "meshes"
    texture_dir = tmp_path / "textures"
    mesh_dir.mkdir()
    texture_dir.mkdir()
    mesh_texture = texture_dir / "mesh.png"
    mesh_texture.write_bytes(b"mesh-image")
    urdf_texture = texture_dir / "robot.png"
    urdf_texture.write_bytes(b"robot-image")
    material = mesh_dir / "body.mtl"
    material.write_text("newmtl body\nmap_Kd ../textures/mesh.png\n", encoding="utf-8")
    mesh = mesh_dir / "body.obj"
    mesh.write_text("mtllib body.mtl\nv 0 0 0\n", encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="fixture"><link name="base"><visual>
  <geometry><mesh filename="meshes/body.obj"/></geometry>
  <material name="m"><texture filename="textures/robot.png"/></material>
</visual></link></robot>""",
        encoding="utf-8",
    )
    scene = _single_object_scene(tmp_path, [urdf], load_type="urdf")

    pkg = import_env_gen(scene)
    genesis = next(rep for rep in pkg.assets[0].representations if rep.backend == "genesis")

    assert genesis.metadata["dependency_errors"] == []
    assert [item["uri"] for item in genesis.metadata["dependencies"]] == sorted(
        [str(material), str(mesh), str(mesh_texture), str(urdf_texture)]
    )


def test_articulation_is_preserved_per_instance_for_shared_asset(tmp_path: Path):
    fixture = FIX.with_name("cabinet_articulated.resolved_scene.json")
    data = json.loads(fixture.read_text(encoding="utf-8"))
    first = data["objects"][0]
    first["source_files"] = [str(fixture.parent / "fixture_assets" / "cabinet.urdf")]
    second = json.loads(json.dumps(first))
    second["object_id"] = "cabinet_2"
    second["articulation_qpos"] = [0.01, 0.02, 0.03]
    second["articulation_state"] = "instance-specific"
    data["objects"] = [first, second]
    data["relations"] = []
    scene = tmp_path / "shared_asset.resolved_scene.json"
    scene.write_text(json.dumps(data), encoding="utf-8")

    pkg = import_env_gen(scene)
    by_id = {obj.instance_id: obj for obj in pkg.env.objects}

    assert len(pkg.assets) == 1
    assert by_id["cabinet_1"].metadata["articulation"]["qpos"] == [
        0.0675,
        0.0675,
        0.0675,
    ]
    assert by_id["cabinet_2"].metadata["articulation"] == {
        "joint_names": ["joint_1", "joint_2", "joint_3"],
        "joint_limits": first["articulation_joint_limits"],
        "qpos": [0.01, 0.02, 0.03],
        "state": "instance-specific",
    }

"""Portable asset conversion, physical evidence and multi-library attack cases."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import import_simfoundry_assets as importer
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis import validate_single_asset as physics


def source(tmp_path):
    scene = tmp_path / "source"
    root = scene / "s11_sim/objects/box/model/urdf"
    root.mkdir(parents=True)
    (root / "mesh.obj").write_text(trimesh.creation.box((0.1, 0.2, 0.3)).export(file_type="obj"))
    (root / "model.urdf").write_text("""<robot name="box"><link name="body">
    <inertial><origin xyz="0.01 0.02 0.03" rpy="0.2 0.4 0.6"/>
    <mass value="0.2"/><inertia ixx="0.002" iyy="0.003" izz="0.004"
      ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.01 0 0" rpy="0 0.2 0"/><geometry>
    <mesh filename="mesh.obj" scale="1 2 1"/></geometry></visual>
    <collision><geometry><mesh filename="mesh.obj"/></geometry></collision>
    </link></robot>""")
    (scene / "s11_sim/scene_objects_info.json").write_text(
        json.dumps({"0": dict(name="iter_0", category="box", model="model", friction=0.4)})
    )
    return scene, root / "model.urdf"


def test_equivalent_inertia_portability_and_tamper(tmp_path):
    scene, original = source(tmp_path)
    authored = standard.inspect(original)
    inventory = importer.import_assets(scene, tmp_path / "library")
    item = inventory["assets"][0]
    assert item["status"] == "imported"
    package = tmp_path / "library" / item["package"]
    copied = tmp_path / "independent"
    shutil.copytree(package.parent, copied)
    shutil.rmtree(scene)
    data, entry, _ = standard.verify_package(copied / "asset.json")
    actual = standard.inspect(entry)
    for key in authored:
        assert np.allclose(authored[key], actual[key], atol=1e-14)
    assert 'rpy="0 0 0"' in entry.read_text()
    (copied / "physics.json").write_text('{"friction":0}')
    with pytest.raises(ValueError):
        standard.verify_package(copied / "asset.json")


@pytest.mark.parametrize("replacement", ['<joint name="bad"/>', '<link name="second"/>'])
def test_multibody_rejected(tmp_path, replacement):
    scene, entry = source(tmp_path)
    entry.write_text(entry.read_text().replace("</robot>", replacement + "</robot>"))
    result = importer.import_assets(scene, tmp_path / "library")
    assert result["assets"][0]["status"] == "failed"
    assert "single rigid" in result["assets"][0]["error"]


@pytest.mark.parametrize("change", ["mass", "collision", "texture", "scale"])
def test_bad_assets_retained_as_failures(tmp_path, change):
    scene, entry = source(tmp_path)
    text = entry.read_text()
    if change == "mass":
        text = text.replace('value="0.2"', 'value="nan"')
    elif change == "collision":
        text = text.replace("<collision>", "<not_collision>").replace(
            "</collision>", "</not_collision>"
        )
    elif change == "texture":
        with (entry.parent / "mesh.obj").open("a") as stream:
            stream.write("\nmtllib missing.mtl\n")
    else:
        text = text.replace('scale="1 2 1"', 'scale="1 -2 1"')
    entry.write_text(text)
    result = importer.import_assets(scene, tmp_path / "library")
    assert result["assets"][0]["status"] == "failed"


def trajectory():
    return [
        dict(
            step=i,
            time_s=i * 0.004,
            position=[0, 0, 0.1],
            orientation_wxyz=[1, 0, 0, 0],
            velocity=[0, 0, 0],
            angular_velocity=[0, 0, 0],
            contacts=[],
            ground_up_force_n=None if i == 0 else 1.0,
        )
        for i in range(1001)
    ]


def test_no_contact_cannot_pass_and_penetration_not_hidden():
    rows = trajectory()
    cfg = physics.asset_physics.settings("baseline")
    # Stationary screenshots cannot establish support: force must actually be observed.
    for row in rows[1:]:
        row["ground_up_force_n"] = 0
    assert not next(c for c in physics.evaluate(rows, cfg) if c["name"] == "support_fraction")[
        "passed"
    ]
    rows = trajectory()
    rows[10]["contacts"] = [{"penetration": 0.00101}]
    assert not next(c for c in physics.evaluate(rows, cfg) if c["name"] == "penetration_m")[
        "passed"
    ]
    with pytest.raises(ValueError, match="incomplete"):
        physics.evaluate(rows[:-1], cfg)
    rows[8]["step"] = 9
    with pytest.raises(ValueError, match="nonsequential"):
        physics.evaluate(rows, cfg)


def test_changed_scale_or_fixed_body_rejected():
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    geom = SimpleNamespace(get_vverts=lambda: points * 2)
    entity = SimpleNamespace(
        get_quat=lambda: np.array([1.0, 0, 0, 0]),
        get_pos=lambda: np.zeros(3),
        links=[SimpleNamespace(vgeoms=[geom])],
    )
    with pytest.raises(ValueError, match="geometry differs"):
        standard.audit(entity, {"visual": points}, collision=False)
    geom.get_vverts = lambda: points
    entity.get_verts = lambda: points
    entity.geoms = [geom]
    entity.n_dofs = 6
    entity.base_link = SimpleNamespace(is_fixed=True)
    with pytest.raises(ValueError, match="fixed"):
        standard.audit(entity, dict(visual=points, collision=points, collision_count=1))


def test_inertia_coordinate_mismatch_is_not_accepted():
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    geom = SimpleNamespace(get_vverts=lambda: points, get_friction=lambda: np.array(0.4))
    link = SimpleNamespace(
        vgeoms=[geom],
        is_fixed=False,
        desc=SimpleNamespace(inertial_quat=[1, 0, 0, 0], inertial_pos=[0, 0, 0]),
    )
    entity = SimpleNamespace(
        get_quat=lambda: np.array([1.0, 0, 0, 0]),
        get_pos=lambda: np.zeros(3),
        get_verts=lambda: points,
        links=[link],
        base_link=link,
        get_mass=lambda: np.array(0.2),
        get_links_inertia=lambda: np.diag([0.002, 0.003, 0.004]),
        geoms=[geom],
        n_dofs=6,
    )
    expected = dict(
        visual=points,
        collision=points,
        collision_count=1,
        mass=0.2,
        com=np.zeros(3),
        inertia=np.diag([0.003, 0.002, 0.004]),
    )
    with pytest.raises(ValueError, match="inertia changed"):
        standard.audit(entity, expected, friction=0.4)


def test_union_gate_rejects_forged_or_other_asset_evidence(tmp_path):
    with pytest.raises(ValueError, match="no passing"):
        physics.verify_evidence(tmp_path, dict(physics_status="passed"))
    report = tmp_path / "physics_result.json"
    report.write_text(
        json.dumps(
            dict(exit_code=0, physics_status="passed", steps_executed=0, simulation_executed=False)
        )
    )
    binding = dict(
        physics_status="passed", physics_evidence=physics.official.fingerprint(report, tmp_path)
    )
    with pytest.raises(ValueError, match="did not complete"):
        physics.verify_evidence(tmp_path, binding)

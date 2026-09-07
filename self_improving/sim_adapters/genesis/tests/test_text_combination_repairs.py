"""Regression evidence from real existing-asset language combination failures."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import asset_extraction as extraction
from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis.tests.test_text_repair import case


def test_table_center_wording_expresses_support_but_bowl_interior_does_not():
    objects = [dict(object_id="cup"), dict(object_id="table")]
    for text in ["桌子中间放着一个黄色杯子。", "桌面中央放杯子。", "桌子的中心放杯子。"]:
        relation = dict(relation="on", source="cup", target="table", evidence=text)
        assert extraction.clean_relations(dict(relations=[relation], ambiguities=[]), text, objects)
    text = "杯子在碗中间。"
    with pytest.raises(ValueError, match="not expressed"):
        extraction.clean_relations(
            dict(
                relations=[dict(relation="on", source="cup", target="table", evidence=text)],
                ambiguities=[],
            ),
            text,
            objects,
        )


def test_explicit_center_is_sampled_without_losing_geometry_gates():
    data, _ = case()
    assets = data["assets"]
    poses = {"table": data["poses"]["table"]}
    polygon = np.asarray(assets["table"]["surface"]["polygon_xy_m"])
    expected = (polygon.min(0) + polygon.max(0)) / 2
    for seed in [0, 42, 87]:
        selected, _ = geo.sample_object(
            assets, poses, "a", [], [dict(object_id="a", region="center")], seed, 0
        )
        assert selected is not None
        actual = geo.inverse([selected["pose"]["position"]], poses["table"])[0][:2]
        assert np.linalg.norm(actual - expected) < 1e-9
        assert not geo.geometric_checks(assets, dict(poses, a=selected["pose"]), [])


def test_export_preserves_tiny_offdiagonal_tensor_through_mujoco(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    tensor = np.array(
        [
            [0.0018396996892988682, 3.08183864253709e-11, 1.46538702438137e-12],
            [3.08183864253709e-11, 0.0018396996892989638, 3.61504920609053e-10],
            [1.46538702438137e-12, 3.61504920609053e-10, 0.003205756656825447],
        ]
    )
    path = prep.write_collision_model(
        [trimesh.creation.box([0.1, 0.1, 0.1])], tmp_path, 0.5, np.zeros(3), tensor
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    axes = geo.rotation(model.body_iquat[1])
    actual = axes @ np.diag(model.body_inertia[1]) @ axes.T
    assert np.allclose(actual, tensor, rtol=1e-10, atol=1e-14)
    assert abs(actual[1, 2]) > 3e-10
    assert ET.parse(path).find(".//inertial").get("fullinertia") is None


def test_ordinal_reference_does_not_add_another_quantity():
    text = "桌上放着两个黄色杯子，第一个杯子在第二个杯子左边。"
    objects = [
        dict(
            object_id="table_1",
            category="table",
            description="桌子",
            attributes=[],
            mentions=["桌"],
        )
    ]
    for number, ordinal in [(1, "第一"), (2, "第二")]:
        objects.append(
            dict(
                object_id=f"cup_{number}",
                category="cup",
                description="黄色杯子",
                attributes=["黄色"],
                mentions=["两个黄色杯子", f"{ordinal}个杯子"],
            )
        )
    assert extraction.clean_objects(dict(objects=objects, ambiguities=[]), text) == objects


def test_one_model_correction_is_recorded_and_revalidated(tmp_path):
    import json
    from self_improving.sim_adapters.genesis.tests.test_asset_extraction import config

    text = "桌上放着一个黄色杯子。"
    objects = [
        dict(
            object_id="table_1",
            category="table",
            description="桌子",
            attributes=[],
            mentions=["桌"],
        ),
        dict(
            object_id="cup_1",
            category="cup",
            description="黄色杯子",
            attributes=["黄色"],
            mentions=["一个黄色杯子"],
        ),
    ]
    broken = json.loads(json.dumps(objects))
    broken[0]["mentions"].append("不存在的桌面")
    responses = [
        dict(objects=broken, ambiguities=[]),
        dict(objects=objects, ambiguities=[]),
        dict(
            relations=[dict(relation="on", source="cup_1", target="table_1", evidence=text)],
            ambiguities=[],
        ),
    ]
    requests = []

    def send(system, user):
        requests.append(json.loads(user))
        return json.dumps(responses[len(requests) - 1], ensure_ascii=False)

    provider = extraction.AssetProvider(config(), transport_fn=send, cache_dir=tmp_path)
    result = provider.extract(text)
    assert result["objects"] == objects
    assert len(requests) == 3
    assert "validation_error" in requests[1]
    assert provider.evidence()["stages"]["objects"]["rejected_response"]["objects"] == broken


def test_repeated_invalid_correction_stops_after_two_calls(tmp_path):
    import json
    from self_improving.sim_adapters.genesis.tests.test_asset_extraction import config

    calls = []

    def send(*args):
        calls.append(1)
        return json.dumps(dict(objects=[], ambiguities=[]))

    provider = extraction.AssetProvider(config(), transport_fn=send, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        provider.extract("桌上放杯子。")
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.json"))


def test_visual_containment_batches_bound_ray_expansion(monkeypatch):
    visual = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    # An inscribed cube has artificial internal cut faces: exercise the original
    # solid containment path, preserving every requested sample across batches.
    part = trimesh.creation.box([0.05, 0.05, 0.05])
    seen = []
    original = trimesh.Trimesh.contains

    def bounded(mesh, points):
        seen.append((len(mesh.faces), len(points)))
        assert len(mesh.faces) * len(points) <= 1_000_000
        return original(mesh, points)

    monkeypatch.setattr(trimesh.Trimesh, 'contains', bounded)
    report = prep.proxy_quality(visual, [part])
    assert seen and len(seen) > 1
    # Missing the original sphere surface must still fail despite internal caps.
    assert not report['passed']

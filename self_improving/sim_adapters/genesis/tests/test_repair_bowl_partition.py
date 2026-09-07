"""Attacks on exact measured-profile partitioning without vendor assets or Genesis."""

import json

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_bowl_partition as bowl
from self_improving.sim_adapters.genesis import repair_collision_v2 as v2


def lathe(profile, segments=64):
    vertices = []
    rings = []
    for r, z in profile:
        if r == 0:
            rings.append([len(vertices)] * segments)
            vertices.append([0, 0, z])
        else:
            rings.append(list(range(len(vertices), len(vertices) + segments)))
            vertices.extend(
                [
                    [r * np.cos(a), r * np.sin(a), z]
                    for a in np.arange(segments) * 2 * np.pi / segments
                ]
            )
    faces = []
    for lower, upper in zip(rings[:-1], rings[1:]):
        for k in range(segments):
            a, b, c, d = lower[k], upper[k], upper[(k + 1) % segments], lower[(k + 1) % segments]
            faces.extend(f for f in [[a, c, b], [a, d, c]] if len(set(f)) == 3)
    return trimesh.Trimesh(vertices, faces, process=True)


def cylinder():
    return lathe([[0, 0], [0.05, 0], [0.05, 0.04], [0, 0.04]])


def test_exact_profile_partition_retains_concave_rim_and_all_boundary_edges():
    profile = np.array(
        [
            [0, 0],
            [0.05, 0],
            [0.08, 0.04],
            [0.09, 0.045],
            [0.085, 0.05],
            [0.08, 0.046],
            [0.055, 0.015],
            [0.04, 0.004],
            [0, 0.004],
        ]
    )
    parts = bowl.partition_profile(profile)
    assert len(parts) > 1
    assert sum(bowl.area(profile[p]) for p in parts) == pytest.approx(bowl.area(profile))
    edges = []
    for p in parts:
        xy = profile[p]
        assert all(
            bowl.cross(xy[k] - xy[k - 1], xy[(k + 1) % len(p)] - xy[k]) >= -1e-14
            for k in range(len(p))
        )
        edges.extend(zip(p, p[1:] + p[:1]))
    for edge in zip(range(len(profile)), list(range(1, len(profile))) + [0]):
        assert edges.count(edge) == 1 and edges.count(edge[::-1]) == 0
    for edge in edges:
        if edge[1] != (edge[0] + 1) % len(profile):
            assert edges.count(edge[::-1]) == 1


def test_crossing_profile_is_rejected_even_if_its_total_area_is_positive():
    crossed = np.array(
        [
            [0, 0],
            [0.05, 0],
            [0.08, 0.04],
            [0.09, 0.045],
            [0.085, 0.05],
            [0.08, 0.046],
            [0.07, 0.015],
            [0.04, 0.004],
            [0, 0.004],
        ]
    )
    assert bowl.area(crossed) > 0
    with pytest.raises(ValueError, match="self-intersecting"):
        bowl.partition_profile(crossed)


def test_non_rotational_source_is_rejected_before_generation():
    source = cylinder()
    source.vertices[10, 0] += 0.001
    with pytest.raises(ValueError, match="lathe|rotational"):
        bowl.measure_profile(source)


def test_nonzero_secondary_component_cannot_be_dropped():
    extra = trimesh.creation.box([0.01, 0.01, 0.01])
    extra.apply_translation([0.2, 0, 0])
    with pytest.raises(ValueError, match="exactly one closed"):
        bowl.measure_profile(trimesh.util.concatenate([cylinder(), extra]))


def test_parts_use_original_vertices_and_fit_complete_original_surface():
    source = cylinder()
    solid, lookup, measurement = bowl.measure_profile(source)
    parts, records = bowl.generate(
        solid, lookup, measurement, bowl.candidates(len(source.faces))[0]
    )
    assert len(parts) == 16
    for mesh, record in zip(parts, records):
        assert prep.closed_convex(mesh)
        available = {tuple(v) for v in solid.vertices[record["source_vertex_indices"]]}
        assert all(tuple(v) in available for v in mesh.vertices)
    quality = v2.quality_gate(source, parts)
    assert quality["passed"] and quality["bottom_support_check"]["error_m"] == 0


def test_part_budget_rejects_before_building_hulls(monkeypatch):
    source = cylinder()
    solid, lookup, measurement = bowl.measure_profile(source)
    candidate = dict(bowl.candidates(len(source.faces))[0], max_convex_hull=1)
    monkeypatch.setattr(bowl, "convex_mesh", lambda _: pytest.fail("budget must precede geometry"))
    with pytest.raises(ValueError, match="total part budget"):
        bowl.generate(solid, lookup, measurement, candidate)


def test_worker_binds_measured_input_and_preserves_original_reference(tmp_path):
    source = cylinder()
    path = tmp_path / "source.npz"
    np.savez_compressed(path, vertices=source.vertices, faces=source.faces)
    out = tmp_path / "worker"
    out.mkdir()
    request = dict(
        source=str(path),
        source_sha256=prep.lib.sha256(path),
        output_dir=str(out),
        candidate=bowl.candidates(len(source.faces))[0],
        surface=None,
        identity={"visible_mesh_sha256": v2.mesh_hash(source)},
    )
    result = bowl.worker(request)
    assert result["passed"] and len(result["parts"]) == 16
    assert not result["generation_provenance"]["native_mass_properties_modified"]
    assert result["generation_provenance"]["original_visual_sha256"] == v2.mesh_hash(source)
    assert result["parts_sha256"] == result["generation_provenance"]["part_sha256"]
    assert set(result["parts_sha256"]) == set(result["parts"])
    for name, digest in result["parts_sha256"].items():
        assert prep.lib.sha256(out / name) == digest
    for name, digest in result["processed_input_sha256"].items():
        assert prep.lib.sha256(out / name) == digest
    profile = json.loads((out / "inputs/measured_profile.json").read_text())
    assert profile["source_solid_faces"] == len(source.faces)
    assert len(profile["source_vertex_profile_indices"]) == len(source.vertices)


@pytest.mark.parametrize("changed", ["part", "input"])
def test_replacing_a_qualified_file_cannot_reseal_it(tmp_path, monkeypatch, changed):
    source = cylinder()
    path = tmp_path / "source.npz"
    np.savez_compressed(path, vertices=source.vertices, faces=source.faces)
    out = tmp_path / "worker"
    out.mkdir()
    original_quality = v2.quality_gate

    def replaced_after_measurement(*args, **kwargs):
        result = original_quality(*args, **kwargs)
        assert result["passed"]
        if changed == "part":
            trimesh.creation.box([1, 1, 1]).export(out / "parts/part_000.obj")
        else:
            (out / "inputs/measured_profile.json").write_text("{}")
        return result

    monkeypatch.setattr(v2, "quality_gate", replaced_after_measurement)
    with pytest.raises(ValueError, match="changed during qualification"):
        bowl.worker(
            dict(
                source=str(path),
                source_sha256=prep.lib.sha256(path),
                output_dir=str(out),
                candidate=bowl.candidates(len(source.faces))[0],
                surface=None,
                identity={"visible_mesh_sha256": v2.mesh_hash(source)},
            )
        )

"""Attacks on bounded collision preparation, independent of Genesis and real assets."""

import json

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_collision_v2 as v2


def edge_shared_tetrahedra():
    """Two closed tetrahedra share one edge: no boundary, four incident faces."""
    a = trimesh.Trimesh(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        process=True,
    )
    b = a.copy()
    b.vertices[:, 1:] *= -1
    return trimesh.Trimesh(
        *(lambda joined: (joined.vertices, joined.faces))(trimesh.util.concatenate([a, b])),
        process=True,
    )


def test_nonmanifold_closed_looking_mesh_is_rejected():
    report = v2.topology_report(edge_shared_tetrahedra())
    assert report["boundary_edges"] == 0
    assert report["nonmanifold_edges"] == 1
    assert not report["passed"]


def test_decimation_does_not_feed_invalid_topology_to_coacd(monkeypatch):
    original = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    before = v2.mesh_hash(original)
    monkeypatch.setattr(
        trimesh.Trimesh, "simplify_quadric_decimation", lambda *a, **kw: edge_shared_tetrahedra()
    )
    processed, report = v2.prepare_input(original, 100)
    assert report["before"]["passed"]
    assert not report["after"]["passed"]
    assert not report["passed"]
    assert v2.mesh_hash(original) == before
    assert v2.mesh_hash(processed) != before


def test_candidate_order_is_finite_and_does_not_repeat_bad_2000_face_apple():
    apple = v2.candidates("apple", 99994)
    assert [c["face_count"] for c in apple] == [8000, 20000, None, None]
    assert [c["max_convex_hull"] for c in apple] == [64, 64, 64, 128]
    bowl = v2.candidates("bowl", 5376)
    assert all(c["face_count"] is None for c in bowl)
    assert [c["max_convex_hull"] for c in bowl] == [64, 128, 128, 128]
    assert len(apple) == len(bowl) == 4


def test_quality_uses_original_visual_and_preserves_cavity_rejection():
    left = trimesh.creation.box([0.01, 0.05, 0.06])
    left.apply_translation([-0.03, 0, 0])
    right = left.copy()
    right.apply_translation([0.06, 0, 0])
    original = trimesh.util.concatenate([left, right])
    bridge = original.convex_hull
    assert prep.proxy_quality(bridge, [bridge])["passed"]
    assert not prep.proxy_quality(original, [bridge])["passed"]


def test_worker_timeout_is_preserved_and_next_candidate_can_pass(tmp_path, monkeypatch):
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    calls = []

    def worker(request_path, directory, timeout_s):
        request = json.loads(request_path.read_text())
        calls.append((request["candidate"]["id"], timeout_s))
        if len(calls) == 1:
            (directory / "partial.log").write_text("timed out while decomposing")
            return {"passed": False, "status": "timeout", "timeout_s": timeout_s}
        return successful_worker(request, directory, visual)

    monkeypatch.setattr(v2, "run_worker", worker)
    parts, report = v2.decompose(
        visual, None, tmp_path / "asset", category="apple", cache_root=tmp_path / "cache"
    )
    assert len(calls) == 2
    assert all(0 < timeout <= 300 for _, timeout in calls)
    assert len(parts) == 1 and report["quality"]["passed"]
    attempts = json.loads((tmp_path / "asset/collision_attempts.json").read_text())
    assert attempts[0]["status"] == "timeout"
    assert (tmp_path / "asset/collision_candidates/000/partial.log").exists()


def successful_worker(request, directory, visual):
    part_path = directory / "parts/part_000.obj"
    part_path.parent.mkdir(exist_ok=True)
    visual.export(part_path)
    input_path = directory / "inputs/component_000.npz"
    input_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(input_path, vertices=visual.vertices, faces=visual.faces)
    result = {
        "passed": True,
        "status": "passed",
        "parts": ["parts/part_000.obj"],
        "quality": prep.proxy_quality(visual, [visual]),
        "processed_input_sha256": {"inputs/component_000.npz": prep.lib.sha256(input_path)},
        "request": request["identity"],
    }
    prep.clip.write_json(directory / "result.json", result)
    return result


def test_cache_is_success_only_and_revalidates_hashes(tmp_path, monkeypatch):
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    cache = tmp_path / "cache"
    calls = []

    def worker(request_path, directory, timeout_s):
        request = json.loads(request_path.read_text())
        calls.append(request.get("reuse_dir"))
        return successful_worker(request, directory, visual)

    monkeypatch.setattr(v2, "run_worker", worker)
    _, first = v2.decompose(visual, None, tmp_path / "first", category="bowl", cache_root=cache)
    _, second = v2.decompose(visual, None, tmp_path / "second", category="bowl", cache_root=cache)
    assert not first["cache_hit"] and second["cache_hit"]
    assert calls[1] is not None  # Cached parts are still checked by the bounded worker.
    obj = next((cache / "entries").glob("*/parts/part_000.obj"))
    obj.write_text("corrupt mesh")
    with pytest.raises(ValueError, match="hash|mismatch|integrity"):
        v2.decompose(visual, None, tmp_path / "third", category="bowl", cache_root=cache)


def test_asset_deadline_is_total_not_multiplied_by_components(tmp_path, monkeypatch):
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    clock = [0.0]
    monkeypatch.setattr(v2.time, "monotonic", lambda: clock[0])
    calls = []

    def worker(request_path, directory, timeout_s):
        calls.append(timeout_s)
        clock[0] = 1201.0
        return {"passed": False, "status": "timeout"}

    monkeypatch.setattr(v2, "run_worker", worker)
    with pytest.raises(ValueError, match="ASSET_PREPARATION_FAILED"):
        v2.decompose(
            visual, None, tmp_path / "asset", category="apple", cache_root=tmp_path / "cache"
        )
    assert len(calls) == 1
    result = json.loads((tmp_path / "asset/collision_failure.json").read_text())
    assert result["budget_exhausted"]


def test_cache_lock_held_by_other_process_ends_at_shared_asset_deadline(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    cache = tmp_path / "cache"
    cache.mkdir()
    request_identity = v2.identity(visual, None, v2.candidates("apple", len(visual.faces))[0])
    lock_path = cache / f".{v2.digest(request_identity)}.lock"
    child = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); "
         "print('locked'); sys.stdin.read()", str(lock_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        monkeypatch.setattr(
            v2, "run_worker",
            lambda request, directory, timeout: successful_worker(
                json.loads(request.read_text()), directory, visual
            ),
        )
        started = time.monotonic()
        with pytest.raises(ValueError, match="ASSET_PREPARATION_FAILED"):
            v2.decompose(
                visual, None, tmp_path / "asset", category="apple", cache_root=cache,
                deadline=started + 0.5,
            )
        elapsed = time.monotonic() - started
        assert 0.5 <= elapsed < 3.0
        report = json.loads((tmp_path / "asset/collision_failure.json").read_text())
        assert report["budget_exhausted"] and report["phase"] == "cache_publish_lock"
        attempt = report["attempts"][0]
        assert attempt["status"] == "asset_budget_exhausted"
        assert attempt["geometry_passed"] and 0.4 < attempt["elapsed_s"] <= elapsed
        raw = json.loads((tmp_path / "asset/collision_candidates/000/result.json").read_text())
        assert raw["passed"]  # Preserve the valid worker result and publication failure separately.
        assert not list((cache / "entries").iterdir())
        assert not list((cache / "requests").iterdir())
    finally:
        child.communicate(timeout=3)


def test_worker_thread_policy_is_applied_and_bound_to_cache_identity(tmp_path, monkeypatch):
    seen = {}

    class Process:
        def __init__(self, command, **options):
            seen.update(options)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(v2.subprocess, "Popen", Process)
    monkeypatch.setenv("OMP_NUM_THREADS", "42")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "42")
    monkeypatch.setenv("MKL_NUM_THREADS", "42")
    v2.run_worker(tmp_path / "request.json", tmp_path, 0.5)
    expected = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    assert {key: seen["env"][key] for key in expected} == expected
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    candidate = v2.candidates("apple", len(visual.faces))[0]
    first = v2.identity(visual, None, candidate)
    assert first["thread_policy"] == {"openmp": 8, "openblas": 1, "mkl": 1}
    monkeypatch.setattr(v2, "WORKER_OMP_THREADS", 1)
    assert v2.digest(first) != v2.digest(v2.identity(visual, None, candidate))


def test_v2_frozen_mass_does_not_depend_on_collision_proxy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("v2 must not estimate inertia from a replacement proxy")

    monkeypatch.setattr(prep, "proxy_inertia", forbidden)
    frozen = (0.2, np.array([0.01, 0, 0]), np.diag([0.001, 0.002, 0.003]))
    mass, com, inertia, source = prep.select_mass_properties(
        False, frozen, [], 0.1, "text_scene_v2"
    )
    assert mass == frozen[0]
    np.testing.assert_array_equal(com, frozen[1])
    np.testing.assert_array_equal(inertia, frozen[2])
    assert source == "native_loaded_mass_and_inertia_scaled_s3_s5"


def test_legacy_mass_selection_still_estimates_non_mjcf(monkeypatch):
    seen = []
    monkeypatch.setattr(
        prep,
        "proxy_inertia",
        lambda parts, diagonal: seen.append(parts) or (0.1, np.zeros(3), np.eye(3)),
    )
    result = prep.select_mass_properties(False, None, ["part"], 0.1, "legacy")
    assert seen == [["part"]]
    assert result[0] == 0.1


def test_real_worker_enforces_original_quality_and_reuses_exact_inputs(tmp_path):
    visual = trimesh.creation.box([0.04, 0.05, 0.06])
    cache = tmp_path / "cache"
    _, first = v2.decompose(visual, None, tmp_path / "first", category="apple", cache_root=cache)
    _, second = v2.decompose(visual, None, tmp_path / "second", category="apple", cache_root=cache)
    assert first["quality"]["passed"] and second["quality"]["passed"]
    assert first["key"] == second["key"]
    assert second["cache_hit"]
    assert first["operation"] == "generated_candidate"
    assert second["operation"] == "cached_quality_recheck"
    assert not second["coacd_executed"] and second["options"] is None
    assert first["generation_provenance"] == second["generation_provenance"]
    assert first["attempts"][0]["processed_input_sha256"]
    assert (
        first["attempts"][0]["processed_input_sha256"]
        == second["attempts"][0]["processed_input_sha256"]
    )


def test_legacy_recheck_does_not_claim_current_parameters_were_executed(tmp_path, monkeypatch):
    import coacd

    visual = trimesh.creation.box([0.04, 0.05, 0.06])
    source = tmp_path / "visual.npz"
    np.savez_compressed(source, vertices=visual.vertices, faces=visual.faces)
    old = tmp_path / "legacy"
    old.mkdir()
    visual.export(old / "part.obj")
    out = tmp_path / "worker"
    out.mkdir()
    candidate = v2.candidates("bowl", len(visual.faces))[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("a cache quality recheck must not execute CoACD")

    monkeypatch.setattr(coacd, "run_coacd", forbidden)
    result = v2.worker(dict(
        source=str(source), source_sha256=prep.lib.sha256(source), output_dir=str(out),
        candidate=candidate, surface=None, identity=v2.identity(visual, None, candidate),
        reuse_dir=str(old), reuse_parts=["part.obj"],
    ))
    assert result["passed"] and result["operation"] == "cached_quality_recheck"
    assert not result["coacd_executed"] and result["effective_options"] is None
    assert result["configured_options"]["threshold"] == candidate["threshold"]
    assert result["generation_provenance"]["status"] == "unknown"


def test_cached_generation_provenance_is_hash_bound(tmp_path, monkeypatch):
    visual = trimesh.creation.box([0.1, 0.1, 0.1])
    cache = tmp_path / "cache"

    def worker(request_path, directory, timeout):
        result = successful_worker(json.loads(request_path.read_text()), directory, visual)
        result["generation_provenance"] = {"status": "recorded", "coacd_calls": []}
        return result

    monkeypatch.setattr(v2, "run_worker", worker)
    v2.decompose(visual, None, tmp_path / "first", category="apple", cache_root=cache)
    manifest_path = next((cache / "entries").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["descriptor"]["generation_provenance"]["status"] = "forged"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        v2.decompose(visual, None, tmp_path / "second", category="apple", cache_root=cache)


def test_component_budget_limits_the_whole_asset(tmp_path):
    from pathlib import Path

    meshes = []
    for i in range(13):
        box = trimesh.creation.box([0.01, 0.01, 0.01])
        box.apply_translation([i * 0.03, 0, 0])
        meshes.append(box)
    visual = trimesh.util.concatenate(meshes)
    source = tmp_path / "visual.npz"
    np.savez_compressed(source, vertices=visual.vertices, faces=visual.faces)
    out = tmp_path / "worker"
    out.mkdir()
    candidate = v2.candidates("bowl", len(visual.faces))[0]
    request = dict(
        source=str(source),
        source_sha256=prep.lib.sha256(source),
        output_dir=str(out),
        candidate=candidate,
        surface=None,
        identity=v2.identity(visual, None, candidate),
    )
    with pytest.raises(ValueError, match="component limit"):
        v2.worker(request)
    assert not list(Path(out).glob("parts/*"))


def fin_mesh(*, same_winding=False, copies=2):
    box = trimesh.creation.box([0.1, 0.1, 0.1])
    vertices = np.vstack([box.vertices, [0.2, 0.0, 0.0]])
    left = [0, 1, len(vertices) - 1]
    right = left if same_winding else left[::-1]
    fins = [left, right] + [left] * (copies - 2)
    return trimesh.Trimesh(vertices, np.vstack([box.faces, fins]), process=False)


def test_only_exact_opposed_pairs_are_removed_without_changing_volume():
    raw = fin_mesh()
    original_hash = v2.mesh_hash(raw)
    assert not v2.topology_report(raw)["passed"]
    cleaned, report = v2.remove_opposed_duplicate_faces(raw)
    assert report["removed_face_indices"] == [12, 13]
    assert report["volume_preserved"]
    assert v2.topology_report(cleaned)["passed"]
    assert v2.mesh_hash(raw) == original_hash
    assert report["raw_mesh_sha256"] == original_hash
    assert report["cleaned_mesh_sha256"] == v2.mesh_hash(cleaned)


@pytest.mark.parametrize("same_winding,copies", [(True, 2), (False, 3), (False, 4)])
def test_same_winding_and_multiple_overlaps_remain_rejected(same_winding, copies):
    raw = fin_mesh(same_winding=same_winding, copies=copies)
    cleaned, report = v2.remove_opposed_duplicate_faces(raw)
    assert not report["removed_face_indices"]
    assert not v2.topology_report(cleaned)["passed"]


def test_derived_bottom_mismatch_between_1_and_2_mm_is_rejected_in_v2():
    visual = trimesh.creation.box([0.2, 0.2, 0.2])
    proxy = visual.copy()
    proxy.apply_translation([0, 0, 0.0015])
    result = v2.quality_gate(visual, [proxy])
    assert result["original_geometry_check"]["passed"]
    assert not result["bottom_support_check"]["passed"]
    assert not result["passed"]
    assert prep.proxy_quality(visual, [proxy])["passed"]  # Legacy is untouched.


def test_native_qualification_is_bounded_and_retains_a_failed_geometry_result(tmp_path):
    visual = trimesh.creation.box([0.2, 0.2, 0.2])
    proxy = visual.copy()
    proxy.apply_translation([0, 0, 0.003])
    quality = v2.native_quality(
        visual, [proxy], None, tmp_path, deadline=v2.time.monotonic() + 30, native_semantics=True
    )
    assert not quality["passed"]
    attempt = json.loads((tmp_path / "native_quality_worker/attempt.json").read_text())
    assert attempt["status"] == "completed"  # Execution succeeded; the geometry did not.
    assert attempt["elapsed_s"] < 30

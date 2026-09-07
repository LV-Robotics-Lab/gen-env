"""Frozen generation controls must not weaken apple geometry validation."""

import copy
import time

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import repair_apple_fast as fast


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    mesh = trimesh.creation.box([0.02, 0.03, 0.04])
    source = tmp_path / "visual.npz"
    np.savez_compressed(source, vertices=mesh.vertices, faces=mesh.faces)
    native = tmp_path / "native.json"
    native.write_text('{"mass_kg":1,"com":[0,0,0],"inertia":[1,1,1]}')
    for key, value in (("OMP_NUM_THREADS", "8"), ("OPENBLAS_NUM_THREADS", "1"),
                       ("MKL_NUM_THREADS", "1")):
        monkeypatch.setenv(key, value)
    return fast.request(source, native, tmp_path / "trial", fast.CANDIDATES[0],
                        deadline=time.monotonic() + 1200)


def test_search_is_finite_and_seed_geometry_options_are_frozen():
    assert len(fast.CANDIDATES) == 2
    for candidate in fast.CANDIDATES:
        cfg = fast.options(candidate)
        assert (cfg["seed"], cfg["mcts_iterations"], cfg["mcts_nodes"],
                cfg["mcts_max_depth"]) == (0, 5, 20, 3)
        assert cfg["preprocess_mode"] == "off" and not cfg["real_metric"]
    with pytest.raises(ValueError, match="unknown"):
        fast.options(dict(fast.CANDIDATES[0], threshold=1.0))


@pytest.mark.parametrize("attack", ["seed", "threshold", "reuse", "native", "source"])
def test_forged_request_is_rejected(frozen, attack):
    if attack in ("seed", "threshold"):
        frozen["coacd_options"][attack] = 123
    elif attack == "reuse":
        frozen["reuse_dir"] = "/tmp/cached"
    else:
        from pathlib import Path
        path = (frozen["identity"]["native_properties"]["path"] if attack == "native"
                else frozen["source"])
        p = Path(path)
        p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen"):
        fast.validate_request(frozen)


@pytest.mark.parametrize("raises", [False, True])
def test_worker_uses_actual_fast_options_and_restores_shared_defaults(frozen, monkeypatch, raises):
    original = fast.prep.COACD
    before = copy.deepcopy(original)

    def delegate(data):
        assert fast.prep.COACD == frozen["coacd_options"]
        assert data["source"] == frozen["source"] and "reuse_dir" not in data
        if raises:
            raise RuntimeError("generation failed")
        return dict(passed=False, status="quality_rejected", coacd_executed=True,
                    configured_options=copy.deepcopy(fast.prep.COACD), quality=dict(passed=False))

    monkeypatch.setattr(fast.collision, "worker", delegate)
    if raises:
        with pytest.raises(RuntimeError):
            fast.worker(frozen)
    else:
        assert not fast.worker(frozen)["passed"]
    assert fast.prep.COACD is original and fast.prep.COACD == before


def test_failed_quality_cannot_be_reported_as_success(frozen, monkeypatch):
    monkeypatch.setattr(fast.collision, "worker", lambda data: {
        "passed": True, "quality": {"passed": False}})
    with pytest.raises(ValueError, match="geometry"):
        fast.worker(frozen)


def test_parent_environment_cannot_silently_change_thread_policy(frozen, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(ValueError, match="thread"):
        fast.worker(frozen)


def test_v3_strategy_interface_preserves_candidate_options(frozen, monkeypatch):
    candidate = fast.candidates(99994)[0]
    cfg = fast.options(candidate)
    data = dict(frozen, candidate=candidate, strategy=fast.STRATEGY,
                schema_version="genenv.collision_v3_request.v1", coacd_options=cfg)
    data["identity"] = dict(frozen["identity"], strategy=fast.STRATEGY, coacd_options=cfg)
    monkeypatch.setattr(fast.collision, "worker", lambda request: {
        "passed": False, "status": "quality_rejected", "quality": {"passed": False},
        "coacd_executed": True, "configured_options": copy.deepcopy(fast.prep.COACD)})
    assert fast.worker(data)["configured_options"] == cfg
    assert candidate["original_faces"] == 99994
    data["coacd_options"] = dict(cfg, mcts_iterations=100)
    with pytest.raises(ValueError, match="strategy"):
        fast.worker(data)


@pytest.mark.parametrize("mutate", [False, True])
def test_quality_reads_serialized_parts_and_rejects_mid_gate_mutation(frozen, monkeypatch, mutate):
    from pathlib import Path
    out = Path(frozen["output_dir"])
    part_path = out / "parts/part_000.obj"
    part_path.parent.mkdir(parents=True)
    mesh = trimesh.creation.box([0.02, 0.03, 0.04])
    mesh.export(part_path)
    shifted = mesh.copy()
    shifted.apply_translation([10, 0, 0])

    def gate(visual, parts, surface, **flags):
        assert np.allclose(parts[0].bounds, mesh.bounds)
        if mutate:
            part_path.write_text(part_path.read_text() + "# modified during gate\n")
        return {"passed": True}

    monkeypatch.setattr(fast.collision, "quality_gate", gate)
    monkeypatch.setattr(fast.collision, "worker", lambda data: dict(
        passed=True, parts=["parts/part_000.obj"],
        quality=fast.collision.quality_gate(mesh, [shifted], None)))
    if mutate:
        with pytest.raises(ValueError, match="changed during qualification"):
            fast.worker(frozen)
    else:
        result = fast.worker(frozen)
        assert result["parts_sha256"] == {"parts/part_000.obj": fast.prep.lib.sha256(part_path)}


def test_native_call_timeout_kills_only_child_and_preserves_inputs(frozen, monkeypatch):
    import subprocess
    from pathlib import Path

    import coacd
    killed = []
    waits = []

    class Child:
        def wait(self, timeout=None):
            waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("coacd", timeout)
            return -9

        def kill(self):
            killed.append(True)

    def popen(args, **kwargs):
        assert args[2] == fast.MODULE and args[3] == "--coacd-call"
        assert not kwargs.get("start_new_session", False)
        return Child()

    monkeypatch.setattr(fast.subprocess, "Popen", popen)
    monkeypatch.setattr(fast, "WORKER_LIMIT_S", 0.05)
    mesh = trimesh.creation.box()
    with pytest.raises(TimeoutError, match="CoACD"):
        fast.bounded_coacd(coacd.Mesh(mesh.vertices, mesh.faces), frozen["coacd_options"],
                           frozen, 0, time.monotonic()+1200)
    assert killed == [True] and 0 < waits[0] <= 0.05 and waits[1] is None
    out = Path(frozen["output_dir"]) / "coacd_calls/000"
    assert (out / "input.npz").exists() and (out / "options.json").exists()
    assert fast.prep.lib.read_json(out / "timing.json")["status"] == "coacd_timeout"


def test_native_subprocess_uses_frozen_options_and_returns_real_parts(frozen):
    import coacd
    mesh = trimesh.creation.box([0.02, 0.03, 0.04])
    record, parts = fast.bounded_coacd(coacd.Mesh(mesh.vertices, mesh.faces),
                                      frozen["coacd_options"], frozen, 0,
                                      time.monotonic()+30)
    assert record["result"]["passed"] and record["timing"]["elapsed_s"] < 30
    assert len(parts) == 1
    assert np.allclose(trimesh.Trimesh(*parts[0], process=False).bounds, mesh.bounds)

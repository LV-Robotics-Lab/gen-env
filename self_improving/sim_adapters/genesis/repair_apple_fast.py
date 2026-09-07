"""Isolated finite apple generation experiments with the unchanged visual quality gate.

These candidates are experimental, not registered asset overrides. Every candidate
starts from the complete original visual mesh. Only its isolated subprocess changes
CoACD generation defaults; mass, COM, inertia and shared source files are untouched.
"""

from __future__ import annotations

import copy
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_collision_v2 as collision

POLICY = "apple_fast_coacd_experiment_v2"
MODULE = "self_improving.sim_adapters.genesis.repair_apple_fast"
WORKER_LIMIT_S = 300.0
ASSET_BUDGET_S = 1200.0
THREADS = dict(openmp=8, openblas=1, mkl=1)
STRATEGY = "apple_fast_v1"
FAST_COACD = dict(prep.COACD, preprocess_mode="off", real_metric=False,
                  mcts_iterations=5, mcts_nodes=20, mcts_max_depth=3, seed=0)
CANDIDATES = (
    dict(id="apple_fast_8k_t01_h256_r2000", face_count=8000,
         threshold=0.01, max_convex_hull=256, resolution=2000),
    dict(id="apple_fast_8k_t01_h512_r2000", face_count=8000,
         threshold=0.01, max_convex_hull=512, resolution=2000),
)


def candidates(original_faces):
    return [dict(row, original_faces=original_faces) for row in CANDIDATES]


def options(candidate):
    stripped = {key: value for key, value in candidate.items() if key != "original_faces"}
    if stripped not in CANDIDATES:
        raise ValueError("unknown experimental apple candidate")
    result = copy.deepcopy(FAST_COACD)
    result.update({key: candidate[key] for key in ("threshold", "max_convex_hull", "resolution")})
    return result


def request(source, native_properties, output_dir, candidate, *, deadline=None):
    source, native_properties = Path(source).resolve(), Path(native_properties).resolve()
    cfg = options(candidate)
    with np.load(source) as arrays:
        visual = trimesh.Trimesh(arrays["vertices"], arrays["faces"], process=False)
    identity = collision.identity(visual, None, candidate)
    identity.update(policy=POLICY, coacd_defaults=cfg, thread_policy=THREADS,
                    worker_limit_s=WORKER_LIMIT_S, asset_budget_s=ASSET_BUDGET_S,
                    native_properties=dict(path=str(native_properties),
                                           sha256=prep.lib.sha256(native_properties)),
                    reference="complete original visual mesh; native mass/COM/inertia unchanged")
    identity["implementation_sha256"][Path(__file__).name] = prep.lib.sha256(Path(__file__))
    result = dict(schema_version="genenv.apple_fast_request.v1", identity=identity,
                candidate=copy.deepcopy(candidate), coacd_options=cfg,
                source=str(source), source_sha256=prep.lib.sha256(source),
                surface=None, output_dir=str(Path(output_dir).resolve()))
    if deadline is not None:
        if not isinstance(deadline, (float, int)) or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("invalid frozen asset deadline")
        result["asset_deadline_monotonic"] = deadline
        identity["asset_deadline_monotonic"] = deadline
    return result


def validate_request(data):
    native = data["identity"]["native_properties"]
    expected = request(data["source"], native["path"], data["output_dir"], data["candidate"],
                       deadline=data.get("asset_deadline_monotonic"))
    if data != expected:
        raise ValueError("frozen apple experiment input or implementation mismatch")
    return data["coacd_options"]


def validate_worker_request(data):
    if data.get("schema_version") == "genenv.apple_fast_request.v1":
        return validate_request(data)
    cfg = options(data["candidate"])
    identity = data["identity"]
    if (data.get("strategy") != STRATEGY or identity.get("strategy") != STRATEGY
            or data.get("coacd_options") != cfg or identity.get("coacd_options") != cfg
            or identity.get("implementation_sha256", {}).get(Path(__file__).name)
            != prep.lib.sha256(Path(__file__))):
        raise ValueError("frozen apple strategy options or implementation mismatch")
    return cfg


def worker(data):
    """Scoped override lives only in this isolated worker and is restored on exit."""
    cfg = validate_worker_request(data)
    actual_threads = {"openmp": os.environ.get("OMP_NUM_THREADS"),
                      "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
                      "mkl": os.environ.get("MKL_NUM_THREADS")}
    if actual_threads != {key: str(value) for key, value in THREADS.items()}:
        raise ValueError("apple worker thread policy mismatch")
    import coacd

    previous_run = coacd.run_coacd
    bounded_calls = []
    deadline = data.get("asset_deadline_monotonic", math.inf)
    if math.isnan(deadline) or deadline <= time.monotonic():
        raise ValueError("apple asset deadline expired")
    def run_bounded(mesh, **effective):
        record, parts = bounded_coacd(mesh, effective, data, len(bounded_calls), deadline)
        bounded_calls.append(record)
        return parts

    qualified_parts = {}
    previous_quality = collision.quality_gate
    def serialized_quality(visual, parts, surface=None, **flags):
        directory = Path(data["output_dir"])
        names = sorted((directory / "parts").glob("*.obj"))
        if len(names) != len(parts):
            raise ValueError("serialized apple part count mismatch")
        frozen = {p.relative_to(directory).as_posix(): prep.lib.sha256(p) for p in names}
        actual = [trimesh.load(p, force="mesh", process=False) for p in names]
        quality = previous_quality(visual, actual, surface, **flags)
        if sorted((directory / "parts").glob("*.obj")) != names or any(
                prep.lib.sha256(directory / n) != h for n, h in frozen.items()):
            raise ValueError("serialized apple parts changed during qualification")
        qualified_parts.update(frozen)
        return quality

    previous = prep.COACD
    collision.quality_gate = serialized_quality
    coacd.run_coacd = run_bounded
    prep.COACD = copy.deepcopy(cfg)
    try:
        result = collision.worker(data)
    finally:
        prep.COACD = previous
        coacd.run_coacd = previous_run
        collision.quality_gate = previous_quality
    validate_worker_request(data)
    if result.get("coacd_executed") and result["configured_options"] != cfg:
        raise ValueError("executed CoACD configuration differs from frozen options")
    if prep.lib.sha256(Path(data["source"])) != data["source_sha256"]:
        raise ValueError("original visual changed during apple generation")
    if "parts" in result and (set(result["parts"]) != set(qualified_parts) or any(
            prep.lib.sha256(Path(data["output_dir"]) / n) != h
            for n, h in qualified_parts.items())):
        raise ValueError("reported apple parts differ from qualified serialized bytes")
    result["parts_sha256"] = qualified_parts
    if result.get("passed") and not result.get("quality", {}).get("passed"):
        raise ValueError("apple success lacks unchanged geometry gate")
    if "generation_provenance" in result:
        result["generation_provenance"]["bounded_coacd_calls"] = bounded_calls
    for path in sorted((Path(data["output_dir"]) / "coacd_calls").rglob("*")):
        if path.is_file():
            name = path.relative_to(data["output_dir"]).as_posix()
            result.setdefault("processed_input_sha256", {})[name] = prep.lib.sha256(path)
    result.update(experiment_policy=POLICY, thread_policy=actual_threads,
                  bounded_coacd_calls=bounded_calls,
                  native_properties=data["identity"].get("native_properties"))
    return result


def run_candidate(data, *, deadline):
    """Generation and full quality share remaining asset time; each CoACD gets 300 s."""
    if data.get("asset_deadline_monotonic") is None:
        data = request(data["source"], data["identity"]["native_properties"]["path"],
                       data["output_dir"], data["candidate"], deadline=deadline)
    elif data["asset_deadline_monotonic"] != deadline:
        raise ValueError("candidate deadline differs from the shared asset deadline")
    validate_request(data)
    out = Path(data["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    path = out / "request.json"
    prep.clip.write_json(path, data)
    digest = prep.lib.sha256(path)
    started = time.monotonic()
    timeout = min(ASSET_BUDGET_S, deadline - started)
    if timeout <= 0:
        result = dict(passed=False, status="asset_budget_exhausted", elapsed_s=0.0)
    else:
        env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen([sys.executable, "-m", MODULE, str(path)],
                                     cwd=Path(__file__).resolve().parents[3], env=env,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
                result = dict(passed=False, status="asset_budget_exhausted", timeout_s=timeout)
            else:
                result_path = out / "result.json"
                result = (prep.lib.read_json(result_path) if result_path.exists()
                          else dict(passed=False, status="worker_error"))
            if code:
                result["passed"] = False
            result.update(exit_code=code, elapsed_s=time.monotonic() - started)
    validate_request(data)
    if prep.lib.sha256(path) != digest:
        raise ValueError("frozen request bytes changed")
    if time.monotonic() > deadline:
        result.update(passed=False, status="asset_budget_exhausted")
    if result.get("passed"):
        if result.get("request") != data["identity"] or not result["quality"]["passed"]:
            raise ValueError("worker result is not bound to the requested experiment")
        for name, expected in result["processed_input_sha256"].items():
            if prep.lib.sha256(out / collision._safe_relative(name)) != expected:
                raise ValueError("processed input changed")
        if set(result["parts_sha256"]) != set(result["parts"]) or any(
                prep.lib.sha256(out / collision._safe_relative(name)) != expected
                for name, expected in result["parts_sha256"].items()):
            raise ValueError("qualified serialized apple part hash mismatch")
    result.update(request_sha256=digest, coacd_call_limit_s=WORKER_LIMIT_S,
                  worker_asset_time_remaining_s=timeout if timeout > 0 else 0.0)
    prep.clip.write_json(out / "attempt.json", result)
    prep.clip.write_json(out / "manifest.json", dict(
        request_sha256=digest, files=[prep.official.fingerprint(p, out)
                                     for p in sorted(out.rglob("*"))
                                     if p.is_file() and p.name != "manifest.json"]))
    return result


def bounded_coacd(mesh, options, data, index, deadline):
    """Native CoACD has a separate killable process inside the asset worker's group."""
    directory = Path(data["output_dir"]) / "coacd_calls" / f"{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "input.npz", vertices=mesh.vertices, faces=mesh.indices)
    prep.clip.write_json(directory / "options.json", options)
    call = dict(output_dir=str(directory), input_sha256=prep.lib.sha256(directory / "input.npz"),
                options_sha256=prep.lib.sha256(directory / "options.json"),
                parent_identity_sha256=collision.digest(data["identity"]),
                implementation_sha256=prep.lib.sha256(Path(__file__)))
    prep.clip.write_json(directory / "request.json", call)
    started = time.monotonic()
    timeout = min(WORKER_LIMIT_S, deadline - started)
    if timeout <= 0:
        raise TimeoutError("asset deadline expired before CoACD")
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        child = subprocess.Popen([sys.executable, "-m", MODULE, "--coacd-call",
                                  str(directory / "request.json")], stdout=stdout, stderr=stderr)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            prep.clip.write_json(directory / "timing.json", dict(
                status="coacd_timeout", elapsed_s=time.monotonic()-started, timeout_s=timeout))
            raise TimeoutError("native CoACD exceeded its bounded call deadline") from None
    timing = dict(status="completed" if code == 0 else "native_error", exit_code=code,
                  elapsed_s=time.monotonic()-started, timeout_s=timeout)
    prep.clip.write_json(directory / "timing.json", timing)
    if code != 0:
        raise RuntimeError(f"native CoACD exited {code}; see {directory}")
    report = prep.lib.read_json(directory / "result.json")
    if (not report["passed"] or report["input_sha256"] != call["input_sha256"]
            or report["options_sha256"] != call["options_sha256"]
            or prep.lib.sha256(directory / "parts.npz") != report["parts_sha256"]):
        raise ValueError("native CoACD output identity mismatch")
    with np.load(directory / "parts.npz") as arrays:
        parts = [(arrays[f"vertices_{i}"].copy(), arrays[f"faces_{i}"].copy())
                 for i in range(report["parts_count"])]
    return dict(call, timing=timing, result=report), parts


def native_coacd_call(data):
    import coacd

    directory = Path(data["output_dir"])
    if (prep.lib.sha256(Path(__file__)) != data["implementation_sha256"]
            or prep.lib.sha256(directory / "input.npz") != data["input_sha256"]
            or prep.lib.sha256(directory / "options.json") != data["options_sha256"]):
        raise ValueError("native CoACD frozen input mismatch")
    options = prep.lib.read_json(directory / "options.json")
    coacd.set_log_level("info")
    with np.load(directory / "input.npz") as arrays:
        mesh = coacd.Mesh(arrays["vertices"], arrays["faces"])
    started = time.monotonic()
    parts = coacd.run_coacd(mesh, **options)
    elapsed = time.monotonic() - started
    arrays = {}
    for i, (vertices, faces) in enumerate(parts):
        arrays[f"vertices_{i}"] = vertices
        arrays[f"faces_{i}"] = faces
    np.savez_compressed(directory / "parts.npz", **arrays)
    return dict(data, passed=True, status="completed", coacd_elapsed_s=elapsed,
                parts_count=len(parts), parts_sha256=prep.lib.sha256(directory / "parts.npz"))


def main():
    import resource
    import traceback

    resource.setrlimit(resource.RLIMIT_AS,
                       (collision.WORKER_MEMORY_BYTES, collision.WORKER_MEMORY_BYTES))
    native_call = sys.argv[1] == "--coacd-call"
    data = prep.lib.read_json(Path(sys.argv[2] if native_call else sys.argv[1]))
    try:
        result = native_coacd_call(data) if native_call else worker(data)
    except Exception as exc:
        result = dict(passed=False, status="coacd_timeout" if isinstance(exc, TimeoutError)
                      else "worker_error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    prep.clip.write_json(Path(data["output_dir"]) / "result.json", result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""User-authorized collision generation extension with unchanged geometric acceptance."""

from __future__ import annotations

import copy
import hashlib
import importlib
import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_collision_v2 as gate

POLICY = "collision_repair_v3"
CACHE = prep.CACHE.parent / "repair_collision_cache_v3"
STRATEGIES = {"apple": "apple_fast_v1", "bowl": "bowl_partition_v1"}
MODULES = {
    "apple_fast_v1": "repair_apple_fast",
    "bowl_partition_v1": "repair_bowl_partition",
}


def strategy_module(strategy):
    if strategy not in MODULES:
        raise ValueError("unknown collision generation strategy")
    return importlib.import_module(f"self_improving.sim_adapters.genesis.{MODULES[strategy]}")


def identity(visual, surface, candidate, strategy):
    module = strategy_module(strategy)
    result = gate.identity(visual, surface, candidate)
    result.update(policy=POLICY, strategy=strategy)
    result["implementation_sha256"].update({
        Path(__file__).name: prep.lib.sha256(__file__),
        Path(module.__file__).name: prep.lib.sha256(module.__file__),
    })
    if strategy == "apple_fast_v1":
        result["coacd_options"] = module.options(candidate)
    else:
        result["partition_configuration"] = candidate
    return result


def run_worker(request_path, directory, timeout_s):
    env = dict(os.environ, OMP_NUM_THREADS=str(gate.WORKER_OMP_THREADS),
               OPENBLAS_NUM_THREADS=str(gate.WORKER_BLAS_THREADS),
               MKL_NUM_THREADS=str(gate.WORKER_BLAS_THREADS))
    started = time.monotonic()
    with (directory / "stdout.log").open("wb") as stdout, (
        directory / "stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(request_path)],
            cwd=Path(__file__).resolve().parents[3], env=env,
            stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return dict(passed=False, status="timeout", phase="worker", timeout_s=timeout_s,
                        elapsed_s=time.monotonic() - started, exit_code=process.returncode)
    result_path = directory / "result.json"
    result = prep.lib.read_json(result_path) if result_path.exists() else dict(
        passed=False, status="worker_error", error="worker produced no result"
    )
    if code:
        result.update(passed=False, exit_code=code)
    result["elapsed_s"] = time.monotonic() - started
    return result


def decompose(visual, surface, out, *, category, check=lambda: None, deadline=None,
              cache_root=None):
    """Qualify one finite strategy, publishing only independently qualified geometry."""
    if category not in STRATEGIES:
        return gate.decompose(visual, surface, out, category=category, check=check,
                              deadline=deadline)
    started = time.monotonic()
    deadline = started + gate.ASSET_BUDGET_S if deadline is None else deadline
    cache_root = CACHE if cache_root is None else Path(cache_root)
    strategy = STRATEGIES[category]
    module = strategy_module(strategy)
    plans = module.candidates(len(visual.faces))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "collision_reference.npz"
    np.savez_compressed(source, vertices=visual.vertices, faces=visual.faces)
    attempts = []
    budget_phase = None
    for index, candidate in enumerate(plans):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            budget_phase = "candidate_start"
            attempts.extend(dict(candidate=c, passed=False, status="budget_not_run",
                                 elapsed_s=0.0) for c in plans[index:])
            break
        check()
        directory = out / "collision_candidates" / f"{index:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        frozen = identity(visual, surface, candidate, strategy)
        reused = gate._cache_hit(cache_root, frozen)
        request = dict(strategy=strategy, identity=frozen, candidate=candidate,
                       source=str(source.resolve()), source_sha256=prep.lib.sha256(source),
                       surface=surface, output_dir=str(directory.resolve()),
                       coacd_timeout_s=gate.WORKER_LIMIT_S)
        if strategy == "apple_fast_v1":
            request["coacd_options"] = module.options(candidate)
        if reused:
            cached_manifest = prep.lib.read_json(Path(reused["directory"]) / "manifest.json")
            request["reuse_parts_sha256"] = cached_manifest["descriptor"]["parts_sha256"]
            request.update(reuse_dir=reused["directory"], reuse_parts=reused["parts"],
                           reuse_inputs=reused["processed_input_sha256"],
                           reuse_generation_provenance=reused["generation_provenance"])
        path = directory / "request.json"
        prep.clip.write_json(path, request)
        print(f"{POLICY}: {category} candidate {index + 1}/{len(plans)} {candidate}", flush=True)
        candidate_started = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            budget_phase = "worker_start"
        result = (
            run_worker(path, directory, remaining)
            if remaining > 0 else dict(passed=False, status="budget_not_run", elapsed_s=0.0)
        )
        result.update(candidate=candidate, directory=str(directory), cache_hit=bool(reused))
        attempts.append(result)
        prep.clip.write_json(directory / "attempt.json", result)
        prep.clip.write_json(out / "collision_attempts.json", attempts)
        check()
        if not result.get("passed"):
            if reused:
                raise ValueError("cached collision no longer passes geometry qualification")
            continue
        if result.get("request") != frozen or not result.get("quality", {}).get("passed"):
            raise ValueError("collision worker qualification is not bound to its input")
        for name, digest in result["processed_input_sha256"].items():
            if prep.lib.sha256(directory / gate._safe_relative(name)) != digest:
                raise ValueError("processed collision input changed")
        if set(result.get("parts_sha256", {})) != set(result["parts"]):
            raise ValueError("worker did not freeze every qualified collision part")
        parts = []
        for name in result["parts"]:
            snapshot = (directory / gate._safe_relative(name)).read_bytes()
            if hashlib.sha256(snapshot).hexdigest() != result["parts_sha256"][name]:
                raise ValueError("qualified collision part changed")
            parts.append(trimesh.load(io.BytesIO(snapshot), file_type="obj",
                                      force="mesh", process=False))
        if not parts or len(parts) > candidate["max_convex_hull"] or any(
            not prep.closed_convex(part) for part in parts
        ):
            raise ValueError("invalid collision parts or part count")
        try:
            key = gate._publish_cache(cache_root, frozen, directory, result,
                                      deadline=deadline, check=check)
        except gate.AssetBudgetExpired as exc:
            budget_phase = exc.phase
            result.update(passed=False, status="asset_budget_exhausted", phase=exc.phase,
                          geometry_passed=True, worker_elapsed_s=result.get("elapsed_s"),
                          elapsed_s=time.monotonic() - candidate_started)
            prep.clip.write_json(directory / "attempt.json", result)
            break
        published = gate._cache_hit(cache_root, frozen)
        manifest = prep.lib.read_json(cache_root / "entries" / key / "manifest.json")
        if (not published or published["key"] != key
                or manifest["descriptor"]["parts_sha256"] != result["parts_sha256"]):
            raise ValueError("cache publication differs from qualified collision bytes")
        report = dict(method=strategy, policy=POLICY, strategy=strategy, cache_hit=bool(reused),
                      key=key, quality=result["quality"], parts=len(parts),
                      options=result.get("effective_options"),
                      configured_options=result.get("configured_options", candidate),
                      operation=result.get("operation"),
                      coacd_executed=result.get("coacd_executed", False),
                      generation_provenance=result.get("generation_provenance"),
                      selected_candidate=candidate, request_identity=frozen,
                      preprocessing=dict(asset_budget_s=gate.ASSET_BUDGET_S,
                                         coacd_limit_s=gate.WORKER_LIMIT_S,
                                         qualification_budget="remaining asset budget"),
                      attempts=attempts)
        prep.clip.write_json(out / "collision_success.json", report)
        return parts, report
    failure = dict(policy=POLICY, strategy=strategy, attempts=attempts,
                   budget_exhausted=budget_phase is not None, phase=budget_phase,
                   elapsed_s=time.monotonic() - started,
                   asset_budget_s=gate.ASSET_BUDGET_S, coacd_limit_s=gate.WORKER_LIMIT_S)
    prep.clip.write_json(out / "collision_attempts.json", attempts)
    prep.clip.write_json(out / "collision_failure.json", failure)
    raise ValueError("ASSET_PREPARATION_FAILED: authorized generation candidates exhausted")


def worker(request):
    if request.get("strategy") != request["identity"].get("strategy"):
        raise ValueError("collision strategy identity mismatch")
    if request.get("candidate") != request["identity"].get("candidate") or (
        request.get("surface") != request["identity"].get("surface")
    ):
        raise ValueError("collision candidate or support surface differs from its identity")
    if request.get("coacd_timeout_s") != gate.WORKER_LIMIT_S:
        raise ValueError("CoACD call deadline differs from the fixed 300 second limit")
    if request["strategy"] == "apple_fast_v1" and (
        request.get("coacd_options") != request["identity"].get("coacd_options")
    ):
        raise ValueError("CoACD parameters differ from their identity")
    module = strategy_module(request["strategy"])
    expected = request["identity"]["implementation_sha256"]
    for name, path in ((Path(__file__).name, __file__),
                       (Path(module.__file__).name, module.__file__),
                       (Path(gate.__file__).name, gate.__file__),
                       (Path(prep.__file__).name, prep.__file__)):
        if expected.get(name) != prep.lib.sha256(path):
            raise ValueError("collision implementation changed")
    if request.get("reuse_dir"):
        # Reuse the same complete visual/bottom gate without executing either generator.
        part_hashes = request["reuse_parts_sha256"]
        if set(part_hashes) != set(request["reuse_parts"]):
            raise ValueError("cached collision part fingerprint set mismatch")
        for name, digest in part_hashes.items():
            if prep.lib.sha256(Path(request["reuse_dir"]) / gate._safe_relative(name)) != digest:
                raise ValueError("cached collision part changed before qualification")
        copied = copy.deepcopy(request)
        copied["candidate"].setdefault("threshold", prep.COACD["threshold"])
        copied["candidate"].setdefault("resolution", prep.COACD["resolution"])
        result = gate.worker(copied)
        result["configured_options"] = request.get("coacd_options", request["candidate"])
        result["parts_sha256"] = {
            name: part_hashes[request["reuse_parts"][index]]
            for index, name in enumerate(result["parts"])
        }
        for name, digest in result["parts_sha256"].items():
            if prep.lib.sha256(Path(request["output_dir"]) / gate._safe_relative(name)) != digest:
                raise ValueError("cached collision part changed during qualification")
        return result
    return module.worker(request)


def main():
    import resource
    import traceback

    resource.setrlimit(resource.RLIMIT_AS, (gate.WORKER_MEMORY_BYTES, gate.WORKER_MEMORY_BYTES))
    request = prep.lib.read_json(sys.argv[1])
    try:
        result = worker(request)
    except Exception as exc:
        result = dict(passed=False, status="worker_error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    prep.clip.write_json(Path(request["output_dir"]) / "result.json", result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Finite, topology-protected collision candidates; each worker has one total deadline.

The original visible mesh is always the quality reference. Successful cache entries
include the exact processed inputs; failed candidates remain in the task directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep

POLICY = "collision_repair_v2"
ASSET_BUDGET_S = 1200.0
WORKER_LIMIT_S = 300.0
WORKER_MEMORY_BYTES = 8 * 1024**3
WORKER_OMP_THREADS = 8
WORKER_BLAS_THREADS = 1
MAX_COMPONENTS = 12
CACHE = prep.CACHE.parent / "repair_collision_cache_v2"
QUALITY_VERSION = "complete_visual_external_boundary_support_bottom_1mm_v2"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def mesh_hash(mesh):
    h = hashlib.sha256()
    for values, dtype in [(mesh.vertices, "<f8"), (mesh.faces, "<i8")]:
        array = np.ascontiguousarray(values, dtype=dtype)
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def topology_report(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    finite = bool(np.isfinite(vertices).all())
    report = dict(vertices=len(vertices), faces=len(faces), finite=finite, passed=False)
    if not finite or not len(faces) or not len(vertices):
        return report
    if faces.min() < 0 or faces.max() >= len(vertices):
        return dict(report, invalid_indices=True)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    d = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    degenerate = int(np.count_nonzero(mesh.area_faces <= max(d * d * 1e-14, 1e-24)))
    duplicates = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    volume = float(mesh.volume)
    winding = bool(mesh.is_winding_consistent)
    report.update(
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        nonmanifold_edge_face_counts=counts[counts > 2].tolist(),
        degenerate_faces=degenerate,
        duplicate_faces=duplicates,
        winding_consistent=winding,
        volume_m3=volume,
        mesh_sha256=mesh_hash(mesh),
        passed=bool(
            not boundary
            and not nonmanifold
            and not degenerate
            and not duplicates
            and winding
            and np.isfinite(volume)
            and volume > 0
        ),
    )
    return report


def remove_opposed_duplicate_faces(mesh):
    """Remove only two exactly coincident, oppositely wound generated triangles."""
    faces = np.asarray(mesh.faces)
    _, groups, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    removed = []
    pairs = []
    for group in np.flatnonzero(counts == 2):
        indices = np.flatnonzero(groups == group)
        left, right = faces[indices[0]], faces[indices[1]]
        if len(set(left.tolist())) != 3:
            continue
        if any(np.array_equal(left, np.roll(right[::-1], i)) for i in range(3)):
            removed.extend(indices.tolist())
            pairs.append(dict(face_indices=indices.tolist(), vertex_indices=left.tolist()))
    cleaned = mesh.copy()
    keep = np.ones(len(faces), dtype=bool)
    keep[removed] = False
    cleaned.update_faces(keep)
    cleaned.remove_unreferenced_vertices()
    before, after = float(mesh.volume), float(cleaned.volume)
    unchanged = abs(before - after) <= max(1e-14, abs(before) * 1e-10)
    report = dict(
        removed_face_indices=sorted(removed),
        opposed_pairs=pairs,
        raw_mesh_sha256=mesh_hash(mesh),
        cleaned_mesh_sha256=mesh_hash(cleaned),
        volume_before_m3=before,
        volume_after_m3=after,
        volume_preserved=unchanged,
        applied=bool(removed),
    )
    if not unchanged:
        raise ValueError("opposed face cleanup changed enclosed volume")
    return cleaned, report


def input_surface_errors(original, prepared):
    measurements = {}
    for name, target, source in [
        ("original_to_prepared_m", prepared, original),
        ("prepared_to_original_m", original, prepared),
    ]:
        points = prep.sample_surface(source)
        maximum = 0.0
        for block in np.array_split(points, max(1, int(np.ceil(len(points) / 128)))):
            maximum = max(maximum, float(trimesh.proximity.closest_point(target, block)[1].max()))
        measurements[name] = maximum
    return dict(
        measurements,
        reference="original cleaned connected component",
        sampling="vertices and face interiors, at most 15000 per direction",
        final_proxy_reference="complete original visible mesh, unchanged",
    )


def prepare_input(component, face_count, *, raw_path=None):
    before = topology_report(component)
    prepared = component.copy()
    simplified = before["passed"] and face_count is not None and len(component.faces) > face_count
    if simplified:
        prepared = component.simplify_quadric_decimation(face_count=face_count)
    raw_topology = topology_report(prepared)
    if raw_path is not None:
        np.savez_compressed(raw_path, vertices=prepared.vertices, faces=prepared.faces)
    cleanup = dict(applied=False, removed_face_indices=[])
    if simplified:
        prepared, cleanup = remove_opposed_duplicate_faces(prepared)
    after = topology_report(prepared)
    report = dict(
        before=before,
        raw=raw_topology,
        after=after,
        cleanup=cleanup,
        requested_faces=face_count,
        passed=before["passed"] and after["passed"],
    )
    if cleanup["applied"] and report["passed"]:
        report["input_surface_errors"] = input_surface_errors(component, prepared)
    return prepared, report


def candidates(category, original_faces):
    if category == "apple":
        rows = [
            (8000, 64, 0.005, 2000),
            (20000, 64, 0.005, 2000),
            (None, 64, 0.005, 2000),
            (None, 128, 0.0025, 4000),
        ]
    elif category == "bowl":
        rows = [
            (None, 64, 0.005, 2000),
            (None, 128, 0.005, 2000),
            (None, 128, 0.0025, 2000),
            (None, 128, 0.0025, 4000),
        ]
    else:
        rows = [
            (2000, 64, 0.005, 2000),
            (8000, 64, 0.005, 2000),
            (None, 128, 0.005, 2000),
            (None, 128, 0.0025, 4000),
        ]
    return [
        dict(
            id=f"{category}_{i}",
            face_count=n,
            max_convex_hull=h,
            threshold=t,
            resolution=r,
            original_faces=original_faces,
        )
        for i, (n, h, t, r) in enumerate(rows)
    ]


def identity(visual, surface, candidate):
    return dict(
        policy=POLICY,
        topology_version="closed_edges_positive_volume_opposed_pairs_v2",
        quality_version=QUALITY_VERSION,
        visible_mesh_sha256=mesh_hash(visual),
        surface=surface,
        candidate=candidate,
        coacd_defaults=prep.COACD,
        versions={
            n: version(n) for n in ("coacd", "trimesh", "fast-simplification", "numpy", "scipy")
        },
        genesis_commit=prep.official.GENESIS_COMMIT,
        implementation_sha256={
            Path(p).name: prep.lib.sha256(Path(p)) for p in (__file__, prep.__file__)
        },
        max_components=MAX_COMPONENTS,
        worker_memory_bytes=WORKER_MEMORY_BYTES,
        worker_limit_s=WORKER_LIMIT_S,
        asset_budget_s=ASSET_BUDGET_S,
        thread_policy=dict(openmp=WORKER_OMP_THREADS, openblas=WORKER_BLAS_THREADS,
                           mkl=WORKER_BLAS_THREADS),
    )


def _safe_relative(value):
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("cache integrity: invalid relative path")
    return p


def _cache_hit(cache_root, request_identity):
    index = cache_root / "requests" / f"{digest(request_identity)}.json"
    if not index.exists():
        return None
    key = prep.lib.read_json(index)["key"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("cache integrity: invalid key")
    entry = cache_root / "entries" / key
    manifest = prep.lib.read_json(entry / "manifest.json")
    if (
        digest(manifest["descriptor"]) != key
        or manifest["descriptor"]["request"] != request_identity
    ):
        raise ValueError("cache identity mismatch")
    for record in manifest["files"]:
        _safe_relative(record["path"])
    prep.official.verify_files(entry, manifest["files"])
    for name, expected in manifest["descriptor"]["processed_input_sha256"].items():
        if prep.lib.sha256(entry / _safe_relative(name)) != expected:
            raise ValueError("cache processed input hash mismatch")
    for name in manifest["parts"]:
        path = entry / _safe_relative(name)
        if prep.lib.sha256(path) != manifest["descriptor"]["parts_sha256"][name]:
            raise ValueError("cache collision part hash mismatch")
    return dict(
        directory=str(entry),
        parts=manifest["parts"],
        key=key,
        processed_input_sha256=manifest["descriptor"]["processed_input_sha256"],
        generation_provenance=manifest["descriptor"].get("generation_provenance", {
            "status": "unknown", "reason": "cache has no recorded generation parameters"
        }),
    )


class AssetBudgetExpired(ValueError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__(f"ASSET_PREPARATION_FAILED: asset deadline exhausted during {phase}")


def _publish_cache(cache_root, request_identity, directory, result, *, deadline, check):
    import fcntl

    def require_time(phase):
        if time.monotonic() >= deadline:
            raise AssetBudgetExpired(phase)
        try:
            check()
        finally:
            if time.monotonic() >= deadline:
                raise AssetBudgetExpired(phase)

    require_time("cache_publish_hash")
    descriptor = dict(
        request=request_identity,
        processed_input_sha256=result["processed_input_sha256"],
        generation_provenance=result.get("generation_provenance", {
            "status": "unknown", "reason": "worker has no generation provenance"
        }),
        parts_sha256={n: prep.lib.sha256(directory / _safe_relative(n)) for n in result["parts"]},
    )
    key = digest(descriptor)
    entries, requests = cache_root / "entries", cache_root / "requests"
    entries.mkdir(parents=True, exist_ok=True)
    requests.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f".{digest(request_identity)}.lock"
    with lock_path.open("a") as lock:
        while True:
            require_time("cache_publish_lock")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
        require_time("cache_publish_copy")
        target = entries / key
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=entries))
            try:
                names = sorted(set(result["parts"]) | set(result["processed_input_sha256"]))
                for name in names:
                    relative = _safe_relative(name)
                    dst = temporary / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(directory / relative, dst)
                    require_time("cache_publish_copy")
                prep.clip.write_json(
                    temporary / "manifest.json",
                    dict(
                        descriptor=descriptor,
                        parts=result["parts"],
                        quality=result["quality"],
                        files=[prep.official.fingerprint(temporary / n, temporary) for n in names],
                    ),
                )
                require_time("cache_publish_manifest")
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        pointer = requests / f"{digest(request_identity)}.json"
        temporary_pointer = pointer.with_suffix(f".{os.getpid()}.tmp")
        try:
            prep.clip.write_json(temporary_pointer, dict(key=key))
            require_time("cache_publish_pointer")
            temporary_pointer.replace(pointer)
        finally:
            temporary_pointer.unlink(missing_ok=True)
    return key


def _legacy_cache(visual):
    """Import an already qualified v1 derivative without rerunning its preparation."""
    key = hashlib.sha256(
        visual.vertices.tobytes()
        + visual.faces.tobytes()
        + json.dumps(prep.COACD, sort_keys=True).encode()
        + prep.COACD_VERSION.encode()
        + b"connected_components_v3"
        + json.dumps(prep.COACD_PREPARATION, sort_keys=True).encode()
    ).hexdigest()
    root = prep.CACHE / key
    if not (root / "manifest.json").exists():
        return None
    manifest = prep.lib.read_json(root / "manifest.json")
    for r in manifest["files"]:
        _safe_relative(r["path"])
    prep.official.verify_files(root, manifest["files"])
    return dict(
        directory=str(root), parts=[r["path"] for r in manifest["files"]], key=key,
        generation_provenance={
            "status": "unknown", "reason": "legacy cache records files only",
            "cache_key": key,
        },
    )


def run_worker(request_path, directory, timeout_s):
    """Timeout covers every component plus simplification and the complete quality gate."""
    env = dict(os.environ, OMP_NUM_THREADS=str(WORKER_OMP_THREADS),
               OPENBLAS_NUM_THREADS=str(WORKER_BLAS_THREADS),
               MKL_NUM_THREADS=str(WORKER_BLAS_THREADS))
    started = time.perf_counter()
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(request_path)],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return dict(
                passed=False,
                status="timeout",
                phase="worker",
                timeout_s=timeout_s,
                elapsed_s=time.perf_counter() - started,
                exit_code=process.returncode,
            )
    path = directory / "result.json"
    result = (
        prep.lib.read_json(path) if path.exists() else dict(passed=False, status="worker_error")
    )
    if code:
        result.update(passed=False, exit_code=code)
    result.update(elapsed_s=time.perf_counter() - started)
    return result


def decompose(
    visual, surface, out, *, category, check=lambda: None, cache_root=None, deadline=None
):
    started = time.monotonic()
    if version("coacd") != prep.COACD_VERSION:
        raise ValueError("CoACD version mismatch")
    cache_root = CACHE if cache_root is None else Path(cache_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "collision_reference.npz"
    np.savez_compressed(source, vertices=visual.vertices, faces=visual.faces)
    if deadline is None:
        deadline = time.monotonic() + ASSET_BUDGET_S
    attempts = []
    plans = candidates(category, len(visual.faces))
    # The existing cup derivative is already qualified; retain its exact collision mesh.
    legacy = _legacy_cache(visual) if category not in {"apple", "bowl"} else None
    if legacy:
        plans = [dict(plans[0], id="verified_legacy_cache", legacy_key=legacy["key"])] + plans
    budget_exhausted = False
    budget_phase = None
    for i, candidate in enumerate(plans):
        candidate_started = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            budget_exhausted = True
            budget_phase = "candidate_start"
            attempts.extend(
                dict(candidate=c, passed=False, status="budget_not_run", elapsed_s=0.0,
                     phase=budget_phase) for c in plans[i:]
            )
            break
        check()
        directory = out / "collision_candidates" / f"{i:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        request_identity = identity(visual, surface, candidate)
        hit = _cache_hit(cache_root, request_identity)
        reused = hit or (legacy if candidate["id"] == "verified_legacy_cache" else None)
        request = dict(
            identity=request_identity,
            candidate=candidate,
            source=str(source.resolve()),
            source_sha256=prep.lib.sha256(source),
            surface=surface,
            output_dir=str(directory.resolve()),
        )
        if reused:
            request.update(reuse_dir=reused["directory"], reuse_parts=reused["parts"])
            request["reuse_inputs"] = reused.get("processed_input_sha256", {})
            request["reuse_generation_provenance"] = reused["generation_provenance"]
        request_path = directory / "request.json"
        prep.clip.write_json(request_path, request)
        print(f"{POLICY}: {category} candidate {i + 1}/{len(plans)} {candidate}", flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result = dict(passed=False, status="budget_not_run", elapsed_s=0.0,
                          phase="worker_start")
            budget_exhausted = True
            budget_phase = "worker_start"
        else:
            result = run_worker(request_path, directory, min(WORKER_LIMIT_S, remaining))
        result.update(candidate=candidate, directory=str(directory), cache_hit=bool(reused))
        prep.clip.write_json(directory / "attempt.json", result)
        attempts.append(result)
        prep.clip.write_json(out / "collision_attempts.json", attempts)
        check()
        if not result.get("passed"):
            continue
        if result.get("request") != request_identity or not result.get("quality", {}).get("passed"):
            raise ValueError("collision worker evidence mismatch")
        for name, expected in result["processed_input_sha256"].items():
            if prep.lib.sha256(directory / _safe_relative(name)) != expected:
                raise ValueError("collision worker processed input hash mismatch")
        parts = [
            trimesh.load(directory / _safe_relative(n), force="mesh", process=False)
            for n in result["parts"]
        ]
        if (
            not parts
            or len(parts) > candidate["max_convex_hull"]
            or any(not prep.closed_convex(p) for p in parts)
        ):
            raise ValueError("invalid accepted collision parts")
        try:
            key = _publish_cache(
                cache_root, request_identity, directory, result, deadline=deadline, check=check
            )
        except AssetBudgetExpired as exc:
            budget_exhausted = True
            budget_phase = exc.phase
            result.update(
                passed=False, status="asset_budget_exhausted", phase=exc.phase,
                geometry_passed=True, worker_elapsed_s=result.get("elapsed_s"),
                elapsed_s=time.monotonic() - candidate_started,
            )
            prep.clip.write_json(directory / "attempt.json", result)
            break
        report = dict(
            method="coacd",
            policy=POLICY,
            cache_hit=bool(reused),
            key=key,
            quality=result["quality"],
            options=result.get("effective_options", candidate),
            operation=result.get("operation"),
            configured_options=result.get("configured_options"),
            coacd_executed=result.get("coacd_executed"),
            generation_provenance=result.get("generation_provenance"),
            version=prep.COACD_VERSION,
            parts=len(parts),
            selected_candidate=candidate,
            preprocessing=dict(
                policy=POLICY,
                max_components=MAX_COMPONENTS,
                asset_budget_s=ASSET_BUDGET_S,
                worker_limit_s=WORKER_LIMIT_S,
            ),
            attempts=attempts,
        )
        prep.clip.write_json(out / "collision_success.json", report)
        return parts, report
    failure = dict(
        policy=POLICY,
        attempts=attempts,
        budget_exhausted=budget_exhausted,
        asset_budget_s=ASSET_BUDGET_S,
        worker_limit_s=WORKER_LIMIT_S,
        phase=budget_phase,
        elapsed_s=time.monotonic() - started,
    )
    prep.clip.write_json(out / "collision_attempts.json", attempts)
    prep.clip.write_json(out / "collision_failure.json", failure)
    raise ValueError("ASSET_PREPARATION_FAILED: bounded collision candidates exhausted")


def quality_gate(visual, parts, surface=None, **flags):
    """Retain the complete old gate and additionally bound the support bottom at 1 mm."""
    original = prep.proxy_quality(visual, parts, surface, **flags)
    bottom = original.get("bottom_alignment_error_m")
    bottom_passed = bottom is not None and np.isfinite(bottom) and bottom <= 0.001
    return dict(
        original,
        passed=bool(original["passed"] and bottom_passed),
        original_geometry_check=original,
        bottom_support_check=dict(passed=bool(bottom_passed), error_m=bottom, limit_m=0.001),
    )


def native_quality(
    visual,
    parts,
    surface,
    out,
    *,
    deadline,
    check=lambda: None,
    fixed_native=False,
    native_semantics=False,
):
    """Original collision preference uses the same bounded worker and asset deadline."""
    directory = Path(out) / "native_quality_worker"
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=False)
    check()
    source = directory / "original_visual.npz"
    np.savez_compressed(source, vertices=visual.vertices, faces=visual.faces)
    names = []
    for i, part in enumerate(parts):
        path = directory / f"native_{i:03d}.npz"
        np.savez_compressed(path, vertices=part.vertices, faces=part.faces)
        names.append(path.name)
    request = dict(
        operation="native_quality",
        source=str(source.resolve()),
        source_sha256=prep.lib.sha256(source),
        surface=surface,
        output_dir=str(directory.resolve()),
        native_parts=names,
        native_part_sha256={n: prep.lib.sha256(directory / n) for n in names},
        fixed_native=fixed_native,
        native_semantics=native_semantics,
        identity=dict(visible_mesh_sha256=mesh_hash(visual)),
    )
    path = directory / "request.json"
    prep.clip.write_json(path, request)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        result = dict(passed=False, status="asset_budget_exhausted", phase="native_worker_start",
                      elapsed_s=time.monotonic() - started)
    else:
        result = run_worker(path, directory, min(WORKER_LIMIT_S, remaining))
    prep.clip.write_json(directory / "attempt.json", result)
    check()
    if not result.get("passed"):
        raise ValueError(f"ASSET_PREPARATION_FAILED: native quality {result.get('status')}")
    if result.get("request") != request["identity"]:
        raise ValueError("native quality worker identity mismatch")
    return result["quality"]


def worker(request):
    """One subprocess owns a candidate; it never initializes or steps Genesis."""
    import coacd

    np.random.seed(0)
    directory = Path(request["output_dir"])
    source = Path(request["source"])
    if prep.lib.sha256(source) != request["source_sha256"]:
        raise ValueError("original visual hash mismatch")
    with np.load(source) as data:
        visual = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    if mesh_hash(visual) != request["identity"]["visible_mesh_sha256"]:
        raise ValueError("original visual identity mismatch")
    if request.get("operation") == "native_quality":
        parts = []
        for name in request["native_parts"]:
            path = directory / _safe_relative(name)
            if prep.lib.sha256(path) != request["native_part_sha256"][name]:
                raise ValueError("native part hash mismatch")
            with np.load(path) as data:
                parts.append(trimesh.Trimesh(data["vertices"], data["faces"], process=False))
        quality = quality_gate(
            visual,
            parts,
            request["surface"],
            fixed_native=request["fixed_native"],
            native_semantics=request["native_semantics"],
        )
        return dict(passed=True, status="completed", quality=quality, request=request["identity"])
    inputs = directory / "inputs"
    inputs.mkdir()
    candidate = request["candidate"]
    options = dict(prep.COACD, preprocess_mode="off", real_metric=False)
    for key in ("max_convex_hull", "threshold", "resolution"):
        options[key] = candidate[key]
    parts, input_hashes, executed_calls = [], {}, []
    part_dir = directory / "parts"
    part_dir.mkdir()

    def preserve_parts():
        for part_index, part in enumerate(parts):
            part.export(part_dir / f"part_{part_index:03d}.obj")

    if request.get("reuse_dir"):
        parts = [
            trimesh.load(
                Path(request["reuse_dir"]) / _safe_relative(n), force="mesh", process=False
            )
            for n in request["reuse_parts"]
        ]
        for name, expected in request.get("reuse_inputs", {}).items():
            relative = _safe_relative(name)
            saved = Path(request["reuse_dir"]) / relative
            if prep.lib.sha256(saved) != expected:
                raise ValueError("reused processed input hash mismatch")
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, target)
            input_hashes[name] = expected
        if not request.get("reuse_inputs"):
            prep.clip.write_json(
                inputs / "legacy_provenance.json",
                dict(
                    source_cache=request["reuse_dir"],
                    note="Qualified legacy parts reused; no new CoACD input claimed",
                ),
            )
    else:
        if not np.isfinite(visual.vertices).all():
            raise ValueError("nonfinite original visual geometry")
        components = trimesh.Trimesh(visual.vertices, visual.faces, process=True).split(
            only_watertight=False
        )
        solids = []
        for i, component in enumerate(components):
            singular = np.linalg.svd(
                component.vertices - component.vertices.mean(0), compute_uv=False
            )
            if abs(component.volume) <= 1e-12 and singular[-1] <= 1e-7:
                prep.clip.write_json(
                    inputs / f"skipped_{i:03d}.json",
                    dict(
                        reason="zero-volume coplanar patch; complete visual gate retained",
                        volume_m3=float(component.volume),
                        singular_values_m=singular.tolist(),
                    ),
                )
            else:
                solids.append(component)
        if not solids or len(solids) > MAX_COMPONENTS:
            raise ValueError(
                f"component limit: {len(solids)} nonzero components, maximum {MAX_COMPONENTS}"
            )
        coacd.set_log_level("info")
        for i, component in enumerate(solids):
            raw_path = inputs / f"component_{i:03d}_raw.npz"
            prepared, topology = prepare_input(
                component, candidate["face_count"], raw_path=raw_path
            )
            input_hashes[raw_path.relative_to(directory).as_posix()] = prep.lib.sha256(raw_path)
            path = inputs / f"component_{i:03d}.npz"
            np.savez_compressed(path, vertices=prepared.vertices, faces=prepared.faces)
            name = path.relative_to(directory).as_posix()
            input_hashes[name] = prep.lib.sha256(path)
            prep.clip.write_json(inputs / f"component_{i:03d}.json", topology)
            if not topology["passed"]:
                return dict(
                    passed=False,
                    status="topology_rejected",
                    component=i,
                    topology=topology,
                    processed_input_sha256=input_hashes,
                )
            if prepared.is_convex:
                parts.append(prepared.convex_hull)
                preserve_parts()
                continue
            effective = dict(
                options,
                max_convex_hull=candidate["max_convex_hull"] - len(parts) - (len(solids) - i - 1),
            )
            if effective["max_convex_hull"] < 1:
                raise ValueError("asset convex part budget exhausted")
            prep.clip.write_json(inputs / f"options_{i:03d}.json", effective)
            executed_calls.append(dict(
                component=i, options=effective, processed_input=name,
                processed_input_sha256=input_hashes[name],
            ))
            result = coacd.run_coacd(coacd.Mesh(prepared.vertices, prepared.faces), **effective)
            parts.extend(trimesh.Trimesh(v, f, process=True).convex_hull for v, f in result)
            preserve_parts()
    names = []
    for i, part in enumerate(parts):
        path = part_dir / f"part_{i:03d}.obj"
        if request.get("reuse_dir"):
            source_part = Path(request["reuse_dir"]) / _safe_relative(request["reuse_parts"][i])
            shutil.copy2(source_part, path)
        else:
            part.export(path)
        names.append(path.relative_to(directory).as_posix())
    if not parts or len(parts) > candidate["max_convex_hull"]:
        raise ValueError("empty or excessive collision parts")
    quality = quality_gate(visual, parts, request["surface"])
    reused = bool(request.get("reuse_dir"))
    generation = (
        request.get("reuse_generation_provenance", {
            "status": "unknown", "reason": "cache has no recorded generation parameters"
        })
        if reused else dict(
            status="recorded", request_identity_sha256=digest(request["identity"]),
            coacd_calls=executed_calls,
        )
    )
    return dict(
        passed=bool(quality["passed"]),
        status="passed" if quality["passed"] else "quality_rejected",
        parts=names,
        quality=quality,
        processed_input_sha256=input_hashes,
        operation="cached_quality_recheck" if reused else "generated_candidate",
        configured_options=options,
        coacd_executed=bool(executed_calls),
        effective_options=executed_calls[0]["options"] if len(executed_calls) == 1 else None,
        generation_provenance=generation,
        request=request["identity"],
    )


def main():
    import resource
    import traceback

    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    request = prep.lib.read_json(Path(sys.argv[1]))
    out = Path(request["output_dir"])
    try:
        result = worker(request)
    except Exception as exc:
        result = dict(passed=False, status="worker_error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    prep.clip.write_json(out / "result.json", result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

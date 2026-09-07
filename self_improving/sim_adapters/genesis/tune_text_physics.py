"""Audited numerical comparisons on frozen text-scene poses, without asset retrieval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis.storage_paths import local_path

SEEDS = (0, 42, 87)
SCHEMA = "genenv.text_numerics_comparison.v1"


def verify_assets(data):
    for asset in data["assets"].values():
        for root_key, files_key in (("source_root", "source_files"),
                                    ("derived_root", "derived_files")):
            official.verify_files(local_path(asset[root_key]), asset[files_key])


def implementation_files():
    root = Path(__file__).resolve().parent
    names = (
        "tune_text_physics.py", "repair_numerics.py", "repair_physics.py", "repair_assets.py",
        "repair_geometry.py", "physics_math.py", "validate_asset_scene.py", "asset_physics.py",
    )
    return {str(root / name): library.sha256(root / name) for name in names}


def run_trial(source_input, output_dir, numerics_profile):
    """Run one new trajectory and independently recompute its frozen verdict."""
    source_input, out = Path(source_input).resolve(), Path(output_dir).resolve()
    source_hash = library.sha256(source_input)
    original = library.read_json(source_input)
    if original["settings"] != physics.settings(original["profile"]):
        raise ValueError("comparison source must use an unmodified legacy physics profile")
    if set(original["assets"]) != {"table_1", "cup_1"}:
        raise ValueError("numerical comparison requires the frozen table/cup isolation scene")
    verify_assets(original)
    data = physics.frozen_input(
        original["assets"], original["poses"], original["relations"], original["random_seed"],
        profile="text_repair_v1", numerics_profile=numerics_profile,
    )
    out.mkdir(parents=True, exist_ok=False)
    path = out / "physics_input.json"
    official.write_json(path, data)
    digest = library.sha256(path)
    implementation = implementation_files()
    official.write_json(out / "source_manifest.json", dict(
        source_input=str(source_input), source_input_sha256=source_hash,
        implementation_sha256=implementation,
        purpose="frozen table/cup diagnostic; not complete four-asset acceptance",
    ))

    def check():
        if library.sha256(source_input) != source_hash or library.sha256(path) != digest:
            raise ValueError("comparison input changed")
        for name, expected in implementation.items():
            if library.sha256(name) != expected:
                raise ValueError("comparison implementation changed during the run")
        verify_assets(data)

    report = dict(
        schema_version=SCHEMA, status="execution_failed", exit_code=1,
        numerics_profile=numerics_profile, acceptance_profile="text_repair_v1",
        simulation_executed=False, steps_executed=0, input_sha256=digest,
    )
    started = time.monotonic()
    try:
        result = physics.simulate(data, out, check)
        rows = [json.loads(line) for line in (out / "trace.jsonl").open()]
        if physics.evaluate(data, rows, initial_rejected=not result["complete"]) != result:
            raise ValueError("persisted trajectory verdict differs")
        check()
        official.write_json(out / "validation_result.json", result)
        report.update(
            status="physics_passed" if result["passed"] else "physics_failed",
            exit_code=0 if result["passed"] else 2, result=result,
            simulation_executed=result["simulation_executed"],
            steps_executed=result["steps_executed"],
            trace_sha256=library.sha256(out / "trace.jsonl"),
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        loaded = out / "asset_physics_report.json"
        if loaded.exists():
            actual = library.read_json(loaded)
            for key in ("simulation_executed", "steps_executed"):
                report[key] = actual.get(key, report[key])
    finally:
        report["duration_s"] = time.monotonic() - started
        report["artifacts"] = [official.fingerprint(p, out) for p in sorted(out.rglob("*"))
                               if p.is_file() and p.name != "trial_result.json"]
        official.write_json(out / "trial_result.json", report)
    return report


def verify_trial(directory, *, source_hash=None, numerics_profile=None,
                 expected_implementation=None):
    directory = Path(directory)
    report = library.read_json(directory / "trial_result.json")
    official.verify_files(directory, report["artifacts"])
    source = library.read_json(directory / "source_manifest.json")
    if expected_implementation is not None and (
        source["implementation_sha256"] != expected_implementation
    ):
        raise ValueError("trial implementation differs from frozen comparison")
    source_path = Path(source["source_input"])
    if library.sha256(source_path) != source["source_input_sha256"]:
        raise ValueError("trial source input changed")
    if source_hash is not None and source["source_input_sha256"] != source_hash:
        raise ValueError("trial belongs to a different frozen source")
    if numerics_profile is not None and report["numerics_profile"] != numerics_profile:
        raise ValueError("trial belongs to a different numerical profile")
    if report["exit_code"] not in (0, 2):
        raise ValueError(f"comparison execution error: {report.get('error')}")
    data = library.read_json(directory / "physics_input.json")
    if data.get("numerics_profile", "legacy") != report["numerics_profile"]:
        raise ValueError("trial numerical profile differs from its input")
    if data["profile"] != "text_repair_v1":
        raise ValueError("comparison requires the unchanged 95% acceptance profile")
    original = library.read_json(source_path)
    if original["settings"] != physics.settings(original["profile"]):
        raise ValueError("comparison source must use an unmodified legacy physics profile")
    expected_data = physics.frozen_input(
        original["assets"], original["poses"], original["relations"], original["random_seed"],
        profile="text_repair_v1", numerics_profile=report["numerics_profile"],
    )
    if data != expected_data:
        raise ValueError("trial changes the frozen source assets or initial state")
    verify_assets(data)
    if library.sha256(directory / "physics_input.json") != report["input_sha256"]:
        raise ValueError("trial input fingerprint differs")
    rows = [json.loads(line) for line in (directory / "trace.jsonl").open()]
    result = physics.evaluate(data, rows, initial_rejected=not report["result"]["complete"])
    if result != report["result"]:
        raise ValueError("trial result is not supported by its trajectory")
    expected = dict(
        exit_code=0 if result["passed"] else 2,
        status="physics_passed" if result["passed"] else "physics_failed",
        steps_executed=result["steps_executed"],
        simulation_executed=result["simulation_executed"],
    )
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("trial summary contradicts the independently evaluated trajectory")
    return report


def choose_candidate(matrix, sensitivity, ordered_profiles):
    """A candidate is eligible only after all base and half-step seeds pass."""
    for profile in ordered_profiles:
        base, half = matrix.get(profile, {}), sensitivity.get(profile, {})
        if all(base.get(str(s), {}).get("passed") is True for s in SEEDS) and all(
            half.get(str(s), {}).get("passed") is True for s in SEEDS
        ):
            return profile
    return None


def run_matrix(source_root, output_dir, *, resume=False):
    from self_improving.sim_adapters.genesis import repair_numerics as numerics

    source_root, out = Path(source_root).resolve(), Path(output_dir).resolve()
    sources = {str(seed): source_root / f"seed_{seed}" / "03_physics/physics_input.json"
               for seed in SEEDS}
    source_hashes = {seed: library.sha256(path) for seed, path in sources.items()}
    for seed, path in sources.items():
        if library.read_json(path)["random_seed"] != int(seed):
            raise ValueError("source input random_seed differs from declared comparison seed")
    if out.is_relative_to(source_root) or source_root.is_relative_to(out):
        raise ValueError("comparison and source directories must be separate")
    frozen = dict(schema_version=SCHEMA, source_root=str(source_root),
                  source_input_sha256=source_hashes, seeds=list(SEEDS),
                  ordered_profiles=list(numerics.CANDIDATE_PROFILES),
                  acceptance_profile="text_repair_v1",
                  implementation_sha256=implementation_files())
    if resume:
        if library.read_json(out / "comparison_input.json") != frozen:
            raise ValueError("resume inputs differ from the original comparison")
    else:
        out.mkdir(parents=True, exist_ok=False)
        official.write_json(out / "comparison_input.json", frozen)
    summary = dict(frozen, status="running", matrix={}, sensitivity={}, selected_profile=None)
    official.write_json(out / "comparison_result.json", summary)

    def execute(profile, seed):
        target = out / "trials" / profile / f"seed_{seed}"
        if not (target / "trial_result.json").exists():
            if target.exists():
                raise ValueError(
                    f"interrupted trial retained at {target}; use a new output directory"
                )
            log_path = out / "logs" / f"{profile}_seed_{seed}.log"
            log_path.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-m",
                       "self_improving.sim_adapters.genesis.tune_text_physics",
                       "trial", "--source-input", str(sources[str(seed)]),
                       "--output-dir", str(target), "--numerics-profile", profile]
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            with log_path.open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, check=False)
            if completed.returncode not in (0, 2):
                raise RuntimeError(f"trial execution failed; see {log_path}")
        result = verify_trial(target, source_hash=source_hashes[str(seed)],
                              numerics_profile=profile,
                              expected_implementation=frozen["implementation_sha256"])
        record = dict(passed=result["result"]["passed"], directory=str(target),
                      result_sha256=library.sha256(target / "trial_result.json"),
                      objects=result["result"]["objects"],
                      failures=result["result"]["failures"])
        print(
            json.dumps(dict(profile=profile, seed=seed, **record), ensure_ascii=False), flush=True
        )
        return record

    try:
        for profile in numerics.CANDIDATE_PROFILES:
            summary["matrix"][profile] = {}
            for seed in SEEDS:
                summary["matrix"][profile][str(seed)] = execute(profile, seed)
                official.write_json(out / "comparison_result.json", summary)
        for profile in numerics.CANDIDATE_PROFILES:
            if not all(r["passed"] for r in summary["matrix"][profile].values()):
                continue
            half = numerics.half_dt_profile(profile)
            summary["sensitivity"][profile] = {}
            for seed in SEEDS:
                summary["sensitivity"][profile][str(seed)] = execute(half, seed)
                official.write_json(out / "comparison_result.json", summary)
            chosen = choose_candidate(summary["matrix"], summary["sensitivity"],
                                      numerics.CANDIDATE_PROFILES)
            if chosen:
                summary.update(status="passed", selected_profile=chosen,
                               half_dt_profile=numerics.half_dt_profile(chosen))
                break
        if summary["selected_profile"] is None:
            from self_improving.sim_adapters.genesis import text_plane_control

            summary.update(status="physics_failed", analytic_plane_control={})
            official.write_json(out / "comparison_result.json", summary)
            for profile in numerics.CANDIDATE_PROFILES:
                destination = out / "analytic_plane_control" / profile
                if destination.exists():
                    report = library.read_json(destination / "diagnostic_result.json")
                    official.verify_files(destination, report["artifacts"])
                else:
                    report = text_plane_control.run(
                        sources["0"], destination, numerics_profile=profile
                    )
                summary["analytic_plane_control"][profile] = dict(
                    directory=str(destination), report=report,
                    report_sha256=library.sha256(destination / "diagnostic_result.json"),
                )
                official.write_json(out / "comparison_result.json", summary)
    except Exception as exc:
        summary.update(status="execution_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if implementation_files() != frozen["implementation_sha256"]:
            summary.update(status="execution_failed", selected_profile=None,
                           error="comparison implementation changed")
        for seed, path in sources.items():
            if library.sha256(path) != source_hashes[seed]:
                summary.update(status="execution_failed", error="source input changed")
        official.write_json(out / "comparison_result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--source-input", type=Path, required=True)
    trial.add_argument("--output-dir", type=Path, required=True)
    trial.add_argument("--numerics-profile", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--source-root", type=Path, required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    report = run_trial(**args) if command == "trial" else run_matrix(**args)
    print(json.dumps(report, ensure_ascii=False))
    return report.get("exit_code", 0 if report["status"] == "passed" else
                      1 if report["status"] == "execution_failed" else 2)


if __name__ == "__main__":
    raise SystemExit(main())

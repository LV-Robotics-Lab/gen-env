"""Attack tests for numerical trial provenance and matrix candidate acceptance."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_text_repair import case

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import tune_text_physics as tune

NUMERICS = "dt2ms_tau10ms_authored_v1"


def write(path, value):
    official.write_json(path, value)


def reseal(ctx):
    ctx.report["input_sha256"] = library.sha256(ctx.directory / "physics_input.json")
    ctx.report["trace_sha256"] = library.sha256(ctx.directory / "trace.jsonl")
    ctx.report["artifacts"] = [
        official.fingerprint(p, ctx.directory) for p in sorted(ctx.directory.rglob("*"))
        if p.is_file() and p.name != "trial_result.json"
    ]
    write(ctx.directory / "trial_result.json", ctx.report)


@pytest.fixture
def trial(tmp_path):
    original, rows = case()
    mapping = {"table": "table_1", "a": "cup_1", "ground": "ground"}
    assets = {}
    source_bytes, derived_bytes = [], []
    for name in ("table", "a"):
        asset = copy.deepcopy(original["assets"][name])
        asset["support"] = mapping[asset["support"]]
        asset["category"] = "table" if name == "table" else "cup"
        for kind, tracked in (("source", source_bytes), ("derived", derived_bytes)):
            root = tmp_path / kind / mapping[name]
            root.mkdir(parents=True)
            path = root / "mesh.dat"
            path.write_bytes(b"frozen asset mesh")
            asset[kind + "_root"] = str(root)
            asset[kind + "_files"] = [official.fingerprint(path, root)]
            tracked.append(path)
        assets[mapping[name]] = asset
    poses = {mapping[n]: original["poses"][n] for n in ("table", "a")}
    original = physics.frozen_input(assets, poses, [], 0)
    source = tmp_path / "source_input.json"
    write(source, original)
    data = physics.frozen_input(assets, poses, [], 0, numerics_profile=NUMERICS)
    for row in rows:
        row["objects"] = {mapping[n]: s for n, s in row["objects"].items() if n in mapping}
        row["contacts"] = [c for c in row["contacts"] if c["a"] in mapping and c["b"] in mapping]
        for contact in row["contacts"]:
            contact["a"], contact["b"] = mapping[contact["a"]], mapping[contact["b"]]
    directory = tmp_path / "trial"
    directory.mkdir()
    write(directory / "physics_input.json", data)
    (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    implementation = tune.implementation_files()
    manifest = dict(source_input=str(source), source_input_sha256=library.sha256(source),
                    implementation_sha256=implementation,
                    purpose="frozen table/cup diagnostic; not complete four-asset acceptance")
    write(directory / "source_manifest.json", manifest)
    result = physics.evaluate(data, rows)
    assert result["passed"]
    write(directory / "validation_result.json", result)
    report = dict(
        schema_version=tune.SCHEMA, status="physics_passed", exit_code=0,
        numerics_profile=NUMERICS, acceptance_profile="text_repair_v1",
        simulation_executed=True, steps_executed=data["settings"]["steps"], result=result,
    )
    ctx = SimpleNamespace(directory=directory, source=source, original=original,
                          data=data, rows=rows, report=report, manifest=manifest,
                          implementation=implementation, source_bytes=source_bytes,
                          derived_bytes=derived_bytes)
    reseal(ctx)
    return ctx


def verify(ctx):
    return tune.verify_trial(ctx.directory, source_hash=library.sha256(ctx.source),
                             numerics_profile=NUMERICS,
                             expected_implementation=ctx.implementation)


def test_valid_trial_remains_replayable(trial):
    assert verify(trial)["result"]["passed"]


@pytest.mark.parametrize("field,value", [
    ("exit_code", 2), ("status", "physics_failed"),
    ("steps_executed", 1499), ("simulation_executed", False),
])
def test_top_level_report_cannot_contradict_recomputed_verdict(trial, field, value):
    trial.report[field] = value
    write(trial.directory / "trial_result.json", trial.report)
    with pytest.raises(ValueError):
        verify(trial)


def test_failed_trajectory_cannot_be_selected_by_forging_exit_zero(trial):
    # Above the limit this trial's own settings derive from its step size, sustained long
    # enough to be motion rather than a one-sample contact-detection artefact.
    speed = 2*trial.data["settings"]["effective_speed_mps"]
    for row in trial.rows[-26:]:
        row["objects"]["cup_1"]["velocity"] = [speed, 0, 0]
    result = physics.evaluate(trial.data, trial.rows)
    assert not result["passed"]
    trial.report["result"] = result
    write(trial.directory / "validation_result.json", result)
    (trial.directory / "trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in trial.rows)
    )
    # The trajectory and all artifact hashes are valid, but exit_code/status claim success.
    reseal(trial)
    with pytest.raises(ValueError):
        verify(trial)


def test_trial_from_different_implementation_cannot_be_reused(trial):
    trial.manifest["implementation_sha256"] = dict(trial.implementation)
    key = next(iter(trial.implementation))
    trial.manifest["implementation_sha256"][key] = "0" * 64
    write(trial.directory / "source_manifest.json", trial.manifest)
    reseal(trial)
    with pytest.raises(ValueError):
        verify(trial)


@pytest.mark.parametrize("kind", ["source_bytes", "derived_bytes"])
def test_changed_external_asset_bytes_invalidate_resume(trial, kind):
    getattr(trial, kind)[0].write_bytes(b"changed asset after trial")
    with pytest.raises(ValueError):
        verify(trial)


@pytest.mark.parametrize("field", ["assets", "poses", "relations", "random_seed"])
def test_resealed_trial_cannot_claim_another_source_scene(trial, field):
    # Change the source, bind its true new hash, but retain the previous trial input/trace.
    original = copy.deepcopy(trial.original)
    if field == "assets":
        original["assets"]["cup_1"]["category"] = "apple"
    elif field == "poses":
        original["poses"]["cup_1"]["position"][0] += .01
    elif field == "relations":
        original["relations"] = [dict(source="cup_1", relation="on", target="table_1")]
    else:
        original["random_seed"] = 42
    write(trial.source, original)
    trial.manifest["source_input_sha256"] = library.sha256(trial.source)
    write(trial.directory / "source_manifest.json", trial.manifest)
    reseal(trial)
    with pytest.raises(ValueError):
        verify(trial)


def test_changed_source_file_cannot_hide_behind_manifest_hash(trial):
    changed = copy.deepcopy(trial.original)
    changed["random_seed"] = 42
    write(trial.source, changed)
    # No expected source hash is supplied; verifier must still read and validate the source.
    with pytest.raises(ValueError):
        tune.verify_trial(trial.directory, numerics_profile=NUMERICS,
                          expected_implementation=trial.implementation)


def test_matrix_rejects_same_seed_renamed_into_three_directories(trial, tmp_path, monkeypatch):
    root = tmp_path / "seeds"
    for seed in tune.SEEDS:
        path = root / f"seed_{seed}" / "03_physics" / "physics_input.json"
        path.parent.mkdir(parents=True)
        write(path, trial.original)  # Every file actually declares seed 0.
    monkeypatch.setattr(
        tune.subprocess, "run", lambda *a, **k: pytest.fail("must reject before run")
    )
    try:
        result = tune.run_matrix(root, tmp_path / "matrix")
    except ValueError as exc:
        assert "seed" in str(exc).lower()
    else:
        assert result["status"] == "execution_failed"
        assert "seed" in result["error"].lower()


def test_resealed_source_cannot_change_original_acceptance_settings(trial):
    original = copy.deepcopy(trial.original)
    original["settings"]["support_fraction"] = .1
    write(trial.source, original)
    trial.manifest["source_input_sha256"] = library.sha256(trial.source)
    write(trial.directory / "source_manifest.json", trial.manifest)
    reseal(trial)
    with pytest.raises(ValueError):
        verify(trial)

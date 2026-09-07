"""Half-step acceptance must replay the original release without another placement search."""

import copy
import json

import pytest
from test_task_output import make_task, snapshot
from test_text_repair import case

from self_improving.sim_adapters.genesis import construct_asset_scene as construction
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import replay_text_half_dt as replay


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def baseline(tmp_path):
    source = make_task(tmp_path)
    source.start_scene()
    data, rows = case()
    asset_dir = source.stage("scene") / "assets"
    asset_dir.mkdir()
    model = asset_dir / "fixture.xml"
    model.write_text("offline bound fixture")
    files = [replay.official.fingerprint(model, asset_dir)]
    for asset in data["assets"].values():
        asset.update(
            source_root=str(asset_dir),
            source_files=files,
            derived_root=str(asset_dir),
            derived_files=files,
        )
    data = physics.frozen_input(
        data["assets"],
        data["poses"],
        data["relations"],
        7,
        numerics_profile=numerics.CANDIDATE_PROFILES[0],
        repair_preset="text_scene_v2",
    )
    replay.clip.write_json(
        source.stage("scene") / "scene_v0.json",
        dict(graph=dict(relations=data["relations"], order=list(data["assets"]))),
    )
    source.finish_scene(dict(status="scene_built"))
    source.start_physics()
    attempt = source.stage("physics") / "attempts" / "000"
    attempt.mkdir(parents=True)
    replay.clip.write_json(attempt / "physics_input.json", data)
    # A distinct stable terminal pose demonstrates that it cannot become the new release.
    rows[-1]["objects"]["a"]["position"][0] += 0.0001
    write_rows(attempt / "trace.jsonl", rows)
    replay.clip.write_json(attempt / "final_state.json", dict(state=rows[-1]))
    result = physics.evaluate(data, rows)
    assert result["passed"]
    result.update(
        input_sha256=replay.lib.sha256(attempt / "physics_input.json"),
        trace_sha256=replay.lib.sha256(attempt / "trace.jsonl"),
    )
    replay.clip.write_json(attempt / "validation_result.json", result)
    replay.clip.write_json(
        attempt / "manifest.json",
        dict(files=[replay.official.fingerprint(p, attempt) for p in sorted(attempt.iterdir())]),
    )
    source.report["construction"] = dict(
        repair_preset="text_scene_v2",
        attempts=[
            dict(
                directory=str(attempt),
                passed=True,
                result_sha256=replay.lib.sha256(attempt / "validation_result.json"),
            )
        ],
    )
    source.finish_physics(
        dict(status="physics_passed", physics_status="passed", render_status="not_run")
    )
    return source, data, rows


def test_half_input_changes_only_registered_numerics():
    data, _ = case()
    data = physics.frozen_input(
        data["assets"],
        data["poses"],
        data["relations"],
        31,
        numerics_profile=numerics.CANDIDATE_PROFILES[0],
        repair_preset="text_scene_v2",
    )
    before = copy.deepcopy(data)
    half = replay.half_input(data)
    assert data == before
    assert {k for k in half if half[k] != data[k]} == {"settings", "numerics_profile"}
    assert half["numerics_profile"] == numerics.half_dt_profile(data["numerics_profile"])
    assert half["settings"]["dt"] == data["settings"]["dt"] / 2
    assert half["settings"]["steps"] == data["settings"]["steps"] * 2
    for field in ("assets", "poses", "relations", "random_seed", "repair_preset"):
        assert half[field] == data[field]


@pytest.mark.parametrize("mode", ["passed", "failed", "execution_error"])
def test_replay_preserves_source_and_release_and_gates_final(tmp_path, monkeypatch, mode):
    source, original, source_rows = baseline(tmp_path / "source")
    before = snapshot(source.root)
    scene_before = snapshot(source.stage("scene"))
    obj_before = snapshot(source.stage("objects"))
    received = []
    media = []
    monkeypatch.setattr(construction, "repair_loop", lambda *a, **k: pytest.fail("no repair loop"))
    monkeypatch.setattr(construction, "make_initial", lambda *a, **k: pytest.fail("no resampling"))

    def simulator(data, out, check):
        received.append(copy.deepcopy(data))
        assert data["poses"] == original["poses"]
        assert data["poses"]["a"]["position"] != source_rows[-1]["objects"]["a"]["position"]
        check()
        rows = [
            copy.deepcopy(source_rows[0 if not i else 1])
            for i in range(data["settings"]["steps"] + 1)
        ]
        for i, row in enumerate(rows):
            row.update(step=i, time_s=i * data["settings"]["dt"])
        if mode == "execution_error":
            write_rows(out / "trace.jsonl", rows[:4])
            raise RuntimeError("interrupted after three steps")
        if mode == "failed":
            for row in rows[-51:]:
                row["objects"]["a"]["velocity"] = [0.01, 0, 0]
        write_rows(out / "trace.jsonl", rows)
        return physics.evaluate(data, rows)

    def recorder(data, trace, out, check, **kwargs):
        check()
        media.append(kwargs)
        out.mkdir(parents=True)
        (out / "fixture.json").write_text(json.dumps(kwargs))

    result = replay.run(
        source.root, tmp_path / "half", render=True, simulator=simulator, recorder=recorder
    )
    assert len(received) == 1 and result["placement_calls"] == 0
    assert snapshot(source.root) == before
    target = replay.TaskOutput(tmp_path / "half")
    source.verify()
    target.verify()
    assert snapshot(target.stage("scene")) == scene_before
    assert snapshot(target.stage("objects")) == obj_before
    assert "construction" not in target.report
    for item in result["source_baseline"]["files"].values():
        assert replay.lib.sha256(target.stage("physics") / item["archived_path"]) == item["sha256"]
    assert (
        replay.lib.sha256(target.stage("physics") / result["source_baseline"]["archived_manifest"])
        == (result["source_baseline"]["source_manifest_sha256"])
    )
    if mode == "passed":
        assert result["status"] == "physics_passed" and result["exit_code"] == 0
        assert media == [dict(physics_passed=True), dict(final=True, physics_passed=True)]
        assert (target.stage("physics") / "validated_scene.json").exists()
    elif mode == "failed":
        assert result["status"] == "physics_failed" and result["exit_code"] == 2
        assert media == [dict(physics_passed=False)]
        assert not (target.stage("physics") / "validated_scene.json").exists()
    else:
        assert result["status"] == "execution_failed" and result["exit_code"] == 1
        assert result["steps_executed"] == 3 and result["simulation_executed"]
        assert not media
    assert bool(list(target.stage("final_render").iterdir())) == (mode == "passed")
    assert result["source_baseline"]["files"]["input"]["sha256"] == replay.lib.sha256(
        source.stage("physics") / "attempts/000/physics_input.json"
    )


def test_nonpassing_baseline_is_rejected_before_copy(tmp_path):
    source, _, _ = baseline(tmp_path / "source")
    source.finish_physics(
        dict(status="physics_failed", physics_status="failed", render_status="not_run")
    )
    before = snapshot(source.root)
    with pytest.raises(ValueError, match="passing text_scene_v2"):
        replay.run(source.root, tmp_path / "half")
    assert not (tmp_path / "half").exists()
    assert snapshot(source.root) == before

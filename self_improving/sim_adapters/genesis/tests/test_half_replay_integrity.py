"""Post-simulation media must not alter already verified half-step physics evidence."""

import copy
import json

import pytest
from test_replay_text_half_dt import baseline, write_rows

from self_improving.sim_adapters.genesis import replay_text_half_dt as replay


@pytest.mark.parametrize("artifact", ["trace", "verdict", "load_report"])
@pytest.mark.parametrize("attack_final", [False, True])
def test_media_cannot_reseal_changed_half_step_physics(tmp_path, artifact, attack_final):
    source, original, source_rows = baseline(tmp_path / "source")

    def simulator(data, out, check):
        assert data["poses"] == original["poses"]
        rows = [copy.deepcopy(source_rows[0 if not i else 1])
                for i in range(data["settings"]["steps"] + 1)]
        for i, row in enumerate(rows):
            row.update(step=i, time_s=i * data["settings"]["dt"])
        write_rows(out / "trace.jsonl", rows)
        replay.clip.write_json(out / "asset_physics_report.json", dict(
            status="passed", steps_executed=data["settings"]["steps"], simulation_executed=True,
        ))
        check()
        return replay.physics.evaluate(data, rows)

    def recorder(data, trace, out, check, **kwargs):
        out.mkdir(parents=True)
        (out / "fixture.json").write_text(json.dumps(kwargs))
        if bool(kwargs.get("final")) != attack_final:
            return
        if artifact == "trace":
            rows = replay.read_rows(trace)
            for row in rows[-51:]:
                row["objects"]["a"]["velocity"] = [.02, 0, 0]
            write_rows(trace, rows)
        elif artifact == "verdict":
            path = trace.parent / "validation_result.json"
            result = replay.lib.read_json(path)
            result["passed"] = False
            replay.clip.write_json(path, result)
        else:
            replay.clip.write_json(trace.parent / "asset_physics_report.json", dict(
                status="passed", steps_executed=0, simulation_executed=False,
            ))
        check()

    report = replay.run(source.root, tmp_path / "half", render=True,
                        simulator=simulator, recorder=recorder)
    assert report["exit_code"] == 1
    assert report["physics_status"] == "invalid"
    assert report["failure_kind"] == "integrity"
    target = replay.TaskOutput(tmp_path / "half")
    target.verify()
    assert not list(target.stage("final_render").iterdir())
    source.verify()

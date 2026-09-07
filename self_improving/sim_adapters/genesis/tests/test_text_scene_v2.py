"""Controlled text-scene repair lifecycle and durable media contracts."""

import copy
import json

import pytest
from test_task_output import make_task, snapshot
from test_text_repair import case

from self_improving.sim_adapters.genesis import construct_asset_scene as entry
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video


@pytest.mark.parametrize("dt,name,slowdown", [
    (0.002, "physics_replay_10x_slow.mp4", 10),
    (0.001, "physics_replay_20x_slow.mp4", 20),
    (0.004, "physics_replay_5x_slow.mp4", 5),
])
def test_replay_name_reports_actual_timestep(dt, name, slowdown):
    result = video.replay_timing(dt)
    assert result["filename"] == name
    assert result["slowdown_factor"] == slowdown
    assert result["playback_speed"] == pytest.approx(1 / slowdown)


@pytest.mark.parametrize("dt", [0, -0.001, float("nan"), float("inf")])
def test_replay_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match="timestep"):
        video.replay_timing(dt)


@pytest.mark.parametrize("failures,complete,expected", [
    ({"a": ["stable_velocity"], "b": []}, True, True),
    ({"a": ["support_preserved", "stable_velocity"]}, True, True),
    ({"a": ["support_preserved", "unexpected_contact"]}, True, False),
    ({"a": ["stable_velocity"], "b": ["support_geometry"]}, True, False),
    ({"a": []}, True, False),
    ({"a": ["stable_velocity"]}, False, False),
])
def test_contact_only_failures_require_all_other_gates(failures, complete, expected):
    assert entry.contact_dynamics_failed(dict(failures=failures, complete=complete)) is expected


@pytest.mark.parametrize("mode,status,kind,code", [
    ("prepare_error", "preparation_failed", "asset_preparation", 1),
    ("placement_error", "scene_failed", "PLACEMENT_FAILED", 2),
    ("contact", "physics_failed", "CONTACT_DYNAMICS_FAILED", 2),
    ("execution_error", "execution_failed", "physics", 1),
    ("passed", "physics_passed", None, 0),
])
def test_v2_task_state_and_no_blind_contact_retries(
    tmp_path, monkeypatch, mode, status, kind, code
):
    source = make_task(tmp_path / "source")
    before = snapshot(source.root)
    index = tmp_path / "index" / "index.json"
    index.parent.mkdir()
    index.write_text("{}")
    data, template = case(True)
    document = dict(
        request=source.root.name,
        objects=[dict(object_id=n, category=a["category"], description=n)
                 for n, a in data["assets"].items()],
        relations=[dict(relation="on", source="a", target="table", evidence=""),
                   dict(relation="on", source="b", target="a", evidence="")],
    )
    bindings = {n: dict(source_root=str(index.parent), source_files=[])
                for n in data["assets"]}
    monkeypatch.setattr(entry, "bind_inputs", lambda *a: (document, bindings))
    prepared = []

    def preparer(doc, binding, out, fixed, metadata, check, *, repair_preset):
        prepared.append(repair_preset)
        check()
        if mode == "prepare_error":
            raise ValueError("invalid collision")
        assets = copy.deepcopy(data["assets"])
        for a in assets.values():
            a.update(derived_root=str(out), derived_files=[])
        return assets

    def initial(*args):
        if mode == "placement_error":
            raise entry.PlacementFailure("no valid initial placement")
        return copy.deepcopy(data["poses"])

    monkeypatch.setattr(entry, "make_initial", initial)
    monkeypatch.setattr(
        geo, "sample_object", lambda *a: pytest.fail("pure contact failure retried")
    )

    def simulator(current, out, check):
        if mode == "execution_error":
            raise RuntimeError("solver unavailable")
        rows = copy.deepcopy(template)
        if mode == "contact":
            for row in rows[-26:]:
                row["objects"]["a"]["velocity"] = [0.01, 0, 0]
        (out / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (out / "final_state.json").write_text(json.dumps(dict(state=rows[-1])))
        check()
        return physics.evaluate(current, rows)

    rendered = []

    def recorder(data, trace, out, check, **kwargs):
        check()
        out.mkdir(parents=True)
        (out / "fixture.json").write_text(json.dumps(kwargs))
        rendered.append(kwargs)

    target = tmp_path / "result"
    report = entry.run(source.root, target, index, fixed_objects=["table"], render=True,
                       repair_preset="text_scene_v2", preparer=preparer,
                       simulator=simulator, recorder=recorder)
    assert prepared == ["text_scene_v2"]
    assert (report["status"], report["failure_kind"], report["exit_code"]) == (status, kind, code)
    assert snapshot(source.root) == before
    task = entry.TaskOutput(target)
    task.verify()
    if mode in {"prepare_error", "placement_error"}:
        assert report["physics_status"] == "not_run"
        assert not report["simulation_executed"] and report["steps_executed"] == 0
        assert not list(task.stage("physics").iterdir())
        assert (task.stage("scene") / "construction_result.json").exists()
    elif mode == "contact":
        assert len(report["attempts"]) == 1
        assert report["steps_executed"] == 1500
        assert report["attempts"][0]["stop_reason"] == "CONTACT_DYNAMICS_FAILED"
    elif mode == "execution_error":
        assert report["physics_status"] == "invalid"
        assert report["steps_executed"] == 0
    assert bool(list(task.stage("final_render").iterdir())) == (mode == "passed")
    if mode == "passed":
        assert rendered[-1] == dict(final=True, physics_passed=True)


@pytest.mark.parametrize("dt,final", [(0.002, False), (0.001, False), (0.002, True)])
def test_media_keeps_every_sample_and_exports_final_views(tmp_path, monkeypatch, dt, final):
    from types import SimpleNamespace

    import numpy as np

    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    geometry = tmp_path / "geometry.npz"
    np.savez(geometry, vertices=vertices)
    asset = dict(model_entrypoint=str(tmp_path / "asset.glb"), scale=1,
                 anchor_m=[0, 0, 0], hull=vertices.tolist(), geometry_file=str(geometry))
    data = dict(settings=dict(dt=dt), assets=dict(a=asset))
    rows = [dict(step=i, time_s=i * dt, objects=dict(a=dict(
        position=[i * 1e-6, 0, 0], orientation_wxyz=[1, 0, 0, 0])))
        for i in range(round(3 / dt) + 1)]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "physics_input.json").write_text(json.dumps(data))

    class Entity:
        def __init__(self):
            self.position = [0, 0, 0]
            self.orientation = [1, 0, 0, 0]

        def set_pos(self, value):
            self.position = value

        def set_quat(self, value):
            self.orientation = value

    class Camera:
        def __init__(self):
            self.frames = 0

        def set_pose(self, **kwargs):
            pass

        def render(self, **kwargs):
            assert kwargs["force_render"]
            self.frames += 1
            return np.full((3, 4, 3), self.frames % 256, dtype=np.uint8), None, None, None

    class Scene:
        def __init__(self, **kwargs):
            pass

        def add_entity(self, *args, **kwargs):
            return Entity()

        def add_camera(self, **kwargs):
            return Camera()

        def build(self):
            pass

        def step(self):
            pytest.fail("media replay must never simulate")

    def ignored(**kwargs):
        return None
    gs = SimpleNamespace(Scene=Scene, destroy=lambda: None,
                         renderers=SimpleNamespace(Rasterizer=ignored),
                         options=SimpleNamespace(VisOptions=ignored),
                         morphs=SimpleNamespace(Plane=ignored, Mesh=ignored, MJCF=ignored),
                         materials=SimpleNamespace(Rigid=ignored),
                         surfaces=SimpleNamespace(Default=ignored))
    monkeypatch.setattr(video.repair_assets, "init_genesis", lambda: gs)
    monkeypatch.setattr(video.native, "pose", lambda e: dict(
        position=e.position, orientation_wxyz=e.orientation))
    monkeypatch.setattr(video.repair_assets, "visible_geometry", lambda e, path: (
        geo.transform(vertices, dict(position=e.position, orientation_wxyz=e.orientation)), None))
    monkeypatch.setattr(video.physics, "evaluate", lambda *args: dict(passed=True))
    encoded = []

    class Sink:
        def write(self, value):
            encoded.append(len(value))

        def close(self):
            pass

    class Process:
        stdin = Sink()

        def wait(self):
            return 0

        def poll(self):
            return 0

    def encode(path, fps):
        path.write_bytes(b"mock encoded video")
        return Process()

    monkeypatch.setattr(video, "encode", encode)
    monkeypatch.setattr(video, "verify_video", lambda path, count, fps: dict(
        file=path.name, frame_count=count, fps=fps, duration_s=count / fps))
    out = tmp_path / "render"
    original_step = Scene.step
    result = video.render(data, trace, out, lambda: None, final=final, physics_passed=final)
    count = 180 if final else len(rows)
    assert len(encoded) == result["frame_count"] == result["saved_frame_count"] == count
    assert len(list((out / "frames").glob("*.png"))) == count
    ledger = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert [r["source_step"] for r in ledger] == (
        [rows[-1]["step"]] * 180 if final else list(range(len(rows))))
    for item in (ledger[0], ledger[-1]):
        assert video.lib.sha256(out / item["png_path"]) == item["png_sha256"]
    assert result["physics_steps"] == 0 and Scene.step is original_step
    assert result["physics_duration_s"] == pytest.approx(3)
    if final:
        assert {p["path"] for p in result["final_images"]} == {
            "overview.png", "top.png", "side.png"
        }
        assert result["file"] == "orbit.mp4"
    else:
        assert result["file"] == video.replay_timing(dt)["filename"]
        assert result["playback_speed"] == pytest.approx(dt * 50)

"""Preserve upstream support and poses without retrieval or acceptance claims."""
import json
from unittest.mock import Mock

import pytest

from self_improving.sim_adapters.genesis import reconstruct_media as media
from self_improving.sim_adapters.genesis.task_output import TaskOutput
from self_improving.sim_adapters.genesis.tests import test_simfoundry_scene


@pytest.fixture
def case(tmp_path):
    return test_simfoundry_scene.case.__wrapped__(tmp_path)


@pytest.mark.parametrize("enabled,visible", [(True, True), (True, False), (False, False)])
def test_existing_scene_preserves_plane_and_pose(
    case, tmp_path, monkeypatch, enabled, visible,
):
    scene, _, _, state, _ = case
    state["init_info"] = {"args": {"use_floor_plane": enabled}}
    state["ground_plane_info"]["visible"] = visible
    raw = scene / "s14_og/reconstructed_og_scene.json"
    raw.write_text(json.dumps(state))
    source_bytes = raw.read_bytes()
    previews = []

    def preview(package, target, **options):
        layout = media.scene_import.verify(package)
        previews.append((layout, options))
        target.mkdir()
        (target / "preview_report.json").write_text('{"status":"passed"}')

    monkeypatch.setattr(media.scene_import, "preview", preview)
    # Still forbidden: choosing a support asset, and re-solving the poses. Those would
    # replace the source placement this mode exists to preserve. Extracting the support
    # observation is neither -- it only reads what the reconstruction already recorded.
    forbidden = Mock(side_effect=AssertionError("retrieval must not run"))
    monkeypatch.setattr(media.support, "rank", forbidden)
    monkeypatch.setattr(media.support, "choose", forbidden)
    monkeypatch.setattr(media.workflow, "run", forbidden)
    out = tmp_path / "按上游平面转换场景"
    report = media.import_upstream_reconstruction(scene, out)
    assert report["status"] == "scene_built" and report["exit_code"] == 0
    # These fixtures place nothing on the plane, so there is no support graph to validate.
    # Reported with its reason rather than crashing, and never as a pass.
    assert report["physics_status"] == report["render_status"] == "not_run"
    assert "support chain" in report["physics_not_run_reason"]
    assert report["stages"] == dict(objects="passed", scene="passed", physics="not_run",
                                     final_render="not_run")
    layout, options = previews[0]
    assert options == {"orbit": True}
    assert layout["environment"]["ground"] == ("genesis_builtin_plane" if enabled else None)
    assert layout["environment"]["visible"] == visible
    assert layout["environment"]["position_m"] == [0, 0, 0.4]
    assert [obj["object_id"] for obj in layout["objects"]] == ["iter_0"]
    assert layout["objects"][0]["translation_m"] == [1, 2, 3]
    assert layout["relations"] == []
    assert not list((out / "04_final_render").iterdir())
    assert raw.read_bytes() == source_bytes
    snapshot = out / "01_obj/source_outputs/s14_og/reconstructed_og_scene.json"
    assert snapshot.read_bytes() == source_bytes
    forbidden.assert_not_called()
    TaskOutput(out).verify()
    with pytest.raises(FileExistsError):
        media.import_upstream_reconstruction(scene, out)


def test_upstream_config_does_not_read_clip_index(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    config = tmp_path / "llm.yaml"
    config.write_text("profiles: {}")
    result = media.task_config("image", source, None, config, ())
    assert result["support_mode"] == "upstream"
    assert result["clip_index"] is result["clip_index_sha256"] is None
    with pytest.raises(ValueError, match="requires --clip-index"):
        media.task_config("image", source, None, config, (), "retrieved")


def test_image_default_and_resume_do_not_select_support(case, tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    from PIL import Image

    scene = case[0]
    source = tmp_path / "鼠标.png"
    Image.new("RGB", (32, 24)).save(source)
    config = tmp_path / "llm.yaml"
    config.write_text("profiles: {}")
    out = tmp_path / "鼠标按原始平面重建"
    calls = []

    def runner(*args, **kwargs):
        calls.append("reconstruct")
        shutil.copytree(scene, out / "01_obj/reconstruction")
        return SimpleNamespace(returncode=0)

    def preview(package, target, **kwargs):
        calls.append("preview")
        target.mkdir()
        (target / "preview_report.json").write_text('{"status":"passed"}')

    monkeypatch.setattr(media.scene_import, "preview", preview)
    monkeypatch.setattr(media, "verify_sampled_frames", lambda *args: {})
    # What upstream must never do is retrieve a support asset or re-solve the poses: the
    # source placement is the claim under test. Running physics on it is not in that set --
    # free replay changes nothing, it only measures. support.observe is likewise allowed,
    # since upstream needs the reference camera it extracts, not the retrieval it feeds.
    forbidden = Mock(side_effect=AssertionError("finite support path must not run"))
    monkeypatch.setattr(media.support, "rank", forbidden)
    monkeypatch.setattr(media.workflow, "run", forbidden)
    never = Mock(side_effect=AssertionError("physics must not run without a support graph"))
    monkeypatch.setattr(media, "_physics_attempt", never)
    for resume in (False, True):
        result = media.run(source, "image", out, None, config,
                           process_runner=runner, resume=resume)
        assert result["status"] == "scene_built", result
        assert result["stages"]["scene"] == "passed"
        # This fixture's bodies rest on nothing, so there is no support graph to validate.
        # The scene is reported as unvalidated with the reason, never simulated anyway and
        # never passed, and the import itself still succeeds.
        assert result["physics_status"] == "not_run"
        assert "support chain" in result["physics_not_run_reason"]
        assert any("not validated" in line for line in result["limitations"])
        assert result["stages"]["final_render"] == "not_run"
        TaskOutput(out).verify()
    assert calls == ["reconstruct", "preview"]
    never.assert_not_called()
    forbidden.assert_not_called()

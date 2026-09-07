"""Moving shared assets must preserve historical bytes and reject changed dependencies."""
import pytest
from test_asset_library import library as make_library

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import extract_assets as extraction
from self_improving.sim_adapters.genesis import storage_paths as storage


def test_moved_library_and_selected_binding_keep_original_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    old, current = tmp_path / "ouput", tmp_path / "output"
    fixture = make_library.__wrapped__(old)
    selection = old / "selection"
    assert clip.select(output_dir=selection, **fixture.kwargs)["status"] == "selected"
    path = old / "clip/index.json"
    digest = lib.sha256(path)
    index, _ = clip.load_index(path)
    obj = dict(description=fixture.kwargs["query"])
    binding = extraction.verified_binding(obj, selection, index, digest)
    before = {str(p.relative_to(old)): lib.sha256(p) for p in old.rglob('*') if p.is_file()}
    old.rename(current)
    old.symlink_to("output", target_is_directory=True)
    moved, _ = clip.load_index(current / "clip/index.json")
    assert moved == index and lib.sha256(current / "clip/index.json") == digest
    checked = extraction.verified_binding(obj, current / "selection", moved, digest)
    assert checked["model_entrypoint"] == binding["model_entrypoint"]
    assert checked["source_root"] == binding["source_root"]
    assert before == {str(p.relative_to(current)): lib.sha256(p)
                      for p in current.rglob('*') if p.is_file()}
    source = current / "download/sources/bowl.urdf"
    source.write_text(source.read_text() + "<!-- tamper -->")
    with pytest.raises(ValueError, match="integrity"):
        clip.load_index(current / "clip/index.json")


def test_only_explicit_migration_alias_is_recognized(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    current = tmp_path / "output"
    current.mkdir()
    old = tmp_path / "ouput"
    assert not storage.same_evidence_path(old / "asset", current / "asset")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old.symlink_to(elsewhere)
    assert not storage.same_evidence_path(old / "asset", current / "asset")
    old.unlink()
    old.symlink_to("output")
    assert storage.same_evidence_path(old / "asset", current / "asset")
    assert not storage.same_evidence_path(old / "../asset", tmp_path / "asset")
    assert not storage.same_evidence_path(old / "asset", current / "different")


@pytest.mark.parametrize("legacy_name", ["output", "ouput"])
def test_resource_move_without_output_symlinks(tmp_path, monkeypatch, legacy_name):
    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    old = tmp_path / legacy_name / "genesis_assets"
    fixture = make_library.__wrapped__(old)
    selection = tmp_path / "selected"
    assert clip.select(output_dir=selection, **fixture.kwargs)["status"] == "selected"
    index, _ = clip.load_index(old / "clip/index.json")
    digest = lib.sha256(old / "clip/index.json")
    obj = dict(description=fixture.kwargs["query"])
    binding = extraction.verified_binding(obj, selection, index, digest)
    before = {p.relative_to(old).as_posix(): lib.sha256(p)
              for p in old.rglob("*") if p.is_file()}
    new = tmp_path / "assets/genesis"
    new.parent.mkdir()
    old.rename(new)
    assert not old.exists() and not old.is_symlink()
    moved, _ = clip.load_index(new / "clip/index.json")
    assert moved == index
    assert clip.load_index(old / "clip/index.json")[0] == index
    assert before == {p.relative_to(new).as_posix(): lib.sha256(p)
                      for p in new.rglob("*") if p.is_file()}
    assert extraction.verified_binding(obj, selection, moved, digest) == binding
    assert storage.local_path(binding["model_entrypoint"]).is_file()
    source = new / "download/sources/bowl.urdf"
    source.write_text(source.read_text() + "<!-- changed -->")
    with pytest.raises(ValueError, match="integrity"):
        clip.load_index(new / "clip/index.json")


def test_resource_move_rejects_shadow_roots_and_destination_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    new = tmp_path / "assets/genesis"
    new.mkdir(parents=True)
    old = tmp_path / "output/genesis_assets"
    assert storage.local_path(old / "asset") == new / "asset"
    assert storage.local_path(old / "../escape") != new / "../escape"
    old.mkdir(parents=True)
    assert storage.local_path(old / "asset") == old / "asset"
    old.rmdir()
    new.rmdir()
    other = tmp_path / "other"
    other.mkdir()
    new.symlink_to(other)
    assert storage.local_path(old / "asset") == old / "asset"


def test_task_locks_stay_outside_output(tmp_path, monkeypatch):
    from self_improving.sim_adapters.genesis import task_output
    monkeypatch.setattr(task_output, "CACHE_ROOT", tmp_path / ".cache/genesis")
    task = task_output.TaskOutput(tmp_path / "output/request")
    with task.lock():
        assert not (tmp_path / "output/.task_locks").exists()
        assert len(list((tmp_path / ".cache/genesis/task_locks").glob("*.lock"))) == 1
        with pytest.raises(ValueError, match="already running"):
            with task.lock():
                pass

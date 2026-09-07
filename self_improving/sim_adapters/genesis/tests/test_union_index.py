"""Union source identity, routing and post-selection failure propagation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_clip_select import response
from test_clip_select import setup as _setup

from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import union_index

setup_fixture = pytest.fixture(name="setup")(_setup.__wrapped__)


def union(s):
    p = s.root / "union"
    union_index.build([f"official={s.index}/index.json", f"generated={s.index}/index.json"], p)
    return p / "index.json"


def test_union_duplicate_local_ids_and_image_roots(setup):
    path = union(setup)
    index, vectors = clip.load_index(path)
    assert len({a["asset_id"] for a in index["assets"]}) == 8
    assert vectors.shape[0] == 48
    candidates = clip.retrieve(index["rows"], vectors, [[1, 0]], 3)
    messages = clip.image_messages("cup", candidates, Path("/nonexistent"))
    assert len([c for c in messages[1]["content"] if c["type"] == "image_url"]) == 6
    assert all(a["source_root"] == str(setup.assets) for a in index["assets"])
    old = json.loads(path.read_text())
    old["rows"][0]["preview_root"] = "/wrong"
    clip.write_json(path, old)
    with pytest.raises(ValueError, match="binding"):
        clip.load_index(path)


def test_union_member_mutation(setup):
    path = union(setup)
    with (setup.index / "index.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="member changed"):
        clip.load_index(path)


@pytest.mark.parametrize("exit_code,status", [(0, "passed"), (2, "failed"), (1, "not_evaluated")])
def test_selection_then_physics_no_fallback(setup, monkeypatch, exit_code, status):
    path = union(setup)
    calls = []

    def worker(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        binding = Path(command[command.index("--binding") + 1])
        assert binding.exists()  # selected before validation
        calls.append(json.loads(binding.read_text())["asset_id"])
        output.mkdir()
        clip.write_json(
            output / "physics_result.json", dict(exit_code=exit_code, physics_status=status)
        )
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(clip.subprocess, "run", worker)
    result = clip.select(
        **(
            setup.kwargs
            | dict(
                clip_index=path,
                output_dir=setup.root / "selected",
                vlm=lambda _: response(number=1),
            )
        )
    )
    assert len(calls) == 1
    assert result["exit_code"] == exit_code
    assert result["status"] == ("selected" if exit_code == 0 else "error")
    selected = json.loads((setup.root / "selected/selected_asset.json").read_text())
    assert selected["asset_id"] == calls[0] and selected["physics_status"] == status
    assert result["vlm_calls"] == 1

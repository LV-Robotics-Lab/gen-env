"""Opt-in four-asset acceptance, separate from physics acceptance experiments."""

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_official_index as index


@pytest.mark.skipif(os.environ.get("GENESIS_INDEX_REAL") != "1",
                    reason="set GENESIS_INDEX_REAL=1 and a new GENESIS_INDEX_OUTPUT")
def test_four_official_assets_zero_step_previews(monkeypatch):
    import genesis as gs

    def forbidden_step(*args, **kwargs):
        pytest.fail("asset previews must never call Scene.step")

    monkeypatch.setattr(gs.Scene, "step", forbidden_step)
    output = Path(os.environ["GENESIS_INDEX_OUTPUT"])
    assert index.build(output)["status"] == "passed"
    assert index.verify_index(output)["status"] == "passed"
    assert len(list(output.glob("previews/*/view_*.png"))) == 24
    assert len(list(output.glob("previews/*/contact_sheet.png"))) == 4
    with Image.open(output / "overview.png") as picture:
        picture.load()
        assert picture.size == (1024, 1072)
    for asset_id in index.ASSETS:
        evidence = json.loads((output / f"previews/{asset_id}/preview_result.json").read_text())
        assert evidence["physics_steps"] == 0
        assert evidence["loaded_collision_parts"] == 32
        assert evidence["loaded_visual_parts"] == (10 if asset_id == "donut_0" else 1)
        assert evidence["geometry"]["max_bounds_error_m"] <= 1e-6
        assert evidence["geometry"]["max_vertex_error_m"] <= 1e-6
        for view in evidence["views"]:
            with Image.open(output / view["image"]["path"]) as picture:
                picture.load()
                assert picture.size == (512, 512)
            assert view["visibility"]["pixels"] > 0

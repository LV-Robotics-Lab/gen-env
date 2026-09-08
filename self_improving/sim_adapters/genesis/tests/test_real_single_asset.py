"""Explicit real single-body control, independent of generated assets and network."""

import json
import os
from pathlib import Path

import pytest

from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import validate_single_asset as validator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENESIS_SINGLE_ASSET_REAL") != "1",
    reason="explicit real Genesis acceptance only",
)


def test_authored_contact_margin_control(tmp_path):
    source = Path(__file__).parent / "fixtures/single_asset/drop_control.xml"
    source = source.resolve()
    binding = tmp_path / "binding.json"
    clip.write_json(
        binding,
        dict(
            model_entrypoint=str(source),
            source_root=str(source.parent),
            source_files=[clip.official.fingerprint(source, source.parent)],
        ),
    )
    result = validator.run(tmp_path / "physics", binding=binding)
    assert result["exit_code"] == 0, result
    assert result["steps_executed"] == 1000
    assert result["video"]["total_frames"] == 101


def test_at_rest_release_is_not_accepted_as_drop_evidence(tmp_path):
    """The easier release must not be able to stand in for the drop test downstream.

    An at-rest run never lands, so its trajectory says nothing about what the asset does on
    impact. It is a legitimate way to validate a scene whose pose was authored at rest, and
    an illegitimate way to qualify an asset for selection -- verify_evidence separates them.
    """
    source = (Path(__file__).parent / "fixtures/single_asset/drop_control.xml").resolve()
    binding = tmp_path / "binding.json"
    clip.write_json(
        binding,
        dict(
            model_entrypoint=str(source),
            source_root=str(source.parent),
            source_files=[clip.official.fingerprint(source, source.parent)],
        ),
    )
    result = validator.run(tmp_path / "physics", binding=binding, at_rest=True)
    assert result["release_mode"] == "at_rest"
    assert result["steps_executed"] == 1000

    frozen = json.loads((tmp_path / "physics/physics_input.json").read_text())
    assert frozen["clearance_m"] == 0.0

    selected = json.loads(binding.read_text())
    selected.update(
        physics_status="passed",
        physics_evidence=clip.official.fingerprint(
            tmp_path / "physics/physics_result.json", tmp_path / "physics"
        ),
    )
    with pytest.raises(ValueError, match="drop test"):
        validator.verify_evidence(tmp_path / "physics", selected)

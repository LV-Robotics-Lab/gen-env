"""Explicit real single-body control, independent of generated assets and network."""

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

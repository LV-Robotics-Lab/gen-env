"""Opt-in six-query acceptance, preserving failures and using no simulator runtime."""
import json
import os
from pathlib import Path

import pytest

from self_improving.sim_adapters.genesis import clip_select as clip

pytestmark = pytest.mark.skipif(os.environ.get("GENESIS_CLIP_REAL") != "1",
                                reason="explicit real CLIP and online VLM opt-in required")
CASES = [
    ("apple", "苹果", {"apple_15"}),
    ("donut", "甜甜圈", {"donut_0"}),
    ("yellow_cup", "黄色杯子", {"cup_2"}),
    ("handled_cup", "带把手的杯子", {"mug_1"}),
    ("hammer", "锤子", set()),
    ("mickey_cup", "印有米老鼠的杯子", {"mug_1", "cup_2"}),
]


@pytest.fixture(scope="module")
def real_runs():
    destination = Path(os.environ["GENESIS_CLIP_OUTPUT"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    index = Path(os.environ.get("GENESIS_CLIP_INDEX", "assets/genesis/clip_v1/index.json"))
    config = Path(os.environ.get("GENESIS_CLIP_CONFIG", "configs/llm.yaml"))
    runs = {}
    summary = []
    # All six execute before assertions, retaining every failure without changing expected assets.
    for name, query, expected in CASES:
        output = destination / name
        report = clip.select(index, query, output, vlm_config=config,
                             cache_dir=destination / "cache")
        retrieval = json.loads((output / "retrieval_result.json").read_text())
        evidence = json.loads((output / "vlm_selection.json").read_text())
        binding = (json.loads((output / "selected_asset.json").read_text())
                   if (output / "selected_asset.json").exists() else None)
        candidate_ids = [c["asset_id"] for c in retrieval["candidates"]]
        row = dict(name=name, query=query, expected_assets=sorted(expected),
                   candidates=candidate_ids, top3_hit=bool(expected.intersection(candidate_ids)),
                   status=report["status"], selected_asset=binding["asset_id"] if binding else None,
                   timings_s=report["timings_s"], vlm_calls=report["vlm_calls"],
                   cache=report["cache"], error=report.get("error"),
                   visible_differences=binding["visible_differences"] if binding else [])
        summary.append(row)
        clip.write_json(destination / "acceptance_summary.json", summary)
        runs[name] = (report, row, evidence)
    cache_report = clip.select(index, "苹果", destination / "apple_cached", vlm_config=config,
                               cache_dir=destination / "cache")
    runs["cache"] = cache_report
    return runs


@pytest.mark.parametrize("name,query,expected", CASES)
def test_real_selection(real_runs, name, query, expected):
    report, row, evidence = real_runs[name]
    assert report["status"] != "error", report
    assert report["vlm_calls"] == 1
    if name == "hammer":
        assert report["status"] == "rejected" and row["selected_asset"] is None
    elif name == "mickey_cup":
        assert report["status"] == "rejected" or row["selected_asset"] in expected
        if report["status"] == "selected":
            differences = " ".join(evidence["selection"]["visible_differences"])
            assert "米老鼠" in differences
            assert any(term in differences for term in ("未", "无", "不", "没有", "无法"))
    else:
        assert row["top3_hit"], row
        assert report["status"] == "selected" and row["selected_asset"] in expected, row


def test_real_cache_hit(real_runs):
    report = real_runs["cache"]
    assert report["status"] == "selected" and report["cache"]["hit"]
    assert report["vlm_calls"] == 0 and report["timings_s"]["vlm"] == 0

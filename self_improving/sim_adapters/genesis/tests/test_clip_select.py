"""Offline selection contracts. No network, CLIP downloads or Genesis runtime."""
import base64
import io
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_official_index import fake_preview, fixture_asset

from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.vision_request import (
    ChatVisionClient,
    SelectionError,
    validate_selection,
)


class FakeEncoder:
    device = "cpu"
    metadata = {"model": clip.MODEL, "revision": clip.REVISION, "test": True}

    def __init__(self):
        self.image_calls = 0

    def encode_images(self, images):
        self.image_calls += 1
        return np.tile([[5., self.image_calls]], (len(images), 1))

    def encode_text(self, query):
        if len(query) > 64:
            raise SelectionError("query_too_long")
        return [[1., 0.]]


def response(status="selected", number=2):
    return json.dumps(dict(status=status, candidate_id=number, reason="类别合理",
                           visible_differences=["未确认米老鼠印花"]), ensure_ascii=False)


@pytest.fixture
def setup(tmp_path):
    assets = tmp_path / "official"
    clip.official.build(assets, downloader=fixture_asset, renderer=fake_preview)
    encoder = FakeEncoder()
    index = tmp_path / "clip"
    weights = tmp_path / "weights"
    clip.build_index(assets / "asset_index.json", index, weights_dir=weights, encoder=encoder)
    config = clip.LLMProviderConfig(endpoint="https://example.invalid/v1", model="gpt-4o",
                                    api_key="TEST_SECRET_NEVER_LOG")
    calls = []

    def vlm(messages):
        calls.append(messages)
        return response()

    kwargs = dict(clip_index=index / "index.json", query="印有米老鼠的杯子", encoder=encoder,
                  vlm_config=config, vlm=vlm, weights_dir=weights, cache_dir=tmp_path / "cache")
    return SimpleNamespace(root=tmp_path, assets=assets, index=index, encoder=encoder,
                           config=config, calls=calls, kwargs=kwargs)


def run(s, name="run", **changes):
    return clip.select(output_dir=s.root / name, **(s.kwargs | changes))


def read(s, name):
    return json.loads((s.root / "run" / name).read_text())


def test_normalize_and_cosine():
    assert np.allclose(clip.normalize([[3, 4], [-3, 4]]), [[.6, .8], [-.6, .8]])
    for bad in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], []):
        with pytest.raises(SelectionError):
            clip.normalize(bad)


def test_six_view_max_dedup_ties_and_all_scores():
    rows = [dict(asset_id=a, view=v) for a in ("b", "a") for v in clip.VIEWS]
    vectors = [[0, 1]] * 5 + [[1, 0]] + [[1, 0]] + [[0, 1]] * 5
    found = clip.retrieve(rows, vectors, [[1, 0]], 5)
    assert [c["asset_id"] for c in found] == ["a", "b"]
    assert len(found) == 2
    assert all(c["score"] == 1 and len(c["views"]) == 6 for c in found)
    assert found[1]["selected_views"][0]["view"] == "bottom"
    assert len({v["view"] for v in found[0]["selected_views"]}) == 2
    with pytest.raises(SelectionError, match="six_distinct"):
        clip.retrieve(rows + rows[:1], vectors + vectors[:1], [[1, 0]])


@pytest.mark.parametrize("k", [0, 6, -1, 1.0, True])
def test_k_invalid(setup, k):
    result = run(setup, top_k=k)
    assert result["status"] == "error" and result["vlm_calls"] == 0
    assert not (setup.root / "run/selected_asset.json").exists()


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_k_boundary(setup, k):
    result = run(setup, top_k=k, vlm=lambda _: response(number=1))
    assert result["status"] == "selected"
    assert read(setup, "retrieval_result.json")["actual_k"] == min(k, 4)


def test_nonfirst_binding_differences_and_neutral_images(setup):
    result = run(setup)
    assert result["status"] == "selected" and result["vlm_calls"] == 1
    retrieval = read(setup, "retrieval_result.json")
    binding = read(setup, "selected_asset.json")
    assert binding["asset_id"] != retrieval["candidates"][0]["asset_id"]
    assert binding["model_entrypoint"].endswith("model.xml")
    assert binding["visible_differences"] == ["未确认米老鼠印花"]
    assert len(binding["source_files"]) == 5
    messages = setup.calls[0]
    wire = json.dumps(messages)
    assert not any(a in wire for a in clip.official.ASSETS)
    assert "score" not in wire and "sha256" not in wire
    images = [v for v in messages[1]["content"] if v["type"] == "image_url"]
    assert len(images) == 6
    for part in images:
        image = Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",")[1])))
        assert image.size == (512, 512) and not image.info
    assert setup.encoder.image_calls == 4  # only build-index
    for path in (setup.root / "run").glob("*"):
        assert setup.config.api_key not in path.read_text()


@pytest.mark.parametrize("raw,expected", [
    (response("rejected", None), "rejected"),
    ("```json\n{}\n```", "error"),
    (response(number=99), "error"),
    (response(number=True), "error"),
    (response("rejected", 1), "error"),
    ('{"status":"selected","status":"rejected"}', "error"),
    ('{"status":"rejected","candidate_id":null,"reason":"ok",'
     '"visible_differences":[],"path":"evil.xml"}', "error"),
])
def test_rejection_and_invalid_no_fallback(setup, raw, expected):
    result = run(setup, vlm=lambda _: raw)
    assert result["status"] == expected and result["vlm_calls"] == 1
    assert not (setup.root / "run/selected_asset.json").exists()
    assert bool(list((setup.root / "cache").glob("*.json"))) == (expected == "rejected")


@pytest.mark.parametrize("error", [TimeoutError("TEST_SECRET_NEVER_LOG"),
                                  RuntimeError("TEST_SECRET_NEVER_LOG")])
def test_failure_no_retry_binding_or_cache(setup, error):
    calls = []

    def fail(_):
        calls.append(1)
        raise error

    result = run(setup, vlm=fail)
    assert result["status"] == "error" and result["vlm_calls"] == len(calls) == 1
    assert not (setup.root / "run/selected_asset.json").exists()
    assert not list((setup.root / "cache").glob("*.json"))
    assert setup.config.api_key not in json.dumps(result)


@pytest.mark.parametrize("status", ["selected", "rejected"])
def test_cached_selected_and_rejected_zero_requests(setup, status):
    number = 2 if status == "selected" else None
    first = run(setup, vlm=lambda _: response(status, number))
    second = run(setup, "again", vlm=lambda _: pytest.fail("cache called VLM"))
    assert first["status"] == second["status"] == status
    assert first["vlm_calls"] == 1 and second["vlm_calls"] == 0
    assert second["cache"]["hit"] and second["timings_s"]["vlm"] == 0


@pytest.mark.parametrize("change", ["query", "k", "endpoint", "temperature", "timeout", "prompt"])
def test_cache_key_changes(setup, monkeypatch, change):
    first = run(setup)
    kwargs = {}
    if change == "query":
        kwargs["query"] = "黄色杯子"
    elif change == "k":
        kwargs["top_k"] = 4
    elif change == "endpoint":
        kwargs["vlm_config"] = replace(setup.config, endpoint="https://other.invalid/v1")
    elif change == "temperature":
        kwargs["vlm_config"] = replace(setup.config, temperature=.5)
    elif change == "timeout":
        kwargs["timeout_s"] = 10
    else:
        monkeypatch.setattr(clip, "PROMPT_VERSION", "new")
    second = run(setup, "again", **kwargs)
    assert second["vlm_calls"] == 1 and not second["cache"]["hit"]
    assert second["cache"]["key"] != first["cache"]["key"]


@pytest.mark.parametrize("target", ["image", "vector", "source", "row"])
def test_tampering_before_cache_hit(setup, target):
    run(setup)
    if target == "image":
        path = setup.assets / "previews/apple_15/view_top.png"
    elif target == "source":
        path = setup.assets / "sources/apple_15/model.xml"
    elif target == "vector":
        path = setup.index / "vectors.npy"
    else:
        path = setup.index / "index.json"
        value = json.loads(path.read_text())
        value["rows"][0]["asset_id"] = "donut_0"
        clip.write_json(path, value)
        path = None
    if path:
        path.write_bytes(path.read_bytes() + b"tampering")
    result = run(setup, "again")
    assert result["status"] == "error" and result["vlm_calls"] == 0
    assert not (setup.root / "again/selected_asset.json").exists()


def test_cache_corruption_is_miss(setup):
    first = run(setup)
    cache = setup.root / "cache" / (first["cache"]["key"] + ".json")
    value = json.loads(cache.read_text())
    value["payload"]["selection"]["candidate_id"] = 99
    clip.write_json(cache, value)
    assert run(setup, "again")["vlm_calls"] == 1


def test_long_query_no_call(setup):
    result = run(setup, query="杯" * 1000)
    assert result["error"] == "query_too_long" and result["vlm_calls"] == 0
    assert not (setup.root / "run/selected_asset.json").exists()


def test_native_length_gate_does_not_truncate(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    captured = []

    def tokenizer(query, **kwargs):
        captured.append((query, kwargs))
        return {"input_ids": np.ones((1, 8), dtype=np.int64)}

    encoder = object.__new__(clip.ChineseClipEncoder)
    encoder.processor = SimpleNamespace(tokenizer=tokenizer)
    encoder.max_length = 7
    with pytest.raises(SelectionError, match="query_too_long"):
        encoder.encode_text("完整查询")
    assert captured == [("完整查询", {"truncation": False, "return_tensors": "pt"})]


def test_output_protection(setup):
    run(setup)
    before = (setup.root / "run/run_report.json").read_bytes()
    with pytest.raises(SelectionError, match="exists"):
        run(setup)
    assert (setup.root / "run/run_report.json").read_bytes() == before
    for destination in (setup.assets / "new", setup.index / "new", setup.root / "cache/new"):
        with pytest.raises(SelectionError, match="overlap|official_package"):
            clip.select(output_dir=destination, **setup.kwargs)
        assert not destination.exists()
    with pytest.raises(SelectionError, match="exists"):
        clip.build_index(setup.assets / "asset_index.json", setup.index,
                         weights_dir=setup.root / "weights", encoder=setup.encoder)


def test_wrong_preprocessing_no_request(setup):
    setup.encoder.metadata = dict(setup.encoder.metadata, changed=True)
    result = run(setup)
    assert result["error"] == "encoder_preprocessing_mismatch_rebuild_index"
    assert result["vlm_calls"] == 0


def test_provider_secret_echo_is_redacted(setup):
    result = run(setup, vlm=lambda _: "not json " + setup.config.api_key)
    assert result["status"] == "error"
    assert setup.config.api_key not in (setup.root / "run/vlm_selection.json").read_text()


def test_http_payload_no_retries_and_timeout(setup, monkeypatch):
    captured = []

    class Opener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            raise TimeoutError("secret")

    import urllib.request
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: Opener())
    config = replace(setup.config, timeout_s=60)
    with pytest.raises(SelectionError, match="vlm_timeout"):
        ChatVisionClient(config)([{"role": "user", "content": []}])
    assert len(captured) == 1 and captured[0][1] == 60
    payload = json.loads(captured[0][0].data)
    assert payload["model"] == "gpt-4o" and payload["response_format"] == {"type": "json_object"}
    assert captured[0][0].full_url.endswith("/v1/chat/completions")


def test_strict_validation_differences():
    assert validate_selection(response(), {1, 2, 3})["visible_differences"] == ["未确认米老鼠印花"]


def test_only_preview_passed_assets_enter_index(tmp_path):
    def partial_preview(record, output):
        if record["asset_id"] == "apple_15":
            raise ValueError("preview failed")
        return fake_preview(record, output)

    root = tmp_path / "official"
    assert clip.official.build(root, downloader=fixture_asset,
                               renderer=partial_preview)["status"] == "failed"
    result = clip.build_index(root / "asset_index.json", tmp_path / "clip",
                              weights_dir=tmp_path / "weights", encoder=FakeEncoder())
    assert len(result["rows"]) == 18 and len(result["assets"]) == 3
    assert "apple_15" not in {a["asset_id"] for a in result["assets"]}


def test_valid_new_asset_package_changes_cache(setup):
    first = run(setup)
    assets = setup.root / "official_new"
    clip.official.build(assets, downloader=fixture_asset, renderer=fake_preview)
    new_index = setup.root / "clip_new"
    clip.build_index(assets / "asset_index.json", new_index,
                     weights_dir=setup.root / "weights", encoder=FakeEncoder())
    second = run(setup, "again", clip_index=new_index / "index.json")
    assert second["status"] == "selected" and second["vlm_calls"] == 1
    assert second["cache"]["key"] != first["cache"]["key"]


def test_vector_norm_gate_even_with_updated_manifest(setup):
    path = setup.index / "vectors.npy"
    np.save(path, np.ones((24, 2), dtype=np.float32), allow_pickle=False)
    index = json.loads((setup.index / "index.json").read_text())
    index["vectors"] = clip.official.fingerprint(path, setup.index)
    clip.write_json(setup.index / "index.json", index)
    result = run(setup)
    assert result["error"] == "invalid_normalized_vectors" and result["vlm_calls"] == 0


def test_integrity_rechecked_after_vlm(setup):
    def mutate(_):
        path = setup.assets / "previews/apple_15/view_top.png"
        path.write_bytes(path.read_bytes() + b"changed during VLM")
        return response()

    result = run(setup, vlm=mutate)
    assert result["status"] == "error" and result["vlm_calls"] == 1
    assert not list((setup.root / "cache").glob("*.json"))
    assert not (setup.root / "run/selected_asset.json").exists()


def test_malformed_index_gets_error_report(setup):
    (setup.index / "index.json").write_text("not JSON")
    result = run(setup)
    assert result["status"] == "error" and result["vlm_calls"] == 0
    assert len(list((setup.root / "run").glob("*"))) == 4


def test_other_sealed_package_cannot_be_cache_or_weights(setup):
    other = setup.root / "another_official"
    clip.official.build(other, downloader=fixture_asset, renderer=fake_preview)
    for field in ("cache_dir", "weights_dir"):
        with pytest.raises(SelectionError, match="official_package"):
            run(setup, **{field: other})
    clip.official.verify_index(other)


def test_failed_binding_write_not_cached(setup, monkeypatch):
    original = clip.write_json

    def fail_binding(path, value):
        if path.name == "selected_asset.json":
            raise OSError("disk full")
        original(path, value)

    monkeypatch.setattr(clip, "write_json", fail_binding)
    result = run(setup)
    assert result["status"] == "error"
    assert not list((setup.root / "cache").glob("*.json"))


def test_model_reuse_function(monkeypatch):
    models = []

    def create(path):
        models.append(path)
        return FakeEncoder()

    clip.get_encoder.cache_clear()
    monkeypatch.setattr(clip, "ChineseClipEncoder", create)
    assert clip.get_encoder("fake_weights") is clip.get_encoder("fake_weights")
    assert models == ["fake_weights"]
    clip.get_encoder.cache_clear()

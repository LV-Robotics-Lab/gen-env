from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from scene_gen.builder import build_scene_package, verify_package
from scene_gen.catalog import load_catalog
from scene_gen.llm_provider import (
    MAX_LLM_CONFIG_BYTES,
    MAX_PARSE_CACHE_BYTES,
    LLMProviderConfig,
    LLMProviderError,
    LLMSceneProvider,
    _NoRedirectHandler,
    load_llm_provider_config,
)
from scene_gen.parser import parse_provider_payload, parse_rule_based, parse_with_provider
from scene_gen.schema import RelationType, SceneSpecError
from scene_gen.solver import solve_scene

ROOT = Path(__file__).resolve().parents[2]


class FakeTransport:
    def __init__(self, *responses: str | dict[str, Any] | Exception) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("unexpected provider request")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
        return response


def _config(
    tmp_path: Path,
    *,
    api_key: str = "unit-test-secret",
    max_attempts: int = 3,
) -> LLMProviderConfig:
    return LLMProviderConfig(
        endpoint="https://llm.invalid/v1",
        model="fake-model",
        api_key=api_key,
        max_attempts=max_attempts,
        cache_dir=tmp_path / "cache",
    )


def _objects(*items: dict[str, Any], ambiguities: list[str] | None = None) -> dict[str, Any]:
    return {"objects": list(items), "ambiguities": ambiguities or []}


def _relations(
    *,
    topology: list[dict[str, Any]],
    lateral: list[dict[str, Any]] | None = None,
    ambiguities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "topology": topology,
        "lateral": lateral or [],
        "ambiguities": ambiguities or [],
    }


def _on_table(object_id: str) -> dict[str, str]:
    return {"relation": "on_table", "source": object_id, "target": "table"}


def _object(
    category: str,
    *,
    index: int = 1,
    **fields: Any,
) -> dict[str, Any]:
    return {"object_id": f"{category}_{index}", "category": category, **fields}


def _all_on_table(*object_ids: str) -> list[dict[str, str]]:
    return [_on_table(object_id) for object_id in object_ids]


def test_fake_transport_runs_two_stages_and_cleans_attributes_and_regions(
    tmp_path: Path,
) -> None:
    request = "Place a red ceramic mug on the left side of the table and a blue plate on the table."
    transport = FakeTransport(
        _objects(
            {
                "object_id": "mug_1",
                "category": "mug",
                "color": " RED ",
                "material": "陶瓷",
                "region": "left",
            },
            {
                "object_id": "plate_1",
                "category": "plate",
                "color": " BLUE ",
                "material": "felt",
                # The request did not authorize the right region, so this must
                # not become an invented placement constraint.
                "region": "right",
            },
        ),
        _relations(topology=[_on_table("mug_1"), _on_table("plate_1")]),
    )
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    spec = parse_with_provider(provider, request=request, seed=17)

    assert len(transport.calls) == 2
    assert json.loads(transport.calls[0][1]) == {"request": request}
    assert json.loads(transport.calls[1][1]) == {
        "object_ids": ["mug_1", "plate_1"],
        "request": request,
    }
    by_id = {item.object_id: item for item in spec.objects}
    assert by_id["mug_1"].color == "red"
    assert by_id["mug_1"].material == "ceramic"
    assert by_id["mug_1"].region == "left"
    assert by_id["plate_1"].color == "blue"
    assert by_id["plate_1"].material is None
    assert by_id["plate_1"].region == "center"
    assert spec.request == request
    assert spec.seed == 17
    assert provider.evidence()["status"] == "pass"


def test_fenced_json_is_accepted_for_both_stages(tmp_path: Path) -> None:
    objects = _objects(
        {
            "object_id": "cup_1",
            "category": "cup",
            "color": None,
            "material": None,
            "region": "center",
        }
    )
    relations = _relations(topology=[_on_table("cup_1")])
    transport = FakeTransport(
        f"```json\n{json.dumps(objects)}\n```",
        f"```\n{json.dumps(relations)}\n```",
    )
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    spec = parse_with_provider(provider, request="Place a cup on the table.", seed=3)

    assert [item.object_id for item in spec.objects] == ["cup_1"]
    assert spec.relations[0].relation == RelationType.ON_TABLE


def test_stack_lateral_and_distance_relations_survive_bounded_cleaning(
    tmp_path: Path,
) -> None:
    request = "Stack a block on a plate, with the plate left of a bowl and at least 0.20 m away."
    transport = FakeTransport(
        _objects(
            {"object_id": "block_1", "category": "block"},
            {"object_id": "plate_1", "category": "plate"},
            {"object_id": "bowl_1", "category": "bowl"},
        ),
        _relations(
            topology=[
                {
                    "relation": "on_top_of",
                    "source": "block_1",
                    "target": "plate_1",
                },
                _on_table("plate_1"),
                _on_table("bowl_1"),
            ],
            lateral=[
                {"relation": "left_of", "source": "plate_1", "target": "bowl_1"},
                {
                    "relation": "distance_at_least",
                    "source": "plate_1",
                    "target": "bowl_1",
                    "min_distance_m": "0.20",
                },
            ],
        ),
    )
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    spec = parse_with_provider(provider, request=request, seed=5)

    relations = {(item.relation, item.source, item.target): item for item in spec.relations}
    assert (RelationType.ON_TOP_OF, "block_1", "plate_1") in relations
    assert (RelationType.LEFT_OF, "plate_1", "bowl_1") in relations
    distance = relations[(RelationType.DISTANCE_AT_LEAST, "plate_1", "bowl_1")]
    assert distance.min_distance_m == pytest.approx(0.2)


def test_invalid_stage_outputs_are_retried_with_bounded_feedback(tmp_path: Path) -> None:
    transport = FakeTransport(
        "not json",
        _objects({"object_id": "cup_1", "category": "cup"}),
        _relations(topology=[{"relation": "on_table", "source": "cup_1", "target": "floor"}]),
        _relations(topology=[_on_table("cup_1")]),
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=2),
        transport=transport,
    )

    spec = parse_with_provider(provider, request="Place a cup on the table.", seed=8)

    assert spec.objects[0].object_id == "cup_1"
    assert len(transport.calls) == 4
    object_retry = json.loads(transport.calls[1][1])
    relation_retry = json.loads(transport.calls[3][1])
    assert "strict JSON object" in object_retry["retry_feedback"]
    assert "on_table must target table" in relation_retry["retry_feedback"]
    assert provider.evidence()["stages"]["objects"]["attempts"] == 2
    assert provider.evidence()["stages"]["relations"]["attempts"] == 2


def test_exhausted_attempts_fail_without_a_cache_entry(tmp_path: Path) -> None:
    transport = FakeTransport("not json", RuntimeError("provider body must stay private"))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=2),
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=9)

    error = captured.value
    assert error.stage == "objects"
    assert error.attempts == 2
    assert error.failure_kind == "attempts_exhausted"
    assert error.safe_details()["failure_kind"] == "attempts_exhausted"
    assert "provider body must stay private" not in str(error)
    assert "provider body must stay private" not in json.dumps(provider.evidence())
    assert len(transport.calls) == 2
    assert not list((_config(tmp_path).cache_dir).glob("*.json"))
    assert provider.evidence()["status"] == "fail"


@pytest.mark.parametrize("ambiguous_stage", ["objects", "relations"])
def test_reported_ambiguity_fails_closed(
    tmp_path: Path,
    ambiguous_stage: str,
) -> None:
    object_response = _objects(
        {"object_id": "cup_1", "category": "cup"},
        ambiguities=["which cup is intended"] if ambiguous_stage == "objects" else [],
    )
    responses: list[dict[str, Any]] = [object_response]
    if ambiguous_stage == "relations":
        responses.append(
            _relations(
                topology=[_on_table("cup_1")],
                ambiguities=["support target is unclear"],
            )
        )
    transport = FakeTransport(*responses)
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup somewhere.", seed=10)

    assert captured.value.stage == ambiguous_stage
    assert captured.value.failure_kind == "ambiguous_request"
    assert captured.value.attempts == 1
    assert provider.evidence()["status"] == "fail"
    assert provider.evidence()["stage"] == ambiguous_stage
    assert len(transport.calls) == (1 if ambiguous_stage == "objects" else 2)
    assert not list((_config(tmp_path).cache_dir.glob("*.json")))


def test_cache_hit_skips_transport_and_cache_identity_excludes_api_key(tmp_path: Path) -> None:
    request = "Place a cup on the table."
    first_transport = FakeTransport(
        _objects({"object_id": "cup_1", "category": "cup"}),
        _relations(topology=[_on_table("cup_1")]),
    )
    first = LLMSceneProvider(
        _config(tmp_path, api_key="first-super-secret"),
        transport=first_transport,
    )
    first_payload = first.parse_scene(request=request, seed=11)
    first_evidence = first.evidence()
    key = first_evidence["cache"]["key"]
    cache_path = first.config.cache_dir / f"{key}.json"

    second_transport = FakeTransport()
    second = LLMSceneProvider(
        _config(tmp_path, api_key="second-super-secret"),
        transport=second_transport,
    )
    second_payload = second.parse_scene(request=request, seed=11)

    assert first_payload == second_payload
    assert second_transport.calls == []
    assert second.evidence() == first_evidence
    assert second.evidence()["cache"] == {"key": key}
    serialized_cache = cache_path.read_text(encoding="utf-8")
    for secret in ("first-super-secret", "second-super-secret"):
        assert secret not in key
        assert secret not in serialized_cache
        assert secret not in json.dumps(second.evidence(), sort_keys=True)


def test_direct_environment_configuration_is_complete_and_secret_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENENV_LLM_ENDPOINT", "https://example.invalid/v1/")
    monkeypatch.setenv("GENENV_LLM_API_KEY", "environment-secret")
    monkeypatch.setenv("GENENV_LLM_MODEL", "environment-model")
    monkeypatch.setenv("GENENV_LLM_API_MODE", "responses")
    monkeypatch.setenv("GENENV_LLM_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("GENENV_LLM_CACHE_DIR", str(tmp_path / "env-cache"))

    config = load_llm_provider_config()

    assert config.endpoint == "https://example.invalid/v1"
    assert config.model == "environment-model"
    assert config.api_mode == "responses"
    assert config.max_attempts == 4
    assert config.api_key == "environment-secret"
    assert "environment-secret" not in repr(config)
    assert "environment-secret" not in json.dumps(config.safe_dict(), sort_keys=True)


def test_yaml_configuration_reads_key_from_named_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "endpoint: https://example.invalid/v1",
                "model: yaml-model",
                "api_key_env: TEST_ONLY_LLM_KEY",
                "max_attempts: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_ONLY_LLM_KEY", "yaml-environment-secret")

    config = load_llm_provider_config(config_path)

    assert config.source == str(config_path.resolve())
    assert config.api_key == "yaml-environment-secret"
    assert config.max_attempts == 2
    assert "yaml-environment-secret" not in config.fingerprint()


def test_yaml_configuration_supports_inline_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://example.invalid/v1\n"
        "model: yaml-model\n"
        'api_key: "yaml-inline-secret"\n'
        "api_mode: responses\n"
        f"cache_dir: {tmp_path.as_posix()}/cache\n",
        encoding="utf-8",
    )

    config = load_llm_provider_config(config_path)

    assert config.source == str(config_path.resolve())
    assert config.api_key == "yaml-inline-secret"
    assert config.api_key_env == "direct"
    assert config.api_mode == "responses"
    serialized_safe_state = json.dumps(
        {
            "repr": repr(config),
            "safe_dict": config.safe_dict(),
            "fingerprint": config.fingerprint(),
        },
        sort_keys=True,
    )
    assert "yaml-inline-secret" not in serialized_safe_state
    provider = LLMSceneProvider(
        config,
        transport=FakeTransport(
            _objects(_object("cup")),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )

    provider.parse_scene(request="Place a cup on the table.", seed=301)

    serialized_runtime_state = json.dumps(provider.evidence(), sort_keys=True) + "".join(
        path.read_text(encoding="utf-8") for path in config.cache_dir.glob("*.json")
    )
    assert "yaml-inline-secret" not in serialized_runtime_state


def test_profiled_yaml_selects_only_the_requested_inline_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "active_profile: first\n"
        "profiles:\n"
        "  first:\n"
        "    endpoint: https://first.invalid/v1\n"
        "    model: first-model\n"
        '    api_key: "first-inline-secret"\n'
        "  second:\n"
        "    endpoint: https://second.invalid/v1\n"
        "    model: second-model\n"
        '    api_key: "second-inline-secret"\n',
        encoding="utf-8",
    )

    config = load_llm_provider_config(config_path, profile="second")

    assert config.profile == "second"
    assert config.endpoint == "https://second.invalid/v1"
    assert config.api_key == "second-inline-secret"
    safe = json.dumps(config.safe_dict(), sort_keys=True)
    assert "first-inline-secret" not in safe
    assert "second-inline-secret" not in safe


def test_yaml_configuration_rejects_api_key_and_api_key_env_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    monkeypatch.setenv("TEST_ONLY_LLM_KEY", "environment-secret")
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://example.invalid/v1\n"
        "model: yaml-model\n"
        'api_key: "inline-secret-must-not-leak"\n'
        "api_key_env: TEST_ONLY_LLM_KEY\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError, match="exactly one") as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"
    assert "inline-secret-must-not-leak" not in str(captured.value)
    assert "environment-secret" not in str(captured.value)


def test_malformed_active_profile_never_echoes_inline_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "malformed-profile.yaml"
    config_path.write_text(
        "active_profile:\n  api_key: active-profile-secret-must-not-leak\nprofiles: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    serialized_error = json.dumps(captured.value.safe_details(), sort_keys=True)
    assert captured.value.failure_kind == "invalid_configuration"
    assert "active-profile-secret-must-not-leak" not in serialized_error


def test_unknown_requested_profile_never_echoes_its_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "profiles.yaml"
    config_path.write_text(
        "active_profile: safe\n"
        "profiles:\n"
        "  safe:\n"
        "    endpoint: https://example.invalid/v1\n"
        "    model: safe-model\n"
        "    api_key: safe-inline-secret\n",
        encoding="utf-8",
    )
    supplied_profile = "secret-looking-profile-name"

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path, profile=supplied_profile)

    assert supplied_profile not in json.dumps(captured.value.safe_details(), sort_keys=True)


def test_complete_direct_environment_precedes_explicit_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://yaml.invalid/v1\nmodel: yaml-model\napi_key: yaml-inline-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GENENV_LLM_ENDPOINT", "https://env.invalid/v1")
    monkeypatch.setenv("GENENV_LLM_API_KEY", "environment-secret")
    monkeypatch.setenv("GENENV_LLM_MODEL", "environment-model")

    config = load_llm_provider_config(config_path)

    assert config.source == "environment"
    assert config.endpoint == "https://env.invalid/v1"
    assert config.model == "environment-model"
    assert config.api_key == "environment-secret"


def test_partial_direct_environment_blocks_explicit_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://yaml.invalid/v1\n"
        "model: yaml-model\n"
        "api_key: yaml-inline-secret-must-not-leak\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GENENV_LLM_ENDPOINT", "https://env.invalid/v1")
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)

    with pytest.raises(LLMProviderError, match="incomplete direct LLM environment") as captured:
        load_llm_provider_config(config_path)

    assert captured.value.failure_kind == "invalid_configuration"
    assert "yaml-inline-secret-must-not-leak" not in str(captured.value)


@pytest.mark.parametrize(
    ("numeric_field", "yaml_value", "expected"),
    [
        ("timeout_s", "[]", "expected a number"),
        ("timeout_s", ".nan", "finite number"),
        ("max_attempts", "{}", "expected an integer"),
        ("max_attempts", "3.5", "expected an integer"),
        ("max_attempts", "true", "expected an integer"),
    ],
)
def test_yaml_configuration_rejects_malformed_numeric_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    numeric_field: str,
    yaml_value: str,
    expected: str,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    monkeypatch.setenv("TEST_ONLY_LLM_KEY", "yaml-environment-secret")
    config_path = tmp_path / f"bad-{numeric_field}.yaml"
    config_path.write_text(
        "endpoint: https://example.invalid/v1\n"
        "model: yaml-model\n"
        "api_key_env: TEST_ONLY_LLM_KEY\n"
        f"{numeric_field}: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError, match=expected) as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"


@pytest.mark.parametrize(
    "secret_field",
    ["apikey", "key", "secret", "token", "api_key_file"],
)
def test_yaml_configuration_rejects_unsupported_credential_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret_field: str,
) -> None:
    config_path = tmp_path / f"{secret_field}.yaml"
    config_path.write_text(
        "endpoint: https://example.invalid/v1\n"
        "model: yaml-model\n"
        f"{secret_field}: must-not-be-loaded\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "inline_secret_forbidden"


def test_provider_payload_preserves_compatible_envelope_but_rejects_changes() -> None:
    request = "Place a cup on the table."
    seed = 23
    payload = {
        "objects": [{"object_id": "cup_1", "category": "cup"}],
        "relations": [_on_table("cup_1")],
    }

    bound = parse_provider_payload(payload, request=request, seed=seed)
    assert bound.request == request
    assert bound.seed == seed
    assert bound.language == "en"
    assert bound.scene_id != "provider_scene"
    assert (
        parse_provider_payload(bound.canonical_dict(), request=request, seed=seed).digest()
        == bound.digest()
    )
    assert (
        parse_provider_payload(payload | {"workspace": {}}, request=request, seed=seed).digest()
        == bound.digest()
    )

    for injected in (
        {"scene_id": "provider_scene"},
        {"language": "zh"},
        {"workspace": {"table_height_m": 0.8}},
    ):
        with pytest.raises(SceneSpecError, match="changed caller-owned field"):
            parse_provider_payload(payload | injected, request=request, seed=seed)

    with pytest.raises(SceneSpecError, match="changed the user request"):
        parse_provider_payload(
            payload | {"request": "Use a different request."},
            request=request,
            seed=seed,
        )
    with pytest.raises(SceneSpecError, match="changed the deterministic seed"):
        parse_provider_payload(payload | {"seed": seed + 1}, request=request, seed=seed)


def test_llm_evidence_is_hash_bound_as_an_additional_package_file(tmp_path: Path) -> None:
    catalog = load_catalog(ROOT / "tests" / "fixtures" / "asset_catalog.json")
    spec = parse_rule_based("Place a can on top of a plate.", seed=29)
    resolved = solve_scene(spec, catalog)
    package = tmp_path / "package"
    package.mkdir()
    evidence_path = package / "llm_parse_evidence.json"
    evidence_path.write_text(
        json.dumps({"schema_version": "robotwin.llm_scene_extraction.v1", "status": "pass"}) + "\n",
        encoding="utf-8",
    )

    manifest = build_scene_package(
        spec,
        resolved,
        package,
        additional_files=("llm_parse_evidence.json",),
    )

    records = {item["path"]: item for item in manifest["files"]}
    assert "llm_parse_evidence.json" in records
    assert verify_package(package)["status"] == "pass"
    evidence_path.write_text('{"status":"tampered"}\n', encoding="utf-8")
    report = verify_package(package)
    assert report["status"] == "fail"
    assert any(
        item["path"] == "llm_parse_evidence.json" and not item["pass"] for item in report["checks"]
    )

    with pytest.raises(ValueError, match="escapes package root"):
        build_scene_package(
            spec,
            resolved,
            tmp_path / "escape-package",
            additional_files=("../escape",),
        )


@pytest.mark.parametrize(
    "lateral",
    [
        [],
        [{"relation": "left_of", "source": "bowl_1", "target": "cup_1"}],
    ],
)
def test_explicit_relation_omission_or_reversal_fails_closed(
    tmp_path: Path,
    lateral: list[dict[str, str]],
) -> None:
    transport = FakeTransport(
        _objects(
            {"object_id": "cup_1", "category": "cup"},
            {"object_id": "bowl_1", "category": "bowl"},
        ),
        _relations(
            topology=[_on_table("cup_1"), _on_table("bowl_1")],
            lateral=lateral,
        ),
    )
    provider = LLMSceneProvider(_config(tmp_path, max_attempts=1), transport=transport)

    with pytest.raises(LLMProviderError, match="missing or reversed") as captured:
        provider.parse_scene(request="A cup is left of a bowl.", seed=31)

    assert captured.value.stage == "relations"
    assert len(transport.calls) == 2
    assert provider.evidence()["status"] == "fail"


def test_hallucinated_known_object_fails_closed(tmp_path: Path) -> None:
    transport = FakeTransport(
        _objects(
            {"object_id": "cup_1", "category": "cup"},
            {"object_id": "bowl_1", "category": "bowl"},
        )
    )
    provider = LLMSceneProvider(_config(tmp_path, max_attempts=1), transport=transport)

    with pytest.raises(LLMProviderError, match="object count for 'bowl'") as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=32)

    assert captured.value.stage == "objects"
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {
            "objects": [
                {
                    "object_id": "cup_1",
                    "category": "cup",
                    "qpos": [0.5],
                }
            ],
            "ambiguities": [],
        },
        ('{"objects":[{"object_id":"cup_1","category":"cup"}],"objects":[],"ambiguities":[]}'),
        ('{"objects":[{"object_id":"cup_1","category":"cup","region":NaN}],"ambiguities":[]}'),
    ],
)
def test_unknown_fields_duplicate_keys_and_nonfinite_json_are_rejected(
    tmp_path: Path,
    response: dict[str, Any] | str,
) -> None:
    transport = FakeTransport(response)
    provider = LLMSceneProvider(_config(tmp_path, max_attempts=1), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=33)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"


@pytest.mark.parametrize(
    ("user_text", "seed"),
    [
        ("  ", 0),
        ("x" * 2001, 0),
        ("Place a cup on the table.", -1),
        ("Place a cup on the table.", True),
    ],
)
def test_invalid_request_or_seed_never_calls_transport(
    tmp_path: Path,
    user_text: str,
    seed: int,
) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(SceneSpecError):
        parse_with_provider(provider, request=user_text, seed=seed)

    assert transport.calls == []


def test_unsupported_negation_fails_before_transport(tmp_path: Path) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        parse_with_provider(
            provider,
            request="Place a cup on the table without a bowl.",
            seed=34,
        )

    assert captured.value.stage == "semantic_precheck"
    assert captured.value.failure_kind == "unsupported_request"
    assert transport.calls == []


def test_chinese_request_uses_english_canonical_category_and_prompt_contract(
    tmp_path: Path,
) -> None:
    request = "把一个杯子放在桌面上。"
    transport = FakeTransport(
        _objects({"object_id": "cup_1", "category": "cup"}),
        _relations(topology=[_on_table("cup_1")]),
    )
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    spec = parse_with_provider(provider, request=request, seed=35)

    assert spec.objects[0].category == "cup"
    objects_prompt = transport.calls[0][0]
    assert "support table" in objects_prompt
    assert "Translate Chinese category words" in objects_prompt


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.invalid/v1",
        "ftp://example.invalid/v1",
        "https://user:password@example.invalid/v1",
        "https://example.invalid/v1?redirect=1",
    ],
)
def test_provider_config_rejects_unsafe_endpoints(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(LLMProviderError, match="endpoint"):
        LLMProviderConfig(
            endpoint=endpoint,
            model="fake-model",
            api_key="secret",
            cache_dir=tmp_path / "cache",
        )

    local = LLMProviderConfig(
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        api_key="local-secret",
        cache_dir=tmp_path / "local-cache",
    )
    assert local.endpoint.startswith("http://127.0.0.1:")


def test_redirect_handler_never_forwards_authorization() -> None:
    handler = _NoRedirectHandler()

    assert (
        handler.redirect_request(
            None,
            None,
            302,
            "Found",
            {"Location": "https://attacker.invalid/"},
            "https://attacker.invalid/",
        )
        is None
    )


def test_credential_scope_changes_fingerprint_without_exposing_secret(tmp_path: Path) -> None:
    first = LLMProviderConfig(
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_key="first-secret",
        api_key_env="FIRST_ACCOUNT_KEY",
        cache_dir=tmp_path / "cache",
    )
    second = LLMProviderConfig(
        endpoint="https://example.invalid/v1",
        model="fake-model",
        api_key="second-secret",
        api_key_env="SECOND_ACCOUNT_KEY",
        cache_dir=tmp_path / "cache",
    )

    assert first.fingerprint() != second.fingerprint()
    safe = json.dumps([first.safe_dict(), second.safe_dict()], sort_keys=True)
    assert "first-secret" not in safe
    assert "second-secret" not in safe


def test_cache_with_tampered_evidence_is_ignored(tmp_path: Path) -> None:
    request = "Place a cup on the table."
    config = _config(tmp_path)
    first = LLMSceneProvider(
        config,
        transport=FakeTransport(
            _objects({"object_id": "cup_1", "category": "cup"}),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )
    first.parse_scene(request=request, seed=36)
    key = first.evidence()["cache"]["key"]
    cache_path = config.cache_dir / f"{key}.json"
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["evidence"]["prompt_hashes"]["llm_objects.md"] = "0" * 64
    cache_path.write_text(json.dumps(record), encoding="utf-8")

    replacement_transport = FakeTransport(
        _objects({"object_id": "cup_1", "category": "cup"}),
        _relations(topology=[_on_table("cup_1")]),
    )
    second = LLMSceneProvider(config, transport=replacement_transport)

    second.parse_scene(request=request, seed=36)

    assert len(replacement_transport.calls) == 2


def _clean_cli_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GENENV_LLM_")
    }
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def test_llm_cli_missing_configuration_writes_structured_failure(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "script" / "generate_scene.py"),
            "--provider",
            "llm",
            "--llm-config",
            str(tmp_path / "missing.yaml"),
            "--prompt",
            "Place a cup on the table.",
            "--seed",
            "37",
            "--asset-catalog",
            str(ROOT / "tests" / "fixtures" / "asset_catalog.json"),
            "--out-root",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=_clean_cli_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    reports = list((tmp_path / "output" / "_failures").glob("*/failure_report.json"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["stage"] == "llm_scene_extraction"
    assert report["provider"] == "llm"
    assert report["details"][0]["failure_kind"] == "missing_configuration"


def test_llm_cli_catalog_miss_is_a_grounding_failure_not_parse_failure(
    tmp_path: Path,
) -> None:
    request = "Place a spoon on the table."
    cache_dir = tmp_path / "cache"
    config = LLMProviderConfig(
        endpoint="https://llm.invalid/v1",
        model="fake-model",
        api_key="cache-builder-secret",
        api_key_env="CLI_TEST_LLM_KEY",
        cache_dir=cache_dir,
    )
    provider = LLMSceneProvider(
        config,
        transport=FakeTransport(
            _objects({"object_id": "spoon_1", "category": "spoon"}),
            _relations(topology=[_on_table("spoon_1")]),
        ),
    )
    provider.parse_scene(request=request, seed=38)
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://llm.invalid/v1\n"
        "model: fake-model\n"
        "api_key_env: CLI_TEST_LLM_KEY\n"
        f"cache_dir: {json.dumps(str(cache_dir))}\n",
        encoding="utf-8",
    )
    environment = _clean_cli_environment()
    environment["CLI_TEST_LLM_KEY"] = "runtime-secret"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "script" / "generate_scene.py"),
            "--provider",
            "llm",
            "--llm-config",
            str(config_path),
            "--prompt",
            request,
            "--seed",
            "38",
            "--asset-catalog",
            str(ROOT / "tests" / "fixtures" / "asset_catalog.json"),
            "--out-root",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    reports = list((tmp_path / "output" / "_failures").glob("*/failure_report.json"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["stage"] == "asset_grounding"
    assert report["provider"] == "llm"
    assert report["llm_extraction"]["status"] == "pass"
    assert report["llm_extraction"]["cache"] == {"key": provider.evidence()["cache"]["key"]}
    evidence_paths = list((tmp_path / "output").glob("*/llm_parse_evidence.json"))
    assert len(evidence_paths) == 1
    scene_dir = evidence_paths[0].parent
    assert json.loads(evidence_paths[0].read_text())["status"] == "pass"
    assert (scene_dir / "objects" / "spoon_1.json").is_file()
    assert (scene_dir / "relations.json").is_file()
    resolution = json.loads((scene_dir / "asset_resolution.json").read_text())
    assert resolution["objects"][0]["status"] == "missing"
    assert not (scene_dir / "resolved_scene.json").exists()


def test_llm_cli_success_hash_binds_stable_parse_evidence(tmp_path: Path) -> None:
    request = "Place a can on top of a plate."
    seed = 39
    cache_dir = tmp_path / "cache"
    config = LLMProviderConfig(
        endpoint="https://llm.invalid/v1",
        model="fake-model",
        api_key="cache-builder-secret",
        api_key_env="CLI_TEST_LLM_KEY",
        cache_dir=cache_dir,
    )
    provider = LLMSceneProvider(
        config,
        transport=FakeTransport(
            _objects(
                {"object_id": "can_1", "category": "can"},
                {"object_id": "plate_1", "category": "plate"},
            ),
            _relations(
                topology=[
                    {
                        "relation": "on_top_of",
                        "source": "can_1",
                        "target": "plate_1",
                    },
                    _on_table("plate_1"),
                ]
            ),
        ),
    )
    provider.parse_scene(request=request, seed=seed)
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "endpoint: https://llm.invalid/v1\n"
        "model: fake-model\n"
        "api_key_env: CLI_TEST_LLM_KEY\n"
        f"cache_dir: {json.dumps(str(cache_dir))}\n",
        encoding="utf-8",
    )
    environment = _clean_cli_environment()
    environment["CLI_TEST_LLM_KEY"] = "runtime-secret"
    output_root = tmp_path / "output"
    command = [
        sys.executable,
        str(ROOT / "script" / "generate_scene.py"),
        "--provider",
        "llm",
        "--llm-config",
        str(config_path),
        "--prompt",
        request,
        "--seed",
        str(seed),
        "--asset-catalog",
        str(ROOT / "tests" / "fixtures" / "asset_catalog.json"),
        "--out-root",
        str(output_root),
    ]

    first = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr or first.stdout
    scene_dirs = [item for item in output_root.iterdir() if item.name != "_failures"]
    assert len(scene_dirs) == 1
    scene_dir = scene_dirs[0]
    manifest_path = scene_dir / "package_manifest.json"
    evidence_path = scene_dir / "llm_parse_evidence.json"
    first_manifest = manifest_path.read_bytes()
    first_evidence = evidence_path.read_bytes()
    manifest = json.loads(first_manifest)
    records = {item["path"]: item for item in manifest["files"]}
    assert "llm_parse_evidence.json" in records
    assert verify_package(scene_dir)["status"] == "pass"
    evidence = json.loads(first_evidence)
    assert evidence["status"] == "pass"
    assert evidence["cache"] == {"key": provider.evidence()["cache"]["key"]}

    second = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr or second.stdout
    assert manifest_path.read_bytes() == first_manifest
    assert evidence_path.read_bytes() == first_evidence


_HTTPResponse = tuple[int, dict[str, str], dict[str, Any] | str | bytes]
_HTTPResponder = Callable[[dict[str, Any]], _HTTPResponse]


class _LoopbackHandler(BaseHTTPRequestHandler):
    def _handle_request(self) -> None:
        server = self.server
        assert isinstance(server, _LoopbackServer)
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b""
        try:
            body: Any = json.loads(raw_body) if raw_body else None
        except json.JSONDecodeError:
            body = raw_body
        record = {
            "method": self.command,
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        }
        server.requests.append(record)
        status, headers, response = server.responder(record)
        if isinstance(response, dict):
            encoded = json.dumps(response).encode("utf-8")
            headers = {"Content-Type": "application/json", **headers}
        elif isinstance(response, str):
            encoded = response.encode("utf-8")
        else:
            encoded = response
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        self._handle_request()

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, responder: _HTTPResponder) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        super().__init__(("127.0.0.1", 0), _LoopbackHandler)

    @property
    def endpoint(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


def _start_loopback_server(
    responder: _HTTPResponder,
) -> tuple[_LoopbackServer, threading.Thread]:
    server = _LoopbackServer(responder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_loopback_server(server: _LoopbackServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    ("api_mode", "expected_path"),
    [("chat", "/v1/chat/completions"), ("responses", "/v1/responses")],
)
def test_loopback_http_transport_request_and_response_envelopes(
    tmp_path: Path,
    api_mode: str,
    expected_path: str,
) -> None:
    def respond(record: dict[str, Any]) -> _HTTPResponse:
        body = record["body"]
        message_key = "messages" if api_mode == "chat" else "input"
        user_payload = json.loads(body[message_key][-1]["content"])
        stage_payload = (
            _relations(topology=[_on_table("cup_1")])
            if "object_ids" in user_payload
            else _objects({"object_id": "cup_1", "category": "cup"})
        )
        serialized = json.dumps(stage_payload)
        if api_mode == "chat":
            envelope = {"choices": [{"message": {"content": serialized}}]}
        elif "object_ids" in user_payload:
            envelope = {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": serialized}],
                    }
                ]
            }
        else:
            envelope = {"output_text": serialized}
        return 200, {}, envelope

    server, thread = _start_loopback_server(respond)
    try:
        config = LLMProviderConfig(
            endpoint=f"{server.endpoint}/v1",
            model="loopback-model",
            api_key="loopback-secret",
            api_mode=api_mode,
            max_attempts=1,
            cache_dir=tmp_path / api_mode,
        )
        provider = LLMSceneProvider(config)

        spec = parse_with_provider(
            provider,
            request="Place a cup on the table.",
            seed=41,
        )
    finally:
        _stop_loopback_server(server, thread)

    assert spec.objects[0].object_id == "cup_1"
    assert len(server.requests) == 2
    for index, record in enumerate(server.requests):
        assert record["method"] == "POST"
        assert record["path"] == expected_path
        assert record["headers"]["authorization"] == "Bearer loopback-secret"
        assert record["headers"]["content-type"] == "application/json"
        body = record["body"]
        assert body["model"] == "loopback-model"
        message_key = "messages" if api_mode == "chat" else "input"
        assert [item["role"] for item in body[message_key]] == ["system", "user"]
        user_payload = json.loads(body[message_key][-1]["content"])
        assert user_payload["request"] == "Place a cup on the table."
        assert ("object_ids" in user_payload) is (index == 1)
        if api_mode == "chat":
            assert set(body) == {"model", "messages", "stream"}
            assert body["stream"] is False
        else:
            assert set(body) == {"input", "max_output_tokens", "model", "store"}
            assert body["max_output_tokens"] == 4096
            assert body["store"] is False


def test_loopback_http_transport_rejects_redirect_without_forwarding_bearer(
    tmp_path: Path,
) -> None:
    attacker, attacker_thread = _start_loopback_server(
        lambda record: (200, {}, {"choices": [{"message": {"content": "{}"}}]})
    )

    def redirect(_: dict[str, Any]) -> _HTTPResponse:
        return 302, {"Location": f"{attacker.endpoint}/capture"}, b""

    origin, origin_thread = _start_loopback_server(redirect)
    try:
        provider = LLMSceneProvider(
            LLMProviderConfig(
                endpoint=f"{origin.endpoint}/v1",
                model="loopback-model",
                api_key="redirect-secret",
                max_attempts=1,
                cache_dir=tmp_path / "redirect-cache",
            )
        )

        with pytest.raises(LLMProviderError, match="HTTP 302") as captured:
            provider.parse_scene(request="Place a cup on the table.", seed=42)
    finally:
        _stop_loopback_server(origin, origin_thread)
        _stop_loopback_server(attacker, attacker_thread)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert len(origin.requests) == 1
    assert origin.requests[0]["headers"]["authorization"] == "Bearer redirect-secret"
    assert attacker.requests == []


def _articulation(state: str) -> dict[str, Any]:
    return {
        "state": state,
        "open_fraction": 1.0 if state == "open" else 0.0,
        "joint_selector": "all_movable",
    }


def test_explicit_attributes_are_bound_to_their_exact_objects(tmp_path: Path) -> None:
    request = "Place a red ceramic cup on the table. Place a blue metal bowl on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup", color="red", material="ceramic"),
                _object("bowl", color="blue", material="metal"),
            ),
            _relations(topology=_all_on_table("cup_1", "bowl_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=101)

    by_id = {item.object_id: item for item in spec.objects}
    assert (by_id["cup_1"].color, by_id["cup_1"].material) == ("red", "ceramic")
    assert (by_id["bowl_1"].color, by_id["bowl_1"].material) == ("blue", "metal")


@pytest.mark.parametrize(
    "candidate_objects",
    [
        [
            _object("cup", color=None, material=None),
            _object("bowl", color="blue", material="metal"),
        ],
        [
            _object("cup", color="blue", material="metal"),
            _object("bowl", color="red", material="ceramic"),
        ],
    ],
    ids=("omitted", "swapped"),
)
def test_explicit_attribute_omission_or_swap_fails_closed(
    tmp_path: Path,
    candidate_objects: list[dict[str, Any]],
) -> None:
    transport = FakeTransport(_objects(*candidate_objects))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(
            request="Place a red ceramic cup on the table. Place a blue metal bowl on the table.",
            seed=102,
        )

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert len(transport.calls) == 1


def test_postnominal_color_and_material_are_bound(tmp_path: Path) -> None:
    request = "Place a cup that is red and made of ceramic on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup", color="red", material="ceramic")),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=103)

    assert spec.objects[0].color == "red"
    assert spec.objects[0].material == "ceramic"


def test_articulation_states_are_bound_to_their_exact_objects(tmp_path: Path) -> None:
    request = "Place an open cabinet and a closed box on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cabinet", articulation=_articulation("open")),
                _object("box", articulation=_articulation("closed")),
            ),
            _relations(topology=_all_on_table("cabinet_1", "box_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=104)

    by_id = {item.object_id: item for item in spec.objects}
    assert by_id["cabinet_1"].articulation is not None
    assert by_id["cabinet_1"].articulation.state == "open"
    assert by_id["box_1"].articulation is not None
    assert by_id["box_1"].articulation.state == "closed"


def test_swapped_articulation_states_fail_closed(tmp_path: Path) -> None:
    transport = FakeTransport(
        _objects(
            _object("cabinet", articulation=_articulation("closed")),
            _object("box", articulation=_articulation("open")),
        )
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="articulation.*contradicts") as captured:
        provider.parse_scene(
            request="Place an open cabinet and a closed box on the table.",
            seed=105,
        )

    assert captured.value.stage == "objects"
    assert len(transport.calls) == 1


def test_preposed_table_regions_are_bound_without_becoming_relations(tmp_path: Path) -> None:
    request = (
        "On the left side of the table, place a cup; on the right side of the table, place a bowl."
    )
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup", region="left"),
                _object("bowl", region="right"),
            ),
            _relations(topology=_all_on_table("cup_1", "bowl_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=106)

    assert [(item.object_id, item.region) for item in spec.objects] == [
        ("cup_1", "left"),
        ("bowl_1", "right"),
    ]
    assert {item.relation for item in spec.relations} == {RelationType.ON_TABLE}


def test_swapped_preposed_table_regions_fail_closed(tmp_path: Path) -> None:
    transport = FakeTransport(
        _objects(
            _object("cup", region="right"),
            _object("bowl", region="left"),
        )
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="region.*contradicts") as captured:
        provider.parse_scene(
            request=(
                "On the left side of the table, place a cup; "
                "on the right side of the table, place a bowl."
            ),
            seed=107,
        )

    assert captured.value.stage == "objects"


def test_chinese_table_regions_do_not_authorize_lateral_relations(tmp_path: Path) -> None:
    request = "把一个杯子放在桌面左侧，把一个碗放在桌面右侧。"
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup", region="left"),
                _object("bowl", region="right"),
            ),
            _relations(topology=_all_on_table("cup_1", "bowl_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=108)

    assert {item.relation for item in spec.relations} == {RelationType.ON_TABLE}


def test_open_vocabulary_object_can_coexist_with_known_object(tmp_path: Path) -> None:
    request = "Place a cup and a fork on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup"), _object("fork")),
            _relations(topology=_all_on_table("cup_1", "fork_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=109)

    assert [item.object_id for item in spec.objects] == ["cup_1", "fork_1"]


def test_open_vocabulary_object_omission_fails_closed(tmp_path: Path) -> None:
    transport = FakeTransport(_objects(_object("cup")))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="object count for 'fork'") as captured:
        provider.parse_scene(request="Place a cup and a fork on the table.", seed=110)

    assert captured.value.stage == "objects"


def test_merged_open_vocabulary_category_fails_closed(tmp_path: Path) -> None:
    transport = FakeTransport(_objects(_object("cup_fork")))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="object count") as captured:
        provider.parse_scene(request="Place a cup and a fork on the table.", seed=111)

    assert captured.value.stage == "objects"


def test_color_word_used_as_noun_and_second_open_category_are_preserved(
    tmp_path: Path,
) -> None:
    request = "Place an orange and a banana on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("orange"), _object("banana")),
            _relations(topology=_all_on_table("orange_1", "banana_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=112)

    assert [(item.category, item.color) for item in spec.objects] == [
        ("orange", None),
        ("banana", None),
    ]


def test_multiword_open_vocabulary_category_uses_snake_case(tmp_path: Path) -> None:
    request = "Place a wine glass on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("wine_glass")),
            _relations(topology=[_on_table("wine_glass_1")]),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=113)

    assert spec.objects[0].category == "wine_glass"


def test_incidental_open_vocabulary_word_does_not_authorize_an_object(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(_objects(_object("cup"), _object("banana")))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="object count for 'banana'") as captured:
        provider.parse_scene(
            request="Place a cup on the table; the instruction label says banana.",
            seed=114,
        )

    assert captured.value.stage == "objects"


@pytest.mark.parametrize(
    "reserved_category",
    ["robot_arm", "table_surface", "workspace_area", "world_frame"],
)
def test_reserved_context_compounds_cannot_become_scene_objects(
    tmp_path: Path,
    reserved_category: str,
) -> None:
    transport = FakeTransport(_objects(_object("cup"), _object(reserved_category)))
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="reserved context category") as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=115)

    assert captured.value.stage == "objects"
    assert len(transport.calls) == 1


def test_same_pair_can_carry_multiple_explicit_relation_cues(tmp_path: Path) -> None:
    request = "A cup is left of and near a bowl, while a plate is on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl"), _object("plate")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "plate_1"),
                lateral=[
                    {"relation": "left_of", "source": "cup_1", "target": "bowl_1"},
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    },
                ],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=116)

    relations = {(item.relation, item.source, item.target) for item in spec.relations}
    assert (RelationType.LEFT_OF, "cup_1", "bowl_1") in relations
    assert (RelationType.NEAR, "cup_1", "bowl_1") in relations


def test_preposed_direction_preserves_source_and_target(tmp_path: Path) -> None:
    request = "To the left of a bowl, place a cup."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("bowl"), _object("cup")),
            _relations(
                topology=_all_on_table("bowl_1", "cup_1"),
                lateral=[{"relation": "left_of", "source": "cup_1", "target": "bowl_1"}],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=117)

    assert any(
        item.relation == RelationType.LEFT_OF and item.source == "cup_1" and item.target == "bowl_1"
        for item in spec.relations
    )


def test_reversed_preposed_direction_fails_closed(tmp_path: Path) -> None:
    transport = FakeTransport(
        _objects(_object("bowl"), _object("cup")),
        _relations(
            topology=_all_on_table("bowl_1", "cup_1"),
            lateral=[{"relation": "left_of", "source": "bowl_1", "target": "cup_1"}],
        ),
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="missing or reversed") as captured:
        provider.parse_scene(
            request="To the left of a bowl, place a cup.",
            seed=118,
        )

    assert captured.value.stage == "relations"


def test_oxford_comma_sources_all_bind_to_one_container(tmp_path: Path) -> None:
    request = "Place a cup, a can, and an apple inside a basket."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup"),
                _object("can"),
                _object("apple"),
                _object("basket"),
            ),
            _relations(
                topology=[
                    {
                        "relation": "inside",
                        "source": source,
                        "target": "basket_1",
                    }
                    for source in ("cup_1", "can_1", "apple_1")
                ]
                + [_on_table("basket_1")]
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=119)

    inside_sources = {
        item.source
        for item in spec.relations
        if item.relation == RelationType.INSIDE and item.target == "basket_1"
    }
    assert inside_sources == {"cup_1", "can_1", "apple_1"}


@pytest.mark.parametrize(
    ("user_text", "relation", "source_category", "target_category"),
    [
        ("A cup is in a basket.", "inside", "cup", "basket"),
        ("A cup is atop a plate.", "on_top_of", "cup", "plate"),
    ],
)
def test_explicit_in_and_atop_require_nested_topology(
    tmp_path: Path,
    user_text: str,
    relation: str,
    source_category: str,
    target_category: str,
) -> None:
    source = f"{source_category}_1"
    target = f"{target_category}_1"
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object(source_category), _object(target_category)),
            _relations(
                topology=[
                    {"relation": relation, "source": source, "target": target},
                    _on_table(target),
                ]
            ),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=120)

    assert any(
        item.relation.value == relation and item.source == source and item.target == target
        for item in spec.relations
    )


@pytest.mark.parametrize(
    "user_text",
    [
        "A cup is in a basket.",
        "A cup is atop a plate.",
    ],
)
def test_explicit_in_or_atop_cannot_be_replaced_with_on_table(
    tmp_path: Path,
    user_text: str,
) -> None:
    target_category = "basket" if "basket" in user_text else "plate"
    transport = FakeTransport(
        _objects(_object("cup"), _object(target_category)),
        _relations(topology=_all_on_table("cup_1", f"{target_category}_1")),
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="missing or reversed") as captured:
        provider.parse_scene(request=user_text, seed=121)

    assert captured.value.stage == "relations"


@pytest.mark.parametrize(
    ("user_text", "lateral", "expected_relation", "expected_distance"),
    [
        (
            "Place a cup within 0.20 m of a bowl.",
            [
                {
                    "relation": "near",
                    "source": "cup_1",
                    "target": "bowl_1",
                    "max_distance_m": 0.20,
                }
            ],
            RelationType.NEAR,
            0.20,
        ),
        (
            "Place a cup and a bowl at least 0.30 m apart.",
            [
                {
                    "relation": "distance_at_least",
                    "source": "cup_1",
                    "target": "bowl_1",
                    "min_distance_m": 0.30,
                }
            ],
            RelationType.DISTANCE_AT_LEAST,
            0.30,
        ),
    ],
)
def test_numeric_distance_cues_preserve_exact_value_and_endpoint_order(
    tmp_path: Path,
    user_text: str,
    lateral: list[dict[str, Any]],
    expected_relation: RelationType,
    expected_distance: float,
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=lateral,
            ),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=122)

    relation = next(item for item in spec.relations if item.relation == expected_relation)
    actual_distance = (
        relation.max_distance_m
        if expected_relation == RelationType.NEAR
        else relation.min_distance_m
    )
    assert actual_distance == pytest.approx(expected_distance)
    assert (relation.source, relation.target) == ("cup_1", "bowl_1")


@pytest.mark.parametrize(
    ("user_text", "lateral"),
    [
        (
            "Place a cup within 0.20 m of a bowl.",
            [
                {
                    "relation": "near",
                    "source": "cup_1",
                    "target": "bowl_1",
                    "max_distance_m": 0.25,
                }
            ],
        ),
        (
            "Place a cup within 0.20 m of a bowl.",
            [
                {
                    "relation": "near",
                    "source": "bowl_1",
                    "target": "cup_1",
                    "max_distance_m": 0.20,
                }
            ],
        ),
        (
            "Place a cup and a bowl at least 0.30 m apart.",
            [
                {
                    "relation": "distance_at_least",
                    "source": "cup_1",
                    "target": "bowl_1",
                    "min_distance_m": 0.20,
                }
            ],
        ),
        (
            "Place a cup and a bowl at least 0.30 m apart.",
            [
                {
                    "relation": "distance_at_least",
                    "source": "bowl_1",
                    "target": "cup_1",
                    "min_distance_m": 0.30,
                }
            ],
        ),
    ],
    ids=("within-value", "within-endpoints", "minimum-value", "minimum-endpoints"),
)
def test_numeric_distance_change_or_reversal_fails_closed(
    tmp_path: Path,
    user_text: str,
    lateral: list[dict[str, Any]],
) -> None:
    transport = FakeTransport(
        _objects(_object("cup"), _object("bowl")),
        _relations(
            topology=_all_on_table("cup_1", "bowl_1"),
            lateral=lateral,
        ),
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="preserve|missing or reversed") as captured:
        provider.parse_scene(request=user_text, seed=123)

    assert captured.value.stage == "relations"


@pytest.mark.parametrize(
    ("user_text", "object_document", "relation_document", "expected_stage"),
    [
        (
            "Place a red cup on the table.",
            _objects(_object("cup", color=["red"])),
            None,
            "objects",
        ),
        (
            "Place a wooden cup on the table.",
            _objects(_object("cup", material={"value": "wood"})),
            None,
            "objects",
        ),
        (
            "Place a cup on the left side of the table.",
            _objects(_object("cup", region=["left"])),
            None,
            "objects",
        ),
        (
            "Place an open cabinet on the table.",
            _objects(
                _object(
                    "cabinet",
                    articulation={
                        "state": ["open"],
                        "open_fraction": 1.0,
                        "joint_selector": "all_movable",
                    },
                )
            ),
            None,
            "objects",
        ),
        (
            "Place a cup on the table.",
            _objects(_object("cup")),
            _relations(topology=[{"relation": "on_table", "source": ["cup_1"], "target": "table"}]),
            "relations",
        ),
        (
            "Place a cup on the table.",
            _objects(_object("cup")),
            _relations(
                topology=[{"relation": "on_table", "source": "cup_1", "target": {"id": "table"}}]
            ),
            "relations",
        ),
        (
            "Place a cup near a bowl.",
            _objects(_object("cup"), _object("bowl")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": [0.25],
                    }
                ],
            ),
            "relations",
        ),
    ],
    ids=(
        "color-list",
        "material-object",
        "region-list",
        "articulation-state-list",
        "relation-source-list",
        "relation-target-object",
        "distance-list",
    ),
)
def test_nested_malformed_values_return_structured_provider_errors(
    tmp_path: Path,
    user_text: str,
    object_document: dict[str, Any],
    relation_document: dict[str, Any] | None,
    expected_stage: str,
) -> None:
    responses = (
        (object_document,) if relation_document is None else (object_document, relation_document)
    )
    transport = FakeTransport(*responses)
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request=user_text, seed=124)

    assert captured.value.stage == expected_stage
    assert captured.value.failure_kind == "attempts_exhausted"
    assert captured.value.safe_details()["failure_kind"] == "attempts_exhausted"
    assert provider.evidence()["status"] == "fail"
    json.dumps(provider.evidence(), sort_keys=True)


def _deep_json_value(depth: int = 70) -> Any:
    value: Any = "leaf"
    for _ in range(depth):
        value = [value]
    return value


def test_deep_stage_json_is_rejected_as_a_structured_failure(tmp_path: Path) -> None:
    response = json.dumps(
        {"objects": _deep_json_value(), "ambiguities": []},
        ensure_ascii=False,
    )
    transport = FakeTransport(response)
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError, match="complexity limits") as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=125)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert len(transport.calls) == 1


def _seed_simple_cache(
    tmp_path: Path,
    *,
    request: str = "Place a cup on the table.",
    seed: int = 126,
) -> tuple[LLMProviderConfig, Path]:
    config = _config(tmp_path)
    provider = LLMSceneProvider(
        config,
        transport=FakeTransport(
            _objects(_object("cup")),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )
    provider.parse_scene(request=request, seed=seed)
    key = provider.evidence()["cache"]["key"]
    return config, config.cache_dir / f"{key}.json"


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "root-field",
        "evidence-field",
        "invalid-attempts",
        "semantic-evidence",
        "semantic-payload",
    ],
)
def test_cache_injection_is_revalidated_and_refilled(
    tmp_path: Path,
    mutation: str,
) -> None:
    request = "Place a cup on the table."
    seed = 126
    config, cache_path = _seed_simple_cache(
        tmp_path,
        request=request,
        seed=seed,
    )
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    if mutation == "root-field":
        record["api_key"] = "injected-secret"
    elif mutation == "evidence-field":
        record["evidence"]["api_key"] = "injected-secret"
    elif mutation == "invalid-attempts":
        record["evidence"]["stages"]["objects"]["attempts"] = 0
    elif mutation == "semantic-evidence":
        record["evidence"]["stages"]["objects"]["semantic_checks"] = {
            "version": "injected",
            "known_object_counts": {"banana": 99},
            "status": "pass",
        }
    else:
        record["payload"]["objects"][0]["color"] = "red"
        record["evidence"]["stages"]["objects"]["objects"][0]["color"] = "red"
        payload_sha256 = _canonical_sha256(record["payload"])
        record["payload_sha256"] = payload_sha256
        record["evidence"]["payload_sha256"] = payload_sha256
    cache_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    replacement_transport = FakeTransport(
        _objects(_object("cup")),
        _relations(topology=[_on_table("cup_1")]),
    )
    replacement = LLMSceneProvider(config, transport=replacement_transport)

    payload = replacement.parse_scene(request=request, seed=seed)

    assert len(replacement_transport.calls) == 2
    assert payload["objects"][0]["color"] is None
    serialized = cache_path.read_text(encoding="utf-8")
    assert "injected-secret" not in serialized
    assert '"version": "injected"' not in serialized
    assert '"banana": 99' not in serialized


def test_deep_cache_json_is_ignored_and_refilled(tmp_path: Path) -> None:
    request = "Place a cup on the table."
    seed = 127
    config, cache_path = _seed_simple_cache(
        tmp_path,
        request=request,
        seed=seed,
    )
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["evidence"]["deep"] = _deep_json_value()
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    replacement_transport = FakeTransport(
        _objects(_object("cup")),
        _relations(topology=[_on_table("cup_1")]),
    )
    replacement = LLMSceneProvider(config, transport=replacement_transport)

    replacement.parse_scene(request=request, seed=seed)

    assert len(replacement_transport.calls) == 2
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "deep" not in refreshed["evidence"]


def test_deep_loopback_http_envelope_is_a_structured_failure(
    tmp_path: Path,
) -> None:
    def respond(_: dict[str, Any]) -> _HTTPResponse:
        return (
            200,
            {},
            {
                "choices": [{"message": {"content": json.dumps(_objects(_object("cup")))}}],
                "deep": _deep_json_value(),
            },
        )

    server, thread = _start_loopback_server(respond)
    try:
        provider = LLMSceneProvider(
            LLMProviderConfig(
                endpoint=f"{server.endpoint}/v1",
                model="loopback-model",
                api_key="loopback-secret",
                max_attempts=1,
                cache_dir=tmp_path / "deep-http-cache",
            )
        )

        with pytest.raises(LLMProviderError, match="valid JSON") as captured:
            provider.parse_scene(request="Place a cup on the table.", seed=128)
    finally:
        _stop_loopback_server(server, thread)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("field", "yaml_value"),
    [
        ("model", "[yaml-model]"),
        ("cache_dir", "{path: cache}"),
    ],
)
def test_yaml_configuration_rejects_non_scalar_text_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    yaml_value: str,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    monkeypatch.setenv("TEST_ONLY_LLM_KEY", "yaml-environment-secret")
    config_path = tmp_path / f"bad-{field}.yaml"
    model_line = "" if field == "model" else "model: yaml-model\n"
    config_path.write_text(
        "endpoint: https://example.invalid/v1\n"
        f"{model_line}"
        "api_key_env: TEST_ONLY_LLM_KEY\n"
        f"{field}: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError, match=f"invalid {field}: expected text") as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"


def test_endpoint_trailing_slashes_are_canonicalized(tmp_path: Path) -> None:
    config = LLMProviderConfig(
        endpoint="https://example.invalid/v1///",
        model="yaml-model",
        api_key="secret",
        cache_dir=tmp_path / "cache",
    )

    assert config.endpoint == "https://example.invalid/v1"


def test_generate_missing_assets_error_writes_structured_failure(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "script" / "generate_scene.py"),
            "--provider",
            "rule",
            "--prompt",
            "Place a cup on top of a plate.",
            "--seed",
            "129",
            "--asset-catalog",
            str(ROOT / "tests" / "fixtures" / "asset_catalog.json"),
            "--out-root",
            str(tmp_path / "output"),
            "--generate-missing-assets",
            "--generated-objects-root",
            str(tmp_path / "generated-assets"),
        ],
        cwd=ROOT,
        env=_clean_cli_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    reports = list((tmp_path / "output" / "_failures").glob("*/failure_report.json"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["stage"] == "asset_grounding"
    assert report["provider"] == "rule"
    assert report["error_type"] == "SceneSpecError"


def test_missing_asset_catalog_writes_structured_grounding_failure(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "script" / "generate_scene.py"),
            "--provider",
            "rule",
            "--prompt",
            "Place a cup on the table.",
            "--seed",
            "130",
            "--asset-catalog",
            str(tmp_path / "missing-catalog.json"),
            "--out-root",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=_clean_cli_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    reports = list((tmp_path / "output" / "_failures").glob("*/failure_report.json"))
    assert completed.returncode == 2
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["stage"] == "asset_grounding"
    assert report["provider"] == "rule"
    assert report["blocker"] == "asset catalog could not be loaded or validated"
    assert report["error_type"] == "FileNotFoundError"


def test_same_category_response_order_is_canonicalized_by_object_id(
    tmp_path: Path,
) -> None:
    request = "Place a red cup on the table. Place a blue cup on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup", index=2, color="blue"),
                _object("cup", index=1, color="red"),
            ),
            _relations(topology=_all_on_table("cup_1", "cup_2")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=130)

    assert [(item.object_id, item.color) for item in spec.objects] == [
        ("cup_1", "red"),
        ("cup_2", "blue"),
    ]


def test_reported_ambiguity_with_extra_fields_still_stops_after_first_attempt(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        _objects(
            _object("cup"),
            ambiguities=["the intended cup is unclear"],
        )
        | {"asset_path": "/must/not/be/processed"}
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=3),
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=131)

    assert captured.value.failure_kind == "ambiguous_request"
    assert captured.value.attempts == 1
    assert len(transport.calls) == 1


def test_bad_http_status_line_is_wrapped_without_response_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadStatusOpener:
        def open(self, *args: Any, **kwargs: Any) -> Any:
            raise http.client.BadStatusLine("private-response-text")

    monkeypatch.setattr(
        "scene_gen.llm_provider.urllib.request.build_opener",
        lambda *args: BadStatusOpener(),
    )
    provider = LLMSceneProvider(
        LLMProviderConfig(
            endpoint="https://loopback.invalid/v1",
            model="fake-model",
            api_key="secret",
            max_attempts=1,
            cache_dir=tmp_path / "cache",
        )
    )

    with pytest.raises(LLMProviderError, match="BadStatusLine") as captured:
        provider.parse_scene(request="Place a cup on the table.", seed=132)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert "private-response-text" not in str(captured.value)
    assert "private-response-text" not in json.dumps(provider.evidence())


def test_modal_can_is_not_extracted_as_a_can_object(tmp_path: Path) -> None:
    request = "Place a cup near a bowl if you can."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=201)

    assert [item.object_id for item in spec.objects] == ["cup_1", "bowl_1"]
    assert all(item.category != "can" for item in spec.objects)


@pytest.mark.parametrize(
    ("user_text", "cup_fields"),
    [
        ("Place a cup that is red near a bowl.", {"color": "red"}),
        ("Place a cup made of metal near a bowl.", {"material": "metal"}),
    ],
    ids=("relative-color", "made-of-material"),
)
def test_postnominal_attribute_owner_does_not_shift_to_the_next_object(
    tmp_path: Path,
    user_text: str,
    cup_fields: dict[str, str],
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup", **cup_fields), _object("bowl")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=202)

    by_id = {item.object_id: item for item in spec.objects}
    assert by_id["cup_1"].color == cup_fields.get("color")
    assert by_id["cup_1"].material == cup_fields.get("material")
    assert by_id["bowl_1"].color is None
    assert by_id["bowl_1"].material is None


def test_incidental_lighting_color_does_not_become_object_color(tmp_path: Path) -> None:
    request = "Use red lighting. Place a cup on the table."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup", color=None)),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=203)

    assert spec.objects[0].color is None


@pytest.mark.parametrize(
    ("user_text", "cabinet_state", "box_state", "lateral"),
    [
        (
            "Place a cabinet near an open box.",
            None,
            "open",
            [
                {
                    "relation": "near",
                    "source": "cabinet_1",
                    "target": "box_1",
                    "max_distance_m": 0.25,
                }
            ],
        ),
        (
            "Place a cabinet and a box on the table, both open.",
            "open",
            "open",
            [],
        ),
    ],
    ids=("next-object", "coordinated-both"),
)
def test_articulation_scope_does_not_bleed_between_objects(
    tmp_path: Path,
    user_text: str,
    cabinet_state: str | None,
    box_state: str | None,
    lateral: list[dict[str, Any]],
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(
                _object(
                    "cabinet",
                    articulation=_articulation(cabinet_state) if cabinet_state else None,
                ),
                _object("box", articulation=_articulation(box_state) if box_state else None),
            ),
            _relations(
                topology=_all_on_table("cabinet_1", "box_1"),
                lateral=lateral,
            ),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=204)

    by_id = {item.object_id: item for item in spec.objects}
    actual = {
        object_id: item.articulation.state if item.articulation else None
        for object_id, item in by_id.items()
    }
    assert actual == {"cabinet_1": cabinet_state, "box_1": box_state}


@pytest.mark.parametrize(
    ("user_text", "objects", "topology", "relation"),
    [
        (
            "Place a basket that contains a cup on the table.",
            (_object("basket"), _object("cup")),
            (
                {"relation": "inside", "source": "cup_1", "target": "basket_1"},
                _on_table("basket_1"),
            ),
            RelationType.INSIDE,
        ),
        (
            "Place a plate that supports a cup on the table.",
            (_object("plate"), _object("cup")),
            (
                {"relation": "on_top_of", "source": "cup_1", "target": "plate_1"},
                _on_table("plate_1"),
            ),
            RelationType.ON_TOP_OF,
        ),
        (
            "Place a cup that rests on a plate on the table.",
            (_object("cup"), _object("plate")),
            (
                {"relation": "on_top_of", "source": "cup_1", "target": "plate_1"},
                _on_table("plate_1"),
            ),
            RelationType.ON_TOP_OF,
        ),
    ],
    ids=("contains", "supports", "rests-on"),
)
def test_support_topology_paraphrases_cannot_bleed_into_on_table(
    tmp_path: Path,
    user_text: str,
    objects: tuple[dict[str, Any], ...],
    topology: tuple[dict[str, Any], ...],
    relation: RelationType,
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(*objects),
            _relations(topology=list(topology)),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=205)

    nested = next(item for item in spec.relations if item.relation == relation)
    assert (nested.source, nested.target) == ("cup_1", topology[0]["target"])


def test_shared_subject_keeps_each_relation_on_the_original_source(tmp_path: Path) -> None:
    request = "Place a cup to the left of a bowl and to the right of a plate."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl"), _object("plate")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "plate_1"),
                lateral=[
                    {"relation": "left_of", "source": "cup_1", "target": "bowl_1"},
                    {"relation": "right_of", "source": "cup_1", "target": "plate_1"},
                ],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=206)

    lateral = {
        (item.relation, item.source, item.target)
        for item in spec.relations
        if item.relation != RelationType.ON_TABLE
    }
    assert lateral == {
        (RelationType.LEFT_OF, "cup_1", "bowl_1"),
        (RelationType.RIGHT_OF, "cup_1", "plate_1"),
    }


def test_shared_subject_numeric_cues_keep_endpoint_and_value_pairs(tmp_path: Path) -> None:
    request = "Place a cup within 0.2 m of a bowl and within 0.3 m of a plate."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl"), _object("plate")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "plate_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.2,
                    },
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "plate_1",
                        "max_distance_m": 0.3,
                    },
                ],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=207)

    near = {
        (item.source, item.target): item.max_distance_m
        for item in spec.relations
        if item.relation == RelationType.NEAR
    }
    assert near[("cup_1", "bowl_1")] == pytest.approx(0.2)
    assert near[("cup_1", "plate_1")] == pytest.approx(0.3)


def test_modified_definite_reference_reuses_the_uniquely_matching_object(
    tmp_path: Path,
) -> None:
    request = "Place a red cup and a blue cup on the table. Put the red cup to the left of a bowl."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(
                _object("cup", index=1, color="red"),
                _object("cup", index=2, color="blue"),
                _object("bowl"),
            ),
            _relations(
                topology=_all_on_table("cup_1", "cup_2", "bowl_1"),
                lateral=[{"relation": "left_of", "source": "cup_1", "target": "bowl_1"}],
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=208)

    assert [item.object_id for item in spec.objects] == ["cup_1", "cup_2", "bowl_1"]
    assert any(
        item.relation == RelationType.LEFT_OF and item.source == "cup_1" and item.target == "bowl_1"
        for item in spec.relations
    )


@pytest.mark.parametrize(
    "user_text",
    [
        "Place either a cup or a bowl on the table.",
        "Place a cup near or behind a bowl.",
        "Place a cup beside a drawing of an apple.",
        "Place a cup under a bowl.",
        "Place a cup above a bowl.",
    ],
    ids=("object-disjunction", "relation-disjunction", "depiction", "under", "above"),
)
def test_ambiguous_or_unsupported_semantics_fail_before_transport(
    tmp_path: Path,
    user_text: str,
) -> None:
    transport = FakeTransport(
        _objects(_object("cup"), _object("bowl")),
        _relations(topology=_all_on_table("cup_1", "bowl_1")),
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request=user_text, seed=209)

    assert captured.value.failure_kind in {"ambiguous_request", "unsupported_request"}
    assert captured.value.attempts == 0
    assert transport.calls == []


@pytest.mark.parametrize(
    ("user_text", "object_document", "relation_document"),
    [
        (
            "Place a cup near a bowl if you can.",
            _objects(_object("cup"), _object("bowl"), _object("can")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "can_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
        (
            "Place a cup that is red near a bowl.",
            _objects(_object("cup", color=None), _object("bowl", color="red")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
        (
            "Place a cup made of metal near a bowl.",
            _objects(_object("cup", material=None), _object("bowl", material="metal")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
        (
            "Place a cabinet near an open box.",
            _objects(
                _object("cabinet", articulation=_articulation("open")),
                _object("box", articulation=_articulation("open")),
            ),
            _relations(
                topology=_all_on_table("cabinet_1", "box_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cabinet_1",
                        "target": "box_1",
                        "max_distance_m": 0.25,
                    }
                ],
            ),
        ),
        (
            "Place a basket that contains a cup on the table.",
            _objects(_object("basket"), _object("cup")),
            _relations(topology=_all_on_table("basket_1", "cup_1")),
        ),
        (
            "Place a cup to the left of a bowl and to the right of a plate.",
            _objects(_object("cup"), _object("bowl"), _object("plate")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "plate_1"),
                lateral=[
                    {"relation": "left_of", "source": "cup_1", "target": "bowl_1"},
                    {"relation": "right_of", "source": "bowl_1", "target": "plate_1"},
                ],
            ),
        ),
        (
            "Place a cup within 0.2 m of a bowl and within 0.3 m of a plate.",
            _objects(_object("cup"), _object("bowl"), _object("plate")),
            _relations(
                topology=_all_on_table("cup_1", "bowl_1", "plate_1"),
                lateral=[
                    {
                        "relation": "near",
                        "source": "cup_1",
                        "target": "bowl_1",
                        "max_distance_m": 0.3,
                    },
                    {
                        "relation": "near",
                        "source": "bowl_1",
                        "target": "plate_1",
                        "max_distance_m": 0.2,
                    },
                ],
            ),
        ),
        (
            "Place a red cup and a blue cup on the table. Put the red cup to the left of a bowl.",
            _objects(
                _object("cup", index=1, color="red"),
                _object("cup", index=2, color="blue"),
                _object("cup", index=3, color="red"),
                _object("bowl"),
            ),
            _relations(
                topology=_all_on_table("cup_1", "cup_2", "cup_3", "bowl_1"),
                lateral=[{"relation": "left_of", "source": "cup_3", "target": "bowl_1"}],
            ),
        ),
    ],
    ids=(
        "modal-can-object",
        "relative-color-rebound",
        "postnominal-material-rebound",
        "next-articulation-leaks-backward",
        "contains-replaced-by-on-table",
        "shared-subject-rebound",
        "numeric-cues-permuted",
        "modified-reference-becomes-new-object",
    ),
)
def test_known_false_accept_candidates_fail_closed(
    tmp_path: Path,
    user_text: str,
    object_document: dict[str, Any],
    relation_document: dict[str, Any],
) -> None:
    transport = FakeTransport(object_document, relation_document)
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )

    with pytest.raises(LLMProviderError):
        provider.parse_scene(request=user_text, seed=212)

    assert provider.evidence()["status"] == "fail"


def test_modal_plain_on_preserves_object_support_topology(tmp_path: Path) -> None:
    request = "A cup should be on a plate on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup"), _object("plate")),
            _relations(
                topology=[
                    {
                        "relation": "on_top_of",
                        "source": "cup_1",
                        "target": "plate_1",
                    },
                    _on_table("plate_1"),
                ]
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=213)

    assert any(
        item.relation == RelationType.ON_TOP_OF
        and item.source == "cup_1"
        and item.target == "plate_1"
        for item in spec.relations
    )


def test_modal_plain_on_cannot_be_replaced_by_two_on_table_relations(
    tmp_path: Path,
) -> None:
    request = "A cup should be on a plate on the table."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup"), _object("plate")),
            _relations(topology=_all_on_table("cup_1", "plate_1")),
        ),
    )

    with pytest.raises(LLMProviderError, match="missing or reversed"):
        provider.parse_scene(request=request, seed=214)


@pytest.mark.parametrize(
    "user_text",
    [
        "Place a large cup on the table.",
        "Place a cup on a large plate.",
        "Place a cup that is large on the table.",
    ],
)
def test_unsupported_local_object_modifiers_fail_before_transport(
    tmp_path: Path,
    user_text: str,
) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError, match="modifier") as captured:
        provider.parse_scene(request=user_text, seed=215)

    assert captured.value.stage == "semantic_precheck"
    assert transport.calls == []


def test_open_vocabulary_compound_starting_with_known_noun_is_preserved(
    tmp_path: Path,
) -> None:
    request = "Place a can opener and a cup on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("can_opener"), _object("cup")),
            _relations(topology=_all_on_table("can_opener_1", "cup_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=216)

    assert [item.category for item in spec.objects] == ["can_opener", "cup"]


def test_open_vocabulary_compound_cannot_be_truncated_to_known_head(
    tmp_path: Path,
) -> None:
    request = "Place a can opener and a cup on the table."
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("can"), _object("cup")),
        ),
    )

    with pytest.raises(LLMProviderError, match="object count"):
        provider.parse_scene(request=request, seed=217)


@pytest.mark.parametrize(
    ("surface", "category"),
    [
        ("coffee mug", "mug"),
        ("drawer cabinet", "cabinet"),
        ("storage box", "box"),
        ("notebook computer", "laptop"),
        ("soda can", "can"),
    ],
)
def test_existing_multiword_aliases_keep_their_canonical_category(
    tmp_path: Path,
    surface: str,
    category: str,
) -> None:
    request = f"Place a {surface} on the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object(category)),
            _relations(topology=[_on_table(f"{category}_1")]),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=218)

    assert spec.objects[0].category == category


@pytest.mark.parametrize(
    "user_text",
    [
        "Open the cabinet and the box.",
        "Open the cabinet and box.",
    ],
)
def test_coordinated_articulation_command_fails_before_transport(
    tmp_path: Path,
    user_text: str,
) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request=user_text, seed=219)

    assert captured.value.failure_kind == "ambiguous_request"
    assert transport.calls == []


def test_two_explicit_table_regions_do_not_become_object_support(
    tmp_path: Path,
) -> None:
    request = "Place a cup on the left side of the table and a bowl on the right side of the table."
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(
                _object("cup", region="left"),
                _object("bowl", region="right"),
            ),
            _relations(topology=_all_on_table("cup_1", "bowl_1")),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=220)

    assert [item.region for item in spec.objects] == ["left", "right"]


def test_fresh_payload_mutation_cannot_change_parse_evidence(tmp_path: Path) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup")),
            _relations(topology=[_on_table("cup_1")]),
        ),
    )
    payload = provider.parse_scene(request="Place a cup on the table.", seed=221)
    evidence_before = provider.evidence()

    payload["objects"][0]["category"] = "tampered"

    assert provider.evidence() == evidence_before
    assert provider.evidence()["stages"]["objects"]["objects"][0]["category"] == "cup"


def test_oversized_parse_cache_is_ignored_and_refilled(tmp_path: Path) -> None:
    request = "Place a cup on the table."
    seed = 222
    config, cache_path = _seed_simple_cache(
        tmp_path,
        request=request,
        seed=seed,
    )
    cache_path.write_bytes(b" " * (MAX_PARSE_CACHE_BYTES + 1))
    transport = FakeTransport(
        _objects(_object("cup")),
        _relations(topology=[_on_table("cup_1")]),
    )
    provider = LLMSceneProvider(config, transport=transport)

    provider.parse_scene(request=request, seed=seed)

    assert len(transport.calls) == 2
    assert cache_path.stat().st_size < MAX_PARSE_CACHE_BYTES


def test_direct_provider_call_applies_full_prompt_boundary(tmp_path: Path) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(
            request="Use asset_id 071_can and model_id 0.",
            seed=223,
        )

    assert captured.value.stage == "semantic_precheck"
    assert captured.value.failure_kind == "unsupported_request"
    assert transport.calls == []


@pytest.mark.parametrize(
    "document",
    [
        ("endpoint: https://example.invalid/v1\nmodel: first-model\nmodel: second-model\n"),
        (
            "active_profile: default\n"
            "profiles:\n"
            "  default:\n"
            "    endpoint: https://example.invalid/v1\n"
            "    model: first-model\n"
            "    model: second-model\n"
        ),
    ],
)
def test_yaml_configuration_rejects_duplicate_keys_at_every_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"
    assert "first-model" not in str(captured.value)
    assert "second-model" not in str(captured.value)


def test_yaml_configuration_size_limit_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "oversized.yaml"
    config_path.write_bytes(b"x" * (MAX_LLM_CONFIG_BYTES + 1))

    with pytest.raises(LLMProviderError, match="size limit") as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"


def test_yaml_giant_integer_is_a_structured_configuration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    config_path = tmp_path / "giant-integer.yaml"
    config_path.write_text(
        f"endpoint: https://example.invalid/v1\nmodel: model-name\ntimeout_s: {'9' * 5000}\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"


@pytest.mark.parametrize(
    "response",
    [
        '{"objects":' + "9" * 5000 + "}",
        (
            '{"objects":[{"object_id":"cabinet_1","category":"cabinet",'
            '"articulation":{"state":"partially_open","open_fraction":'
            + "9" * 4000
            + ',"joint_selector":"all_movable"}}],"ambiguities":[]}'
        ),
    ],
)
def test_giant_provider_numbers_are_structured_stage_failures(
    tmp_path: Path,
    response: str,
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(response),
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cabinet on the table.", seed=224)

    assert captured.value.stage == "objects"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert provider.evidence()["status"] == "fail"


def test_giant_relation_distance_is_a_structured_stage_failure(tmp_path: Path) -> None:
    huge = "9" * 4000
    relation_response = (
        '{"topology":['
        '{"relation":"on_table","source":"cup_1","target":"table"},'
        '{"relation":"on_table","source":"bowl_1","target":"table"}],'
        '"lateral":[{"relation":"near","source":"cup_1","target":"bowl_1",'
        f'"max_distance_m":{huge}}}],"ambiguities":[]}}'
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=FakeTransport(
            _objects(_object("cup"), _object("bowl")),
            relation_response,
        ),
    )

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup near a bowl.", seed=225)

    assert captured.value.stage == "relations"
    assert captured.value.failure_kind == "attempts_exhausted"
    assert provider.evidence()["status"] == "fail"


def test_failed_second_request_cannot_retain_first_request_pass_evidence(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        _objects(_object("cup")),
        _relations(topology=[_on_table("cup_1")]),
        '{"objects":' + "9" * 5000 + "}",
    )
    provider = LLMSceneProvider(
        _config(tmp_path, max_attempts=1),
        transport=transport,
    )
    provider.parse_scene(request="Place a cup on the table.", seed=226)
    prior = provider.evidence()

    with pytest.raises(LLMProviderError):
        provider.parse_scene(request="Place a bowl on the table.", seed=227)

    current = provider.evidence()
    assert prior["status"] == "pass"
    assert current["status"] == "fail"
    assert current != prior
    assert current["stage"] == "objects"


def test_deterministic_ambiguity_is_not_retried(tmp_path: Path) -> None:
    transport = FakeTransport(
        _objects(_object("cup")),
        _relations(topology=[_on_table("cup_1")]),
    )
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request="Place a cup next to the cup.", seed=228)

    assert captured.value.stage == "relations"
    assert captured.value.attempts == 1
    assert captured.value.failure_kind == "ambiguous_request"
    assert len(transport.calls) == 2
    assert provider.evidence()["failure_kind"] == "ambiguous_request"


@pytest.mark.parametrize(
    "user_text",
    [
        "Place a non-red cup on the table.",
        "放一个非红色杯子在桌上。",
        "放一个非金属杯子在桌上。",
        "Place a cup nowhere near a bowl.",
        "Place a cup anywhere but inside a basket.",
        "Place a cup far away from a bowl.",
        "Place large cup on the table.",
        "Put tiny bowl on the table.",
        "Place a large widget on the table.",
        "Place mice on the table.",
        "Place a widgets on the table.",
    ],
)
def test_unrepresentable_semantics_fail_before_transport(
    tmp_path: Path,
    user_text: str,
) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request=user_text, seed=229)

    assert captured.value.stage == "semantic_precheck"
    assert transport.calls == []


@pytest.mark.parametrize(
    "user_text",
    [
        "把杯垫放在桌上。",
        "把瓶盖放在桌上。",
        "把苹果汁放在桌上。",
        "把刀叉放在桌上。",
        "把盘架放在桌上。",
        "把杯套放在桌上。",
    ],
)
def test_chinese_compound_substrings_fail_before_transport(
    tmp_path: Path,
    user_text: str,
) -> None:
    transport = FakeTransport()
    provider = LLMSceneProvider(_config(tmp_path), transport=transport)

    with pytest.raises(LLMProviderError) as captured:
        provider.parse_scene(request=user_text, seed=230)

    assert captured.value.stage == "semantic_precheck"
    assert transport.calls == []


@pytest.mark.parametrize(
    ("user_text", "articulation"),
    [
        (
            "把柜子打开并放在桌上。",
            {"state": "open", "open_fraction": 1.0, "joint_selector": "all_movable"},
        ),
        (
            "把柜子打开一半并放在桌上。",
            {
                "state": "partially_open",
                "open_fraction": 0.5,
                "joint_selector": "all_movable",
            },
        ),
        (
            "打开的柜子放在桌上。",
            {"state": "open", "open_fraction": 1.0, "joint_selector": "all_movable"},
        ),
        (
            "关闭的柜子放在桌上。",
            {"state": "closed", "open_fraction": 0.0, "joint_selector": "all_movable"},
        ),
    ],
)
def test_supported_chinese_articulation_survives_compound_guard(
    tmp_path: Path,
    user_text: str,
    articulation: dict[str, Any],
) -> None:
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cabinet", articulation=articulation)),
            _relations(topology=[_on_table("cabinet_1")]),
        ),
    )

    spec = parse_with_provider(provider, request=user_text, seed=231)

    assert spec.objects[0].articulation is not None
    assert spec.objects[0].articulation.open_fraction == articulation["open_fraction"]


def test_supported_chinese_direct_locative_relation_is_enforced(tmp_path: Path) -> None:
    request = "杯子位于盘子上面。"
    provider = LLMSceneProvider(
        _config(tmp_path),
        transport=FakeTransport(
            _objects(_object("cup"), _object("plate")),
            _relations(
                topology=[
                    {"relation": "on_top_of", "source": "cup_1", "target": "plate_1"},
                    _on_table("plate_1"),
                ]
            ),
        ),
    )

    spec = parse_with_provider(provider, request=request, seed=232)

    assert any(item.relation == RelationType.ON_TOP_OF for item in spec.relations)


def test_yaml_rejects_unsupported_credential_alias_in_an_unselected_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    monkeypatch.setenv("GOOD_PROFILE_KEY", "environment-only-secret")
    config_path = tmp_path / "hidden-secret.yaml"
    config_path.write_text(
        "active_profile: good\n"
        "profiles:\n"
        "  good:\n"
        "    endpoint: https://example.invalid/v1\n"
        "    model: safe-model\n"
        "    api_key_env: GOOD_PROFILE_KEY\n"
        "  unused:\n"
        "    endpoint: https://unused.invalid/v1\n"
        "    model: unused-model\n"
        "    token: plaintext-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "inline_secret_forbidden"
    assert "plaintext-secret" not in str(captured.value)


def test_yaml_alias_graph_is_rejected_without_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENENV_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("GENENV_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GENENV_LLM_MODEL", raising=False)
    lines = ["n0: &n0 [leaf]"]
    lines.extend(f"n{index}: &n{index} [*n{index - 1}, *n{index - 1}]" for index in range(1, 29))
    config_path = tmp_path / "alias-graph.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LLMProviderError) as captured:
        load_llm_provider_config(config_path)

    assert config_path.stat().st_size < 2048
    assert captured.value.stage == "configuration"
    assert captured.value.failure_kind == "invalid_configuration"
    assert "YAML aliases" not in str(captured.value)

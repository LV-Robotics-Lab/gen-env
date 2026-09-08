"""Optional two-stage LLM extraction behind the bounded ``SceneSpec`` boundary.

The model is only a semantic candidate generator. It may name objects,
attributes, and relations; it cannot select assets, provide paths, poses,
coordinates, or simulator code. Every response is deterministically cleaned
and validated by :mod:`scene_gen.schema` before it can reach grounding.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError

from .parser import (
    ARTICULATED_CATEGORIES,
    COLOR_TERMS,
    MATERIAL_TERMS,
    MAX_SCENE_OBJECTS,
    REGION_TERMS,
    parse_provider_payload,
    validate_prompt_boundary,
)
from .schema import RelationType, SceneSpecError
from .semantic_checks import (
    SEMANTIC_CHECK_VERSION,
    AmbiguousReference,
    SemanticMismatch,
    reject_unsupported_request_semantics,
    validate_object_extraction,
    validate_relation_extraction,
)

PROVIDER_VERSION = "scene_gen.llm_extractor.v4"
EVIDENCE_SCHEMA = "robotwin.llm_scene_extraction.v1"
DEFAULT_CACHE_DIR = Path("data/scene_gen/parse_cache")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 3
MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PARSE_CACHE_BYTES = 4 * 1024 * 1024
MAX_LLM_CONFIG_BYTES = 256 * 1024

_PROMPT_DIR = Path(__file__).with_name("prompts")
_OBJECTS_PROMPT = _PROMPT_DIR / "llm_objects.md"
_RELATIONS_PROMPT = _PROMPT_DIR / "llm_relations.md"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CATEGORY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOPOLOGY = {
    RelationType.ON_TABLE.value,
    RelationType.ON_TOP_OF.value,
    RelationType.INSIDE.value,
}
_LATERAL = {
    RelationType.LEFT_OF.value,
    RelationType.RIGHT_OF.value,
    RelationType.FRONT_OF.value,
    RelationType.BEHIND.value,
    RelationType.NEAR.value,
    RelationType.DISTANCE_AT_LEAST.value,
}
_YAML_FIELDS = {
    "endpoint",
    "base_url",
    "model",
    "model_name",
    "api_key",
    "api_key_env",
    "api_mode",
    "timeout_s",
    "max_attempts",
    "cache_dir",
    "temperature",
}
_UNSUPPORTED_CREDENTIAL_FIELDS = {"apikey", "key", "secret", "token", "api_key_file"}
_RESERVED_OBJECT_CATEGORIES = {"robot", "table", "tabletop", "workspace", "world"}

Transport = Callable[[str, str], str]


class LLMProviderError(SceneSpecError):
    """A safe, structured failure from optional LLM extraction."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        attempts: int = 0,
        failure_kind: str = "provider_error",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts
        self.failure_kind = failure_kind

    def safe_details(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "attempts": self.attempts,
            "failure_kind": self.failure_kind,
            "message": str(self),
        }


class _StageOutputError(ValueError):
    """A model response was syntactically or semantically unusable."""


class _ReportedAmbiguity(ValueError):
    """A model explicitly reported ambiguity that must not be retried away."""

    def __init__(self, ambiguities: list[str]) -> None:
        super().__init__("; ".join(ambiguities))
        self.ambiguities = ambiguities


class _TransportError(RuntimeError):
    """A network or provider response failed without retaining response bodies."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent bearer credentials from crossing redirect boundaries."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _is_unroutable_host(hostname: str | None) -> bool:
    """True for hosts the public internet cannot reach, where plaintext HTTP is allowed.

    The bearer token is sent in a header, so the endpoint must be HTTPS whenever it leaves
    this network. It need not be for a destination that is not globally routable: loopback,
    RFC 1918, link-local, or the 100.64/10 shared range a WireGuard/Tailscale peer sits in,
    which carries its own transport encryption. Only literal addresses and `localhost`
    qualify -- any other name is rejected because DNS is not authenticated and could point
    a plaintext request with a live credential at an arbitrary host.
    """
    if hostname == "localhost":
        return True
    try:
        return not ipaddress.ip_address(hostname or "").is_global
    except ValueError:
        return False


@dataclass(frozen=True)
class LLMProviderConfig:
    """Resolved provider settings; the API key is excluded from repr and digests."""

    endpoint: str
    model: str
    api_key: str = field(repr=False, compare=False)
    api_mode: str = "chat"
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    cache_dir: Path = DEFAULT_CACHE_DIR
    source: str = field(default="environment", compare=False)
    profile: str | None = field(default=None, compare=False)
    api_key_env: str = field(default="direct", compare=False)
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise LLMProviderError(
                "LLM endpoint is empty",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if self.endpoint != self.endpoint.strip():
            raise LLMProviderError(
                "LLM endpoint cannot contain surrounding whitespace",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in self.endpoint):
            raise LLMProviderError(
                "LLM endpoint cannot contain control characters",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        endpoint = self.endpoint.rstrip("/")
        try:
            endpoint.encode("utf-8")
        except UnicodeError as exc:
            raise LLMProviderError(
                "LLM endpoint must be valid UTF-8 text",
                stage="configuration",
                failure_kind="invalid_configuration",
            ) from exc
        object.__setattr__(self, "endpoint", endpoint)
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMProviderError(
                "LLM model is empty",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        model = self.model.strip()
        if any(ord(character) < 32 or ord(character) == 127 for character in model):
            raise LLMProviderError(
                "LLM model cannot contain control characters",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        try:
            model.encode("utf-8")
        except UnicodeError as exc:
            raise LLMProviderError(
                "LLM model must be valid UTF-8 text",
                stage="configuration",
                failure_kind="invalid_configuration",
            ) from exc
        object.__setattr__(self, "model", model)
        if not isinstance(self.api_mode, str):
            raise LLMProviderError(
                "LLM api_mode must be text",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        object.__setattr__(self, "api_mode", self.api_mode.strip().lower())
        if not isinstance(self.cache_dir, (str, os.PathLike)) or not str(self.cache_dir).strip():
            raise LLMProviderError(
                "LLM cache_dir must be a non-empty path",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        try:
            cache_dir = Path(self.cache_dir).expanduser()
            str(cache_dir).encode("utf-8")
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            raise LLMProviderError(
                "LLM cache_dir is not a valid local path",
                stage="configuration",
                failure_kind="invalid_configuration",
            ) from exc
        object.__setattr__(self, "cache_dir", cache_dir)
        try:
            parsed_endpoint = urllib.parse.urlsplit(self.endpoint)
            parsed_endpoint.port
            hostname = parsed_endpoint.hostname
            username = parsed_endpoint.username
            password = parsed_endpoint.password
        except ValueError as exc:
            raise LLMProviderError(
                "LLM endpoint is not a valid URL",
                stage="configuration",
                failure_kind="invalid_configuration",
            ) from exc
        is_plaintext_ok = parsed_endpoint.scheme == "http" and _is_unroutable_host(hostname)
        if parsed_endpoint.scheme != "https" and not is_plaintext_ok:
            raise LLMProviderError(
                "LLM endpoint must use HTTPS; HTTP is allowed only for a loopback or "
                "non-routable host",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if (
            not hostname
            or username is not None
            or password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise LLMProviderError(
                "LLM endpoint must not contain userinfo, a query, or a fragment",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise LLMProviderError(
                "LLM API key is empty; set api_key, GENENV_LLM_API_KEY, or api_key_env",
                stage="configuration",
                failure_kind="missing_api_key",
            )
        api_key = self.api_key.strip()
        if any(ord(character) < 32 or ord(character) == 127 for character in api_key):
            raise LLMProviderError(
                "LLM API key cannot contain control characters",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        try:
            api_key.encode("utf-8")
        except UnicodeError as exc:
            raise LLMProviderError(
                "LLM API key must be valid UTF-8 text",
                stage="configuration",
                failure_kind="invalid_configuration",
            ) from exc
        object.__setattr__(self, "api_key", api_key)
        api_key_env = self.api_key_env or "direct"
        if not isinstance(api_key_env, str) or (
            api_key_env != "direct"
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env) is None
        ):
            raise LLMProviderError(
                "api_key_env must be 'direct' or a valid environment variable name",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        object.__setattr__(self, "api_key_env", api_key_env)
        if self.api_mode not in {"chat", "responses"}:
            raise LLMProviderError(
                f"unsupported LLM api_mode {self.api_mode!r}; expected 'chat' or 'responses'",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        timeout_s: float | None = None
        if not isinstance(self.timeout_s, bool) and isinstance(self.timeout_s, (int, float)):
            try:
                timeout_s = float(self.timeout_s)
            except (TypeError, ValueError, OverflowError):
                timeout_s = None
        if timeout_s is None or not math.isfinite(timeout_s) or not 0.0 < timeout_s <= 3600.0:
            raise LLMProviderError(
                "LLM timeout_s must be finite and within (0, 3600]",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        object.__setattr__(self, "timeout_s", timeout_s)
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
            or self.max_attempts > 10
        ):
            raise LLMProviderError(
                "LLM max_attempts must be between 1 and 10",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if self.temperature is not None:
            temperature: float | None = None
            if not isinstance(self.temperature, bool) and isinstance(
                self.temperature, (int, float)
            ):
                try:
                    temperature = float(self.temperature)
                except (TypeError, ValueError, OverflowError):
                    temperature = None
            if (
                temperature is None
                or not math.isfinite(temperature)
                or not 0.0 <= temperature <= 2.0
            ):
                raise LLMProviderError(
                    "LLM temperature must be null or a finite number in [0, 2]",
                    stage="configuration",
                    failure_kind="invalid_configuration",
                )
            object.__setattr__(self, "temperature", temperature)

    def safe_dict(self) -> dict[str, Any]:
        """Return behavior-affecting configuration without endpoint or credential."""

        return {
            "model": self.model,
            "api_mode": self.api_mode,
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "endpoint_sha256": hashlib.sha256(
                self.endpoint.rstrip("/").encode("utf-8")
            ).hexdigest(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.safe_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _float_setting(value: Any, *, name: str, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise LLMProviderError(
            f"invalid {name}: expected a number",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMProviderError(
            f"invalid {name}: expected a number",
            stage="configuration",
            failure_kind="invalid_configuration",
        ) from exc
    if not math.isfinite(result):
        raise LLMProviderError(
            f"invalid {name}: expected a finite number",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    return result


def _optional_float_setting(value: Any, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _float_setting(value, name=name, default=0.0)


def _int_setting(value: Any, *, name: str, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not (
        isinstance(value, int)
        or (isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()))
    ):
        raise LLMProviderError(
            f"invalid {name}: expected an integer",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMProviderError(
            f"invalid {name}: expected an integer",
            stage="configuration",
            failure_kind="invalid_configuration",
        ) from exc
    return result


def _string_setting(value: Any, *, name: str, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise LLMProviderError(
            f"invalid {name}: expected text",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    return value.strip()


def _config_from_values(
    values: dict[str, Any],
    *,
    source: str,
    profile: str | None,
    direct_api_key: str = "",
) -> LLMProviderConfig:
    unsupported_credential_fields = sorted(
        set(values) & _UNSUPPORTED_CREDENTIAL_FIELDS,
        key=str,
    )
    if unsupported_credential_fields:
        raise LLMProviderError(
            "unsupported LLM credential field; use exactly one of api_key or api_key_env",
            stage="configuration",
            failure_kind="inline_secret_forbidden",
        )
    unknown = sorted(set(values) - _YAML_FIELDS, key=str)
    if unknown:
        raise LLMProviderError(
            f"unknown LLM config fields: {unknown}",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    endpoint = _string_setting(values.get("endpoint"), name="endpoint")
    base_url = _string_setting(values.get("base_url"), name="base_url")
    if endpoint and base_url and endpoint.rstrip("/") != base_url.rstrip("/"):
        raise LLMProviderError(
            "endpoint and base_url aliases disagree",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    endpoint = endpoint or base_url
    model = _string_setting(values.get("model"), name="model")
    model_name = _string_setting(values.get("model_name"), name="model_name")
    if model and model_name and model != model_name:
        raise LLMProviderError(
            "model and model_name aliases disagree",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    model = model or model_name
    inline_api_key = _string_setting(values.get("api_key"), name="api_key")
    api_key_env = _string_setting(values.get("api_key_env"), name="api_key_env")
    if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env):
        raise LLMProviderError(
            "api_key_env must be a valid environment variable name",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    if "api_key" in values and "api_key_env" in values:
        raise LLMProviderError(
            "LLM config must set exactly one of api_key or api_key_env, not both",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    api_key = (
        direct_api_key.strip()
        or inline_api_key
        or (os.environ.get(api_key_env, "").strip() if api_key_env else "")
    )
    if api_key_env and not api_key:
        raise LLMProviderError(
            f"LLM API key environment variable {api_key_env} is not set",
            stage="configuration",
            failure_kind="missing_api_key",
        )
    cache_value = _string_setting(values.get("cache_dir"), name="cache_dir")
    cache_dir: str | Path = cache_value or DEFAULT_CACHE_DIR
    return LLMProviderConfig(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        api_mode=_string_setting(values.get("api_mode"), name="api_mode", default="chat"),
        timeout_s=_float_setting(
            values.get("timeout_s"), name="timeout_s", default=DEFAULT_TIMEOUT_S
        ),
        max_attempts=_int_setting(
            values.get("max_attempts"),
            name="max_attempts",
            default=DEFAULT_MAX_ATTEMPTS,
        ),
        cache_dir=cache_dir,
        source=source,
        profile=profile,
        api_key_env=api_key_env or "direct",
        temperature=_optional_float_setting(values.get("temperature"), name="temperature"),
    )


def _reject_duplicate_yaml_keys(
    node: yaml.Node | None,
    seen: set[int] | None = None,
) -> None:
    """Reject duplicate keys and aliases at every depth."""

    if node is None:
        return
    if seen is None:
        seen = set()
    node_identity = id(node)
    if node_identity in seen:
        raise yaml.YAMLError("YAML aliases are not supported in LLM configuration")
    seen.add(node_identity)

    if isinstance(node, yaml.MappingNode):
        keys: set[tuple[str, str]] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise yaml.YAMLError("LLM config mapping keys must be scalar values")
            identity = (key_node.tag, key_node.value)
            if identity in keys:
                raise yaml.YAMLError(f"duplicate LLM config key {key_node.value!r}")
            keys.add(identity)
            _reject_duplicate_yaml_keys(key_node, seen)
            _reject_duplicate_yaml_keys(value_node, seen)
        return
    if isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _reject_duplicate_yaml_keys(child, seen)


def _reject_yaml_secret_fields(value: Any) -> None:
    """Reject unsupported credential aliases anywhere in a configuration document."""

    stack = [value]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, (dict, list)):
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(item, dict):
            if any(
                isinstance(key, str) and key.strip().lower() in _UNSUPPORTED_CREDENTIAL_FIELDS
                for key in item
            ):
                raise LLMProviderError(
                    "unsupported LLM credential field; use exactly one of api_key or api_key_env",
                    stage="configuration",
                    failure_kind="inline_secret_forbidden",
                )
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _load_yaml_config(path: Path, *, requested_profile: str | None) -> LLMProviderConfig:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_LLM_CONFIG_BYTES + 1)
        if len(raw) > MAX_LLM_CONFIG_BYTES:
            raise LLMProviderError(
                f"LLM config {path} exceeds the size limit",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        text = raw.decode("utf-8")
        document = yaml.compose(text, Loader=yaml.SafeLoader)
        _reject_duplicate_yaml_keys(document)
        loaded = yaml.safe_load(text) or {}
    except LLMProviderError:
        raise
    except (
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
        yaml.YAMLError,
    ) as exc:
        raise LLMProviderError(
            f"failed to load LLM config {path}: {type(exc).__name__}",
            stage="configuration",
            failure_kind="invalid_configuration",
        ) from exc
    if not isinstance(loaded, dict):
        raise LLMProviderError(
            f"LLM config {path} must be a mapping",
            stage="configuration",
            failure_kind="invalid_configuration",
        )
    _reject_yaml_secret_fields(loaded)

    profiles = loaded.get("profiles")
    if profiles is None:
        values = loaded
        selected_profile = requested_profile
    else:
        unexpected_root = sorted(set(loaded) - {"active_profile", "profiles"}, key=str)
        if unexpected_root:
            raise LLMProviderError(
                f"unknown profiled LLM config fields: {unexpected_root}",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if not isinstance(profiles, dict):
            raise LLMProviderError(
                f"LLM config {path} profiles must be a mapping",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        raw_profile = (
            requested_profile if requested_profile is not None else loaded.get("active_profile")
        )
        if raw_profile is None or raw_profile == "":
            raise LLMProviderError(
                f"LLM config {path} has profiles but no active_profile",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        if not isinstance(raw_profile, str):
            raise LLMProviderError(
                "LLM profile name must be text",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        selected_profile = raw_profile.strip()
        if _PROFILE_NAME.fullmatch(selected_profile) is None:
            raise LLMProviderError(
                "LLM profile name must be a safe identifier",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        values = profiles.get(selected_profile)
        if not isinstance(values, dict):
            raise LLMProviderError(
                "selected LLM profile was not found",
                stage="configuration",
                failure_kind="invalid_configuration",
            )
    return _config_from_values(
        values,
        source=str(path.resolve()),
        profile=selected_profile,
    )


def _config_path(value: str | Path) -> Path:
    try:
        return Path(value).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LLMProviderError(
            "LLM config path is invalid",
            stage="configuration",
            failure_kind="invalid_configuration",
        ) from exc


def load_llm_provider_config(
    path: str | Path | None = None, *, profile: str | None = None
) -> LLMProviderConfig:
    """Resolve direct environment settings, then an explicit or discovered YAML file.

    Direct settings win only as a complete group. YAML resolution is explicit
    ``path`` -> ``GENENV_LLM_CONFIG`` -> ``X2ENV_LLM_CONFIG`` ->
    ``SCENE_GEN_LLM_CONFIG`` -> ``configs/llm.yaml``.
    """

    direct_names = ("GENENV_LLM_ENDPOINT", "GENENV_LLM_API_KEY", "GENENV_LLM_MODEL")
    direct = {name: os.environ.get(name, "").strip() for name in direct_names}
    present = [name for name, value in direct.items() if value]
    if present:
        missing = [name for name, value in direct.items() if not value]
        if missing:
            raise LLMProviderError(
                "incomplete direct LLM environment; missing " + ", ".join(missing),
                stage="configuration",
                failure_kind="invalid_configuration",
            )
        return _config_from_values(
            {
                "endpoint": direct["GENENV_LLM_ENDPOINT"],
                "model": direct["GENENV_LLM_MODEL"],
                "api_key_env": "GENENV_LLM_API_KEY",
                "api_mode": os.environ.get("GENENV_LLM_API_MODE", "chat"),
                "timeout_s": os.environ.get("GENENV_LLM_TIMEOUT_S", ""),
                "max_attempts": os.environ.get("GENENV_LLM_MAX_ATTEMPTS", ""),
                "cache_dir": os.environ.get("GENENV_LLM_CACHE_DIR", ""),
                "temperature": os.environ.get("GENENV_LLM_TEMPERATURE", ""),
            },
            source="environment",
            profile=None,
            direct_api_key=direct["GENENV_LLM_API_KEY"],
        )

    selected: Path | None
    if path is not None:
        selected = _config_path(path)
    else:
        configured = (
            os.environ.get("GENENV_LLM_CONFIG", "").strip()
            or os.environ.get("X2ENV_LLM_CONFIG", "").strip()
            or os.environ.get("SCENE_GEN_LLM_CONFIG", "").strip()
        )
        selected = _config_path(configured) if configured else None
        if selected is None:
            repository_default = Path(__file__).resolve().parents[1] / "configs" / "llm.yaml"
            if repository_default.is_file():
                selected = repository_default
    if selected is None or not selected.is_file():
        location = str(selected) if selected is not None else "configs/llm.yaml"
        raise LLMProviderError(
            "LLM provider is not configured; set GENENV_LLM_ENDPOINT, "
            "GENENV_LLM_API_KEY and GENENV_LLM_MODEL together, or create "
            f"{location} from configs/llm.example.yaml",
            stage="configuration",
            failure_kind="missing_configuration",
        )
    selected_profile = profile or os.environ.get("GENENV_LLM_PROFILE", "").strip() or None
    return _load_yaml_config(selected, requested_profile=selected_profile)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_prompt(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMProviderError(
            f"LLM prompt is unavailable: {path.name}",
            stage="configuration",
            failure_kind="missing_prompt",
        ) from exc
    return text, _sha256_text(text)


def _reject_json_constant(value: str) -> None:
    raise _StageOutputError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _StageOutputError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _strict_json_loads(text: str) -> Any:
    value = json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 64 or nodes > 100_000:
            raise _StageOutputError("JSON structure exceeds complexity limits")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = _strict_json_loads(cleaned)
    except _StageOutputError:
        raise
    except (ValueError, RecursionError, OverflowError) as exc:
        raise _StageOutputError("response is not one strict JSON object") from exc
    if not isinstance(value, dict):
        raise _StageOutputError("response JSON root must be an object")
    return value


def _response_text(data: dict[str, Any], *, api_mode: str) -> str:
    if api_mode == "chat":
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _TransportError("chat response is missing choices[0].message.content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text") for item in content if isinstance(item, dict)]
            if parts and all(isinstance(item, str) for item in parts):
                return "".join(parts)
        raise _TransportError("chat response content is not text")

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    parts: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    if not parts:
        raise _TransportError("responses result is missing output text")
    return "\n".join(parts)


def _http_transport(config: LLMProviderConfig, system: str, user: str) -> str:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    if config.api_mode == "chat":
        url = (
            config.endpoint
            if config.endpoint.endswith("/chat/completions")
            else f"{config.endpoint}/chat/completions"
        )
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
    else:
        url = (
            config.endpoint
            if config.endpoint.endswith("/responses")
            else f"{config.endpoint}/responses"
        )
        payload = {
            "model": config.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": 4096,
            "store": False,
        }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener.open(request, timeout=config.timeout_s) as response:
            raw_bytes = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
        if len(raw_bytes) > MAX_PROVIDER_RESPONSE_BYTES:
            raise _TransportError("provider response exceeds the size limit")
        raw = raw_bytes.decode("utf-8")
        data = _strict_json_loads(raw)
    except urllib.error.HTTPError as exc:
        raise _TransportError(f"provider returned HTTP {exc.code}") from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        socket.timeout,
        OSError,
        OverflowError,
    ) as exc:
        raise _TransportError(f"provider request failed: {type(exc).__name__}") from exc
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _TransportError("provider response envelope is not valid JSON") from exc
    if not isinstance(data, dict):
        raise _TransportError("provider response envelope must be a JSON object")
    return _response_text(data, api_mode=config.api_mode)


def _canonical_term(value: Any, lexicon: dict[str, tuple[str, ...]]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _StageOutputError("color and material values must be text or null")
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    for canonical, variants in lexicon.items():
        if normalized == canonical or normalized in variants:
            return canonical
    return None


def _region_is_authorized(request: str, region: str) -> bool:
    if region == "center":
        return True
    normalized = re.sub(r"\s+", " ", request.strip().lower())
    return any(term.lower() in normalized for term in REGION_TERMS[region])


def _clean_ambiguities(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _StageOutputError("ambiguities must be an array")
    ambiguities: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _StageOutputError("each ambiguity must be non-empty text")
        ambiguities.append(item.strip()[:500])
    return ambiguities


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _StageOutputError(f"{context} contains unsupported fields: {unknown}")


def _clean_articulation(value: Any, *, category: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _StageOutputError("articulation must be an object or null")
    _reject_unknown_keys(
        value,
        {"state", "open_fraction", "joint_selector"},
        context="articulation",
    )
    if category not in ARTICULATED_CATEGORIES:
        raise _StageOutputError(f"category {category!r} does not support articulation")
    joint_selector = value.get("joint_selector", "all_movable")
    if not isinstance(joint_selector, str) or joint_selector != "all_movable":
        raise _StageOutputError("articulation joint_selector must be all_movable")
    state = value.get("state")
    if not isinstance(state, str) or state not in {"closed", "open", "partially_open"}:
        raise _StageOutputError(f"unsupported articulation state {state!r}")

    raw_fraction = value.get("open_fraction")
    if raw_fraction is None:
        supplied_fraction = None
    else:
        if isinstance(raw_fraction, bool):
            raise _StageOutputError("articulation open_fraction must be a finite number")
        try:
            supplied_fraction = float(raw_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _StageOutputError("articulation open_fraction must be a finite number") from exc
        if not math.isfinite(supplied_fraction):
            raise _StageOutputError("articulation open_fraction must be a finite number")

    expected_fraction = 0.0 if state == "closed" else 1.0 if state == "open" else None
    if expected_fraction is not None:
        if supplied_fraction is not None and supplied_fraction != expected_fraction:
            raise _StageOutputError(
                f"{state} articulation requires open_fraction={expected_fraction:g}"
            )
        fraction = expected_fraction
    else:
        fraction = 0.5 if supplied_fraction is None else supplied_fraction
        if not 0.0 < fraction < 1.0:
            raise _StageOutputError(
                "partially_open articulation requires open_fraction within (0, 1)"
            )
    return {
        "state": state,
        "open_fraction": fraction,
        "joint_selector": "all_movable",
    }


def _clean_objects(document: dict[str, Any], *, request: str) -> dict[str, Any]:
    ambiguities = _clean_ambiguities(document.get("ambiguities"))
    if ambiguities:
        raise _ReportedAmbiguity(ambiguities)
    _reject_unknown_keys(document, {"objects", "ambiguities"}, context="objects stage response")
    raw_objects = document.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise _StageOutputError("objects must be a non-empty array")
    if len(raw_objects) > MAX_SCENE_OBJECTS:
        raise _StageOutputError(f"objects exceeds the limit of {MAX_SCENE_OBJECTS}")
    counts: dict[str, int] = {}
    seen: set[str] = set()
    objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        if not isinstance(raw, dict):
            raise _StageOutputError("every object must be a JSON object")
        _reject_unknown_keys(
            raw,
            {"object_id", "category", "color", "material", "region", "articulation"},
            context="object",
        )
        object_id = raw.get("object_id")
        category = raw.get("category")
        if not isinstance(object_id, str) or not _IDENTIFIER.fullmatch(object_id):
            raise _StageOutputError(f"invalid object_id {object_id!r}")
        if not isinstance(category, str) or not _CATEGORY.fullmatch(category):
            raise _StageOutputError(f"invalid category {category!r}")
        category_tokens = set(category.split("_"))
        if category_tokens & _RESERVED_OBJECT_CATEGORIES:
            raise _StageOutputError(f"reserved context category {category!r} is not a scene object")
        if category_tokens & {"and", "or", "plus"}:
            raise _StageOutputError(f"category {category!r} merges multiple object phrases")
        prefix = f"{category}_"
        instance = object_id[len(prefix) :] if object_id.startswith(prefix) else ""
        if re.fullmatch(r"[1-9][0-9]*", instance) is None:
            raise _StageOutputError(
                f"object_id {object_id!r} must use the form {category!r}_<positive integer>"
            )
        counts[category] = counts.get(category, 0) + 1
        if object_id in seen:
            raise _StageOutputError(f"duplicate object_id {object_id!r}")
        seen.add(object_id)
        raw_region = raw.get("region")
        if raw_region is None:
            region = "center"
        elif not isinstance(raw_region, str):
            raise _StageOutputError("table region must be text or null")
        else:
            region = raw_region
        if region not in REGION_TERMS:
            raise _StageOutputError(f"unsupported table region {region!r}")
        if not _region_is_authorized(request, region):
            region = "center"
        objects.append(
            {
                "object_id": object_id,
                "category": category,
                "color": _canonical_term(raw.get("color"), COLOR_TERMS),
                "material": _canonical_term(raw.get("material"), MATERIAL_TERMS),
                "region": region,
                "articulation": _clean_articulation(raw.get("articulation"), category=category),
            }
        )
    for category, count in sorted(counts.items()):
        actual_ids = {item["object_id"] for item in objects if item["category"] == category}
        expected_ids = {f"{category}_{index}" for index in range(1, count + 1)}
        if actual_ids != expected_ids:
            raise _StageOutputError(
                f"category {category!r} must use deterministic ids {sorted(expected_ids)!r}"
            )
    semantic_checks = validate_object_extraction(request, objects)
    object_rank = {
        object_id: index for index, object_id in enumerate(semantic_checks["object_order"])
    }
    objects.sort(key=lambda item: object_rank[item["object_id"]])
    return {
        "objects": objects,
        "ambiguities": ambiguities,
        "semantic_checks": semantic_checks,
    }


def _distance(value: Any, *, name: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise _StageOutputError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _StageOutputError(f"{name} must be a number") from exc
    if not 0.0 < result <= 1.0:
        raise _StageOutputError(f"{name} must be within (0, 1]")
    return result


def _clean_relation(
    raw: Any, *, allowed: set[str], object_ids: set[str], topology: bool
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _StageOutputError("every relation must be a JSON object")
    relation = raw.get("relation")
    allowed_keys = {"relation", "source", "target"}
    if relation == RelationType.NEAR.value:
        allowed_keys.add("max_distance_m")
    elif relation == RelationType.DISTANCE_AT_LEAST.value:
        allowed_keys.add("min_distance_m")
    _reject_unknown_keys(raw, allowed_keys, context="relation")
    source = raw.get("source")
    target = raw.get("target")
    if not isinstance(relation, str) or relation not in allowed:
        raise _StageOutputError(f"unsupported relation {relation!r}")
    if not isinstance(source, str) or source not in object_ids:
        raise _StageOutputError(f"unknown relation source {source!r}")
    if not isinstance(target, str):
        raise _StageOutputError(f"invalid relation target {target!r}")
    if source == target:
        raise _StageOutputError("self-relations are forbidden")
    if relation == RelationType.ON_TABLE.value:
        if target != "table":
            raise _StageOutputError("on_table must target table")
    elif target not in object_ids:
        raise _StageOutputError(f"unknown relation target {target!r}")
    cleaned: dict[str, Any] = {
        "relation": relation,
        "source": source,
        "target": target,
    }
    if not topology and relation == RelationType.NEAR.value:
        cleaned["max_distance_m"] = _distance(
            raw.get("max_distance_m"), name="max_distance_m", default=0.25
        )
    elif not topology and relation == RelationType.DISTANCE_AT_LEAST.value:
        cleaned["min_distance_m"] = _distance(raw.get("min_distance_m"), name="min_distance_m")
    return cleaned


def _clean_relations(
    document: dict[str, Any],
    *,
    objects: list[dict[str, Any]],
    request: str,
    seed: int,
) -> dict[str, Any]:
    ambiguities = _clean_ambiguities(document.get("ambiguities"))
    if ambiguities:
        raise _ReportedAmbiguity(ambiguities)
    _reject_unknown_keys(
        document, {"topology", "lateral", "ambiguities"}, context="relations stage response"
    )
    raw_topology = document.get("topology")
    raw_lateral = document.get("lateral")
    if not isinstance(raw_topology, list):
        raise _StageOutputError("topology must be an array")
    if not isinstance(raw_lateral, list):
        raise _StageOutputError("lateral must be an array")
    object_ids = {item["object_id"] for item in objects}
    topology = [
        _clean_relation(raw, allowed=_TOPOLOGY, object_ids=object_ids, topology=True)
        for raw in raw_topology
    ]
    lateral = [
        _clean_relation(raw, allowed=_LATERAL, object_ids=object_ids, topology=False)
        for raw in raw_lateral
    ]
    object_rank = {item["object_id"]: index for index, item in enumerate(objects)}
    topology.sort(
        key=lambda item: (
            object_rank[item["source"]],
            item["relation"],
            object_rank.get(item["target"], -1),
            item["target"],
        )
    )
    lateral.sort(
        key=lambda item: (
            object_rank[item["source"]],
            item["relation"],
            object_rank[item["target"]],
            item.get("max_distance_m", 0.0),
            item.get("min_distance_m", 0.0),
        )
    )
    supports: dict[str, str] = {}
    for relation in topology:
        source = relation["source"]
        if source in supports:
            raise _StageOutputError(f"object {source!r} has more than one topology relation")
        supports[source] = relation["target"]
    missing = sorted(object_ids - set(supports))
    if missing:
        raise _StageOutputError(f"objects are missing topology relations: {missing}")
    for relation in lateral:
        source_support = supports[relation["source"]]
        target_support = supports[relation["target"]]
        if source_support != target_support:
            raise _StageOutputError(
                "lateral relation endpoints must share the same immediate support"
            )
    relation_keys = [
        (
            item["relation"],
            item["source"],
            item["target"],
            item.get("max_distance_m"),
            item.get("min_distance_m"),
        )
        for item in (*topology, *lateral)
    ]
    if len(relation_keys) != len(set(relation_keys)):
        raise _StageOutputError("duplicate relations are forbidden")
    try:
        parse_provider_payload(
            {"objects": objects, "relations": [*topology, *lateral]},
            request=request,
            seed=seed,
        )
    except (SceneSpecError, ValidationError) as exc:
        raise _StageOutputError(f"relations violate SceneSpec: {exc}") from exc
    semantic_checks = validate_relation_extraction(
        request,
        [*topology, *lateral],
        objects,
    )
    return {
        "topology": topology,
        "lateral": lateral,
        "ambiguities": ambiguities,
        "semantic_checks": semantic_checks,
    }


class LLMSceneProvider:
    """Two-stage objects/relations extractor for ``parse_with_provider``."""

    def __init__(
        self,
        config: LLMProviderConfig | None = None,
        *,
        config_path: str | Path | None = None,
        profile: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or load_llm_provider_config(config_path, profile=profile)
        self._objects_prompt, objects_hash = _read_prompt(_OBJECTS_PROMPT)
        self._relations_prompt, relations_hash = _read_prompt(_RELATIONS_PROMPT)
        self._prompt_hashes = {
            "llm_objects.md": objects_hash,
            "llm_relations.md": relations_hash,
        }
        self._transport = transport
        self.last_evidence: dict[str, Any] = {}

    def _send(self, system: str, user: str) -> str:
        if self._transport is not None:
            try:
                return self._transport(system, user)
            except _TransportError:
                raise
            except Exception as exc:
                raise _TransportError(f"custom transport failed: {type(exc).__name__}") from exc
        return _http_transport(self.config, system, user)

    def _cache_key(self, *, request: str, seed: int) -> str:
        return _sha256_text(
            _canonical_json(
                {
                    "provider_version": PROVIDER_VERSION,
                    "semantic_check_version": SEMANTIC_CHECK_VERSION,
                    "request": request,
                    "seed": seed,
                    "config_fingerprint": self.config.fingerprint(),
                    "prompt_hashes": self._prompt_hashes,
                }
            )
        )

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir.expanduser() / f"{key}.json"

    def _success_evidence(
        self,
        *,
        key: str,
        payload: dict[str, Any],
        objects_stage: dict[str, Any],
        relations_stage: dict[str, Any],
        object_attempts: int,
        relation_attempts: int,
    ) -> dict[str, Any]:
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "pass",
            "provider_version": PROVIDER_VERSION,
            "semantic_check_version": SEMANTIC_CHECK_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "model": self.config.model,
            "api_mode": self.config.api_mode,
            "prompt_hashes": self._prompt_hashes,
            "cache": {"key": key},
            "stages": {
                "objects": {
                    "attempts": object_attempts,
                    "objects": objects_stage["objects"],
                    "ambiguities": [],
                    "semantic_checks": objects_stage["semantic_checks"],
                },
                "relations": {
                    "attempts": relation_attempts,
                    "topology": relations_stage["topology"],
                    "lateral": relations_stage["lateral"],
                    "ambiguities": [],
                    "semantic_checks": relations_stage["semantic_checks"],
                },
            },
            "payload_sha256": _sha256_text(_canonical_json(payload)),
        }
        return json.loads(json.dumps(evidence))

    def _read_cache(self, key: str, *, request: str, seed: int) -> dict[str, Any] | None:
        path = self._cache_path(key)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_PARSE_CACHE_BYTES + 1)
            if len(raw) > MAX_PARSE_CACHE_BYTES:
                return None
            cached = _strict_json_loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, RecursionError, OverflowError):
            return None
        if (
            not isinstance(cached, dict)
            or set(cached)
            != {"schema_version", "cache_key", "payload_sha256", "payload", "evidence"}
            or cached.get("schema_version") != "robotwin.llm_parse_cache.v1"
            or cached.get("cache_key") != key
        ):
            return None
        payload = cached.get("payload")
        evidence = cached.get("evidence")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"objects", "relations"}
            or not isinstance(evidence, dict)
        ):
            return None
        payload_sha256 = _sha256_text(_canonical_json(payload))
        if cached.get("payload_sha256") != payload_sha256:
            return None
        evidence_fields = {
            "schema_version",
            "status",
            "provider_version",
            "semantic_check_version",
            "config_fingerprint",
            "model",
            "api_mode",
            "prompt_hashes",
            "cache",
            "stages",
            "payload_sha256",
        }
        if set(evidence) != evidence_fields:
            return None
        stages = evidence.get("stages")
        if not isinstance(stages, dict) or set(stages) != {"objects", "relations"}:
            return None
        objects_stage = stages.get("objects")
        relations_stage = stages.get("relations")
        if not isinstance(objects_stage, dict) or not isinstance(relations_stage, dict):
            return None
        if set(objects_stage) != {"attempts", "objects", "ambiguities", "semantic_checks"}:
            return None
        if set(relations_stage) != {
            "attempts",
            "topology",
            "lateral",
            "ambiguities",
            "semantic_checks",
        }:
            return None
        object_attempts = objects_stage.get("attempts")
        relation_attempts = relations_stage.get("attempts")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= self.config.max_attempts
            for value in (object_attempts, relation_attempts)
        ):
            return None
        if objects_stage.get("ambiguities") != [] or relations_stage.get("ambiguities") != []:
            return None
        if (
            not isinstance(objects_stage.get("objects"), list)
            or not isinstance(relations_stage.get("topology"), list)
            or not isinstance(relations_stage.get("lateral"), list)
        ):
            return None
        try:
            cleaned_objects = _clean_objects(
                {"objects": objects_stage["objects"], "ambiguities": []},
                request=request,
            )
            cleaned_relations = _clean_relations(
                {
                    "topology": relations_stage["topology"],
                    "lateral": relations_stage["lateral"],
                    "ambiguities": [],
                },
                objects=cleaned_objects["objects"],
                request=request,
                seed=seed,
            )
        except (SceneSpecError, ValidationError, _StageOutputError, SemanticMismatch, TypeError):
            return None
        staged_payload = {
            "objects": cleaned_objects["objects"],
            "relations": [
                *cleaned_relations["topology"],
                *cleaned_relations["lateral"],
            ],
        }
        if staged_payload != payload:
            return None
        rebuilt = self._success_evidence(
            key=key,
            payload=payload,
            objects_stage=cleaned_objects,
            relations_stage=cleaned_relations,
            object_attempts=object_attempts,
            relation_attempts=relation_attempts,
        )
        if evidence != rebuilt:
            return None
        self.last_evidence = json.loads(json.dumps(rebuilt))
        return payload

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "robotwin.llm_parse_cache.v1",
            "cache_key": key,
            "payload_sha256": _sha256_text(_canonical_json(payload)),
            "payload": payload,
            "evidence": self.last_evidence,
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, indent=2, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except Exception:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _run_stage(
        self,
        *,
        stage: str,
        system: str,
        user_payload: dict[str, Any],
        cleaner: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any], int]:
        last_reason = "unknown provider failure"
        for attempt in range(1, self.config.max_attempts + 1):
            payload = dict(user_payload)
            if attempt > 1:
                payload["retry_feedback"] = last_reason
            try:
                response = self._send(system, _canonical_json(payload))
                if not isinstance(response, str):
                    raise _StageOutputError("provider stage content must be text")
                try:
                    response_size = len(response.encode("utf-8"))
                except UnicodeError as exc:
                    raise _StageOutputError(
                        "provider stage content is not valid UTF-8 text"
                    ) from exc
                if response_size > MAX_PROVIDER_RESPONSE_BYTES:
                    raise _StageOutputError("provider stage content exceeds the size limit")
                document = _parse_json_object(response)
                return cleaner(document), attempt
            except _ReportedAmbiguity as exc:
                self.last_evidence = {
                    "schema_version": EVIDENCE_SCHEMA,
                    "status": "fail",
                    "provider_version": PROVIDER_VERSION,
                    "stage": stage,
                    "attempts": attempt,
                    "failure_kind": "ambiguous_request",
                    "ambiguities": exc.ambiguities,
                    "config_fingerprint": self.config.fingerprint(),
                    "prompt_hashes": self._prompt_hashes,
                }
                raise LLMProviderError(
                    f"LLM reported ambiguity during {stage} extraction: {exc}",
                    stage=stage,
                    attempts=attempt,
                    failure_kind="ambiguous_request",
                ) from exc
            except AmbiguousReference as exc:
                self.last_evidence = {
                    "schema_version": EVIDENCE_SCHEMA,
                    "status": "fail",
                    "provider_version": PROVIDER_VERSION,
                    "stage": stage,
                    "attempts": attempt,
                    "failure_kind": "ambiguous_request",
                    "ambiguities": [str(exc)[:500]],
                    "config_fingerprint": self.config.fingerprint(),
                    "prompt_hashes": self._prompt_hashes,
                }
                raise LLMProviderError(
                    f"ambiguous request during {stage} extraction: {exc}",
                    stage=stage,
                    attempts=attempt,
                    failure_kind="ambiguous_request",
                ) from exc
            except (_TransportError, _StageOutputError, SemanticMismatch) as exc:
                last_reason = str(exc)[:500]
        self.last_evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "fail",
            "provider_version": PROVIDER_VERSION,
            "stage": stage,
            "attempts": self.config.max_attempts,
            "failure_kind": "attempts_exhausted",
            "config_fingerprint": self.config.fingerprint(),
            "prompt_hashes": self._prompt_hashes,
        }
        raise LLMProviderError(
            f"LLM {stage} extraction failed after {self.config.max_attempts} attempts: "
            f"{last_reason}",
            stage=stage,
            attempts=self.config.max_attempts,
            failure_kind="attempts_exhausted",
        )

    def parse_scene(self, *, request: str, seed: int) -> dict[str, Any]:
        self.last_evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "fail",
            "provider_version": PROVIDER_VERSION,
            "stage": "initialization",
            "attempts": 0,
            "failure_kind": "request_in_progress",
            "config_fingerprint": self.config.fingerprint(),
            "prompt_hashes": self._prompt_hashes,
        }
        try:
            return self._parse_scene(request=request, seed=seed)
        except LLMProviderError:
            raise
        except Exception as exc:
            self.last_evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "status": "fail",
                "provider_version": PROVIDER_VERSION,
                "stage": "provider",
                "attempts": 0,
                "failure_kind": "unexpected_error",
                "config_fingerprint": self.config.fingerprint(),
                "prompt_hashes": self._prompt_hashes,
            }
            raise LLMProviderError(
                f"unexpected LLM provider failure: {type(exc).__name__}",
                stage="provider",
                failure_kind="unexpected_error",
            ) from exc

    def _parse_scene(self, *, request: str, seed: int) -> dict[str, Any]:
        boundary_error: str | None = None
        boundary_kind = "unsupported_request"
        try:
            validate_prompt_boundary(request)
        except SceneSpecError as exc:
            boundary_error = str(exc)
        if boundary_error is None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647
        ):
            boundary_error = "seed must be an integer in [0, 2147483647]"
        if boundary_error is None:
            try:
                reject_unsupported_request_semantics(request)
            except AmbiguousReference as exc:
                boundary_error = str(exc)
                boundary_kind = "ambiguous_request"
            except SemanticMismatch as exc:
                boundary_error = str(exc)
        if boundary_error is not None:
            self.last_evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "status": "fail",
                "provider_version": PROVIDER_VERSION,
                "stage": "semantic_precheck",
                "attempts": 0,
                "failure_kind": boundary_kind,
                "config_fingerprint": self.config.fingerprint(),
                "prompt_hashes": self._prompt_hashes,
            }
            raise LLMProviderError(
                boundary_error,
                stage="semantic_precheck",
                failure_kind=boundary_kind,
            )

        key = self._cache_key(request=request, seed=seed)
        cached = self._read_cache(key, request=request, seed=seed)
        if cached is not None:
            return cached

        objects_stage, object_attempts = self._run_stage(
            stage="objects",
            system=self._objects_prompt,
            user_payload={"request": request},
            cleaner=lambda value: _clean_objects(value, request=request),
        )
        if objects_stage["ambiguities"]:
            self.last_evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "status": "fail",
                "provider_version": PROVIDER_VERSION,
                "stage": "objects",
                "failure_kind": "ambiguous_request",
                "ambiguities": objects_stage["ambiguities"],
                "config_fingerprint": self.config.fingerprint(),
                "prompt_hashes": self._prompt_hashes,
            }
            raise LLMProviderError(
                "LLM reported an ambiguous object extraction: "
                + "; ".join(objects_stage["ambiguities"]),
                stage="objects",
                attempts=object_attempts,
                failure_kind="ambiguous_request",
            )

        object_ids = [item["object_id"] for item in objects_stage["objects"]]
        relations_stage, relation_attempts = self._run_stage(
            stage="relations",
            system=self._relations_prompt,
            user_payload={"request": request, "object_ids": object_ids},
            cleaner=lambda value: _clean_relations(
                value,
                objects=objects_stage["objects"],
                request=request,
                seed=seed,
            ),
        )
        if relations_stage["ambiguities"]:
            self.last_evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "status": "fail",
                "provider_version": PROVIDER_VERSION,
                "stage": "relations",
                "failure_kind": "ambiguous_request",
                "ambiguities": relations_stage["ambiguities"],
                "config_fingerprint": self.config.fingerprint(),
                "prompt_hashes": self._prompt_hashes,
            }
            raise LLMProviderError(
                "LLM reported an ambiguous relation extraction: "
                + "; ".join(relations_stage["ambiguities"]),
                stage="relations",
                attempts=relation_attempts,
                failure_kind="ambiguous_request",
            )

        payload = {
            "objects": objects_stage["objects"],
            "relations": [
                *relations_stage["topology"],
                *relations_stage["lateral"],
            ],
        }
        parse_provider_payload(payload, request=request, seed=seed)
        self.last_evidence = self._success_evidence(
            key=key,
            payload=payload,
            objects_stage=objects_stage,
            relations_stage=relations_stage,
            object_attempts=object_attempts,
            relation_attempts=relation_attempts,
        )
        try:
            self._write_cache(key, payload)
        except OSError as exc:
            self.last_evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "status": "fail",
                "provider_version": PROVIDER_VERSION,
                "stage": "cache",
                "attempts": 0,
                "failure_kind": "cache_write_failed",
                "config_fingerprint": self.config.fingerprint(),
                "prompt_hashes": self._prompt_hashes,
                "payload_sha256": _sha256_text(_canonical_json(payload)),
            }
            raise LLMProviderError(
                f"failed to write complete LLM parse cache: {type(exc).__name__}",
                stage="cache",
                failure_kind="cache_write_failed",
            ) from exc
        return payload

    def evidence(self) -> dict[str, Any]:
        """Return a defensive copy of the latest secret-free extraction evidence."""

        return json.loads(json.dumps(self.last_evidence))

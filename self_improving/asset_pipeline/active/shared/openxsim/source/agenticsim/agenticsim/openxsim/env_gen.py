"""Import env-gen resolved_scene.json into the Open-X-Sim IR (first-class importer)."""

from __future__ import annotations

import hashlib
import json
import shlex
import struct
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .importers import EnvironmentImportError, _task_from_contract, _valid_identifier
from .ir import (
    AssetBundle,
    AssetRepresentation,
    EnvironmentPackage,
    EnvSpec,
    Pose,
    SceneObject,
)

_MESH_SUFFIXES = {"glb", "obj", "dae", "stl", "ply", "usd", "usda", "usdc"}
_GENESIS_MESH_SUFFIXES = {"obj", "stl", "dae", "glb", "gltf"}
_DEPENDENCY_DISCOVERY = "env_gen.local_asset_dependencies.v1"
_MTL_TEXTURE_DIRECTIVES = {
    "bump",
    "decal",
    "disp",
    "map_bump",
    "map_d",
    "map_ka",
    "map_kd",
    "map_ke",
    "map_ks",
    "map_ns",
    "norm",
    "refl",
}


def _is_env_gen(data: dict[str, Any]) -> bool:
    return str(data.get("compiler_version", "")).startswith("scene_gen")


def _file_fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _fingerprinted_representation(
    *,
    fmt: str,
    resolved: Path,
    backend: str,
    role: str = "visual_and_collision",
    metadata: dict[str, Any] | None = None,
) -> AssetRepresentation:
    try:
        sha256, size_bytes = _file_fingerprint(resolved)
    except OSError as exc:
        raise EnvironmentImportError(
            f"env-gen asset fingerprint failed for {resolved}: {exc}"
        ) from exc
    return AssetRepresentation(
        format=fmt,
        uri=str(resolved),
        backend=backend,
        role=role,
        sha256=sha256,
        size_bytes=size_bytes,
        metadata=dict(metadata or {}),
    )


def _tokenized_lines(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [
        shlex.split(line, comments=True, posix=True)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _gltf_references(document: Any) -> list[str]:
    if not isinstance(document, dict):
        raise ValueError("glTF JSON root must be an object")
    references: list[str] = []
    for section in ("buffers", "images"):
        values = document.get(section) or []
        if not isinstance(values, list):
            raise ValueError(f"glTF {section} must be an array")
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("uri"), str):
                references.append(value["uri"])
    return references


def _glb_references(path: Path) -> list[str]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise ValueError("invalid GLB header")
    version, declared_size = struct.unpack_from("<II", payload, 4)
    if version != 2:
        raise ValueError(f"unsupported GLB version: {version}")
    if declared_size != len(payload):
        raise ValueError(
            f"GLB declared size {declared_size} does not match file size {len(payload)}"
        )
    offset = 12
    document: Any | None = None
    while offset < declared_size:
        if offset + 8 > declared_size:
            raise ValueError("truncated GLB chunk header")
        chunk_size, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_size
        if end > declared_size:
            raise ValueError("truncated GLB chunk payload")
        if chunk_type == 0x4E4F534A and document is None:
            text = payload[offset:end].decode("utf-8").rstrip("\x00 \t\r\n")
            document = json.loads(text)
        offset = end
    if document is None:
        raise ValueError("GLB has no JSON chunk")
    return _gltf_references(document)


def _dependency_references(path: Path) -> tuple[list[str], list[str]]:
    """Return direct asset references and deterministic parse errors for one file."""

    suffix = path.suffix.lower()
    try:
        if suffix == ".obj":
            references = [
                value
                for tokens in _tokenized_lines(path)
                if tokens and tokens[0].lower() == "mtllib"
                for value in tokens[1:]
            ]
        elif suffix == ".mtl":
            references = [
                tokens[-1]
                for tokens in _tokenized_lines(path)
                if len(tokens) >= 2 and tokens[0].lower() in _MTL_TEXTURE_DIRECTIVES
            ]
        elif suffix == ".gltf":
            references = _gltf_references(json.loads(path.read_text(encoding="utf-8-sig")))
        elif suffix == ".glb":
            references = _glb_references(path)
        elif suffix == ".dae":
            root = ET.parse(path).getroot()
            references = [
                str(child.text).strip()
                for image in root.iter()
                if image.tag.rsplit("}", 1)[-1].lower() == "image"
                for child in image
                if child.tag.rsplit("}", 1)[-1].lower() == "init_from"
                and child.text
                and str(child.text).strip()
            ]
        elif suffix == ".urdf":
            root = ET.parse(path).getroot()
            references = [
                str(node.attrib["filename"]).strip()
                for node in root.iter()
                if node.tag.rsplit("}", 1)[-1].lower() in {"mesh", "texture"}
                and node.attrib.get("filename")
            ]
        else:
            references = []
    except (ET.ParseError, json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
        return [], [f"dependency parse failed for {path}: {type(exc).__name__}: {exc}"]
    return sorted(set(references)), []


def _resolve_dependency(reference: str, owner: Path) -> tuple[Path | None, str | None]:
    value = reference.strip()
    if not value or value.startswith("#") or value.lower().startswith("data:"):
        return None, None
    try:
        parsed = urlparse(value)
        if parsed.scheme:
            if parsed.scheme.lower() != "file":
                return None, f"unsupported dependency URI {value!r} referenced by {owner}"
            if parsed.netloc not in {"", "localhost"}:
                return None, (f"unsupported file URI authority in {value!r} referenced by {owner}")
        if not parsed.path:
            raise ValueError("dependency URI has no path")
        candidate = Path(unquote(parsed.path))
        if not candidate.is_absolute():
            candidate = owner.parent / candidate
        return candidate.expanduser().resolve(), None
    except (OSError, RuntimeError, ValueError) as exc:
        return None, (
            f"dependency URI resolution failed for {value!r} referenced by {owner}: "
            f"{type(exc).__name__}: {exc}"
        )


def _dependency_metadata(primary: Path) -> dict[str, Any]:
    pending = [primary.resolve()]
    visited: set[str] = set()
    dependencies: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    while pending:
        owner = pending.pop()
        owner_uri = str(owner)
        if owner_uri in visited:
            continue
        visited.add(owner_uri)
        references, parse_errors = _dependency_references(owner)
        errors.extend(parse_errors)
        for reference in references:
            dependency, resolution_error = _resolve_dependency(reference, owner)
            if resolution_error:
                errors.append(resolution_error)
                continue
            if dependency is None:
                continue
            dependency_uri = str(dependency)
            if not dependency.is_file():
                errors.append(f"missing dependency {dependency} referenced by {owner}")
                continue
            try:
                sha256, size_bytes = _file_fingerprint(dependency)
            except OSError as exc:
                errors.append(
                    f"dependency fingerprint failed for {dependency}: {type(exc).__name__}: {exc}"
                )
                continue
            if dependency != primary:
                dependencies[dependency_uri] = {
                    "uri": dependency_uri,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
            if dependency_uri not in visited:
                pending.append(dependency)

    return {
        "dependencies": [dependencies[key] for key in sorted(dependencies)],
        "dependency_discovery": _DEPENDENCY_DISCOVERY,
        "dependency_errors": sorted(set(errors)),
    }


def _representation(
    load_type: str, source_files: list[str], base: Path
) -> AssetRepresentation | None:
    def _resolve(p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (base / path).resolve()

    def _make(fmt: str, resolved: Path) -> AssetRepresentation | None:
        if not resolved.is_file():
            return None
        return _fingerprinted_representation(
            fmt=fmt,
            resolved=resolved,
            backend="sapien",
        )

    if load_type == "urdf":
        for value in source_files:
            if value.lower().endswith(".urdf"):
                representation = _make("urdf", _resolve(value))
                if representation is not None:
                    return representation
        return None
    for value in source_files:
        suffix = Path(value).suffix.lower().lstrip(".")
        if suffix in _MESH_SUFFIXES:
            representation = _make(suffix, _resolve(value))
            if representation is not None:
                return representation
    return None


def _genesis_representation(
    load_type: str, source_files: list[str], base: Path
) -> AssetRepresentation | None:
    """Return a render-capable Genesis representation when one is available.

    SAPIEN and Genesis representations are selected independently. Unsupported
    or absent Genesis inputs are left for the Genesis compiler to report as
    backend blockers.
    """

    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base / path).resolve()

    def _role(value: str) -> str:
        parts = {part.lower() for part in value.replace("\\", "/").split("/")}
        if "visual" in parts:
            return "visual"
        if "collision" in parts:
            return "collision"
        return "generic"

    if load_type == "urdf":
        for value in source_files:
            if Path(value).suffix.lower() != ".urdf":
                continue
            resolved = _resolve(value)
            if resolved.is_file():
                return _fingerprinted_representation(
                    fmt="urdf",
                    resolved=resolved,
                    backend="genesis",
                    role="visual_and_collision",
                    metadata=_dependency_metadata(resolved),
                )
        return None

    candidates: list[tuple[int, int, str, str, Path]] = []
    for index, value in enumerate(source_files):
        suffix = Path(value).suffix.lower().lstrip(".")
        if suffix not in _GENESIS_MESH_SUFFIXES:
            continue
        role = _role(value)
        if role == "collision":
            continue
        # A declared visual mesh wins regardless of source_files ordering.
        priority = 0 if role == "visual" else 1
        candidates.append((priority, index, role, suffix, _resolve(value)))

    for _, _, role, suffix, resolved in sorted(candidates):
        if not resolved.is_file():
            continue
        metadata = _dependency_metadata(resolved)
        metadata["file_meshes_are_zup"] = suffix not in {"glb", "gltf"}
        return _fingerprinted_representation(
            fmt=suffix,
            resolved=resolved,
            backend="genesis",
            role="visual" if role == "visual" else "visual_and_collision",
            metadata=metadata,
        )
    return None


def import_env_gen(path: str | Path) -> EnvironmentPackage:
    source = Path(path).expanduser().resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentImportError(f"env-gen scene parse failed for {source}: {exc}") from exc
    if not _is_env_gen(data):
        raise EnvironmentImportError(
            f"not an env-gen resolved_scene (compiler_version={data.get('compiler_version')!r})"
        )

    ws = data.get("workspace") or {}
    x_lo, x_hi = ws.get("x_bounds_m") or [-1.0, 1.0]
    y_lo, y_hi = ws.get("y_bounds_m") or [-1.0, 1.0]
    table = float(ws.get("table_height_m", 0.0))
    workspace_bounds = (
        float(x_lo),
        float(y_lo),
        table,
        float(x_hi),
        float(y_hi),
        table + 0.5,
    )

    objects: list[SceneObject] = []
    assets: list[AssetBundle] = []
    seen: set[str] = set()
    for index, obj in enumerate(data.get("objects") or []):
        instance_id = _valid_identifier(str(obj.get("object_id") or f"object_{index}"), "obj")
        asset_id = _valid_identifier(f"{obj.get('asset_id')}_m{obj.get('model_id', 0)}", "asset")
        pose = obj.get("pose") or {}
        joints = list(obj.get("articulation_joint_names") or [])
        instance_articulation = {
            "joint_names": joints,
            "joint_limits": list(obj.get("articulation_joint_limits") or []),
            "qpos": list(obj.get("articulation_qpos") or []),
            "state": obj.get("articulation_state"),
        }
        objects.append(
            SceneObject(
                instance_id=instance_id,
                asset_id=asset_id,
                pose=Pose(
                    position=tuple(float(v) for v in (pose.get("position_m") or [0.0, 0.0, 0.0])),
                    orientation_wxyz=tuple(
                        float(v) for v in (pose.get("orientation_wxyz") or [1.0, 0.0, 0.0, 0.0])
                    ),
                ),
                static=bool(obj.get("is_static", False)),
                scale=tuple(float(v) for v in (obj.get("mesh_scale") or [1.0, 1.0, 1.0])),
                metadata={
                    "category": obj.get("category"),
                    "color": obj.get("color"),
                    "material": obj.get("material"),
                    "z_policy": obj.get("z_policy") or "origin_on_table",
                    "support_relation": obj.get("support_relation"),
                    "support_target": obj.get("support_target"),
                    "grounding_score": obj.get("grounding_score"),
                    "articulation": instance_articulation,
                },
            )
        )
        if asset_id in seen:
            continue
        seen.add(asset_id)
        load_type = str(obj.get("load_type") or "rigid")
        source_files = list(obj.get("source_files") or [])
        representations: list[AssetRepresentation] = []
        sapien_representation = _representation(
            load_type,
            source_files,
            source.parent,
        )
        if sapien_representation is not None:
            representations.append(sapien_representation)
        genesis_representation = _genesis_representation(
            load_type,
            source_files,
            source.parent,
        )
        if genesis_representation is not None:
            representations.append(genesis_representation)
        if not representations:
            raise EnvironmentImportError(
                f"no existing mesh/urdf representation in source_files: {source_files}"
            )
        assets.append(
            AssetBundle(
                asset_id=asset_id,
                category=str(obj.get("category") or "object"),
                representations=tuple(representations),
                physical={
                    "dimensions_m": obj.get("dimensions_m"),
                    "mass_kg": {"status": "unknown"},
                    "inertia": {"status": "unknown"},
                    "friction": {"status": "unknown"},
                },
                articulation=(dict(instance_articulation) if joints else {}),
                source={
                    "kind": "env_gen",
                    "asset_id": obj.get("asset_id"),
                    "model_id": obj.get("model_id"),
                    "asset_provenance": obj.get("asset_provenance"),
                    "source_files": source_files,
                },
                tags=tuple(t for t in (obj.get("color"), obj.get("material")) if t),
            )
        )

    task, limitations = _task_from_contract(None, backend="env_gen")
    task = replace(
        task,
        instruction=str(data.get("request") or task.instruction),
        intent="env_gen_scene_import",
    )
    name = _valid_identifier(str(data.get("scene_id") or source.stem), "env_gen")
    package = EnvironmentPackage(
        package_id=name,
        env=EnvSpec(
            name=name,
            objects=tuple(objects),
            gravity_mps2=(0.0, 0.0, -9.81),
            workspace_bounds_m=workspace_bounds,
            metadata={
                "request": data.get("request"),
                "seed": data.get("seed"),
                "compiler_version": data.get("compiler_version"),
                "source_scene_spec_sha256": data.get("source_scene_spec_sha256"),
                "asset_catalog_sha256": data.get("asset_catalog_sha256"),
                "relations": data.get("relations") or [],
            },
        ),
        assets=tuple(assets),
        task=task,
        source={
            "mode": "existing_environment_import",
            "backend": "env_gen",
            "path": str(source),
            "limitations": limitations,
        },
        target_backends=("robotwin",),
    )
    package.validate()
    return package

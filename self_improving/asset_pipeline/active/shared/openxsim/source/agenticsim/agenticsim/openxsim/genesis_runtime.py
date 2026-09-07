"""Render an OpenXSim Genesis scene without advancing physics.

The compiler-facing input is ``agenticsim.genesis_render_scene.v1``.  This
module intentionally imports Genesis only inside :func:`render_scene`, after
headless environment variables have been set.  The helpers for validation,
camera planning, canonical hashing, and frame uniqueness therefore remain
usable in unit tests on machines without Genesis.

This runtime produces rendering evidence only.  It never calls ``Scene.step``
and its evidence must not be treated as physical runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These must be set before any Genesis (and therefore pyglet/OpenGL) import.
os.environ["GS_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "1"


SCENE_SCHEMA = "agenticsim.genesis_render_scene.v1"
EVIDENCE_SCHEMA = "agenticsim.genesis_render_evidence.v1"
MANIFEST_SCHEMA = "agenticsim.genesis_render_manifest.v1"
POSE_TOLERANCE = 1e-6
QPOS_TOLERANCE = 1e-6

STATIC_OUTPUTS = {
    "front_high": "preview_head.png",
    "world_left": "preview_world_left.png",
    "world_right": "preview_world_right.png",
}
SEGMENTATION_OUTPUT = "preview_segmentation.png"
OBSERVER_OUTPUTS = {
    "start": "observer_start.png",
    "mid": "observer_mid.png",
    "end": "observer_end.png",
}
VIDEO_OUTPUT = "observer_runtime.mp4"
EVIDENCE_OUTPUT = "genesis_render_evidence.json"
MANIFEST_OUTPUT = "render_manifest.json"

_RENDER_OUTPUT_NAMES = frozenset(
    {
        *STATIC_OUTPUTS.values(),
        SEGMENTATION_OUTPUT,
        *OBSERVER_OUTPUTS.values(),
        VIDEO_OUTPUT,
        EVIDENCE_OUTPUT,
        MANIFEST_OUTPUT,
    }
)
_SUPPORTED_KINDS = {"box", "mesh", "urdf"}
_SUPPORTED_MESH_FORMATS = {"obj", "stl", "dae", "glb", "gltf"}
_SHA256_RE = frozenset("0123456789abcdef")


def _clear_render_outputs(output_dir: Path) -> None:
    """Remove only this runner's known outputs before or after an attempted run."""

    for name in _RENDER_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


class GenesisRenderError(RuntimeError):
    """Raised for invalid scene artifacts or failed render acceptance gates."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class RenderOptions:
    """Runtime options that override the compiler's suggested render settings."""

    width: int = 640
    height: int = 480
    frames: int = 120
    fps: int = 12
    compute_backend: str = "cpu"

    def validate(self) -> None:
        if self.width < 16 or self.height < 16:
            raise GenesisRenderError("width and height must both be at least 16 pixels")
        if self.width % 2 or self.height % 2:
            raise GenesisRenderError("width and height must be even for the H.264 MP4 output")
        if self.frames < 3:
            raise GenesisRenderError("frames must be at least 3")
        if self.fps <= 0:
            raise GenesisRenderError("fps must be positive")
        if self.compute_backend not in {"cpu", "gpu", "cuda"}:
            raise GenesisRenderError(
                "compute_backend must be one of: cpu, gpu, cuda"
            )


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for *payload*."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Match ``EnvironmentPackage.digest`` canonical JSON hashing."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def unique_frame_count(frames: Iterable[Any]) -> int:
    """Count byte-distinct frames; accepts bytes or array-like objects."""

    digests: set[str] = set()
    for frame in frames:
        if isinstance(frame, bytes):
            payload = frame
        elif isinstance(frame, bytearray):
            payload = bytes(frame)
        elif isinstance(frame, memoryview):
            payload = frame.tobytes()
        elif hasattr(frame, "tobytes"):
            payload = frame.tobytes()
        else:
            payload = bytes(frame)
        digests.add(sha256_bytes(payload))
    return len(digests)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_RE)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and set(value.lower()).issubset(_SHA256_RE)
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise GenesisRenderError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GenesisRenderError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise GenesisRenderError(f"{field} must be a finite number")
    return result


def _vector(
    value: Any,
    length: int,
    field: str,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GenesisRenderError(f"{field} must contain {length} numbers")
    result = tuple(
        _finite_number(component, f"{field}[{index}]")
        for index, component in enumerate(value)
    )
    if len(result) != length:
        raise GenesisRenderError(f"{field} must contain {length} numbers")
    if positive and any(component <= 0.0 for component in result):
        raise GenesisRenderError(f"{field} values must be positive")
    if unit_interval and any(not 0.0 <= component <= 1.0 for component in result):
        raise GenesisRenderError(f"{field} values must be in [0, 1]")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GenesisRenderError(f"{field} must be an object")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenesisRenderError(f"{field} must be a non-empty string")
    return value


def validate_scene_config(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a compiled Genesis render scene without importing Genesis.

    The original mapping is returned unchanged so callers can retain compiler
    metadata.  File existence and package digest binding are checked later,
    relative to the scene artifact's location.
    """

    if not isinstance(data, Mapping):
        raise GenesisRenderError("scene JSON root must be an object")
    if data.get("schema") != SCENE_SCHEMA:
        raise GenesisRenderError(
            f"unsupported scene schema {data.get('schema')!r}; expected {SCENE_SCHEMA!r}"
        )
    _nonempty_string(data.get("package_id"), "package_id")
    if not _is_sha256(data.get("package_digest")):
        raise GenesisRenderError("package_digest must be a lowercase SHA-256 digest")
    _nonempty_string(data.get("package_path"), "package_path")

    environment = _mapping(data.get("environment"), "environment")
    _nonempty_string(environment.get("name"), "environment.name")
    workspace = _vector(
        environment.get("workspace_bounds_m"),
        6,
        "environment.workspace_bounds_m",
    )
    if any(workspace[index] >= workspace[index + 3] for index in range(3)):
        raise GenesisRenderError(
            "environment.workspace_bounds_m minimums must be below maximums"
        )
    table_height = _finite_number(
        environment.get("table_height_m"), "environment.table_height_m"
    )

    table = _mapping(data.get("table"), "table")
    table_center = _vector(table.get("center"), 3, "table.center")
    table_size = _vector(table.get("size"), 3, "table.size", positive=True)
    _vector(table.get("color_rgb"), 3, "table.color_rgb", unit_interval=True)
    table_top = table_center[2] + 0.5 * table_size[2]
    if abs(table_top - table_height) > 1e-9:
        raise GenesisRenderError(
            "table top must equal environment.table_height_m exactly within 1e-9 m"
        )

    render = data.get("render") or {}
    render = _mapping(render, "render")
    if render.get("renderer") is not None:
        _nonempty_string(render.get("renderer"), "render.renderer")
    for key in ("width", "height", "frames", "fps"):
        if key in render and (isinstance(render[key], bool) or int(render[key]) <= 0):
            raise GenesisRenderError(f"render.{key} must be a positive integer")
    if "compute_backend" in render:
        _nonempty_string(render["compute_backend"], "render.compute_backend")

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise GenesisRenderError("objects must be an array")
    instance_ids: set[str] = set()
    for index, item in enumerate(objects):
        obj = _mapping(item, f"objects[{index}]")
        prefix = f"objects[{index}]"
        instance_id = _nonempty_string(obj.get("instance_id"), f"{prefix}.instance_id")
        if instance_id in instance_ids:
            raise GenesisRenderError(f"duplicate object instance_id: {instance_id}")
        instance_ids.add(instance_id)
        _nonempty_string(obj.get("asset_id"), f"{prefix}.asset_id")

        kind = _nonempty_string(obj.get("kind"), f"{prefix}.kind").lower()
        if kind not in _SUPPORTED_KINDS:
            blocker = obj.get("blocker") or f"unsupported object kind {kind!r}"
            raise GenesisRenderError(f"{instance_id}: {blocker}")
        if obj.get("render_fixed") is not True:
            raise GenesisRenderError(f"{instance_id}: render_fixed must be true")

        pose = _mapping(obj.get("pose"), f"{prefix}.pose")
        _vector(pose.get("position"), 3, f"{prefix}.pose.position")
        quat = _vector(
            pose.get("orientation_wxyz"),
            4,
            f"{prefix}.pose.orientation_wxyz",
        )
        quat_norm = math.sqrt(sum(component * component for component in quat))
        if abs(quat_norm - 1.0) > 1e-5:
            raise GenesisRenderError(
                f"{instance_id}: pose.orientation_wxyz must be unit length"
            )
        _vector(obj.get("scale"), 3, f"{prefix}.scale", positive=True)
        _vector(
            obj.get("dimensions_m"),
            3,
            f"{prefix}.dimensions_m",
            positive=True,
        )
        z_policy = _nonempty_string(obj.get("z_policy"), f"{prefix}.z_policy")
        if z_policy not in {"origin_on_table", "center_on_table"}:
            raise GenesisRenderError(
                f"{prefix}.z_policy must be 'origin_on_table' or 'center_on_table'"
            )
        color_rgb = obj.get("color_rgb")
        if color_rgb is not None:
            _vector(
                color_rgb,
                3,
                f"{prefix}.color_rgb",
                unit_interval=True,
            )

        articulation = _mapping(obj.get("articulation") or {}, f"{prefix}.articulation")
        joint_names = articulation.get("joint_names") or []
        qpos = articulation.get("qpos") or []
        if not isinstance(joint_names, list) or not all(
            isinstance(name, str) and name for name in joint_names
        ):
            raise GenesisRenderError(
                f"{prefix}.articulation.joint_names must be an array of names"
            )
        if len(set(joint_names)) != len(joint_names):
            raise GenesisRenderError(f"{prefix}.articulation.joint_names has duplicates")
        if not isinstance(qpos, list) or len(qpos) != len(joint_names):
            raise GenesisRenderError(
                f"{prefix}.articulation.qpos must match joint_names length"
            )
        for q_index, value in enumerate(qpos):
            _finite_number(value, f"{prefix}.articulation.qpos[{q_index}]")
        if joint_names and kind != "urdf":
            raise GenesisRenderError(
                f"{instance_id}: articulation is supported only for URDF objects"
            )

        if kind == "box":
            _vector(obj.get("size_m"), 3, f"{prefix}.size_m", positive=True)
        else:
            _nonempty_string(obj.get("uri"), f"{prefix}.uri")
            if not _is_sha256(obj.get("source_sha256")):
                raise GenesisRenderError(f"{prefix}.source_sha256 must be a SHA-256 digest")
            source_size = obj.get("source_size_bytes")
            if isinstance(source_size, bool) or not isinstance(source_size, int) or source_size < 0:
                raise GenesisRenderError(
                    f"{prefix}.source_size_bytes must be a non-negative integer"
                )
            _nonempty_string(
                obj.get("source_dependency_discovery"),
                f"{prefix}.source_dependency_discovery",
            )
            dependencies = obj.get("source_dependencies")
            if not isinstance(dependencies, list):
                raise GenesisRenderError(f"{prefix}.source_dependencies must be an array")
            for dependency_index, dependency_value in enumerate(dependencies):
                dependency = _mapping(
                    dependency_value,
                    f"{prefix}.source_dependencies[{dependency_index}]",
                )
                dependency_prefix = f"{prefix}.source_dependencies[{dependency_index}]"
                _nonempty_string(dependency.get("uri"), f"{dependency_prefix}.uri")
                if not _is_sha256(dependency.get("sha256")):
                    raise GenesisRenderError(
                        f"{dependency_prefix}.sha256 must be a SHA-256 digest"
                    )
                dependency_size = dependency.get("size_bytes")
                if (
                    isinstance(dependency_size, bool)
                    or not isinstance(dependency_size, int)
                    or dependency_size < 0
                ):
                    raise GenesisRenderError(
                        f"{dependency_prefix}.size_bytes must be a non-negative integer"
                    )
            fmt = _nonempty_string(obj.get("format"), f"{prefix}.format").lower()
            if kind == "mesh" and fmt not in _SUPPORTED_MESH_FORMATS:
                raise GenesisRenderError(
                    f"{instance_id}: unsupported Genesis mesh format {fmt!r}"
                )
            if kind == "urdf" and fmt != "urdf":
                raise GenesisRenderError(f"{instance_id}: URDF object format must be 'urdf'")
            if kind == "mesh":
                if not isinstance(obj.get("file_meshes_are_zup"), bool):
                    raise GenesisRenderError(
                        f"{prefix}.file_meshes_are_zup must be boolean"
                    )
                _vector(
                    obj.get("genesis_scale"),
                    3,
                    f"{prefix}.genesis_scale",
                    positive=True,
                )
            if kind == "urdf":
                uniform_scale = _finite_number(
                    obj.get("uniform_scale"), f"{prefix}.uniform_scale"
                )
                if uniform_scale <= 0.0:
                    raise GenesisRenderError(f"{prefix}.uniform_scale must be positive")

    return data


def load_scene_config(path: str | Path) -> Mapping[str, Any]:
    """Read and validate one compiled scene artifact."""

    scene_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenesisRenderError(f"could not read Genesis scene {scene_path}: {exc}") from exc
    return validate_scene_config(data)


def verify_package_binding(
    scene: Mapping[str, Any], scene_path: str | Path
) -> dict[str, Any]:
    """Verify package digest and reproduce the exact compiled scene contract."""

    declared = str(scene["package_digest"])
    package_value = _nonempty_string(scene.get("package_path"), "package_path")
    path = Path(package_value).expanduser()
    if not path.is_absolute():
        path = Path(scene_path).expanduser().resolve().parent / path
    path = path.resolve()
    try:
        package_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenesisRenderError(f"could not read environment package {path}: {exc}") from exc
    actual = canonical_json_sha256(package_data)
    if actual != declared:
        raise GenesisRenderError(
            f"environment package digest mismatch: declared {declared}, computed {actual}"
        )
    if str(package_data.get("package_id")) != str(scene["package_id"]):
        raise GenesisRenderError("environment package_id does not match render scene package_id")

    # Recompilation is intentionally dependency-free: GenesisCompiler emits JSON
    # but never imports Genesis.  Exact comparison binds every pose, object,
    # articulation, scale, table, task, and source fingerprint to the package.
    from .backends import BackendCompileError, GenesisCompiler
    from .ir import EnvironmentPackage

    try:
        package = EnvironmentPackage.from_dict(package_data)
        with tempfile.TemporaryDirectory(prefix="openxsim-genesis-verify-") as temporary:
            result = GenesisCompiler().compile(package, temporary, strict=True)
            expected_scene = json.loads(
                Path(result.artifact_path).read_text(encoding="utf-8")
            )
    except (BackendCompileError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise GenesisRenderError(
            f"could not reproduce Genesis scene from environment package: {exc}"
        ) from exc

    expected_sha = canonical_json_sha256(expected_scene)
    scene_sha = canonical_json_sha256(scene)
    if expected_scene != scene:
        raise GenesisRenderError(
            "compiled Genesis scene does not match its environment package "
            f"(expected {expected_sha}, got {scene_sha})"
        )
    return {
        "declared_digest": declared,
        "verified": True,
        "verification": "canonical_digest_and_deterministic_recompile",
        "path": str(path),
        "file_sha256": sha256_file(path),
        "scene_contract_verified": True,
        "expected_scene_sha256": expected_sha,
        "scene_sha256": scene_sha,
    }


def verify_asset_integrity(
    scene: Mapping[str, Any], scene_path: str | Path
) -> dict[str, dict[str, Any]]:
    """Verify primary assets and every compiler-declared local dependency."""

    source_path = Path(scene_path).expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    for spec in scene["objects"]:
        instance_id = str(spec["instance_id"])
        if spec["kind"] == "box":
            records[instance_id] = {
                "verified": True,
                "kind": "primitive_box",
                "primary": None,
                "dependencies": [],
            }
            continue
        primary = _resolve_uri(str(spec["uri"]), source_path)
        expected_sha = str(spec["source_sha256"])
        actual_sha = sha256_file(primary)
        expected_size = int(spec["source_size_bytes"])
        actual_size = primary.stat().st_size
        if actual_sha != expected_sha or actual_size != expected_size:
            raise GenesisRenderError(
                f"{instance_id}: primary asset fingerprint mismatch for {primary}"
            )
        dependency_records: list[dict[str, Any]] = []
        for dependency_spec in spec["source_dependencies"]:
            dependency = _resolve_uri(str(dependency_spec["uri"]), source_path)
            dependency_sha = sha256_file(dependency)
            dependency_size = dependency.stat().st_size
            if (
                dependency_sha != dependency_spec["sha256"]
                or dependency_size != dependency_spec["size_bytes"]
            ):
                raise GenesisRenderError(
                    f"{instance_id}: dependency fingerprint mismatch for {dependency}"
                )
            dependency_records.append(
                {
                    "uri": str(dependency),
                    "sha256": dependency_sha,
                    "size_bytes": dependency_size,
                }
            )
        records[instance_id] = {
            "verified": True,
            "kind": str(spec["kind"]),
            "dependency_discovery": spec["source_dependency_discovery"],
            "primary": {
                "uri": str(primary),
                "sha256": actual_sha,
                "size_bytes": actual_size,
            },
            "dependencies": dependency_records,
        }
    return records


def _object_bounds(scene: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    objects = scene.get("objects") or []
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    for item in objects:
        position = _vector(item["pose"]["position"], 3, "pose.position")
        dimensions = _vector(item["dimensions_m"], 3, "dimensions_m", positive=True)
        for axis in range(2):
            lower[axis] = min(lower[axis], position[axis] - 0.5 * dimensions[axis])
            upper[axis] = max(upper[axis], position[axis] + 0.5 * dimensions[axis])
        if item["z_policy"] == "origin_on_table":
            object_z_lower = position[2]
            object_z_upper = position[2] + dimensions[2]
        else:
            object_z_lower = position[2] - 0.5 * dimensions[2]
            object_z_upper = position[2] + 0.5 * dimensions[2]
        lower[2] = min(lower[2], object_z_lower)
        upper[2] = max(upper[2], object_z_upper)
    if objects:
        return lower, upper

    workspace = _vector(
        scene["environment"]["workspace_bounds_m"],
        6,
        "environment.workspace_bounds_m",
    )
    return list(workspace[:3]), list(workspace[3:])


def build_camera_plan(scene: Mapping[str, Any], frames: int) -> dict[str, Any]:
    """Build deterministic static and one-turn orbit camera poses.

    The camera target and distance come from declared object poses and metric
    dimensions, not from backend-dependent mesh bounds.  A small deterministic
    elevation oscillation makes the orbit informative for symmetric scenes.
    """

    validate_scene_config(scene)
    if frames < 3:
        raise GenesisRenderError("frames must be at least 3")
    lower, upper = _object_bounds(scene)
    table_height = float(scene["environment"]["table_height_m"])
    center_x = 0.5 * (lower[0] + upper[0])
    center_y = 0.5 * (lower[1] + upper[1])
    center_z = max(table_height + 0.03, 0.5 * (lower[2] + upper[2]))
    span = max(
        upper[0] - lower[0],
        upper[1] - lower[1],
        upper[2] - min(lower[2], table_height),
        0.08,
    )
    fov_deg = 45.0
    minimum_distance = 0.65
    framing_distance = 0.85 * span / math.tan(math.radians(fov_deg / 2.0))
    radius = max(minimum_distance, framing_distance)
    elevation = max(0.28, 0.48 * radius)
    target = (center_x, center_y, center_z)
    up = (0.0, 0.0, 1.0)

    static_views = [
        {
            "name": "front_high",
            "position": (center_x, center_y - radius, center_z + elevation),
            "lookat": target,
            "up": up,
        },
        {
            "name": "world_left",
            "position": (
                center_x - 0.90 * radius,
                center_y - 0.60 * radius,
                center_z + elevation,
            ),
            "lookat": target,
            "up": up,
        },
        {
            "name": "world_right",
            "position": (
                center_x + 0.90 * radius,
                center_y - 0.60 * radius,
                center_z + elevation,
            ),
            "lookat": target,
            "up": up,
        },
    ]

    start_angle = -0.5 * math.pi
    angle_step = 2.0 * math.pi / frames
    orbit = []
    for index in range(frames):
        angle = start_angle + index * angle_step
        z_wobble = 0.08 * radius * math.sin(2.0 * angle + 0.3)
        orbit.append(
            {
                "index": index,
                "angle_rad": angle,
                "position": (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                    center_z + elevation + z_wobble,
                ),
                "lookat": target,
                "up": up,
            }
        )
    return {
        "target": target,
        "radius_m": radius,
        "base_elevation_m": elevation,
        "elevation_wobble_amplitude_m": 0.08 * radius,
        "fov_deg": fov_deg,
        "near_m": 0.01,
        "far_m": max(20.0, 5.0 * radius),
        "static_views": static_views,
        "orbit": orbit,
        "orbit_start_angle_rad": start_angle,
        "orbit_angle_step_rad": angle_step,
        "duplicates_endpoint": False,
    }


# Short alias for callers that prefer a noun-style helper name.
camera_plan = build_camera_plan


def _resolve_uri(value: str, scene_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = scene_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise GenesisRenderError(f"asset file does not exist: {path}")
    return path


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _rgb_uint8(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - Genesis itself requires numpy
        raise GenesisRenderError("numpy is required for Genesis rendering") from exc
    array = np.asarray(_as_numpy(value))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise GenesisRenderError(f"unexpected RGB buffer shape: {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if finite.size and float(finite.max()) <= 1.0 + 1e-6:
            finite = finite * 255.0
        array = finite
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _segmentation_array(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise GenesisRenderError("numpy is required for Genesis rendering") from exc
    array = np.asarray(_as_numpy(value))
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise GenesisRenderError(f"unexpected segmentation buffer shape: {array.shape}")
    return array.astype(np.int64, copy=False)


def _save_png(path: Path, array: Any) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise GenesisRenderError("Pillow is required to write Genesis render images") from exc
    image = _rgb_uint8(array)
    Image.fromarray(image).save(path, format="PNG", optimize=False)


def _fallback_colorize_segmentation(segmentation: Any) -> Any:
    import numpy as np

    seg = _segmentation_array(segmentation)
    output = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label in np.unique(seg):
        if int(label) == 0:
            continue
        value = int(label)
        output[seg == label] = (
            64 + (value * 67) % 192,
            64 + (value * 109) % 192,
            64 + (value * 149) % 192,
        )
    return output


class _VideoWriter:
    """Stream RGB frames to MP4, preferring the installed ffmpeg executable."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self._path = path
        self._process: subprocess.Popen[bytes] | None = None
        self._imageio_writer: Any = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [
                ffmpeg,
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise GenesisRenderError(
                "MP4 encoding requires ffmpeg or imageio with an ffmpeg plugin"
            ) from exc
        try:
            self._imageio_writer = imageio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
        except Exception as exc:
            raise GenesisRenderError(f"could not open MP4 encoder: {exc}") from exc

    def append(self, frame: Any) -> None:
        image = _rgb_uint8(frame)
        if self._process is not None:
            if self._process.stdin is None:
                raise GenesisRenderError("ffmpeg stdin is unavailable")
            try:
                self._process.stdin.write(image.tobytes())
            except (BrokenPipeError, OSError) as exc:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read()
                message = stderr.decode("utf-8", errors="replace").strip()
                raise GenesisRenderError(f"ffmpeg rejected a frame: {message}") from exc
        else:
            self._imageio_writer.append_data(image)

    def close(self) -> None:
        if self._process is not None:
            process = self._process
            self._process = None
            if process.stdin is not None:
                process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise GenesisRenderError(
                    f"ffmpeg exited with status {return_code}: {message}"
                )
        elif self._imageio_writer is not None:
            writer = self._imageio_writer
            self._imageio_writer = None
            try:
                writer.close()
            except Exception as exc:
                raise GenesisRenderError(f"could not finalize MP4: {exc}") from exc

    def abort(self) -> None:
        if self._process is not None:
            process = self._process
            self._process = None
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._imageio_writer is not None:
            writer = self._imageio_writer
            self._imageio_writer = None
            try:
                writer.close()
            except Exception:
                pass

    def __enter__(self) -> "_VideoWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


def verify_encoded_video(
    path: str | Path,
    *,
    expected_frames: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Decode the finished MP4 and enforce its frame, size, and diversity contract."""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise GenesisRenderError("imageio is required to verify the encoded MP4") from exc

    video_path = Path(path).expanduser().resolve()
    frame_hashes: list[str] = []
    try:
        reader = imageio.get_reader(video_path)
        try:
            for index, frame in enumerate(reader):
                image = _rgb_uint8(frame)
                if image.shape[:2] != (height, width):
                    raise GenesisRenderError(
                        f"encoded MP4 frame {index} has size "
                        f"{image.shape[1]}x{image.shape[0]}, expected {width}x{height}"
                    )
                frame_hashes.append(sha256_bytes(image.tobytes()))
        finally:
            reader.close()
    except GenesisRenderError:
        raise
    except Exception as exc:
        raise GenesisRenderError(f"could not decode encoded MP4 {video_path}: {exc}") from exc

    decoded_frames = len(frame_hashes)
    if decoded_frames != expected_frames:
        raise GenesisRenderError(
            f"encoded MP4 has {decoded_frames} frames; expected exactly {expected_frames}"
        )
    unique_frames = len(set(frame_hashes))
    minimum_unique = min(expected_frames, 30)
    if unique_frames < minimum_unique:
        raise GenesisRenderError(
            f"encoded MP4 has {unique_frames} distinct decoded frames; "
            f"expected at least {minimum_unique}"
        )
    return {
        "decoded_frame_count": decoded_frames,
        "decoded_unique_frame_count": unique_frames,
        "minimum_unique_frame_count": minimum_unique,
        "decoded_frame_sha256": frame_hashes,
        "width": width,
        "height": height,
    }


def _surface(gs: Any, color_rgb: Sequence[float] | None, *, double_sided: bool) -> Any:
    if color_rgb is None:
        return gs.surfaces.Default(double_sided=double_sided, vis_mode="visual")
    rgba = tuple(float(value) for value in color_rgb) + (1.0,)
    return gs.surfaces.Plastic(
        color=rgba,
        roughness=0.72,
        double_sided=double_sided,
        vis_mode="visual",
    )


def _add_table(scene: Any, gs: Any, table: Mapping[str, Any]) -> Any:
    morph = gs.morphs.Box(
        pos=tuple(table["center"]),
        size=tuple(table["size"]),
        collision=False,
        fixed=True,
    )
    return scene.add_entity(
        morph=morph,
        material=gs.materials.Kinematic(),
        surface=_surface(gs, table["color_rgb"], double_sided=False),
        name="openxsim_table",
    )


def _add_object(
    scene: Any,
    gs: Any,
    spec: Mapping[str, Any],
    scene_path: Path,
) -> Any:
    kind = str(spec["kind"])
    pose = spec["pose"]
    common = {
        "pos": tuple(pose["position"]),
        "quat": tuple(pose["orientation_wxyz"]),
        "collision": False,
        "fixed": True,
    }
    if kind == "box":
        morph = gs.morphs.Box(size=tuple(spec["size_m"]), **common)
    elif kind == "mesh":
        path = _resolve_uri(str(spec["uri"]), scene_path)
        file_meshes_are_zup = spec.get("file_meshes_are_zup")
        if file_meshes_are_zup is None:
            file_meshes_are_zup = str(spec["format"]).lower() not in {"glb", "gltf"}
        morph = gs.morphs.Mesh(
            file=str(path),
            scale=tuple(spec["genesis_scale"]),
            align=False,
            file_meshes_are_zup=bool(file_meshes_are_zup),
            **common,
        )
    elif kind == "urdf":
        path = _resolve_uri(str(spec["uri"]), scene_path)
        morph = gs.morphs.URDF(
            file=str(path),
            scale=float(spec["uniform_scale"]),
            align=False,
            merge_fixed_links=False,
            **common,
        )
    else:  # guarded by validate_scene_config
        raise GenesisRenderError(f"unsupported object kind: {kind}")
    return scene.add_entity(
        morph=morph,
        material=gs.materials.Kinematic(),
        surface=_surface(gs, spec.get("color_rgb"), double_sided=True),
        name=str(spec["instance_id"]),
    )


def _tensor_list(value: Any) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise GenesisRenderError("numpy is required for Genesis rendering") from exc
    array = np.asarray(_as_numpy(value), dtype=np.float64).reshape(-1)
    return [float(component) for component in array]


def _apply_articulation(entity: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    articulation = spec.get("articulation") or {}
    joint_names = list(articulation.get("joint_names") or [])
    requested = [float(value) for value in articulation.get("qpos") or []]
    if not joint_names:
        return {"joint_names": [], "requested_qpos": [], "applied_qpos": []}

    indices: list[int] = []
    for joint_name in joint_names:
        try:
            joint = entity.get_joint(name=joint_name)
        except Exception as exc:
            raise GenesisRenderError(
                f"{spec['instance_id']}: Genesis joint {joint_name!r} was not found"
            ) from exc
        if int(joint.n_dofs) != 1 or int(joint.n_qs) != 1:
            raise GenesisRenderError(
                f"{spec['instance_id']}: joint {joint_name!r} must map to one DoF and one q"
            )
        local_indices = list(joint.qs_idx_local)
        if len(local_indices) != 1:
            raise GenesisRenderError(
                f"{spec['instance_id']}: joint {joint_name!r} has ambiguous q indices"
            )
        indices.append(int(local_indices[0]))
    if len(set(indices)) != len(indices):
        raise GenesisRenderError(f"{spec['instance_id']}: joint q indices are not unique")

    entity.set_qpos(requested, qs_idx_local=indices)
    applied = _tensor_list(entity.get_qpos(qs_idx_local=indices))
    if len(applied) != len(requested):
        raise GenesisRenderError(
            f"{spec['instance_id']}: Genesis returned {len(applied)} qpos values; "
            f"expected {len(requested)}"
        )
    max_error = max(
        (abs(expected - observed) for expected, observed in zip(requested, applied)),
        default=0.0,
    )
    if max_error > QPOS_TOLERANCE:
        raise GenesisRenderError(
            f"{spec['instance_id']}: applied articulation qpos differs from resolved "
            f"state by {max_error:.9g}, above {QPOS_TOLERANCE:.1e}",
            details={
                "instance_id": spec["instance_id"],
                "joint_names": joint_names,
                "requested_qpos": requested,
                "applied_qpos": applied,
                "max_abs_qpos_error": max_error,
                "qpos_tolerance": QPOS_TOLERANCE,
            },
        )
    return {
        "joint_names": joint_names,
        "requested_qpos": requested,
        "applied_qpos": applied,
        "max_abs_qpos_error": max_error,
        "qpos_tolerance": QPOS_TOLERANCE,
        "q_indices_local": indices,
    }


def _entity_pose(entity: Any) -> dict[str, list[float]]:
    return {
        "position": _tensor_list(entity.get_pos(relative=True)),
        "orientation_wxyz": _tensor_list(entity.get_quat(relative=True)),
    }


def pose_error(
    resolved: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, float]:
    """Return max-absolute pose error with quaternion sign equivalence."""

    resolved_position = _vector(resolved.get("position"), 3, "resolved.position")
    observed_position = _vector(observed.get("position"), 3, "observed.position")
    resolved_quat = _vector(
        resolved.get("orientation_wxyz"), 4, "resolved.orientation_wxyz"
    )
    observed_quat = _vector(
        observed.get("orientation_wxyz"), 4, "observed.orientation_wxyz"
    )
    direct = max(abs(left - right) for left, right in zip(resolved_quat, observed_quat))
    negated = max(abs(left + right) for left, right in zip(resolved_quat, observed_quat))
    return {
        "position_max_abs_m": max(
            abs(left - right) for left, right in zip(resolved_position, observed_position)
        ),
        "quaternion_sign_invariant_max_abs": min(direct, negated),
    }


def _segmentation_indices(scene: Any, entities: Mapping[str, Any]) -> dict[str, int | None]:
    raw_mapping = scene.visualizer.segmentation_idx_dict
    mapping = {int(index): key for index, key in raw_mapping.items()}
    result: dict[str, int | None] = {}
    for name, entity in entities.items():
        entity_index = int(entity.idx)
        result[name] = next(
            (index for index, key in mapping.items() if key == entity_index),
            None,
        )
    return result


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _discover_genesis_commit(gs: Any) -> str | None:
    module_path = Path(gs.__file__).resolve()
    for candidate in (module_path.parent, *module_path.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = result.stdout.strip()
        return commit if _is_git_commit(commit) else None
    return None


def _camera_evidence(plan: Mapping[str, Any], frames: int) -> dict[str, Any]:
    orbit = plan["orbit"]
    selected = [orbit[0], orbit[frames // 2], orbit[-1]]
    return {
        "target": list(plan["target"]),
        "radius_m": plan["radius_m"],
        "base_elevation_m": plan["base_elevation_m"],
        "elevation_wobble_amplitude_m": plan["elevation_wobble_amplitude_m"],
        "fov_deg": plan["fov_deg"],
        "near_m": plan["near_m"],
        "far_m": plan["far_m"],
        "static_views": [
            {
                **view,
                "position": list(view["position"]),
                "lookat": list(view["lookat"]),
                "up": list(view["up"]),
            }
            for view in plan["static_views"]
        ],
        "orbit": {
            "frame_count": frames,
            "start_angle_rad": plan["orbit_start_angle_rad"],
            "angle_step_rad": plan["orbit_angle_step_rad"],
            "duplicates_endpoint": plan["duplicates_endpoint"],
            "selected_poses": [
                {
                    **pose,
                    "position": list(pose["position"]),
                    "lookat": list(pose["lookat"]),
                    "up": list(pose["up"]),
                }
                for pose in selected
            ],
        },
    }


def render_scene(
    scene_config: Mapping[str, Any],
    *,
    scene_path: str | Path,
    output_dir: str | Path,
    options: RenderOptions | None = None,
) -> dict[str, Any]:
    """Render a validated scene and return its success evidence.

    This function raises :class:`GenesisRenderError` on any build, render,
    visibility, or unique-frame failure.  The CLI converts it into failed
    evidence and a nonzero exit status.
    """

    validate_scene_config(scene_config)
    options = options or RenderOptions()
    options.validate()
    source_path = Path(scene_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _clear_render_outputs(destination)
    binding = verify_package_binding(scene_config, source_path)
    asset_integrity = verify_asset_integrity(scene_config, source_path)
    plan = build_camera_plan(scene_config, options.frames)

    try:
        import numpy as np
    except ImportError as exc:
        raise GenesisRenderError("numpy is required for Genesis rendering") from exc
    try:
        import genesis as gs
    except ImportError as exc:
        raise GenesisRenderError(
            "Genesis is not installed; install the pinned external/genesis-world checkout"
        ) from exc

    backend_attribute = options.compute_backend
    gs_backend = getattr(gs, backend_attribute, None)
    if gs_backend is None:
        raise GenesisRenderError(
            f"this Genesis build does not expose the {backend_attribute!r} backend"
        )

    object_entities: dict[str, Any] = {}
    object_results: dict[str, dict[str, Any]] = {}
    frame_hashes: list[str] = []
    output_paths = [
        *(destination / name for name in STATIC_OUTPUTS.values()),
        destination / SEGMENTATION_OUTPUT,
        *(destination / name for name in OBSERVER_OUTPUTS.values()),
        destination / VIDEO_OUTPUT,
    ]

    try:
        gs.init(backend=gs_backend, logging_level=logging.WARNING, seed=0)
        scene = gs.Scene(
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
            vis_options=gs.options.VisOptions(segmentation_level="entity"),
        )
        _add_table(scene, gs, scene_config["table"])
        for spec in scene_config["objects"]:
            instance_id = str(spec["instance_id"])
            entity = _add_object(scene, gs, spec, source_path)
            object_entities[instance_id] = entity

        initial_view = plan["static_views"][0]
        camera = scene.add_camera(
            res=(options.width, options.height),
            pos=initial_view["position"],
            lookat=initial_view["lookat"],
            up=initial_view["up"],
            fov=plan["fov_deg"],
            near=plan["near_m"],
            far=plan["far_m"],
            GUI=False,
        )
        scene.build()

        for spec in scene_config["objects"]:
            instance_id = str(spec["instance_id"])
            entity = object_entities[instance_id]
            applied_articulation = _apply_articulation(entity, spec)
            final_pose = _entity_pose(entity)
            resolved_pose_error = pose_error(spec["pose"], final_pose)
            object_results[instance_id] = {
                "instance_id": instance_id,
                "asset_id": spec["asset_id"],
                "kind": spec["kind"],
                "uri": spec.get("uri"),
                "format": spec.get("format"),
                "representation_role": spec.get("representation_role"),
                "source_scale": spec["scale"],
                "genesis_scale": spec.get("genesis_scale"),
                "source_integrity": asset_integrity[instance_id],
                "z_policy": spec["z_policy"],
                "source_static": bool(spec.get("source_static", False)),
                "render_fixed": True,
                "surface_mode": (
                    "color_override" if spec.get("color_rgb") is not None else "asset_material"
                ),
                "color": spec.get("color"),
                "color_rgb": spec.get("color_rgb"),
                "material": spec.get("material"),
                "resolved_pose": spec["pose"],
                "final_pose": final_pose,
                "pose_error": resolved_pose_error,
                "articulation": applied_articulation,
                "entity_index": int(entity.idx),
                "segmentation_index": None,
                "visibility_pixels": {},
            }

        segmentation_indices = _segmentation_indices(scene, object_entities)
        pose_failures = [
            instance_id
            for instance_id, result in object_results.items()
            if result["pose_error"]["position_max_abs_m"] > POSE_TOLERANCE
            or result["pose_error"]["quaternion_sign_invariant_max_abs"] > POSE_TOLERANCE
        ]
        if pose_failures:
            raise GenesisRenderError(
                "resolved/final pose mismatch above 1e-6 for: " + ", ".join(pose_failures),
                details={
                    "pose_tolerance": POSE_TOLERANCE,
                    "objects": list(object_results.values()),
                },
            )

        for instance_id, index in segmentation_indices.items():
            object_results[instance_id]["segmentation_index"] = index

        for view in plan["static_views"]:
            view_name = str(view["name"])
            camera.set_pose(pos=view["position"], lookat=view["lookat"], up=view["up"])
            rgb, _, segmentation, _ = camera.render(
                rgb=True,
                depth=False,
                segmentation=True,
                colorize_seg=False,
                normal=False,
                force_render=True,
            )
            rgb_image = _rgb_uint8(rgb)
            seg_image = _segmentation_array(segmentation)
            _save_png(destination / STATIC_OUTPUTS[view_name], rgb_image)
            for instance_id, seg_index in segmentation_indices.items():
                count = 0 if seg_index is None else int(np.count_nonzero(seg_image == seg_index))
                object_results[instance_id]["visibility_pixels"][view_name] = count
            if view_name == "front_high":
                try:
                    colorized = scene.visualizer.colorize_seg_idxc_arr(seg_image)
                except Exception:
                    colorized = _fallback_colorize_segmentation(seg_image)
                _save_png(destination / SEGMENTATION_OUTPUT, colorized)

        invisible = [
            instance_id
            for instance_id, result in object_results.items()
            if max(result["visibility_pixels"].values(), default=0) <= 0
        ]
        if invisible:
            raise GenesisRenderError(
                f"objects invisible in all three static views: {', '.join(invisible)}",
                details={"objects": list(object_results.values())},
            )

        selected_frames = {
            0: destination / OBSERVER_OUTPUTS["start"],
            options.frames // 2: destination / OBSERVER_OUTPUTS["mid"],
            options.frames - 1: destination / OBSERVER_OUTPUTS["end"],
        }
        with _VideoWriter(
            destination / VIDEO_OUTPUT,
            options.width,
            options.height,
            options.fps,
        ) as video:
            for pose in plan["orbit"]:
                camera.set_pose(pos=pose["position"], lookat=pose["lookat"], up=pose["up"])
                rgb, _, _, _ = camera.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                    normal=False,
                    force_render=True,
                )
                frame = _rgb_uint8(rgb)
                digest = sha256_bytes(frame.tobytes())
                frame_hashes.append(digest)
                video.append(frame)
                selected_path = selected_frames.get(int(pose["index"]))
                if selected_path is not None:
                    _save_png(selected_path, frame)

        submitted_unique_count = len(set(frame_hashes))
        minimum_unique = min(options.frames, 30)
        if submitted_unique_count < minimum_unique:
            raise GenesisRenderError(
                f"orbit video has {submitted_unique_count} unique submitted frames; "
                f"expected at least {minimum_unique}",
                details={
                    "objects": list(object_results.values()),
                    "video": {
                        "total_frame_count": options.frames,
                        "unique_frame_count": submitted_unique_count,
                        "minimum_unique_frame_count": minimum_unique,
                        "frame_sha256": frame_hashes,
                    },
                },
            )

        encoded_video = verify_encoded_video(
            destination / VIDEO_OUTPUT,
            expected_frames=options.frames,
            width=options.width,
            height=options.height,
        )

        genesis_info = {
            "version": str(getattr(gs, "__version__", "unknown")),
            "commit": _discover_genesis_commit(gs),
            "module_path": str(Path(gs.__file__).resolve()),
            "compute_backend": options.compute_backend,
            "renderer": "genesis.renderers.Rasterizer",
            "headless": True,
        }
    except GenesisRenderError:
        raise
    except Exception as exc:
        raise GenesisRenderError(f"Genesis render failed: {exc}") from exc
    finally:
        if getattr(gs, "_initialized", False):
            gs.destroy()

    artifacts = {path.name: _file_record(path) for path in output_paths}
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "success",
        "backend": "genesis",
        "mode": "render_only_no_physics",
        "physical_runtime_evidence": False,
        "package_id": scene_config["package_id"],
        "package_digest": scene_config["package_digest"],
        "package_binding": binding,
        "scene": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "schema": scene_config["schema"],
        },
        "genesis": genesis_info,
        "render": {
            "width": options.width,
            "height": options.height,
            "fps": options.fps,
            "total_frame_count": encoded_video["decoded_frame_count"],
            "unique_frame_count": encoded_video["decoded_unique_frame_count"],
            "minimum_unique_frame_count": encoded_video["minimum_unique_frame_count"],
            "frame_sha256": encoded_video["decoded_frame_sha256"],
            "encoded_video_verification": encoded_video,
            "submitted_frame_count": len(frame_hashes),
            "submitted_unique_frame_count": submitted_unique_count,
            "submitted_frame_sha256": frame_hashes,
            "camera": _camera_evidence(plan, options.frames),
        },
        "objects": list(object_results.values()),
        "artifacts": artifacts,
        "manifest_file": MANIFEST_OUTPUT,
        "conformance": {
            "L0": "pass",
            "L1": "pass",
            "L2": "not_evaluated",
            "L3": "not_evaluated",
            "L4": "not_evaluated",
        },
        "scene_step_calls": 0,
    }
    evidence_path = destination / EVIDENCE_OUTPUT
    _write_json_atomic(evidence_path, evidence)

    manifest_artifacts = {
        **artifacts,
        EVIDENCE_OUTPUT: _file_record(evidence_path),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "success",
        "backend": "genesis",
        "package_id": scene_config["package_id"],
        "package_digest": scene_config["package_digest"],
        "artifacts": manifest_artifacts,
        "self_hash_omitted": True,
    }
    _write_json_atomic(destination / MANIFEST_OUTPUT, manifest)
    return evidence


def _failure_evidence(
    *,
    scene_path: Path,
    scene_config: Mapping[str, Any] | None,
    options: RenderOptions,
    error: BaseException,
) -> dict[str, Any]:
    scene_record: dict[str, Any] = {"path": str(scene_path)}
    if scene_path.is_file():
        scene_record["sha256"] = sha256_file(scene_path)
    details = error.details if isinstance(error, GenesisRenderError) else {}
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "failed",
        "backend": "genesis",
        "mode": "render_only_no_physics",
        "physical_runtime_evidence": False,
        "package_id": scene_config.get("package_id") if scene_config else None,
        "package_digest": scene_config.get("package_digest") if scene_config else None,
        "scene": scene_record,
        "render": {
            "width": options.width,
            "height": options.height,
            "frames": options.frames,
            "fps": options.fps,
            "compute_backend": options.compute_backend,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "details": details,
        "conformance": {
            "L0": "failed",
            "L1": "failed",
            "L2": "not_evaluated",
            "L3": "not_evaluated",
            "L4": "not_evaluated",
        },
        "scene_step_calls": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an OpenXSim Genesis scene without running physics."
    )
    parser.add_argument("--scene", required=True, help="Compiled Genesis scene.json")
    parser.add_argument("--output-dir", required=True, help="Render artifact directory")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--compute-backend",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = RenderOptions(
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        compute_backend=args.compute_backend,
    )
    scene_path = Path(args.scene).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_render_outputs(output_dir)

    scene_config: Mapping[str, Any] | None = None
    try:
        options.validate()
        scene_config = load_scene_config(scene_path)
        evidence = render_scene(
            scene_config,
            scene_path=scene_path,
            output_dir=output_dir,
            options=options,
        )
    except Exception as exc:
        _clear_render_outputs(output_dir)
        failed = _failure_evidence(
            scene_path=scene_path,
            scene_config=scene_config,
            options=options,
            error=exc,
        )
        try:
            _write_json_atomic(output_dir / EVIDENCE_OUTPUT, failed)
        except OSError as write_exc:
            print(f"could not write failed render evidence: {write_exc}", file=sys.stderr)
        print(f"Genesis render failed: {exc}", file=sys.stderr)
        if os.environ.get("AGENTICSIM_GENESIS_TRACEBACK") == "1":
            traceback.print_exc()
        return 1

    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backend compilers for EnvironmentPackage v1."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .ir import AssetBundle, AssetRepresentation, EnvironmentPackage
from .robotwin import RoboTwinExportError, write_robotwin_bundle


class BackendCompileError(RuntimeError):
    """Raised when strict backend compilation cannot represent the package."""


@dataclass(frozen=True)
class CompileResult:
    backend: str
    status: str
    artifact_path: str
    manifest_path: str
    package_path: str
    package_digest: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    runtime_command: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path) -> "CompileResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            backend=str(data["backend"]),
            status=str(data["status"]),
            artifact_path=str(data["artifact_path"]),
            manifest_path=str(data["manifest_path"]),
            package_path=str(data["package_path"]),
            package_digest=str(data["package_digest"]),
            blockers=tuple(str(value) for value in data.get("blockers", [])),
            warnings=tuple(str(value) for value in data.get("warnings", [])),
            runtime_command=tuple(str(value) for value in data.get("runtime_command", [])),
            metadata=dict(data.get("metadata") or {}),
        )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return result or "object"


def _path_uri(value: str) -> Path | None:
    if value.startswith(("primitive://", "robotwin://", "http://", "https://")):
        return None
    return Path(value).expanduser().resolve()


def _asset_map(package: EnvironmentPackage) -> dict[str, AssetBundle]:
    return {asset.asset_id: asset for asset in package.assets}


def _primitive(asset: AssetBundle) -> AssetRepresentation | None:
    return next((item for item in asset.representations if item.format == "primitive_box"), None)


def _representation(
    asset: AssetBundle,
    *,
    backend: str,
    formats: Iterable[str],
) -> AssetRepresentation | None:
    accepted = tuple(format.lower() for format in formats)
    for representation in asset.representations:
        if representation.backend == backend and representation.format.lower() in accepted:
            return representation
    for representation in asset.representations:
        if representation.backend == "portable" and representation.format.lower() in accepted:
            return representation
    for representation in asset.representations:
        if representation.format.lower() in accepted:
            return representation
    return None


def _write_manifest(
    package: EnvironmentPackage,
    output: Path,
    *,
    backend: str,
    status: str,
    artifact: Path,
    blockers: list[str],
    warnings: list[str],
    runtime_command: tuple[str, ...],
    metadata: dict[str, Any] | None = None,
) -> CompileResult:
    package_path = package.write_json(output / "environment_package.json")
    manifest_path = output / "compile_manifest.json"
    result = CompileResult(
        backend=backend,
        status=status,
        artifact_path=str(artifact),
        manifest_path=str(manifest_path),
        package_path=str(package_path),
        package_digest=package.digest(),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        runtime_command=runtime_command,
        metadata={
            "schema": "agenticsim.backend_compile.v1",
            "task_contract": package.task.semantic_contract(),
            "object_ids": [obj.instance_id for obj in package.env.objects],
            "asset_ids": [asset.asset_id for asset in package.assets],
            **(metadata or {}),
        },
    )
    manifest_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _status(blockers: list[str]) -> str:
    return "compiled" if not blockers else "partial"


class BackendCompiler:
    backend = "unknown"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        raise NotImplementedError

    @staticmethod
    def _finish(result: CompileResult, *, strict: bool) -> CompileResult:
        if strict and result.blockers:
            raise BackendCompileError("; ".join(result.blockers))
        return result


_GENESIS_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    # Keep these values byte-for-byte aligned with scene_gen.colors.COLOR_RGB.
    # The adapter copies the stable table instead of importing the core compiler.
    "black": (0.08, 0.09, 0.10),
    "blue": (0.10, 0.32, 0.78),
    "brown": (0.38, 0.19, 0.08),
    "green": (0.12, 0.55, 0.28),
    "orange": (0.95, 0.38, 0.06),
    "pink": (0.93, 0.35, 0.58),
    "purple": (0.48, 0.18, 0.72),
    "red": (0.82, 0.10, 0.12),
    "white": (0.82, 0.84, 0.86),
    "yellow": (0.92, 0.72, 0.08),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _genesis_asset_integrity_blockers(
    representation: AssetRepresentation,
    path: Path,
) -> list[str]:
    """Validate the primary representation and its local dependency closure."""

    problems: list[str] = []
    if not representation.sha256:
        problems.append("representation is missing a required sha256")
    elif _sha256_file(path) != representation.sha256:
        problems.append("representation sha256 does not match the asset on disk")
    if path.stat().st_size != representation.size_bytes:
        problems.append(
            "representation size_bytes does not match the asset on disk "
            f"({representation.size_bytes} declared, {path.stat().st_size} actual)"
        )

    metadata = representation.metadata
    discovery = metadata.get("dependency_discovery")
    dependencies = metadata.get("dependencies")
    dependency_errors = metadata.get("dependency_errors")
    if not isinstance(discovery, str) or not discovery.strip():
        problems.append("representation is missing dependency_discovery metadata")
    if not isinstance(dependencies, list) or not isinstance(dependency_errors, list):
        problems.append("representation does not declare a local dependency closure")
        return problems
    problems.extend(f"dependency discovery: {value}" for value in dependency_errors)
    for index, record in enumerate(dependencies):
        if not isinstance(record, dict):
            problems.append(f"dependency {index} is not an object")
            continue
        dependency = _path_uri(str(record.get("uri") or ""))
        if dependency is None or not dependency.is_file():
            problems.append(f"dependency {index} does not exist: {record.get('uri')!r}")
            continue
        expected_sha = str(record.get("sha256") or "")
        expected_size = record.get("size_bytes")
        if _sha256_file(dependency) != expected_sha:
            problems.append(f"dependency sha256 mismatch: {dependency}")
        if expected_size != dependency.stat().st_size:
            problems.append(f"dependency size mismatch: {dependency}")
    return problems


def _genesis_dimensions(asset: AssetBundle) -> list[float]:
    values = asset.physical.get("dimensions_m")
    if isinstance(values, (list, tuple)) and len(values) == 3:
        try:
            dimensions = [float(value) for value in values]
        except (TypeError, ValueError):
            dimensions = []
        if len(dimensions) == 3 and all(value > 0.0 for value in dimensions):
            return dimensions
    return [0.1, 0.1, 0.1]


def _genesis_representation(asset: AssetBundle) -> AssetRepresentation | None:
    formats = {"obj", "stl", "dae", "glb", "gltf", "urdf"}
    for backend in ("genesis", "portable"):
        for representation in asset.representations:
            if representation.backend != backend:
                continue
            if representation.format.lower() not in formats:
                continue
            if representation.role not in {"visual", "visual_and_collision"}:
                continue
            return representation
    return None


def _genesis_articulation_blocker(
    articulation: dict[str, Any], path: Path
) -> str | None:
    names = [str(value) for value in articulation.get("joint_names") or []]
    qpos = list(articulation.get("qpos") or [])
    limits = list(articulation.get("joint_limits") or [])
    if not names and not qpos and not limits:
        return None
    if len(names) != len(qpos) or (limits and len(limits) != len(names)):
        return "articulation joint_names, qpos, and joint_limits have inconsistent lengths"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return f"cannot parse URDF articulation: {exc}"
    joint_types = {
        str(joint.get("name")): str(joint.get("type") or "")
        for joint in root.findall("joint")
        if joint.get("name")
    }
    missing = [name for name in names if name not in joint_types]
    if missing:
        return f"URDF is missing declared joints {missing}"
    unsupported = {
        name: joint_types[name]
        for name in names
        if joint_types[name] not in {"continuous", "prismatic", "revolute"}
    }
    if unsupported:
        return f"Genesis render v1 requires one-DoF articulation joints, got {unsupported}"
    return None


class GenesisCompiler(BackendCompiler):
    """Compile a portable package into a deterministic Genesis render bundle."""

    backend = "genesis"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "scene.json"
        runner = output / "run_genesis_render.py"
        assets = _asset_map(package)
        blockers: list[str] = []
        warnings: list[str] = []
        object_specs: list[dict[str, Any]] = []

        for obj in package.env.objects:
            asset = assets[obj.asset_id]
            dimensions = _genesis_dimensions(asset)
            color_name = str(obj.metadata.get("color") or "").strip().lower()
            color_rgb = _GENESIS_COLOR_RGB.get(color_name)
            if color_name and color_rgb is None:
                blockers.append(
                    f"{obj.instance_id}: unsupported explicit color override {color_name!r}"
                )
            item: dict[str, Any] = {
                "instance_id": obj.instance_id,
                "asset_id": obj.asset_id,
                "kind": "",
                "pose": {
                    "position": list(obj.pose.position),
                    "orientation_wxyz": list(obj.pose.orientation_wxyz),
                },
                "source_static": obj.static,
                "render_fixed": True,
                "scale": list(obj.scale),
                "dimensions_m": dimensions,
                "z_policy": obj.metadata.get("z_policy") or "center_on_table",
                "color": color_name or None,
                "color_rgb": list(color_rgb) if color_rgb is not None else None,
                "material": obj.metadata.get("material"),
                "articulation": dict(
                    obj.metadata.get("articulation") or asset.articulation
                ),
            }
            primitive = _primitive(asset)
            if primitive is not None:
                half = primitive.metadata.get("half_size_m") or [0.025, 0.025, 0.025]
                item.update(
                    {
                        "kind": "box",
                        "size_m": [
                            2.0 * float(half[index]) * obj.scale[index]
                            for index in range(3)
                        ],
                        "color_rgb": list(
                            color_rgb
                            or primitive.metadata.get("color_rgb")
                            or [0.8, 0.8, 0.8]
                        ),
                    }
                )
                object_specs.append(item)
                continue

            representation = _genesis_representation(asset)
            path = _path_uri(representation.uri) if representation is not None else None
            if representation is None or path is None or not path.is_file():
                available = sorted({rep.format.lower() for rep in asset.representations})
                blocker = (
                    f"{obj.instance_id}: no existing Genesis visual representation "
                    f"(supported: dae, glb, gltf, obj, stl, urdf; available: {available})"
                )
                blockers.append(blocker)
                item.update({"kind": "missing", "blocker": blocker})
                object_specs.append(item)
                continue

            fmt = representation.format.lower()
            integrity_blockers = _genesis_asset_integrity_blockers(representation, path)
            if integrity_blockers:
                blocker = f"{obj.instance_id}: " + "; ".join(integrity_blockers)
                blockers.append(blocker)
                item.update({"kind": "missing", "blocker": blocker})
                object_specs.append(item)
                continue
            item.update(
                {
                    "kind": "urdf" if fmt == "urdf" else "mesh",
                    "format": fmt,
                    "uri": str(path),
                    "representation_role": representation.role,
                    "source_sha256": representation.sha256,
                    "source_size_bytes": representation.size_bytes,
                    "source_dependencies": list(
                        representation.metadata.get("dependencies") or []
                    ),
                    "source_dependency_discovery": representation.metadata.get(
                        "dependency_discovery"
                    ),
                }
            )
            if fmt == "urdf":
                if max(obj.scale) - min(obj.scale) > 1e-9:
                    blocker = (
                        f"{obj.instance_id}: Genesis URDF requires uniform scale, "
                        f"got {list(obj.scale)}"
                    )
                    blockers.append(blocker)
                    item.update({"kind": "missing", "blocker": blocker})
                else:
                    articulation_blocker = _genesis_articulation_blocker(
                        item["articulation"], path
                    )
                    if articulation_blocker is not None:
                        blocker = f"{obj.instance_id}: {articulation_blocker}"
                        blockers.append(blocker)
                        item.update({"kind": "missing", "blocker": blocker})
                    else:
                        item["uniform_scale"] = float(obj.scale[0])
            else:
                item["file_meshes_are_zup"] = bool(
                    representation.metadata.get(
                        "file_meshes_are_zup", fmt not in {"glb", "gltf"}
                    )
                )
                item["genesis_scale"] = (
                    list(obj.scale)
                    if item["file_meshes_are_zup"]
                    else [obj.scale[0], obj.scale[2], obj.scale[1]]
                )
            object_specs.append(item)

        x_min, y_min, table_height, x_max, y_max, _ = package.env.workspace_bounds_m
        table_thickness = 0.04
        payload = {
            "schema": "agenticsim.genesis_render_scene.v1",
            "package_id": package.package_id,
            "package_digest": package.digest(),
            "package_path": "environment_package.json",
            "environment": {
                "name": package.env.name,
                "workspace_bounds_m": list(package.env.workspace_bounds_m),
                "table_height_m": float(table_height),
            },
            "table": {
                "center": [
                    (x_min + x_max) / 2.0,
                    (y_min + y_max) / 2.0,
                    table_height - table_thickness / 2.0,
                ],
                "size": [x_max - x_min, y_max - y_min, table_thickness],
                "color_rgb": [0.46, 0.31, 0.20],
            },
            "render": {
                "renderer": "genesis_rasterizer",
                "width": 640,
                "height": 480,
                "frames": 120,
                "fps": 12,
                "compute_backend": "cpu",
            },
            "objects": object_specs,
            "task_contract": package.task.semantic_contract(),
        }
        artifact.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runner.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n\n"
            "from agenticsim.openxsim.genesis_runtime import main\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=artifact,
            blockers=blockers,
            warnings=warnings,
            runtime_command=(
                sys.executable,
                str(runner),
                "--scene",
                str(artifact),
                "--output-dir",
                str(output / "render"),
            ),
            metadata={
                "artifact_format": "agenticsim.genesis_render_scene.v1",
                "render_runner": str(runner),
                "renderer": "genesis_rasterizer",
                "render_only": True,
            },
        )
        return self._finish(result, strict=strict)


def _usda_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


class IsaacSimCompiler(BackendCompiler):
    backend = "isaacsim"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "scene.usda"
        assets = _asset_map(package)
        blockers: list[str] = []
        warnings: list[str] = []
        lines = [
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            f'    upAxis = "{package.env.up_axis}"',
            ")",
            "",
            'def Xform "World"',
            "{",
            '    def PhysicsScene "PhysicsScene"',
            "    {",
            f"        vector3f physics:gravityDirection = ({package.env.gravity_mps2[0]}, {package.env.gravity_mps2[1]}, {package.env.gravity_mps2[2]})",
            "        float physics:gravityMagnitude = 1",
            "    }",
            '    def Cube "Ground"',
            "    {",
            "        double size = 1",
            "        double3 xformOp:scale = (4, 4, 0.02)",
            "        double3 xformOp:translate = (0, 0, -0.02)",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            "    }",
            '    def Scope "Objects"',
            "    {",
        ]
        for obj in package.env.objects:
            asset = assets[obj.asset_id]
            name = _slug(obj.instance_id)
            position = obj.pose.position
            orientation = obj.pose.orientation_wxyz
            primitive = _primitive(asset)
            if primitive:
                half = primitive.metadata.get("half_size_m") or [0.025, 0.025, 0.025]
                color = primitive.metadata.get("color_rgb") or [0.8, 0.8, 0.8]
                lines.extend(
                    [
                        f'        def Cube "{name}"',
                        "        {",
                        "            double size = 1",
                        f"            color3f[] primvars:displayColor = [({float(color[0])}, {float(color[1])}, {float(color[2])})]",
                        f"            double3 xformOp:scale = ({2 * float(half[0]) * obj.scale[0]}, {2 * float(half[1]) * obj.scale[1]}, {2 * float(half[2]) * obj.scale[2]})",
                        f"            quatd xformOp:orient = ({orientation[0]}, {orientation[1]}, {orientation[2]}, {orientation[3]})",
                        f"            double3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})",
                        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
                        "        }",
                    ]
                )
                continue
            representation = _representation(asset, backend=self.backend, formats=("usd", "usda", "usdc"))
            path = _path_uri(representation.uri) if representation else None
            if not representation or path is None or not path.is_file():
                blockers.append(f"{obj.instance_id}: no existing USD representation")
                lines.extend(
                    [
                        f'        def Xform "{name}"',
                        "        {",
                        f"            custom string agenticsim:blocker = {_usda_string('missing USD representation')}",
                        "        }",
                    ]
                )
                continue
            relative = os.path.relpath(path, artifact.parent).replace(os.sep, "/")
            lines.extend(
                [
                    f'        def Xform "{name}" (',
                    f"            prepend references = @{relative}@",
                    "        )",
                    "        {",
                    f"            quatd xformOp:orient = ({orientation[0]}, {orientation[1]}, {orientation[2]}, {orientation[3]})",
                    f"            double3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})",
                    f"            double3 xformOp:scale = ({obj.scale[0]}, {obj.scale[1]}, {obj.scale[2]})",
                    '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
                    "        }",
                ]
            )
        lines.extend(["    }", '    def Scope "Regions"', "    {"])
        for region in package.env.regions:
            name = _slug(str(region.get("id") or "region"))
            center = region.get("center") or [0.0, 0.0, 0.0]
            size = region.get("size") or [0.1, 0.1, 0.005]
            color = (region.get("color") or [0.2, 0.4, 0.8])[:3]
            lines.extend(
                [
                    f'        def Cube "{name}"',
                    "        {",
                    "            double size = 1",
                    f"            color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]",
                    f"            double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})",
                    f"            double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
                    '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                    "        }",
                ]
            )
        lines.extend(["    }", "}", ""])
        artifact.write_text("\n".join(lines), encoding="utf-8")
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=artifact,
            blockers=blockers,
            warnings=warnings,
            runtime_command=(
                "python",
                str(output / "run_isaac_smoke.py"),
                "--package",
                str(output / "environment_package.json"),
                "--output",
                str(output / "runtime_evidence.json"),
            ),
            metadata={"artifact_format": "usda", "runtime_scene_builder": str(output / "run_isaac_smoke.py")},
        )
        (output / "run_isaac_smoke.py").write_text(_isaac_smoke_script(), encoding="utf-8")
        return self._finish(result, strict=strict)


def _isaac_smoke_script() -> str:
    return '''#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--package", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "hide_ui": True, "fast_shutdown": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False, "max_gpu_count": 1})
try:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage

    package = json.loads(Path(args.package).read_text(encoding="utf-8"))
    contract = package["task"]
    semantic_contract = {key: contract[key] for key in ("reset", "action", "observation", "success", "termination")}
    contract_hash = hashlib.sha256(json.dumps(semantic_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    action_interface = str(contract.get("action", {}).get("interface", ""))
    success_types = {str(item.get("type", "")) for item in contract.get("success", [])}
    supported_success = {"state_trace_available", "object_below", "settled"}
    asset_map = {item["asset_id"]: item for item in package["assets"]}
    world = World(physics_dt=0.01, rendering_dt=0.01, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    actors = {}
    dynamic_names = []
    external_static_names = []
    for item in package["env"]["objects"]:
        asset = asset_map[item["asset_id"]]
        primitive = next((rep for rep in asset["representations"] if rep["format"] == "primitive_box"), None)
        if primitive is None:
            representation = next(
                (
                    rep
                    for rep in asset["representations"]
                    if rep["format"] in {"usd", "usda", "usdc"} and Path(rep["uri"]).is_file()
                ),
                None,
            )
            if representation is None:
                raise RuntimeError(f"Isaac smoke needs primitive_box or USD for {item['instance_id']}")
            name = item["instance_id"]
            prim_path = f"/World/OpenXSim/{name}"
            add_reference_to_stage(usd_path=representation["uri"], prim_path=prim_path)
            actors[name] = SingleXFormPrim(
                prim_path=prim_path,
                name=name,
                position=np.asarray(item["pose"]["position"]),
                orientation=np.asarray(item["pose"]["orientation_wxyz"]),
                scale=np.asarray(item.get("scale", [1, 1, 1])),
            )
            external_static_names.append(name)
            continue
        half = primitive["metadata"].get("half_size_m", [0.025, 0.025, 0.025])
        scale = np.asarray([2 * half[i] * item.get("scale", [1, 1, 1])[i] for i in range(3)])
        color = np.asarray(primitive["metadata"].get("color_rgb", [0.8, 0.8, 0.8]))
        position = np.asarray(item["pose"]["position"])
        orientation = np.asarray(item["pose"]["orientation_wxyz"])
        name = item["instance_id"]
        cls = FixedCuboid if item.get("static", False) else DynamicCuboid
        kwargs = {"prim_path": f"/World/OpenXSim/{name}", "name": name, "position": position, "orientation": orientation, "scale": scale, "color": color}
        if cls is DynamicCuboid:
            kwargs["mass"] = float(asset.get("physical", {}).get("mass_kg", 0.1))
            dynamic_names.append(name)
        actors[name] = world.scene.add(cls(**kwargs))
    world.reset()
    for name in dynamic_names:
        actors[name].set_linear_velocity(np.zeros(3))
        actors[name].set_angular_velocity(np.zeros(3))

    def snapshot(step):
        return {"step": step, "objects": {name: [float(value) for value in actor.get_world_pose()[0]] for name, actor in actors.items()}, "contacts": []}

    trajectory = [snapshot(0)]
    for step in range(1, args.steps + 1):
        world.step(render=False)
        trajectory.append(snapshot(step))
    runtime_stage = Path(args.output).with_name("runtime_scene.usda")
    world.stage.GetRootLayer().Export(str(runtime_stage))
    payload = {
        "schema": "agenticsim.runtime_evidence.v1",
        "backend": "isaacsim",
        "reset_ok": True,
        "step_ok": True,
        "steps": args.steps,
        "action_interface": action_interface,
        "action_interface_bound": action_interface in {"none", "zero_action"},
        "success_evaluator_bound": bool(success_types) and success_types <= supported_success,
        "task_contract_hash": contract_hash,
        "observation_keys": ["contact", "object_pose"],
        "trajectory_mode": "zero_action_physics_rollout",
        "trajectory": trajectory,
        "runtime_stage": str(runtime_stage),
        "external_asset_static_references": external_static_names,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
finally:
    app.close()
'''


class MuJoCoCompiler(BackendCompiler):
    backend = "mujoco"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "scene.xml"
        assets = _asset_map(package)
        blockers: list[str] = []
        mesh_entries: list[str] = []
        body_entries: list[str] = []
        for obj in package.env.objects:
            asset = assets[obj.asset_id]
            name = _slug(obj.instance_id)
            position = " ".join(f"{value:.9g}" for value in obj.pose.position)
            quat = " ".join(f"{value:.9g}" for value in obj.pose.orientation_wxyz)
            primitive = _primitive(asset)
            if primitive:
                half = primitive.metadata.get("half_size_m") or [0.025, 0.025, 0.025]
                size = " ".join(f"{float(half[index]) * obj.scale[index]:.9g}" for index in range(3))
                color = list(primitive.metadata.get("color_rgb") or [0.8, 0.8, 0.8]) + [1.0]
                rgba = " ".join(f"{float(value):.9g}" for value in color[:4])
                joint = "" if obj.static else f'<freejoint name="{name}_free"/>'
                body_entries.append(
                    f'    <body name="{html.escape(name)}" pos="{position}" quat="{quat}">{joint}'
                    f'<geom name="{html.escape(name)}_geom" type="box" size="{size}" rgba="{rgba}"/></body>'
                )
                continue
            representation = _representation(asset, backend=self.backend, formats=("obj", "stl"))
            path = _path_uri(representation.uri) if representation else None
            if not representation or path is None or not path.is_file():
                blockers.append(f"{obj.instance_id}: no existing OBJ/STL representation")
                continue
            mesh_name = f"{name}_mesh"
            relative = os.path.relpath(path, artifact.parent).replace(os.sep, "/")
            scale = " ".join(f"{value:.9g}" for value in obj.scale)
            mesh_entries.append(
                f'    <mesh name="{html.escape(mesh_name)}" file="{html.escape(relative)}" scale="{scale}"/>'
            )
            joint = "" if obj.static else f'<freejoint name="{name}_free"/>'
            body_entries.append(
                f'    <body name="{html.escape(name)}" pos="{position}" quat="{quat}">{joint}'
                f'<geom name="{html.escape(name)}_geom" type="mesh" mesh="{html.escape(mesh_name)}"/></body>'
            )
        for region in package.env.regions:
            name = _slug(str(region.get("id") or "region"))
            center = region.get("center") or [0.0, 0.0, 0.0]
            size = region.get("size") or [0.1, 0.1, 0.005]
            half = [float(value) / 2.0 for value in size]
            color = list(region.get("color") or [0.2, 0.4, 0.8, 0.35])
            body_entries.append(
                f'    <body name="{html.escape(name)}" pos="{" ".join(str(value) for value in center)}">'
                f'<geom type="box" size="{" ".join(str(value) for value in half)}" rgba="{" ".join(str(value) for value in color[:4])}" contype="0" conaffinity="0"/></body>'
            )
        task_text = html.escape(json.dumps(package.task.semantic_contract(), sort_keys=True, separators=(",", ":")))
        content = "\n".join(
            [
                f'<mujoco model="{html.escape(package.package_id)}">',
                '  <compiler angle="radian" coordinate="local"/>',
                f'  <option gravity="{" ".join(str(value) for value in package.env.gravity_mps2)}" timestep="0.01"/>',
                "  <asset>",
                *mesh_entries,
                "  </asset>",
                f'  <custom><text name="agenticsim_task_contract" data="{task_text}"/></custom>',
                "  <worldbody>",
                '    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.3 0.3 0.3 1"/>',
                *body_entries,
                "  </worldbody>",
                "</mujoco>",
                "",
            ]
        )
        artifact.write_text(content, encoding="utf-8")
        try:
            ET.parse(artifact)
        except ET.ParseError as exc:
            raise BackendCompileError(f"generated MJCF is not XML: {exc}") from exc
        smoke = output / "run_mujoco_smoke.py"
        smoke.write_text(_mujoco_smoke_script(), encoding="utf-8")
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=artifact,
            blockers=blockers,
            warnings=[],
            runtime_command=("python", str(smoke), "--model", str(artifact), "--output", str(output / "runtime_evidence.json")),
            metadata={"artifact_format": "mjcf", "xml_validated": True},
        )
        return self._finish(result, strict=strict)


def _mujoco_smoke_script() -> str:
    return '''#!/usr/bin/env python3
import argparse, hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
root = ET.parse(args.model).getroot()
custom = root.find("./custom/text[@name='agenticsim_task_contract']")
contract = json.loads(custom.get("data")) if custom is not None else {}
contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
action_interface = str(contract.get("action", {}).get("interface", ""))
success_types = {str(item.get("type", "")) for item in contract.get("success", [])}
supported_success = {"state_trace_available", "object_below", "settled"}
model = mujoco.MjModel.from_xml_path(args.model)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
initial = data.qpos.tolist()
body_ids = [index for index in range(1, model.nbody) if model.body_jntnum[index] > 0]

def snapshot(step):
    objects = {}
    for body_id in body_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        objects[name] = [float(value) for value in data.xpos[body_id]]
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        names = sorted(filter(None, [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)]))
        if names:
            contacts.append(":".join(names))
    return {"step": step, "objects": objects, "contacts": sorted(set(contacts))}

trajectory = [snapshot(0)]
for step in range(1, args.steps + 1):
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    trajectory.append(snapshot(step))
payload = {
    "schema": "agenticsim.runtime_evidence.v1",
    "backend": "mujoco",
    "reset_ok": True,
    "step_ok": True,
    "steps": args.steps,
    "qpos_initial": initial,
    "qpos_final": data.qpos.tolist(),
    "action_interface": action_interface,
    "action_interface_bound": action_interface in {"none", "zero_action"},
    "success_evaluator_bound": bool(success_types) and success_types <= supported_success,
    "task_contract_hash": contract_hash,
    "observation_keys": ["contact", "object_pose"],
    "trajectory_mode": "zero_action_physics_rollout",
    "trajectory": trajectory,
}
Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
'''


class SapienCompiler(BackendCompiler):
    backend = "sapien"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "scene_spec.json"
        runner = output / "run_sapien_smoke.py"
        assets = _asset_map(package)
        blockers: list[str] = []
        object_specs: list[dict[str, Any]] = []
        for obj in package.env.objects:
            asset = assets[obj.asset_id]
            primitive = _primitive(asset)
            item: dict[str, Any] = {
                "name": obj.instance_id,
                "position": list(obj.pose.position),
                "orientation_wxyz": list(obj.pose.orientation_wxyz),
                "static": obj.static,
                "scale": list(obj.scale),
            }
            if primitive:
                item.update(
                    {
                        "kind": "box",
                        "half_size_m": list(primitive.metadata.get("half_size_m") or [0.025, 0.025, 0.025]),
                        "color_rgb": list(primitive.metadata.get("color_rgb") or [0.8, 0.8, 0.8]),
                    }
                )
            else:
                representation = _representation(
                    asset, backend=self.backend, formats=("urdf", "obj", "stl", "ply", "glb")
                )
                path = _path_uri(representation.uri) if representation else None
                if not representation or path is None or not path.exists():
                    blockers.append(f"{obj.instance_id}: no existing SAPIEN-loadable representation")
                    item.update({"kind": "missing", "blocker": blockers[-1]})
                else:
                    item.update({"kind": representation.format, "path": str(path)})
            object_specs.append(item)
        payload = {
            "schema": "agenticsim.sapien_scene.v1",
            "package_id": package.package_id,
            "gravity_mps2": list(package.env.gravity_mps2),
            "objects": object_specs,
            "regions": list(package.env.regions),
            "task_contract": package.task.semantic_contract(),
        }
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        runner.write_text(_sapien_smoke_script(), encoding="utf-8")
        ast.parse(runner.read_text(encoding="utf-8"))
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=artifact,
            blockers=blockers,
            warnings=[],
            runtime_command=("python", str(runner), "--spec", str(artifact), "--output", str(output / "runtime_evidence.json")),
            metadata={"artifact_format": "sapien_scene_json", "runner_syntax_validated": True},
        )
        return self._finish(result, strict=strict)


def _sapien_smoke_script() -> str:
    return '''#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import sapien

parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
contract = spec.get("task_contract", {})
contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
action_interface = str(contract.get("action", {}).get("interface", ""))
success_types = {str(item.get("type", "")) for item in contract.get("success", [])}
supported_success = {"state_trace_available", "object_below", "settled"}

if hasattr(sapien, "Engine"):
    engine = sapien.Engine()
    scene = engine.create_scene()
else:
    scene = sapien.Scene()
scene.set_timestep(0.01)
if hasattr(scene, "set_gravity"):
    scene.set_gravity(spec["gravity_mps2"])
scene.add_ground(0.0)
actors = {}
for item in spec["objects"]:
    if item["kind"] == "missing":
        raise RuntimeError(item["blocker"])
    if item["kind"] == "urdf":
        loader = scene.create_urdf_loader()
        loader.fix_root_link = bool(item["static"])
        actor = loader.load(item["path"])
    else:
        builder = scene.create_actor_builder()
        if item["kind"] == "box":
            half = [item["half_size_m"][i] * item["scale"][i] for i in range(3)]
            builder.add_box_collision(half_size=half)
            builder.add_box_visual(half_size=half, material=item["color_rgb"])
        else:
            builder.add_visual_from_file(item["path"], scale=item["scale"])
            builder.add_multiple_convex_collisions_from_file(item["path"], scale=item["scale"])
        actor = builder.build_static(name=item["name"]) if item["static"] else builder.build(name=item["name"])
    actor.set_pose(sapien.Pose(item["position"], item["orientation_wxyz"]))
    actors[item["name"]] = actor
initial = {name: actor.get_pose().p.tolist() for name, actor in actors.items()}

def snapshot(step):
    contacts = []
    if hasattr(scene, "get_contacts"):
        try:
            for contact in scene.get_contacts():
                names = sorted(filter(None, [getattr(getattr(contact, "bodies", [None, None])[0], "name", None), getattr(getattr(contact, "bodies", [None, None])[1], "name", None)]))
                if names:
                    contacts.append(":".join(names))
        except Exception:
            contacts = []
    return {"step": step, "objects": {name: [float(value) for value in actor.get_pose().p] for name, actor in actors.items()}, "contacts": sorted(set(contacts))}

trajectory = [snapshot(0)]
for step in range(1, args.steps + 1):
    scene.step()
    trajectory.append(snapshot(step))
final = {name: actor.get_pose().p.tolist() for name, actor in actors.items()}
payload = {
    "schema": "agenticsim.runtime_evidence.v1",
    "backend": "sapien",
    "reset_ok": True,
    "step_ok": True,
    "steps": args.steps,
    "object_positions_initial": initial,
    "object_positions_final": final,
    "action_interface": action_interface,
    "action_interface_bound": action_interface in {"none", "zero_action"},
    "success_evaluator_bound": bool(success_types) and success_types <= supported_success,
    "task_contract_hash": contract_hash,
    "observation_keys": ["contact", "object_pose"],
    "trajectory_mode": "zero_action_physics_rollout",
    "trajectory": trajectory,
}
Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
'''


class MetaSimCompiler(BackendCompiler):
    backend = "metasim"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "scenario.py"
        config_path = output / "scenario.json"
        assets = _asset_map(package)
        blockers: list[str] = []
        config_objects: list[dict[str, Any]] = []
        expressions: list[str] = []
        for obj in package.env.objects:
            asset = assets[obj.asset_id]
            primitive = _primitive(asset)
            if primitive:
                half = primitive.metadata.get("half_size_m") or [0.025, 0.025, 0.025]
                size = [2 * float(half[index]) * obj.scale[index] for index in range(3)]
                color = list(primitive.metadata.get("color_rgb") or [0.8, 0.8, 0.8])
                kwargs = {
                    "name": obj.instance_id,
                    "size": size,
                    "color": color,
                    "mass": float(asset.physical.get("mass_kg", 0.1)),
                    "default_position": list(obj.pose.position),
                    "default_orientation": list(obj.pose.orientation_wxyz),
                    "fix_base_link": obj.static,
                }
                expressions.append(f"PrimitiveCubeCfg(**{kwargs!r})")
                config_objects.append({"class": "PrimitiveCubeCfg", **kwargs})
                continue
            paths: dict[str, str | None] = {"mesh_path": None, "usd_path": None, "urdf_path": None, "mjcf_path": None}
            mappings = {
                "mesh_path": ("obj", "stl", "ply", "glb", "gltf"),
                "usd_path": ("usd", "usda", "usdc"),
                "urdf_path": ("urdf",),
                "mjcf_path": ("mjcf",),
            }
            for field_name, formats in mappings.items():
                representation = _representation(asset, backend=self.backend, formats=formats)
                if representation:
                    path = _path_uri(representation.uri)
                    if path and path.is_file():
                        paths[field_name] = str(path)
            if not any(paths.values()):
                blockers.append(f"{obj.instance_id}: no existing MetaSim-loadable representation")
                continue
            kwargs = {
                "name": obj.instance_id,
                **paths,
                "default_position": list(obj.pose.position),
                "default_orientation": list(obj.pose.orientation_wxyz),
                "scale": list(obj.scale),
                "fix_base_link": obj.static,
            }
            expressions.append(f"RigidObjCfg(**{kwargs!r})")
            config_objects.append({"class": "RigidObjCfg", **kwargs})
        script = "\n".join(
            [
                '"""Generated by AgenticSim Open-X-Sim."""',
                "from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg",
                "from metasim.scenario.scenario import ScenarioCfg",
                "",
                "objects = [",
                *[f"    {expression}," for expression in expressions],
                "]",
                f"cfg = ScenarioCfg(objects=objects, gravity={package.env.gravity_mps2!r}, headless=True, simulator=None)",
                "",
            ]
        )
        artifact.write_text(script, encoding="utf-8")
        ast.parse(script)
        config_path.write_text(
            json.dumps(
                {
                    "schema": "agenticsim.metasim_scenario.v1",
                    "package_id": package.package_id,
                    "simulator": None,
                    "gravity": list(package.env.gravity_mps2),
                    "objects": config_objects,
                    "task_contract": package.task.semantic_contract(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=artifact,
            blockers=blockers,
            warnings=[],
            runtime_command=("python", "-c", f"exec(open({str(artifact)!r}).read()); print(cfg)"),
            metadata={
                "artifact_format": "metasim_scenario_cfg",
                "scenario_json": str(config_path),
                "runner_syntax_validated": True,
            },
        )
        return self._finish(result, strict=strict)


class RoboTwinCompiler(BackendCompiler):
    backend = "robotwin"

    def compile(self, package: EnvironmentPackage, output_dir: str | Path, *, strict: bool = False) -> CompileResult:
        package.validate()
        output = Path(output_dir).expanduser().resolve() / self.backend
        output.mkdir(parents=True, exist_ok=True)
        blockers: list[str] = []
        try:
            placement_path, task_program_path = write_robotwin_bundle(package, output)
        except RoboTwinExportError as exc:
            blockers.append(str(exc))
            placement_path = output / "placement.json"
            task_program_path = output / "task_program.json"
            placement_path.write_text("{}\n", encoding="utf-8")
            task_program_path.write_text("{}\n", encoding="utf-8")
        result = _write_manifest(
            package,
            output,
            backend=self.backend,
            status=_status(blockers),
            artifact=task_program_path,
            blockers=blockers,
            warnings=[],
            runtime_command=(
                "python",
                "scripts/run_generated_selection2env_rollout_probe.py",
                "--robotwin-root",
                "external/RoboTwin",
                "--task-program-input",
                str(task_program_path),
                "--out-dir",
                str(output / "runtime"),
            ),
            metadata={
                "artifact_format": "robotwin_selection2env_task_program",
                "placement_path": str(placement_path),
            },
        )
        return self._finish(result, strict=strict)


COMPILERS: dict[str, type[BackendCompiler]] = {
    "genesis": GenesisCompiler,
    "isaac": IsaacSimCompiler,
    "isaacsim": IsaacSimCompiler,
    "mujoco": MuJoCoCompiler,
    "sapien": SapienCompiler,
    "sapien3": SapienCompiler,
    "metasim": MetaSimCompiler,
    "robotwin": RoboTwinCompiler,
}


def compile_package(
    package: EnvironmentPackage,
    output_dir: str | Path,
    backends: Iterable[str],
    *,
    strict: bool = False,
) -> dict[str, CompileResult]:
    """Compile one canonical package through every requested backend."""

    results: dict[str, CompileResult] = {}
    for name in dict.fromkeys(value.lower() for value in backends):
        compiler_type = COMPILERS.get(name)
        if compiler_type is None:
            raise BackendCompileError(f"unknown backend: {name}")
        compiler = compiler_type()
        results[compiler.backend] = compiler.compile(package, output_dir, strict=strict)
    return results

"""Early object records and read-only reuse of complete, hash-bound CLI outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .builder import _write_json, verify_package
from .catalog import AssetCatalog, load_catalog
from .grounding import resolve_scene_assets
from .schema import ResolvedSceneSpec, SceneSpec, SceneSpecError
from .validator import validate_resolved_scene


def parse_evidence(spec: SceneSpec, evidence: dict | None) -> dict | None:
    if evidence is None:
        return None
    return {
        **evidence,
        "scene_spec_sha256": spec.digest(),
        "request_sha256": hashlib.sha256(spec.request.encode("utf-8")).hexdigest(),
    }


def object_documents(spec: SceneSpec, evidence: dict | None) -> dict:
    documents = {
        "scene_spec.json": spec.canonical_dict(),
        "relations.json": [r.model_dump(mode="json") for r in spec.relations],
    }
    documents.update(
        {f"objects/{obj.object_id}.json": obj.model_dump(mode="json") for obj in spec.objects}
    )
    if evidence is not None:
        documents["llm_parse_evidence.json"] = evidence
    return documents


def write_object_records(spec: SceneSpec, out: Path, evidence: dict | None) -> tuple[str, ...]:
    """Only write into a new/empty output; never modify a previous attempt."""
    out.mkdir(parents=True, exist_ok=True)
    if out.is_symlink() or any(out.iterdir()):
        raise SceneSpecError("output already exists; use a new --out-root")
    (out / "objects").mkdir()
    (out / "request.txt").write_text(spec.request + "\n", encoding="utf-8")
    documents = object_documents(spec, evidence)
    for name, value in documents.items():
        _write_json(out / name, value)
    return tuple(name for name in documents if name != "scene_spec.json")


def resolution_error(spec: SceneSpec, error: Exception) -> dict:
    return {
        "schema_version": "robotwin.asset_resolution.v1",
        "scene_id": spec.scene_id,
        "source_scene_spec_sha256": spec.digest(),
        "asset_catalog_sha256": None,
        "status": "error",
        "error_type": type(error).__name__,
        "reason": str(error),
        "objects": [],
    }


def check_asset_bindings(report: dict, resolved: ResolvedSceneSpec) -> None:
    if report["status"] != "matched":
        raise SceneSpecError("cannot solve an unresolved object list")
    declared = {o["object_id"]: (o["asset_id"], o["model_id"]) for o in report["objects"]}
    actual = {o.object_id: (o.asset_id, o.model_id) for o in resolved.objects}
    if (
        declared != actual
        or report["source_scene_spec_sha256"] != resolved.source_scene_spec_sha256
        or report["asset_catalog_sha256"] != resolved.asset_catalog_sha256
    ):
        raise SceneSpecError("asset resolution and resolved scene bindings differ")


def reuse_scene_package(
    spec: SceneSpec,
    catalog: AssetCatalog,
    out: Path,
    evidence: dict | None,
    generation_options: dict,
) -> dict:
    """Reuse is verification, not regeneration or repair of an existing directory."""
    try:
        previous_validation = json.loads(
            (out / "validation_report.json").read_text(encoding="utf-8")
        )
        if (
            previous_validation["status"] not in {"pass", "incomplete"}
            or (out / "failure_report.json").exists()
        ):
            raise ValueError("previous attempt did not finish successfully")
        if out.is_symlink() or verify_package(out)["status"] != "pass":
            raise ValueError("package integrity failed")
        manifest = json.loads((out / "package_manifest.json").read_text(encoding="utf-8"))
        documents = object_documents(spec, evidence)
        if {p.name for p in (out / "objects").iterdir()} != {
            f"{obj.object_id}.json" for obj in spec.objects
        }:
            raise ValueError("object file set differs")
        required = {
            *documents,
            "request.txt",
            "resolved_scene.json",
            "generated_scene.py",
            "asset_resolution.json",
        }
        if generation_options["enabled"]:
            required.update({"asset_generation_report.json", "effective_asset_catalog.json"})
        listed = {f["path"] for f in manifest["files"]}
        if not required <= listed:
            raise ValueError("required object records not hash-bound")
        if (out / "request.txt").read_text(encoding="utf-8") != spec.request + "\n":
            raise ValueError("request differs")
        for name, value in documents.items():
            if json.loads((out / name).read_text(encoding="utf-8")) != value:
                raise ValueError(f"parsed input differs: {name}")
        if evidence is None and (out / "llm_parse_evidence.json").exists():
            raise ValueError("parser provider differs")
        report = json.loads((out / "asset_resolution.json").read_text(encoding="utf-8"))
        if (
            report["input_asset_catalog_sha256"] != catalog.digest()
            or report["generation_options"] != generation_options
        ):
            raise ValueError("catalog or asset generation options differ")
        effective = (
            load_catalog(out / "effective_asset_catalog.json")
            if generation_options["enabled"]
            else catalog
        )
        expected = resolve_scene_assets(spec, effective)
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError("asset resolution differs")
        resolved = ResolvedSceneSpec.model_validate_json(
            (out / "resolved_scene.json").read_text(encoding="utf-8")
        )
        check_asset_bindings(report, resolved)
        validation = validate_resolved_scene(resolved, package_root=out, require_runtime=False)
        if validation["status"] not in {"pass", "incomplete"}:
            raise ValueError("static validation failed")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SceneSpecError(
            f"existing output cannot be reused ({error}); use a new --out-root"
        ) from error

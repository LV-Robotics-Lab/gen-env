#!/usr/bin/env python3
"""Compile text into a deterministic RoboTwin generated-scene package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from scene_gen.asset_generator import ensure_assets_for_scene
from scene_gen.builder import _write_json, build_scene_package
from scene_gen.catalog import load_catalog
from scene_gen.grounding import resolve_scene_assets
from scene_gen.object_records import (
    check_asset_bindings,
    parse_evidence,
    resolution_error,
    reuse_scene_package,
    write_object_records,
)
from scene_gen.parser import parse_rule_based, parse_with_provider
from scene_gen.schema import SceneSpecError
from scene_gen.solver import SceneSolveError, solve_scene
from scene_gen.validator import validate_resolved_scene

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


def _write_input_failure(
    error: Exception,
    *,
    out_root: Path,
    prompt: str,
    seed: int,
    provider: str,
    stage: str,
    blocker: str,
    llm_evidence: dict[str, object] | None = None,
) -> Path:
    legacy_rule_failure = provider == "rule" and stage == "scene_spec_validation"
    identity = (
        f"{seed}\0{prompt}" if legacy_rule_failure else f"{seed}\0{prompt}\0{provider}\0{stage}"
    )
    failure_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    failure_path = out_root / "_failures" / failure_id / "failure_report.json"
    if isinstance(error, ValidationError):
        details = error.errors()
    elif hasattr(error, "safe_details"):
        details = [error.safe_details()]
    else:
        details = [{"message": str(error)}]
    report: dict[str, object] = {
        "schema_version": "robotwin.scene_generation_failure.v1",
        "status": "fail",
        "stage": stage,
        "blocker": blocker,
        "error_type": type(error).__name__,
        "request": prompt,
        "seed": seed,
        "details": details,
    }
    if not legacy_rule_failure:
        report["provider"] = provider
    if llm_evidence:
        report["llm_extraction"] = llm_evidence
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return failure_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asset-catalog", required=True)
    parser.add_argument(
        "--out-root", default=str(DEFAULT_OUTPUT_ROOT), help="场景输出根目录（默认：仓库 output/）"
    )
    parser.add_argument("--generate-missing-assets", action="store_true")
    parser.add_argument("--generated-objects-root")
    parser.add_argument("--provider", choices=("rule", "llm"), default="rule")
    parser.add_argument("--llm-config")
    parser.add_argument("--llm-profile")
    args = parser.parse_args()
    if args.provider != "llm" and (args.llm_config or args.llm_profile):
        parser.error("--llm-config/--llm-profile require --provider llm")

    out_root = Path(args.out_root)
    llm_provider = None
    try:
        if args.provider == "rule":
            spec = parse_rule_based(args.prompt, seed=args.seed)
        else:
            from scene_gen.llm_provider import LLMSceneProvider

            llm_provider = LLMSceneProvider(
                config_path=args.llm_config,
                profile=args.llm_profile,
            )
            spec = parse_with_provider(llm_provider, request=args.prompt, seed=args.seed)
    except (SceneSpecError, ValidationError) as error:
        is_llm = args.provider == "llm"
        failure_path = _write_input_failure(
            error,
            out_root=out_root,
            prompt=args.prompt,
            seed=args.seed,
            provider=args.provider,
            stage="llm_scene_extraction" if is_llm else "scene_spec_validation",
            blocker=(
                "LLM extraction rejected before grounding"
                if is_llm
                else "request rejected before grounding"
            ),
            llm_evidence=llm_provider.evidence() if llm_provider is not None else None,
        )
        print(f"FAIL {failure_path}")
        return 2

    out_dir = out_root / spec.scene_id
    evidence = parse_evidence(spec, llm_provider.evidence() if llm_provider else None)
    generation_options = {
        "enabled": args.generate_missing_assets,
        "objects_root": (
            str(Path(args.generated_objects_root).expanduser().resolve())
            if args.generate_missing_assets and args.generated_objects_root
            else None
        ),
    }
    existing_output = out_dir.is_symlink() or (
        out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir()))
    )
    owns_output = False

    def fail(error, *, stage, blocker):
        failure_path = _write_input_failure(
            error,
            out_root=out_root,
            prompt=args.prompt,
            seed=args.seed,
            provider=args.provider,
            stage=stage,
            blocker=blocker,
            llm_evidence=llm_provider.evidence() if llm_provider is not None else None,
        )
        if owns_output:
            _write_json(
                out_dir / "failure_report.json",
                json.loads(failure_path.read_text(encoding="utf-8")),
            )
        print(f"FAIL {failure_path}")
        return 2

    additional_files: tuple[str, ...] = ()
    if not existing_output:
        try:
            additional_files = write_object_records(spec, out_dir, evidence)
            owns_output = True
        except (OSError, SceneSpecError) as error:
            return fail(error, stage="output_preservation", blocker="use a new --out-root")

    try:
        catalog = load_catalog(Path(args.asset_catalog))
    except (OSError, UnicodeError, ValidationError) as error:
        if owns_output:
            _write_json(out_dir / "asset_resolution.json", resolution_error(spec, error))
        return fail(
            error, stage="asset_grounding", blocker="asset catalog could not be loaded or validated"
        )

    if existing_output:
        try:
            manifest = reuse_scene_package(spec, catalog, out_dir, evidence, generation_options)
        except SceneSpecError as error:
            return fail(error, stage="output_preservation", blocker="use a new --out-root")
        print(
            f"PASS scene_id={spec.scene_id} "
            f"resolved_sha256={manifest['resolved_scene_sha256']} reused=true"
        )
        return 0

    input_catalog_digest = catalog.digest()
    try:
        if args.generate_missing_assets:
            catalog, generation_report = ensure_assets_for_scene(
                spec,
                catalog,
                objects_root=(
                    Path(args.generated_objects_root) if args.generated_objects_root else None
                ),
            )
            (out_dir / "asset_generation_report.json").write_text(
                json.dumps(generation_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (out_dir / "effective_asset_catalog.json").write_text(
                json.dumps(catalog.canonical_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            additional_files += ("asset_generation_report.json", "effective_asset_catalog.json")
        resolution = resolve_scene_assets(spec, catalog)
        resolution.update(
            input_asset_catalog_sha256=input_catalog_digest, generation_options=generation_options
        )
        _write_json(out_dir / "asset_resolution.json", resolution)
        additional_files += ("asset_resolution.json",)
        if resolution["status"] != "matched":
            unresolved = [
                f"{item['object_id']}={item['status']}"
                for item in resolution["objects"]
                if item["status"] != "matched"
            ]
            raise SceneSpecError("asset resolution incomplete: " + ", ".join(unresolved))
        resolved = solve_scene(spec, catalog)
        check_asset_bindings(resolution, resolved)
    except SceneSpecError as error:
        if not (out_dir / "asset_resolution.json").exists():
            failure = resolution_error(spec, error)
            failure.update(
                input_asset_catalog_sha256=input_catalog_digest,
                asset_catalog_sha256=catalog.digest(),
                generation_options=generation_options,
            )
            _write_json(out_dir / "asset_resolution.json", failure)
        return fail(
            error,
            stage="asset_grounding",
            blocker="semantic category or attributes cannot reach a usable catalog asset",
        )
    except SceneSolveError as error:
        solve_report = error.report
        if llm_provider is not None:
            solve_report = {
                **solve_report,
                "provider": "llm",
                "llm_extraction": llm_provider.evidence(),
            }
        (out_dir / "failure_report.json").write_text(
            json.dumps(solve_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"FAIL {out_dir / 'failure_report.json'}")
        return 2
    manifest = build_scene_package(
        spec,
        resolved,
        out_dir,
        additional_files=additional_files,
    )
    report = validate_resolved_scene(resolved, package_root=out_dir, require_runtime=False)
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"PASS scene_id={spec.scene_id} resolved_sha256={manifest['resolved_scene_sha256']} "
        f"validation={report['status']}"
    )
    return 0 if report["status"] in {"pass", "incomplete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

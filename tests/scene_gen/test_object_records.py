from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scene_gen.builder import verify_package
from scene_gen.catalog import AssetCatalog, load_catalog
from scene_gen.grounding import ground_object, resolve_scene_assets
from scene_gen.object_records import check_asset_bindings, object_documents, write_object_records
from scene_gen.parser import parse_provider_payload, parse_rule_based
from scene_gen.schema import ResolvedSceneSpec, SceneSpecError
from scene_gen.solver import SceneSolveError, solve_scene
from script import generate_scene as cli

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "asset_catalog.json"
PROMPT = "Place a can on top of a plate."


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def invoke(monkeypatch, root, *, prompt=PROMPT, catalog=CATALOG, seed=0, extra=()):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scene.py",
            "--prompt",
            prompt,
            "--seed",
            str(seed),
            "--asset-catalog",
            str(catalog),
            "--out-root",
            str(root),
            *extra,
        ],
    )
    return cli.main()


def scene_dir(root, prompt=PROMPT):
    return root / parse_rule_based(prompt).scene_id


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_success_records_match_spec_and_original_solver(tmp_path, monkeypatch):
    spec = parse_rule_based(PROMPT, seed=0)
    catalog = load_catalog(CATALOG)
    expected = solve_scene(spec, catalog)
    assert invoke(monkeypatch, tmp_path) == 0
    out = scene_dir(tmp_path)
    assert read(out / "scene_spec.json") == spec.canonical_dict()
    for name, value in object_documents(spec, None).items():
        assert read(out / name) == value
    assert read(out / "resolved_scene.json") == expected.canonical_dict()
    report = read(out / "asset_resolution.json")
    assert report["status"] == "matched"
    assert report["input_asset_catalog_sha256"] == catalog.digest()
    check_asset_bindings(report, expected)
    records = {r["path"] for r in read(out / "package_manifest.json")["files"]}
    assert {
        "objects/can_1.json",
        "objects/plate_1.json",
        "relations.json",
        "asset_resolution.json",
    } <= records
    assert verify_package(out)["status"] == "pass"
    assert not (out / "llm_parse_evidence.json").exists()


def test_records_exist_before_catalog_load(tmp_path, monkeypatch):
    def inspect_catalog(path):
        out = scene_dir(tmp_path)
        assert (out / "objects" / "can_1.json").is_file()
        assert (out / "relations.json").is_file()
        assert (out / "request.txt").is_file()
        return load_catalog(path)

    monkeypatch.setattr(cli, "load_catalog", inspect_catalog)
    assert invoke(monkeypatch, tmp_path) == 0


def test_same_category_instances_can_share_one_asset(tmp_path):
    spec = parse_rule_based("Place two cans on the table.")
    assert len(spec.objects) == 2
    write_object_records(spec, tmp_path, None)
    results = resolve_scene_assets(spec, load_catalog(CATALOG))["objects"]
    assert len({o["object_id"] for o in results}) == 2
    assert len({(o["asset_id"], o["model_id"]) for o in results}) == 1
    assert len(list((tmp_path / "objects").glob("*.json"))) == 2


def test_collects_all_misses_and_never_solves_or_generates(tmp_path, monkeypatch):
    spec = parse_provider_payload(
        {
            "objects": [{"object_id": f"{c}_1", "category": c} for c in ("spoon", "can", "fork")],
            "relations": [
                {"relation": "on_table", "source": f"{c}_1", "target": "table"}
                for c in ("spoon", "can", "fork")
            ],
        },
        request="Place a spoon, a can and a fork on the table.",
        seed=0,
    )
    monkeypatch.setattr(cli, "parse_rule_based", lambda *a, **k: spec)
    monkeypatch.setattr(cli, "solve_scene", lambda *a: pytest.fail("solver called"))
    monkeypatch.setattr(cli, "ensure_assets_for_scene", lambda *a, **k: pytest.fail("generated"))
    assert invoke(monkeypatch, tmp_path, prompt=spec.request) == 2
    out = tmp_path / spec.scene_id
    report = read(out / "asset_resolution.json")
    assert {o["object_id"]: o["status"] for o in report["objects"]} == {
        "can_1": "matched",
        "fork_1": "missing",
        "spoon_1": "missing",
    }
    assert len(list((out / "objects").glob("*.json"))) == 3
    assert (out / "failure_report.json").is_file()
    assert not (out / "resolved_scene.json").exists()
    assert not (out / "package_manifest.json").exists()


@pytest.mark.parametrize(
    "kind", ["entry_unavailable", "model_unusable", "no_models", "wrong_color"]
)
def test_missing_and_blocked_are_distinct(kind):
    catalog = load_catalog(CATALOG)
    entry = next(e for e in catalog.entries if e.category == "can")
    query = parse_rule_based("Place a red can on the table.")
    if kind == "entry_unavailable":
        entry = entry.model_copy(update={"available": False, "availability_reasons": ("offline",)})
    elif kind == "model_unusable":
        entry = entry.model_copy(
            update={
                "models": tuple(
                    m.model_copy(update={"usable": False, "missing": ("collision missing",)})
                    for m in entry.models
                )
            }
        )
    elif kind == "no_models":
        entry = entry.model_copy(update={"models": ()})
    else:
        entry = entry.model_copy(
            update={
                "models": tuple(m.model_copy(update={"colors": ("blue",)}) for m in entry.models)
            }
        )
    catalog = catalog.model_copy(update={"entries": (entry,)})
    report = resolve_scene_assets(query, catalog)
    result = report["objects"][0]
    assert result["status"] == ("missing" if kind == "wrong_color" else "blocked")
    assert result["rejected_candidates"]
    with pytest.raises(SceneSpecError):
        ground_object(query.objects[0], catalog, seed=0)


@pytest.mark.parametrize("content", [None, "{invalid", "{}"])
def test_catalog_errors_keep_parsed_records(tmp_path, monkeypatch, content):
    catalog = tmp_path / "catalog.json"
    if content is not None:
        catalog.write_text(content)
    root = tmp_path / "output"
    assert invoke(monkeypatch, root, catalog=catalog) == 2
    out = scene_dir(root)
    assert (out / "objects" / "can_1.json").is_file()
    report = read(out / "asset_resolution.json")
    assert report["status"] == "error"
    assert report["objects"] == []
    assert report["asset_catalog_sha256"] is None
    assert not (out / "package_manifest.json").exists()


def test_solver_failure_retains_all_records(tmp_path, monkeypatch):
    def fail(*args):
        raise SceneSolveError({"status": "fail", "blocker": "cannot place"})

    monkeypatch.setattr(cli, "solve_scene", fail)
    assert invoke(monkeypatch, tmp_path) == 2
    out = scene_dir(tmp_path)
    assert read(out / "asset_resolution.json")["status"] == "matched"
    assert (out / "objects" / "can_1.json").is_file()
    assert read(out / "failure_report.json")["status"] == "fail"


def test_reuse_does_not_solve_or_modify_output(tmp_path, monkeypatch):
    assert invoke(monkeypatch, tmp_path) == 0
    out = scene_dir(tmp_path)
    before = snapshot(out)
    monkeypatch.setattr(cli, "solve_scene", lambda *a: pytest.fail("solver called during reuse"))
    assert invoke(monkeypatch, tmp_path) == 0
    assert snapshot(out) == before


@pytest.mark.parametrize("changed", ["catalog", "seed", "evidence", "generation_options"])
def test_changed_input_never_overwrites_success(tmp_path, monkeypatch, changed):
    root = tmp_path / "output"
    assert invoke(monkeypatch, root) == 0
    out = scene_dir(root)
    before = snapshot(out)
    kwargs = {}
    if changed == "catalog":
        catalog = load_catalog(CATALOG).model_copy(update={"entries": ()})
        path = tmp_path / "changed_catalog.json"
        path.write_text(json.dumps(catalog.canonical_dict()))
        kwargs["catalog"] = path
    elif changed == "seed":
        kwargs["seed"] = 5
    elif changed == "evidence":
        monkeypatch.setattr(cli, "parse_evidence", lambda *a: {"model": "changed"})
    else:
        kwargs["extra"] = ("--generate-missing-assets",)
    assert invoke(monkeypatch, root, **kwargs) == 2
    assert snapshot(out) == before


@pytest.mark.parametrize("name", ["objects/can_1.json", "relations.json", "asset_resolution.json"])
def test_record_tampering_fails_verification_and_reuse(tmp_path, monkeypatch, name):
    assert invoke(monkeypatch, tmp_path) == 0
    out = scene_dir(tmp_path)
    (out / name).write_text("{}\n")
    assert verify_package(out)["status"] == "fail"
    before = snapshot(out)
    assert invoke(monkeypatch, tmp_path) == 2
    assert snapshot(out) == before


def test_existing_failure_directory_is_not_reused(tmp_path, monkeypatch):
    catalog = tmp_path / "missing.json"
    root = tmp_path / "output"
    assert invoke(monkeypatch, root, catalog=catalog) == 2
    out = scene_dir(root)
    before = snapshot(out)
    assert invoke(monkeypatch, root) == 2
    assert snapshot(out) == before


def test_binding_mismatch_stops_packaging(tmp_path, monkeypatch):
    original = cli.solve_scene

    def wrong(spec, catalog):
        resolved = original(spec, catalog)
        first = resolved.objects[0].model_copy(update={"asset_id": "wrong"})
        return resolved.model_copy(update={"objects": (first, *resolved.objects[1:])})

    monkeypatch.setattr(cli, "solve_scene", wrong)
    assert invoke(monkeypatch, tmp_path) == 2
    assert not (scene_dir(tmp_path) / "package_manifest.json").exists()


def test_explicit_legacy_generation_reports_effective_catalog_and_reuses(tmp_path, monkeypatch):
    catalog = AssetCatalog(
        robotwin_root=str(tmp_path / "assets_root"),
        objects_root=str(tmp_path / "assets_root" / "objects"),
        entries=(),
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog.canonical_dict()))
    prompt = "Place a purple hexagonal pedestal on the table."
    root = tmp_path / "output"
    kwargs = dict(prompt=prompt, catalog=path, extra=("--generate-missing-assets",))
    assert invoke(monkeypatch, root, **kwargs) == 0
    out = scene_dir(root, prompt)
    report = read(out / "asset_resolution.json")
    assert report["status"] == "matched"
    assert report["input_asset_catalog_sha256"] == catalog.digest()
    assert report["asset_catalog_sha256"] != catalog.digest()
    assert verify_package(out)["status"] == "pass"
    resolved = ResolvedSceneSpec.model_validate(read(out / "resolved_scene.json"))
    assert resolved.objects[0].asset_provenance == "procedural_generated"
    monkeypatch.setattr(cli, "ensure_assets_for_scene", lambda *a, **k: pytest.fail("regenerated"))
    assert invoke(monkeypatch, root, **kwargs) == 0


@pytest.mark.parametrize("stage", ["solver", "catalog", "success"])
def test_llm_evidence_is_saved_before_downstream_failure(tmp_path, monkeypatch, stage):
    from scene_gen import llm_provider

    responses = [
        {
            "objects": [
                {"object_id": "can_1", "category": "can"},
                {"object_id": "plate_1", "category": "plate"},
            ],
            "ambiguities": [],
        },
        {
            "topology": [
                {"relation": "on_top_of", "source": "can_1", "target": "plate_1"},
                {"relation": "on_table", "source": "plate_1", "target": "table"},
            ],
            "lateral": [],
            "ambiguities": [],
        },
    ]
    calls = []

    def transport(*args):
        calls.append(args)
        return json.dumps(responses.pop(0))

    provider = llm_provider.LLMSceneProvider(
        llm_provider.LLMProviderConfig(
            endpoint="https://llm.invalid/v1",
            model="object-record-test",
            api_key="test-only-key",
            cache_dir=tmp_path / "cache",
        ),
        transport=transport,
    )
    monkeypatch.setattr(llm_provider, "LLMSceneProvider", lambda **kwargs: provider)
    root = tmp_path / "output"
    catalog = CATALOG
    if stage == "solver":

        def fail(*args):
            raise SceneSolveError({"status": "fail", "blocker": "cannot place"})

        monkeypatch.setattr(cli, "solve_scene", fail)
    elif stage == "catalog":
        catalog = tmp_path / "absent.json"
    code = invoke(monkeypatch, root, catalog=catalog, extra=("--provider", "llm"))
    assert code == (0 if stage == "success" else 2)
    out = scene_dir(root)
    assert read(out / "llm_parse_evidence.json")["status"] == "pass"
    assert (out / "objects" / "can_1.json").is_file()
    assert len(calls) == 2
    if stage == "success":
        assert invoke(monkeypatch, root, extra=("--provider", "llm")) == 0
        assert len(calls) == 2
    else:
        assert not (out / "resolved_scene.json").exists()


def test_rule_path_never_constructs_llm(tmp_path, monkeypatch):
    from scene_gen import llm_provider

    monkeypatch.setattr(
        llm_provider, "LLMSceneProvider", lambda *a, **k: pytest.fail("rule path constructed LLM")
    )
    assert invoke(monkeypatch, tmp_path) == 0


def test_parse_failure_has_no_object_exports(tmp_path, monkeypatch):
    assert invoke(monkeypatch, tmp_path, prompt="Place a can at (1, 2, 3).") == 2
    assert not list(tmp_path.glob("*/objects"))
    assert not list(tmp_path.glob("*/scene_spec.json"))


def test_cli_defaults_to_unified_output_root(tmp_path, monkeypatch):
    assert cli.DEFAULT_OUTPUT_ROOT.name == "output"
    root = tmp_path / "output"
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_ROOT", root)
    monkeypatch.setattr(
        sys, "argv", ["generate_scene.py", "--prompt", PROMPT, "--asset-catalog", str(CATALOG)]
    )
    assert cli.main() == 0
    out = scene_dir(root)
    assert (out / "objects" / "can_1.json").is_file()
    assert (out / "asset_resolution.json").is_file()


@pytest.mark.parametrize("attack", ["interrupted", "failed", "extra_object"])
def test_incomplete_or_mixed_output_cannot_be_reused(tmp_path, monkeypatch, attack):
    assert invoke(monkeypatch, tmp_path) == 0
    out = scene_dir(tmp_path)
    if attack == "interrupted":
        (out / "validation_report.json").unlink()
    elif attack == "failed":
        (out / "failure_report.json").write_text('{"status":"fail"}')
    else:
        (out / "objects" / "stale_object.json").write_text("{}")
    before = snapshot(out)
    assert invoke(monkeypatch, tmp_path) == 2
    assert snapshot(out) == before

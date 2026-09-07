"""Offline official-index tests: no downloads, credentials, or simulator required."""

# ruff: noqa: E402
import ast
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_official_index as index


def fixture_asset(asset_id, source):
    source.mkdir(parents=True)
    (source / "visual").mkdir()
    (source / "collision").mkdir()
    (source / "visual" / "model.obj").write_text(
        "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (source / "visual" / "material.mtl").write_text("newmtl test\nmap_Kd texture.png\n")
    Image.new("RGB", (8, 8), "red").save(source / "visual" / "texture.png")
    (source / "collision" / "part.obj").write_text("v 0 0 0\n")
    visuals = 10 if asset_id == "donut_0" else 1
    geoms = '<geom group="1" mesh="visual"/>' * visuals
    geoms += '<geom group="0" mesh="collision"/>' * 32
    (source / "model.xml").write_text(
        '<mujoco><asset><mesh name="visual" file="visual/model.obj" '
        'scale=".1 .2 .3" refquat="0 0 0 1"/>'
        '<mesh name="collision" file="collision/part.obj"/></asset>'
        f'<worldbody><body>{geoms}</body></worldbody></mujoco>')


def fake_preview(record, output):
    saved = json.loads((output / "assets" / f"{record['asset_id']}.json").read_text())
    assert saved["status"] == "structure_checked"
    destination = output / "previews" / record["asset_id"]
    destination.mkdir(parents=True)
    views = []
    for name in ("az000", "az090", "az180", "az270", "top", "bottom"):
        path = destination / f"view_{name}.png"
        Image.new("RGB", (512, 512), "red").save(path)
        views.append(dict(image=index.fingerprint(path, output)))
    index.make_sheet([(str(i), output / view["image"]["path"])
                      for i, view in enumerate(views)], destination / "contact_sheet.png", 3)
    result = dict(status="passed", views=views, geometry=dict(loaded_bounds_m=[[0]*3, [1]*3]),
                  source_files=record["source_files"], physics_steps=0, mode=index.MODE,
                  contact_sheet=index.fingerprint(destination / "contact_sheet.png", output))
    index.write_json(destination / "preview_result.json", result)
    return result


def test_pinned_scope():
    assert index.ASSETS == ("mug_1", "cup_2", "apple_15", "donut_0")
    assert index.REPOSITORY == "Genesis-Intelligence/assets"
    assert index.REVISION == "4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc"


@pytest.mark.parametrize("asset_id,visuals", [("mug_1", 1), ("donut_0", 10)])
def test_inspect_preserves_parts_transforms_and_bytes(tmp_path, asset_id, visuals):
    source = tmp_path / "sources" / asset_id
    fixture_asset(asset_id, source)
    before = (source / "model.xml").read_bytes()
    record = index.inspect_asset(source, tmp_path, asset_id)
    assert record["visual_parts"] == visuals
    assert record["collision_parts"] == 32
    assert record["entrypoint"].endswith("/model.xml")
    assert (source / "model.xml").read_bytes() == before
    assert len(record["source_files"]) == 5
    assert len(record["dependencies"]) == 4
    assert not {"category", "colors", "materials", "interior_bounds"} & record.keys()
    index.verify_files(tmp_path, record["source_files"])


@pytest.mark.parametrize("bad", ["../outside.obj", "/tmp/outside.obj", "..\\outside.obj"])
def test_reject_escaping_dependencies(tmp_path, bad):
    source = tmp_path / "mug_1"
    fixture_asset("mug_1", source)
    xml = source / "model.xml"
    xml.write_text(xml.read_text().replace("visual/model.obj", bad))
    with pytest.raises(ValueError, match="unsafe"):
        index.inspect_asset(source, tmp_path, "mug_1")


def test_missing_texture(tmp_path):
    source = tmp_path / "mug_1"
    fixture_asset("mug_1", source)
    (source / "visual" / "texture.png").unlink()
    with pytest.raises(ValueError, match="missing dependency"):
        index.inspect_asset(source, tmp_path, "mug_1")


def test_symlink_rejected(tmp_path):
    source = tmp_path / "mug_1"
    fixture_asset("mug_1", source)
    (source / "extra.obj").symlink_to(source / "visual" / "model.obj")
    with pytest.raises(ValueError, match="symlink"):
        index.inspect_asset(source, tmp_path, "mug_1")


def test_source_tampering(tmp_path):
    source = tmp_path / "mug_1"
    fixture_asset("mug_1", source)
    record = index.inspect_asset(source, tmp_path, "mug_1")
    (source / "visual" / "model.obj").write_text("tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        index.verify_files(tmp_path, record["source_files"])


def test_camera_fits_all_corners():
    import itertools
    box = np.array([[-.1, -.2, -.3], [.2, .3, .4]])
    corners = np.array(list(itertools.product(*zip(*box))))
    views = index.camera_views(box)
    assert len(views) == 6
    for view in views:
        pos, target = np.array(view["pos"]), np.array(view["lookat"])
        forward = (target - pos) / np.linalg.norm(target - pos)
        right = np.cross(forward, view["up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = corners - pos
        depth = relative @ forward
        assert (depth > view["near"]).all() and (depth < view["far"]).all()
        half_width = depth * np.tan(np.deg2rad(view["fov"] / 2))
        assert (np.abs(relative @ right) < half_width * .95).all()
        assert (np.abs(relative @ up) < half_width * .95).all()


@pytest.mark.parametrize("bad", ["empty", "border", "blank", "nan", "wrong_size"])
def test_bad_previews_rejected(bad):
    rgb = np.full((512, 512, 3), 255., dtype=float)
    rgb[100:400, 100:400] = 100
    seg = np.full((512, 512), -1)
    seg[100:400, 100:400] = 3
    if bad == "empty":
        seg[:] = -1
    elif bad == "border":
        seg[0, 100] = 3
    elif bad == "blank":
        rgb[:] = 255
    elif bad == "nan":
        rgb[150, 150] = np.nan
    elif bad == "wrong_size":
        rgb = rgb[:256]
    with pytest.raises(ValueError):
        index.check_visibility(rgb, seg, 3)


def test_symmetric_views_allowed():
    rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
    rgb[100:400, 100:400] = 100
    seg = np.full((512, 512), -1)
    seg[100:400, 100:400] = 3
    assert index.check_visibility(rgb, seg, 3)["pixels"] == 90000
    assert index.check_visibility(rgb, seg, 3)["pixels"] == 90000


def test_geometry_does_not_accept_only_matching_outer_bounds():
    points = np.array([[0., 0, 0], [1, 1, 1], [.5, .5, .5]])
    with pytest.raises(ValueError, match="vertices disagree"):
        index.compare_geometry([points], [points[:2]])


def test_geometry_alignment_and_missing_parts():
    points = np.array([[0., 0, 0], [1, 1, 1]])
    assert index.compare_geometry([points], [points])["max_bounds_error_m"] == 0
    with pytest.raises(ValueError, match="parts mismatch"):
        index.compare_geometry([points, points], [points])
    with pytest.raises(ValueError, match="bounds disagree"):
        index.compare_geometry([points], [points + 2e-6])


def test_visual_classification_ignores_collision_duplicates():
    visual = SimpleNamespace(metadata={"mesh_path": "visual/model.obj"})
    collision = SimpleNamespace(metadata={"mesh_path": "collision/part.obj"})
    entity = SimpleNamespace(links=[SimpleNamespace(vgeoms=[visual, collision])])
    record = dict(visual_meshes=["visual/model.obj"], collision_meshes=["collision/part.obj"])
    assert index.visual_geometries(entity, record) == [visual]
    collision.metadata["mesh_path"] = "unknown.obj"
    with pytest.raises(ValueError, match="unrecognized"):
        index.visual_geometries(entity, record)


def test_full_build_and_relative_hash_manifest(tmp_path):
    output = tmp_path / "new"
    report = index.build(output, downloader=fixture_asset, renderer=fake_preview)
    assert report["status"] == "passed"
    assert (output / "overview.png").exists()
    manifest = index.verify_index(output)
    assert all(not Path(f["path"]).is_absolute() for f in manifest["files"])
    assert len(list(output.glob("previews/*/view_*.png"))) == 24
    with pytest.raises(ValueError, match="must be new"):
        index.build(output)


def test_verify_relative_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index.build(Path("relative"), downloader=fixture_asset, renderer=fake_preview)
    assert index.verify_index("relative")["status"] == "passed"
    assert index.main(["--output-dir", "relative", "--verify-only"]) == 0


def test_late_source_tampering_does_not_create_success(tmp_path):
    def render(record, output):
        result = fake_preview(record, output)
        if record["asset_id"] == "donut_0":
            (output / "sources/mug_1/visual/model.obj").write_text("tampered late")
        return result

    output = tmp_path / "new"
    report = index.build(output, downloader=fixture_asset, renderer=render)
    assert report["status"] == "failed"
    assert report["assets"][0]["status"] == "failed"
    assert not (output / "overview.png").exists()
    assert index.verify_index(output)["status"] == "failed"


@pytest.mark.parametrize("target", ["assets/mug_1.json", "previews/mug_1/view_top.png"])
def test_output_tamper_detected(tmp_path, target):
    output = tmp_path / "new"
    index.build(output, downloader=fixture_asset, renderer=fake_preview)
    (output / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        index.verify_index(output)


@pytest.mark.parametrize("stage", ["download", "preview"])
def test_failure_continues_and_retains_records(tmp_path, stage):
    calls = []

    def download(asset_id, source):
        calls.append(asset_id)
        if stage == "download" and asset_id == "mug_1":
            raise OSError("test network failure")
        fixture_asset(asset_id, source)

    def render(record, output):
        if stage == "preview" and record["asset_id"] == "mug_1":
            raise ValueError("test renderer failure")
        return fake_preview(record, output)

    output = tmp_path / "new"
    assert index.build(output, downloader=download, renderer=render)["status"] == "failed"
    assert tuple(calls) == index.ASSETS
    assert not (output / "overview.png").exists()
    assert len(list((output / "assets").glob("*.json"))) == 4
    record = json.loads((output / "assets/mug_1.json").read_text())
    assert record["failure_stage"] == stage
    if stage == "preview":
        assert record["source_files"] and record["entrypoint"].endswith("model.xml")
    index.verify_index(output)


def test_no_step_reset_converter_or_semantic_dependency():
    tree = ast.parse(inspect.getsource(index))
    forbidden = {"step", "mj_step", "set_pos", "set_quat", "convert_mug"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]
    for name in ("scene_gen", "agenticsim", "validate_physics", "prepare_cases"):
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module and n.module.startswith(name)]


def test_help(capsys):
    with pytest.raises(SystemExit) as result:
        index.main(["--help"])
    assert result.value.code == 0
    assert "--output-dir" in capsys.readouterr().out

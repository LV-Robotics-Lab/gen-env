"""Index four pinned official assets and render previews, never physical evidence.

This is not scene_gen's semantic catalog. No conversion, captioning or retrieval.
Genesis and MuJoCo are imported only by the reference/preview operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shlex
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis.storage_paths import evidence_path, local_path

ASSETS = ("mug_1", "cup_2", "apple_15", "donut_0")
REPOSITORY = "Genesis-Intelligence/assets"
REVISION = "4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc"
GENESIS_COMMIT = "0e74bf392781884ccad765c3f344419c86b872ca"
ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "genenv.genesis_asset_index.v1"
MODE = "asset_preview_zero_physics_steps"
SIZE = 512
TOLERANCE = 1e-6
LIGHTS = [
    dict(type="directional", dir=(-1, -1, -1), color=(1, 1, 1), intensity=3.0),
    dict(type="directional", dir=(1, 1, 1), color=(1, 1, 1), intensity=2.0),
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path, root):
    path = Path(evidence_path(Path(path).absolute()))
    root = Path(evidence_path(Path(root).absolute()))
    safe_file(root, str(path.relative_to(root)))
    return dict(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def safe_file(root, relative):
    """Reject escaping paths and symlinks, including links pointing back inside."""
    root = local_path(root).resolve()
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError(f"unsafe dependency path: {relative}")
    path = root.joinpath(*rel.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"symlink dependency: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing dependency: {relative}")
    return path


def verify_files(root, records):
    for record in records:
        if fingerprint(safe_file(root, record["path"]), root) != record:
            raise ValueError(f"file integrity mismatch: {record['path']}")


def download_asset(asset_id, destination):
    """Only the allowlisted subtree; authenticate bytes against pinned Hub metadata."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    if asset_id not in ASSETS:
        raise ValueError("asset not in official preview allowlist")
    destination.mkdir(parents=True, exist_ok=False)
    entries = list(HfApi().list_repo_tree(REPOSITORY, path_in_repo=asset_id,
                                        recursive=True, repo_type="dataset", revision=REVISION))

    def fetch(entry):
        if not isinstance(entry, RepoFile):
            return
        relative = PurePosixPath(entry.path).relative_to(asset_id)
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("unsafe remote path")
        path = destination.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{entry.path}"
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if len(data) != entry.size:
            raise ValueError(f"remote size mismatch: {entry.path}")
        if entry.lfs:
            actual = hashlib.sha256(data).hexdigest()
            expected = entry.lfs.sha256
        else:
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            expected = entry.blob_id
        if actual != expected:
            raise ValueError(f"remote digest mismatch: {entry.path}")
        with path.open("xb") as stream:
            stream.write(data)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, entries))
    if not (destination / "model.xml").is_file():
        raise ValueError("official model.xml absent")


def inspect_asset(source, output, asset_id):
    """Audit this small official template; do not guess general MJCF semantics."""
    source, output = Path(source), Path(output)
    dependencies = set()
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not accepted in official sources")
        if not path.is_file():
            continue
        refs = []
        if path.suffix == ".xml":
            content = path.read_text()
            if "<!DOCTYPE" in content or "<!ENTITY" in content:
                raise ValueError("XML entities are not supported")
            root = ET.fromstring(content)
            if root.tag != "mujoco" or root.findall(".//include"):
                raise ValueError("unsupported official XML structure")
            if any(n.attrib.get(k) for n in root.findall("compiler")
                   for k in ("assetdir", "meshdir", "texturedir")):
                raise ValueError("directory overrides outside the supported template")
            refs = [n.attrib["file"] for n in root.iter() if "file" in n.attrib]
        elif path.suffix.lower() in (".obj", ".mtl"):
            for line in path.read_text().splitlines():
                words = shlex.split(line, comments=True)
                if not words:
                    continue
                if words[0] == "mtllib":
                    refs.extend(words[1:])
                elif words[0].lower().startswith("map_") or words[0].lower() in (
                    "bump", "disp", "decal", "norm", "refl"
                ):
                    if len(words) != 2 or words[1].startswith("-"):
                        raise ValueError("unsupported MTL texture options")
                    refs.append(words[1])
        for ref in refs:
            # Check ref before prefixing: joining an absolute path discards its base.
            if PurePosixPath(ref).is_absolute() or ".." in PurePosixPath(ref).parts:
                raise ValueError(f"unsafe dependency path: {ref}")
            relative = (path.parent.relative_to(source) / ref).as_posix()
            dependencies.add(safe_file(source, relative).relative_to(output).as_posix())
    xml = safe_file(source, "model.xml")
    root = ET.parse(xml).getroot()
    geoms = root.findall(".//geom")
    visuals = [n for n in geoms if n.get("group") == "1"]
    collisions = [n for n in geoms if n.get("group") == "0"]
    expected_visuals = 10 if asset_id == "donut_0" else 1
    if root.findall(".//joint") or len(visuals) != expected_visuals or len(collisions) != 32:
        raise ValueError("model does not match the pinned official rigid template")
    if len(geoms) != len(visuals) + len(collisions):
        raise ValueError("unclassified geometry")
    meshes = {n.attrib["name"]: n.attrib["file"] for n in root.findall("./asset/mesh")}
    visual_meshes = [meshes[n.attrib["mesh"]] for n in visuals]
    collision_meshes = [meshes[n.attrib["mesh"]] for n in collisions]
    if set(visual_meshes) & set(collision_meshes):
        raise ValueError("visual and collision meshes must be independent")
    records = [fingerprint(p, output) for p in sorted(source.rglob("*")) if p.is_file()]
    return dict(asset_id=asset_id, status="structure_checked", source=dict(
        repository=REPOSITORY, revision=REVISION, byte_verification="pinned_hub_file_digest"),
        entrypoint=xml.relative_to(output).as_posix(), dependencies=sorted(dependencies),
        source_files=records, visual_parts=len(visuals), collision_parts=len(collisions),
        visual_meshes=visual_meshes, collision_meshes=collision_meshes,
        preview_status="not_started", physics_status="not_evaluated")


def visual_geometries(entity, record):
    """MJCF adds transparent collision vgeoms; classify by source mesh, not mesh.color."""
    visuals = []
    for link in entity.links:
        for geom in link.vgeoms:
            source = geom.metadata.get("mesh_path")
            if source in record["visual_meshes"]:
                visuals.append(geom)
            elif source not in record["collision_meshes"]:
                raise ValueError(f"unrecognized loaded visual mesh: {source}")
    return visuals


def reference_geometry(xml):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # Geometry resolution only; never mj_step.
    if model.njnt or len(set(model.geom_bodyid.tolist())) != 1:
        raise ValueError("expected one rigid body without joints")
    parts = []
    for i in range(model.ngeom):
        if model.geom_group[i] != 1:
            continue
        mesh = int(model.geom_dataid[i])
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("only official mesh visual geometry is supported")
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start + count] @ data.geom_xmat[i].reshape(3, 3).T
        points = points + data.geom_xpos[i]
        parts.append(points)
    return parts


def bounds(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("missing or nonfinite geometry")
    return np.array([points.min(axis=0), points.max(axis=0)])


def compare_geometry(expected, actual):
    if len(expected) != len(actual):
        raise ValueError(f"visual parts mismatch: expected {len(expected)}, got {len(actual)}")
    reference = bounds(np.concatenate(expected))
    loaded = bounds(np.concatenate(actual))
    error = float(np.max(np.abs(reference - loaded)))
    if error > TOLERANCE:
        raise ValueError(f"visual bounds disagree with MJCF: {error} m")
    # Also compare all vertices: matching outer boxes alone can hide lost details.
    from scipy.spatial import cKDTree
    reference_points, actual_points = np.concatenate(expected), np.concatenate(actual)
    vertex_error = float(max(cKDTree(reference_points).query(actual_points)[0].max(),
                             cKDTree(actual_points).query(reference_points)[0].max()))
    if vertex_error > TOLERANCE:
        raise ValueError(f"visual vertices disagree with MJCF: {vertex_error} m")
    return dict(reference_bounds_m=reference.tolist(), loaded_bounds_m=loaded.tolist(),
                dimensions_m=(loaded[1] - loaded[0]).tolist(), max_bounds_error_m=error,
                max_vertex_error_m=vertex_error, tolerance_m=TOLERANCE)


def camera_views(box):
    box = np.asarray(box, dtype=float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or (box[1] <= box[0]).any():
        raise ValueError("invalid visual bounds")
    center = box.mean(axis=0)
    radius = float(np.linalg.norm(box[1] - box[0]) / 2)
    distance = radius / math.sin(math.radians(35 / 2)) * 1.2
    directions = [(math.cos(a), math.sin(a), math.tan(math.radians(30)))
                  for a in np.arange(4) * math.pi / 2]
    directions.extend([(0, 0, 1), (0, 0, -1)])
    names = ["az000", "az090", "az180", "az270", "top", "bottom"]
    return [dict(name=name, pos=(center + distance * np.array(d) / np.linalg.norm(d)).tolist(),
                 lookat=center.tolist(), up=[0, 1, 0] if i >= 4 else [0, 0, 1],
                 fov=35, near=max(radius * .01, 1e-5), far=distance + radius * 3)
            for i, (name, d) in enumerate(zip(names, directions))]


def check_visibility(rgb, segmentation, entity_id):
    rgb, segmentation = np.asarray(rgb), np.asarray(segmentation).squeeze()
    if rgb.shape != (SIZE, SIZE, 3) or segmentation.shape != (SIZE, SIZE):
        raise ValueError("unexpected rendered image dimensions")
    if not np.isfinite(rgb).all() or not np.isfinite(segmentation).all():
        raise ValueError("nonfinite render")
    mask = segmentation == entity_id
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty object segmentation")
    if min(x.min(), y.min(), SIZE - 1 - x.max(), SIZE - 1 - y.max()) < 4:
        raise ValueError("object touches preview border")
    if np.ptp(rgb.astype(float)) == 0:
        raise ValueError("blank RGB image")
    return dict(pixels=int(len(x)), bbox_xyxy=[int(x.min()), int(y.min()),
                                            int(x.max()), int(y.max())])


def make_sheet(items, destination, columns):
    sheet = Image.new("RGB", (SIZE * columns, (SIZE + 24) * math.ceil(len(items) / columns)),
                      "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(items):
        x, y = (i % columns) * SIZE, (i // columns) * (SIZE + 24)
        with Image.open(path) as picture:
            sheet.paste(picture.convert("RGB"), (x, y + 24))
        draw.text((x + 8, y + 6), label, fill="black")
    sheet.save(destination)


def render_preview(record, output):
    import genesis as gs

    checkout = Path(gs.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != GENESIS_COMMIT:
        raise ValueError(f"Genesis revision mismatch: {commit}")
    verify_files(output, record["source_files"])
    source = safe_file(output, record["entrypoint"])
    reference = reference_geometry(source)
    views = camera_views(bounds(np.concatenate(reference)))
    destination = output / "previews" / record["asset_id"]
    destination.mkdir(parents=True, exist_ok=False)
    evidence = dict(mode=MODE, status="running", physics_steps=0, physics_status="not_evaluated",
                    genesis_commit=commit, backend="cpu", seed=0, renderer="Rasterizer",
                    resolution=[SIZE, SIZE], background=[1, 1, 1], lights=LIGHTS,
                    ambient_light=[.4, .4, .4], source_files=record["source_files"], views=[])
    write_json(destination / "preview_result.json", evidence)
    initialized = False
    try:
        gs.init(backend=gs.cpu, seed=0, logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer(),
                         vis_options=gs.options.VisOptions(background_color=(1, 1, 1),
                             ambient_light=(.4, .4, .4), lights=LIGHTS, shadow=False,
                             segmentation_level="entity"))
        entity = scene.add_entity(gs.morphs.MJCF(file=str(source), scale=1.0,
                                  convexify=False, decimate=False, watertighten=None),
                                  material=gs.materials.Rigid(), vis_mode="visual")
        camera = scene.add_camera(res=(SIZE, SIZE), GUI=False,
                                  **{k: v for k, v in views[0].items() if k != "name"})
        scene.build()
        actual = [g.get_vverts().detach().cpu().numpy().reshape(-1, 3)
                  for g in visual_geometries(entity, record)]
        evidence["geometry"] = compare_geometry(reference, actual)
        evidence["loaded_visual_parts"] = len(actual)
        if entity.n_geoms != record["collision_parts"]:
            raise ValueError("loaded collision part count mismatch")
        evidence["loaded_collision_parts"] = entity.n_geoms
        segmentation_id = next((int(i) for i, key in
                                scene.visualizer.segmentation_idx_dict.items()
                                if key == int(entity.idx)), None)
        if segmentation_id is None:
            raise ValueError("entity missing from segmentation mapping")
        sheets = []
        for view in views:
            camera.set_pose(pos=view["pos"], lookat=view["lookat"], up=view["up"])
            rgb, _, segmentation, _ = camera.render(rgb=True, segmentation=True)
            visibility = check_visibility(rgb, segmentation, segmentation_id)
            path = destination / f"view_{view['name']}.png"
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
            with Image.open(path) as decoded:
                decoded.load()
                if decoded.size != (SIZE, SIZE):
                    raise ValueError("PNG decode dimensions mismatch")
            evidence["views"].append(dict(camera=view, visibility=visibility,
                                          image=fingerprint(path, output)))
            sheets.append((view["name"], path))
        make_sheet(sheets, destination / "contact_sheet.png", 3)
        evidence["contact_sheet"] = fingerprint(destination / "contact_sheet.png", output)
        verify_files(output, record["source_files"])
        evidence["status"] = "passed"
    except Exception as exc:
        evidence.update(status="failed", error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        write_json(destination / "preview_result.json", evidence)
        if initialized:
            gs.destroy()
    return evidence


def verify_index(output):
    output = Path(output).absolute()
    index = json.loads((output / "asset_index.json").read_text())
    if index["schema_version"] != SCHEMA:
        raise ValueError("wrong official index schema")
    if index["source_repository"] != REPOSITORY or index["source_revision"] != REVISION:
        raise ValueError("wrong official source revision")
    verify_files(output, index["files"])
    expected = {r["path"] for r in index["files"]}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != expected | {"asset_index.json"}:
        raise ValueError("index file set mismatch")
    if tuple(item["asset_id"] for item in index["assets"]) != ASSETS:
        raise ValueError("index asset set mismatch")
    for item in index["assets"]:
        record_path = f"assets/{item['asset_id']}.json"
        if item["record"] != record_path or record_path not in expected:
            raise ValueError("asset record missing from manifest")
        record = json.loads(safe_file(output, record_path).read_text())
        if record["asset_id"] != item["asset_id"] or record["status"] != item["status"]:
            raise ValueError("asset record binding mismatch")
        if record["status"] == "preview_passed":
            verify_files(output, record["source_files"])
            preview_path = record["preview_result"]
            if preview_path not in expected:
                raise ValueError("preview record missing from manifest")
            preview = json.loads(safe_file(output, preview_path).read_text())
            if (preview["status"] != "passed" or len(preview["views"]) != 6
                    or preview["source_files"] != record["source_files"]
                    or preview["geometry"] != record["geometry"]
                    or preview["physics_steps"] != 0 or preview["mode"] != MODE):
                raise ValueError("preview/source binding mismatch")
            verify_files(output, [v["image"] for v in preview["views"]])
            verify_files(output, [preview["contact_sheet"]])
    all_passed = all(a["status"] == "preview_passed" for a in index["assets"])
    if (index["status"] == "passed") != all_passed:
        raise ValueError("inconsistent index success status")
    if all_passed and "overview.png" not in expected:
        raise ValueError("overview missing from manifest")
    return index


def build(output, *, downloader=download_asset, renderer=render_preview):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory must be new; choose another --output-dir")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version="genenv.genesis_asset_build.v1", mode=MODE,
                  status="running", physics_status="not_evaluated", assets=[])
    write_json(output / "build_report.json", report)
    overview = []
    for asset_id in ASSETS:
        record = dict(asset_id=asset_id, status="failed", preview_status="not_started",
                      physics_status="not_evaluated", source=dict(repository=REPOSITORY,
                                                                 revision=REVISION))
        stage = "download"
        print(f"[{asset_id}] downloading pinned official source", flush=True)
        try:
            source = output / "sources" / asset_id
            downloader(asset_id, source)
            stage = "structure"
            record = inspect_asset(source, output, asset_id)
            write_json(output / "assets" / f"{asset_id}.json", record)
            stage = "preview"
            print(f"[{asset_id}] structure checked; rendering six views", flush=True)
            preview = renderer(record, output)
            if preview["status"] != "passed" or len(preview["views"]) != 6:
                raise ValueError("preview did not complete all six views")
            verify_files(output, record["source_files"])
            record.update(status="preview_passed", preview_status="passed",
                          geometry=preview["geometry"],
                          preview_result=f"previews/{asset_id}/preview_result.json")
            overview.append((asset_id, output / preview["views"][0]["image"]["path"]))
        except Exception as exc:
            record.update(status="failed", failure_stage=stage, error_type=type(exc).__name__,
                          reason=str(exc), preview_status="failed" if stage == "preview"
                          else "not_started")
            print(f"[{asset_id}] {stage} failed: {exc}", file=sys.stderr, flush=True)
        write_json(output / "assets" / f"{asset_id}.json", record)
        report["assets"].append(dict(asset_id=asset_id, status=record["status"],
                                     record=f"assets/{asset_id}.json",
                                     reason=record.get("reason")))
        write_json(output / "build_report.json", report)
    report["status"] = "passed" if len(overview) == len(ASSETS) else "failed"
    # Recheck early assets after later renders, before recording overall success.
    for item in report["assets"]:
        if item["status"] != "preview_passed":
            continue
        record_path = output / item["record"]
        record = json.loads(record_path.read_text())
        try:
            verify_files(output, record["source_files"])
            preview = json.loads(safe_file(output, record["preview_result"]).read_text())
            verify_files(output, [v["image"] for v in preview["views"]])
            verify_files(output, [preview["contact_sheet"]])
        except Exception as exc:
            record.update(status="failed", preview_status="failed", failure_stage="integrity",
                          error_type=type(exc).__name__, reason=str(exc))
            write_json(record_path, record)
            item.update(status="failed", reason=str(exc))
            report["status"] = "failed"
    if report["status"] == "passed":
        make_sheet(overview, output / "overview.png", 2)
    write_json(output / "build_report.json", report)
    index = dict(schema_version=SCHEMA, status=report["status"], mode=MODE,
                 source_repository=REPOSITORY, source_revision=REVISION,
                 assets=report["assets"], files=[fingerprint(p, output)
                 for p in sorted(output.rglob("*")) if p.is_file()])
    write_json(output / "asset_index.json", index)
    verify_index(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory, e.g. assets/genesis/v1")
    parser.add_argument("--verify-only", action="store_true",
                        help="Read-only verification of an existing index and all linked files")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            index = verify_index(args.output_dir)
            print(json.dumps(dict(integrity="passed", build_status=index["status"])))
            return 0
        report = build(args.output_dir)
        print(json.dumps(dict(status=report["status"], output_dir=str(args.output_dir))))
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Import SimFoundry rigid assets as portable, source-bound Genesis URDF packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import standard_urdf as standard


def import_assets(scene_dir, output_dir, only=()):
    scene, out = Path(scene_dir).resolve(), Path(output_dir).resolve()
    if out.is_relative_to(scene) or scene.is_relative_to(out):
        raise ValueError("source and output must be separate")
    metadata = scene / "s11_sim/scene_objects_info.json"
    objects = library.read_json(metadata)
    if only and set(only) - {v["name"] for v in objects.values()}:
        raise ValueError("unknown requested object ID")
    out.mkdir(parents=True, exist_ok=False)
    inventory = dict(
        schema_version=standard.SCHEMA,
        source_metadata={"path": str(metadata), "sha256": library.sha256(metadata)},
        assets=[],
    )
    for item in objects.values():
        if only and item["name"] not in only:
            continue
        result = dict(object_id=item["name"], category=item["category"], status="failed")
        try:
            if any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", item[k]) for k in ("category", "model", "name")
            ):
                raise ValueError("unsafe source identifier")
            root = scene / "s11_sim/objects" / item["category"] / item["model"]
            entry = root / "urdf" / (item["model"] + ".urdf")
            records = {
                p.relative_to(root).as_posix(): official.fingerprint(p, root)
                for p in root.rglob("*")
                if p.is_file()
            }
            closure = library.dependencies(entry, root, records)
            source_files = [records[p] for p in closure]
            standard.inspect(entry)
            friction = item["friction"]
            if not isinstance(friction, (float, int)) or not 0 <= friction < float("inf"):
                raise ValueError("invalid source friction")
            identity = hashlib.sha256(
                json.dumps(
                    dict(
                        files=source_files,
                        friction=friction,
                        category=item["category"],
                        conversion="link_inertia_v1",
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            asset_id = f"simfoundry_{item['category']}_{identity[:16]}"
            dest = out / "packages" / asset_id
            dest.mkdir(parents=True)
            mapping = []
            for relative in closure:
                src, target = root / relative, dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                content = src.read_bytes()
                refs = library.direct_references(src)
                if refs:
                    text = content.decode()
                    for ref in refs:
                        resolved = library.local_reference(src, ref, root)
                        new_ref = os.path.relpath(dest / resolved.relative_to(root), target.parent)
                        text = text.replace(ref, new_ref)
                    content = text.encode()
                if src == entry:
                    # Equivalent URDF representation avoids a native parser losing inertial RPY.
                    xml = ET.fromstring(content)
                    inertial = xml.find("link/inertial")
                    authored = standard.inspect(entry)
                    origin = inertial.find("origin")
                    if origin is None:
                        origin = ET.SubElement(inertial, "origin", xyz="0 0 0")
                    origin.set("rpy", "0 0 0")
                    tensor = authored["inertia"]
                    for key, a, b in [
                        ("ixx", 0, 0),
                        ("ixy", 0, 1),
                        ("ixz", 0, 2),
                        ("iyy", 1, 1),
                        ("iyz", 1, 2),
                        ("izz", 2, 2),
                    ]:
                        inertial.find("inertia").set(key, format(tensor[a, b], ".17g"))
                    content = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
                target.write_bytes(content)
                mapping.append(
                    dict(source=records[relative], output=official.fingerprint(target, dest))
                )
            official.write_json(
                dest / "physics.json",
                dict(
                    friction=friction,
                    mass_source="SimFoundry VLM estimate",
                    friction_source="SimFoundry VLM estimate",
                    inertia_source="authored SimFoundry URDF",
                    units="SI",
                ),
            )
            package = dict(
                schema_version="genenv.standard_urdf_asset.v1",
                asset_id=asset_id,
                category=item["category"],
                entrypoint=entry.relative_to(root).as_posix(),
                physics_file="physics.json",
                source=dict(
                    root=str(root), metadata=inventory["source_metadata"], object_id=item["name"]
                ),
                path_mapping=mapping,
                files=[
                    official.fingerprint(p, dest) for p in sorted(dest.rglob("*")) if p.is_file()
                ],
            )
            official.write_json(dest / "asset.json", package)
            _, converted, _ = standard.verify_package(dest / "asset.json")
            a, b = standard.inspect(entry), standard.inspect(converted)
            import numpy as np

            if any(not np.allclose(a[k], b[k], rtol=1e-12, atol=1e-15) for k in a):
                raise ValueError("conversion changed geometry or inertial data")
            official.verify_files(root, source_files)
            result.update(
                status="imported",
                asset_id=asset_id,
                package=(dest / "asset.json").relative_to(out).as_posix(),
                package_sha256=library.sha256(dest / "asset.json"),
            )
        except Exception as exc:
            result.update(error=f"{type(exc).__name__}: {exc}")
        inventory["assets"].append(result)
        official.write_json(out / "library.json", inventory)
    if library.sha256(metadata) != inventory["source_metadata"]["sha256"]:
        raise ValueError("source metadata changed during import")
    return inventory


def previews(library_path, output_dir):
    library_path, out = Path(library_path).resolve(), Path(output_dir).resolve()
    from self_improving.sim_adapters.genesis.clip_select import separate

    separate(out, library_path.parent)
    inventory = library.read_json(library_path)
    if inventory["schema_version"] != standard.SCHEMA:
        raise ValueError("invalid standard library")
    out.mkdir(parents=True, exist_ok=False)
    reference = dict(path=str(library_path), sha256=library.sha256(library_path))
    assets = []
    for item in inventory["assets"]:
        if item["status"] != "imported":
            continue
        package_path = library_path.parent / item["package"]
        pkg, _, _ = standard.verify_package(package_path)
        if library.sha256(package_path) != item["package_sha256"]:
            raise ValueError("package manifest changed")
        aid = item["asset_id"]
        candidate = dict(
            asset_id=aid,
            entrypoint=pkg["entrypoint"],
            format="urdf",
            discovery_error=None,
            dependency_paths=[f["path"] for f in pkg["files"]],
        )
        job = out / "jobs" / f"{aid}.json"
        official.write_json(
            job,
            dict(
                output=str(out),
                source_root=str(package_path.parent),
                candidate=candidate,
                standard_urdf=True,
                source_records={f["path"]: f for f in pkg["files"]},
            ),
        )
        log = out / "logs" / f"{aid}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("w") as stream:
            try:
                child = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "self_improving.sim_adapters.genesis.build_library_previews",
                        "--worker",
                        str(job),
                    ],
                    stdout=stream,
                    stderr=stream,
                    timeout=300,
                    env=dict(os.environ, OMP_NUM_THREADS="2"),
                )
                error = f"worker_exit_{child.returncode}" if child.returncode else None
            except subprocess.TimeoutExpired:
                error = "preview_timeout"
        preview_path = f"previews/{aid}/preview_result.json"
        preview = library.read_json(out / preview_path) if (out / preview_path).exists() else {}
        status = "preview_passed" if not error and preview.get("status") == "passed" else "failed"
        record = dict(
            candidate,
            source_root=str(package_path.parent),
            source_files=pkg["files"],
            source_inventory=reference,
            standard_package=str(package_path),
            standard_package_sha256=item["package_sha256"],
            preview_result=preview_path,
            status=status,
            error=error or preview.get("error"),
        )
        record_path = f"assets/{aid}.json"
        official.write_json(out / record_path, record)
        assets.append(dict(asset_id=aid, status=status, record=record_path))
        print(aid, status, record["error"], flush=True)
    index = dict(
        schema_version=standard.PREVIEW_SCHEMA,
        source_inventory=reference,
        assets=assets,
        files=[official.fingerprint(p, out) for p in sorted(out.rglob("*")) if p.is_file()],
    )
    official.write_json(out / "asset_index.json", index)
    standard.verify_preview_index(out / "asset_index.json")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--scene-dir", required=True, type=Path)
    imp.add_argument("--output-dir", required=True, type=Path)
    imp.add_argument("--only", action="append", default=[])
    pre = sub.add_parser("previews")
    pre.add_argument("--library-path", required=True, type=Path)
    pre.add_argument("--output-dir", required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = import_assets(**args) if command == "import" else previews(**args)
    print(json.dumps(result["assets"], ensure_ascii=False))
    return 2 if any(a["status"] == "failed" for a in result["assets"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

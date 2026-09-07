"""Verified union of existing CLIP libraries; assets and images retain their own roots."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis.storage_paths import local_path

SCHEMA = "genenv.clip_union.v1"


def collect(members):
    from self_improving.sim_adapters.genesis import clip_select as clip

    assets, rows, matrices = [], [], []
    metadata = None
    names = set()
    for member in members:
        if member["namespace"] in names or not member["namespace"]:
            raise ValueError("duplicate/empty union namespace")
        names.add(member["namespace"])
        path = local_path(member["path"]).resolve()
        if clip.file_hash(path) != member["sha256"]:
            raise ValueError("union member changed")
        if clip.strict_json(path.read_text())["schema_version"] != clip.INDEX_SCHEMA:
            raise ValueError("only leaf CLIP indexes can be union members")
        index, vectors = clip.load_index(path)
        signature = (index["encoder"], index["aggregation"], index["shape"][1])
        if metadata is not None and signature != metadata:
            raise ValueError("incompatible union encoders/preprocessing/normalization")
        metadata = signature
        root = local_path(index["official_index"]["path"]).resolve().parent
        for source in index["assets"]:
            asset = copy.deepcopy(source)
            asset.update(
                asset_id=member["namespace"] + ":" + source["asset_id"],
                member_asset_id=source["asset_id"],
                member_index=member,
                preview_root=str(root),
                source_root=source.get("source_root", str(root)),
                model_format=source.get("model_format", Path(source["entrypoint"]).suffix[1:]),
                source_inventory=source.get("source_inventory", index["official_index"]),
            )
            assets.append(asset)
        for source in index["rows"]:
            rows.append(
                dict(
                    source,
                    asset_id=member["namespace"] + ":" + source["asset_id"],
                    preview_root=str(root),
                )
            )
        matrices.append(vectors)
    if not matrices:
        raise ValueError("empty union")
    return assets, rows, np.concatenate(matrices), metadata[0]


def build(clip_index, output_dir):
    from self_improving.sim_adapters.genesis import clip_select as clip

    output = Path(output_dir).resolve()
    clip.writable_storage(output)
    members = []
    for value in clip_index:
        namespace, separator, raw = str(value).partition("=")
        if not separator or not namespace.replace("_", "").isalnum():
            raise ValueError("expected namespace=/path/to/index.json")
        path = local_path(raw).resolve()
        clip.separate(output, path.parent)
        members.append(dict(namespace=namespace, path=str(path), sha256=clip.file_hash(path)))
    assets, rows, vectors, encoder = collect(members)
    output.mkdir(parents=True, exist_ok=False)
    clip.write_json(output / "members.json", members)
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    index = dict(
        schema_version=SCHEMA,
        members=members,
        assets=assets,
        rows=rows,
        encoder=encoder,
        shape=list(vectors.shape),
        aggregation="max_of_six_cosine",
        official_index=dict(
            path=str(output / "members.json"), sha256=clip.file_hash(output / "members.json")
        ),
        vectors=official.fingerprint(output / "vectors.npy", output),
        physics_after_selection=True,
    )
    clip.write_json(output / "index.json", index)
    load(output / "index.json")
    return index


def load(path):
    from self_improving.sim_adapters.genesis import clip_select as clip

    path = Path(path).resolve()
    index = clip.strict_json(path.read_text())
    if index["schema_version"] != SCHEMA or index.get("physics_after_selection") is not True:
        raise ValueError("invalid union schema or disabled validation")
    if (
        clip.file_hash(index["official_index"]["path"]) != index["official_index"]["sha256"]
        or clip.strict_json(Path(index["official_index"]["path"]).read_text()) != index["members"]
    ):
        raise ValueError("union member manifest changed")
    assets, rows, expected, encoder = collect(index["members"])
    if (assets, rows, encoder) != (index["assets"], index["rows"], index["encoder"]):
        raise ValueError("union source binding changed")
    official.verify_files(path.parent, [index["vectors"]])
    vectors = np.load(official.safe_file(path.parent, index["vectors"]["path"]), allow_pickle=False)
    if (
        not np.array_equal(vectors, expected)
        or vectors.dtype != np.float32
        or list(vectors.shape) != index["shape"]
        or index["aggregation"] != "max_of_six_cosine"
    ):
        raise ValueError("union vector binding changed")
    return index, vectors

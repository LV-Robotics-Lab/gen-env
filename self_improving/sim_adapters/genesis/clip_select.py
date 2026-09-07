"""Pinned Chinese-CLIP retrieval and one visual choice, producing only an asset binding."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scene_gen.llm_provider import LLMProviderConfig, load_llm_provider_config
from self_improving.sim_adapters.genesis import asset_library, standard_urdf, union_index
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis.storage_paths import (
    CACHE_ROOT,
    local_path,
    same_evidence_path,
)
from self_improving.sim_adapters.genesis.vision_request import (
    PROMPT,
    PROMPT_VERSION,
    ChatVisionClient,
    SelectionError,
    strict_json,
    validate_selection,
)

MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"
REVISION = "36e679e65c2a2fead755ae21162091293ad37834"
INDEX_SCHEMA = "genenv.clip_index.v1"
SELECTION_VERSION = "genenv.clip_select.v2"
WEIGHTS_DIR = CACHE_ROOT / "model_weights/chinese_clip"
CACHE_DIR = CACHE_ROOT / "asset_selection_cache"
VIEWS = ("az000", "az090", "az180", "az270", "top", "bottom")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(local_path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".selection-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def separate(*paths):
    resolved = [Path(p).resolve() for p in paths]
    for i, left in enumerate(resolved):
        for right in resolved[i + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise SelectionError("storage_directories_overlap")
    # Protect sealed official packages even if a caller supplies a different package.
    for path in resolved:
        if any((p / marker).is_file() for p in path.parents
               for marker in ("asset_index.json", "download_manifest.json")):
            raise SelectionError("storage_inside_official_package")


def is_standard_package(path):
    for filename, schema in (("asset.json", "genenv.standard_urdf_asset.v1"),
                             ("library.json", standard_urdf.SCHEMA)):
        candidate = path / filename
        if candidate.is_file():
            try:
                if strict_json(candidate.read_text()).get("schema_version") == schema:
                    return True
            except (ValueError, OSError):
                continue
    return False


def writable_storage(*paths):
    for path in paths:
        resolved = Path(path).resolve()
        if any(is_standard_package(p) for p in (resolved, *resolved.parents)):
            raise SelectionError("storage_inside_standard_package")
        if any((p / marker).is_file() for p in (resolved, *resolved.parents)
               for marker in ("asset_index.json", "download_manifest.json")):
            raise SelectionError("storage_inside_official_package")


def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise SelectionError("invalid_vectors")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise SelectionError("zero_vector")
    return (array / norms).astype(np.float32)


class ChineseClipEncoder:
    def __init__(self, weights_dir):
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        started = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        options = dict(revision=REVISION, cache_dir=str(weights_dir))
        self.processor = ChineseCLIPProcessor.from_pretrained(MODEL, **options)
        self.model = ChineseCLIPModel.from_pretrained(MODEL, **options).to(self.device).eval()
        self.max_length = min(self.processor.tokenizer.model_max_length,
                              self.model.config.text_config.max_position_embeddings)
        self.metadata = dict(
            model=MODEL, revision=REVISION, interface="transformers.ChineseCLIP",
            transformers_version=importlib.metadata.version("transformers"),
            torch_version=importlib.metadata.version("torch"),
            pillow_version=importlib.metadata.version("Pillow"),
            preprocessing_version="native_chinese_clip_rgb.v1",
            image_processor=self.processor.image_processor.to_dict(),
            tokenizer_class=type(self.processor.tokenizer).__name__,
            vocabulary_sha256=digest(self.processor.tokenizer.get_vocab()),
            max_text_tokens=self.max_length)
        self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _array(features):
        # Transformers 4 returns a Tensor; 5 returns projected pooler_output.
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features.detach().float().cpu().numpy()

    def encode_images(self, images):
        import torch
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            return self._array(self.model.get_image_features(**inputs))

    def encode_text(self, query):
        import torch
        inputs = self.processor.tokenizer(query, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[-1] > self.max_length:
            raise SelectionError("query_too_long")
        with torch.inference_mode():
            return self._array(self.model.get_text_features(**inputs.to(self.device)))


@lru_cache(maxsize=1)
def get_encoder(weights_dir):
    return ChineseClipEncoder(weights_dir)


def official_assets(asset_index):
    path = local_path(asset_index).resolve()
    if path.name != "asset_index.json":
        raise SelectionError("expected_official_asset_index")
    snapshot_hash = file_hash(path)
    schema = strict_json(path.read_text()).get("schema_version")
    standard = schema == standard_urdf.PREVIEW_SCHEMA
    extended = schema == asset_library.SCHEMA or standard
    index = (standard_urdf.verify_preview_index(path) if standard else
             asset_library.verify_preview_index(path) if extended else
             official.verify_index(path.parent))
    rows, assets = [], []
    for item in sorted(index["assets"], key=lambda v: v["asset_id"]):
        if item["status"] != "preview_passed":
            continue
        record = strict_json(official.safe_file(path.parent, item["record"]).read_text())
        preview = strict_json(official.safe_file(path.parent, record["preview_result"]).read_text())
        entry = record["entrypoint"]
        valid_entry = (Path(entry).suffix.lower() in asset_library.FORMATS if extended
                       else Path(entry).name == "model.xml")
        if not valid_entry or entry not in {f["path"] for f in record["source_files"]}:
            raise SelectionError("invalid_official_entrypoint")
        views = {Path(v["image"]["path"]).stem.removeprefix("view_"): v["image"]
                 for v in preview["views"]}
        if set(views) != set(VIEWS) or len(preview["views"]) != 6:
            raise SelectionError("expected_six_distinct_views")
        asset = dict(asset_id=item["asset_id"], record=item["record"],
                     entrypoint=entry, source_files=record["source_files"])
        if extended:
            asset.update(source_root=record["source_root"], model_format=record["format"],
                         source_inventory=index["source_inventory"])
        if standard:
            asset.update(standard_package=record["standard_package"],
                         standard_package_sha256=record["standard_package_sha256"])
        assets.append(asset)
        for view in VIEWS:
            image = views[view]
            if image["path"] != f"previews/{item['asset_id']}/view_{view}.png":
                raise SelectionError("invalid_preview_path")
            rows.append(dict(asset_id=item["asset_id"], view=view, image=image))
    if not assets:
        raise SelectionError("no_available_assets")
    if file_hash(path) != snapshot_hash:
        raise SelectionError("official_index_changed")
    return dict(path=str(path), sha256=snapshot_hash), assets, rows


def build_index(asset_index, output_dir, *, weights_dir=WEIGHTS_DIR, encoder=None):
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    separate(output, Path(asset_index).resolve().parent, weights_dir)
    writable_storage(output, weights_dir)
    if output.exists():
        raise SelectionError("output_directory_exists")
    reference, assets, rows = official_assets(asset_index)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="error", device=None, timings_s={})
    try:
        load_start = time.perf_counter()
        try:
            encoder = encoder or get_encoder(str(Path(weights_dir).resolve()))
        finally:
            report["timings_s"]["model_load"] = time.perf_counter() - load_start
        report["device"] = encoder.device
        vectors = []
        image_start = time.perf_counter()
        for start in range(0, len(rows), 6):
            images = []
            for row in rows[start:start + 6]:
                with Image.open(official.safe_file(Path(asset_index).parent,
                                                   row["image"]["path"])) as image:
                    if image.size != (512, 512):
                        raise SelectionError("invalid_preview_dimensions")
                    images.append(image.convert("RGB"))
            vectors.append(encoder.encode_images(images))
        vectors = normalize(np.concatenate(vectors))
        if len(vectors) != len(rows):
            raise SelectionError("vector_row_count_mismatch")
        report["timings_s"]["image_encoding"] = time.perf_counter() - image_start
        if official_assets(asset_index) != (reference, assets, rows):
            raise SelectionError("official_index_changed")
        np.save(output / "vectors.npy", vectors, allow_pickle=False)
        index = dict(schema_version=INDEX_SCHEMA, official_index=reference, assets=assets,
                     rows=rows, encoder=encoder.metadata, device=encoder.device,
                     vectors=official.fingerprint(output / "vectors.npy", output),
                     shape=list(vectors.shape), aggregation="max_of_six_cosine",
                     build_timings_s=report["timings_s"])
        write_json(output / "index.json", index)
        report.update(status="passed", asset_count=len(assets), image_count=len(rows),
                      index_sha256=file_hash(output / "index.json"))
        return index
    except Exception as exc:
        report["error"] = str(exc) if isinstance(exc, SelectionError) else "index_build_failed"
        raise SelectionError(report["error"]) from None
    finally:
        report["timings_s"]["total"] = time.perf_counter() - started
        write_json(output / "build_report.json", report)


def load_index(path):
    path = local_path(path).resolve()
    index = strict_json(path.read_text())
    if index["schema_version"] == union_index.SCHEMA:
        return union_index.load(path)
    if index["schema_version"] != INDEX_SCHEMA:
        raise SelectionError("invalid_clip_index_schema")
    if index["encoder"]["model"] != MODEL or index["encoder"]["revision"] != REVISION:
        raise SelectionError("clip_model_revision_mismatch")
    reference, assets, rows = official_assets(index["official_index"]["path"])
    recorded_reference = index["official_index"]
    if same_evidence_path(reference["path"], recorded_reference.get("path")):
        reference = dict(reference, path=recorded_reference["path"])
    if (reference, assets, rows) != (recorded_reference, index["assets"], index["rows"]):
        raise SelectionError("clip_official_binding_mismatch")
    official.verify_files(path.parent, [index["vectors"]])
    vectors = np.load(official.safe_file(path.parent, index["vectors"]["path"]), allow_pickle=False)
    if (vectors.dtype != np.float32 or list(vectors.shape) != index["shape"]
            or len(vectors) != len(rows) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise SelectionError("invalid_normalized_vectors")
    return index, vectors


def retrieve(rows, vectors, text_vector, top_k=3):
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise SelectionError("top_k_must_be_1_to_5")
    scores = normalize(vectors) @ normalize(text_vector)[0]
    if len(scores) != len(rows):
        raise SelectionError("vector_row_count_mismatch")
    grouped = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row["asset_id"], []).append(dict(row, score=float(score)))
    candidates = []
    for asset_id, views in grouped.items():
        if len(views) != 6 or {v["view"] for v in views} != set(VIEWS):
            raise SelectionError("expected_six_distinct_views")
        ordered = sorted(views, key=lambda v: (-v["score"], v["view"]))
        candidates.append(dict(asset_id=asset_id, score=ordered[0]["score"],
                               views=views, selected_views=ordered[:2]))
    candidates.sort(key=lambda v: (-v["score"], v["asset_id"]))
    # Neutral numbering in asset-ID order avoids encoding the retrieval rank for the VLM.
    numbers = {a: i + 1 for i, a in enumerate(sorted(c["asset_id"] for c in candidates[:top_k]))}
    return [dict(c, candidate_id=numbers[c["asset_id"]]) for c in candidates[:top_k]]


def image_messages(query, candidates, root):
    content = [{"type": "text", "text": json.dumps({"query": query}, ensure_ascii=False)}]
    for candidate in sorted(candidates, key=lambda c: c["candidate_id"]):
        content.append({"type": "text", "text": f"候选 {candidate['candidate_id']}"})
        for view in candidate["selected_views"]:
            image_root = local_path(view.get("preview_root", root))
            path = official.safe_file(image_root, view["image"]["path"])
            official.verify_files(image_root, [view["image"]])
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise SelectionError("invalid_preview_dimensions")
                buffer = io.BytesIO()
                # Re-encode RGB to remove metadata and semantic filenames from the wire.
                image.convert("RGB").save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]


def read_cache(path, key, candidate_ids):
    try:
        if path.stat().st_size > 1024 * 1024:
            return None
        cached = strict_json(path.read_text())
        if (cached["key"] != key or digest(cached["payload"]) != cached["payload_sha256"]):
            return None
        payload = cached["payload"]
        if validate_selection(payload["response"], candidate_ids) != payload["selection"]:
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def select(clip_index, query, output_dir, *, top_k=3, vlm_config=None, profile=None,
           timeout_s=60, cache_dir=CACHE_DIR, weights_dir=WEIGHTS_DIR, encoder=None, vlm=None):
    """Repeatable in-process entry point; returns a persisted report, including errors.

    Encoder and VLM injection support offline tests. Production calls reuse the native model.
    Existing or protected output directories raise before any write or model/network work.
    """
    started = time.perf_counter()
    output, clip_path = Path(output_dir).resolve(), local_path(clip_index).resolve()
    separate(output, clip_path.parent, cache_dir, weights_dir)
    writable_storage(output, cache_dir, weights_dir)
    # A malformed/missing index is reported inside the new query output as an integrity error.
    try:
        unverified = strict_json(clip_path.read_text())
        root = local_path(unverified["official_index"]["path"]).resolve().parent
    except (OSError, ValueError, KeyError, TypeError):
        root = None
    if root is not None:
        separate(*dict.fromkeys((output, root, clip_path.parent, cache_dir, weights_dir)))
    if output.exists():
        raise SelectionError("output_directory_exists")
    output.mkdir(parents=True, exist_ok=False)
    timings = dict(model_load=0.0, text_encoding=0.0, similarity=0.0, vlm=0.0,
                   integrity=0.0, total=0.0)
    report = dict(schema_version=SELECTION_VERSION, status="error", vlm_calls=0,
                  cache=dict(hit=False, key=None), timings_s=timings, device=None,
                  model_reused=False, image_encoding_calls=0, physics_status="not_evaluated")
    retrieval = dict(status="not_started", candidates=[])
    evidence = dict(status="not_started", candidates=[], response=None, validation=None)
    selection = None
    config = None
    stage = "configuration"
    try:
        if not isinstance(query, str) or not query.strip():
            raise SelectionError("empty_query")
        (output / "request.txt").write_text(query, encoding="utf-8")
        report["query_sha256"] = hashlib.sha256(query.encode()).hexdigest()
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise SelectionError("top_k_must_be_1_to_5")
        config = (vlm_config if isinstance(vlm_config, LLMProviderConfig)
                  else load_llm_provider_config(vlm_config, profile=profile))
        config = replace(config, timeout_s=timeout_s, max_attempts=1)
        if config.api_mode != "chat" or config.model != "gpt-4o":
            raise SelectionError("expected_gpt_4o_chat_configuration")
        report["vlm_config"] = config.safe_dict()
        stage = "integrity"
        index_hash = file_hash(clip_path)
        tick = time.perf_counter()
        try:
            index, vectors = load_index(clip_path)
        finally:
            timings["integrity"] += time.perf_counter() - tick
        root = local_path(index["official_index"]["path"]).parent
        report["clip_index_sha256"] = index_hash
        report["official_index"] = index["official_index"]
        stage = "clip_model"
        load_start = time.perf_counter()
        before = get_encoder.cache_info().hits
        try:
            encoder = encoder or get_encoder(str(Path(weights_dir).resolve()))
        finally:
            timings["model_load"] = time.perf_counter() - load_start
        report["model_reused"] = get_encoder.cache_info().hits > before
        report["device"] = encoder.device
        if encoder.metadata != index["encoder"]:
            raise SelectionError("encoder_preprocessing_mismatch_rebuild_index")
        stage = "text_encoding"
        tick = time.perf_counter()
        try:
            text_vector = encoder.encode_text(query)
        finally:
            timings["text_encoding"] = time.perf_counter() - tick
        stage = "similarity"
        tick = time.perf_counter()
        candidates = retrieve(index["rows"], vectors, text_vector, top_k)
        timings["similarity"] = time.perf_counter() - tick
        retrieval = dict(status="passed", query=query, requested_k=top_k,
                         actual_k=len(candidates), aggregation="max_of_six_cosine",
                         clip_index_sha256=index_hash, candidates=candidates)
        ids = {c["candidate_id"] for c in candidates}
        evidence["candidates"] = [dict(candidate_id=c["candidate_id"],
                                       images=[v["image"] for v in c["selected_views"]])
                                  for c in sorted(candidates, key=lambda c: c["candidate_id"])]
        key = digest(dict(query=query, top_k=top_k, index_sha256=index_hash,
                          transport="injected" if vlm is not None else "http",
                          encoder=encoder.metadata, config=config.safe_dict(),
                          prompt_version=PROMPT_VERSION, prompt_sha256=digest(PROMPT),
                          selection_version=SELECTION_VERSION))
        report["cache"]["key"] = key
        evidence.update(prompt_version=PROMPT_VERSION, model=config.model,
                        transport="injected" if vlm is not None else "http")
        cache_path = Path(cache_dir) / f"{key}.json"
        cached = read_cache(cache_path, key, ids)
        if cached is not None:
            report["cache"]["hit"] = True
            evidence["response"] = cached["response"]
            selection = cached["selection"]
        else:
            stage = "vlm"
            messages = image_messages(query, candidates, root)
            client = vlm or ChatVisionClient(config)
            report["vlm_calls"] = 1
            tick = time.perf_counter()
            try:
                raw = client(messages)
            except TimeoutError:
                raise SelectionError("vlm_timeout") from None
            finally:
                timings["vlm"] = time.perf_counter() - tick
            if not isinstance(raw, str) or len(raw.encode()) > 1024 * 1024:
                raise SelectionError("invalid_vlm_response")
            # Do not persist a provider echo of the credential, even on a malformed response.
            raw = raw.replace(config.api_key, "[REDACTED]")
            evidence["response"] = raw
            stage = "vlm_validation"
            selection = validate_selection(raw, ids)
        stage = "final_integrity"
        if file_hash(clip_path) != index_hash:
            raise SelectionError("clip_index_changed")
        tick = time.perf_counter()
        try:
            load_index(clip_path)
        finally:
            timings["integrity"] += time.perf_counter() - tick
        evidence.update(status=selection["status"], selection=selection, validation="passed")
        if selection["status"] == "selected":
            candidate = next(c for c in candidates
                             if c["candidate_id"] == selection["candidate_id"])
            asset = next(a for a in index["assets"] if a["asset_id"] == candidate["asset_id"])
            binding = dict(schema_version="genenv.selected_asset.v1", query=query,
                           query_sha256=report["query_sha256"], selection_key=key,
                           clip_index_sha256=index_hash, official_index=index["official_index"],
                           asset_id=asset["asset_id"], official_record=asset["record"],
                           model_entrypoint=str(
                               Path(asset.get("source_root", root)) / asset["entrypoint"]),
                           source_files=asset["source_files"], reason=selection["reason"],
                           visible_differences=selection["visible_differences"],
                           meaning="本次选中的候选；不表示所有描述、几何条件或物理要求已满足。",
                           physics_status="not_evaluated")
            if "source_root" in asset:
                binding.update(source_root=asset["source_root"],
                               model_format=asset["model_format"],
                               source_inventory=asset["source_inventory"])
            for field in ("standard_package", "standard_package_sha256", "member_index",
                          "member_asset_id", "preview_root"):
                if field in asset:
                    binding[field] = asset[field]
            write_json(output / "selected_asset.json", binding)
        if not report["cache"]["hit"]:
            payload = dict(response=evidence["response"], selection=selection)
            write_json(cache_path, dict(key=key, payload=payload, payload_sha256=digest(payload)))
        report["status"] = selection["status"]
        if selection["status"] == "selected" and (index.get("physics_after_selection")
                or "standard_package" in asset):
            stage = "physics"
            physics_dir = output / "asset_physics"
            with (output / "physics.log").open("w") as log:
                child = subprocess.run([sys.executable, "-m",
                    "self_improving.sim_adapters.genesis.validate_single_asset", "--binding",
                    str(output / "selected_asset.json"), "--output-dir", str(physics_dir)],
                    stdout=log, stderr=log, timeout=600,
                    env=dict(os.environ, OMP_NUM_THREADS="2"))
            physics = strict_json((physics_dir / "physics_result.json").read_text())
            load_index(clip_path)
            if child.returncode != physics["exit_code"]:
                raise SelectionError("physics_process_result_mismatch")
            binding["physics_status"] = physics["physics_status"]
            binding["physics_evidence"] = official.fingerprint(
                physics_dir / "physics_result.json", output)
            write_json(output / "selected_asset.json", binding)
            report.update(physics_status=physics["physics_status"], exit_code=physics["exit_code"],
                          selected_asset_id=binding["asset_id"])
            if child.returncode:
                report.update(status="error", error="selected_asset_physics_failed",
                              error_stage="physics")
    except Exception as exc:
        code = str(exc) if isinstance(exc, SelectionError) else f"{stage}_failed"
        if config is not None:
            code = code.replace(config.api_key, "[REDACTED]")
        report.update(status="error", error=code, error_stage=stage)
        evidence.update(status="error", validation=code)
        if stage != "physics":
            (output / "selected_asset.json").unlink(missing_ok=True)
        else:
            report.update(exit_code=1, physics_status="error")
    finally:
        if not (output / "request.txt").exists():
            (output / "request.txt").write_text("", encoding="utf-8")
        write_json(output / "retrieval_result.json", retrieval)
        write_json(output / "vlm_selection.json", evidence)
        report["files"] = [official.fingerprint(output / name, output) for name in
                           ("request.txt", "retrieval_result.json", "vlm_selection.json",
                            "selected_asset.json") if (output / name).exists()]
        timings["total"] = time.perf_counter() - started
        write_json(output / "run_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index", help="Encode six verified views per asset once")
    build.add_argument("--asset-index", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    union = commands.add_parser("build-union-index")
    union.add_argument("--clip-index", action="append", required=True)
    union.add_argument("--output-dir", required=True, type=Path)
    choose = commands.add_parser("select", help="Retrieve Top-K and make at most one VLM request")
    choose.add_argument("--clip-index", required=True, type=Path)
    choose.add_argument("--query", required=True)
    choose.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    choose.add_argument("--vlm-config", type=Path, required=True)
    choose.add_argument("--profile")
    choose.add_argument("--timeout-s", type=float, default=60)
    choose.add_argument("--output-dir", required=True, type=Path)
    choose.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    choose.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in {"build-index", "build-union-index"}:
            index = (build_index(**args) if command == "build-index" else union_index.build(**args))
            print(json.dumps(dict(status="passed", assets=len(index["assets"]))))
            return 0
        report = select(**args)
        print(json.dumps(report, ensure_ascii=False))
        return report.get("exit_code", {"selected": 0, "rejected": 2, "error": 1}[report["status"]])
    except Exception as exc:
        code = str(exc) if isinstance(exc, SelectionError) else "initialization_failed"
        print(json.dumps(dict(status="error", error=code)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

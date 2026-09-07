"""Deterministic semantic grounding from SceneSpec queries to real assets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .catalog import AssetCatalog, CatalogEntry, CatalogModel
from .schema import RejectedCandidate, SceneObjectSpec, SceneSpec, SceneSpecError


@dataclass(frozen=True)
class GroundedSelection:
    query: SceneObjectSpec
    entry: CatalogEntry
    model: CatalogModel
    score: float
    reasons: tuple[str, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]


class GroundingMiss(SceneSpecError):
    """An expected lookup miss, distinct from catalog or programming errors."""

    def __init__(self, query, *, blocked, rejected):
        super().__init__(
            f"no usable catalog candidate for {query.object_id} category={query.category} "
            f"color={query.color} material={query.material}"
        )
        self.status = "blocked" if blocked else "missing"
        self.rejected_candidates = tuple(rejected[:25])


def _tie_break(seed: int, object_id: str, asset_id: str, model_id: int) -> str:
    return hashlib.sha256(f"{seed}:{object_id}:{asset_id}:{model_id}".encode("utf-8")).hexdigest()


def _semantic_score(query: SceneObjectSpec, entry: CatalogEntry) -> tuple[float, list[str]] | None:
    category = query.category.lower()
    aliases = {alias.lower().replace(" ", "_") for alias in entry.aliases}
    semantic = entry.semantic_name.lower().replace(" ", "_")
    if entry.category == category:
        score = 100.0
        reasons = ["exact category match"]
    elif semantic == category:
        score = 95.0
        reasons = ["exact semantic_name match"]
    elif category in aliases:
        score = 90.0
        reasons = ["exact alias match"]
    else:
        return None
    if query.color:
        if query.color in entry.colors:
            score += 5.0
            reasons.append(f"color metadata matches {query.color}")
        elif entry.colors:
            return None
        else:
            reasons.append(f"color metadata unknown; query {query.color} preserved")
    if query.material:
        if query.material in entry.materials:
            score += 5.0
            reasons.append(f"material metadata matches {query.material}")
        elif entry.materials:
            return None
        else:
            reasons.append(f"material metadata unknown; query {query.material} preserved")
    if "geometry_compatible_derived" in entry.source_notes:
        score += 3.0
        reasons.append("derived asset satisfies nested geometry")
    return score, reasons


def ground_object(query: SceneObjectSpec, catalog: AssetCatalog, *, seed: int) -> GroundedSelection:
    accepted: list[tuple[float, str, CatalogEntry, CatalogModel, tuple[str, ...]]] = []
    rejected: list[RejectedCandidate] = []
    blocked = False
    for entry in catalog.entries:
        semantic = _semantic_score(query, entry)
        if semantic is None:
            continue
        base_score, semantic_reasons = semantic
        if not entry.available:
            blocked = True
            rejected.append(
                RejectedCandidate(
                    asset_id=entry.asset_id,
                    score=base_score,
                    reasons=tuple(["asset unavailable", *entry.availability_reasons]),
                )
            )
            continue
        if not entry.models:
            blocked = True
            rejected.append(
                RejectedCandidate(
                    asset_id=entry.asset_id, score=base_score, reasons=("asset has no models",)
                )
            )
        for model in entry.models:
            score = base_score
            reasons = list(semantic_reasons)
            if not model.usable:
                # A known attribute mismatch is missing, not an unusable match.
                if not query.color or not model.colors or query.color in model.colors:
                    blocked = True
                rejected.append(
                    RejectedCandidate(
                        asset_id=entry.asset_id,
                        model_id=model.model_id,
                        score=score,
                        reasons=tuple(["model unusable", *model.missing]),
                    )
                )
                continue
            if query.color:
                if query.color in model.colors:
                    # a measured true-colour model outranks unknown-colour
                    # candidates in EVERY entry, so "a red cup" finds the
                    # actually-red cup instead of tinting the first cup
                    # (measured 2026-08-15: true hits 9/44 -> 23/44)
                    score += 4.0
                    reasons.append(f"model color matches {query.color}")
                elif model.colors:
                    rejected.append(
                        RejectedCandidate(
                            asset_id=entry.asset_id,
                            model_id=model.model_id,
                            score=score,
                            reasons=(f"model colors {list(model.colors)} lack {query.color}",),
                        )
                    )
                    continue
            if model.collision_path or model.urdf_path:
                score += 2.0
                reasons.append("collision representation available")
            if model.dimensions_m:
                score += 1.0
                reasons.append("normalized dimensions available")
            accepted.append(
                (
                    -score,
                    _tie_break(seed, query.object_id, entry.asset_id, model.model_id),
                    entry,
                    model,
                    tuple(reasons),
                )
            )
    if not accepted:
        rejected.sort(key=lambda item: (-item.score, item.asset_id, item.model_id or -1))
        raise GroundingMiss(query, blocked=blocked, rejected=rejected)
    accepted.sort(key=lambda item: (item[0], item[1], item[2].asset_id, item[3].model_id))
    neg_score, _, entry, model, reasons = accepted[0]
    for other_neg_score, _, other_entry, other_model, other_reasons in accepted[1:]:
        rejected.append(
            RejectedCandidate(
                asset_id=other_entry.asset_id,
                model_id=other_model.model_id,
                score=-other_neg_score,
                reasons=tuple(["lower deterministic rank", *other_reasons]),
            )
        )
    rejected.sort(key=lambda item: (-item.score, item.asset_id, item.model_id or -1))
    return GroundedSelection(
        query=query,
        entry=entry,
        model=model,
        score=-neg_score,
        reasons=reasons,
        rejected_candidates=tuple(rejected[:25]),
    )


def ground_scene(
    spec_objects: tuple[SceneObjectSpec, ...], catalog: AssetCatalog, *, seed: int
) -> dict[str, GroundedSelection]:
    return {
        query.object_id: ground_object(query, catalog, seed=seed)
        for query in sorted(spec_objects, key=lambda item: item.object_id)
    }


def resolve_scene_assets(spec: SceneSpec, catalog: AssetCatalog) -> dict:
    """Inspect every object without changing the existing selection algorithm."""
    objects = []
    for query in sorted(spec.objects, key=lambda obj: obj.object_id):
        try:
            selected = ground_object(query, catalog, seed=spec.seed)
        except GroundingMiss as error:
            objects.append(
                {
                    "object_id": query.object_id,
                    "status": error.status,
                    "reasons": [str(error)],
                    "rejected_candidates": [
                        r.model_dump(mode="json") for r in error.rejected_candidates
                    ],
                }
            )
        else:
            objects.append(
                {
                    "object_id": query.object_id,
                    "status": "matched",
                    "asset_id": selected.entry.asset_id,
                    "model_id": selected.model.model_id,
                    "score": selected.score,
                    "reasons": list(selected.reasons),
                    "rejected_candidates": [
                        r.model_dump(mode="json") for r in selected.rejected_candidates
                    ],
                }
            )
    return {
        "schema_version": "robotwin.asset_resolution.v1",
        "scene_id": spec.scene_id,
        "source_scene_spec_sha256": spec.digest(),
        "asset_catalog_sha256": catalog.digest(),
        "status": "matched" if all(o["status"] == "matched" for o in objects) else "unresolved",
        "objects": objects,
    }

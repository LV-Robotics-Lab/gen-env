"""Deterministic high-confidence checks for optional LLM semantic extraction.

These checks do not attempt to replace the language model. They reject
representations that contradict lexical evidence we can identify reliably:
known object counts, unsupported negation, invented attributes, and direct
binary-relation phrases. Unrecognized language remains the LLM's job and is
still constrained by SceneSpec downstream.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .parser import (
    COLOR_TERMS,
    MATERIAL_TERMS,
    OBJECT_TERMS,
    REGION_TERMS,
    MentionGroup,
    extract_mentions,
)
from .schema import RelationType, SceneSpecError

SEMANTIC_CHECK_VERSION = "scene_gen.llm_semantic_checks.v4"

_NEGATION = re.compile(
    r"\b(?:no|not|never|without|except|excluding|exclude|remove|omit|"
    r"leave\s+out|cannot|avoid|neither|nor|nowhere|anything\s+but|"
    r"anywhere\s+but|other\s+than|(?:do|does|did)\s+not|"
    r"(?:don|doesn|didn|isn|aren|wasn|weren|can|won|shouldn|mustn)['’]t)\b"
    r"|\bnon[-\s]+[a-z]"
    r"|不要|不能|不得|不含|不包括|没有|不是|不在|并非|勿|除了|移除|删除|排除|"
    r"非(?=黑|白|红|蓝|棕|绿|橙|粉|紫|黄|陶瓷|玻璃|金属|塑料|木质|木制)|"
    r"别(?:把|将|让|放|置|打开|关闭)|"
    r"未(?:被|曾|将|把|让|放|置|打开|关闭|包含|包括|使用|指定|在|是)",
    flags=re.IGNORECASE,
)


class SemanticMismatch(ValueError):
    """The candidate contradicts deterministic evidence in the request."""


class AmbiguousReference(SemanticMismatch):
    """The request contains a reference that cannot be bound deterministically."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _mask_table_region_phrases(text: str) -> str:
    masked = list(text)
    for terms in REGION_TERMS.values():
        for term in terms:
            for start, end in _term_spans(text, term):
                masked[start:end] = " " * (end - start)
    return "".join(masked)


def _term_present(text: str, term: str) -> bool:
    if re.search(r"[a-z0-9]", term):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return term in text


def _term_spans(text: str, term: str) -> list[tuple[int, int]]:
    if re.search(r"[a-z0-9]", term):
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    else:
        pattern = re.escape(term)
    return [match.span() for match in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _attribute_bindings(
    request: str,
    mentions: list[MentionGroup],
    lexicon: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Bind explicit attributes only through local noun-phrase syntax."""

    normalized = _normalize(request)
    occurrences: set[tuple[int, int, str]] = set()
    for canonical, terms in lexicon.items():
        for term in {canonical, *terms}:
            for start, end in _term_spans(normalized, term):
                occurrences.add((start, end, canonical))

    all_attribute_terms = {
        term
        for attribute_lexicon in (COLOR_TERMS, MATERIAL_TERMS)
        for canonical, terms in attribute_lexicon.items()
        for term in {canonical, *terms}
        if re.fullmatch(r"[a-z][a-z0-9_-]*", term)
    }
    attribute_pattern = "|".join(
        re.escape(term) for term in sorted(all_attribute_terms, key=len, reverse=True)
    )

    chinese_attribute_terms = {
        term
        for attribute_lexicon in (COLOR_TERMS, MATERIAL_TERMS)
        for canonical, terms in attribute_lexicon.items()
        for term in {canonical, *terms}
        if re.search(r"[\u3400-\u9fff]", term)
    }
    chinese_attribute_pattern = "|".join(
        re.escape(term) for term in sorted(chinese_attribute_terms, key=len, reverse=True)
    )
    prenominal_gap = re.compile(
        rf"(?:\s*(?:(?:and\s+)?(?:{attribute_pattern})\s+)*|"
        rf"\s*(?:{chinese_attribute_pattern})*\s*的?\s*)$",
        flags=re.IGNORECASE,
    )
    postnominal_gap = re.compile(
        rf"(?:\s*(?:(?:that|which)\s+)?(?:(?:is|was)\s+)?"
        rf"(?:(?:{attribute_pattern})\s+(?:and\s+))*"
        rf"(?:(?:made\s+of|colored|painted)(?:\s+(?:in|with))?\s+)?)$"
        r"|(?:\s*(?:是|为|呈|由)\s*)$",
        flags=re.IGNORECASE,
    )

    bindings: dict[str, str] = {}
    for start, end, canonical in sorted(occurrences):
        before = normalized[max(0, start - 20) : start]
        after = normalized[end : min(len(normalized), end + 24)]
        if re.search(r"(?:\btable\b|桌面|桌子)(?:\s+is|\s*是)?\s*$", before) or re.match(
            r"\s*(?:\btable\b|桌面|桌子)", after
        ):
            continue
        if any(
            start < mention.end
            and mention.start < end
            and normalized[start:end]
            in set(mention.category.replace("_", " ").replace("-", " ").split())
            for mention in mentions
        ):
            continue

        owners: list[MentionGroup] = []
        for mention in mentions:
            if end <= mention.start:
                gap = normalized[end : mention.start]
                if len(gap) <= 48 and prenominal_gap.fullmatch(gap):
                    owners.append(mention)
            elif mention.end <= start:
                gap = normalized[mention.end : start]
                if len(gap) <= 80 and postnominal_gap.fullmatch(gap):
                    owners.append(mention)
        identities = {owner.object_ids for owner in owners}
        if not identities:
            continue
        if len(identities) != 1:
            raise SemanticMismatch(
                f"{canonical!r} attribute cannot be bound to one object unambiguously"
            )
        owner_ids = next(iter(identities))
        for object_id in owner_ids:
            existing = bindings.get(object_id)
            if existing is not None and existing != canonical:
                raise SemanticMismatch(
                    f"multiple conflicting attributes are bound to {object_id!r}"
                )
            bindings[object_id] = canonical
    return bindings


def reject_unsupported_negation(request: str) -> None:
    """Reject negative instructions because the current IR cannot encode them."""

    if _NEGATION.search(request):
        raise SemanticMismatch(
            "request contains negation that the SceneSpec relation language cannot represent"
        )


def _mentions_or_empty(request: str) -> list[MentionGroup]:
    try:
        return extract_mentions(request)
    except SceneSpecError as exc:
        if str(exc) == "no supported tabletop object found":
            return []
        raise SemanticMismatch(str(exc)) from exc


_ATTRIBUTE_TOKENS = {
    term
    for lexicon in (COLOR_TERMS, MATERIAL_TERMS)
    for terms in lexicon.values()
    for term in terms
    if re.fullmatch(r"[a-z][a-z0-9_-]*", term)
}

_UNSUPPORTED_OBJECT_MODIFIERS = {
    "big",
    "broken",
    "damaged",
    "gray",
    "grey",
    "heavy",
    "huge",
    "large",
    "light",
    "narrow",
    "paper",
    "round",
    "rubber",
    "short",
    "small",
    "square",
    "steel",
    "tall",
    "tiny",
    "wide",
}

_IRREGULAR_OPEN_VOCAB_PLURALS = {"children", "feet", "geese", "men", "mice", "teeth", "women"}


def _semantic_category(mention: MentionGroup) -> str:
    if mention.category in OBJECT_TERMS:
        return mention.category
    words = re.findall(r"[a-z0-9]+", mention.surface.lower())
    while len(words) > 1 and words[0] in _ATTRIBUTE_TOKENS:
        words.pop(0)
    category = "_".join(words)[:64]
    return category or mention.category


def _canonical_attribute(
    token: str,
    lexicon: dict[str, tuple[str, ...]],
) -> str | None:
    for canonical, terms in lexicon.items():
        if token in terms:
            return canonical
    return None


def _resolve_modified_reference(
    *,
    request: str,
    category: str,
    descriptor_text: str,
    introductions: list[MentionGroup],
) -> tuple[str, ...]:
    category_mentions = [item for item in introductions if item.category == category]
    instance_ids = [object_id for item in category_mentions for object_id in item.object_ids]
    descriptor_tokens = re.findall(r"[a-z][a-z0-9_-]*", descriptor_text)

    ordinal_positions = {"first": 0, "second": 1, "third": 2}
    ordinals = [
        ordinal_positions[token] for token in descriptor_tokens if token in ordinal_positions
    ]
    if ordinals:
        if len(set(ordinals)) != 1 or ordinals[0] >= len(instance_ids):
            raise AmbiguousReference(
                f"ordinal reference to {category!r} does not identify one existing instance"
            )
        candidates = {instance_ids[ordinals[0]]}
    else:
        candidates = set(instance_ids)

    requested_colors = {
        value
        for token in descriptor_tokens
        if (value := _canonical_attribute(token, COLOR_TERMS)) is not None
    }
    requested_materials = {
        value
        for token in descriptor_tokens
        if (value := _canonical_attribute(token, MATERIAL_TERMS)) is not None
    }
    if len(requested_colors) > 1 or len(requested_materials) > 1:
        raise AmbiguousReference(f"reference to {category!r} has conflicting descriptors")
    if requested_colors:
        color_bindings = _attribute_bindings(request, introductions, COLOR_TERMS)
        requested_color = next(iter(requested_colors))
        candidates = {item for item in candidates if color_bindings.get(item) == requested_color}
    if requested_materials:
        material_bindings = _attribute_bindings(request, introductions, MATERIAL_TERMS)
        requested_material = next(iter(requested_materials))
        candidates = {
            item for item in candidates if material_bindings.get(item) == requested_material
        }
    if len(candidates) != 1:
        raise AmbiguousReference(
            f"definite reference to {category!r} has multiple possible instances"
        )
    return (next(iter(candidates)),)


def _semantic_mentions(request: str) -> tuple[list[MentionGroup], list[MentionGroup]]:
    """Resolve high-confidence definite repeats without inventing new instances."""

    normalized = _normalize(request)
    counts: Counter[str] = Counter()
    latest_ids: dict[str, tuple[str, ...]] = {}
    resolved: list[MentionGroup] = []
    introductions: list[MentionGroup] = []
    for mention in _independent_semantic_mentions(request):
        category = _semantic_category(mention)
        prefix = normalized[max(0, mention.start - 48) : mention.start]
        descriptors = "|".join(
            re.escape(term)
            for term in sorted(
                {*_ATTRIBUTE_TOKENS, "first", "second", "third", "other"},
                key=len,
                reverse=True,
            )
        )
        reference_match = (
            re.search(
                rf"(?:\b(?:the|this|that|same)\s+"
                rf"(?P<descriptors>(?:(?:{descriptors})\s+)*)|"
                rf"(?P<chinese>这个|那个|该|上述))$",
                prefix,
            )
            if category in latest_ids
            else None
        )
        is_reference = reference_match is not None
        if is_reference:
            if counts[category] == 1:
                object_ids = latest_ids[category]
            else:
                descriptor_text = reference_match.group("descriptors") or ""
                object_ids = _resolve_modified_reference(
                    request=request,
                    category=category,
                    descriptor_text=descriptor_text,
                    introductions=introductions,
                )
            quantity = 1
        else:
            first_index = counts[category] + 1
            object_ids = tuple(
                f"{category}_{index}"
                for index in range(first_index, first_index + mention.quantity)
            )
            counts[category] += mention.quantity
            latest_ids[category] = object_ids
            quantity = mention.quantity
        semantic = MentionGroup(
            category=category,
            surface=mention.surface,
            start=mention.start,
            end=mention.end,
            group_id=mention.group_id,
            quantity=quantity,
            object_ids=object_ids,
        )
        resolved.append(semantic)
        if not is_reference:
            introductions.append(semantic)
    return resolved, introductions


def _pluralize_surface(surface: str) -> str:
    words = surface.split()
    noun = words[-1]
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        noun = f"{noun[:-1]}ies"
    elif noun.endswith(("s", "x", "z", "ch", "sh")):
        noun = f"{noun}es"
    else:
        noun = f"{noun}s"
    return " ".join((*words[:-1], noun))


def _is_composite_generic_mention(mention: MentionGroup) -> bool:
    if mention.category in OBJECT_TERMS:
        return False
    return bool(
        re.search(
            r"\b(?:and|plus)\s+(?:a|an|one)\b|,|以及|和|与|、",
            mention.surface,
            flags=re.IGNORECASE,
        )
    )


def _split_composite_generic_mention(mention: MentionGroup) -> list[MentionGroup]:
    if not _is_composite_generic_mention(mention):
        return [mention]

    delimiter = re.compile(
        r"\s*,\s*|\s+(?:and|plus)\s+(?=(?:a|an|one)\s+)",
        flags=re.IGNORECASE,
    )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in delimiter.finditer(mention.surface):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(mention.surface)))

    results: list[MentionGroup] = []
    for local_start, local_end in spans:
        segment = mention.surface[local_start:local_end]
        words = list(re.finditer(r"[a-z0-9_-]+", segment, flags=re.IGNORECASE))
        if words and words[0].group(0).lower() in {"a", "an", "the", "one"}:
            words.pop(0)
        while len(words) > 1 and words[0].group(0).lower() in _ATTRIBUTE_TOKENS:
            words.pop(0)
        if not words:
            raise SemanticMismatch("generic object list contains an empty object phrase")
        category = "_".join(word.group(0).lower() for word in words)[:64]
        start = mention.start + local_start + words[0].start()
        end = mention.start + local_start + words[-1].end()
        results.append(
            MentionGroup(
                category=category,
                surface=mention.surface[
                    local_start + words[0].start() : local_start + words[-1].end()
                ],
                start=start,
                end=end,
                group_id=mention.group_id,
                quantity=1,
                object_ids=(),
            )
        )
    return results


def _supplemental_generic_mentions(
    request: str,
    existing: list[MentionGroup],
) -> list[MentionGroup]:
    """Recover explicit list tails and relation targets hidden by known mentions."""

    normalized = _normalize(request)
    noun = r"[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3}?"
    relation = (
        r"on\s+top\s+of|atop|upon|inside|into|near|next\s+to|beside|"
        r"close\s+to|adjacent\s+to|(?:to\s+the\s+)?left\s+of|"
        r"(?:to\s+the\s+)?right\s+of|in\s+front\s+of|behind|in|on"
    )
    patterns = (
        rf"\b(?:and|plus)\s+(?:a|an|one)\s+(?P<object>{noun})"
        rf"(?=\s+(?:{relation})\b|[.,]|$)",
        rf"\b(?:{relation})\s+(?:a|an|one)\s+(?P<object>{noun})"
        r"(?=\s+(?:and|while|then)\b|[.,]|$)",
    )
    results: list[MentionGroup] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            start, end = match.span("object")
            if (start, end) in seen_spans or any(
                start < item.end and item.start < end for item in [*existing, *results]
            ):
                continue
            words = list(
                re.finditer(
                    r"[a-z0-9_-]+",
                    normalized[start:end],
                    flags=re.IGNORECASE,
                )
            )
            while len(words) > 1 and words[0].group(0).lower() in _ATTRIBUTE_TOKENS:
                words.pop(0)
            if not words:
                continue
            category = "_".join(word.group(0).lower() for word in words)[:64]
            if set(category.split("_")) & {
                "robot",
                "table",
                "tabletop",
                "workspace",
                "world",
            }:
                continue
            object_start = start + words[0].start()
            object_end = start + words[-1].end()
            seen_spans.add((start, end))
            results.append(
                MentionGroup(
                    category=category,
                    surface=normalized[object_start:object_end],
                    start=object_start,
                    end=object_end,
                    group_id=-1,
                    quantity=1,
                    object_ids=(),
                )
            )
    return results


def _enumerated_generic_mentions(
    request: str,
    existing: list[MentionGroup],
) -> list[MentionGroup]:
    """Recover simple open-vocabulary objects from placement lists."""

    normalized = _normalize(request)
    clause_pattern = re.compile(
        r"\b(?:place|put|add|create|generate|set)\s+"
        r"(?P<items>[^.!?;]{1,240}?)"
        r"(?=\s+(?:on\s+(?:the\s+)?table|onto\s+(?:the\s+)?table|"
        r"on\s+top\s+of|inside|into|near|next\s+to|beside|close\s+to|"
        r"adjacent\s+to|(?:to\s+the\s+)?(?:left|right)\s+of|"
        r"in\s+front\s+of|behind|within\s+\d+(?:\.\d+)?\s*"
        r"(?:m\b|meters?\b)(?:\s+of)?|\bin\b)|[.!?;]|$)",
        flags=re.IGNORECASE,
    )
    delimiter = re.compile(
        r"\s*,\s*(?:and\s+)?|\s+(?:and|plus)\s+",
        flags=re.IGNORECASE,
    )
    forbidden = {
        "behind",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "on",
        "onto",
        "top",
        "under",
        "with",
    }
    reserved = {"robot", "table", "tabletop", "workspace", "world"}
    results: list[MentionGroup] = []
    for clause in clause_pattern.finditer(normalized):
        items = clause.group("items")
        separators = list(delimiter.finditer(items))
        spans: list[tuple[int, int]] = []
        cursor = 0
        for separator in separators:
            spans.append((cursor, separator.start()))
            cursor = separator.end()
        spans.append((cursor, len(items)))
        for local_start, local_end in spans:
            segment = items[local_start:local_end]
            words = list(re.finditer(r"[a-z0-9_-]+", segment, flags=re.IGNORECASE))
            leading_determiner = (
                words[0].group(0).lower()
                if words and words[0].group(0).lower() in {"a", "an", "the", "one"}
                else None
            )
            explicit_singular = leading_determiner in {"a", "an", "one"}
            if leading_determiner is not None:
                words.pop(0)
            raw_tokens = [word.group(0).lower() for word in words]
            if raw_tokens and raw_tokens[0] in _UNSUPPORTED_OBJECT_MODIFIERS:
                raise SemanticMismatch(
                    f"object modifier {raw_tokens[0]!r} cannot be represented by SceneSpec"
                )
            while len(words) > 1 and words[0].group(0).lower() in _ATTRIBUTE_TOKENS:
                words.pop(0)
            tokens = [word.group(0).lower() for word in words]
            if not tokens or len(tokens) > 4 or set(tokens) & forbidden:
                continue
            looks_plural = tokens[-1] in _IRREGULAR_OPEN_VOCAB_PLURALS or (
                tokens[-1].endswith("s") and not tokens[-1].endswith(("ss", "us", "is", "ns"))
            )
            if looks_plural and (not explicit_singular or leading_determiner in {"a", "an", "one"}):
                raise SemanticMismatch(
                    "open-vocabulary plural requires an explicit supported quantity"
                )
            category = "_".join(tokens)[:64]
            if set(category.split("_")) & reserved:
                continue
            start = clause.start("items") + local_start + words[0].start()
            end = clause.start("items") + local_start + words[-1].end()
            if any(start < item.end and item.start < end for item in [*existing, *results]):
                continue
            results.append(
                MentionGroup(
                    category=category,
                    surface=normalized[start:end],
                    start=start,
                    end=end,
                    group_id=-1,
                    quantity=1,
                    object_ids=(),
                )
            )
    return results


def _attribute_noun_mentions(
    request: str,
    existing: list[MentionGroup],
) -> list[MentionGroup]:
    """Recover cases such as an orange where a color word is the noun."""

    normalized = _normalize(request)
    results: list[MentionGroup] = []
    relation_suffix = re.compile(
        r"\s+(?:on\s+top\s+of|inside|into|in\b|on\b|near|next\s+to|"
        r"beside|close\s+to|adjacent\s+to|to\s+the\s+(?:left|right)|"
        r"in\s+front\s+of|behind|atop|upon|and\b|plus\b)|\s*[.,]|\s*$",
        flags=re.IGNORECASE,
    )
    for category in sorted(_ATTRIBUTE_TOKENS, key=len, reverse=True):
        variants = ((category, False), (_pluralize_surface(category), True))
        for variant, plural in variants:
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])",
                normalized,
                flags=re.IGNORECASE,
            ):
                if any(match.start() < item.end and item.start < match.end() for item in existing):
                    continue
                if relation_suffix.match(normalized[match.end() :]) is None:
                    continue
                prefix = normalized[max(0, match.start() - 24) : match.start()]
                quantity_match = re.search(
                    r"\b(a|an|one|two|three|\d+)\s+$",
                    prefix,
                    flags=re.IGNORECASE,
                )
                if quantity_match is None:
                    continue
                token = quantity_match.group(1).lower()
                quantity = {
                    "a": 1,
                    "an": 1,
                    "one": 1,
                    "two": 2,
                    "three": 3,
                }.get(token)
                quantity = int(token) if quantity is None else quantity
                if not 1 <= quantity <= 12:
                    raise SemanticMismatch(
                        f"quantity for open-vocabulary category {category!r} is unsupported"
                    )
                if (quantity > 1) != plural:
                    raise SemanticMismatch(
                        f"quantity does not agree with open-vocabulary category {category!r}"
                    )
                results.append(
                    MentionGroup(
                        category=category,
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        group_id=-1,
                        quantity=quantity,
                        object_ids=(),
                    )
                )
    return results


def _is_modal_can_mention(request: str, mention: MentionGroup) -> bool:
    if mention.category != "can" or mention.surface.lower() != "can":
        return False
    normalized = _normalize(request)
    prefix = normalized[max(0, mention.start - 80) : mention.start]
    suffix = normalized[mention.end : mention.end + 96]
    placement = r"(?:place|put|add|create|generate|set|stack|sit|rest)"
    if re.match(
        rf"\s+(?:you|we|it|the\s+robot)\s+(?:please\s+)?{placement}\b",
        suffix,
    ):
        return True
    if mention.start == 0 and re.match(
        r"\s+(?:a|an|the)\s+[^.!?]{1,72}\s+be\s+(?:placed|put|set|added|stacked)\b",
        suffix,
    ):
        return True
    if re.search(r"\bif\s+(?:you|we|it)\s*$", prefix):
        return True
    if re.search(r"\b(?:robot|we|you|it)\s*$", prefix) and re.match(
        rf"\s+(?:please\s+)?{placement}\b", suffix
    ):
        return True
    if not re.search(r"\b(?:a|an|the|one)\s*$", prefix) and re.match(
        r"\s+be\s+(?:placed|put|set|added|stacked)\b", suffix
    ):
        return True
    return False


def _reject_embedded_chinese_object_term(request: str, mention: MentionGroup) -> None:
    """Reject known CJK substrings embedded in an unknown compound noun."""

    if re.fullmatch(r"[\u3400-\u9fff]+", mention.surface) is None:
        return
    normalized = _normalize(request)
    chinese_attributes = {
        term
        for lexicon in (COLOR_TERMS, MATERIAL_TERMS)
        for terms in lexicon.values()
        for term in terms
        if re.search(r"[\u3400-\u9fff]", term)
    }
    prefix_endings = {
        "、",
        "一",
        "一个",
        "一只",
        "二",
        "三",
        "两",
        "两个",
        "两只",
        "三个",
        "三只",
        "于",
        "与",
        "及",
        "和",
        "在",
        "将",
        "把",
        "给",
        *chinese_attributes,
        *(f"{term}的" for term in chinese_attributes),
        "半开的",
        "打开的",
        "开启的",
        "关闭的",
        "闭合的",
        "关上的",
    }
    suffix_starts = (
        "、",
        "上",
        "下",
        "与",
        "为",
        "内",
        "关闭",
        "及",
        "和",
        "在",
        "外",
        "左",
        "开",
        "打",
        "位",
        "必须",
        "放",
        "是",
        "有",
        "的",
        "相邻",
        "离",
        "置",
        "装",
        "里",
        "右",
        "靠近",
    )
    prefix = normalized[: mention.start]
    suffix = normalized[mention.end :]
    if (
        prefix
        and re.search(r"[\u3400-\u9fff]$", prefix)
        and not any(prefix.endswith(ending) for ending in prefix_endings)
    ):
        raise SemanticMismatch(
            f"known object term {mention.surface!r} is embedded in an unsupported Chinese compound"
        )
    if suffix and re.match(r"[\u3400-\u9fff]", suffix) and not suffix.startswith(suffix_starts):
        raise SemanticMismatch(
            f"known object term {mention.surface!r} is embedded in an unsupported Chinese compound"
        )


def _independent_semantic_mentions(request: str) -> list[MentionGroup]:
    normalized = _normalize(request)
    mentions: list[MentionGroup] = []
    for mention in _mentions_or_empty(request):
        _reject_embedded_chinese_object_term(request, mention)
        if _is_modal_can_mention(request, mention):
            continue
        mentions.extend(_split_composite_generic_mention(mention))
    for candidate in _enumerated_generic_mentions(request, []):
        overlapping = [
            item for item in mentions if candidate.start < item.end and item.start < candidate.end
        ]
        if not overlapping:
            mentions.append(candidate)
        elif (
            len(overlapping) == 1
            and candidate.start == overlapping[0].start
            and candidate.end > overlapping[0].end
        ):
            extension = normalized[overlapping[0].end : candidate.end].strip()
            extension_tokens = re.findall(r"[a-z0-9_-]+", extension)
            unsafe_extension_tokens = {
                *_ATTRIBUTE_TOKENS,
                "adjacent",
                "atop",
                "behind",
                "beside",
                "close",
                "and",
                "anywhere",
                "are",
                "at",
                "can",
                "colored",
                "could",
                "here",
                "if",
                "is",
                "least",
                "nowhere",
                "made",
                "minimum",
                "must",
                "or",
                "painted",
                "should",
                "somewhere",
                "that",
                "there",
                "was",
                "were",
                "which",
                "would",
                "in",
                "inside",
                "into",
                "near",
                "next",
                "of",
                "on",
                "onto",
                "stacked",
                "to",
                "upon",
                "within",
            }
            if (
                extension_tokens
                and len(extension_tokens) <= 3
                and not any(token[0].isdigit() for token in extension_tokens)
                and not set(extension_tokens) & unsafe_extension_tokens
            ):
                mentions.remove(overlapping[0])
                mentions.append(candidate)
    mentions.extend(_supplemental_generic_mentions(request, mentions))
    mentions.extend(_attribute_noun_mentions(request, mentions))
    mentions.sort(key=lambda item: (item.start, item.end, item.category))
    return mentions


def _reject_unsupported_noun_modifiers(request: str) -> None:
    """Reject modifiers only when local noun-phrase syntax binds them exactly."""

    normalized = _normalize(request)
    mentions = _independent_semantic_mentions(request)
    supported = {
        *_ATTRIBUTE_TOKENS,
        "open",
        "opened",
        "closed",
        "shut",
        "half",
        "half-open",
        "partially",
        "halfway",
    }
    relation_words = {
        "adjacent",
        "atop",
        "behind",
        "beside",
        "close",
        "colored",
        "contains",
        "in",
        "inside",
        "left",
        "made",
        "near",
        "next",
        "on",
        "onto",
        "painted",
        "rests",
        "right",
        "stacked",
        "supports",
        "to",
        "upon",
        "within",
    }
    for index, mention in enumerate(mentions):
        previous_end = mentions[index - 1].end if index else 0
        prefix = normalized[previous_end : mention.start]
        determiners = list(re.finditer(r"\b(?:a|an|the|one)\s+", prefix))
        modifier_text = prefix[determiners[-1].end() :] if determiners else ""
        if determiners and re.fullmatch(
            r"(?:[a-z][a-z0-9_-]*\s+)*",
            modifier_text,
        ):
            modifiers = set(re.findall(r"[a-z][a-z0-9_-]*", modifier_text))
            unsupported = sorted(modifiers - supported)
            if unsupported:
                raise SemanticMismatch(
                    f"object modifier cannot be represented by SceneSpec: {unsupported!r}"
                )
        elif not determiners:
            clause_prefix = re.split(r"[.!?;,]", prefix)[-1]
            clause_prefix = re.sub(
                r"^\s*(?:place|put|add|create|generate|set)\s+",
                "",
                clause_prefix,
            )
            unsupported = sorted(
                set(re.findall(r"[a-z][a-z0-9_-]*", clause_prefix)) & _UNSUPPORTED_OBJECT_MODIFIERS
            )
            if unsupported:
                raise SemanticMismatch(
                    f"object modifier cannot be represented by SceneSpec: {unsupported!r}"
                )

        next_start = mentions[index + 1].start if index + 1 < len(mentions) else len(normalized)
        suffix = normalized[mention.end : next_start]
        postnominal = re.match(
            r"\s*(?:(?:that|which)\s+)?(?:is|was)\s+"
            r"(?P<modifier>[a-z][a-z0-9_-]*)\b",
            suffix,
        )
        if postnominal is not None:
            modifier = postnominal.group("modifier")
            if modifier not in supported and modifier not in relation_words:
                raise SemanticMismatch(
                    f"object modifier {modifier!r} cannot be represented by SceneSpec"
                )


_UNSUPPORTED_SPATIAL_CUE = re.compile(
    r"\b(?:under(?:neath)?|below|above|over|far(?:\s+away)?\s+from|away\s+from|"
    r"across\s+from|between|outside(?:\s+of)?|off|touching|contacting|"
    r"overlapping|around|surrounding)\b"
    r"|下方|下面|上方|正上方|正下方|远离|之间|外面|外部|之外|接触|相接|重叠|周围|围绕",
    flags=re.IGNORECASE,
)
_DEPICTION_REFERENCE = re.compile(
    r"\b(?:drawing|picture|image|photo|photograph|painting|sketch|"
    r"illustration|depiction)\s+of\b|(?:图片|图像|照片|画)中的?",
    flags=re.IGNORECASE,
)
_AMBIGUOUS_CHOICE = re.compile(
    r"\beither\b[^.!?]{0,160}\bor\b|"
    r"\band\s*/\s*or\b|\bor\b|"
    r"或者|或",
    flags=re.IGNORECASE,
)
_UNSUPPORTED_COMPLEX_CLAUSE = re.compile(
    r"\b(?:as\s+well\s+as|along\s+with|together\s+with)\b|"
    r"\b(?:which|that)\s+(?:is\s+)?(?:to\s+the\s+)?"
    r"(?:left|right)\s+of\b|"
    r"\b(?:which|that)\s+(?:is\s+)?(?:near|behind|in\s+front\s+of)\b|"
    r"在[^。.!?]{0,80}上(?:面)?放|放到[^。.!?]{0,80}(?:里|内部)",
    flags=re.IGNORECASE,
)
_RELATION_DISJUNCTION = re.compile(
    r"\b(?:near|behind|inside|atop|(?:to\s+the\s+)?(?:left|right)\s+of)"
    r"\s+or\s+(?:near|behind|inside|atop|(?:to\s+the\s+)?(?:left|right)\s+of)\b",
    flags=re.IGNORECASE,
)
_GROUP_REFERENCE = re.compile(r"\bboth\b|两者|二者", flags=re.IGNORECASE)
_GROUP_ARTICULATION = re.compile(
    r"\bboth\s+(?P<state>open|opened|closed|shut)\b",
    flags=re.IGNORECASE,
)
_COORDINATED_ARTICULATION_COMMAND = re.compile(
    r"(?:^|[.!?]\s*)(?:open|close|shut)\s+(?:a|an|the)\s+"
    r"[^.!?]{1,64}\s+(?:and|,)\s+(?:(?:a|an|the)\s+)?[a-z]",
    flags=re.IGNORECASE,
)
_INEXACT_ARTICULATION = re.compile(
    r"\b(?:ajar|slightly\s+open|cracked\s+open)\b|微开|略微打开|留一条缝",
    flags=re.IGNORECASE,
)
_REFERENCE_PRONOUN = re.compile(
    r"(?<![a-z0-9])(?P<pronoun>it|them|they|former|latter)(?![a-z0-9])"
    r"|(?P<chinese>它们?|前者|后者)",
    flags=re.IGNORECASE,
)


def reject_unsupported_request_semantics(request: str) -> None:
    """Reject request constructs that cannot be bound to this IR exactly."""

    reject_unsupported_negation(request)
    normalized = _normalize(request)
    _reject_unsupported_noun_modifiers(request)
    if _UNSUPPORTED_SPATIAL_CUE.search(normalized):
        raise SemanticMismatch(
            "request contains a spatial relation that the SceneSpec language cannot represent"
        )
    if _DEPICTION_REFERENCE.search(normalized):
        raise SemanticMismatch(
            "request contains a depiction reference whose pictured subject is not a scene object"
        )
    if _AMBIGUOUS_CHOICE.search(normalized):
        raise AmbiguousReference("request contains an unresolved either/or object choice")
    if _UNSUPPORTED_COMPLEX_CLAUSE.search(normalized):
        raise AmbiguousReference(
            "request uses a coordinated or relative clause without exact endpoint binding"
        )
    if _RELATION_DISJUNCTION.search(normalized):
        raise AmbiguousReference("request contains a disjunction between relation constraints")
    if _GROUP_REFERENCE.search(normalized) and _GROUP_ARTICULATION.search(normalized) is None:
        raise AmbiguousReference("group reference cannot be bound exactly in the current IR")
    if _COORDINATED_ARTICULATION_COMMAND.search(normalized):
        raise AmbiguousReference(
            "coordinated articulation commands require an explicit per-object state"
        )
    if _INEXACT_ARTICULATION.search(normalized):
        raise AmbiguousReference(
            "request contains an articulation state without an exact supported fraction"
        )
    reference = _REFERENCE_PRONOUN.search(normalized)
    if reference is not None:
        token = (reference.group("pronoun") or reference.group("chinese")).lower()
        raise AmbiguousReference(
            f"reference {token!r} is not accepted without an explicit noun owner"
        )


def _region_bindings(
    request: str,
    mentions: list[MentionGroup],
) -> dict[str, str]:
    normalized = _normalize(request)
    occurrences: set[tuple[int, int, str]] = set()
    for region, terms in REGION_TERMS.items():
        for term in terms:
            for start, end in _term_spans(normalized, term):
                occurrences.add((start, end, region))
    if not occurrences:
        return {}

    bindings: dict[str, str] = {}
    all_ids = {object_id for mention in mentions for object_id in mention.object_ids}
    if len(all_ids) == 1:
        regions = {item[2] for item in occurrences}
        if len(regions) != 1:
            raise SemanticMismatch("multiple table regions are stated for one object")
        return {next(iter(all_ids)): next(iter(regions))}

    for start, end, region in sorted(occurrences):
        previous = [item for item in mentions if item.end <= start]
        following = [item for item in mentions if item.start >= end]
        owner: MentionGroup | None = None
        if previous:
            separator = normalized[previous[-1].end : start]
            if not re.search(r"[,;.]|\b(?:and|then|while)\b|以及|然后|和|与", separator):
                owner = previous[-1]
        if owner is None and following:
            separator = normalized[end : following[0].start]
            if re.fullmatch(
                r"\s*,?\s*(?:(?:then\s+)?(?:place|put|add|set)\s+)"
                r"(?:(?:a|an|the|one)\s+)?",
                separator,
            ):
                owner = following[0]
        if owner is None:
            distances: list[tuple[int, MentionGroup]] = []
            for mention in mentions:
                if end <= mention.start:
                    distance = mention.start - end
                elif start >= mention.end:
                    distance = start - mention.end
                else:
                    distance = 0
                distances.append((distance, mention))
            nearest_distance = min(item[0] for item in distances)
            nearest = [item[1] for item in distances if item[0] == nearest_distance]
            if len({item.object_ids for item in nearest}) != 1 or nearest_distance > 48:
                raise SemanticMismatch(
                    f"table region {region!r} cannot be bound to one object unambiguously"
                )
            owner = nearest[0]
        for object_id in owner.object_ids:
            existing = bindings.get(object_id)
            if existing is not None and existing != region:
                raise SemanticMismatch(f"multiple table regions are bound to {object_id!r}")
            bindings[object_id] = region
    return bindings


def _nearest_articulation(
    request: str,
    mention: MentionGroup,
    previous_end: int | None,
    next_start: int | None,
) -> dict[str, Any] | None:
    """Bind articulation only when its syntax explicitly owns this mention."""

    normalized = _normalize(request)
    prefix = normalized[max(previous_end or 0, mention.start - 96) : mention.start]
    suffix = normalized[mention.end : min(next_start or len(normalized), mention.end + 96)]
    attributes = "|".join(
        re.escape(term) for term in sorted(_ATTRIBUTE_TOKENS, key=len, reverse=True)
    )
    trailing_attributes = rf"(?:(?:{attributes})\s+)*"
    candidates: list[tuple[str, float]] = []

    def add_state(token: str) -> None:
        normalized_token = token.lower().replace("-", " ")
        if normalized_token in {
            "half open",
            "partially open",
            "halfway open",
            "open halfway",
            "opened halfway",
            "半开",
            "打开一半",
        }:
            candidates.append(("partially_open", 0.5))
        elif normalized_token in {"closed", "shut", "close", "关闭", "闭合", "关上"}:
            candidates.append(("closed", 0.0))
        else:
            candidates.append(("open", 1.0))

    prenominal = re.search(
        rf"(?P<state>half[ -]?open|partially\s+open|halfway\s+open|"
        rf"open|opened|closed|shut)\s+{trailing_attributes}$",
        prefix,
        flags=re.IGNORECASE,
    )
    imperative = re.search(
        rf"(?:^|[.!?]\s*)(?P<state>open|close|shut)\s+"
        rf"(?:a|an|the)\s+{trailing_attributes}$",
        prefix,
        flags=re.IGNORECASE,
    )
    chinese_prefix = re.search(
        r"(?P<state>打开一半|半开|打开|开启|关闭|闭合|关上)\s*(?:的)?\s*$",
        prefix,
    )
    for match in (imperative, prenominal, chinese_prefix):
        if match is not None:
            add_state(match.group("state"))

    postnominal = re.match(
        r"\s*(?:(?:that|which)\s+)?(?:(?:is|was)\s+)?"
        r"(?P<state>half[ -]?open|partially\s+open|halfway\s+open|open(?:ed)?\s+halfway|"
        r"open|opened|closed|shut)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    chinese_post = re.match(
        r"\s*(?:是|为)?\s*(?P<state>打开一半|半开|打开|开启|开着|关闭|闭合|关上)",
        suffix,
    )
    for match in (postnominal, chinese_post):
        if match is not None:
            add_state(match.group("state"))

    percentage_first = re.match(
        r"\s*(?:(?:that|which)\s+)?(?:(?:is|was)\s+)?"
        r"(\d{1,3})\s*%\s*(?:open|opened)\b",
        suffix,
        flags=re.IGNORECASE,
    )
    chinese_percentage = re.match(
        r"\s*(?:是|为)?\s*(?:打开|开启)\s*(\d{1,3})\s*%",
        suffix,
    )
    if (percentage_first is not None or chinese_percentage is not None) and not candidates:
        add_state("open")

    identities = set(candidates)
    if not identities:
        return None
    if len(identities) != 1:
        raise SemanticMismatch(f"articulation state for {mention.object_ids!r} is ambiguous")
    state, fraction = next(iter(identities))

    percentage = re.match(
        r"\s*(?:(?:(?:that|which)\s+)?(?:(?:is|was)\s+)?"
        r"(?:open|opened)\s+)?(?:to|by)?\s*(\d{1,3})\s*%",
        suffix,
        flags=re.IGNORECASE,
    )
    percentage = percentage or percentage_first or chinese_percentage
    if percentage is not None:
        if state != "open":
            raise SemanticMismatch("articulation percentage conflicts with a closed state")
        fraction = float(percentage.group(1)) / 100.0
        if not 0.0 < fraction < 1.0:
            raise SemanticMismatch("articulation percentage must be between 1% and 99%")
        state = "partially_open"

    return {
        "state": state,
        "open_fraction": fraction,
        "joint_selector": "all_movable",
    }


def _shared_group_articulation(
    request: str,
    introductions: list[MentionGroup],
) -> dict[str, Any] | None:
    match = _GROUP_ARTICULATION.search(_normalize(request))
    if match is None:
        return None
    object_ids = [object_id for mention in introductions for object_id in mention.object_ids]
    if len(object_ids) != 2:
        raise AmbiguousReference("'both' articulation must resolve to exactly two object instances")
    state_token = match.group("state").lower()
    if state_token in {"closed", "shut"}:
        state = "closed"
        fraction = 0.0
    else:
        state = "open"
        fraction = 1.0
    return {
        "state": state,
        "open_fraction": fraction,
        "joint_selector": "all_movable",
    }


def validate_object_extraction(
    request: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-check high-confidence object, attribute, and state evidence."""

    reject_unsupported_request_semantics(request)
    resolved_mentions, introductions = _semantic_mentions(request)
    expected = Counter(
        {
            category: sum(item.quantity for item in introductions if item.category == category)
            for category in {item.category for item in introductions}
        }
    )
    actual = Counter(item["category"] for item in objects)
    categories = set(expected) | set(actual)
    for category in sorted(categories):
        if actual[category] != expected[category]:
            raise SemanticMismatch(
                f"object count for {category!r} contradicts the request: "
                f"expected {expected[category]}, got {actual[category]}"
            )

    by_id = {item["object_id"]: item for item in objects}
    color_bindings = _attribute_bindings(request, resolved_mentions, COLOR_TERMS)
    material_bindings = _attribute_bindings(request, resolved_mentions, MATERIAL_TERMS)
    region_bindings = _region_bindings(request, introductions)
    shared_articulation = _shared_group_articulation(request, introductions)
    previous_end: int | None = None
    for index, mention in enumerate(introductions):
        next_start = introductions[index + 1].start if index + 1 < len(introductions) else None
        articulation = _nearest_articulation(
            request,
            mention,
            previous_end,
            next_start,
        )
        if (
            articulation is not None
            and shared_articulation is not None
            and articulation != shared_articulation
        ):
            raise SemanticMismatch("local and group articulation states conflict")
        articulation = shared_articulation or articulation
        for object_id in mention.object_ids:
            expected_fields = {
                "color": color_bindings.get(object_id),
                "material": material_bindings.get(object_id),
                "region": region_bindings.get(object_id, "center"),
                "articulation": articulation,
            }
            candidate = by_id.get(object_id)
            if candidate is None:
                raise SemanticMismatch(f"request object {object_id!r} was not extracted")
            for field, expected_value in expected_fields.items():
                if candidate.get(field) != expected_value:
                    raise SemanticMismatch(
                        f"{field} for {object_id!r} contradicts the request: "
                        f"expected {expected_value!r}, got {candidate.get(field)!r}"
                    )
        previous_end = mention.end

    return {
        "version": SEMANTIC_CHECK_VERSION,
        "known_object_counts": dict(sorted(expected.items())),
        "object_order": [
            object_id for mention in introductions for object_id in mention.object_ids
        ],
        "status": "pass",
    }


def _inverse_relations(normalized: str, first: MentionGroup, second: MentionGroup) -> set[str]:
    between = normalized[first.end : second.start]
    relations: set[str] = set()
    if re.search(r"\b(?:that\s+|which\s+)?contains?\s+(?:a|an|the|one)?\s*$", between):
        relations.add(RelationType.INSIDE.value)
    if re.search(r"\b(?:that\s+|which\s+)?supports?\s+(?:a|an|the|one)?\s*$", between):
        relations.add(RelationType.ON_TOP_OF.value)
    return relations


def _direct_relations(
    normalized: str,
    first: MentionGroup,
    second: MentionGroup,
) -> set[str]:
    relation_text = _mask_table_region_phrases(normalized)
    between = relation_text[first.end : second.start]
    after = relation_text[second.end : min(len(relation_text), second.end + 16)]
    relations: set[str] = set()
    chinese_locative = re.search(r"(?:放|置)?(?:在|于)\s*$|位于\s*$", between) is not None

    if re.search(
        r"\bwithin\s+\d+(?:\.\d+)?\s*(?:m\b|meters?\b)(?:\s+of)?",
        between,
    ):
        relations.add(RelationType.NEAR.value)

    attributes = "|".join(
        re.escape(term) for term in sorted(_ATTRIBUTE_TOKENS, key=len, reverse=True)
    )
    descriptor = rf"(?:(?:a|an|the|one)\s+)?(?:(?:{attributes})\s+)*"
    plain_in = re.fullmatch(
        rf"\s*(?:(?:that|which)\s+)?"
        rf"(?:(?:(?:should|must|can|could|would)\s+be|"
        rf"is|was|sits?|lies?|placed?|put|set)\s+)?"
        rf"in\s+{descriptor}",
        between,
    )
    if (
        re.search(r"\b(?:inside|into)\b|\bwithin\b(?!\s+\d)", between)
        or plain_in
        or re.search(r"放进|装进|放入|置于.*(?:里面|内部)", between)
        or (chinese_locative and re.match(r"\s*(?:里(?:面)?|内(?:部)?)", after))
    ):
        relations.add(RelationType.INSIDE.value)

    explicit_on = re.search(
        r"\b(?:on\s+top\s+of|stacked?\s+(?:on|onto)|onto|atop|upon)\b",
        between,
    )
    plain_on = re.fullmatch(
        rf"\s*(?:(?:that|which)\s+)?"
        rf"(?:(?:(?:should|must|can|could|would)\s+be|"
        rf"is|was|sits?|stands?|lies?|rests?|placed?|put|set)\s+)?"
        rf"on\s+{descriptor}",
        between,
    )
    if (
        explicit_on
        or plain_on
        or (re.search(r"放在|叠在|堆在", between) and re.search(r"上|顶部", after))
        or (chinese_locative and re.match(r"\s*(?:上(?:面|方)?|顶部)", after))
    ):
        relations.add(RelationType.ON_TOP_OF.value)

    if re.search(r"\b(?:(?:to|on)\s+the\s+)?left\s+(?:side\s+)?of\b", between) or re.search(
        r"(?:的)?左边|左侧", after
    ):
        relations.add(RelationType.LEFT_OF.value)
    if re.search(r"\b(?:(?:to|on)\s+the\s+)?right\s+(?:side\s+)?of\b", between) or re.search(
        r"(?:的)?右边|右侧", after
    ):
        relations.add(RelationType.RIGHT_OF.value)
    if re.search(r"\bin\s+front\s+of\b", between) or re.search(r"(?:的)?前方|前面", after):
        relations.add(RelationType.FRONT_OF.value)
    if re.search(r"\bbehind\b", between) or re.search(r"(?:的)?后方|后面", after):
        relations.add(RelationType.BEHIND.value)
    if re.search(r"\b(?:near|next\s+to|beside|close\s+to|adjacent\s+to)\b", between) or re.search(
        r"靠近|旁边|邻近|相邻", f"{between} {after}"
    ):
        relations.add(RelationType.NEAR.value)
    return relations


def _preposed_relations(
    normalized: str,
    target: MentionGroup,
    source: MentionGroup,
) -> set[str]:
    relation_text = _mask_table_region_phrases(normalized)
    connector = relation_text[target.end : source.start]
    if re.fullmatch(r"\s*(?:里|里面|内部)\s*(?:放|置)(?:一个|一只)?\s*", connector):
        return {RelationType.INSIDE.value}
    if re.fullmatch(
        r"\s*(?:上|上面|顶部)\s*(?:放|置)(?:一个|一只)?\s*",
        connector,
    ):
        return {RelationType.ON_TOP_OF.value}

    if not re.fullmatch(
        r"\s*,\s*(?:place|put|add|set|stack)\s+"
        r"(?:(?:a|an|the|one)\s+)?",
        connector,
    ):
        return set()

    attributes = "|".join(
        re.escape(term) for term in sorted(_ATTRIBUTE_TOKENS, key=len, reverse=True)
    )
    descriptor = rf"(?:(?:a|an|the|one)\s+)(?:(?:{attributes})\s+)*"
    prefix = relation_text[max(0, target.start - 96) : target.start]
    patterns = {
        RelationType.INSIDE.value: rf"(?:^|[.!?]\s*)inside\s+{descriptor}$",
        RelationType.ON_TOP_OF.value: (
            rf"(?:^|[.!?]\s*)(?:on\s+top\s+of|atop|upon)\s+{descriptor}$"
        ),
        RelationType.LEFT_OF.value: (rf"(?:^|[.!?]\s*)(?:to\s+the\s+)?left\s+of\s+{descriptor}$"),
        RelationType.RIGHT_OF.value: (rf"(?:^|[.!?]\s*)(?:to\s+the\s+)?right\s+of\s+{descriptor}$"),
        RelationType.FRONT_OF.value: (rf"(?:^|[.!?]\s*)in\s+front\s+of\s+{descriptor}$"),
        RelationType.BEHIND.value: rf"(?:^|[.!?]\s*)behind\s+{descriptor}$",
        RelationType.NEAR.value: (
            rf"(?:^|[.!?]\s*)(?:near|next\s+to|beside|close\s+to|"
            rf"adjacent\s+to)\s+{descriptor}$"
        ),
    }
    return {
        relation
        for relation, pattern in patterns.items()
        if re.search(pattern, prefix, flags=re.IGNORECASE)
    }


def _coordinated_sources(
    normalized: str, mentions: list[MentionGroup], source_index: int
) -> list[MentionGroup]:
    sources = [mentions[source_index]]
    cursor = source_index
    closing_connector = re.compile(
        r"\s*,?\s*(?:and|plus|以及|和|与|、)\s*"
        r"(?:(?:a|an|the|one)\s+|(?:一个|一只))?\s*",
        flags=re.IGNORECASE,
    )
    comma_connector = re.compile(
        r"\s*,\s*(?:(?:a|an|the|one)\s+|(?:一个|一只))?\s*",
        flags=re.IGNORECASE,
    )
    saw_closing_connector = False
    while cursor > 0:
        previous = mentions[cursor - 1]
        current = mentions[cursor]
        connector = normalized[previous.end : current.start]
        if closing_connector.fullmatch(connector):
            saw_closing_connector = True
        elif not (saw_closing_connector and comma_connector.fullmatch(connector)):
            break
        previous_index = cursor - 1
        if previous_index > 0:
            predecessor = mentions[previous_index - 1]
            if _direct_relations(normalized, predecessor, previous) or _preposed_relations(
                normalized, predecessor, previous
            ):
                break
        sources.insert(0, previous)
        cursor -= 1
    return sources


_SHARED_PREDICATE = re.compile(
    r"\s*,?\s*(?:and|then)\s+(?:"
    r"(?:to\s+the\s+)?(?:left|right)\s+of|in\s+front\s+of|behind|"
    r"near|next\s+to|beside|close\s+to|adjacent\s+to|"
    r"within\s+\d+(?:\.\d+)?\s*(?:m\b|meters?\b)(?:\s+of)?|"
    r"(?:at\s+least|minimum)\s+\d+(?:\.\d+)?\s*(?:m\b|meters?\b)"
    r"(?:\s+from)?|inside|into|on\s+top\s+of|atop|upon)\b",
    flags=re.IGNORECASE,
)
_PLAIN_COORDINATION = re.compile(
    r"\s*,?\s*(?:and|plus)\s*(?:(?:a|an|the|one)\s+)?",
    flags=re.IGNORECASE,
)


def _continues_previous_predicate(
    normalized: str, first: MentionGroup, second: MentionGroup
) -> bool:
    relation_text = _mask_table_region_phrases(normalized)
    return _SHARED_PREDICATE.match(relation_text[first.end : second.start]) is not None


def _expected_relations(
    request: str,
    objects: list[dict[str, Any]],
) -> tuple[set[tuple[str, str, str]], list[MentionGroup]]:
    normalized = _normalize(request)
    mentions, _ = _semantic_mentions(request)
    mentioned_ids = {object_id for item in mentions for object_id in item.object_ids}
    object_ids = {item["object_id"] for item in objects}
    if mentioned_ids != object_ids:
        raise SemanticMismatch(
            "relation-stage objects no longer match request-derived object instances"
        )

    expected: set[tuple[str, str, str]] = set()
    last_direct_sources: tuple[str, ...] = ()
    for index, (first, second) in enumerate(zip(mentions, mentions[1:], strict=False)):
        if set(first.object_ids) == set(second.object_ids):
            if (
                _direct_relations(normalized, first, second)
                or _inverse_relations(normalized, first, second)
                or _preposed_relations(normalized, first, second)
            ):
                raise AmbiguousReference(
                    "request contains an explicit relation from an object to itself"
                )
            continue
        inverse = _inverse_relations(normalized, first, second)
        if inverse:
            if first.quantity != 1:
                raise AmbiguousReference("inverse relation target is an unresolved plural group")
            target = first.object_ids[0]
            for source in second.object_ids:
                if source != target:
                    for relation in inverse:
                        expected.add((relation, source, target))
            last_direct_sources = ()
            continue
        direct = _direct_relations(normalized, first, second)
        if direct:
            if second.quantity != 1:
                raise AmbiguousReference("relation target is an unresolved plural object group")
            target = second.object_ids[0]
            if _continues_previous_predicate(normalized, first, second):
                if not last_direct_sources:
                    raise AmbiguousReference("coordinated relation has no exact shared subject")
                source_ids = last_direct_sources
            else:
                source_ids = tuple(
                    source
                    for group in _coordinated_sources(normalized, mentions, index)
                    for source in group.object_ids
                )
            for source in source_ids:
                if source != target:
                    for relation in direct:
                        expected.add((relation, source, target))
            last_direct_sources = source_ids
        else:
            connector = normalized[first.end : second.start]
            previous_direct = (
                _direct_relations(normalized, mentions[index - 1], first) if index > 0 else set()
            )
            next_direct = (
                _direct_relations(normalized, second, mentions[index + 2])
                if index + 2 < len(mentions)
                else set()
            )
            second_has_table_support = re.match(
                r"\s+(?:(?:is|was|sits?|stands?|lies?|rests?)\s+)?"
                r"on\s+(?:the\s+)?(?:table|tabletop)\b",
                normalized[second.end :],
            )
            if (
                previous_direct
                and _PLAIN_COORDINATION.fullmatch(connector)
                and not next_direct
                and second_has_table_support is None
            ):
                raise AmbiguousReference(
                    "coordinated relation targets cannot be bound to exact endpoints"
                )
            relation_text = _mask_table_region_phrases(connector)
            without_table_support = re.sub(
                r"\bon\s+(?:the\s+)?(?:table|tabletop)\b",
                " ",
                relation_text,
            )
            has_region = any(
                _term_present(connector, term) for terms in REGION_TERMS.values() for term in terms
            )
            if not has_region and (
                re.search(r"\bon\b", without_table_support)
                or re.search(r"\bin\b(?!\s+front\s+of)", relation_text)
            ):
                raise SemanticMismatch(
                    "support relation cue cannot be bound to exact object endpoints"
                )
            last_direct_sources = ()

        preposed = _preposed_relations(normalized, first, second)
        if preposed:
            if first.quantity != 1:
                raise AmbiguousReference("preposed relation target is an unresolved plural group")
            target = first.object_ids[0]
            for source in second.object_ids:
                if source != target:
                    for relation in preposed:
                        expected.add((relation, source, target))
    return expected, mentions


def _cue_types(request: str) -> set[str]:
    normalized = _mask_table_region_phrases(_normalize(request))
    patterns = {
        RelationType.INSIDE.value: (
            r"\b(?:inside|into|contains?)\b|\bwithin\b(?!\s+\d)|"
            r"放进|装进|放入|里面|内部"
        ),
        RelationType.ON_TOP_OF.value: (
            r"\b(?:on\s+top\s+of|stacked?\s+(?:on|onto)|onto|atop|upon|"
            r"supports?|rests?\s+on)\b|叠在|堆在"
        ),
        RelationType.LEFT_OF.value: (r"\b(?:(?:to|on)\s+the\s+)?left\s+(?:side\s+)?of\b|左边|左侧"),
        RelationType.RIGHT_OF.value: (
            r"\b(?:(?:to|on)\s+the\s+)?right\s+(?:side\s+)?of\b|右边|右侧"
        ),
        RelationType.FRONT_OF.value: r"\bin\s+front\s+of\b|前方|前面",
        RelationType.BEHIND.value: r"\bbehind\b|后方|后面",
        RelationType.NEAR.value: (
            r"\b(?:near|next\s+to|beside|close\s+to|adjacent\s+to)\b"
            r"|\bwithin\s+\d+(?:\.\d+)?\s*(?:m\b|meters?\b)|靠近|旁边|邻近|相邻"
        ),
        RelationType.DISTANCE_AT_LEAST.value: r"\b(?:at\s+least|minimum)\b|至少",
    }
    return {
        relation
        for relation, pattern in patterns.items()
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    }


_MIN_DISTANCE_CUE = re.compile(
    r"(?:at\s+least|minimum|至少(?:相距|距离)?)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:m\b|meters?\b|米)",
    flags=re.IGNORECASE,
)
_MAX_DISTANCE_CUE = re.compile(
    r"\bwithin\s+(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:m\b|meters?\b)(?:\s+of)?",
    flags=re.IGNORECASE,
)


def _distinct_single_mentions(
    mentions: list[MentionGroup],
) -> list[MentionGroup]:
    result: list[MentionGroup] = []
    seen: set[str] = set()
    for mention in mentions:
        if len(mention.object_ids) != 1:
            raise SemanticMismatch(
                "distance_at_least cannot target an unresolved plural object group"
            )
        object_id = mention.object_ids[0]
        if object_id not in seen:
            result.append(mention)
            seen.add(object_id)
    return result


def _distance_expectations(
    request: str,
    mentions: list[MentionGroup],
    pattern: re.Pattern[str],
) -> list[tuple[str, str, float]]:
    normalized = _normalize(request)
    expectations: list[tuple[str, str, float]] = []
    previous_source: str | None = None
    for match in pattern.finditer(normalized):
        before = _distinct_single_mentions([item for item in mentions if item.end <= match.start()])
        after = _distinct_single_mentions([item for item in mentions if item.start >= match.end()])
        shared_subject = False
        if previous_source is not None and before:
            bridge = normalized[before[-1].end : match.start()]
            shared_subject = bool(re.fullmatch(r"\s*,?\s*(?:and|then|以及|然后)\s*", bridge))
        separator = normalized[match.end() : after[0].start] if after else ""
        if shared_subject:
            if not after:
                raise SemanticMismatch("coordinated distance cue has no explicit target object")
            source = previous_source
            target = after[0].object_ids[0]
        elif pattern is _MAX_DISTANCE_CUE and before and after:
            source = before[-1].object_ids[0]
            target = after[0].object_ids[0]
        elif before and after and re.search(r"\bfrom\b|与|和", separator):
            source = before[-1].object_ids[0]
            target = after[0].object_ids[0]
        elif len(before) >= 2:
            source = before[-2].object_ids[0]
            target = before[-1].object_ids[0]
        elif len(after) >= 2:
            source = after[0].object_ids[0]
            target = after[1].object_ids[0]
        elif before and after:
            source = before[-1].object_ids[0]
            target = after[0].object_ids[0]
        else:
            raise SemanticMismatch("distance cue cannot be bound to two explicit object instances")
        if source == target:
            raise SemanticMismatch("distance cue resolves to the same object on both endpoints")
        expectations.append((source, target, float(match.group("value"))))
        previous_source = source
    return expectations


def validate_relation_extraction(
    request: str,
    relations: list[dict[str, Any]],
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject missing/reversed direct cues and unsupported invented relations."""

    reject_unsupported_request_semantics(request)
    expected, mentions = _expected_relations(request, objects)
    actual = {(item["relation"], item["source"], item["target"]) for item in relations}
    missing = sorted(expected - actual)
    if missing:
        raise SemanticMismatch(f"explicit request relations are missing or reversed: {missing}")

    cue_types = _cue_types(request)
    expected_types = {item[0] for item in expected}
    unbound_cues = cue_types - expected_types - {RelationType.DISTANCE_AT_LEAST.value}
    if unbound_cues:
        raise SemanticMismatch(
            f"relation cues cannot be bound to exact request endpoints: {sorted(unbound_cues)}"
        )
    for relation, source, target in sorted(actual):
        if relation in {RelationType.ON_TABLE.value, RelationType.DISTANCE_AT_LEAST.value}:
            continue
        if (relation, source, target) not in expected:
            raise SemanticMismatch(
                f"relation {(relation, source, target)!r} is not an exact request relation"
            )

    distance_expectations = _distance_expectations(request, mentions, _MIN_DISTANCE_CUE)
    if RelationType.DISTANCE_AT_LEAST.value in cue_types and not distance_expectations:
        raise SemanticMismatch(
            "distance_at_least cue must include a supported numeric distance in meters"
        )
    remaining_distances = [
        item for item in relations if item["relation"] == RelationType.DISTANCE_AT_LEAST.value
    ]
    for source, target, requested in distance_expectations:
        matching = [
            item
            for item in remaining_distances
            if item["source"] == source
            and item["target"] == target
            and math.isclose(
                item.get("min_distance_m", math.nan),
                requested,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if len(matching) != 1:
            raise SemanticMismatch(
                "distance_at_least does not preserve the explicit distance and endpoint order"
            )
        remaining_distances.remove(matching[0])
    if remaining_distances:
        raise SemanticMismatch(
            "distance_at_least contains an unstated distance, endpoint, or duplicate relation"
        )

    max_distance_expectations = _distance_expectations(request, mentions, _MAX_DISTANCE_CUE)
    remaining_near = [item for item in relations if item["relation"] == RelationType.NEAR.value]
    explicit_near_pairs: set[tuple[str, str]] = set()
    for source, target, requested in max_distance_expectations:
        explicit_near_pairs.add((source, target))
        matching = [
            item
            for item in remaining_near
            if item["source"] == source
            and item["target"] == target
            and math.isclose(
                item.get("max_distance_m", math.nan),
                requested,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if len(matching) != 1:
            raise SemanticMismatch(
                "near does not preserve the explicit maximum distance and endpoint order"
            )
        remaining_near.remove(matching[0])
    for item in remaining_near:
        pair = (item["source"], item["target"])
        if pair in explicit_near_pairs or not math.isclose(
            item.get("max_distance_m", math.nan),
            0.25,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise SemanticMismatch(
                "near contains an unstated maximum distance or duplicate relation"
            )

    return {
        "version": SEMANTIC_CHECK_VERSION,
        "direct_relations": [list(item) for item in sorted(expected)],
        "distance_relations": [list(item) for item in distance_expectations],
        "max_distance_relations": [list(item) for item in max_distance_expectations],
        "status": "pass",
    }

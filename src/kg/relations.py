# -*- coding: utf-8 -*-
"""High-precision relation patterns.

A relation is emitted only when a pattern word sits between two adjacent
mentions (at most 40 characters apart). A keyword elsewhere in the
sentence does not count. No edge is stored without mention ids.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.kg.entries import Entry
from src.kg.mentions import Mention

_PATTERNS = (
    ("married_to", re.compile(r"\b(wife|husband|married)\b", re.I), "PER", "PER"),
    ("engaged_to", re.compile(r"\b(engaged|engagement|fiance|fiancé|betrothed)\b", re.I), "PER", "PER"),
    ("patient_of", re.compile(r"\bpatient\b", re.I), "PER", "PER"),
    ("employed_by", re.compile(r"\b(employer|clerk)\b", re.I), "PER", "PER"),
    ("resides_at", re.compile(r"\b(house at|lives at)\b", re.I), "PER", "FAC"),
)

DOCTOR_HONORIFICS = {"dr", "doctor", "professor"}


def find_relations(entries: list[Entry], mentions: list[Mention]) -> list[dict]:
    by_sent: dict[tuple, list[Mention]] = defaultdict(list)
    for mention in mentions:
        if mention.entity_id and mention.mention_class in {"PROP", "NOM"}:
            by_sent[(mention.entry_index, mention.sent_id)].append(mention)
    text_of = {entry.entry_index: entry.text for entry in entries}
    edges: dict[tuple, dict] = {}

    for (entry_index, _sent_id), group in by_sent.items():
        text = text_of[entry_index]
        ordered = sorted(group, key=lambda m: m.start)
        seen_pair = set()
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if left.entity_id == right.entity_id:
                    continue
                gap = text[left.end:right.start]
                # The pattern word has to sit between the two spans, and the
                # spans have to be adjacent. A nearby "married" or "patient"
                # is not enough.
                if len(gap) > 40 or not gap.strip():
                    continue
                pair_key = (left.entity_id, right.entity_id, gap)
                if pair_key in seen_pair:
                    continue
                seen_pair.add(pair_key)
                for rel, pattern, left_type, right_type in _PATTERNS:
                    if pattern.search(gap) is None:
                        continue
                    if left_type == right_type:
                        if left.entity_type != left_type or right.entity_type != right_type:
                            continue
                        source, target = left.entity_id, right.entity_id
                        if rel == "patient_of":
                            source, target = _patient_direction(left, right, ordered)
                            if source is None:
                                continue
                        a, b = left, right
                    else:
                        if {left.entity_type, right.entity_type} != {left_type, right_type}:
                            continue
                        person = left if left.entity_type == "PER" else right
                        place = right if person is left else left
                        if person.entity_type != left_type or place.entity_type != right_type:
                            continue
                        source, target = person.entity_id, place.entity_id
                        a, b = person, place
                    key = (rel, source, target)
                    slot = edges.setdefault(key, {
                        "type": rel,
                        "source": source,
                        "target": target,
                        "evidence": [],
                    })
                    slot["evidence"].append(a.mention_id)
                    slot["evidence"].append(b.mention_id)
    for edge in edges.values():
        edge["evidence"] = sorted(set(edge["evidence"]))
    return list(edges.values())


def _patient_direction(a: Mention, b: Mention, group: list[Mention]) -> tuple[str | None, str | None]:
    """Point patient_of from the non-doctor to the doctor, when one side is marked."""
    from src.kg.resolve import parse_name

    def is_doctor(mention: Mention) -> bool:
        if parse_name(mention.surface).honorific in DOCTOR_HONORIFICS:
            return True
        return any(
            other.entity_id == mention.entity_id and parse_name(other.surface).honorific in DOCTOR_HONORIFICS
            for other in group
        )

    a_doc, b_doc = is_doctor(a), is_doctor(b)
    if a_doc == b_doc:
        return None, None
    if a_doc:
        return b.entity_id, a.entity_id
    return a.entity_id, b.entity_id

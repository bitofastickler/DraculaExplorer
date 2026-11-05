# -*- coding: utf-8 -*-
"""Relations between resolved entities, two ways.

``find_relations``: high-precision patterns. A relation is emitted only
when a pattern word sits between two adjacent mentions (at most 40
characters apart). The novel rarely writes it that way, so this usually
finds nothing.

``find_stated_relations``: relations known from reading the book, each
kept only when a sentence in the text supports it. No edge is stored
without mention ids.
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
                        "method": "pattern",
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


# Sentence ends: . ! ? (plus closing quotes) followed by a space, but not
# after an abbreviation. The entry text has no line breaks.
_SENT_END = re.compile(r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)[.!?][\"')\]_]*\s+")
_FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
CUE_WINDOW = 40
_SECTION = re.compile(r"\*(?:\s\*){4}|_(?=[A-Z])[^_]{2,150}_")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _SENT_END.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def find_stated_relations(
    entries: list[Entry],
    mentions: list[Mention],
    stated: list[dict],
    narrator_ids: dict[str, str] | None = None,
) -> list[dict]:
    """Keep the known relations that a sentence in the text supports.

    ``stated`` rows come from ``data/gold/relations.yml``. A row becomes an
    edge only when one sentence mentions both entities and a cue word sits
    within ``CUE_WINDOW`` characters of a mention of each ("to Jonathan, my
    husband"). A cue elsewhere in a long sentence is usually about someone
    else. A first-person pronoun stands for the narrator only outside
    dialogue: the sentence has no quote mark and an even number of quote
    marks comes before it in the entry. It is not trusted in a section
    whose narration calls the narrator by title ("Dr. Seward was out on
    business"): that section is another writer's pages. The quote
    kept is the sentence where the cue sits closest to both names.
    """
    from bisect import bisect_right

    from src.kg.resolve import parse_name

    narrator_ids = narrator_ids or {}
    cue_of = {
        id(row): re.compile(r"\b(?:" + "|".join(re.escape(c) for c in row["cues"]) + r")", re.I)
        for row in stated
    }
    by_entry: dict[int, list[Mention]] = defaultdict(list)
    for mention in mentions:
        by_entry[mention.entry_index].append(mention)

    edges: dict[tuple, dict] = {}
    for entry in entries:
        group = by_entry.get(entry.entry_index)
        if not group:
            continue
        text = entry.text
        spans = _sentence_spans(text)
        starts = [s for s, _e in spans]
        in_sent: dict[int, list[Mention]] = defaultdict(list)
        for mention in group:
            in_sent[bisect_right(starts, mention.start) - 1].append(mention)
        narrator = narrator_ids.get(entry.narrator or "")

        def narrated(pos: int) -> bool:
            return text.count('"', 0, pos) % 2 == 0

        # Sections start at "* * * * *" or an italic heading such as
        # "_Letter, Sister Agatha, ..._". A section whose narration calls the
        # narrator by title belongs to someone else.
        cuts = [0] + [b.start() for b in _SECTION.finditer(text)]
        untrusted = {
            bisect_right(cuts, m.start) - 1
            for m in group
            if m.entity_id == narrator and m.mention_class == "PROP"
            and parse_name(m.surface).honorific and narrated(m.start)
        } if narrator else set()

        for k, ms in in_sent.items():
            s_start, s_end = spans[k]
            sentence = text[s_start:s_end]
            present: dict[str, list[Mention]] = defaultdict(list)
            for m in ms:
                if m.entity_id and m.mention_class in {"PROP", "NOM"}:
                    present[m.entity_id].append(m)
            if (
                narrator and '"' not in sentence and narrated(s_start)
                and bisect_right(cuts, s_start) - 1 not in untrusted
            ):
                for m in ms:
                    if m.mention_class == "PRON" and m.surface.lower() in _FIRST_PERSON:
                        present[narrator].append(m)
            if len(present) < 2:
                continue
            for row in stated:
                src, dst = row["source"], row["target"]
                if src not in present or dst not in present:
                    continue
                best = None
                for cue in cue_of[id(row)].finditer(sentence):
                    at, end = s_start + cue.start(), s_start + cue.end()

                    def gap(group: list[Mention]) -> int:
                        return min(max(m.start - end, at - m.end, 0) for m in group)

                    d_src, d_dst = gap(present[src]), gap(present[dst])
                    if d_src <= CUE_WINDOW and d_dst <= CUE_WINDOW and (best is None or d_src + d_dst < best):
                        best = d_src + d_dst
                if best is None:
                    continue
                key = (row["rel"], src, dst)
                slot = edges.get(key)
                if slot is None:
                    slot = edges[key] = {
                        "type": row["rel"],
                        "source": src,
                        "target": dst,
                        "method": "stated",
                        "evidence": [],
                        "support": 0,
                        "_best": None,
                    }
                slot["support"] += 1
                slot["evidence"].extend(m.mention_id for m in present[src] + present[dst])
                if slot["_best"] is None or best < slot["_best"]:
                    slot["_best"] = best
                    slot["entry_index"] = entry.entry_index
                    slot["quote"] = " ".join(sentence.split())[:300]
    for edge in edges.values():
        edge["evidence"] = sorted(set(edge["evidence"]))
        del edge["_best"]
    return list(edges.values())


def _plain(text: str) -> str:
    return " ".join(text.replace("_", "").split())


def _find_quote(plain: list[tuple[Entry, str]], quote: str) -> Entry | None:
    parts = [_plain(p) for p in quote.split(" ... ")]
    for entry, text in plain:
        at = 0
        for part in parts:
            at = text.find(part, at)
            if at < 0:
                break
            at += len(part)
        else:
            return entry
    return None


def find_quoted_relations(entries: list[Entry], stated: list[dict]) -> list[dict]:
    """Edges for rows of ``relations.yml`` that carry a ``quote``.

    A quote is copied from the novel. Its parts, joined by " ... ", must
    appear in that order in one document, ignoring italics underscores and
    whitespace. ``quote`` may be a list when the relationship is plain from
    context across documents (the Demeter's arrival, the Count's boxes).
    A quote that is not found raises ``ValueError``: the file is
    hand-written, so a typo should stop the build rather than drop the edge.
    """
    plain = [(entry, _plain(entry.text)) for entry in entries]
    edges = []
    for row in stated:
        quotes = row.get("quote")
        if not quotes:
            continue
        if isinstance(quotes, str):
            quotes = [quotes]
        found = []
        for quote in quotes:
            hit = _find_quote(plain, quote)
            if hit is None:
                raise ValueError(f"quote for {row['rel']} {row['source']} -> {row['target']} not found: {quote[:80]}")
            found.append({"text": _plain(quote), "entry_index": hit.entry_index})
        edges.append({
            "type": row["rel"],
            "source": row["source"],
            "target": row["target"],
            "method": "quoted",
            "evidence": [],
            "support": len(found),
            "entry_index": found[0]["entry_index"],
            "quote": found[0]["text"],
            "quotes": found,
        })
    return edges

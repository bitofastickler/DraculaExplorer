# -*- coding: utf-8 -*-
"""Mention detection: spaCy spans plus a small honorific gazetteer.

The gazetteer exists because the small English model splits or drops
forms this book depends on (``Van Helsing``, ``the Count``, ``Castle Dracula``).
On an overlap, the longer span wins. ``Castle Dracula`` is forced to FAC
so it cannot cluster with the person.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.kg.entries import Entry

SPACY_MAP = {
    "PERSON": "PER",
    "FAC": "FAC",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "LOC",
}

PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
    "we", "us", "our", "ours",
    "you", "your", "yours", "yourself",
}

# Only abbreviations take a period. "the Count. Then he" is a sentence
# break, not a man called Count Then.
_HONOR = (
    r"(?:(?:Dr|Mr|Mrs)\.?|Count|Countess|Doctor|Professor|Lord|Miss|Captain|Madam|Sir)"
)
_NAME = r"(?:(?:Van|Von|De|Di)\s+)?[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+){0,3}"
HONOR_NAME = re.compile(rf"\b{_HONOR}\s+{_NAME}\b")
VAN_HELSING = re.compile(r"\b(?:Abraham\s+)?Van\s+Helsing\b")
CASTLE = re.compile(r"\bCastle\s+Dracula\b")
DEMETER = re.compile(r"\bthe\s+Demeter\b|\bDemeter\b")
TITLE_NOM = re.compile(r"\bthe\s+(Count|Countess|Professor|Doctor|Captain)\b")
_DIRTY = re.compile(r"[:,\"“”]|--|---|&")
# A period after a word that is not an abbreviation ends a sentence.
_SENT_BREAK = re.compile(r"\b(?!(?:Dr|Mr|Mrs|St)\.)[A-Za-z]{2,}\.\s")
_STOP = {"on", "of", "the", "and", "to", "in", "for", "with", "from", "by", "at", "a", "an"}
# spaCy entity edges with these tags are not part of the name
# ("Happily Van Helsing", "Jonathan knelt", "Tell Arthur").
_EDGE_POS = {"ADV", "VERB", "AUX", "ADP", "CCONJ", "SCONJ", "PRON", "DET", "PART", "INTJ", "PUNCT", "NUM"}


def _usable_prop(surface: str) -> bool:
    """Drop spans that swallowed a verb, a quote, or a line-break artifact.

    The small NER model sometimes returns ``gravely:-- "Madam Mina``.
    Those spans invent surnames and block real merges. A comma inside a
    span ("Van Helsing, Art") joins two names; the apposition rule in
    ``resolve`` handles that pattern between separate spans.
    """
    if _DIRTY.search(surface) or _SENT_BREAK.search(surface + " "):
        return False
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", surface)]
    if not tokens or len(tokens) > 5:
        return False
    if any(t in _STOP for t in tokens):
        return False
    return True

# Longer patterns run first so "Castle Dracula" beats a bare "Dracula" inside it.
GAZETTEER = (
    ("FAC", "PROP", CASTLE),
    ("VEH", "PROP", DEMETER),
    ("PER", "PROP", HONOR_NAME),
    ("PER", "PROP", VAN_HELSING),
    ("PER", "NOM", TITLE_NOM),
)


@dataclass
class Mention:
    mention_id: str
    entry_index: int
    chapter_number: int | None
    sent_id: int
    start: int
    end: int
    surface: str
    entity_type: str
    mention_class: str
    extractor: str
    entity_id: str | None = None
    link_method: str | None = None
    link_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "mention_id": self.mention_id,
            "entry_index": self.entry_index,
            "chapter_number": self.chapter_number,
            "sent_id": self.sent_id,
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "entity_type": self.entity_type,
            "mention_class": self.mention_class,
            "extractor": self.extractor,
            "entity_id": self.entity_id,
            "link_method": self.link_method,
            "link_score": self.link_score,
        }


@dataclass
class _Span:
    start: int
    end: int
    surface: str
    entity_type: str
    mention_class: str
    extractor: str
    sent_id: int = 0
    proper: bool = False


def _overlaps(a: _Span, b: _Span) -> bool:
    return a.start < b.end and b.start < a.end


def _prefer(spans: Sequence[_Span]) -> list[_Span]:
    """Keep the longest span in each overlap. Gazetteer beats spaCy on a tie."""
    ranked = sorted(spans, key=lambda s: (-(s.end - s.start), 0 if s.extractor == "gazetteer" else 1, s.start))
    kept: list[_Span] = []
    for span in ranked:
        if any(_overlaps(span, prev) for prev in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s.start)
    return kept


def _sent_id(sents, start: int) -> int:
    for i, sent in enumerate(sents):
        if sent.start_char <= start < sent.end_char:
            return i
    return 0


def _trim_edges(ent):
    """Drop leading and trailing tokens that are not part of a name."""
    start, end = ent.start, ent.end
    doc = ent.doc
    while start < end and doc[start].pos_ in _EDGE_POS:
        start += 1
    while end > start and doc[end - 1].pos_ in _EDGE_POS:
        end -= 1
    if start == end:
        return None
    return doc[start:end]


def _class_from_doc(doc, start: int, end: int, fallback: str) -> str:
    fragment = doc.char_span(start, end, alignment_mode="expand")
    if fragment is None:
        return fallback
    if any(tok.pos_ == "PROPN" for tok in fragment):
        return "PROP"
    return fallback


def extract_mentions(entries: Iterable[Entry], nlp=None) -> list[Mention]:
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    mentions: list[Mention] = []
    for entry in entries:
        doc = nlp(entry.text)
        sents = list(doc.sents)
        spans: list[_Span] = []

        for ent in doc.ents:
            etype = SPACY_MAP.get(ent.label_)
            if etype is None:
                continue
            ent = _trim_edges(ent)
            if ent is None:
                continue
            mention_class = "PROP" if any(t.pos_ == "PROPN" for t in ent) else "NOM"
            spans.append(_Span(ent.start_char, ent.end_char, ent.text, etype, mention_class, "spacy", _sent_id(sents, ent.start_char)))

        for tok in doc:
            if tok.pos_ == "PRON" and tok.lower_ in PRONOUNS:
                spans.append(_Span(tok.idx, tok.idx + len(tok.text), tok.text, "PER", "PRON", "spacy", _sent_id(sents, tok.idx)))

        for etype, mention_class, pattern in GAZETTEER:
            for match in pattern.finditer(entry.text):
                surface = match.group(0)
                kind = mention_class
                if kind != "NOM":
                    kind = _class_from_doc(doc, match.start(), match.end(), mention_class)
                spans.append(_Span(match.start(), match.end(), surface, etype, kind, "gazetteer", _sent_id(sents, match.start())))

        spans = [s for s in spans if s.mention_class != "PROP" or _usable_prop(s.surface)]
        for span in _prefer(spans):
            surface = " ".join(span.surface.split())
            if re.search(r"\b(journal|diary)\b", surface, flags=re.I):
                continue
            etype = span.entity_type
            if re.search(r"\bCastle\s+Dracula\b", surface):
                etype = "FAC"
                surface_norm_type = "FAC"
            else:
                surface_norm_type = etype
            if re.fullmatch(r"the\s+Demeter", surface, flags=re.I) or surface == "Demeter":
                surface_norm_type = "VEH"
                etype = "VEH"
            mentions.append(
                Mention(
                    mention_id=f"m{entry.entry_index:04d}-{span.start}-{span.end}",
                    entry_index=entry.entry_index,
                    chapter_number=entry.chapter_number,
                    sent_id=span.sent_id,
                    start=span.start,
                    end=span.end,
                    surface=entry.text[span.start:span.end],
                    entity_type=etype if surface_norm_type == etype else surface_norm_type,
                    mention_class=span.mention_class,
                    extractor=span.extractor,
                )
            )
    return mentions


def load_mentions(path) -> list[Mention]:
    """Read detected mentions back from ``mentions.jsonl`` with links cleared.

    Lets the resolver be rerun without spaCy. Types already carry
    ``apply_alias_types``, which is safe to apply again.
    """
    import json
    from pathlib import Path

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.update(entity_id=None, link_method=None, link_score=None)
        rows.append(Mention(**row))
    return rows


def drop_surfaces(mentions: list[Mention], surfaces: Iterable[str]) -> list[Mention]:
    """Remove proper mentions whose surface is a known non-entity.

    The small NER model tags words such as "Telegram" or "Vampire" as
    names. The list lives in ``configs/kg.yml`` under ``not_entities``.
    Matching ignores case and a trailing possessive.
    """
    stop = {s.lower().strip() for s in surfaces}
    if not stop:
        return mentions

    def key(surface: str) -> str:
        text = " ".join(surface.lower().replace("’", "'").split())
        return text[:-2] if text.endswith("'s") else text

    return [m for m in mentions if m.mention_class != "PROP" or key(m.surface) not in stop]


def mentions_in(entry: Entry, mentions: Sequence[Mention]) -> list[Mention]:
    return [m for m in mentions if m.entry_index == entry.entry_index]


def apply_alias_types(mentions: list[Mention], gold_entities: list[dict]) -> None:
    """Set the entity type from the cast and place list when the surface matches.

    ``en_core_web_sm`` tags ``Mina`` and ``Bistritz`` as organizations. The
    alias list is closed-world knowledge about this book, same as the
    honorific gazetteer. A surface that matches two types is left alone.
    """
    from src.kg.resolve import parse_name

    lookup: dict[str, set[str]] = {}
    for entity in gold_entities:
        etype = entity.get("entity_type")
        if not etype:
            continue
        for alias in entity.get("aliases") or []:
            lookup.setdefault(alias.lower().strip(), set()).add(etype)
    for mention in mentions:
        if mention.mention_class == "PRON":
            continue
        part = parse_name(mention.surface)
        forms = {part.token_str}
        if part.honorific:
            forms.add(f"{part.honorific} {part.token_str}".strip())
        forms.add(" ".join(mention.surface.lower().split()))
        types: set[str] = set()
        for form in forms:
            types |= lookup.get(form, set())
        if len(types) == 1:
            mention.entity_type = types.pop()

# -*- coding: utf-8 -*-
"""One document per diary entry, rebuilt from overlapping RAG chunks.

``load_corpus`` joins chunk text with newlines and keeps the overlap.
That is fine for TF-IDF. Mention offsets are not. Consecutive chunks in
``dracula_ascii_rag.json`` share about 150 characters; ``deoverlap``
keeps the first chunk and appends only the new tail of each later chunk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Entry:
    entry_index: int
    chapter_number: int | None
    section_title: str | None
    narrator: str | None
    date_str: str | None
    date_iso: str | None
    kind: str
    text: str
    n_chunks: int
    source_entry: int | None = None


def overlap_len(left: str, right: str, min_n: int = 20, max_n: int = 400) -> int:
    """Length of the longest suffix of ``left`` that is a prefix of ``right``."""
    limit = min(max_n, len(left), len(right))
    for n in range(limit, min_n - 1, -1):
        if left[-n:] == right[:n]:
            return n
    return 0


def deoverlap(chunks: Sequence[str], min_n: int = 20, max_n: int = 400) -> str:
    if not chunks:
        return ""
    out = chunks[0]
    for nxt in chunks[1:]:
        n = overlap_len(out, nxt, min_n=min_n, max_n=max_n)
        out = out + nxt[n:]
    return out


def naive_join(chunks: Sequence[str]) -> str:
    """The join ``load_corpus`` uses. Kept here so tests can show the duplication."""
    return "\n".join(chunks)


def document_kind(section_title: str | None, narrator: str | None) -> str:
    title = section_title or ""
    narr = narrator or ""
    if narr == "Lucy & Mina (Letters)" or title.startswith("Letters"):
        return "letter"
    if narr == "Newspaper cutting" or "Dailygraph" in title or title.startswith("Cutting"):
        return "newspaper"
    if "Phonograph" in title:
        return "phonograph"
    if "Diary" in title:
        return "diary"
    if "Journal" in title:
        return "journal"
    return "document"


def _clean(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def load_entries(path: str | Path) -> list[Entry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["entries"] if isinstance(raw, dict) else raw
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["entry_index"]), []).append(row)

    entries: list[Entry] = []
    for idx in sorted(groups):
        chunks = sorted(groups[idx], key=lambda r: int(r.get("chunk_index") or 0))
        head = chunks[0]
        title = _clean(head.get("section_title"))
        narrator = _clean(head.get("narrator"))
        chapter = head.get("chapter_number")
        entries.append(
            Entry(
                entry_index=idx,
                chapter_number=int(chapter) if chapter is not None else None,
                section_title=title,
                narrator=narrator,
                date_str=_clean(head.get("date_str")),
                date_iso=_clean(head.get("date_iso")),
                kind=document_kind(title, narrator),
                # The file wraps lines mid-sentence. Collapse that whitespace
                # so "Mrs.\nHarker" is one name. Offsets refer to this string.
                text=re.sub(r"\s+", " ", deoverlap([c.get("text") or "" for c in chunks])).strip(),
                n_chunks=len(chunks),
            )
        )
    return entries


# Section headings inside the text. The source file attaches each heading
# to the end of the entry before it, and labels whole runs of entries with
# the chapter's first narrator, so chapters 9-10 read as "Mina Murray"
# although they are Seward's diary and letters.
# Each heading is matched by its own opening words, not by pairing
# underscores: a stray "_" elsewhere would shift every pair after it.
_OWNED = re.compile(
    r"_((?:Dr\. )?[A-Z][a-z]+(?: [A-Z][a-z]+)?)'s (Phonograph )?(Journal|Diary|Memorandum)(?:--continued)?\.?_"
)
_UPPER = re.compile(r"\b((?:DR\. )?[A-Z]+(?: [A-Z]+)?)'S (JOURNAL|DIARY)(?:\.|--_continued_)")
_SENT = re.compile(
    r"_(Letter|Telegram|Report|Note|Memorandum)\b(?: \(by hand\))?,? (?:from |left by |by )?([A-Z][^_]*?)(?: to [^_]+?)?\.?_"
)
_PAPER = re.compile(r'_"The ([A-Z][A-Za-z ]+?)," \d+ [A-Z][a-z]+\.?_')
SECTION_KIND = {"journal": "journal", "diary": "diary", "memorandum": "memorandum",
                "letter": "letter", "telegram": "telegram", "report": "letter", "note": "letter"}


@dataclass(frozen=True)
class Heading:
    start: int
    end: int
    title: str
    writer: str
    kind: str


def _writer_label(name: str, labels: dict[str, str]) -> str:
    name = re.sub(r"^(?:Messrs\.|Hon\.|Mr\.|Miss) ", "", name.strip())
    name = re.split(r" in (?:his|her) ", name)[0]
    if "Paterson" in name:
        return "Carter, Paterson & Co."
    if name.startswith("Mitchell"):
        return "Mitchell, Sons & Candy"
    name = name.split(", ")[0].strip()
    return labels.get(name, name)


def find_headings(text: str, labels: dict[str, str] | None = None) -> list[Heading]:
    """Section headings such as ``_Dr. Seward's Diary._`` or ``_Letter, A to B._``.

    ``labels`` maps a writer as written ("Seward", "Mrs. Harker") to the
    narrator label used elsewhere ("Dr. John Seward", "Mina Harker").
    An italic span that names no writer ("_Telegram, October 24th._") is
    not a heading.
    """
    labels = labels or {}
    found: list[Heading] = []
    for m in _OWNED.finditer(text):
        kind = "phonograph" if m.group(2) else SECTION_KIND[m.group(3).lower()]
        title = f"{m.group(1)}'s {m.group(2) or ''}{m.group(3)}"
        found.append(Heading(m.start(), m.end(), title, _writer_label(m.group(1), labels), kind))
    for m in _UPPER.finditer(text):
        owner = m.group(1).title()
        title = f"{owner}'s {m.group(2).title()}"
        found.append(Heading(m.start(), m.end(), title, _writer_label(owner, labels), SECTION_KIND[m.group(2).lower()]))
    for m in _PAPER.finditer(text):
        found.append(Heading(m.start(), m.end(), m.group(0).strip("_").rstrip("."), "Newspaper cutting", "newspaper"))
    for m in _SENT.finditer(text):
        if re.match(r"[A-Z][a-z]+ \d", m.group(2)):
            continue
        title = " ".join(m.group(0).strip("_").split()).rstrip(".")
        found.append(Heading(m.start(), m.end(), title, _writer_label(m.group(2), labels), SECTION_KIND[m.group(1).lower()]))
    found.sort(key=lambda h: h.start)
    out: list[Heading] = []
    for h in found:
        if out and h.start < out[-1].end:
            continue
        out.append(h)
    return out


def split_documents(
    entries: Sequence[Entry],
    labels: dict[str, str] | None = None,
    overrides: dict[int, dict] | None = None,
    carry_max: int = 160,
) -> list[Entry]:
    """Cut the source entries at their section headings.

    Each document is one journal or diary entry, letter, telegram or
    cutting, with the narrator taken from its own heading. A heading with
    almost no text after it at the end of an entry ("..._Dr. Seward's
    Diary._") opens the next entry, as in the printed book. Text before
    any heading continues the previous section's writer, except at the
    start of a chapter, where the source label is the chapter's first
    section. ``overrides`` (source entry index -> narrator, title, kind)
    fixes entries whose opening heading is missing from the source.
    Documents are numbered from 1; ``source_entry`` keeps the original
    index. The first document cut from an entry keeps its date.
    """
    labels = labels or {}
    overrides = overrides or {}
    docs: list[Entry] = []
    writer = title = kind = None
    carried = ""
    prev_chapter = None
    for entry in entries:
        text = entry.text
        heads = find_headings(text, labels)
        if entry.chapter_number != prev_chapter and not carried:
            writer, title, kind = entry.narrator, entry.section_title, entry.kind
        prev_chapter = entry.chapter_number
        if entry.entry_index in overrides:
            o = overrides[entry.entry_index]
            writer, title, kind = o["narrator"], o.get("title", title), o.get("kind", kind)
        # A trailing heading with almost no text after it opens the next entry.
        tail = None
        if heads and len(text) - heads[-1].end <= carry_max:
            tail = heads.pop()
        cuts = [0] + [h.start for h in heads] + [tail.start if tail else len(text)]
        pending = list(heads)
        first = True
        for a, b in zip(cuts, cuts[1:]):
            if pending and pending[0].start == a:
                h = pending.pop(0)
                writer, title, kind = h.writer, h.title, h.kind
            body = text[a:b].strip()
            if not body:
                continue
            body = f"{carried} {body}".strip() if first and carried else body
            docs.append(Entry(
                entry_index=len(docs) + 1,
                chapter_number=entry.chapter_number,
                section_title=title,
                narrator=writer,
                date_str=entry.date_str if first else None,
                date_iso=entry.date_iso if first else None,
                kind=kind or "document",
                text=body,
                n_chunks=entry.n_chunks,
                source_entry=entry.entry_index,
            ))
            first = False
        carried = ""
        if tail:
            carried = text[tail.start:].strip()
            writer, title, kind = tail.writer, tail.title, tail.kind
    return docs


def load_documents(path: str | Path, labels: dict[str, str] | None = None,
                   overrides: dict[int, dict] | None = None) -> list[Entry]:
    return split_documents(load_entries(path), labels, overrides)

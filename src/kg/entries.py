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

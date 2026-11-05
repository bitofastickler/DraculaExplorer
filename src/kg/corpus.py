# -*- coding: utf-8 -*-
"""Build ``data/dracula_ascii_rag.json`` from the Project Gutenberg text.

The earlier corpus file dropped some sections (Mina's letter from
Buda-Pesth that opens chapter IX, Van Helsing's phonograph diary that
opens chapter XXIV, Jonathan's journal that opens chapter XX), attached
each section heading to the end of the entry before it, and labelled runs
of entries with the chapter's first narrator. This module rebuilds it from
``data/source/pg345.txt`` (Gutenberg eBook #345) in the same schema.

One entry is one dated diary or journal entry, one letter, telegram,
memorandum or note, or one newspaper cutting. Each entry starts with its
section heading when it opens a section, so the heading travels with the
text. Chapter numbers and narrators come from the book's own chapter and
section headings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from src.kg.entries import _writer_label, find_headings

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
YEAR = 1897  # The novel never states the year; the old corpus used 1897.

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}
_ASCII = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "--",
          "æ": "ae", "£": "L", "™": "(TM)", "•": "*"}
_DATE_LEAD = re.compile(r'^_([^_]{1,80})_\.?--')
_DAY_MONTH = re.compile(r"(\d{1,2})(?:-\d{1,2})?\s+(" + "|".join(MONTHS) + r")")
_MONTH_DAY = re.compile(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})")
_TIME = re.compile(r"\d{1,2}(?::\d\d)? [ap]\. m\.")
# Uppercase section headings that open a chapter or a section.
_UPPER_OWNED = re.compile(r"((?:DR\. )?[A-Z]+(?: [A-Z]+)?)'S (JOURNAL|DIARY)\.?")
_SIGNATURE = re.compile(r'^"?[A-Z][A-Z .,&\'-]*\."?_?$')


def to_ascii(text: str) -> str:
    for src, dst in _ASCII.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def roman(value: str) -> int:
    total = 0
    for a, b in zip(value, value[1:] + " "):
        n = _ROMAN[a]
        total += -n if b != " " and _ROMAN.get(b, 0) > n else n
    return total


@dataclass
class Section:
    chapter: int
    title: str
    narrator: str
    paragraphs: list[str] = field(default_factory=list)
    date_str: str | None = None
    date_iso: str | None = None


def _date_of(lead: str) -> tuple[str | None, str | None]:
    m = _DAY_MONTH.search(lead)
    if m:
        day, month = int(m.group(1)), m.group(2)
    else:
        m = _MONTH_DAY.search(lead)
        if not m:
            return None, None
        month, day = m.group(1), int(m.group(2))
    return f"{day} {month}", date(YEAR, MONTHS[month], day).isoformat()


def _heading(para: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """(title, narrator) when a paragraph is a section heading."""
    one = " ".join(para.split())
    if one == "DR. SEWARD'S PHONOGRAPH DIARY, SPOKEN BY VAN HELSING":
        return "Dr. Seward's Phonograph Diary, spoken by Van Helsing", "Abraham Van Helsing"
    if one.startswith('CUTTING FROM "THE DAILYGRAPH,"'):
        return 'Cutting from "The Dailygraph," 8 August', "Newspaper cutting"
    if one == "NOTE":
        return "Note", "Jonathan Harker"
    m = _UPPER_OWNED.fullmatch(one)
    if m:
        owner = m.group(1).title().replace("Dr. ", "Dr. ")
        title = f"{owner}'s {m.group(2).title()}"
        return title, _writer_label(owner, labels)
    heads = find_headings(one, labels)
    if heads and heads[0].start == 0 and heads[0].end >= len(one.rstrip(".")) - 1:
        return heads[0].title, heads[0].writer
    return None


def parse_book(raw: str, labels: dict[str, str]) -> list[Section]:
    """Cut the Gutenberg text into entries with chapter, title, narrator and date."""
    text = to_ascii(raw.replace("\r\n", "\n"))
    toc = {roman(n): t.strip() for n, t in re.findall(r"^CHAPTER ([IVXL]+)\. (.+)$", text, flags=re.M)}
    body = text[text.index("\nCHAPTER I\n") + 1:re.search(r"\n\s*THE END\s*\n", text).start()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    sections: list[Section] = []
    chapter = 0
    title = narrator = None
    last_iso = None
    pending_telegram = False

    def open_section(t: str, n: str) -> Section:
        sec = Section(chapter, t, n, date_iso=last_iso)
        sections.append(sec)
        return sec

    current: Section | None = None
    for para in paras:
        m = re.fullmatch(r"CHAPTER ([IVXL]+)", para)
        if m:
            chapter = roman(m.group(1))
            # The chapter title names the first section's writer; a heading
            # on the next paragraph may restate it.
            title = toc[chapter]
            owned = _heading(title.upper() if "'s" in title else title, labels)
            if title.startswith("Letters"):
                narrator = "Mina Murray"
            elif owned:
                title, narrator = owned
            else:
                narrator = "Newspaper cutting"
            current = open_section(title, narrator)
            continue
        one = " ".join(para.split())
        if pending_telegram:
            # "_Telegram, October 24th._" then "_Rufus Smith, Lloyd's, London, to ..._"
            sender = re.match(r"_([^_,]+),", one)
            if sender:
                current.narrator = _writer_label(sender.group(1), labels)
                current.title = f"Telegram, {one.strip('_').rstrip('.')}"
            pending_telegram = False
        head = None if one.startswith("_Telegram, October") else _heading(para, labels)
        if one.startswith("_Telegram, October"):
            current = open_section("Telegram", "Rufus Smith")
            current.paragraphs.append(para)
            pending_telegram = True
            continue
        if head:
            heading_title, heading_writer = head
            # A chapter's opening heading restates the chapter title.
            if not (current and not current.paragraphs and current.narrator == heading_writer):
                current = open_section(heading_title, heading_writer)
            current.title = heading_title
            current.paragraphs.append(para)
            continue
        lead = _DATE_LEAD.match(one)
        if lead and (_date_of(lead.group(1))[0] or _TIME.search(lead.group(1))):
            d_str, d_iso = _date_of(lead.group(1))
            if d_iso:
                last_iso = d_iso
            if current.paragraphs and not _only_headings(current, labels):
                current = open_section(current.title, current.narrator)
            current.date_str = d_str or lead.group(1).rstrip(".")
            current.date_iso = last_iso
        elif current.date_str is None and current.paragraphs and len(current.paragraphs) <= 3:
            # Letters carry their date inside the opening lines: '"_9 May._'
            d_str, d_iso = _date_of(one[:60]) if one.startswith('"_') else (None, None)
            if d_iso:
                current.date_str, current.date_iso = d_str, d_iso
                last_iso = d_iso
        current.paragraphs.append(para)
    return [s for s in sections if s.paragraphs]


def _only_headings(sec: Section, labels: dict[str, str]) -> bool:
    # "(_Kept in shorthand._)", "(Unopened by her.)" are notes under a heading.
    return all(_heading(p, labels) or p.startswith("(") for p in sec.paragraphs)


def chunk(text: str, size: int = 1500, overlap: int = 150) -> list[str]:
    """Overlapping chunks that end on whitespace, like the original file."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size - 200, end)
            end = cut if cut > start else end
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


def build_rows(sections: list[Section]) -> list[dict]:
    rows = []
    for idx, sec in enumerate(sections, start=1):
        text = "\n\n".join(sec.paragraphs)
        for k, piece in enumerate(chunk(text), start=1):
            rows.append({
                "uid": f"DRAC-E{idx:04d}-C{k:03d}",
                "entry_index": idx,
                "chunk_index": k,
                "chapter_number": sec.chapter,
                "section_title": sec.title,
                "narrator": sec.narrator,
                "date_str": sec.date_str,
                "date_iso": sec.date_iso,
                "text": piece,
            })
    return rows

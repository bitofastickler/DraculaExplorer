# -*- coding: utf-8 -*-
from pathlib import Path

from src.kg.entries import deoverlap, load_entries, naive_join

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "dracula_ascii_rag.json"


def test_deoverlap_removes_the_chunk_boundary_repeat():
    import json
    rows = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = [r["text"] for r in sorted((r for r in rows if r["entry_index"] == 1), key=lambda r: r["chunk_index"])]
    joined = naive_join(chunks)
    cleaned = deoverlap(chunks)
    assert len(cleaned) < len(joined)
    overlap = chunks[0][-80:]
    assert cleaned.count(overlap) == 1
    assert joined.count(overlap) >= 1
    assert cleaned.count("Left Munich at 8:35") == 1


def test_load_entries_keeps_narrator_and_chapters():
    entries = load_entries(CORPUS)
    assert len(entries) == 192
    chapters = {e.chapter_number for e in entries}
    assert chapters == set(range(1, 28))
    first = entries[0]
    assert first.narrator == "Jonathan Harker"
    assert first.kind == "journal"
    assert first.date_iso == "1897-05-03"
    kinds = {e.kind for e in entries}
    assert {"letter", "newspaper", "phonograph", "journal", "diary"} <= kinds
    mina = {e.narrator for e in entries if e.narrator and e.narrator.startswith("Mina")}
    assert mina == {"Mina Murray", "Mina Harker"}


def _docs():
    from src.kg.build import documents
    from src.kg.config import load_config
    return documents(ROOT, load_config(ROOT / "configs" / "kg.yml"))


def _writer_of(docs, phrase):
    hits = [d for d in docs if phrase in d.text]
    assert len(hits) == 1, phrase
    return hits[0].narrator


def test_parse_book_cuts_at_headings_and_dates():
    from src.kg.corpus import parse_book
    toc = "CHAPTER I. Jonathan Harker\u2019s Journal\nCHAPTER II. Mina Murray\u2019s Journal\n"
    paras = [
        "CHAPTER I", "JONATHAN HARKER\u2019S JOURNAL", "_3 May._--Left Munich.", "_4 May._--Arrived.",
        "_Letter, Lucy Westenra to Mina Murray_.", "\u201c_17 May._", "\u201cMy dear.\u201d",
        "CHAPTER II", "_Dr. Seward\u2019s Diary._", "_5 June._--Renfield.", "THE END",
    ]
    raw = toc + "\n" + "\n\n".join(paras) + "\n"
    secs = parse_book(raw, {"Dr. Seward": "Dr. John Seward"})
    assert [(s.chapter, s.narrator, s.date_str) for s in secs] == [
        (1, "Jonathan Harker", "3 May"),
        (1, "Jonathan Harker", "4 May"),
        (1, "Lucy Westenra", "17 May"),
        (2, "Dr. John Seward", "5 June"),
    ]
    assert secs[2].paragraphs[0].startswith("_Letter, Lucy Westenra")


def test_document_narrators_match_the_book():
    docs = _docs()
    assert docs[0].narrator == "Jonathan Harker"
    # The first corpus labelled these Mina Murray / Dr. Seward; the book says otherwise.
    assert _writer_of(docs, "The case of Renfield grows even more interesting") == "Dr. John Seward"
    assert _writer_of(docs, "Zoophagous patient still keeps up our interest") == "Dr. John Seward"
    assert _writer_of(docs, "Poor fellows, neither of them is aware") == "Mina Harker"
    assert _writer_of(docs, "When I read to Mina, Van Helsing's message") == "Jonathan Harker"
    assert _writer_of(docs, "I placed a man in the corridor last night") == "Dr. John Seward"
    assert _writer_of(docs, "Won't you just hitch up alongside of me") == "Lucy Westenra"
    assert _writer_of(docs, "Have you said anything to our young friend the lover") == "Dr. John Seward"


def test_sections_missing_from_the_first_corpus_are_back():
    docs = _docs()
    by = lambda phrase: next(d for d in docs if phrase in d.text)
    budapest = by("Buda-Pesth, 24 August")
    assert (budapest.chapter_number, budapest.narrator) == (9, "Mina Harker")
    phonograph = by("This to Jonathan Harker.")
    assert (phonograph.chapter_number, phonograph.narrator, phonograph.kind) == (24, "Abraham Van Helsing", "phonograph")
    snelling = by("I found Thomas Snelling in his house")
    assert (snelling.chapter_number, snelling.narrator) == (20, "Jonathan Harker")
    assert by("Seven years ago we all went through the flames").narrator == "Jonathan Harker"
    # Chapter VIII ends with Seward's 19 August entry; IX opens with Mina's letter.
    assert by("The case of Renfield grows even more interesting").chapter_number == 9

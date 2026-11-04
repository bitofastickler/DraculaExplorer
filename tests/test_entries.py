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
    assert len(entries) == 125
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

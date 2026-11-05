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


def _docs():
    from src.kg.build import documents
    from src.kg.config import load_config
    return documents(ROOT, load_config(ROOT / "configs" / "kg.yml"))


def _writer_of(docs, phrase):
    hits = [d for d in docs if phrase in d.text]
    assert len(hits) == 1, phrase
    return hits[0].narrator


def test_split_documents_follows_the_section_headings():
    from src.kg.entries import Entry, split_documents
    raw = [
        Entry(1, 1, "Mina Murray's Journal", "Mina Murray", "1 May", None, "journal",
              "I wrote today. _Letter, Lucy Westenra to Mina Murray._ \"Dear Mina, all well. _Dr. Seward's Diary._", 1),
        Entry(2, 1, "Mina Murray's Journal", "Mina Murray", "2 May", None, "journal", "Renfield ate a fly.", 1),
    ]
    docs = split_documents(raw, {"Dr. Seward": "Dr. John Seward"})
    assert [d.narrator for d in docs] == ["Mina Murray", "Lucy Westenra", "Dr. John Seward"]
    assert docs[2].text.startswith("_Dr. Seward's Diary._ Renfield")
    assert [d.source_entry for d in docs] == [1, 1, 2]
    assert docs[1].date_str is None and docs[2].date_str == "2 May"


def test_document_narrators_match_the_book():
    docs = _docs()
    assert docs[0].narrator == "Jonathan Harker"
    # The source labels these Mina Murray / Dr. Seward; the text says otherwise.
    assert _writer_of(docs, "The case of Renfield grows even more interesting") == "Dr. John Seward"
    assert _writer_of(docs, "Zoophagous patient still keeps up our interest") == "Dr. John Seward"
    assert _writer_of(docs, "Poor fellows, neither of them is aware") == "Mina Harker"
    assert _writer_of(docs, "When I read to Mina, Van Helsing's message") == "Jonathan Harker"
    assert _writer_of(docs, "I placed a man in the corridor last night") == "Dr. John Seward"
    assert _writer_of(docs, "Won't you just hitch up alongside of me") == "Lucy Westenra"
    assert _writer_of(docs, "Have you said anything to our young friend the lover") == "Dr. John Seward"

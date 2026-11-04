# -*- coding: utf-8 -*-
from pathlib import Path

from src.kg.entries import load_entries
from src.kg.mentions import extract_mentions

ROOT = Path(__file__).resolve().parents[1]


def test_chapter_one_keeps_castle_and_count_apart():
    entries = [e for e in load_entries(ROOT / "data" / "dracula_ascii_rag.json") if e.entry_index == 1]
    mentions = extract_mentions(entries)
    surfaces = {(m.surface, m.entity_type) for m in mentions}
    assert any(m.surface == "Count Dracula" and m.entity_type == "PER" for m in mentions)
    assert any(m.surface == "Castle Dracula" and m.entity_type == "FAC" for m in mentions)
    assert any(m.surface == "Bistritz" for m in mentions)
    assert any(m.surface == "Mina" for m in mentions)
    # Entry 1 names the castle once. De-overlap must not duplicate that span.
    castles = [m for m in mentions if m.surface == "Castle Dracula"]
    assert len(castles) == 1

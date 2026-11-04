# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from src.kg.build import prepare
from src.kg.config import load_yaml
from src.kg.evaluate import answer_queries, pairwise_f1

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    return prepare(ROOT, use_knowledge=True)


def test_graph_queries(prepared):
    answers = answer_queries(prepared["entries"], prepared["mentions"], prepared["graph"])
    expect = load_yaml(ROOT / "data" / "gold" / "queries.yml")["expect"]
    assert answers["ids_differ"] is True
    assert answers["castle_id"] == "castle_dracula"
    assert answers["count_id"] == "count_dracula"
    assert answers["narrators"] == expect["narrators"]
    chapters = sorted((int(k), v) for k, v in answers["mina_surnames_by_chapter"].items())
    assert chapters, "Mina should keep at least one surfaced surname"
    assert expect["mina_early_surname"] in chapters[0][1]
    assert expect["mina_late_surname"] in chapters[-1][1]
    assert answers["seward_entries_mentioning_renfield"]
    assert "count_dracula" in {m.entity_id for m in prepared["mentions"]}
    assert answers["count_dracula_surfaces"]
    held = pairwise_f1(prepared["mentions"], prepared["gold"], set(range(14, 28)))
    assert held["f1"] >= 0.90, held

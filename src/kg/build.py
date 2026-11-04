# -*- coding: utf-8 -*-
"""Run mention detection and resolution for the Dracula corpus."""
from __future__ import annotations

import copy
from pathlib import Path

from src.kg.config import ROOT, load_config, load_yaml
from src.kg.entries import load_entries
from src.kg.evaluate import answer_queries, nominal_stats, pairwise_f1
from src.kg.graph import build_graph
from src.kg.mentions import apply_alias_types, extract_mentions
from src.kg.relations import find_relations
from src.kg.resolve import apply_gold_labels, resolve_mentions


def _pairs(blob: dict, key: str) -> list[tuple[str, str]]:
    rows = []
    for pair in blob.get(key) or []:
        if len(pair) == 2:
            rows.append((str(pair[0]), str(pair[1])))
    return rows


def prepare(root: Path | None = None, use_knowledge: bool = True, mentions=None):
    """Resolve and build the graph. Pass ``mentions`` to skip spaCy.

    Resolution mutates mention objects, so pass a fresh copy per call.
    """
    root = Path(root) if root else ROOT
    entries = load_entries(root / "data" / "dracula_ascii_rag.json")
    if mentions is None:
        mentions = extract_mentions(entries)
    config = load_config(root / "configs" / "kg.yml")
    gold = load_yaml(root / "data" / "gold" / "entities.yml")["entities"]
    apply_alias_types(mentions, gold)
    knowledge = load_yaml(root / "data" / "gold" / "must_links.yml") or {}
    must = _pairs(knowledge, "pairs") if use_knowledge else []
    cannot = _pairs(knowledge, "cannot") if use_knowledge else []
    resolved = resolve_mentions(mentions, entries, config, extra_must=must, extra_cannot=cannot)
    apply_gold_labels(resolved.mentions, gold)
    relations = find_relations(entries, resolved.mentions)
    graph = build_graph(
        entries,
        resolved.mentions,
        relations,
        config,
        source_name="data/dracula_ascii_rag.json",
        gold_entities=gold,
    )
    return {
        "entries": entries,
        "mentions": resolved.mentions,
        "review": resolved.review,
        "relations": relations,
        "graph": graph,
        "gold": gold,
        "config": config,
    }


def score_split(prepared: dict, chapters: set[int]) -> dict:
    return pairwise_f1(prepared["mentions"], prepared["gold"], chapters)


def report(root: Path | None = None) -> dict:
    """Score the held-out chapters with the knowledge file on and off."""
    root = Path(root) if root else ROOT
    entries = load_entries(root / "data" / "dracula_ascii_rag.json")
    # spaCy runs once. Each pass resolves its own copy of the mentions.
    raw = extract_mentions(entries)
    on = prepare(root, use_knowledge=True, mentions=copy.deepcopy(raw))
    off = prepare(root, use_knowledge=False, mentions=copy.deepcopy(raw))
    heldout = set(range(14, 28))
    tune = set(range(1, 14))
    return {
        "on": on,
        "heldout_with_knowledge": pairwise_f1(on["mentions"], on["gold"], heldout),
        "tune_with_knowledge": pairwise_f1(on["mentions"], on["gold"], tune),
        "heldout_without_knowledge": pairwise_f1(off["mentions"], off["gold"], heldout),
        "nominals": nominal_stats(on["mentions"]),
        "queries": answer_queries(on["entries"], on["mentions"], on["graph"]),
        "n_relations": len(on["relations"]),
    }

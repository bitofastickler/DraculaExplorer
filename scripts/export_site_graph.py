# -*- coding: utf-8 -*-
"""Slim data/kg/graph.json into docs/data/graph.json for the GitHub Pages viewer.

Needs only the standard library, so the site can be refreshed without spaCy.
Keeps entities, the entries they appear in, co-appearance links
(two entities appearing in the same entry), and relations with their
evidence sentence. Mention-level REFERS_TO edges
are dropped; they point at mention ids that have no node in graph.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "kg" / "graph.json"
OUT = ROOT / "docs" / "data" / "graph.json"

MIN_MENTIONS = 2   # entities with fewer name or title mentions are left out
MIN_SHARED = 2     # co-appearance links need at least this many shared entries
GROUP = {"PER": "person", "GPE": "place", "LOC": "place", "FAC": "place"}


def main() -> None:
    graph = json.loads(SRC.read_text(encoding="utf-8"))
    nodes = graph["nodes"]
    edges = graph["edges"]

    docs = {}
    for n in nodes:
        if n["label"] == "Document":
            docs[n["entry_index"]] = {
                "chapter": n.get("chapter_number"),
                "title": n.get("section_title"),
                "narrator": n.get("narrator"),
                "date": n.get("date_str"),
                "kind": n.get("kind"),
            }

    entries_of: dict[str, set[int]] = defaultdict(set)
    narrated: dict[str, int] = defaultdict(int)
    # Names plus titles such as "the Count". Pronoun links are a rough
    # same-sentence heuristic, so they are left out of the count.
    named: dict[str, int] = defaultdict(int)
    for e in edges:
        if e["type"] == "REFERS_TO" and e.get("method") != "same_sentence_pronoun":
            named[e["target"]] += 1
        elif e["type"] == "APPEARS_IN":
            entries_of[e["source"]].add(int(e["target"].split(":")[1]))
        elif e["type"] == "NARRATED_BY":
            narrated[e["target"]] += 1

    entities = []
    for n in nodes:
        mentions = named.get(n["id"], 0)
        if n["label"] != "Entity" or mentions < MIN_MENTIONS:
            continue
        entities.append({
            "id": n["id"],
            "name": n["canonical_name"],
            "type": n["entity_type"],
            "group": GROUP.get(n["entity_type"], "other"),
            "mentions": mentions,
            "aliases": n.get("aliases") or [],
            "entries": sorted(entries_of.get(n["id"], ())),
            "narrated": narrated.get(n["id"], 0),
        })
    entities.sort(key=lambda r: -r["mentions"])

    links = []
    for a, b in combinations(entities, 2):
        shared = len(set(a["entries"]) & set(b["entries"]))
        if shared >= MIN_SHARED:
            links.append({"source": a["id"], "target": b["id"], "shared": shared})

    # Stated relations carry the sentence that supports them. Curated edges
    # (part_of from config) have no sentence.
    kept = {r["id"] for r in entities}
    relations = [
        {
            "source": e["source"],
            "target": e["target"],
            "rel": e["rel"],
            "method": e.get("method") or "curated",
            "quote": e.get("quote"),
            "entry": e.get("entry_index"),
            "quotes": [{"text": q["text"], "entry": q["entry_index"]} for q in e.get("quotes") or []],
        }
        for e in edges
        if e["type"] == "RELATION" and e["source"] in kept and e["target"] in kept
    ]

    eval_path = ROOT / "data" / "kg" / "eval.json"
    scores = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "data/kg/graph.json",
        "documents": {str(k): v for k, v in sorted(docs.items())},
        "entities": entities,
        "links": links,
        "relations": relations,
        "heldout": scores.get("heldout_with_knowledge"),
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {OUT} ({len(entities)} entities, {len(links)} links, "
        f"{len(relations)} relations, {OUT.stat().st_size // 1024} KB)"
    )


if __name__ == "__main__":
    main()

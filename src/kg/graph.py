# -*- coding: utf-8 -*-
"""Assemble the property graph from entries, mentions, and relations."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.kg.entries import Entry
from src.kg.mentions import Mention


def _doc_id(entry_index: int) -> str:
    return f"entry:{entry_index}"


def build_graph(
    entries: list[Entry],
    mentions: list[Mention],
    relations: list[dict],
    config: dict,
    source_name: str,
    gold_entities: list[dict] | None = None,
) -> dict:
    gold_by_id = {e["id"]: e for e in gold_entities or []}
    narrator_map = config.get("narrator_to_entity") or {}

    surfaces: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    types: dict[str, str] = {}
    for mention in mentions:
        if not mention.entity_id:
            continue
        types[mention.entity_id] = mention.entity_type
        if mention.mention_class == "PROP":
            surfaces[mention.entity_id][mention.surface] += 1

    for target in narrator_map.values():
        types.setdefault(target, "PER")

    nodes: list[dict] = [{
        "id": "dracula",
        "label": "Work",
        "title": "Dracula",
        "author": "Bram Stoker",
        "source": source_name,
    }]
    edges: list[dict] = []

    for entry in entries:
        node_id = _doc_id(entry.entry_index)
        nodes.append({
            "id": node_id,
            "label": "Document",
            "entry_index": entry.entry_index,
            "chapter_number": entry.chapter_number,
            "section_title": entry.section_title,
            "narrator": entry.narrator,
            "date_str": entry.date_str,
            "date_iso": entry.date_iso,
            "kind": entry.kind,
        })
        edges.append({"type": "PART_OF_WORK", "source": node_id, "target": "dracula"})
        narrator_id = narrator_map.get(entry.narrator or "")
        if narrator_id:
            edges.append({"type": "NARRATED_BY", "source": node_id, "target": narrator_id})
        if entry.kind == "phonograph":
            # The corpus labels these entries as Van Helsing speaking.
            # The section title says the words were taken down on Seward's phonograph.
            edges.append({
                "type": "RECORDED_BY",
                "source": node_id,
                "target": "john_seward",
                "evidence": "section_title",
            })

    for eid, etype in sorted(types.items()):
        gold = gold_by_id.get(eid, {})
        counts = surfaces.get(eid) or {}
        if counts:
            canonical = max(counts, key=lambda s: (counts[s], len(s)))
        else:
            canonical = gold.get("canonical_name", eid)
        if gold.get("canonical_name"):
            canonical = gold["canonical_name"]
        nodes.append({
            "id": eid,
            "label": "Entity",
            "canonical_name": canonical,
            "entity_type": gold.get("entity_type", etype),
            "aliases": sorted(counts),
            "mention_count": sum(counts.values()),
        })

    appears: dict[tuple, int] = defaultdict(int)
    for mention in mentions:
        if not mention.entity_id:
            continue
        edges.append({
            "type": "REFERS_TO",
            "source": mention.mention_id,
            "target": mention.entity_id,
            "method": mention.link_method,
            "score": mention.link_score,
            "surface": mention.surface,
            "entry_index": mention.entry_index,
        })
        appears[(mention.entity_id, mention.entry_index)] += 1

    for (eid, entry_index), count in sorted(appears.items()):
        edges.append({
            "type": "APPEARS_IN",
            "source": eid,
            "target": _doc_id(entry_index),
            "mention_count": count,
        })

    known = set(types)
    for rel in relations:
        if rel["source"] in known and rel["target"] in known and rel["source"] != rel["target"]:
            edges.append({
                "type": "RELATION",
                "rel": rel["type"],
                "source": rel["source"],
                "target": rel["target"],
                "evidence": rel.get("evidence") or [],
            })

    for pair in config.get("part_of") or []:
        if len(pair) != 2:
            continue
        src, dst = pair
        if src in known and dst in known:
            edges.append({
                "type": "RELATION",
                "rel": "part_of",
                "source": src,
                "target": dst,
                "evidence": ["curated"],
            })

    return {"schema_version": 1, "source": source_name, "nodes": nodes, "edges": edges}


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")

# -*- coding: utf-8 -*-
"""Pairwise entity-resolution scores and the ten graph queries."""
from __future__ import annotations

import re
from collections import defaultdict

from src.kg.entries import Entry
from src.kg.mentions import Mention
from src.kg.resolve import parse_name


def _forms(mention: Mention) -> set[str]:
    part = parse_name(mention.surface)
    forms = {part.token_str}
    if part.honorific:
        forms.add(f"{part.honorific} {part.token_str}".strip())
    forms.add(re.sub(r"\s+", " ", mention.surface.lower()).strip())
    return {f for f in forms if f}


def match_gold(mention: Mention, gold_entities: list[dict]) -> list[str]:
    hits = []
    forms = _forms(mention)
    for entity in gold_entities:
        etype = entity.get("entity_type")
        if etype and mention.entity_type != etype:
            continue
        aliases = {a.lower().strip() for a in entity.get("aliases") or []}
        if forms & aliases:
            hits.append(entity["id"])
    return hits


def pairwise_f1(mentions: list[Mention], gold_entities: list[dict], chapters: set[int] | None) -> dict:
    """Pairwise precision/recall on proper mentions that match exactly one gold entity."""
    chosen = []
    for mention in mentions:
        if mention.mention_class != "PROP":
            continue
        if chapters is not None and mention.chapter_number not in chapters:
            continue
        hits = match_gold(mention, gold_entities)
        if len(hits) == 1:
            chosen.append((mention, hits[0]))

    gold_pairs = set()
    pred_pairs = set()
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if chosen[i][1] == chosen[j][1]:
                gold_pairs.add((i, j))
            left, right = chosen[i][0], chosen[j][0]
            if left.entity_id and left.entity_id == right.entity_id:
                pred_pairs.add((i, j))
    tp = len(gold_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "mentions": len(chosen),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_pairs": len(gold_pairs),
        "pred_pairs": len(pred_pairs),
    }


def nominal_stats(mentions: list[Mention]) -> dict:
    nominals = [m for m in mentions if m.mention_class == "NOM" and m.entity_type == "PER"]
    titles = [m for m in nominals if m.surface.lower().startswith("the ")]
    attached = [m for m in titles if m.entity_id]
    return {
        "title_nominals": len(titles),
        "title_nominals_attached": len(attached),
        "unresolved_rate": (1 - len(attached) / len(titles)) if titles else 0.0,
    }


def answer_queries(entries: list[Entry], mentions: list[Mention], graph: dict) -> dict:
    entities = [n for n in graph["nodes"] if n["label"] == "Entity"]
    documents = [n for n in graph["nodes"] if n["label"] == "Document"]
    by_id = {n["id"]: n for n in entities}

    people = [
        {"id": n["id"], "canonical_name": n["canonical_name"], "alias_count": len(n.get("aliases") or [])}
        for n in entities if n["entity_type"] == "PER"
    ]
    people.sort(key=lambda r: (-r["alias_count"], r["id"]))

    mina_by_chapter: dict[int, list[str]] = defaultdict(list)
    seen = set()
    for mention in mentions:
        if mention.entity_id != "mina_harker" or mention.mention_class != "PROP":
            continue
        part = parse_name(mention.surface)
        surname = part.surname
        if not surname or mention.chapter_number is None:
            continue
        key = (mention.chapter_number, surname)
        if key in seen:
            continue
        seen.add(key)
        mina_by_chapter[mention.chapter_number].append(surname)

    narrated = {e["target"] for e in graph["edges"] if e["type"] == "NARRATED_BY"}
    seward_docs = {e["source"] for e in graph["edges"] if e["type"] == "NARRATED_BY" and e["target"] == "john_seward"}
    renfield_docs = {e["target"] for e in graph["edges"] if e["type"] == "APPEARS_IN" and e["source"] == "rm_renfield"}
    seward_renfield = sorted(
        int(d.split(":")[1]) for d in (seward_docs & renfield_docs)
    )

    first_string = None
    for entry in entries:
        if "dracula" in entry.text.lower():
            first_string = entry.entry_index
            break
    first_person = None
    appeared = []
    for edge in graph["edges"]:
        if edge["type"] == "APPEARS_IN" and edge["source"] == "count_dracula":
            appeared.append(int(edge["target"].split(":")[1]))
    if appeared:
        first_person = min(appeared)

    place_order = []
    seen_places = set()
    for mention in mentions:
        if mention.entity_type not in {"GPE", "LOC", "FAC"} or not mention.entity_id:
            continue
        if mention.entity_id in seen_places:
            continue
        seen_places.add(mention.entity_id)
        place_order.append(mention.entity_id)

    narrators = sorted(narrated)
    dracula_surfaces: dict[str, int] = defaultdict(int)
    for mention in mentions:
        if mention.entity_id == "count_dracula":
            dracula_surfaces[mention.surface] += 1

    facilities = sorted(n["id"] for n in entities if n["entity_type"] == "FAC")
    gpes = sorted(n["id"] for n in entities if n["entity_type"] == "GPE")

    early_people = set()
    for mention in mentions:
        if mention.chapter_number is not None and mention.chapter_number <= 4 and mention.entity_type == "PER" and mention.entity_id:
            early_people.add(mention.entity_id)
    never_narrate = sorted(early_people - narrated)

    castle = by_id.get("castle_dracula", {}).get("id")
    person = by_id.get("count_dracula", {}).get("id")

    return {
        "people_alias_counts": people,
        "mina_surnames_by_chapter": {str(k): v for k, v in sorted(mina_by_chapter.items())},
        "seward_entries_mentioning_renfield": seward_renfield,
        "first_entry_string_dracula": first_string,
        "first_entry_person_dracula": first_person,
        "places_first_mention": place_order,
        "narrators": narrators,
        "count_dracula_surfaces": dict(sorted(dracula_surfaces.items(), key=lambda kv: (-kv[1], kv[0]))),
        "facilities": facilities,
        "gpes": gpes,
        "early_people_who_never_narrate": never_narrate,
        "castle_id": castle,
        "count_id": person,
        "ids_differ": bool(castle and person and castle != person),
        "document_count": len(documents),
    }

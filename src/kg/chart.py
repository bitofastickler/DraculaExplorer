# -*- coding: utf-8 -*-
"""Resolved character co-occurrence, drawn with the existing heatmap helper."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.kg.mentions import Mention
from src.visuals import plot_cooccurrence_heatmap


def cooccurrence_frame(mentions: list[Mention], entity_ids: list[str], names: dict[str, str]) -> pd.DataFrame:
    present: dict[int, set[str]] = defaultdict(set)
    for mention in mentions:
        if mention.entity_id in entity_ids and mention.entity_type == "PER":
            present[mention.entry_index].add(mention.entity_id)
    labels = [names.get(eid, eid) for eid in entity_ids]
    mat = pd.DataFrame(0, index=labels, columns=labels)
    id_to_label = dict(zip(entity_ids, labels))
    for ids in present.values():
        ordered = sorted(ids)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                la, lb = id_to_label[a], id_to_label[b]
                mat.loc[la, lb] += 1
                mat.loc[lb, la] += 1
    return mat


def save_resolved_chart(mentions: list[Mention], gold_entities: list[dict], path: str | Path) -> None:
    people = [e for e in gold_entities if e.get("entity_type") == "PER"]
    # Keep the chart to entities the resolver actually found.
    found = {m.entity_id for m in mentions if m.entity_id}
    people = [e for e in people if e["id"] in found]
    names = {e["id"]: e.get("canonical_name") or e["id"] for e in people}
    frame = cooccurrence_frame(mentions, [e["id"] for e in people], names)
    fig = plot_cooccurrence_heatmap(frame, title="Resolved character co-occurrence (by entry)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")

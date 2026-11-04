# -*- coding: utf-8 -*-
"""Rebuild data/kg and the resolved co-occurrence chart from the corpus."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kg.build import prepare, report  # noqa: E402
from src.kg.chart import save_resolved_chart  # noqa: E402
from src.kg.graph import write_json, write_jsonl  # noqa: E402


def main() -> None:
    print("Resolving mentions. The first run loads the spaCy model.")
    # report() runs the pipeline twice (knowledge on and off). Build the
    # saved graph from the knowledge-on pass, which is the published artifact.
    scored = report(ROOT)
    prepared = scored["on"]
    out = ROOT / "data" / "kg"
    write_jsonl(out / "mentions.jsonl", [m.to_dict() for m in prepared["mentions"]])
    entities = [n for n in prepared["graph"]["nodes"] if n["label"] == "Entity"]
    write_jsonl(out / "entities.jsonl", entities)
    write_json(out / "graph.json", prepared["graph"])
    write_json(out / "eval.json", {
        "heldout_with_knowledge": scored["heldout_with_knowledge"],
        "heldout_without_knowledge": scored["heldout_without_knowledge"],
        "tune_with_knowledge": scored["tune_with_knowledge"],
        "nominals": scored["nominals"],
        "n_relations": scored["n_relations"],
        "n_stated_relations": scored["n_stated_relations"],
        "queries": scored["queries"],
    })
    if prepared["review"]:
        import csv
        review_path = out / "review_queue.csv"
        with review_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["surface_a", "surface_b", "entry_a", "entry_b", "score"])
            writer.writeheader()
            writer.writerows(prepared["review"])
    save_resolved_chart(
        prepared["mentions"],
        prepared["gold"],
        ROOT / "assets" / "charts" / "character_cooccurrence_resolved.png",
    )
    held = scored["heldout_with_knowledge"]
    bare = scored["heldout_without_knowledge"]
    print(
        f"Held-out chapters 14-27 pairwise F1 "
        f"with knowledge {held['f1']:.3f} (P {held['precision']:.3f} R {held['recall']:.3f}), "
        f"without {bare['f1']:.3f}."
    )
    print(
        f"Relations: {scored['n_relations']} from patterns, "
        f"{scored['n_stated_relations']} stated relations confirmed in the text."
    )
    print(f"Wrote {out / 'graph.json'}")


if __name__ == "__main__":
    main()

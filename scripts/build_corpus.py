# -*- coding: utf-8 -*-
"""Rebuild data/dracula_ascii_rag.json from data/source/pg345.txt.

Run from the repo root: python scripts/build_corpus.py
Standard library plus pyyaml; no spaCy needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kg.config import load_config  # noqa: E402
from src.kg.corpus import build_rows, parse_book  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "kg.yml")
    raw = (ROOT / "data" / "source" / "pg345.txt").read_text(encoding="utf-8")
    sections = parse_book(raw, config.get("writer_labels") or {})
    rows = build_rows(sections)
    out = ROOT / "data" / "dracula_ascii_rag.json"
    out.write_text(json.dumps(rows, ensure_ascii=True, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(sections)} entries, {len(rows)} chunks)")


if __name__ == "__main__":
    main()

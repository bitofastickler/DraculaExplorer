# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
import requests

def _find_data_file(dd: Path):
    for name in ("dracula_ascii_rag.json", "dracula_corrected_entries.json"):
        p = dd / name
        if p.exists():
            return p
    return None

def main(data_dir="data", model="llama3.2:3b", host="http://localhost:11434"):
    dd = Path(data_dir)
    p = _find_data_file(dd)
    if not p:
        print("[FAIL] No data JSON found under data/.")
        return 1

    # Parse the FULL file (no truncation)
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "entries" in data:
            n = len(data["entries"])
        elif isinstance(data, list):
            n = len(data)
        else:
            n = "?"
        print(f"[OK] Parsed {p} ({n} records).")
    except Exception as e:
        print(f"[FAIL] Could not parse {p}: {e}")
        return 1

    # Ollama reachability
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        print("[OK] Ollama reachable.")
        return 0
    except Exception as e:
        print(f"[FAIL] Ollama not reachable at {host}: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())

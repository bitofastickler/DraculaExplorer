# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
import requests

def main(data_dir="data", model="llama3.2:3b", host="http://localhost:11434"):
    dd = Path(data_dir)
    ok = False
    for name in ["dracula_ascii_rag.json", "dracula_corrected_entries.json"]:
        p = dd / name
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8")[:2048] or "[]")
                print(f"[OK] Found data file: {p}")
                ok = True; break
            except Exception as e:
                print(f"[WARN] Could not parse {p}: {e}")
    if not ok:
        print("[FAIL] No data JSON found under data/."); return 1

    try:
        r = requests.get(f"{host}/api/tags", timeout=5); r.raise_for_status()
        print("[OK] Ollama reachable.")
        return 0
    except Exception as e:
        print(f"[FAIL] Ollama not reachable at {host}: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())

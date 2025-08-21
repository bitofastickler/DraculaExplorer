# -*- coding: utf-8 -*-
"""Simple RAG harness over the Dracula corpus (local/offline friendly).
Retrieval: TF-IDF over chunks or entries.
Generation backends:
  - Ollama (LOCAL): default, runs against http://localhost:11434
  - HF Inference (optional): requires HF_API_TOKEN; not used in offline mode.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os, json, requests
from pathlib import Path
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CorpusChunks:
    df: pd.DataFrame  # columns: text, entry_index, chapter_number, narrator, date_iso

def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_chunks(data_dir: str | Path) -> CorpusChunks:
    data_dir = Path(data_dir)
    ch_path = data_dir / "dracula_ascii_rag.json"
    en_path = data_dir / "dracula_corrected_entries.json"
    if ch_path.exists():
        rows = _read_json(ch_path)
        df = pd.DataFrame(rows)
        keep = ["text","entry_index","chapter_number","narrator","date_iso"]
        for k in keep:
            if k not in df.columns: df[k] = None
        return CorpusChunks(df[keep].copy())
    if en_path.exists():
        data = _read_json(en_path)["entries"]
        df = pd.DataFrame(data)
        if "entry_index" not in df.columns:
            df.insert(0, "entry_index", np.arange(1, len(df)+1))
        keep = ["text","entry_index","chapter_number","narrator","date_iso"]
        for k in keep:
            if k not in df.columns: df[k] = None
        return CorpusChunks(df[keep].copy())
    raise FileNotFoundError("Place dracula_ascii_rag.json OR dracula_corrected_entries.json under data/." )

# Ranks entries and shows best chunks to avoid mixing chapters in top-k
def build_entry_matrix(df, X):
    # X is chunk-by-vocab (scipy csr). Collapse to entry level.
    import numpy as np
    entry_groups = df.groupby('entry_index').indices  # {entry_idx: [row_ids]}
    entry_rows = []
    entry_meta = []
    for eidx, rows in entry_groups.items():
        sub = X[rows]
        vec = sub.mean(axis=0)          # centroid per entry
        entry_rows.append(vec)
        # take first row’s metadata (chapter, narrator, date, etc.)
        r0 = rows[0]
        entry_meta.append({
            'entry_index': int(df.iloc[r0].entry_index),
            'chapter_number': int(df.iloc[r0].chapter_number),
            'narrator': df.iloc[r0].narrator,
            'date_iso': df.iloc[r0].date_iso,
            'section_title': df.iloc[r0].section_title,
        })
    entry_X = np.vstack([v.A if hasattr(v, "A") else v.toarray() for v in entry_rows])
    return entry_X, entry_meta, entry_groups

CANON = {
    r'jon+?athan': 'Jonathan Harker',
    r'\bmina\b': 'Mina Harker',
    r'\blucy\b': 'Lucy Westenra',
    r'seward|dr\.?\s*seward': 'Dr. John Seward',
    r'van\s+helsing|helsing': 'Abraham Van Helsing',
    r'arthur|holmwood|godalming': 'Arthur Holmwood',
    r'\bquincey\b': 'Quincey Morris',
    r'\brenfield\b': 'Renfield',
    r'\bdracula\b': 'Count Dracula',
}

def extract_constraints(q: str):
    ql = q.lower()
    narrator = None
    for pat, name in CANON.items():
        if re.search(pat, ql):
            narrator = name
            break
    first = bool(re.search(r'\b(first|opening|beginning|start)\b', ql))
    m = re.search(r'\bchapter\s+(\d+)\b', ql)
    chapter = int(m.group(1)) if m else None
    return {'narrator': narrator, 'first': first, 'chapter': chapter}

class TfIdfRetriever:
    def __init__(self, min_df=2, max_df=0.9, ngram_range=(1,2)):
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english",
                                   min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        self.X = None
        self.meta = None
        self._texts = None
        # new:
        self.entry_X = None
        self.entry_meta = None
        self.entry_groups = None
        self.df = None

    def fit(self, corpus: CorpusChunks):
        self.df = corpus.df.reset_index(drop=True)
        self._texts = self.df["text"].fillna("").tolist()
        self.meta = self.df[["entry_index","chapter_number","narrator","date_iso"]].to_dict("records")
        self.X = self.vec.fit_transform(self._texts)                 # chunk-level
        # NEW: entry-level centroid matrix + metadata
        self.entry_X, self.entry_meta, self.entry_groups = build_entry_matrix(self.df, self.X)
        return self

    # More sophisticated search with constraints - focuses on narrator, chapter, and keeps a sense of chronology
    def search(self, query: str, topk: int = 5):
        cons = extract_constraints(query)
        qd = self.vec.transform([query]).toarray()                   # dense
        sims = cosine_similarity(qd, self.entry_X).ravel()           # entry-level similarity

        bonus = np.zeros_like(sims, dtype=float)

        if cons['narrator']:
            mask = np.array([m['narrator'] == cons['narrator'] for m in self.entry_meta])
            if mask.sum() >= max(3, topk):   # allow soft filtering if we have enough matches
                sims = sims * (mask.astype(float) * 0.6 + 0.4)   # bias toward narrator
            bonus += np.where(mask, 0.25, 0.0)

        if cons['chapter'] is not None:
            mask = np.array([m['chapter_number'] == cons['chapter'] for m in self.entry_meta])
            bonus += np.where(mask, 0.35, 0.0)

        if cons['first']:
            chapters = np.array([m['chapter_number'] if m['chapter_number'] is not None else 9999
                                 for m in self.entry_meta], dtype=float)
            if cons['narrator']:
                mask = np.array([m['narrator'] == cons['narrator'] for m in self.entry_meta])
            else:
                mask = np.ones_like(chapters, dtype=bool)
            # reward earlier chapters within masked set
            masked = chapters.copy()
            masked[~mask] = masked.max() + 10
            rng = masked.max() - masked.min() + 1e-6
            rank_bonus = (masked.max() - masked) / rng
            bonus += 0.30 * rank_bonus

        final = sims + bonus
        order = np.argsort(-final)[:topk]

        results = []
        for i in order:
            m = dict(self.entry_meta[i])
            # representative chunk: first row for this entry
            rows = list(self.entry_groups[m['entry_index']])
            r0 = rows[0]
            m["score"] = float(final[i])
            m["text"] = self._texts[r0]          # preview
            results.append(m)
        return results

def build_prompt(question: str, passages: List[Dict[str,Any]]) -> str:
    header = "You are a concise literary assistant. Use the CONTEXT from Bram Stoker's Dracula to answer the QUESTION.\n"
    ctx_lines = []
    for j, p in enumerate(passages, 1):
        lead = f"(entry {p.get('entry_index')} | ch {p.get('chapter_number')} | {p.get('narrator')} | {p.get('date_iso')})"
        ctx_lines.append(f"[{j}] {lead}\n{p['text']}")
    instr = "\n\nGuidelines: Cite passage numbers like [1], [2] when relevant. If unknown, say so briefly.\n"
    return f"{header}\nCONTEXT:\n" + "\n\n".join(ctx_lines) + f"\n\nQUESTION: {question}\n{instr}ANSWER:"

# --- Generation backends ---
def generate_with_ollama(prompt: str, model: str = "llama3.2:3b", host: str = "http://localhost:11434") -> str:
    r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")

def generate_with_hf(prompt: str, model_id: str, hf_token: Optional[str] = None, max_new_tokens: int = 400) -> str:
    token = hf_token or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face API token not found. Set HF_API_TOKEN env var.")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "return_full_text": False}}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"]
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    return json.dumps(out)

def answer(question: str, data_dir: str | Path = "data", topk: int = 5,
           backend: str = "ollama", model: str = "llama3.2:3b",
           hf_token: Optional[str] = None):
    corpus = load_chunks(data_dir)
    retr = TfIdfRetriever().fit(corpus)
    passages = retr.search(question, topk=topk)
    prompt = build_prompt(question, passages)
    if backend == "ollama":
        resp = generate_with_ollama(prompt, model=model)
    elif backend == "hf":
        resp = generate_with_hf(prompt, model_id=model, hf_token=hf_token)
    else:
        raise ValueError("backend must be 'ollama' or 'hf'")
    return resp, passages

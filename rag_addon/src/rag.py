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

class TfIdfRetriever:
    def __init__(self, min_df=2, max_df=0.9, ngram_range=(1,2)):
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english",
                                   min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        self.X = None
        self.meta = None
        self._texts = None

    def fit(self, corpus: CorpusChunks):
        self._texts = corpus.df["text"].fillna("").tolist()
        self.meta = corpus.df[["entry_index","chapter_number","narrator","date_iso"]].to_dict("records")
        self.X = self.vec.fit_transform(self._texts)
        return self

    def search(self, query: str, topk: int = 5) -> List[Dict[str,Any]]:
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.X).ravel()
        idx = np.argsort(-sims)[:topk]
        out = []
        for i in idx:
            m = dict(self.meta[i])
            m["score"] = float(sims[i])
            m["text"] = self._texts[i]
            out.append(m)
        return out

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

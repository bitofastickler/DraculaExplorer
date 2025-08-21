# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

CANON_CHARS = ["Dracula","Jonathan","Mina","Lucy","Seward","Van Helsing","Arthur","Quincey","Renfield"]

@dataclass
class Corpus:
    df: pd.DataFrame

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def load_corpus(data_dir: str | Path) -> Corpus:
    data_dir = Path(data_dir)
    entries_path = data_dir / "dracula_corrected_entries.json"
    chunks_path  = data_dir / "dracula_ascii_rag.json"
    if entries_path.exists():
        data = _read_json(entries_path); rows = data["entries"]
        df = pd.DataFrame(rows)
        if "entry_index" not in df.columns: df.insert(0, "entry_index", np.arange(1, len(df)+1))
        keep = ["entry_index","chapter_number","narrator","date_iso","text"]
        for k in keep:
            if k not in df.columns: df[k] = None
        return Corpus(df[keep].copy())
    elif chunks_path.exists():
        rows = _read_json(chunks_path); chdf = pd.DataFrame(rows)
        # Ensure chunk order is stable before joining
        chdf = chdf.sort_values(["entry_index","chunk_index"])
        grp = chdf.groupby("entry_index").agg({"chapter_number":"first","narrator":"first","date_iso":"first","text": lambda s: "\n".join(s.tolist())}).reset_index()
        grp = grp[["entry_index","chapter_number","narrator","date_iso","text"]]
        return Corpus(grp)
    else:
        raise FileNotFoundError("Place dracula_corrected_entries.json or dracula_ascii_rag.json under data/." )

def build_tfidf(corpus: Corpus, min_df: int = 2, max_df: float = 0.9, ngram_range=(1,2)) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = vec.fit_transform(corpus.df["text"].fillna(""))
    return vec, X

def nmf_topics(X, n_topics: int = 8, random_state: int = 42):
    model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)
    return model, W

def kmeans_clusters(X, k: int = 8, random_state: int = 42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels

def top_terms_per_topic(model: NMF, vec: TfidfVectorizer, topn: int = 12) -> Dict[int, List[tuple]]:
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for t in range(model.components_.shape[0]):
        row = model.components_[t]
        idx = np.argsort(row)[::-1][:topn]
        out[t] = [(vocab[i], float(row[i])) for i in idx]
    return out

def chapter_similarity(corpus: Corpus, vec: TfidfVectorizer, X):
    df = corpus.df.copy()
    df["chapter_number"] = df["chapter_number"].fillna(-1).astype(int)
    chapters = sorted([c for c in df["chapter_number"].unique() if c >= 1])
    mats, labels = [], []
    for c in chapters:
        rows = df[df["chapter_number"]==c].index.values
        if len(rows)==0: continue
        mat = X[rows].mean(axis=0)
        mats.append(mat); labels.append(f"Ch {c}")
    M = np.vstack([m.toarray() if hasattr(m,"toarray") else np.asarray(m) for m in mats])
    sim = cosine_similarity(M)
    return sim, labels

def chapter_map(corpus, vec, X, n_components=2):
    df = corpus.df.copy()
    chapters = sorted(df["chapter_number"].dropna().astype(int).unique().tolist())
    mats = []
    for c in chapters:
        rows = df[df["chapter_number"] == c].index.values
        avg = X[rows].mean(axis=0)         # returns a numpy.matrix
        vec2d = np.asarray(avg)            # -> ndarray
        mats.append(vec2d.ravel())         # 1D
    M = np.vstack(mats)
    pca = PCA(n_components=n_components, random_state=42)
    pts = pca.fit_transform(M)
    return pts, [f"Ch {c}" for c in chapters]

def topic_timeline(W, corpus: Corpus, n_topics: int):
    df = corpus.df.copy()
    df["chapter_number"] = df["chapter_number"].fillna(-1).astype(int)
    import pandas as pd
    weights = pd.DataFrame(W, columns=[f"topic_{i}" for i in range(n_topics)])
    merged = pd.concat([df[["chapter_number"]].reset_index(drop=True), weights], axis=1)
    agg = merged.groupby("chapter_number").mean().sort_index()
    agg = agg[agg.index >= 1]
    return agg

def character_cooccurrence(corpus: Corpus):
    names = CANON_CHARS
    df = corpus.df.copy()
    def present_set(txt: str) -> set:
        low = txt.lower() if isinstance(txt,str) else ""
        s = set()
        for n in names:
            if n.lower() in low: s.add(n)
        return s
    sets = df["text"].apply(present_set)
    mat = {a: {b:0 for b in names} for a in names}
    for s in sets:
        for a in s:
            for b in s:
                if a!=b: mat[a][b] += 1
    import pandas as pd
    co = pd.DataFrame(mat).fillna(0).astype(int).T
    return co

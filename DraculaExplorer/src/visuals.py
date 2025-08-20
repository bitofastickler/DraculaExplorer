# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def plot_topic_terms(topic_idx: int, topics: Dict[int, List[Tuple[str,float]]]):
    terms = topics.get(topic_idx, [])
    labels = [w for w,_ in terms][::-1]
    vals = [v for _,v in terms][::-1]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.barh(labels, vals)
    ax.set_title(f"Top terms – Topic {topic_idx}")
    ax.set_xlabel("weight")
    fig.tight_layout(); return fig

def plot_similarity_heatmap(sim: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(sim)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Chapter similarity (cosine)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); return fig

def plot_topic_timeline(df_timeline: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,4))
    x = df_timeline.index.values
    for col in df_timeline.columns:
        ax.plot(x, df_timeline[col], label=col)
    ax.set_xlabel("Chapter"); ax.set_ylabel("Mean topic weight")
    ax.set_title("Topic prevalence across chapters")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout(); return fig

def plot_chapter_map(points: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(points[:,0], points[:,1])
    for i, lab in enumerate(labels):
        ax.annotate(lab, (points[i,0], points[i,1]))
    ax.set_title("Chapter map (TF-IDF → PCA)")
    fig.tight_layout(); return fig

def plot_cooccurrence_heatmap(co: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(co.values)
    ax.set_xticks(range(co.shape[1])); ax.set_yticks(range(co.shape[0]))
    ax.set_xticklabels(co.columns, rotation=45, ha="right")
    ax.set_yticklabels(co.index)
    ax.set_title("Character co-occurrence (by entry)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); return fig

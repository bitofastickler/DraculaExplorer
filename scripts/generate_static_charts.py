# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from src.pipeline import load_corpus, build_tfidf, nmf_topics, top_terms_per_topic, chapter_similarity, chapter_map, topic_timeline, character_cooccurrence
from src.visuals import plot_topic_terms, plot_similarity_heatmap, plot_topic_timeline, plot_chapter_map, plot_cooccurrence_heatmap

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

def main(n_topics=8):
    corpus = load_corpus(DATA_DIR)
    vec, X = build_tfidf(corpus)
    nmf, W = nmf_topics(X, n_topics=n_topics)
    topics = top_terms_per_topic(nmf, vec, topn=12)

    fig = plot_topic_terms(0, topics); fig.savefig(OUT/"topic_terms_t0.png", bbox_inches="tight")
    sim, labels = chapter_similarity(corpus, vec, X)
    fig = plot_similarity_heatmap(sim, labels); fig.savefig(OUT/"chapter_similarity.png", bbox_inches="tight")
    timeline = topic_timeline(W, corpus, n_topics=n_topics)
    fig = plot_topic_timeline(timeline); fig.savefig(OUT/"topic_timeline.png", bbox_inches="tight")
    pts, labs = chapter_map(corpus, vec, X)
    fig = plot_chapter_map(pts, labs); fig.savefig(OUT/"chapter_map.png", bbox_inches="tight")
    co = character_cooccurrence(corpus)
    fig = plot_cooccurrence_heatmap(co); fig.savefig(OUT/"character_cooccurrence.png", bbox_inches="tight")
    print("Saved charts to", OUT)

if __name__ == "__main__":
    main()

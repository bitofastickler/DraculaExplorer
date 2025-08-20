# -*- coding: utf-8 -*-
import gradio as gr
from pathlib import Path
from src.pipeline import load_corpus, build_tfidf, nmf_topics, top_terms_per_topic, chapter_similarity, chapter_map, topic_timeline
from src.visuals import plot_topic_terms, plot_similarity_heatmap, plot_chapter_map, plot_topic_timeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
_state = {"ready": False}

def ensure_models(n_topics=8):
    if _state.get("ready"): return
    corpus = load_corpus(DATA_DIR)
    vec, X = build_tfidf(corpus); nmf, W = nmf_topics(X, n_topics=n_topics)
    topics = top_terms_per_topic(nmf, vec, topn=12)
    sim, labels = chapter_similarity(corpus, vec, X)
    pts, labs = chapter_map(corpus, vec, X)
    timeline = topic_timeline(W, corpus, n_topics=n_topics)
    _state.update(dict(ready=True, corpus=corpus, vec=vec, X=X, nmf=nmf, W=W, topics=topics, sim=sim, labels=labels, pts=pts, labs=labs, timeline=timeline))

def viz_picker(viz_name, topic_idx):
    ensure_models()
    if viz_name == "Topic terms": return plot_topic_terms(int(topic_idx), _state["topics"])
    if viz_name == "Similarity heatmap": return plot_similarity_heatmap(_state["sim"], _state["labels"])
    if viz_name == "Topic timeline": return plot_topic_timeline(_state["timeline"])
    if viz_name == "Chapter map": return plot_chapter_map(_state["pts"], _state["labs"])
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Dracula Explorer – Quick Demo")
    with gr.Row():
        viz = gr.Dropdown(["Topic terms","Similarity heatmap","Topic timeline","Chapter map"], value="Topic terms", label="Visualization")
        topic = gr.Slider(0, 9, value=0, step=1, label="Topic index")
    out = gr.Plot(label="Figure")
    gr.Button("Render").click(viz_picker, [viz, topic], out)

if __name__ == "__main__":
    demo.launch()

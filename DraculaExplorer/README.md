<img width="329" height="504" alt="image" src="https://github.com/user-attachments/assets/570ba84d-b9cb-4a11-812d-0acf2d7129f9" />

# Dracula Explorer (in progress)

Public-domain NLP/RAG mini-project exploring Bram Stoker's *Dracula*. This repo ships a clean JSON corpus
(chapter titles + narrator attribution) and a tiny analysis stack (TF–IDF retrieval, topic modeling, clustering,
and character co‑occurrence) with static README visuals **and** a lightweight Gradio demo.

---

## Repo layout
```
/src
  pipeline.py           # builds TF-IDF, topics, clusters, co-occurrence
  visuals.py            # returns matplotlib Figures
/scripts
  generate_static_charts.py   # writes PNGs to assets/charts
/assets/charts          # PNGs embedded below
/data                   # place JSON here (see below)
app.py                  # Gradio demo
README.md
requirements.txt
```

## Data inputs

Place **one** of the following into `data/`:

- `dracula_corrected_entries.json` ({"entries": [...]} including chapter_number, narrator, text, date_iso)
- or `dracula_ascii_rag.json` (array of chunks with `entry_index`, `chapter_number`, `narrator`, `text`, etc.)

> For quick start, copy the files you generated earlier into `data/`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python scripts/generate_static_charts.py      # writes PNGs to assets/charts
python app.py                                 # launches Gradio demo
```

## Textual Analysis Visuals

### Topic prevalence across chapters
![Topic timeline](assets/charts/topic_timeline.png)

**What it is:** Average topic weights per chapter plotted over chapter index.

**How to read:** Peaks show where a topic dominates; watch for hand‑offs where one topic falls as another rises (often narratorial or plot pivots).

**Caveat:** Vocabulary‑based; great for structure, not literal plot truth.


### Chapter similarity (cosine)
![Similarity heatmap](assets/charts/chapter_similarity.png)

**What it is:** Cosine similarity between chapter TF‑IDF vectors, aggregated by chapter. Brighter = more similar.

**How to read:** The diagonal is 1.0 (self-similarity). Bright off‑diagonal blocks suggest runs of similar chapters (same narrator/setting); dark bands imply stylistic outliers.

**Why it matters:** Quick scan for arcs, section boundaries, and outlier chapters that merit commentary.



### 2D map of chapters (TF-IDF → PCA)
![2D map](assets/charts/chapter_map.png)

**What it is:** Each chapter is turned into a TF‑IDF vector of distinctive words/phrases; PCA projects those vectors to 2D for visualization.

**How to read:** Points close together use similar vocabulary; far‑apart points are lexically different. Outliers often mark narrator shifts or unusual modes (e.g., medical notes vs. travel).

**Why it matters:** Fast way to see clusters of chapters that “talk alike” and to spot structural pivots.



### Character co-occurrence (by entry)
![Character co-occurrence](assets/charts/character_cooccurrence.png)

**What it is:** For each entry (our atomic unit), we detect which canonical character names appear, then count pairwise co‑mentions across entries.

**How to read:** Brighter cells mean two characters are frequently mentioned together.  
**Caveat:** Simple name matching only (no coreference); pronouns aren’t resolved.

**Why it matters:** Rough map of who tends to share scenes/plot beats.

---
## Notes

- Charts use **matplotlib only**.
- Defaults are sane for a small corpus; tune topic count `n_topics`, cluster `k`, etc. in `scripts/generate_static_charts.py`.
- Optional future: Sentence-Transformers + FAISS/Chroma for semantic retrieval; kept out of the core for simplicity.

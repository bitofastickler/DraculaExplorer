# -*- coding: utf-8 -*-
import gradio as gr
from src.rag import answer

def run_rag(q, topk, backend, model):
    if not q:
        return "Ask a question about Dracula.", []
    resp, passages = answer(q, topk=int(topk), backend=backend, model=model)
    rows = [[i+1, p["entry_index"], p["chapter_number"], p["narrator"], p["date_iso"], p["score"], p["text"][:320]]
            for i,p in enumerate(passages)]
    return resp, rows

with gr.Blocks() as demo:
    gr.Markdown("# Dracula RAG – TF-IDF + LLM (Ollama/HF)")
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="e.g., What does Harker describe in his first entry?")
    with gr.Row():
        topk = gr.Slider(1, 8, value=4, step=1, label="Top-k passages")
        backend = gr.Dropdown(["ollama","hf"], value="ollama", label="LLM backend")
        model = gr.Textbox(value="llama3.2:3b", label="Model (Ollama tag or HF model id)")
    go = gr.Button("Ask")
    ans = gr.Markdown(label="Answer")
    tbl = gr.Dataframe(headers=["#","entry","chapter","narrator","date","score","preview"], wrap=True)
    go.click(run_rag, [q, topk, backend, model], [ans, tbl])

if __name__ == "__main__":
    demo.launch()

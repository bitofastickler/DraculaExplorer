# -*- coding: utf-8 -*-
import argparse
from src.rag import answer

def main():
    ap = argparse.ArgumentParser(description="Dracula RAG (TF-IDF + local Ollama or HF)")
    ap.add_argument("--q", "--question", dest="question", required=True, help="Your question")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--backend", choices=["ollama","hf"], default="ollama")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama tag (e.g., llama3.2:3b) or HF model id")
    ap.add_argument("--data", default="data", help="Data directory")
    args = ap.parse_args()

    resp, passages = answer(args.question, data_dir=args.data, topk=args.topk,
                            backend=args.backend, model=args.model)
    print("\n=== ANSWER ===\n")
    print(resp.strip())
    print("\n=== SOURCES ===\n")
    for i, p in enumerate(passages, 1):
        meta = f"(entry {p['entry_index']} | ch {p['chapter_number']} | {p['narrator']} | {p['date_iso']})"
        preview = p['text'][:400].replace('\n',' ')
        print(f"[{i}] {meta}\n{preview}\n")

if __name__ == "__main__":
    main()

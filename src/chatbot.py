# -*- coding: utf-8 -*-
"""
Chat wrapper around the RAG pipeline (TF-IDF + Ollama).
Keeps a lightweight chat history and builds a grounded prompt per turn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .rag import load_chunks, TfIdfRetriever, generate_with_ollama, generate_with_hf
import numpy as np

@dataclass
class ChatTurn:
    role: str   # "user" | "assistant"
    content: str

@dataclass
class ChatState:
    history: List[ChatTurn] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.history.append(ChatTurn(role=role, content=content))

    def last_n_as_text(self, n_pairs: int = 4, max_chars: int = 1500) -> str:
        pairs = []
        u, a = None, None
        for turn in reversed(self.history):
            if turn.role == "assistant" and a is None:
                a = turn.content
            elif turn.role == "user":
                u = turn.content
                if u is not None:
                    pairs.append((u, a or ""))
                    u, a = None, None
            if len(pairs) >= n_pairs:
                break
        pairs.reverse()
        lines = []
        for u, a in pairs:
            if u: lines.append(f"User: {u}")
            if a: lines.append(f"Assistant: {a}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[-max_chars:]
        return out

def build_chat_system_preamble() -> str:
    return ("You are a concise literary assistant helping with Bram Stoker's Dracula.\n"
            "Ground every answer in the provided CONTEXT passages and cite them like [1], [2] where relevant.\n"
            "If the answer is not in CONTEXT, say so briefly.\n"
            "Keep answers short unless the user requests detail.\n")

def build_chat_prompt(question: str, chat_context: str, passages: List[Dict[str,Any]]) -> str:
    head = build_chat_system_preamble()
    convo = f"\nRECENT CONVERSATION (for tone/continuity, not as evidence):\n{chat_context}\n" if chat_context else ""
    ctx_lines = []
    for j, p in enumerate(passages, 1):
        meta = f"(entry {p.get('entry_index')} | ch {p.get('chapter_number')} | {p.get('narrator')} | {p.get('date_iso')})"
        ctx_lines.append(f"[{j}] {meta}\n{p['text']}")
    ctx = "\n\n".join(ctx_lines)
    instr = ("\n\nAnswer the QUESTION using the CONTEXT above. "
             "Cite passage numbers like [1], [2] when you quote or rely on them.\n")
    return f"{head}{convo}\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n{instr}ANSWER:"

def ask_chat(question: str,
             state: ChatState,
             topk: int = 4,
             backend: str = "ollama",
             model: str = "llama3.2:3b",
             data_dir: str | Path = "data"):
    # Build minimal convo context (style only)
    recent = state.last_n_as_text(n_pairs=4, max_chars=1500)

    # Retrieval
    corpus = load_chunks(data_dir)
    retr = TfIdfRetriever().fit(corpus)
    passages = retr.search(question, topk=topk)

    # Prompt
    prompt = build_chat_prompt(question, recent, passages)

    # Generate
    if backend == "ollama":
        text = generate_with_ollama(prompt, model=model)
    elif backend == "hf":
        text = generate_with_hf(prompt, model_id=model)
    else:
        raise ValueError("backend must be 'ollama' or 'hf'")

    return text, passages

# -*- coding: utf-8 -*-
"""
Chat wrapper around the RAG pipeline (TF-IDF + Ollama).
Keeps a lightweight chat history and builds a grounded prompt per turn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .rag import load_chunks, TfIdfRetriever, generate_with_ollama, generate_with_hf, extract_constraints
import numpy as np
import re, json

GLOBAL_PAT = re.compile(
    r"(lucy).*(vampir|bite|stake|bloofer|coffin|kill|slay)|"
    r"(van\s*helsing).*(kill|stake)|"
    r"(did|does)\s+dracula", re.I)

def _coerce_json(s: str):
    s = s.strip()
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.I)
    # try straight parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # fallback: extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise

def _render_answer_first(struct, passages):
    n = len(passages)
    # collect clean evidence (2–4 items max)
    ev = []
    for item in struct.get("evidence", [])[:4]:
        claim = (item.get("claim") or item.get("quote") or "").strip()
        srcs  = [i for i in item.get("sources", []) if isinstance(i, int) and 1 <= i <= n]
        if claim and srcs:
            claim = re.sub(r"\s+", " ", claim).strip()[:240]
            ev.append((claim, srcs))

    ans = (struct.get("answer") or "").strip()
    lines = []
    if ans:
        lines += [f"Answer: {ans}"]
    if ev:
        lines.append("Evidence:")
        for claim, srcs in ev:
            cites = "".join(f"[{i}]" for i in srcs)
            lines.append(f"- {claim} {cites}")
    return "\n".join(lines) if lines else "No supported answer found."

def build_json_prompt(question, chat_context, passages):
    if not passages:
        # Degenerate prompt: tell the model to say it can't answer
        return json.dumps({
            "answer": "The provided context is empty, so I cannot answer.",
            "evidence": []
        })
    n = len(passages)
    head = (
        "You are a concise assistant. Use ONLY the CONTEXT.\n"
        "Return ONLY valid JSON (no markdown) with keys:\n"
        '{"answer": str, "evidence": [{"claim": str, "sources": [int, ...]}]}\n'
        "Requirements:\n"
        "- 'answer': 1–3 sentences that directly answer the question, summary first.\n"
        f"- Provide 2–4 evidence bullets. Each must include 'sources' integers in 1..{n} mapping to the CONTEXT blocks.\n"
        "- Each claim ≤ 240 chars; quote or paraphrase is fine.\n"
        "- Do NOT add extra keys or text outside JSON.\n"
    )
    convo = f"\n(Recent conversation for tone only):\n{chat_context}\n" if chat_context else ""
    ctx = []
    for j, p in enumerate(passages, 1):
        meta = f"(entry {p.get('entry_index')} | ch {p.get('chapter_number')} | {p.get('narrator')} | {p.get('date_iso')})"
        ctx.append(f"[{j}] {meta}\n{p['text']}")
    return head + "\nCONTEXT:\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {question}\nJSON:"

def is_global_fact(q: str) -> bool:
    return bool(GLOBAL_PAT.search(q))

def build_evidence_prompt(question, chat_context, passages):
    head = ("Use ONLY the CONTEXT.\n"
            "First list 2–5 EVIDENCE bullets with [#] cites (quote or paraphrase).\n"
            "Then write a 1–3 sentence ANSWER strictly from those bullets.\n")
    convo = f"\n(Recent conversation for tone only):\n{chat_context}\n" if chat_context else ""
    ctx = []
    for j, p in enumerate(passages, 1):
        meta = f"(entry {p.get('entry_index')} | ch {p.get('chapter_number')} | {p.get('narrator')} | {p.get('date_iso')})"
        ctx.append(f"[{j}] {meta}\n{p['text']}")
    return f"{head}{convo}\nCONTEXT:\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {question}\n\nFormat:\nEVIDENCE:\n- … [#]\n- … [#]\nANSWER:"

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

def ask_chat(question, state, topk=4, backend="ollama", model="llama3.2:3b", data_dir="data"):
    recent = state.last_n_as_text(n_pairs=4, max_chars=1500)

    corpus = load_chunks(data_dir)
    retr = TfIdfRetriever().fit(corpus)

    global_mode = is_global_fact(question)
    hits = retr.search(question, topk=max(topk, 6) if global_mode else topk)

    passages = []
    if global_mode:
        # Take several distinct entries across chapters (arc view)
        used_entries, used_chapters = set(), set()
        for h in hits:
            ch = h["chapter_number"]
            if h["entry_index"] in used_entries or ch in used_chapters:
                continue
            used_entries.add(h["entry_index"]); used_chapters.add(ch)
            bundle = retr.entry_context(h["entry_index"], window=2, max_chars=1400)
            passages.append({
                "entry_index": h["entry_index"],
                "chapter_number": ch,
                "narrator": h["narrator"],
                "date_iso": h["date_iso"],
                "text": bundle,
                "score": h.get("score", 0.0),
            })
            if len(passages) >= 3:  # 3 bundles is plenty for a 3B model
                break
        # Add 1–2 chapter snapshots
        for tc in retr.top_chapter_contexts(question, topn=2, max_chars=1200):
            passages.append({
                "entry_index": None,
                "chapter_number": tc["chapter_number"],
                "narrator": None,
                "date_iso": None,
                "text": tc["text"],
                "score": tc["score"],
            })
    else:
        # Normal mode: single best entry bundle + top chapter
        if hits:
            h0 = hits[0]
            bundle = retr.entry_context(h0["entry_index"], window=2, max_chars=2000)
            passages.append({
                "entry_index": h0["entry_index"],
                "chapter_number": h0["chapter_number"],
                "narrator": h0["narrator"],
                "date_iso": h0["date_iso"],
                "text": bundle,
                "score": h0.get("score", 0.0),
            })
        for tc in retr.top_chapter_contexts(question, topn=1, max_chars=1600):
            passages.append({
                "entry_index": None,
                "chapter_number": tc["chapter_number"],
                "narrator": None,
                "date_iso": None,
                "text": tc["text"],
                "score": tc["score"],
            })

    prompt = build_json_prompt(question, recent, passages)
    raw = generate_with_ollama(prompt, model=model) if backend == "ollama" else generate_with_hf(prompt, model_id=model)
    try:
        obj = _coerce_json(raw)
        text = _render_answer_first(obj, passages)
    except Exception:
        # fallback: show raw if model returned non-JSON
        text = raw
    return text, passages


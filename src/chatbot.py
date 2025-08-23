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

_RETR = None

GLOBAL_PAT = re.compile(
    r"(lucy).*(vampir|bite|stake|bloofer|coffin|kill|slay)|"
    r"(van\s*helsing).*(kill|stake)|"
    r"(did|does)\s+dracula", re.I)

def _get_retriever(data_dir: str):
    global _RETR
    if _RETR is None:
        corpus = load_chunks(data_dir)
        _RETR = TfIdfRetriever().fit(corpus)
    return _RETR

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
    seen = set()
    ev = []
    for item in struct.get("evidence", [])[:6]:
        claim = (item.get("claim") or item.get("quote") or "").strip()
        srcs  = tuple(i for i in item.get("sources", []) if isinstance(i, int) and 1 <= i <= n)
        if claim and srcs and (claim, srcs) not in seen:
            seen.add((claim, srcs))
            claim = re.sub(r"\s+", " ", claim).strip()[:240]
            ev.append((claim, srcs))
        if len(ev) >= 3:  # show up to 3
            break
    ans = (struct.get("answer") or "").strip()
    lines = [f"Answer: {ans}"] if ans else []
    if ev:
        lines.append("Evidence:")
        for claim, srcs in ev:
            lines.append(f"- {claim} " + "".join(f"[{i}]" for i in srcs))
    return "\n".join(lines) if lines else "No supported answer found."

def build_json_prompt(question, chat_context, passages):
    """
    Ask the model for JSON ONLY:
      { "answer": str, "evidence": [ { "claim": str, "sources": [int, ...] }, ... ] }
    The UI will render: Answer first, then Evidence bullets with [#] cites.
    """
    n = len(passages)
    # Safety: if no passages, force a do-nothing JSON so caller can handle nicely.
    if n == 0:
        return (
            '["answer": "Unknown based on the provided context.", "evidence": []]'
        )

    rules = (
        "You are answering questions about Bram Stoker's Dracula using ONLY the provided CONTEXT.\n"
        "Return JSON ONLY (no markdown, no prose before/after). Schema:\n"
        '{ "answer": str, "evidence": [ { "claim": str, "sources": [int, ...] }, ... ] }\n'
        "Requirements:\n"
        f"- sources are integers 1..{n} referring to the numbered CONTEXT blocks.\n"
        "- answer: 1–3 sentences, summary first, directly answering the question.\n"
        "- evidence: 2–4 items. Each item is ≤ 200 chars, either a short quote or precise paraphrase.\n"
        "- Every evidence item MUST include at least one valid source id.\n"
        "- If the context is insufficient, set answer to 'Hmmm, I'm not sure. Can you try asking the question in a different way?' and return evidence: [].\n"
        "- Prefer explicit, conclusive passages over hints. When multiple passages describe a sequence, summarize the outcome plainly.\n"
        "- Do NOT invent details, dates, or actions not present in the CONTEXT.\n"
        "- Output MUST be valid JSON. Do NOT wrap in ```json fences or add extra keys."
    )

    # Recent chat is tone only, never evidence. Keep short so we don't waste tokens.
    convo = f"\n(Conversation context; NOT evidence):\n{chat_context}\n" if chat_context else ""

    ctx_lines = []
    for j, p in enumerate(passages, 1):
        meta = f"(entry {p.get('entry_index')} | ch {p.get('chapter_number')} | {p.get('narrator')} | {p.get('date_iso')})"
        # Keep context blocks compact; they’re numbered so the model can cite them.
        ctx_lines.append(f"[{j}] {meta}\n{p['text']}")

    ctx = "\n\n".join(ctx_lines)
    return (
        f"{rules}"
        f"{convo}\n"
        f"CONTEXT (numbered 1..{n}):\n{ctx}\n\n"
        f"QUESTION: {question}\n"
        "JSON:"
    )

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

def ask_chat(
    question: str,
    state,
    topk: int = 4,
    backend: str = "ollama",
    model: str = "llama3.2:3b",
    data_dir: str = "data",
):
    """
    Closed-book first, RAG second.

    Step A (closed-book): Ask the model from its own knowledge and get {answer, confidence}.
      - If confidence >= threshold and the user didn't request citations -> return immediately (no RAG).

    Step B (grounded): Otherwise, retrieve from the Dracula JSON and answer with citations
      (Answer → Evidence [#]) using your build_json_prompt.
    """
    import re, json
    from typing import List, Dict, Any

    # -------- knobs --------
    CB_CONF_THRESH = 0.70  # skip RAG when closed-book >= this (lower if you want more CB)
    wants_sources = bool(re.search(r"\b(cite|citation|sources?|quote|chapter|where|passage|evidence)\b",
                                   question, re.I))

    # Model-aware context/temperature (helps larger local models like gpt-oss:20b)
    mlow = model.lower()
    is_large = any(tok in mlow for tok in ("20b", "13b", "12b", "9b", "8b", "7b"))
    # retrieval bundle sizing
    bundle_window = 3 if is_large else 2
    max_bundles   = 4 if is_large else 3
    chap_topn     = 2
    bundle_chars  = 1300 if is_large else 1400
    chap_chars    = 1100 if is_large else 1200

    # -------- helpers --------
    def _gen(prompt_text: str) -> str:
        if backend == "ollama":
            opts = {"num_ctx": 4096 if is_large else 3072, "temperature": 0.15}
            try:
                return generate_with_ollama(prompt_text, model=model, options=opts)  # type: ignore[arg-type]
            except TypeError:
                # older helper without options support
                return generate_with_ollama(prompt_text, model=model)
        else:
            return generate_with_hf(prompt_text, model_id=model)

    def _closed_book_prompt(q: str) -> str:
        return (
            "Answer from your general knowledge of Bram Stoker's Dracula (no external text is provided). "
            "Keep it concise (1–3 sentences). "
            "Return JSON ONLY (no markdown): "
            '{ "answer": str, "confidence": float } '
            "Where confidence is in [0,1].\n"
            f"QUESTION: {q}\nJSON:"
        )

    # -------- A) closed-book probe --------
    recent = state.last_n_as_text(n_pairs=4, max_chars=1500)

    proposed_answer, cb_conf = None, 0.0
    try:
        raw_cb = _gen(_closed_book_prompt(question))
        obj_cb = _coerce_json(raw_cb)
        proposed_answer = (obj_cb.get("answer") or "").strip()
        cb_conf = float(obj_cb.get("confidence") or 0.0)
    except Exception:
        # If model returned non-JSON, accept a reasonable plaintext as a low-confidence guess
        raw_cb = locals().get("raw_cb", "")
        guess = raw_cb.strip()
        if len(guess) > 40:
            proposed_answer, cb_conf = guess, 0.50

    # --- short-circuit if closed-book is confident and user didn't ask for citations ---
    if proposed_answer and cb_conf >= CB_CONF_THRESH and not wants_sources:
        return f"Answer: {proposed_answer}", []  # skip RAG entirely

    # -------- B) grounded answer with RAG --------
    # Cache retriever on the function object for speed
    try:
        _cache = ask_chat.__dict__
        retr = _cache.get("_retr")
        if retr is None or _cache.get("_data_dir") != data_dir:
            retr = TfIdfRetriever().fit(load_chunks(data_dir))
            _cache["_retr"] = retr
            _cache["_data_dir"] = data_dir
    except Exception:
        retr = TfIdfRetriever().fit(load_chunks(data_dir))

    global_mode = is_global_fact(question)

    # Main hits
    hits = retr.search(question, topk=max(topk, 6) if global_mode else topk)

    passages: List[Dict[str, Any]] = []
    if global_mode:
        # Gather distinct entry bundles across chapters (to capture an arc)
        used_entries, used_chapters = set(), set()
        for h in hits:
            ch = h.get("chapter_number")
            if h["entry_index"] in used_entries or ch in used_chapters:
                continue
            bundle = retr.entry_context(h["entry_index"], window=bundle_window, max_chars=bundle_chars)
            used_entries.add(h["entry_index"]); used_chapters.add(ch)
            passages.append({
                "entry_index": h["entry_index"],
                "chapter_number": ch,
                "narrator": h.get("narrator"),
                "date_iso": h.get("date_iso"),
                "text": bundle,
                "score": h.get("score", 0.0),
            })
            if len(passages) >= max_bundles:
                break
        # add chapter snapshots
        for tc in retr.top_chapter_contexts(question, topn=chap_topn, max_chars=chap_chars):
            passages.append({
                "entry_index": None,
                "chapter_number": tc["chapter_number"],
                "narrator": None,
                "date_iso": None,
                "text": tc["text"],
                "score": tc["score"],
            })
    else:
        # single best entry bundle + one chapter snapshot
        if hits:
            h0 = hits[0]
            bundle = retr.entry_context(h0["entry_index"], window=2, max_chars=2000)
            passages.append({
                "entry_index": h0["entry_index"],
                "chapter_number": h0.get("chapter_number"),
                "narrator": h0.get("narrator"),
                "date_iso": h0.get("date_iso"),
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

    # If retrieval fails, fall back to closed-book if we have it
    if not passages:
        if proposed_answer:
            return f"Answer: {proposed_answer}", []
        return "Answer: I couldn't retrieve any relevant context for that question.\nEvidence:", []

    # If we *do* have a closed-book guess, pass it as a proposal to verify/correct
    if proposed_answer and cb_conf >= 0.40:
        q_with_proposal = (
            f"{question}\n"
            f"Proposed answer (from general knowledge): {proposed_answer}\n"
            "Use the CONTEXT to verify this; if the proposal conflicts with the evidence, correct it succinctly."
        )
    else:
        q_with_proposal = question

    prompt = build_json_prompt(q_with_proposal, recent, passages)

    try:
        raw = _gen(prompt)
    except Exception as e:
        # If generation fails, still return the closed-book answer if we have one
        if proposed_answer:
            return f"Answer: {proposed_answer}", passages
        return f"Answer: Generation backend error — {type(e).__name__}: {e}", passages

    try:
        obj = _coerce_json(raw)
        text = _render_answer_first(obj, passages)  # Answer first, then Evidence bullets
    except Exception:
        # If JSON parse fails, prefer closed-book answer over raw blob
        text = f"Answer: {proposed_answer or raw.strip()}"

    return text, passages


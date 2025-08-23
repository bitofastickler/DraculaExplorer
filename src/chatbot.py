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
    Hybrid QA:
      1) Closed-book probe from model's native knowledge.
      2) Grounded verification with Dracula JSON (entry bundles + chapter snapshots).
    Renders: 'Answer: …' then 'Evidence:' bullets with [#] cites.
    Also expands the retrieval query and softly reranks for Lucy+Dracula co-mentions.
    """

    import re, json

    CB_CONF_THRESH = 0.70  # require decent confidence before trusting the proposal

    # --- model-awareness for context sizing ---
    mlow = model.lower()
    is_large = any(tok in mlow for tok in ("20b", "13b", "12b", "9b", "8b", "7b"))
    # global-mode knobs
    bundle_window = 3 if is_large else 2       # ± entries per bundle
    max_bundles   = 4 if is_large else 3       # number of entry bundles
    chap_topn     = 2                           # chapter snapshots to add
    bundle_chars  = 1300 if is_large else 1400  # per-bundle cap
    chap_chars    = 1100 if is_large else 1200  # per-chapter cap

    # --- small helpers ---

    def _closed_book_prompt(q: str) -> str:
        return (
            "Answer from your general knowledge of Bram Stoker's *Dracula* only. "
            "Do not assume any external context is available. "
            "Return ONLY valid JSON:\n"
            '{"answer": str, "confidence": float, "needs_context": bool, "verify_queries": [str, ...]}\n'
            "Rules:\n"
            "- confidence in [0,1]\n"
            "- needs_context = true if you are not sure or think evidence would help\n"
            "- verify_queries: 1–3 short phrases you would search for in the text (e.g., 'Lucy staking', 'Bloofer Lady', 'Chapter 16 tomb scene')\n"
            f"QUESTION: {q}\nJSON:"
        )

    def _gen(prompt_text: str) -> str:
        # If your generate_with_ollama() supports options, pass them; else fall back.
        if backend == "ollama":
            ollama_opts = {"num_ctx": 4096 if is_large else 3072, "temperature": 0.15}
            try:
                return generate_with_ollama(prompt_text, model=model, options=ollama_opts)  # type: ignore
            except TypeError:
                return generate_with_ollama(prompt_text, model=model)
        else:
            return generate_with_hf(prompt_text, model_id=model)

    # function-attribute cache for retriever
    try:
        _cache = ask_chat.__dict__
        retr = _cache.get("_retr")
        if retr is None or _cache.get("_data_dir") != data_dir:
            retr = TfIdfRetriever().fit(load_chunks(data_dir))
            _cache["_retr"] = retr
            _cache["_data_dir"] = data_dir
    except Exception:
        retr = TfIdfRetriever().fit(load_chunks(data_dir))

    recent = state.last_n_as_text(n_pairs=4, max_chars=1500)

    # --- Pass 1: closed-book probe ---
    proposed_answer = None
    cb_conf = 0.0
    cb_queries: List[str] = []

    try:
        raw_cb = _gen(_closed_book_prompt(question))
        obj_cb = _coerce_json(raw_cb)
        proposed_answer = (obj_cb.get("answer") or "").strip()
        cb_conf = float(obj_cb.get("confidence") or 0.0)
        cb_need = bool(obj_cb.get("needs_context"))
        cb_queries = [s for s in (obj_cb.get("verify_queries") or []) if isinstance(s, str)]
    except Exception:
        proposed_answer, cb_conf, cb_need, cb_queries = None, 0.0, True, []

    # --- retrieval (Pass 2) ---

    # Simple query expansion to catch synonyms used in the book
    LUCY_SYNS = "Lucy Westenra Miss Westenra Bloofer Lady"
    DRAC_SYNS = "Dracula the Count Count"
    VAMP_SYNS = "vampire vampiric bite blood garlic stake coffin tomb cemetery"
    expanded_q = " ".join([question] + cb_queries[:3] + [LUCY_SYNS, DRAC_SYNS, VAMP_SYNS])

    global_mode = is_global_fact(question)  # your existing classifier
    primary_topk = max(topk, 10) if global_mode else topk  # fish a bit wider for global facts
    hits = retr.search(expanded_q, topk=primary_topk)

    # Soft rerank: prefer bundles that co-mention Lucy & Dracula (not a hard filter)
    LUCY_RE = re.compile(r"\b(lucy|miss\s+westenra|westenra|bloofer\s+lady)\b", re.I)
    DRAC_RE = re.compile(r"\b(dracula|the\s+count|count)\b", re.I)
    VAMP_RE = re.compile(r"\b(vampir\w*|bite\w*|fangs?|blood|garlic|stake\w*|coffin\w*|tomb|grave|cemetery)\b", re.I)

    def _bonus_from_text(txt: str) -> float:
        has_l = bool(LUCY_RE.search(txt))
        has_d = bool(DRAC_RE.search(txt))
        has_v = bool(VAMP_RE.search(txt))
        bonus = 0.0
        if has_l and has_d: bonus += 0.25
        if has_v:           bonus += 0.10
        if has_l ^ has_d:   bonus += 0.06  # only one mentioned
        return bonus

    # Precompute small bundles for reranking (cheap; we need them anyway)
    decorated = []
    for h in hits:
        bundle_preview = retr.entry_context(h["entry_index"], window=2, max_chars=800)
        decorated.append((h, h.get("score", 0.0) + _bonus_from_text(bundle_preview), bundle_preview))

    decorated.sort(key=lambda t: t[1], reverse=True)

    passages: List[Dict[str, Any]] = []
    if global_mode:
        # A few distinct entry bundles across chapters (the arc)
        used_entries, used_chapters = set(), set()
        for h, _, preview in decorated:
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
        # Add chapter snapshots
        for tc in retr.top_chapter_contexts(expanded_q, topn=chap_topn, max_chars=chap_chars):
            passages.append({
                "entry_index": None,
                "chapter_number": tc["chapter_number"],
                "narrator": None,
                "date_iso": None,
                "text": tc["text"],
                "score": tc["score"],
            })

        # Deterministic safety net: ensure Chapter 16 for Lucy fate-style questions
        if re.search(r"\blucy\b.*\b(kill|stake|slay|bloofer)\b", question, re.I):
            have16 = any(p.get("chapter_number") == 16 for p in passages)
            if not have16:
                for tc in retr.top_chapter_contexts("Lucy Westenra staking Chapter 16 tomb", topn=3, max_chars=900):
                    if tc["chapter_number"] == 16:
                        passages.append({
                            "entry_index": None,
                            "chapter_number": 16,
                            "narrator": None,
                            "date_iso": None,
                            "text": tc["text"],
                            "score": tc["score"],
                        })
                        break

    else:
        # Single best entry bundle + 1 chapter snapshot
        if decorated:
            h0, _, _ = decorated[0]
            bundle = retr.entry_context(h0["entry_index"], window=2, max_chars=2000)
            passages.append({
                "entry_index": h0["entry_index"],
                "chapter_number": h0.get("chapter_number"),
                "narrator": h0.get("narrator"),
                "date_iso": h0.get("date_iso"),
                "text": bundle,
                "score": h0.get("score", 0.0),
            })
        for tc in retr.top_chapter_contexts(expanded_q, topn=1, max_chars=1600):
            passages.append({
                "entry_index": None,
                "chapter_number": tc["chapter_number"],
                "narrator": None,
                "date_iso": None,
                "text": tc["text"],
                "score": tc["score"],
            })

    if not passages:
        # Nothing to ground with — return closed-book if we have it, else a friendly miss
        if proposed_answer:
            return f"Answer: {proposed_answer}\nEvidence:", []
        return "Answer: I couldn't retrieve any relevant context for that question.\nEvidence:", []

    # --- Build grounded prompt (Pass 2)
    if proposed_answer and cb_conf >= CB_CONF_THRESH:
        q_with_proposal = (
            f"{question}\n"
            f"Proposed answer (from general knowledge): {proposed_answer}\n"
            "Use the CONTEXT to verify this; if the proposal conflicts with the evidence, correct it succinctly."
        )
    else:
        q_with_proposal = question

    prompt = build_json_prompt(q_with_proposal, recent, passages)

    # --- generation with grounding ---
    try:
        raw = _gen(prompt)
    except Exception as e:
        if proposed_answer:
            return f"Answer: {proposed_answer}\nEvidence:", passages
        return f"Answer: Generation backend error — {type(e).__name__}: {e}", passages

    # --- parse & render (answer first, then bullets) ---
    try:
        obj = _coerce_json(raw)
        text = _render_answer_first(obj, passages)
    except Exception:
        text = f"Answer: {proposed_answer or raw.strip()}"

    return text, passages

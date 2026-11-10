"""Document summarisation: LLM map-reduce when available, extractive fallback."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.chunking import split_text
from core.document_loader import Document
from core.llm import LLMError
from core.preprocessing import split_sentences

MODES = ("short", "detailed", "key_points", "chapter")
MAX_PART_CHARS = 8000
MAX_PARTS = 12

_INSTRUCTIONS = {
    "short": "Write a concise summary of 3-5 sentences.",
    "detailed": "Write a detailed, well-structured summary (several paragraphs) covering all main topics.",
    "key_points": "Extract the key points as a bulleted list of 6-10 short bullets.",
    "chapter": "Write a 2-4 sentence summary of this section.",
}


def _select(docs: List[Document], page_range: Optional[Tuple[int, int]]) -> List[Document]:
    if not page_range or not any(page_range):
        return docs
    lo, hi = page_range
    lo = max(lo or 1, 1)
    hi = hi or 10**9
    return [d for d in docs if d.page is None or lo <= d.page <= hi]


def _by_source(docs: List[Document]) -> Dict[str, List[Document]]:
    out: Dict[str, List[Document]] = {}
    for d in docs:
        out.setdefault(d.source, []).append(d)
    return out


# ---------- extractive fallback ----------
def extractive_summary(text: str, n: int) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = [s for s in split_sentences(text) if 4 <= len(s.split()) <= 70]
    if len(sents) <= n:
        return sents
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:
        return sents[:n]
    lengths = np.array([max(len(s.split()), 1) for s in sents], dtype=float)
    scores = np.asarray(X.sum(axis=1)).ravel() / np.sqrt(lengths)
    scores[0] *= 1.15  # leading sentences often introduce the topic
    top = sorted(np.argsort(-scores)[:n])
    return [sents[i] for i in top]


def _extractive(text: str, mode: str) -> str:
    n_sents = len(split_sentences(text))
    if mode == "short":
        return " ".join(extractive_summary(text, 5))
    if mode == "detailed":
        n = int(min(25, max(10, n_sents * 0.15)))
        return "\n\n".join(extractive_summary(text, n))
    if mode == "key_points":
        pts = extractive_summary(text, 8)
        return "\n".join("- " + (p if len(p) <= 240 else p[:237] + "...") for p in pts)
    return " ".join(extractive_summary(text, 3))


# ---------- LLM map-reduce ----------
def _llm_summary(text: str, mode: str, llm) -> str:
    parts = split_text(text, MAX_PART_CHARS, 0)
    if len(parts) > MAX_PARTS:  # regroup to cap API calls
        size = -(-len(parts) // MAX_PARTS)
        parts = ["\n\n".join(parts[i:i + size]) for i in range(0, len(parts), size)]
    sys_prompt = "You are a precise document summarisation assistant. Use only the given text."
    if len(parts) == 1:
        return llm.generate(f"{_INSTRUCTIONS[mode]}\n\nText:\n{parts[0]}", system=sys_prompt)
    partials = [
        llm.generate(f"Summarise this section of a longer document in 4-6 sentences.\n\nText:\n{p}",
                     system=sys_prompt)
        for p in parts
    ]
    combined = "\n\n".join(f"Section {i}: {s}" for i, s in enumerate(partials, 1))
    return llm.generate(
        f"{_INSTRUCTIONS[mode]}\n\nThese are summaries of consecutive sections of one document:\n{combined}",
        system=sys_prompt,
    )


# ---------- chapter detection ----------
_HEADING = re.compile(
    r"^(#{1,4}\s+.+|(chapter|section|part|unit|module)\s+[\w.]+.*|\d+(\.\d+)*[.)]?\s+[A-Z][^.!?]{2,70})$",
    re.I,
)


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not 3 <= len(line) <= 80 or line.endswith((".", ",", ";")):
        return False
    if _HEADING.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def split_sections(docs: List[Document]) -> List[Tuple[str, str]]:
    """Return [(title, text)] for ONE document's pages."""
    sections: List[Tuple[str, List[str]]] = []
    for d in docs:
        for line in d.text.split("\n"):
            if _is_heading(line):
                sections.append((line.strip().lstrip("# ").strip(), []))
            elif sections:
                sections[-1][1].append(line)
            else:
                sections.append(("Introduction", [line]))
    real = [(t, "\n".join(b).strip()) for t, b in sections if len("\n".join(b).split()) >= 15]
    if len(real) >= 2:
        return real
    # fallback: split by pages or into equal parts
    full = "\n".join(d.text for d in docs)
    paged = [d for d in docs if d.page is not None]
    if len(paged) >= 2:
        group = max(1, -(-len(paged) // 6))
        out = []
        for i in range(0, len(paged), group):
            g = paged[i:i + group]
            title = f"Page {g[0].page}" if len(g) == 1 else f"Pages {g[0].page}–{g[-1].page}"
            out.append((title, "\n".join(x.text for x in g)))
        return out
    parts = split_text(full, max(len(full) // 5 + 1, 600), 0)
    return [(f"Part {i}", p) for i, p in enumerate(parts, 1)]


# ---------- public API ----------
def summarize(docs: List[Document], mode: str = "short", llm=None,
              page_range: Optional[Tuple[int, int]] = None) -> Dict[str, str]:
    """Summarise documents. Returns {text, method, note}."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    docs = _select(docs, page_range)
    if not docs:
        return {"text": "No text found for the selected documents / page range.",
                "method": "none", "note": ""}
    note, use_llm = "", llm is not None
    blocks: List[str] = []
    for source, ds in _by_source(docs).items():
        text = "\n\n".join(d.text for d in ds)
        body = None
        if mode == "chapter":
            secs = split_sections(ds)
            rendered = []
            for title, sec_text in secs:
                s = None
                if use_llm:
                    try:
                        s = _llm_summary(sec_text, "chapter", llm)
                    except LLMError as exc:
                        use_llm, note = False, f"LLM unavailable ({exc}); used extractive summarisation."
                if s is None:
                    s = _extractive(sec_text, "chapter") or sec_text[:300]
                rendered.append(f"**{title}**\n\n{s}")
            body = "\n\n".join(rendered)
        else:
            if use_llm:
                try:
                    body = _llm_summary(text, mode, llm)
                except LLMError as exc:
                    use_llm, note = False, f"LLM unavailable ({exc}); used extractive summarisation."
            if body is None:
                body = _extractive(text, mode) or text[:500]
        blocks.append(f"### {source}\n\n{body}" if len(_by_source(docs)) > 1 else body)
    method = "llm" if use_llm else "extractive"
    if method == "extractive" and not note and llm is None:
        note = "No LLM API key configured: generated an extractive summary (key sentences from the text)."
    return {"text": "\n\n".join(blocks), "method": method, "note": note}

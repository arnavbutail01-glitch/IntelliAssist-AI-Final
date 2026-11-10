"""Sentiment analysis: Hugging Face model when available, lexicon fallback."""
from __future__ import annotations

import importlib.util
import logging
import re
from typing import Dict, List

from config.settings import SENTIMENT_BACKEND, SENTIMENT_MODEL
from core.document_loader import Document
from core.preprocessing import split_sentences

log = logging.getLogger(__name__)
_PIPE = None
_PIPE_FAILED = False

POSITIVE = {
    "excellent", "great", "good", "amazing", "awesome", "fantastic", "wonderful", "love", "loved",
    "helpful", "useful", "reliable", "fast", "efficient", "impressive", "best", "perfect", "easy",
    "improved", "successful", "benefit", "beneficial", "powerful", "accurate", "robust", "happy",
    "recommend", "outstanding", "superb", "smooth", "intuitive", "effective", "strong", "brilliant",
}
NEGATIVE = {
    "bad", "poor", "terrible", "awful", "horrible", "slow", "unreliable", "hate", "hated", "useless",
    "difficult", "buggy", "broken", "worst", "fail", "failed", "failure", "disappointing", "annoying",
    "inaccurate", "confusing", "crash", "crashes", "crashed", "weak", "unhappy", "problem", "problems",
    "frustrating", "expensive", "unstable", "mediocre", "ugly", "lacks", "laggy", "error-prone",
}
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "aren't", "don't", "doesn't", "didn't",
            "cannot", "can't", "won't", "hardly", "without"}


def _lexicon(text: str) -> Dict:
    toks = re.findall(r"[a-z']+(?:-[a-z]+)?", text.lower())
    pos = neg = 0
    for i, t in enumerate(toks):
        flip = any(x in NEGATORS for x in toks[max(0, i - 2):i])
        if t in POSITIVE:
            pos, neg = (pos, neg + 1) if flip else (pos + 1, neg)
        elif t in NEGATIVE:
            pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
    net = pos - neg
    if net > 0:
        label, conf = "Positive", min(0.95, 0.6 + 0.1 * net)
    elif net < 0:
        label, conf = "Negative", min(0.95, 0.6 + 0.1 * -net)
    else:
        label, conf = "Neutral", 0.6 if pos + neg == 0 else 0.5
    return {"label": label, "score": round(conf, 2), "method": "Lexicon (offline)",
            "explanation": f"Found {pos} positive and {neg} negative opinion words."}


def _load_pipe():
    global _PIPE, _PIPE_FAILED
    if _PIPE is not None:
        return _PIPE
    if _PIPE_FAILED or SENTIMENT_BACKEND == "lexicon" or \
            importlib.util.find_spec("transformers") is None:
        return None
    try:
        from transformers import pipeline

        _PIPE = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
        return _PIPE
    except Exception as exc:
        log.warning("HF sentiment model unavailable (%s); using lexicon.", exc)
        _PIPE_FAILED = True
        return None


def analyze_sentiment(text: str) -> Dict:
    """Return {label: Positive|Negative|Neutral, score, method, explanation}."""
    text = (text or "").strip()
    if not text:
        return {"label": "Neutral", "score": 0.0, "method": "n/a", "explanation": "No text provided."}
    pipe = _load_pipe()
    if pipe is not None:
        try:
            out = pipe(text[:2000], truncation=True)[0]
            label, score = out["label"].upper(), float(out["score"])
            if score < 0.75:  # binary model: low confidence == no clear opinion
                lab = "Neutral"
            else:
                lab = "Positive" if label.startswith("POS") else "Negative"
            return {"label": lab, "score": round(score, 2), "method": f"Hugging Face ({SENTIMENT_MODEL})",
                    "explanation": f"Model predicted {label} with {score:.0%} confidence"
                                   + (" (below 75% so reported as Neutral)." if lab == "Neutral" else ".")}
        except Exception as exc:
            log.warning("HF sentiment failed (%s); using lexicon.", exc)
    return _lexicon(text)


def analyze_documents_sentiment(docs: List[Document], max_sentences: int = 40) -> Dict:
    sents: List[str] = []
    for d in docs:
        sents.extend(s for s in split_sentences(d.text) if len(s.split()) >= 5)
    if not sents:
        return {"counts": {"Positive": 0, "Negative": 0, "Neutral": 0}, "overall": "Neutral", "sampled": 0}
    step = max(1, len(sents) // max_sentences)
    sample = sents[::step][:max_sentences]
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for s in sample:
        counts[analyze_sentiment(s)["label"]] += 1
    overall = max(counts, key=counts.get)
    return {"counts": counts, "overall": overall, "sampled": len(sample)}

"""Embedding backends: Hugging Face sentence-transformers (BERT-style) with a
TF-IDF fallback so the app keeps working offline / on low-memory hosts."""
from __future__ import annotations

import importlib.util
import logging
from typing import List, Optional

import numpy as np

from config.settings import EMBEDDING_BACKEND, EMBEDDING_MODEL

log = logging.getLogger(__name__)
_HF_MODEL = None
_HF_FAILED = False


class TfidfEmbedder:
    name = "TF-IDF"
    is_dense = False
    min_score = 0.05  # cosine similarity below this = "not relevant"

    def __init__(self) -> None:
        self.vectorizer = None

    def fit(self, texts: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        for stop in ("english", None):
            try:
                vec = TfidfVectorizer(
                    stop_words=stop, ngram_range=(1, 2), sublinear_tf=True, max_features=4096
                )
                vec.fit(texts)
                self.vectorizer = vec
                return
            except ValueError:  # empty vocabulary
                continue
        raise ValueError("Documents contain no usable words.")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fitted first.")
        return self.vectorizer.transform(texts).toarray().astype("float32")


class HFEmbedder:
    min_score = 0.25
    is_dense = True

    def __init__(self, model) -> None:
        self.model = model
        self.name = f"Hugging Face · {EMBEDDING_MODEL.split('/')[-1]}"

    def fit(self, texts: List[str]) -> None:  # pre-trained: nothing to fit
        return None

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return np.asarray(vecs, dtype="float32")


def _load_hf():
    global _HF_MODEL, _HF_FAILED
    if _HF_MODEL is not None:
        return _HF_MODEL
    if _HF_FAILED or importlib.util.find_spec("sentence_transformers") is None:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _HF_MODEL = SentenceTransformer(EMBEDDING_MODEL)
        return _HF_MODEL
    except Exception as exc:  # network error, out of memory, ...
        log.warning("Hugging Face embeddings unavailable (%s); using TF-IDF.", exc)
        _HF_FAILED = True
        return None


def get_embedder(backend: Optional[str] = None):
    backend = (backend or EMBEDDING_BACKEND).lower()
    if backend != "tfidf":
        model = _load_hf()
        if model is not None:
            return HFEmbedder(model)
    return TfidfEmbedder()

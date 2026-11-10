"""FAISS vector store with source metadata.

With a dense embedder (Hugging Face) the store is *hybrid*: the final score blends
semantic similarity (FAISS) with TF-IDF keyword similarity, which rescues short
factual questions ("Who reviewed the design?") that pure embeddings can miss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.chunking import Chunk
from core.embeddings import TfidfEmbedder, get_embedder

try:
    import faiss
except Exception:  # pragma: no cover - numpy fallback keeps app alive
    faiss = None

DENSE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
LEXICAL_GAIN = 3.0          # TF-IDF cosine values are small; scale before blending
HYBRID_MIN_SCORE = 0.17     # == a pure dense match at the 0.25 cutoff (0.7 * 0.25)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder=None, hybrid: bool = True) -> None:
        self.embedder = embedder or get_embedder()
        self.hybrid = hybrid
        self.chunks: List[Chunk] = []
        self.index = None
        self._matrix: Optional[np.ndarray] = None
        self._lex: Optional[TfidfEmbedder] = None
        self._lex_matrix: Optional[np.ndarray] = None

    @property
    def is_hybrid(self) -> bool:
        return self._lex is not None

    @property
    def min_score(self) -> float:
        return HYBRID_MIN_SCORE if self.is_hybrid else self.embedder.min_score

    def build(self, chunks: List[Chunk]) -> "VectorStore":
        if not chunks:
            raise ValueError("No text chunks to index.")
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        mat = np.ascontiguousarray(self.embedder.encode(texts), dtype="float32")
        if faiss is not None:
            self.index = faiss.IndexFlatIP(mat.shape[1])  # cosine (vectors are L2-normalised)
            self.index.add(mat)
        else:
            self._matrix = mat
        self._lex = self._lex_matrix = None
        if self.hybrid and getattr(self.embedder, "is_dense", False):
            try:
                lex = TfidfEmbedder()
                lex.fit(texts)
                self._lex_matrix = lex.encode(texts)
                self._lex = lex
            except ValueError:  # no usable words for keyword matching: stay dense-only
                self._lex = self._lex_matrix = None
        return self

    def _dense_scores(self, query: str) -> np.ndarray:
        q = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        n = len(self.chunks)
        if self.index is not None:
            dist, idx = self.index.search(q, n)
            scores = np.zeros(n, dtype="float32")
            valid = idx[0] >= 0
            scores[idx[0][valid]] = dist[0][valid]
            return scores
        return (self._matrix @ q.T).ravel()

    def _scores(self, query: str) -> np.ndarray:
        dense = self._dense_scores(query)
        if not self.is_hybrid:
            return dense
        lex = (self._lex_matrix @ self._lex.encode([query]).T).ravel()
        return DENSE_WEIGHT * dense + LEXICAL_WEIGHT * np.minimum(1.0, LEXICAL_GAIN * lex)

    def search(self, query: str, k: int = 4) -> List[Hit]:
        if not self.chunks or not query.strip():
            return []
        scores = self._scores(query)
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]

    def chunks_for_page(self, page: int, query: str, k: int = 4) -> List[Hit]:
        scores = self._scores(query) if query.strip() else np.zeros(len(self.chunks))
        hits = [Hit(c, float(scores[i])) for i, c in enumerate(self.chunks) if c.page == page]
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

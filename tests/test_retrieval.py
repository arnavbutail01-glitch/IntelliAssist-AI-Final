import pytest

from core.chunking import Chunk
from core.vector_store import VectorStore


def top_source(store, q):
    h = store.search(q, 1)[0]
    return h.chunk.source, h.chunk.page


@pytest.mark.parametrize("q,src,page", [
    ("What is retrieval augmented generation?", "RAG_Handbook.pdf", 1),
    ("How are vectors stored and found quickly?", "RAG_Handbook.pdf", 2),
    ("What do the authors recommend evaluating?", "RAG_Handbook.pdf", 3),
    ("Which applications use machine learning for fraud?", "AI_Notes.txt", None),
    ("TF-IDF term weighting", "NLP_Notes.docx", None),
    ("The upload screen crashed", "Pilot_Feedback.txt", None),
])
def test_correct_document_retrieved(store, q, src, page):
    assert top_source(store, q) == (src, page)


def test_scores_sorted_and_k_respected(store):
    hits = store.search("machine learning", 3)
    assert len(hits) == 3 and hits[0].score >= hits[1].score >= hits[2].score


def test_empty_query_and_page_lookup(store):
    assert store.search("   ") == []
    assert all(h.chunk.page == 2 for h in store.chunks_for_page(2, "anything"))
    assert store.chunks_for_page(99, "x") == []


def test_build_requires_chunks():
    with pytest.raises(ValueError):
        VectorStore().build([])


def test_single_chunk_and_stopword_only_corpus():
    s = VectorStore().build([Chunk("the and of to", "a.txt", None, 0)])
    assert len(s.search("the", 3)) == 1
    s2 = VectorStore().build([Chunk("hello world", "a.txt", None, 0)])
    assert s2.search("hello", 1)[0].score > 0


class FakeDense:
    """Weak 'semantic' embedder: hashed bag of letters, so keyword matching must rescue retrieval."""
    name, min_score, is_dense = "fake-dense", 0.25, True

    def fit(self, texts):
        pass

    def encode(self, texts):
        import numpy as np
        out = np.zeros((len(texts), 26), dtype="float32")
        for i, t in enumerate(texts):
            for ch in t.lower():
                if ch.isalpha() and ch.isascii():
                    out[i, ord(ch) - 97] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def test_hybrid_store_uses_keywords_and_threshold(loaded):
    from core.chunking import chunk_documents
    chunks = chunk_documents(loaded.docs)
    hybrid = VectorStore(FakeDense()).build(chunks)
    dense_only = VectorStore(FakeDense(), hybrid=False).build(chunks)
    assert hybrid.is_hybrid and not dense_only.is_hybrid
    assert hybrid.min_score == 0.17 and dense_only.min_score == 0.25
    h = hybrid.search("Who reviewed the design?", 1)[0]
    assert (h.chunk.source, h.chunk.page) == ("RAG_Handbook.pdf", 2)
    assert h.score >= hybrid.min_score
    assert len(hybrid.search("anything", 3)) == 3 and hybrid.chunks_for_page(2, "x")

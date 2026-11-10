"""Retrieval / answer evaluation on the sample documents.
Run: python scripts/evaluate.py   (add --hf to use the Hugging Face embedder)"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SAMPLE_DIR  # noqa: E402
from core.document_loader import load_documents  # noqa: E402
from core.embeddings import get_embedder  # noqa: E402
from core.rag_pipeline import NOT_FOUND, answer_question, build_knowledge_base  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402

# (question, expected source, expected page or None, expected keyword in answer)  -> None source = unanswerable
QA = [
    ("What is retrieval augmented generation?", "RAG_Handbook.pdf", 1, "retrieval"),
    ("Why does RAG reduce hallucination?", "RAG_Handbook.pdf", 1, "hallucination"),
    ("How are embedding vectors stored for fast search?", "RAG_Handbook.pdf", 2, "faiss"),
    ("What chunk sizes are typical?", "RAG_Handbook.pdf", 2, "500"),
    ("Who reviewed the design?", "RAG_Handbook.pdf", 2, "rao"),
    ("What do the authors conclude about RAG?", "RAG_Handbook.pdf", 3, "trustworthy"),
    ("Which metrics should be evaluated?", "RAG_Handbook.pdf", 3, "precision"),
    ("What is machine learning?", "AI_Notes.txt", None, "patterns"),
    ("What are the applications of machine learning?", "AI_Notes.txt", None, "fraud"),
    ("What are the three types of machine learning?", "AI_Notes.txt", None, "reinforcement"),
    ("What does the author conclude about AI ethics?", "AI_Notes.txt", None, "augment"),
    ("How do hospitals use machine learning?", "AI_Notes.txt", None, "scans"),
    ("What is TF-IDF?", "NLP_Notes.docx", None, "rare"),
    ("What are embeddings?", "NLP_Notes.docx", None, "vectors"),
    ("What does Precision at K mean?", "NLP_Notes.docx", None, "relevant"),
    ("What did users complain about in the pilot?", "Pilot_Feedback.txt", None, "crashed"),
    ("How much memory does the server use?", "Pilot_Feedback.txt", None, "2 gb"),
    ("Who won the football match yesterday?", None, None, None),
    ("What is the capital of France?", None, None, None),
    ("Explain quantum entanglement", None, None, None),
]


# Questions whose answer genuinely appears in more than one place: accept any of these (source, page).
ALT = {
    "What are embeddings?": [("RAG_Handbook.pdf", 2)],
    "Which metrics should be evaluated?": [("NLP_Notes.docx", None)],
}


def is_relevant(s, q, src, page):
    valid = [(src, page)] + ALT.get(q, [])
    return any(s["source"] == vs and (vp is None or s["page"] == vp) for vs, vp in valid)


def main(use_hf: bool) -> None:
    docs = load_documents(sorted(SAMPLE_DIR.glob("*"))).docs
    store = VectorStore(get_embedder("hf" if use_hf else "tfidf")).build(
        build_knowledge_base(docs).chunks)
    print(f"Embedder: {store.embedder.name} | chunks: {len(store.chunks)} | questions: {len(QA)}\n")
    p_at_k = hit1 = ans_ok = cite_ok = unsupported = 0
    answerable = [q for q in QA if q[1]]
    refusals = [q for q in QA if not q[1]]
    lat = []
    for q, src, page, kw in QA:
        t = time.perf_counter()
        r = answer_question(q, store, None, k=3)
        lat.append(time.perf_counter() - t)
        if not src:
            unsupported += r["mode"] != "not_found"
            continue
        srcs = r["sources"]
        rel = [s for s in srcs if is_relevant(s, q, src, page)]
        p_at_k += len(rel) / max(len(srcs), 1)
        hit1 += bool(srcs) and is_relevant(srcs[0], q, src, page)
        ans_ok += kw in r["answer"].lower()
        cite_ok += bool(rel)
    n = len(answerable)
    print(f"Precision@3 (mean)      : {p_at_k / n:.2f}")
    print(f"Top-1 retrieval accuracy: {hit1 / n:.2%}")
    print(f"Answer contains key fact: {ans_ok / n:.2%}  (extractive mode, no LLM)")
    print(f"Citation correctness    : {cite_ok / n:.2%}")
    print(f"Unsupported answer rate : {unsupported / len(refusals):.2%}  ({len(refusals)} unanswerable questions)")
    print(f"Mean latency            : {1000 * sum(lat) / len(lat):.1f} ms/question")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    main(ap.parse_args().hf)

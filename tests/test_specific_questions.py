"""Regression tests for answering *specific* questions (who / when / how many / why)."""
import pytest

from core.chunking import chunk_documents
from core.document_loader import Document
from core.rag_pipeline import NOT_FOUND, answer_question, extractive_answer, retrieve_context
from core.textmatch import build_units, stem, unseen_fraction, get_stats
from core.vector_store import VectorStore
from core.embeddings import TfidfEmbedder
import scripts.specific_eval as northwind
import scripts.heldout_eval as heldout


def _store(pages, name):
    docs = [Document(t, name, i + 1, "pdf") for i, t in enumerate(pages)]
    return VectorStore(TfidfEmbedder()).build(chunk_documents(docs))


@pytest.fixture(scope="module")
def nw():
    return _store(northwind.PAGES, "Northwind.pdf")


@pytest.fixture(scope="module")
def held():
    return _store(heldout.PAGES, "CS201.pdf")


def test_stemming_variants_match():
    assert stem("reviewed") == stem("reviews") == stem("review")
    assert stem("policies") == stem("policy") and stem("taught") == stem("teaching") == "teach"
    assert stem("reimbursement") == stem("reimbursed")


def test_abbreviation_is_expanded(nw):
    r = answer_question("Who is the CTO?", nw, None)
    assert r["mode"] == "extractive" and "Daniel Brooks" in r["answer"]


@pytest.mark.parametrize("q,fact,page", [
    ("How many days of annual leave do employees get?", "24", 2),
    ("How far in advance must travel be approved?", "7 days", 3),
    ("What did version 4.2 add?", "holiday", 4),
    ("Where are backups of customer data kept?", "hyderabad", 5),
])
def test_specific_facts_and_pages(nw, q, fact, page):
    r = answer_question(q, nw, None)
    assert fact in r["answer"].lower() and r["sources"][0]["page"] == page


def test_answer_is_focused_not_a_dump(nw):
    r = answer_question("Who founded Northwind Analytics?", nw, None)
    assert r["answer"].count("\n- ") <= 2 and "Meera Iyer" in r["answer"]


@pytest.mark.parametrize("q", ["What is the CEO's salary?", "What is the dress code policy?",
                               "How many employees does Northwind have?"])
def test_absent_facts_are_refused(nw, q):
    r = answer_question(q, nw, None)
    assert r["mode"] == "not_found" and r["answer"] == NOT_FOUND and r["sources"] == []


def test_numeric_question_needs_a_number_next_to_the_noun(nw):
    assert answer_question("How many employees does Northwind have?", nw, None)["mode"] == "not_found"
    assert "26 weeks" in answer_question("How long is parental leave?", nw, None)["answer"]


def test_paraphrased_questions_on_unseen_document(held):
    ok = 0
    qs = [q for q in heldout.QA if q[2]]
    for q, fact, page in qs:
        r = answer_question(q, held, None)
        ok += fact in r["answer"].lower()
    assert ok / len(qs) >= 0.9


def test_unanswerable_on_unseen_document(held):
    for q, _, _ in [x for x in heldout.QA if not x[2]]:
        assert answer_question(q, held, None)["mode"] == "not_found"


def test_overlap_does_not_duplicate_sentences(nw):
    txt = answer_question("What does Forecaster cost?", nw, None)["answer"]
    lines = [l for l in txt.splitlines() if l.startswith("- ")]
    assert len(lines) == len(set(lines))


def test_dense_mode_skips_lexical_refusal(nw):
    units = build_units("What is the dress code policy?")
    assert unseen_fraction(units, get_stats(nw)) >= 0.5
    hits = retrieve_context("What is the dress code policy?", nw)
    # strict=False is what a hybrid/dense store uses: the embedder's own cut-off decides
    assert extractive_answer("What is the dress code policy?", hits or nw.search("policy", 2),
                             stats=get_stats(nw), strict=False) != NOT_FOUND

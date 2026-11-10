import pytest

from tests.helpers import BrokenLLM, FakeLLM
from core.document_loader import Document
from core.intent_analyzer import analyze_intent
from core.sentiment_analyzer import analyze_documents_sentiment, analyze_sentiment
from core.summarizer import MODES, extractive_summary, split_sections, summarize


@pytest.mark.parametrize("text,label", [
    ("The AI system is excellent.", "Positive"),
    ("The application is slow and unreliable.", "Negative"),
    ("The model uses embeddings.", "Neutral"),
    ("This is not good at all", "Negative"),
    ("It is not bad", "Positive"),
])
def test_sentiment(text, label):
    r = analyze_sentiment(text)
    assert r["label"] == label and 0 <= r["score"] <= 1 and r["explanation"]


def test_sentiment_empty():
    assert analyze_sentiment("")["label"] == "Neutral"


def test_document_sentiment(loaded):
    res = analyze_documents_sentiment([d for d in loaded.docs if d.source == "Pilot_Feedback.txt"])
    assert res["sampled"] > 0 and sum(res["counts"].values()) == res["sampled"]
    assert analyze_documents_sentiment([])["sampled"] == 0


@pytest.mark.parametrize("q,intent", [
    ("Summarize this PDF", "Summarization"),
    ("Find the definition of NLP", "Information Retrieval"),
    ("Compare AI and ML", "Comparison"),
    ("Explain this concept", "Explanation"),
    ("List the key points", "Extraction"),
    ("What is the author's opinion?", "Opinion/Analysis"),
    ("Who is mentioned on page 4?", "Information Retrieval"),
    ("hello", "Greeting"),
    ("blah blah", "Information Retrieval"),
])
def test_intent(q, intent):
    r = analyze_intent(q)
    assert r["intent"] == intent and r["category"] and r["explanation"]


def test_intent_empty():
    assert analyze_intent("  ")["intent"] == "Unknown"


@pytest.mark.parametrize("mode", MODES)
def test_extractive_summary_every_mode(loaded, mode):
    docs = [d for d in loaded.docs if d.source == "AI_Notes.txt"]
    r = summarize(docs, mode, None)
    assert r["method"] == "extractive" and len(r["text"]) > 40 and r["note"]


def test_chapter_summary_uses_headings(loaded):
    docs = [d for d in loaded.docs if d.source == "AI_Notes.txt"]
    titles = [t for t, _ in split_sections(docs)]
    assert any("Chapter 2" in t for t in titles)
    assert "Chapter 3" in summarize(docs, "chapter", None)["text"]


def test_chapter_fallback_pages_and_parts():
    pages = [Document(f"Plain paragraph number {i}. " * 20, "x.pdf", i, "pdf") for i in range(1, 4)]
    assert [t for t, _ in split_sections(pages)][0].startswith("Page")
    one = [Document("Some plain text without any heading at all. " * 40, "y.txt", None, "txt")]
    assert split_sections(one)[0][0].startswith("Part")


def test_page_range_and_multi_doc(loaded):
    pdf = [d for d in loaded.docs if d.source == "RAG_Handbook.pdf"]
    r = summarize(pdf, "short", None, (3, 3))
    assert "conclude" in r["text"].lower() and "FAISS" not in r["text"]
    assert "No text found" in summarize(pdf, "short", None, (50, 60))["text"]
    multi = summarize(loaded.docs, "key_points", None)
    assert "### AI_Notes.txt" in multi["text"] and "### RAG_Handbook.pdf" in multi["text"]


def test_llm_summary_and_map_reduce(loaded):
    llm = FakeLLM("LLM SUMMARY")
    r = summarize([d for d in loaded.docs if d.source == "AI_Notes.txt"], "detailed", llm)
    assert r["method"] == "llm" and r["text"] == "LLM SUMMARY"
    big = [Document("Sentence about topic number one. " * 1200, "big.txt")]  # ~40k chars -> several parts
    llm2 = FakeLLM("part")
    r2 = summarize(big, "short", llm2)
    assert r2["method"] == "llm" and 2 < len(llm2.calls) <= 13


def test_llm_failure_in_summary_falls_back(loaded):
    r = summarize(loaded.docs, "short", BrokenLLM())
    assert r["method"] == "extractive" and "quota exceeded" in r["note"] and r["text"]


def test_invalid_mode_and_tiny_input():
    with pytest.raises(ValueError):
        summarize([], "bogus")
    r = summarize([Document("Hi.", "t.txt")], "short", None)
    assert r["text"]
    assert extractive_summary("one two three four five.", 5)

"""End-to-end UI tests using Streamlit's AppTest (runs the real app.py)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui import state as S

TIMEOUT = 90
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def new_app():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            return at.run()
    raise AssertionError(f"button {label!r} not found: {[b.label for b in at.button]}")


def go(at, page):
    at.sidebar.radio[0].set_value(page)
    return at.run()


def ask(at, q):
    at.chat_input[0].set_value(q)
    return at.run()


def text_of(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def app():
    at = new_app()
    assert not at.exception
    return at


@pytest.fixture
def loaded_app(app):
    at = click(app, "Samples")
    assert not at.exception
    return at


def test_launch_without_documents(app):
    assert app.title[0].value == "Ask your documents"
    assert any("Upload a PDF" in i.value for i in app.info)


def test_chat_before_upload_is_friendly(app):
    at = ask(app, "What is AI?")
    assert not at.exception
    assert "upload a document" in at.session_state["messages"][-1]["content"].lower()


def test_load_samples_and_ask(loaded_app):
    at = loaded_app
    assert len(at.session_state["docs"]) == 6 and at.session_state["store"] is not None
    assert any("Loaded" in s.value for s in at.success)
    at = ask(at, "Who reviewed the design?")
    assert not at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "Ananya Rao" in msgs[-1]["content"]
    assert msgs[-1]["sources"][0]["page"] == 2
    assert msgs[0]["meta"]["intent"] == "Information Retrieval"
    assert any("Sources" in e.label for e in at.expander)


def test_not_in_document(loaded_app):
    at = ask(loaded_app, "What is the capital of France?")
    assert "could not find" in at.session_state["messages"][-1]["content"]
    assert at.session_state["messages"][-1]["sources"] == []


def test_summarise_via_chat_and_comparison(loaded_app):
    at = ask(loaded_app, "Summarize the documents")
    assert at.session_state["messages"][0]["meta"]["intent"] == "Summarization"
    assert len(at.session_state["messages"][-1]["content"]) > 50
    at = ask(at, "Compare machine learning and NLP")
    assert not at.exception and len(at.session_state["messages"]) == 4


def test_greeting_and_long_question(loaded_app):
    at = ask(loaded_app, "hello")
    assert "Hello" in at.session_state["messages"][-1]["content"]
    at = ask(at, "what is RAG " * 400)
    assert not at.exception


def test_history_save_open_delete_clear(loaded_app):
    at = ask(loaded_app, "What is machine learning?")
    cid = at.session_state["conv_id"]
    assert cid
    at = click(at, "➕ New chat")
    assert at.session_state["messages"] == [] and at.session_state["conv_id"] is None
    at = go(at, S.PAGE_HISTORY)
    assert not at.exception
    open_btn = [b for b in at.button if b.key == f"open_{cid}"][0]
    open_btn.click()
    at = at.run()
    assert at.session_state["page"] == S.PAGE_CHAT
    assert len(at.session_state["messages"]) == 2 and at.session_state["conv_id"] == cid
    at = click(at, "🧹 Clear chat")
    assert at.session_state["messages"] == []
    at = go(at, S.PAGE_HISTORY)
    assert not any(b.key == f"open_{cid}" for b in at.button)


def test_delete_from_history_page(loaded_app):
    at = ask(loaded_app, "What is RAG?")
    cid = at.session_state["conv_id"]
    at = go(at, S.PAGE_HISTORY)
    [b for b in at.button if b.key == f"del_{cid}"][0].click()
    at = at.run()
    assert not at.exception and at.session_state["conv_id"] is None


@pytest.mark.parametrize("label", ["Short Summary", "Detailed Summary", "Key Points", "Chapter Summary"])
def test_summary_tab_all_types(loaded_app, label):
    at = go(loaded_app, S.PAGE_SUMMARY)
    [r for r in at.radio if r.label == "Summary type"][0].set_value(label)
    at = click(at, "Generate summary")
    assert not at.exception
    assert at.session_state["summary"] and len(at.session_state["summary"]["text"]) > 50


def test_summary_page_range_and_empty_selection(loaded_app):
    at = go(loaded_app, S.PAGE_SUMMARY)
    at.multiselect[0].set_value([])
    at = click(at, "Generate summary")
    assert any("Select at least one" in w.value for w in at.warning)
    at.multiselect[0].set_value(["RAG_Handbook.pdf"])
    at.number_input[0].set_value(3)
    at.number_input[1].set_value(3)
    at = click(at, "Generate summary")
    assert "conclude" in at.session_state["summary"]["text"].lower()


def test_summary_without_documents(app):
    at = go(app, S.PAGE_SUMMARY)
    assert any("Load a document" in i.value for i in at.info)


def test_nlp_page(loaded_app):
    at = go(loaded_app, S.PAGE_NLP)
    at.text_area[0].set_value("The application is slow and unreliable.")
    at = click(at, "Analyse")
    assert not at.exception
    assert any(m.value == "Negative" for m in at.metric)
    at.text_area[0].set_value("Summarize this PDF")
    at = click(at, "Analyse")
    assert any(m.value == "Summarization" for m in at.metric)
    at = click(at, "Analyse loaded documents")
    assert not at.exception
    at.text_area[0].set_value("   ")
    at = click(at, "Analyse")
    assert any("enter some text" in w.value for w in at.warning)


def test_nlp_doc_sentiment_without_docs(app):
    at = go(app, S.PAGE_NLP)
    at = click(at, "Analyse loaded documents")
    assert any("Load a document" in w.value for w in at.warning)


def test_documents_and_about_pages(loaded_app):
    at = go(loaded_app, S.PAGE_DOCS)
    assert not at.exception and len(at.dataframe) == 1
    at = go(at, S.PAGE_ABOUT)
    assert not at.exception and at.title[0].value.startswith("About")
    at = go(at, S.PAGE_HISTORY)
    assert not at.exception


def test_documents_page_empty(app):
    at = go(app, S.PAGE_DOCS)
    assert any("No documents" in i.value for i in at.info)


def test_remove_documents(loaded_app):
    at = click(loaded_app, "🗑️ Remove all documents")
    assert at.session_state["docs"] == [] and at.session_state["store"] is None
    at = ask(at, "What is RAG?")
    assert "upload a document" in at.session_state["messages"][-1]["content"].lower()


def test_process_without_file_warns(app):
    at = click(app, "Process")
    assert any("at least one file" in w.value for w in at.warning)


def test_repeated_sample_load_no_duplicates(loaded_app):
    at = click(loaded_app, "Samples")
    assert len(at.session_state["docs"]) == 6
    keys = [(d.source, d.page) for d in at.session_state["docs"]]
    assert len(keys) == len(set(keys))


def test_api_key_failure_falls_back_gracefully(loaded_app, monkeypatch):
    """A bad/expired key must never crash the UI: answer falls back to extractive."""
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "post", boom)
    at = loaded_app
    at.session_state["api_key"] = "bad-key"
    at = ask(at, "What is RAG?")
    assert not at.exception
    last = at.session_state["messages"][-1]
    assert last["meta"]["mode"] == "extractive" and last["meta"]["error"]
    assert any("could not be used" in w.value for w in at.warning)

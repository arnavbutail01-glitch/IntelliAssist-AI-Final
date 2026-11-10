"""Chat page: RAG question answering with source citations."""
from __future__ import annotations

import streamlit as st

from core.intent_analyzer import analyze_intent
from core.rag_pipeline import answer_question
from core.sentiment_analyzer import analyze_sentiment
from core.summarizer import summarize
from ui import state as S

MAX_QUESTION_CHARS = 2000


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if msg["role"] == "user" and meta.get("intent"):
            st.caption(f"Intent: {meta['intent']} · Sentiment: {meta.get('sentiment', '—')}")
        if msg["role"] == "assistant":
            if meta.get("error"):
                st.warning(f"The LLM could not be used ({meta['error']}). Showing an extractive answer instead.")
            elif meta.get("mode") == "extractive":
                st.caption("ℹ️ Offline extractive mode — add an API key in Settings for fluent, generated answers.")
            sources = msg.get("sources") or []
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for s in sources:
                        label = f"{s['source']} — Page {s['page']}" if s.get("page") else s["source"]
                        score = f" · relevance {s['score']}" if s.get("score") is not None else ""
                        st.markdown(f"**{label}**{score}")
                        if s.get("excerpt"):
                            st.caption(s["excerpt"] + "…")


def _respond(prompt: str, intent: dict) -> dict:
    store = st.session_state.store
    if store is None:
        return {"answer": "Please upload a document (or click **Samples** in the sidebar) first, "
                          "then ask your question.", "sources": [], "mode": "none", "error": None}
    llm = S.current_llm()
    if intent["intent"] == "Summarization":
        res = summarize(st.session_state.docs, "short", llm)
        names = S.doc_names()
        return {"answer": res["text"], "sources": [{"source": n, "page": None} for n in names],
                "mode": res["method"], "error": None}
    if intent["intent"] == "Greeting":
        return {"answer": "Hello! Ask me anything about your uploaded documents.", "sources": [],
                "mode": "none", "error": None}
    k = max(st.session_state.top_k, 6) if intent["intent"] == "Comparison" else st.session_state.top_k
    return answer_question(prompt, store, llm, k=k, history=st.session_state.messages[:-1])


def render_chat() -> None:
    st.title("Ask your documents")
    st.caption("Upload a document and ask questions about its content.")
    if not st.session_state.docs:
        st.info("👈 Upload a PDF, TXT or DOCX in the sidebar and press **Process** — "
                "or click **Samples** to try the bundled documents.")
    for msg in st.session_state.messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt or not prompt.strip():
        return
    prompt = prompt.strip()
    if len(prompt) > MAX_QUESTION_CHARS:
        st.warning(f"Question shortened to {MAX_QUESTION_CHARS} characters.")
        prompt = prompt[:MAX_QUESTION_CHARS]

    history = S.get_history()
    if st.session_state.conv_id is None:
        st.session_state.conv_id = history.create_conversation(S.doc_names())
    cid = st.session_state.conv_id

    intent, sentiment = analyze_intent(prompt), analyze_sentiment(prompt)
    user_msg = {"role": "user", "content": prompt, "sources": [],
                "meta": {"intent": intent["intent"], "sentiment": sentiment["label"]}}
    st.session_state.messages.append(user_msg)
    history.add_message(cid, "user", prompt, meta=user_msg["meta"])

    with st.spinner("Thinking…"):
        try:
            result = _respond(prompt, intent)
        except Exception as exc:  # last line of defence: never show a traceback
            result = {"answer": f"Sorry, something went wrong while answering ({type(exc).__name__}).",
                      "sources": [], "mode": "error", "error": None}
    bot_msg = {"role": "assistant", "content": result["answer"], "sources": result["sources"],
               "meta": {"mode": result["mode"], "error": result.get("error")}}
    st.session_state.messages.append(bot_msg)
    history.add_message(cid, "assistant", result["answer"], result["sources"], bot_msg["meta"])
    st.rerun()

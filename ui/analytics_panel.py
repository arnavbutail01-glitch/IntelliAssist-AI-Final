"""Summary, NLP analysis, history and about pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.intent_analyzer import analyze_intent
from core.sentiment_analyzer import analyze_documents_sentiment, analyze_sentiment
from core.summarizer import summarize
from ui import state as S

SUMMARY_TYPES = {"Short Summary": "short", "Detailed Summary": "detailed",
                 "Key Points": "key_points", "Chapter Summary": "chapter"}


def render_summary() -> None:
    st.title("Document Summary")
    docs = st.session_state.docs
    if not docs:
        st.info("Load a document first (sidebar), then generate a summary here.")
        return
    names = S.doc_names()
    selected = st.multiselect("Documents to summarise", names, default=names)
    label = st.radio("Summary type", list(SUMMARY_TYPES), horizontal=True)
    with st.expander("Summarise only a section (PDF page range)"):
        c1, c2 = st.columns(2)
        start = c1.number_input("From page (0 = start)", min_value=0, value=0, step=1)
        end = c2.number_input("To page (0 = end)", min_value=0, value=0, step=1)
    if st.button("Generate summary", type="primary"):
        chosen = [d for d in docs if d.source in selected]
        if not chosen:
            st.warning("Select at least one document.")
        else:
            with st.spinner("Summarising…"):
                try:
                    res = summarize(chosen, SUMMARY_TYPES[label], S.current_llm(),
                                    (int(start), int(end)) if (start or end) else None)
                    st.session_state.summary = {**res, "label": label}
                except Exception as exc:
                    st.session_state.summary = None
                    st.error(f"Could not generate a summary ({type(exc).__name__}: {exc}).")
    res = st.session_state.summary
    if res:
        st.subheader(res["label"])
        if res.get("note"):
            st.info(res["note"])
        st.markdown(res["text"])
        st.download_button("⬇️ Download summary", res["text"], file_name="summary.md", mime="text/markdown")


def render_nlp() -> None:
    st.title("NLP Analysis")
    st.caption("Sentiment analysis and intent detection.")
    text = st.text_area("Enter text or a question to analyse", height=120, key="nlp_text",
                        placeholder="e.g. The application is slow and unreliable.")
    if st.button("Analyse", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            sent, intent = analyze_sentiment(text), analyze_intent(text)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Sentiment")
                st.metric("Label", sent["label"])
                st.write(f"Confidence: **{sent['score']:.0%}**")
                st.caption(f"{sent['explanation']} · Method: {sent['method']}")
            with c2:
                st.subheader("Intent")
                st.metric("Detected intent", intent["intent"])
                st.write(f"Category: **{intent['category']}** · Confidence: **{intent['confidence']:.0%}**")
                st.caption(intent["explanation"])
    st.divider()
    st.subheader("Document sentiment")
    st.caption("Technical documents are usually Neutral; sentiment is most meaningful for reviews and feedback.")
    if st.button("Analyse loaded documents"):
        if not st.session_state.docs:
            st.warning("Load a document first.")
        else:
            with st.spinner("Analysing…"):
                res = analyze_documents_sentiment(st.session_state.docs)
            st.write(f"Overall: **{res['overall']}** (based on {res['sampled']} sentences)")
            st.bar_chart(pd.DataFrame({"Sentences": res["counts"]}))
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user" and m.get("meta")]
    if user_msgs:
        st.subheader("This conversation")
        st.dataframe(pd.DataFrame([{"Question": m["content"][:80], "Intent": m["meta"].get("intent"),
                                    "Sentiment": m["meta"].get("sentiment")} for m in user_msgs]),
                     use_container_width=True, hide_index=True)


def render_history() -> None:
    st.title("Conversation History")
    convs = S.get_history().list_conversations()
    if not convs:
        st.info("No saved conversations yet. Ask a question in the chat and it will appear here.")
        return
    for conv in convs:
        with st.container(border=True):
            st.markdown(f"**{conv['title']}**")
            docs = ", ".join(conv["document_names"]) or "no documents"
            st.caption(f"{conv['updated_at']} · {conv['message_count']} messages · {docs}")
            c1, c2 = st.columns(2)
            c1.button("Open", key=f"open_{conv['conversation_id']}", on_click=S.open_conversation,
                      args=(conv["conversation_id"],), use_container_width=True)
            c2.button("Delete", key=f"del_{conv['conversation_id']}", on_click=S.delete_conversation,
                      args=(conv["conversation_id"],), use_container_width=True)


def render_about() -> None:
    st.title("About IntelliAssist AI")
    st.markdown(
        """
**IntelliAssist AI** is a smart document assistant: upload PDF, TXT or DOCX files, ask questions,
get answers with source citations, generate summaries and analyse sentiment and intent.

**How it works (RAG):** extract text → clean → chunk (800 chars, 150 overlap) → embed →
store in **FAISS** → retrieve the most relevant chunks → generate a grounded answer with **Gemini / OpenAI**
→ show citations.

**Tech stack:** Python · Streamlit · LangChain text splitters · Hugging Face sentence-transformers (BERT) ·
TF-IDF · FAISS · SQLite.

**Limitations:** scanned/image-only PDFs need OCR (not included); the index lives in memory and is rebuilt
per session; without an API key the app answers in extractive mode.
"""
    )

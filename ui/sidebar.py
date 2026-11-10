"""Sidebar: branding, navigation, uploads, recent chats and settings."""
from __future__ import annotations

import streamlit as st

from config.settings import SAMPLE_DIR, SUPPORTED_EXTENSIONS
from core.document_loader import load_documents
from core.rag_pipeline import build_knowledge_base
from ui import state as S


def ingest(result) -> None:
    """Add loaded documents, rebuild the FAISS index and report the outcome."""
    for err in result.errors:
        st.error(err)
    if not result.docs:
        return
    new_names = {d.source for d in result.docs}
    docs = [d for d in st.session_state.docs if d.source not in new_names] + result.docs
    try:
        with st.spinner("Building the semantic index…"):
            store = build_knowledge_base(docs)
    except Exception as exc:
        st.error(f"Could not index the documents: {exc}")
        return
    st.session_state.docs = docs
    st.session_state.store = store
    st.session_state.summary = None
    st.success(f"✅ Loaded {', '.join(result.loaded)} — {len(store.chunks)} searchable chunks in total.")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🤖 IntelliAssist AI")
        st.caption("Smart Document AI Assistant")
        llm = S.current_llm()
        store = st.session_state.store
        st.markdown(("🟢 **AI Ready**" if store else "🟡 **Waiting for documents**"))
        st.caption(f"LLM: {llm.name if llm else 'offline extractive mode (no API key)'}")
        if store:
            st.caption(f"Embeddings: {store.embedder.name}")

        st.radio("WORKSPACE", S.PAGES, key="page")
        st.divider()

        st.subheader("Upload documents")
        files = st.file_uploader("PDF, TXT or DOCX", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                                 accept_multiple_files=True, key="uploader")
        c1, c2 = st.columns(2)
        if c1.button("Process", use_container_width=True, type="primary"):
            if not files:
                st.warning("Choose at least one file first.")
            else:
                with st.spinner("Reading documents…"):
                    result = load_documents(list(files))
                ingest(result)
        if c2.button("Samples", use_container_width=True, help="Load the bundled sample documents"):
            paths = [p for p in sorted(SAMPLE_DIR.glob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            if not paths:
                st.error("No sample documents found in data/sample_documents.")
            else:
                ingest(load_documents(paths))
        if st.session_state.docs:
            st.button("🗑️ Remove all documents", on_click=S.clear_documents, use_container_width=True)

        st.divider()
        c3, c4 = st.columns(2)
        c3.button("➕ New chat", on_click=S.new_chat, use_container_width=True)
        c4.button("🧹 Clear chat", on_click=S.clear_chat, use_container_width=True)
        if st.session_state.conv_id and st.session_state.messages:
            st.download_button("⬇️ Export chat", S.get_history().export_markdown(st.session_state.conv_id),
                               file_name="intelliassist_chat.md", mime="text/markdown",
                               use_container_width=True)

        recent = S.get_history().list_conversations(limit=5)
        if recent:
            st.subheader("Recent conversations")
            for conv in recent:
                st.button(f"💬 {conv['title'][:28]}", key=f"recent_{conv['conversation_id']}",
                          on_click=S.open_conversation, args=(conv["conversation_id"],),
                          use_container_width=True)

        with st.expander("⚙️ Settings"):
            st.slider("Passages retrieved per question", 2, 8, key="top_k")
            st.selectbox("LLM provider", ["Gemini", "OpenAI"], key="provider")
            st.text_input("API key (kept in this session only)", type="password", key="api_key",
                          help="Leave empty to use GOOGLE_API_KEY / OPENAI_API_KEY from .env or Streamlit secrets.")

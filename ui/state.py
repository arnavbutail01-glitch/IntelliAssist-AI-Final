"""Shared Streamlit session-state helpers."""
from __future__ import annotations

import streamlit as st

from config.settings import TOP_K
from core.chat_history import ChatHistory
from core.llm import get_llm, make_llm

PAGE_CHAT = "💬 AI Chat"
PAGE_DOCS = "📄 My Documents"
PAGE_SUMMARY = "📝 Summary"
PAGE_NLP = "📊 NLP Analysis"
PAGE_HISTORY = "🕘 History"
PAGE_ABOUT = "ℹ️ About"
PAGES = [PAGE_CHAT, PAGE_DOCS, PAGE_SUMMARY, PAGE_NLP, PAGE_HISTORY, PAGE_ABOUT]


def init_state() -> None:
    defaults = {
        "page": PAGE_CHAT, "docs": [], "store": None, "messages": [], "conv_id": None,
        "summary": None, "provider": "Gemini", "api_key": "", "top_k": TOP_K,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource
def get_history() -> ChatHistory:
    return ChatHistory()


def current_llm():
    """LLM from the key typed in the sidebar, else from .env / Streamlit secrets."""
    key = st.session_state.get("api_key", "")
    if key.strip():
        return make_llm(st.session_state.get("provider", "Gemini"), key)
    return get_llm()


def doc_names() -> list:
    seen = []
    for d in st.session_state.docs:
        if d.source not in seen:
            seen.append(d.source)
    return seen


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.conv_id = None


def clear_chat() -> None:
    """Delete the current conversation (messages + DB record) and start fresh."""
    cid = st.session_state.get("conv_id")
    if cid:
        get_history().delete_conversation(cid)
    new_chat()


def open_conversation(cid: str) -> None:
    st.session_state.messages = get_history().get_messages(cid)
    st.session_state.conv_id = cid
    st.session_state.page = PAGE_CHAT


def delete_conversation(cid: str) -> None:
    get_history().delete_conversation(cid)
    if st.session_state.get("conv_id") == cid:
        new_chat()


def clear_documents() -> None:
    st.session_state.docs = []
    st.session_state.store = None
    st.session_state.summary = None

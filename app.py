"""IntelliAssist AI - Smart Document AI Assistant (Streamlit entry point)."""
import streamlit as st

st.set_page_config(page_title="IntelliAssist AI", page_icon="🤖", layout="wide")

from ui import state as S  # noqa: E402
from ui.analytics_panel import render_about, render_history, render_nlp, render_summary  # noqa: E402
from ui.chat_interface import render_chat  # noqa: E402
from ui.document_panel import render_documents  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

S.init_state()
render_sidebar()

PAGE_RENDERERS = {
    S.PAGE_CHAT: render_chat, S.PAGE_DOCS: render_documents, S.PAGE_SUMMARY: render_summary,
    S.PAGE_NLP: render_nlp, S.PAGE_HISTORY: render_history, S.PAGE_ABOUT: render_about,
}
PAGE_RENDERERS.get(st.session_state.page, render_chat)()

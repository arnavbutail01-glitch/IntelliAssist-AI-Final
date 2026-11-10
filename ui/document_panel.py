"""My Documents page."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_documents() -> None:
    st.title("My Documents")
    docs = st.session_state.docs
    if not docs:
        st.info("No documents loaded yet. Upload files in the sidebar or click **Samples**.")
        return
    store = st.session_state.store
    rows = []
    for name in dict.fromkeys(d.source for d in docs):
        parts = [d for d in docs if d.source == name]
        rows.append({
            "Document": name, "Type": parts[0].file_type.upper(),
            "Pages with text": len(parts) if parts[0].page else "—",
            "Words": sum(len(d.text.split()) for d in parts),
            "Chunks": sum(1 for c in store.chunks if c.source == name) if store else 0,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Preview extracted text")
    for name in dict.fromkeys(d.source for d in docs):
        with st.expander(name):
            for d in (d for d in docs if d.source == name):
                if d.page:
                    st.markdown(f"**Page {d.page}**")
                st.text(d.text[:3000] + ("…" if len(d.text) > 3000 else ""))

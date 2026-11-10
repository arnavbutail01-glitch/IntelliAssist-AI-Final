"""Central configuration for IntelliAssist AI."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv is optional at runtime
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_documents"
HISTORY_DIR = BASE_DIR / "storage" / "chat_history"
HISTORY_DB = Path(os.getenv("INTELLIASSIST_DB", str(HISTORY_DIR / "history.db")))

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")
MAX_FILE_MB = 25

# Chunking (engineering starting values - tune using docs/evaluation.md)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

# Model backends. "auto" = use the Hugging Face model if it can be loaded,
# otherwise fall back to a lightweight offline method so the app never crashes.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()  # auto | hf | tfidf
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "auto").lower()  # auto | hf | lexicon
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from environment/.env first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets[name])
    except Exception:
        return default

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Deterministic, offline, isolated: set BEFORE the app modules are imported.
os.environ["EMBEDDING_BACKEND"] = "tfidf"
os.environ["SENTIMENT_BACKEND"] = "lexicon"
os.environ["INTELLIASSIST_DB"] = str(Path(tempfile.mkdtemp()) / "test_history.db")
for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(k, None)

import pytest  # noqa: E402

SAMPLES = ROOT / "data" / "sample_documents"


@pytest.fixture(scope="session")
def sample_paths():
    return sorted(p for p in SAMPLES.glob("*") if p.suffix in (".pdf", ".txt", ".docx"))


@pytest.fixture(scope="session")
def loaded(sample_paths):
    from core.document_loader import load_documents
    return load_documents(sample_paths)


@pytest.fixture(scope="session")
def store(loaded):
    from core.rag_pipeline import build_knowledge_base
    return build_knowledge_base(loaded.docs)

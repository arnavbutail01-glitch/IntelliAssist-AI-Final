# 🤖 IntelliAssist AI — Smart Document AI Assistant

Upload PDF / TXT / DOCX files, ask questions in natural language, and get **grounded answers with source citations**
(file name + page number). Also: summarisation, sentiment & intent analysis, conversation history, Streamlit UI.

## Problem & solution
Students and businesses struggle to search information across PDFs, notes and documents. IntelliAssist AI uses
**RAG (Retrieval-Augmented Generation)**: text is extracted, chunked, embedded and stored in **FAISS**; for every question the most
relevant chunks are retrieved and an LLM (Gemini/OpenAI) answers *only* from them.

## Features
| Feature | Implementation |
|---|---|
| Document upload (PDF/TXT/DOCX) | `core/document_loader.py` (pypdf, python-docx), friendly errors for empty/unsupported/scanned files |
| RAG chatbot | `core/rag_pipeline.py` — document-grounded prompt, refuses when the answer is not in the documents |
| Semantic search | Hugging Face `all-MiniLM-L6-v2` embeddings + FAISS (`core/embeddings.py`, `core/vector_store.py`) |
| Offline fallback | TF-IDF embeddings + extractive answers/summaries when no model/API key is available |
| Summarisation | short / detailed / key points / chapter-wise / page range; LLM map-reduce (`core/summarizer.py`) |
| Sentiment analysis | Hugging Face DistilBERT (+ lexicon fallback) |
| Intent analysis | Rule-based classifier (Summarization, Comparison, Extraction, Explanation, Opinion, Retrieval) |
| Source citations | File + page + relevance score + excerpt in an expandable panel |
| Conversation history | SQLite — save, reopen, delete, export as Markdown |
| UI | Streamlit: chat, documents, summary, NLP analysis, history, about |

## Architecture
```
Upload → extract text (page-aware) → clean → chunk (800/150) → embed → FAISS
Question → intent/sentiment → embed → top-k search → grounded prompt → LLM → answer + citations → SQLite history
```

## Quick start
```bash
python -m venv venv
venv\Scripts\activate            # Windows        (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env   → paste your GOOGLE_API_KEY
streamlit run app.py
```
Click **Samples** in the sidebar to try the bundled documents instantly.
**No API key?** The app still works (extractive answers & summaries). You can also paste a key in *Settings* in the sidebar.

Get a free Gemini key at <https://aistudio.google.com> → *Get API key*.

## Run the tests
```bash
pip install -r requirements-dev.txt
pytest -q                       # 107 tests: loaders, chunking, retrieval, RAG, LLM clients, summaries, NLP, SQLite, full UI flows
python scripts/specific_eval.py && python scripts/heldout_eval.py   # who/when/how-many questions
python scripts/evaluate.py      # retrieval / citation metrics (add --hf for Hugging Face embeddings)
```

## Deploy on Streamlit Community Cloud
1. Push to GitHub (`.env` is git-ignored).
2. <https://share.streamlit.io> → *New app* → choose the repo, branch `main`, main file `app.py`.
3. *Advanced settings → Secrets*:
   ```toml
   GOOGLE_API_KEY = "your_key"
   SENTIMENT_BACKEND = "lexicon"   # recommended on the free tier (≈1 GB RAM)
   ```
4. Deploy, then test the live URL separately from local. If the build runs out of memory, delete the last two lines of
   `requirements.txt` (sentence-transformers, transformers) — the app falls back to TF-IDF automatically.

## Project structure
```
app.py                 Streamlit entry point
config/settings.py     Settings & secrets
core/                  document_loader, preprocessing, chunking, embeddings, vector_store,
                       rag_pipeline, llm, summarizer, sentiment_analyzer, intent_analyzer, chat_history
ui/                    state, sidebar, chat_interface, document_panel, analytics_panel
data/sample_documents/ Sample PDF, DOCX, TXT files (regenerate: python scripts/make_samples.py)
tests/                 pytest suite   |   docs/evaluation.md  evaluation & bug log
```

## Limitations
- Scanned/image-only PDFs have no text layer (OCR not included) — the app reports this clearly.
- The FAISS index lives in memory and is rebuilt per session (re-upload after restart); chat history persists in SQLite,
  but free cloud hosts may reset local files on redeploy.
- Answers from the offline fallback are verbatim excerpts, not generated prose. Fluent answers need an LLM key.
- Intent detection is rule-based; sentiment on technical documents is mostly Neutral by nature.

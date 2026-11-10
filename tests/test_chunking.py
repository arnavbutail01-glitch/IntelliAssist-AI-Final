from core.chunking import _fallback_split, chunk_documents, split_text
from core.document_loader import Document
from core.preprocessing import clean_text, split_sentences


def test_clean_text_preserves_numbers_and_dates():
    t = clean_text("Revenue  grew\r\n\r\n\r\n\r\n12.5%  on 2024-03-01 (Dr. Rao).  ")
    assert "12.5%" in t and "2024-03-01" in t and "\n\n\n" not in t and "  " not in t
    assert clean_text("   \n ") == ""


def test_hyphenated_linebreak_rejoined():
    assert "international" in clean_text("inter-\nnational")


def test_chunk_sizes_and_metadata(loaded):
    from core.chunking import chunk_documents
    chunks = chunk_documents(loaded.docs)
    assert chunks and all(0 < len(c.text) <= 900 for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    pdf = [c for c in chunks if c.source == "RAG_Handbook.pdf"]
    assert {c.page for c in pdf} == {1, 2, 3}
    assert pdf[0].citation().endswith("Page 1")


def test_long_text_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    pieces = split_text(text, 200, 40)
    assert len(pieces) > 3 and all(len(p) <= 200 for p in pieces)
    assert set(pieces[0].split()[-3:]) & set(pieces[1].split()[:12])  # overlap exists


def test_fallback_splitter_terminates():
    out = _fallback_split("a" * 5000, 300, 100)
    assert out and all(len(p) <= 300 for p in out)
    assert _fallback_split("short", 300, 100) == ["short"]


def test_empty_docs():
    assert chunk_documents([]) == []
    assert chunk_documents([Document("", "x.txt")]) == []


def test_sentence_split_handles_abbrev_and_headings():
    s = split_sentences("Overview\n\nDr. Rao reviewed it. It was fine.")
    assert s == ["Dr. Rao reviewed it.", "It was fine."]

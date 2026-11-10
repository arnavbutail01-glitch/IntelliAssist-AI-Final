import io

import docx
import pytest
from reportlab.pdfgen import canvas

from core.document_loader import load_documents


class Upload:  # mimics Streamlit's UploadedFile
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


def test_loads_all_three_types(loaded):
    assert not loaded.errors
    assert set(loaded.loaded) == {"AI_Notes.txt", "NLP_Notes.docx", "Pilot_Feedback.txt", "RAG_Handbook.pdf"}


def test_pdf_pages_preserved(loaded):
    pages = sorted(d.page for d in loaded.docs if d.source == "RAG_Handbook.pdf")
    assert pages == [1, 2, 3]


def test_docx_tables_extracted(loaded):
    text = " ".join(d.text for d in loaded.docs if d.source == "NLP_Notes.docx")
    assert "Precision@K" in text


def test_txt_non_utf8_fallback():
    r = load_documents([Upload("a.txt", "caf\xe9 au lait".encode("latin-1"))])
    assert r.docs and "caf" in r.docs[0].text


def test_unsupported_empty_and_whitespace():
    r = load_documents([Upload("virus.exe", b"MZ"), Upload("e.txt", b""), Upload("w.txt", b"   \n\n ")])
    assert not r.docs and len(r.errors) == 3
    assert "unsupported" in r.errors[0]
    assert "empty" in r.errors[1]
    assert "no readable text" in r.errors[2]


def test_blank_pdf_reports_scanned_hint():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    r = load_documents([Upload("scan.pdf", buf.getvalue())])
    assert not r.docs and "scanned" in r.errors[0]


def test_corrupt_files_do_not_crash():
    r = load_documents([Upload("bad.pdf", b"not a pdf"), Upload("bad.docx", b"not a docx")])
    assert len(r.errors) == 2 and not r.docs


def test_mixed_good_and_bad():
    d = docx.Document()
    d.add_paragraph("Hello docx world")
    b = io.BytesIO()
    d.save(b)
    r = load_documents([Upload("ok.docx", b.getvalue()), Upload("x.png", b"1")])
    assert r.loaded == ["ok.docx"] and len(r.errors) == 1


def test_none_and_empty_list():
    assert load_documents([]).docs == [] and load_documents(None).docs == []


def test_txt_utf16_and_utf8_bom():
    r = load_documents([Upload("u16.txt", "héllo wörld".encode("utf-16")),
                        Upload("u8.txt", "\ufeffhello utf8".encode("utf-8"))])
    assert "wörld" in r.docs[0].text and r.docs[1].text == "hello utf8"

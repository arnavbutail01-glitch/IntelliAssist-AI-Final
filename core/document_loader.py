"""Load PDF, TXT and DOCX files into Document objects (with page numbers)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from config.settings import MAX_FILE_MB, SUPPORTED_EXTENSIONS
from core.preprocessing import clean_text


@dataclass
class Document:
    text: str
    source: str
    page: Optional[int] = None  # 1-based page for PDFs, None for TXT/DOCX
    file_type: str = ""


@dataclass
class LoadResult:
    docs: List[Document] = field(default_factory=list)
    loaded: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _read_bytes(file: Any) -> tuple[str, bytes]:
    """Accept a path, a Streamlit UploadedFile or any file-like object."""
    if isinstance(file, (str, Path)):
        p = Path(file)
        return p.name, p.read_bytes()
    name = getattr(file, "name", "uploaded_file")
    if hasattr(file, "getvalue"):
        return name, file.getvalue()
    if hasattr(file, "seek"):
        file.seek(0)
    return name, file.read()


def load_pdf(data: bytes, name: str) -> List[Document]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception as exc:
            raise ValueError("the PDF is encrypted and could not be opened") from exc
        if not ok:
            raise ValueError("the PDF is password-protected")
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if text:
            docs.append(Document(text, name, i, "pdf"))
    return docs


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # never fails


def load_txt(data: bytes, name: str) -> List[Document]:
    text = clean_text(_decode_text(data))
    return [Document(text, name, None, "txt")] if text else []


def load_docx(data: bytes, name: str) -> List[Document]:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = clean_text("\n".join(parts))
    return [Document(text, name, None, "docx")] if text else []


_LOADERS = {".pdf": load_pdf, ".txt": load_txt, ".docx": load_docx}


def load_documents(files: List[Any]) -> LoadResult:
    """Load many files. Never raises: problems are returned in ``errors``."""
    result = LoadResult()
    for file in files or []:
        try:
            name, data = _read_bytes(file)
        except Exception as exc:
            result.errors.append(f"Could not read a file: {exc}")
            continue
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            result.errors.append(
                f"'{name}': unsupported file type '{ext or 'none'}'. "
                f"Please upload {', '.join(SUPPORTED_EXTENSIONS)}."
            )
            continue
        if not data:
            result.errors.append(f"'{name}': the file is empty.")
            continue
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            result.errors.append(f"'{name}': file is larger than {MAX_FILE_MB} MB.")
            continue
        try:
            docs = _LOADERS[ext](data, name)
        except Exception as exc:
            result.errors.append(f"'{name}': could not be read ({exc}).")
            continue
        if not docs:
            hint = " It may be a scanned/image-only PDF (OCR is not supported)." if ext == ".pdf" else ""
            result.errors.append(f"'{name}': no readable text was found.{hint}")
            continue
        result.docs.extend(docs)
        result.loaded.append(name)
    return result

"""Split documents into overlapping chunks that keep source/page metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from core.document_loader import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback when langchain is unavailable
    RecursiveCharacterTextSplitter = None


@dataclass
class Chunk:
    text: str
    source: str
    page: Optional[int]
    chunk_id: int

    def citation(self) -> str:
        return f"{self.source} — Page {self.page}" if self.page else self.source


def _fallback_split(text: str, size: int, overlap: int) -> List[str]:
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # prefer to break on whitespace
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut != -1 else end
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    overlap = min(overlap, size // 2)
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(text)
    else:
        pieces = _fallback_split(text, size, overlap)
    return [p.strip() for p in pieces if p and p.strip()]


def chunk_documents(docs: List[Document], size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        for piece in split_text(doc.text, size, overlap):
            chunks.append(Chunk(piece, doc.source, doc.page, len(chunks)))
    return chunks

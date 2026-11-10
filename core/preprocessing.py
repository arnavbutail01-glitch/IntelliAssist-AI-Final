"""Light-touch text cleaning. Numbers, dates, names and terms are preserved."""
from __future__ import annotations

import re
from typing import List

_ABBREV = {"e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "fig", "no", "approx"}


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\x0c", "\n").replace("\u00a0", " ")
    # re-join words hyphenated across line breaks: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading_line(line: str) -> bool:
    line = line.strip()
    return (0 < len(line) <= 60 and len(line.split()) < 10 and (line[0].isupper() or line[0].isdigit())
            and not line.endswith((".", "!", "?", ",", ";", ":")))


def split_sentences(text: str, drop_headings: bool = True) -> List[str]:
    """Simple regex sentence splitter. Short heading-like lines are separated from
    body text (and dropped by default) so they do not get glued to sentences."""
    paragraphs: List[str] = []
    buf: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            if buf:
                paragraphs.append(" ".join(buf)); buf = []
        elif _is_heading_line(line) and not buf:
            if not drop_headings:
                paragraphs.append(line.strip())
        else:
            buf.append(line.strip())
    if buf:
        paragraphs.append(" ".join(buf))
    sentences: List[str] = []
    for para in paragraphs:
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", para):
            part = part.strip()
            if not part:
                continue
            if sentences:
                last = sentences[-1].lower().rstrip(".").split()[-1:]
                if last and last[0] in _ABBREV and sentences[-1].endswith("."):
                    sentences[-1] += " " + part
                    continue
            sentences.append(part)
    return sentences

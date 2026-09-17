"""TODO: implement two splitters and a worst-chunk inspector.

- fixed_size(text, size, overlap): word windows WITH overlap.
- heading_aware(text, max_words): split on markdown headings, then size-cap.
- worst_chunk(chunks): return the chunk that makes least sense alone
  (proxy: the shortest non-empty one).

Reference: ../after/src/chunk.py.
"""

from __future__ import annotations

import re


def fixed_size(text: str, size: int = 512, overlap: int = 75) -> list[str]:
    words = text.split()
    step = size - overlap
    out: list[str] = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += step
    return out


def heading_aware(text: str, max_words: int = 512) -> list[str]:
    parts = re.split(r"(?m)^(#{1,6} .*)$", text)
    sections: list[str] = []
    buffer = ""
    for part in parts:
        if re.match(r"^#{1,6} .*$", part):
            if buffer.strip():
                sections.append(buffer)
            buffer = part + "\n"
        else:
            buffer += part
    if buffer.strip():
        sections.append(buffer.strip())

    capped: list[str] = []
    for section in sections:
        if len(section.split()) > max_words:
            capped.append(section)
        else:
            capped.extend(fixed_size(section, size=max_words, overlap=50))
    return capped


def worst_chunk(chunks: list[str]) -> str:
    return min(
        (c for c in chunks if c.strip()), key=lambda c: len(c.split()), default=""
    )


if __name__ == "__main__":
    doc = "# Intro\nHello world. " * 50 + "\n# Details\nMore text here. " * 50
    print("fixed:", len(fixed_size(doc)), "chunks")
    print("heading:", len(heading_aware(doc)), "chunks")

"""Contextual retrieval with real libraries.

- split(text, size, overlap): use langchain_text_splitters.RecursiveCharacterTextSplitter
  (don't hand-roll chunking — this is the splitter real projects use).
- local_summarizer(model): return a callable backed by Ollama's OpenAI-compatible
  endpoint (base_url="http://localhost:11434/v1", api_key="ollama", temperature=0).
- contextualize(doc, chunk, summarize): PREPEND the one-line blurb to the chunk.
- contextualize_all(...): map it over all chunks (the cheap overnight batch job).
"""

from __future__ import annotations

import os
from collections.abc import Callable

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

Summarizer = Callable[[str], str]

load_dotenv()

PROMPT = (
    "In one short sentence, situate this chunk within the document so it can be "
    "found by search. Reply with the sentence only.\n"
    "<document>\n{doc}\n</document>\n<chunk>\n{chunk}\n</chunk>"
)


def split(text: str, chunk_size: int = 512, chunk_overlap: int = 75) -> list[str]:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        ).split_text(text)


def local_summarizer(model: str = "qwen2.5:7b-instruct") -> Summarizer:
    client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("OLLAMA_API_KEY"))

    def summarize(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    return summarize
    

def contextualize(doc: str, chunk: str, summarize: Summarizer) -> str:
    """Return the chunk with the one-line blurb prepended."""
    blurb = summarize(PROMPT.format(doc=doc, chunk=chunk))
    return f"{blurb}\n{chunk}"


def contextualize_all(doc: str, chunks: list[str], summarize: Summarizer) -> list[str]:
    return [contextualize(doc, chunk, summarize) for chunk in chunks]
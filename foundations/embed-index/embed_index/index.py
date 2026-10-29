"""TODO: build a mini vector store.

- normalize(m): L2-normalize rows — and leave a zero row alone rather than
  dividing by zero.
- embed(texts): local embeddings via Ollama (nomic-embed-text), normalized.
- VectorIndex.build(chunks, embedder): embed + hold vectors beside their chunks.
- VectorIndex.search(query, k): cosine top-k (normalized -> a single dot product).

Note the `embedder` parameter: `search` has to embed the query, and taking the
embedder as an argument is what makes that testable without a live model. The same
seam is how you swap nomic-embed-text for BGE-M3, or numpy for Qdrant in Phase 2.

Then try a query phrased with DIFFERENT words than the text and watch it hit;
try an exact ID and watch it MISS (remember that for Phase 2 — hybrid search).

Reference: ../after/src/index.py.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

Embedder = Callable[[list[str]], np.ndarray]

load_dotenv()


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


def embed(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    vecs = np.array([d.embedding for d in response.data])
    return normalize(vecs)


class VectorIndex:
    def __init__(
        self,
        chunks: list[str],
        matrix: np.ndarray,
        embedder: Embedder = embed,
    ) -> None:
        self.chunks = chunks
        self.matrix = matrix
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[str], embedder: Embedder = embed) -> VectorIndex:
        return cls(chunks, embedder(chunks), embedder)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Embed the query, dot it against the matrix, return top-k.

        Because everything is normalized, the dot product *is* cosine similarity —
        no division, no loop over rows.
        """
        q = self.embedder([query])[0]
        scores = self.matrix @ q
        top_k = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    idx = VectorIndex.build(
        ["Hello, world!", "This is a test.", "This is another test."]
    )
    for chunk, score in idx.search("Hello"):
        print(chunk, score)

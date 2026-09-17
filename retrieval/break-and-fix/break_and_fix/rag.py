"""A RAG pipeline with ONE planted bug — the single most common one in the wild.

SYMPTOM: retrieval quality is garbage. Not "slightly worse" — nonsense results,
even for questions whose answer is verbatim in the corpus. Your eval's
context_recall cratered.

Debug it with the back-to-front playbook:
  1. Is the answer ignoring its context?      -> generation/prompt
  2. Is the right doc retrieved at all?       -> retrieval
  3. Is the doc even in the index?            -> ingestion
  4. Retrieved but ranked low / buried?       -> ranking / rerank

(Embedders are injected so this runs offline and deterministically. The bug is
exactly the one you'd hit with real models.)
"""
from __future__ import annotations

from collections.abc import Callable

from qdrant_client import QdrantClient, models

Embedder = Callable[[str], list[float]]


def embedder_a(text: str) -> list[float]:
    """'Model A' — e.g. bge-small-en-v1.5. Direction encodes the topic."""
    t = text.lower()
    payments = float(any(w in t for w in ("invoice", "paid", "payment", "wire", "refund")))
    weather = float(any(w in t for w in ("weather", "sunny", "rain", "cloudy")))
    search = float(any(w in t for w in ("search", "retrieval", "keyword", "vector")))
    return [payments, weather, search]


def embedder_b(text: str) -> list[float]:
    """'Model B' — a DIFFERENT model. Same dimension, but the axes mean other things.

    Two models producing the same vector SIZE does not make their vectors
    comparable. This is the trap.
    """
    t = text.lower()
    search = float(any(w in t for w in ("search", "retrieval", "keyword", "vector")))
    payments = float(any(w in t for w in ("invoice", "paid", "payment", "wire", "refund")))
    weather = float(any(w in t for w in ("weather", "sunny", "rain", "cloudy")))
    return [search, weather, payments]  # axis order differs from model A!


class RAG:
    def __init__(self, docs: list[str], embed: Embedder) -> None:
        self.embed: Embedder = embed or embedder_a
        self.client = QdrantClient(":memory:")
        self.name = "docs"
        self.client.create_collection(
            self.name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        self.client.upsert(self.name, points=[
            models.PointStruct(id=i, vector=self.embed(d), payload={"text": d})
            for i, d in enumerate(docs)
        ])

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        # Fix: use the same embedder for the query as the one used for the index.
        qv = self.embed(query)
        res = self.client.query_points(self.name, query=qv, limit=k, with_payload=True)
        return [(p.payload or {}).get("text", "") for p in res.points]

    def answer(self, query: str) -> str:
        ctx = self.retrieve(query)
        if not ctx:
            return "I don't know"
        return f"Based on the docs: {ctx[0]}"

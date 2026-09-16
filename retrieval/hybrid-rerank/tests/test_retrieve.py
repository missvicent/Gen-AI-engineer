"""Unit tests — real Qdrant API, deterministic fixture vectors, no downloads.

We hand-build tiny vectors so the test is fast and offline, but every call below
goes through the SAME qdrant-client API you'd use in production, including
server-side RRF fusion. The real-model path is covered in test_integration.py.
"""
from qdrant_client import models

from src.retrieve import HybridStore, bm25_search

DOCS = [
    "hybrid search fuses keyword and vector retrieval for better recall",
    "invoice INV-88231 was paid on July third twenty twenty six",
    "the weather is nice and sunny today outside",
]

# Deterministic stand-in vectors (2-D so they're readable):
#   doc0 = "retrieval concepts" direction, doc1 = "payments" direction, doc2 = "weather"
DENSE = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
# Sparse: token-id 7 == "inv-88231" only present on doc1
SPARSE = [
    models.SparseVector(indices=[1], values=[1.0]),
    models.SparseVector(indices=[7], values=[1.0]),
    models.SparseVector(indices=[2], values=[1.0]),
]


def _store() -> HybridStore:
    s = HybridStore(dense_size=2)
    s.upsert(DOCS, DENSE, SPARSE)
    return s


def test_dense_only_misses_the_exact_id():
    """The whole reason hybrid exists: a semantic query vector can't match an ID."""
    store = _store()
    # a query that is semantically about retrieval, not payments
    hits = store.dense_only([1.0, 0.0], k=1)
    assert "INV-88231" not in hits[0].text


def test_hybrid_finds_the_exact_id_via_the_sparse_arm():
    """Same store, same dense query, but the sparse arm carries the identifier."""
    store = _store()
    hits = store.hybrid_search(
        dense_q=[1.0, 0.0],                                     # semantic: retrieval
        sparse_q=models.SparseVector(indices=[7], values=[1.0]),  # keyword: INV-88231
        k=2,
    )
    assert any("INV-88231" in h.text for h in hits)


def test_hybrid_is_one_call_with_server_side_fusion():
    """No fusion code of our own — Qdrant returns an already-fused ranking."""
    store = _store()
    hits = store.hybrid_search([0.0, 1.0], models.SparseVector(indices=[7], values=[1.0]), k=3)
    assert len(hits) == 3
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_bm25_library_finds_the_identifier():
    """rank_bm25 in one call — we never wrote tf/idf."""
    assert "INV-88231" in bm25_search(DOCS, "INV-88231", k=1)[0]

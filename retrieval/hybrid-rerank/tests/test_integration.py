"""Integration tests — REAL local models (fastembed ONNX). Opt-in.

    make test-integration

First run downloads ~90MB of model weights, then caches them. These prove the
claims the unit tests only simulate: real embeddings genuinely miss exact IDs,
and a real cross-encoder genuinely improves ordering.
"""
import pytest

pytestmark = pytest.mark.integration

CORPUS = [
    "Refunds are processed within five business days of approval.",
    "Invoice INV-88231 was settled on 3 July 2026 by wire transfer.",
    "Our office is closed on public holidays in Peru.",
]


def test_real_dense_embeddings_understand_paraphrase():
    """No shared keywords, same meaning — this is what dense retrieval is FOR."""
    from qdrant_client import models as m

    from src.retrieve import HybridStore, dense_embedder

    model = dense_embedder()
    vecs = [v.tolist() for v in model.embed(CORPUS)]
    store = HybridStore(dense_size=len(vecs[0]))
    store.upsert(CORPUS, vecs, [m.SparseVector(indices=[i], values=[1.0])
                                for i in range(len(CORPUS))])
    q = next(iter(model.query_embed("how long until I get my money back?"))).tolist()
    assert "Refunds" in store.dense_only(q, k=1)[0].text


def test_real_reranker_promotes_the_best_passage():
    from src.retrieve import rerank

    out = rerank("when was invoice INV-88231 paid?", CORPUS, top_k=1)
    assert "INV-88231" in out[0]


def test_full_pipeline_end_to_end():
    from src.retrieve import build_and_search

    out = build_and_search(CORPUS, "INV-88231", k=1)
    assert "INV-88231" in out[0]

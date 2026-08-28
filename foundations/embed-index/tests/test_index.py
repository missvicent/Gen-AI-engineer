"""Offline tests: a fake embedder, so `search` itself is under test.

The distinction matters. A test that re-derives the ranking with its own numpy
call proves numpy works — it never touches your code. Injecting the embedder is
what lets these assertions be about `VectorIndex.search`.

The live semantic-search demo is in `__main__` and needs Ollama.
"""
from __future__ import annotations

import numpy as np

from src.index import VectorIndex, normalize

# A two-dimensional toy space: axis 0 is "payments", axis 1 is "weather".
VECTORS: dict[str, list[float]] = {
    "my card got declined": [1.0, 0.0],
    "payment failure": [0.9, 0.1],
    "great weather today": [0.0, 1.0],
    "the transaction did not go through": [0.95, 0.05],  # a query, different words
    "is it sunny": [0.05, 0.95],
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    return normalize(np.array([VECTORS[t] for t in texts], dtype=np.float32))


def index() -> VectorIndex:
    return VectorIndex.build(list(VECTORS)[:3], embedder=fake_embedder)


def test_search_ranks_nearest_first():
    hits = index().search("the transaction did not go through", k=3)
    assert [chunk for chunk, _ in hits] == [
        "my card got declined",
        "payment failure",
        "great weather today",
    ]


def test_different_words_same_meaning_still_lands_on_the_right_chunk():
    """The embeddings 'aha': zero shared words with either payment chunk."""
    top_chunk, top_score = index().search("the transaction did not go through", k=1)[0]
    assert top_chunk == "my card got declined"
    assert top_score > 0.9


def test_search_respects_k():
    assert len(index().search("is it sunny", k=2)) == 2


def test_scores_are_cosines_so_they_stay_within_one():
    for _, score in index().search("is it sunny", k=3):
        assert -1.0001 <= score <= 1.0001


def test_build_keeps_chunks_and_vectors_aligned():
    idx = index()
    assert len(idx.chunks) == idx.matrix.shape[0]
    # Row i must be the vector for chunk i, or every result is silently mislabeled.
    assert np.allclose(idx.matrix[2], fake_embedder(["great weather today"])[0])


def test_normalize_leaves_a_zero_row_alone_instead_of_dividing_by_zero():
    out = normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.isclose(float(np.linalg.norm(out[1])), 1.0)

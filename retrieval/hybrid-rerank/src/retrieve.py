"""
Nobody implements BM25 or cosine similarity at work — they pick the right package.
Your job is the wiring.]

    keyword (sparse)  ->  rank_bm25.BM25Okapi   /  fastembed SparseTextEmbedding
    semantic (dense)  ->  fastembed TextEmbedding("BAAI/bge-small-en-v1.5")
    store + fusion    ->  qdrant_client: Prefetch x2 + FusionQuery(Fusion.RRF)
    rerank            ->  fastembed TextCrossEncoder("BAAI/bge-reranker-base")
"""

from __future__ import annotations

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

DENSE = "dense"
SPARSE = "keywords"


@dataclass
class Hit:
    text: str
    score: float


def dense_embedder(model: str = "BAAI/bge-small-en-v1.5"):
    return TextEmbedding(model_name=model)


def sparse_embedder(model: str = "Qdrant/bm25"):
    return SparseTextEmbedding(model_name=model)


class HybridStore:
    """A Qdrant collection with BOTH a dense and a sparse vector per document."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        name: str = "docs",
        dense_size: int = 384,
    ) -> None:
        self.client = client or QdrantClient(":memory:")
        self.name = name

        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=dense_size, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={SPARSE: models.SparseVectorParams()},
            )

    def upsert(
        self,
        texts: list[str],
        dense_vecs: list[list[float]],
        sparse_vecs: list[models.SparseVector],
    ) -> None:
        self.client.upsert(
            collection_name=self.name,
            points=[
                models.PointStruct(
                    id=i,
                    vector={DENSE: d, SPARSE: s},
                    payload={"text": t},
                )
                for i, (t, d, s) in enumerate(zip(texts, dense_vecs, sparse_vecs))
            ],
        )

    def hybrid_search(
        self,
        dense_q: list[float],
        sparse_q: models.SparseVector,
        k: int = 5,
        candidates: int = 20,
    ) -> list[Hit]:
        res = self.client.query_points(
            collection_name=self.name,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE, limit=candidates),
                models.Prefetch(query=sparse_q, using=SPARSE, limit=candidates),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [Hit(text=p.payload["text"], score=p.score) for p in res.points]


def bm25_search(corpus: list[str], query: str, k: int = 5) -> list[str]:
    bm25 = BM25Okapi([d.lower().split() for d in corpus])
    return bm25.get_top_n(query.lower().split(), corpus, n=k)


def rerank(
    query: str,
    passages: list[str],
    top_k: int = 5,
    model: str = "BAAI/bge-reranker-base",
) -> list[str]:
    """TODO 8: score passages with fastembed TextCrossEncoder, return the best top_k."""

    encoder = TextCrossEncoder(model_name=model)
    scores = list(encoder.rerank(query, passages))
    ranked = sorted(
        zip(passages, scores, strict=True), key=lambda x: x[1], reverse=True
    )
    return [p for p, _ in ranked[:top_k]]


def build_and_search(corpus: list[str], query: str, k: int = 3) -> list[str]:
    """(integration): embed -> index -> hybrid retrieve -> rerank."""
    dense_model, sparse_model = dense_embedder(), sparse_embedder()
    dense_vecs = [v.tolist() for v in dense_model.encode(corpus)]
    sparse_vecs = [
        models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        for s in sparse_model.encode(corpus)
    ]
    store = HybridStore(dense_size=len(dense_vecs[0]))
    store.upsert(corpus, dense_vecs, sparse_vecs)

    dense_q = next(dense_model.query_embed(query)).tolist()
    sparse_q = next(iter(sparse_model.query_embed(query)))
    sq = models.SparseVector(
        indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()
    )

    candidates = [h.text for h in store.hybrid_search(dense_q, sq, k=10)]
    return rerank(query, [h.text for h in candidates], top_k=k)

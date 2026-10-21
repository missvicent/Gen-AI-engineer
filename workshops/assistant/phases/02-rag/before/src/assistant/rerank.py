from __future__ import annotations

import warnings
from dataclasses import replace

from assistant.rag import Chunk, Reranker, texts

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def cross_encoder(model_name: str = DEFAULT_MODEL) -> Reranker | None:

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        warnings.warn(
            "fastembed is not installed: reranking is disabled. "
            "Install it with `uv sync --group release`.",
            stacklevel=2,
        )
        return None

    model = TextCrossEncoder(model_name)

    def rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        texts_to_rank = texts(chunks)
        scores = list(model.rerank(query, texts_to_rank))
        pair_scores = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            replace(chunk, score=float(score), scored_by="rerank") for chunk, score in pair_scores
        ]

    return rerank

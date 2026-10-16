from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant.rag import RagStore
from assistant.rerank import cross_encoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
EVALS = ROOT / "evals" / "golden-set" / "evals"
CORPUS = EVALS / "corpus.jsonl"
GOLDEN = EVALS / "golden.jsonl"
K = 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _to_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row["id"],
            "text": row["text"],
        }
        for row in rows
    ]


def _exact_rows(golden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in golden if row["slice"] == "exact"]


def dense(store: RagStore, question: str) -> list[str]:
    q = question.lower().split()
    positions = store._dense(q)[:K]
    return [store.order[i].source for i in positions]


def fused(store: RagStore, question: str) -> list[str]:
    chunks = store.retrieve(question, k=K)
    return [chunk.source for chunk in chunks]


def recall(
    store: RagStore, rows: list[dict[str, Any]], func: Callable[[RagStore, str], list[str]]
) -> float:
    scores = []
    for row in rows:
        scores.append(
            len(set(func(store, row["question"])) & set(row["supporting_doc_ids"]))
            / len(row["supporting_doc_ids"])
        )
    return sum(scores) / len(scores)


def rerank(corpus_dict: list[dict[str, Any]], golden_exact: list[dict[str, Any]]) -> None:
    encoder = cross_encoder()
    if encoder is not None:
        rerank_score = recall(RagStore(corpus_dict, reranker=encoder), golden_exact, fused)
        print(f"Rerank score: {rerank_score}")
    else:
        print("Rerank score: n/a")


def main() -> None:
    corpus = load_jsonl(CORPUS)
    golden = load_jsonl(GOLDEN)

    corpus_dict = _to_list(corpus)
    golden_exact = _exact_rows(golden)

    store = RagStore(corpus_dict)
    dense_score = recall(store, golden_exact, dense)
    fused_score = recall(store, golden_exact, fused)

    print(f"Dense score: {dense_score}")
    print(f"Fused score: {fused_score}")
    rerank(corpus_dict, golden_exact)


if __name__ == "__main__":
    main()

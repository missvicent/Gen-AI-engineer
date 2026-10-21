from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.rag import RagStore
from assistant.rerank import cross_encoder
from scripts.recall import to_list

EVALS = Path(__file__).resolve().parents[6] / "evals" / "golden-set" / "evals"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


GOLDEN = _jsonl(EVALS / "golden.jsonl")
CORPUS = _jsonl(EVALS / "corpus.jsonl")
ABSTAIN = [c for c in GOLDEN if c.get("expects_abstention")]


@pytest.fixture(scope="module")
def store() -> RagStore:
    # The floor sits on the cross-encoder score; without it nothing abstains.
    reranker = cross_encoder()
    if reranker is None:
        pytest.skip("needs fastembed: uv sync --group release")
    return RagStore(to_list(CORPUS), reranker=reranker)


@pytest.mark.integration
@pytest.mark.parametrize("row", ABSTAIN, ids=lambda r: r["id"])
def test_unanswerable_questions_abstain(store, row):
    assert store.retrieve(row["question"]) == []


@pytest.mark.integration
def test_abstention_does_not_break_an_exact_id(store):
    assert store.retrieve("INV-88231") != []

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.rag import RagStore
from scripts.recall import to_list

EVALS = Path(__file__).resolve().parents[6] / "evals" / "golden-set" / "evals"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


GOLDEN = _jsonl(EVALS / "golden.jsonl")
CORPUS = _jsonl(EVALS / "corpus.jsonl")
ABSTAIN = [c for c in GOLDEN if c.get("expects_abstention")]


@pytest.fixture(scope="module")
def store() -> RagStore:
    corpus_dict = to_list(CORPUS)
    return RagStore(corpus_dict)


@pytest.mark.parametrize("row", ABSTAIN, ids=lambda r: r["id"])
def test_unanswerable_questions_abstain(store, row):
    assert store.retrieve(row["question"]) == []


def test_abstention_does_not_break_an_exact_id(store):
    assert store.retrieve("INV-88231") != []

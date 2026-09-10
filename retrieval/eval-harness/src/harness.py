from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz


@dataclass
class Row:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def load_golden(path: str | Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def answer_similarity_nonllm(row: Row) -> float:
    return fuzz.token_set_ratio(row.answer, row.ground_truth) / 100.0


def context_recall_nonllm(row: Row) -> float:
    if not row.contexts:
        return 0.0
    return max(
        fuzz.token_set_ratio(context, row.ground_truth) / 100.0
        for context in row.contexts
    )


def evaluate_nonllm(rows: list[Row]) -> dict[str, float]:
    if not rows:
        return {
            "answer_similarity_nonllm": 0.0,
            "context_recall_nonllm": 0.0,
        }
    n = len(rows)
    return {
        "answer_similarity_nonllm": sum(answer_similarity_nonllm(row) for row in rows)
        / n,
        "context_recall_nonllm": sum(context_recall_nonllm(row) for row in rows) / n,
    }


def build_dataset(
    golden: list[dict],
    pipeline: Callable[[str], tuple[str, list[str]]],
) -> list[Row]:
    return [
        Row(
            example["question"], *pipeline(example["question"]), example["ground_truth"]
        )
        for example in golden
    ]

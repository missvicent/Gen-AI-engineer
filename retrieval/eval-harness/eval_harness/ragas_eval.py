"""TODO (tier 2): the REAL RAGAS evaluation.

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

Verified against **ragas 0.4.3** (2026-07-31). The metric classes live in
`ragas.metrics.collections` and take a judge built by `llm_factory` over an
`AsyncOpenAI` client. If a tutorial hands you `from ragas.metrics import
Faithfulness` with `evaluate()` and `EvaluationDataset`, that is the pre-0.4
surface: it still imports, it raises a DeprecationWarning, and it will not match
what lesson 3.2 teaches. Metrics score ONE sample at a time via `score(...)`, so
you average across the golden set yourself.

- local_judge(): a FREE judge on Ollama (base_url=http://localhost:11434/v1),
  temperature=0. PIN IT — the judge is part of the ruler.
- as_score(value, metric): validate one verdict before it enters an average.
- aggregate(scored): per-row scores to the two means a gate reads.
- run_ragas(golden, pipeline, judge): score every row, return `aggregate`'s result.
- gate(scores): fail under faithfulness 0.85 / context_recall 0.80.

`as_score` and `aggregate` are separate functions on purpose, and `tests/test_gate.py`
is why: it tests both by the number, offline, with no judge and no `ragas` installed.
That only works because the ragas imports live INSIDE the functions that use them.
Do the same in your own harness — the arithmetic between a judge's verdict and a
merge decision is the part you can test properly, and it is the part that fails
quietly.

Run with: make test-integration (tier 2) or make test (tests/test_gate.py, always)
Reference: ../after/src/ragas_eval.py
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from importlib.metadata import version

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

load_dotenv()

FAITHFULNESS_BAR = 0.85
CONTEXT_RECALL_BAR = 0.80

DEFAULT_LOCAL_MODEL = "qwen3:8b"
DEFAULT_HOSTED_MODEL = "gpt-5.5"
OLLAMA_BASE_URL = os.getenv("BASE_URL")


def local_judge(model: str = DEFAULT_LOCAL_MODEL):
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=os.getenv("OLLAMA_API_KEY"))
    return llm_factory(model, client=client, temperature=0)


def hosted_judge(model: str = DEFAULT_HOSTED_MODEL):
    return llm_factory(
        model,
        client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        temperature=0,
    )


def as_score(value: object, metric: str) -> float:
    if value is None:
        raise ValueError(f"The Judge returned no score for {metric}")
    score = float(value)
    if math.isnan(score):
        raise ValueError(
            f"The Judge returned a NaN score for {metric}, which no threshold can catch"
        )
    if not 0 <= score <= 1:
        raise ValueError(
            f"The Judge returned a score for {metric} outside 0..1: {score}"
        )
    return score


def aggregate(scored: list[dict[str, float]]) -> dict[str, float]:
    """Per row scores to the two numbers a gate reads."""

    """An empty run must not read as a perfect one, and must not crash either.
    nightly jobs die halfway through, and `0.0` fails the gate, which is the
    correct outcome for a measurement that did not happen.
    """
    if not scored:
        return {"faithfulness": 0.0, "context_recall": 0.0}
    return {
        metric: sum(row[metric] for row in scored) / len(scored)
        for metric in ["faithfulness", "context_recall"]
    }


def run_ragas(
    golden: list[dict],
    pipeline: Callable[[str], tuple[str, list[str]]],
    judge=None,
) -> dict:
    judge = judge or local_judge()
    faithfulness, context_recall = Faithfulness(llm=judge), ContextRecall(llm=judge)
    scored: list[dict[str, float]] = []
    for row in golden:
        answer, contexts = pipeline(row["question"])
        scored.append(
            {
                "faithfulness": as_score(
                    faithfulness.score(
                        user_input=row["question"],
                        response=answer,
                        retrieved_contexts=contexts,
                    ),
                    "faithfulness",
                ),
                "context_recall": as_score(
                    context_recall.score(
                        user_input=row["question"],
                        retrieved_contexts=contexts,
                        reference=row["ground_truth"],
                    ),
                    "context_recall",
                ),
            }
        )
    return aggregate(scored)


def describe(model: str = DEFAULT_LOCAL_MODEL) -> dict[str, str]:
    """A score without its instrument is not a measurement. Report this
    beside every number you publish.
    """
    return {
        "judge_model": model,
        "judge_temperature": "0.0",
        "ragas_version": version("ragas"),
    }


def gate(scores: dict) -> tuple[bool, list[str]]:
    reasons = []
    f = scores.get("faithfulness", 0.0)
    c = scores.get("context_recall", 0.0)
    if f < FAITHFULNESS_BAR:
        reasons.append(f"faithfulness {f:.2f} < {FAITHFULNESS_BAR}")
    if c < CONTEXT_RECALL_BAR:
        reasons.append(f"context_recall {c:.2f} < {CONTEXT_RECALL_BAR}")
    return len(reasons) == 0, reasons

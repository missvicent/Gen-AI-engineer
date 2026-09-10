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

from collections.abc import Callable

FAITHFULNESS_BAR = 0.85
CONTEXT_RECALL_BAR = 0.80

DEFAULT_LOCAL_MODEL = "qwen3-coder:30b"
DEFAULT_HOSTED_MODEL = "gpt-5.5"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def local_judge(model: str = DEFAULT_LOCAL_MODEL):
    raise NotImplementedError  # TODO 1


def hosted_judge(model: str = DEFAULT_HOSTED_MODEL):
    """A hosted judge (needs OPENAI_API_KEY). Stronger, costs tokens. Same
    `llm_factory` call as the local one with the base URL left off — which is
    the point: any Ollama model speaks the OpenAI API."""
    raise NotImplementedError  # TODO 1b


def as_score(value: object, metric: str) -> float:
    """TODO 2: one judge verdict, checked before it is allowed into an average.

    Return `value` as a float in 0..1, and raise `ValueError` naming `metric` when
    it is not one. Three inputs to reject, and the middle one is the reason this
    function exists rather than being a `float()` call at the call site:

      * `None` — the judge did not answer. Reading that as 0.0 turns a broken judge
        into a quality regression, and sends someone to debug the retriever.
      * `NaN` — a NaN compares False against every threshold, so `nan < 0.85` is
        False and the gate PASSES. Worse, one NaN row makes the whole mean NaN, so
        a single unparsed verdict silences the gate for the entire run while the
        report still prints a number.
      * outside 0..1 — usually 0..100, because `rapidfuzz` returns percentages and
        RAGAS returns fractions. A faithfulness of 87.0 clears a 0.85 bar by two
        orders of magnitude and reads as a triumph.

    `tests/test_gate.py` has all three, and they are worth reading before you
    write this: they run offline, so you get them green in a second rather than
    after a nightly."""
    raise NotImplementedError  # TODO 2


def aggregate(scored: list[dict[str, float]]) -> dict[str, float]:
    """TODO 3: per-row scores to the two numbers a gate reads.

    The mean of each metric across `scored`. Average the two independently — one
    accumulator for both is a real bug that every passing run hides.

    The empty case is a decision, not an edge case: a nightly job that died before
    its first row has produced no evidence, so return 0.0 for both and let the gate
    fail. Raising here crashes the reporter, and defaulting to 1.0 turns a crashed
    eval into a green build."""
    raise NotImplementedError  # TODO 3


def run_ragas(
    golden: list[dict],
    pipeline: Callable[[str], tuple[str, list[str]]],
    judge=None,
) -> dict:
    """TODO 4: score every row of `golden`, then return `aggregate`'s result.

    `Faithfulness(llm=judge)` and `ContextRecall(llm=judge)`, one sample at a time
    through `score(...)`. Faithfulness takes `user_input`, `response` and
    `retrieved_contexts`; context recall takes `user_input`, `retrieved_contexts`
    and `reference`. Put every `.value` through `as_score` on its way in."""
    raise NotImplementedError  # TODO 4


def describe(model: str = DEFAULT_LOCAL_MODEL) -> dict[str, str]:
    """A score without its instrument is not a measurement. Return the judge
    model, its temperature, and `importlib.metadata.version("ragas")` — the
    three things that make today's 0.91 comparable to last week's 0.88."""
    raise NotImplementedError  # TODO 5


def gate(scores: dict) -> tuple[bool, list[str]]:
    raise NotImplementedError  # TODO 6

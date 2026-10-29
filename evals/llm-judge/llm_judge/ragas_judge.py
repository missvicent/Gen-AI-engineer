"""TODO: the real judge. Implement the `Judge` protocol with RAGAS metrics.

Verified API (ragas 0.4.3, 2026-07-31) — the current classes live in
`ragas.metrics.collections` and take an explicit judge:

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    judge = llm_factory("qwen3-coder:30b",
                        client=AsyncOpenAI(base_url="http://localhost:11434/v1",
                                           api_key="ollama"),
                        temperature=0)                    # pin the ruler
    Faithfulness(llm=judge).score(user_input=q, response=answer,
                                  retrieved_contexts=contexts).value

Note: `from ragas.metrics import Faithfulness` still imports but is deprecated in
favour of the path above. Metric instances have both `ascore()` and a synchronous
`score()`; both return a `MetricResult`, so read `.value`.

Run it with:  make test-integration      (needs Ollama running)

Reference: ../after/src/ragas_judge.py
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "qwen3-coder:30b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class RagasJudge:
    model: str = DEFAULT_MODEL
    base_url: str | None = OLLAMA_BASE_URL
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """TODO 1: build the pinned judge and the two metric objects."""
        raise NotImplementedError

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        """TODO 2: score faithfulness; return the MetricResult's value as a float."""
        raise NotImplementedError

    def context_recall(
        self, question: str, contexts: list[str], reference: str
    ) -> float:
        """TODO 3: score context recall against the reference answer."""
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        """TODO 4: model, temperature, endpoint and the installed ragas version.

        `importlib.metadata.version("ragas")` — record the instrument, always.
        """
        raise NotImplementedError

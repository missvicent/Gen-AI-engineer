"""TODO: the real judge. Implement the `Judge` protocol with RAGAS metrics.

Verified API (ragas 0.4.3, 2026-07-31) — the current classes live in
`ragas.metrics.collections` and take an explicit judge:

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    judge = llm_factory("qwen2.5:7b-instruct",
                        client=AsyncOpenAI(base_url="http://localhost:11434/v1",
                                           api_key="ollama"),
                        temperature=0)                    # pin the ruler
    Faithfulness(llm=judge).score(user_input=q, response=answer,
                                  retrieved_contexts=contexts).value

Note: `from ragas.metrics import Faithfulness` still imports but is deprecated in
favour of the path above. Metric instances have both `ascore()` and a synchronous
`score()`; both return a `MetricResult`, so read `.value`.

Run it with:  make test-integration      (needs Ollama running)
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

DEFAULT_MODEL = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class RagasJudge:
    model: str = DEFAULT_MODEL
    base_url: str | None = OLLAMA_BASE_URL
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.judge = llm_factory(
            self.model,
            client=AsyncOpenAI(base_url=self.base_url, api_key="ollama"),
            temperature=self.temperature,
        )
        self.faithfulness_metric = Faithfulness(llm=self.judge)
        self.context_recall_metric = ContextRecall(llm=self.judge)

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        result = self.faithfulness_metric.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        )

        return float(result.value)

    def context_recall(
        self, question: str, contexts: list[str], reference: str
    ) -> float:
        result = self.context_recall_metric.score(
            user_input=question, retrieved_contexts=contexts, reference=reference
        )

        return float(result.value)

    def describe(self) -> dict[str, str]:
        return {
            "judge_model": self.model,
            "judge_temperature": str(self.temperature),
            "base_url": f"{self.base_url}",
            "ragas_version": importlib.metadata.version("ragas"),
        }

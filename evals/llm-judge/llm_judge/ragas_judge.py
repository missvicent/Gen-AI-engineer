"""`Judge` protocol with RAGAS metric. RAGAS is a framework for evaluating RAG systems."""

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

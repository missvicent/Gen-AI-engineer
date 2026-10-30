"""The real eval: a pipeline's answers, scored by the real RAGAS judge.

    uv run --group integration python run_eval.py      # 2 rows per slice
    uv run --group integration python run_eval.py 10   # 10 rows per slice (all 50)

Needs Ollama running. Every answerable row is several calls to the 7B model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from llm_judge.harness import format_table, load_golden, run_suite
from llm_judge.ragas_judge import RagasJudge

EVALS = Path(__file__).parents[1] / "golden-set" / "evals"
CORPUS = [
    json.loads(line)
    for line in (EVALS / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 3}


def keyword_pipeline(question: str) -> tuple[str, list[str]]:
    """Toy RAG: retrieve the 2 chunks sharing the most words, answer with the best.

    ponytail: word overlap, no embeddings, no LLM answer; swap in the real RAG
    pipeline when it exists.
    """
    q = words(question)
    ranked = sorted(CORPUS, key=lambda d: len(q & words(d["text"])), reverse=True)
    if len(q & words(ranked[0]["text"])) < 2:
        return "Not in the documents.", []
    contexts = [d["text"] for d in ranked[:2]]
    return contexts[0], contexts


if __name__ == "__main__":
    per_slice = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows, seen = [], {}
    for row in load_golden(EVALS / "golden.jsonl"):
        if seen.get(row.slice, 0) < per_slice:
            rows.append(row)
            seen[row.slice] = seen.get(row.slice, 0) + 1

    result = run_suite(rows, keyword_pipeline, RagasJudge())
    for r in result.rows:
        print(r.id, r.slice, r.scores, "judged" if r.judged else "not judged")
    print()
    print(format_table(result))
    print("instrument:", result.instrument)

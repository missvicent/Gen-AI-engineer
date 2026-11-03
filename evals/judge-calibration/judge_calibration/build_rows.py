"""Turn the golden set into judged rows for you to label.

    make rows N=3   # try 3 rows first
    make rows       # all of them

Needs Ollama. Rerunning keeps what is already in the file and only adds what is
missing, so a crash loses nothing and your labels are never overwritten.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

MY_ROWS = Path(__file__).parents[1] / "evals" / "my_rows.jsonl"

PROMPT = """Answer the question using only the context below. Be brief.

Context:
{context}

Question: {question}"""


def build_rows(
    golden: Iterable,
    retrieve: Callable[[str], list[str]],
    write_answer: Callable[[str, list[str]], str],
    faithfulness: Callable[[str, str, list[str]], float],
    path: Path,
    limit: int | None = None,
) -> None:
    done = set()
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        done = {json.loads(line)["id"] for line in lines if line.strip()}

    with path.open("a", encoding="utf-8") as out:
        for row in golden:
            if limit is not None and len(done) >= limit:
                break
            if row.id in done or row.expects_abstention:
                continue
            contexts = retrieve(row.question)
            if not contexts:
                continue

            answer = write_answer(row.question, contexts)
            score = faithfulness(row.question, answer, contexts)
            # A NaN score can't be calibrated against, and it isn't valid JSON.
            if math.isnan(score):
                print(f"{row.id}: judge gave no score, skipped ({answer!r})")
                continue

            labeled = {
                "id": row.id,
                "slice": row.slice,
                "question": row.question,
                "answer": answer,
                "contexts": contexts,
                "judge_score": score,
                "human": "",
                "labeled_by": "",
                "labeled_on": "",
            }
            out.write(json.dumps(labeled) + "\n")
            out.flush()
            done.add(row.id)
            print(f"{row.id}: {len(done)} rows")


if __name__ == "__main__":
    from llm_judge.harness import load_golden
    from llm_judge.ragas_judge import DEFAULT_MODEL, OLLAMA_BASE_URL, RagasJudge
    from openai import OpenAI
    from run_eval import EVALS, keyword_pipeline

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def write_answer(question: str, contexts: list[str]) -> str:
        prompt = PROMPT.format(context="\n\n".join(contexts), question=question)
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    build_rows(
        load_golden(EVALS / "golden.jsonl"),
        # Only the chunks: keyword_pipeline's own answer is a copied chunk.
        lambda question: keyword_pipeline(question)[1],
        write_answer,
        RagasJudge().faithfulness,
        MY_ROWS,
        int(sys.argv[1]) if len(sys.argv) > 1 else None,
    )

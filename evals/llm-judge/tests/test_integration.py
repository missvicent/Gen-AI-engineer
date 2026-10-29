"""The judged tier: real RAGAS metrics, real model. Needs Ollama running.

    make test-integration

Kept to a couple of rows on purpose. The point of this tier is "the judged path
still works against today's library and today's model" — coverage lives offline.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

GOLDEN = Path(__file__).parents[2] / "golden-set" / "evals" / "golden.jsonl"


def test_the_real_judge_scores_a_grounded_answer_highly():
    from llm_judge.harness import load_golden, score_row
    from llm_judge.ragas_judge import RagasJudge

    judge = RagasJudge()
    row = next(r for r in load_golden(GOLDEN) if not r.expects_abstention)
    scored = score_row(row, row.ground_truth, [row.ground_truth], judge)

    assert scored.judged is True
    assert 0.0 <= scored.scores["faithfulness"] <= 1.0
    assert scored.scores["faithfulness"] >= 0.5, (
        "a verbatim-supported answer scoring below 0.5 means the judge or the "
        "rubric is broken — investigate before trusting any number from it"
    )


def test_the_instrument_is_recorded():
    from llm_judge.ragas_judge import RagasJudge

    described = RagasJudge().describe()
    assert described["judge_temperature"] == "0.0"
    assert described["ragas_version"].startswith("0.")

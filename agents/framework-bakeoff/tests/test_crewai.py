"""The agent on CrewAI (bakeoff/run_crewai.py) — live only.

    make live   (asks first; `ollama stop qwen3.5:9b` after)

CrewAI has no scripted test model to pass in, so this can't run offline.
That gap is itself the observability finding for the score.
"""

from __future__ import annotations

import agent_tools.tools as notes
import pytest

pytestmark = pytest.mark.integration


def test_answers_with_the_tool_live():
    from bakeoff.contract import TASK, used_the_tool
    from bakeoff.run_crewai import run_crewai

    r = run_crewai(TASK, notes.read_note)  # model=None: the real qwen3.5:9b

    assert r.framework == "crewai"
    assert "call dentist" in r.answer.lower()
    assert used_the_tool(r)

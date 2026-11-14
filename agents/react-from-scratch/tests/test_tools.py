import httpx
import pytest
from react_from_scratch.agent import Decision, run_agent
from react_from_scratch.tools import TOOL_SPECS, TOOLS, calculator, search


def test_calculator_does_arithmetic_and_nothing_else():
    # The expression comes from the model: it is untrusted input at a tool
    # boundary. Arithmetic must work; anything else must be refused, not run.
    assert calculator("2 + 40") == 42.0
    assert calculator("-(3 * 4) / 2") == -6.0
    with pytest.raises(ValueError):
        calculator("__import__('os').system('true')")
    with pytest.raises(ValueError):
        calculator("().__class__.__mro__")


def test_every_tool_has_a_spec_with_the_same_name():
    # The model picks a tool by the spec's name; the loop looks it up in TOOLS.
    assert {spec["name"] for spec in TOOL_SPECS} == set(TOOLS)


def test_agent_calls_search_then_finishes():
    def fake_tavily(request):
        return httpx.Response(
            200,
            json={"results": [{"title": "ReAct", "url": "https://x", "content": "c"}]},
        )

    fake = httpx.Client(transport=httpx.MockTransport(fake_tavily))
    tools = {"search": lambda query: search(query, client=fake)}
    seq = iter(
        [
            Decision(is_final=False, tool="search", args={"query": "react agent"}),
            Decision(is_final=True, answer="done"),
        ]
    )

    def decide(goal, state):
        if state:  # second call: the search result must be in the history
            _, result = state[-1]
            assert result.startswith("Untrusted web results")
            assert "https://x" in result
        return next(seq)

    assert run_agent("x", tools, decide) == "done"

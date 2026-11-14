import httpx
import pytest
from react_from_scratch.agent import Decision, run_agent
from react_from_scratch.tools import (
    TOOL_SPECS,
    TOOLS,
    calculator,
    search,
    validate,
    with_retry,
)


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


def test_validate_catches_each_kind_of_bad_call():
    def check(tool, **args):
        return validate(Decision(is_final=False, tool=tool, args=args), TOOL_SPECS)

    assert check("calc", expression="1+1") is None
    assert "unknown tool 'calculator'" in check("calculator", expression="1+1")
    assert "missing required args ['expression']" in check("calc")
    assert "unknown args ['x']" in check("calc", expression="1", x=2)
    assert "expression must be a string, got int" in check("calc", expression=5)


def test_with_retry_fixes_a_bad_call_without_using_a_step():
    seen = []

    def scripted(goal, history):
        seen.append(len(history))
        if not history:  # bad: wrong arg name
            return Decision(is_final=False, tool="calc", args={"expr": "6*7"})
        if "error: call not run" in str(history[-1][1]):  # sees the error, fixes it
            return Decision(is_final=False, tool="calc", args={"expression": "6*7"})
        return Decision(is_final=True, answer=f"it is {history[-1][1]}")

    ran = []
    tools = {
        "calc": lambda expression: ran.append(expression) or calculator(expression)
    }
    answer = run_agent("6*7?", tools, with_retry(scripted, TOOL_SPECS), max_steps=2)

    assert answer == "it is 42.0"
    assert ran == ["6*7"]  # the bad call never reached the tool
    assert seen == [0, 1, 1]  # retry saw the error; the loop history didn't keep it


def test_with_retry_gives_up_and_lets_the_loop_report_it():
    def always_bad(goal, history):
        return Decision(is_final=False, tool="nope")

    calls = []
    retrying = with_retry(lambda g, h: calls.append(1) or always_bad(g, h), TOOL_SPECS)
    assert retrying("q", []).tool == "nope"
    assert len(calls) == 3  # 1 try + 2 retries

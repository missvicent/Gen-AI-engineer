import httpx
import pytest
from react_from_scratch.agent import (
    TOOL_SPECS,
    TOOLS,
    Decision,
    calculator,
    run_agent,
    search,
)


def test_agent_calls_a_tool_then_finishes():
    # scripted "brain": first step calls calculator, second step returns the answer
    script = [
        Decision(is_final=False, tool="calc", args={"expression": "2 + 40"}),
        Decision(is_final=True, answer="42"),
    ]
    steps = iter(script)

    def decide(goal, state):
        return next(steps)

    out = run_agent("what is 2+40", {"calc": calculator}, decide)
    assert out == "42"


def test_hard_step_cap_prevents_infinite_loop():
    # a brain that NEVER finishes must still terminate — the leash in code
    def decide(goal, state):
        return Decision(is_final=False, tool="calc", args={"expression": "1+1"})

    out = run_agent("loop forever", {"calc": calculator}, decide, max_steps=3)
    assert "max_steps_exceeded" in out


def test_unknown_tool_becomes_an_observation_not_a_crash():
    seq = [
        Decision(is_final=False, tool="nope", args={}),
        Decision(is_final=True, answer="recovered"),
    ]
    it = iter(seq)
    out = run_agent("x", {"calc": calculator}, lambda g, s: next(it))
    assert out == "recovered"


def test_tool_error_becomes_an_observation():
    seq = [
        Decision(is_final=False, tool="calc", args={"expression": "2 +"}),
        Decision(is_final=True, answer="recovered"),
    ]
    it = iter(seq)

    def decide(goal, state):
        if state:  # second call: the failed step must be in the history
            _, result = state[-1]
            assert result.startswith("error: SyntaxError")
        return next(it)

    out = run_agent("x", {"calc": calculator}, decide)
    assert out == "recovered"


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

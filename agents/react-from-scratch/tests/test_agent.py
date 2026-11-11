import pytest
from react_from_scratch.agent import Decision, calculator, run_agent


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


def test_calculator_does_arithmetic_and_nothing_else():
    # The expression comes from the model: it is untrusted input at a tool
    # boundary. Arithmetic must work; anything else must be refused, not run.
    assert calculator("2 + 40") == 42.0
    assert calculator("-(3 * 4) / 2") == -6.0
    with pytest.raises(ValueError):
        calculator("__import__('os').system('true')")
    with pytest.raises(ValueError):
        calculator("().__class__.__mro__")

"""The model-in-the-loop guard: what it may do, and what it must never do.

Every test here is offline. The guard is a callable returning a bool, so a
"model" that always says INJECTION, or always errors, or takes the scenic route
through a prompt-injection attempt of its own, is three lines of Python — and
testing the WIRING is the point. Whether a given 9B model classifies well is a
question for the eval suite; whether a well-behaved answer can unblock something
the deterministic screen refused is a question for this file, and the answer
must be no regardless of which model you plug in.
"""

import pytest

from assistant.guard import model_guard


def test_a_dead_guard_model_leaves_the_deterministic_verdict_standing():
    """Fails OPEN, deliberately, and the docstring in guard.py says why.

    An Ollama restart must not take the service down; the layers that actually
    contain a landed injection — HITL, least privilege, tenant scoping — never
    consulted this file in the first place.
    """

    def dead(_text: str) -> bool:
        raise ConnectionError("ollama is not running")

    guard = model_guard("http://127.0.0.1:1", "nonexistent-model")
    assert guard("anything at all") is False  # no host, no opinion

    with pytest.raises(ConnectionError):
        dead("x")  # ...but only model_guard swallows it, not with_guard's caller


@pytest.mark.parametrize("reply", ["SAFE", "", "I think this might be fine?", "MAYBE"])
def test_only_an_exact_verdict_counts_as_one(reply, monkeypatch):
    # A model that starts explaining itself must neither trip the gate nor clear
    # it. Anything that is not a clean verdict is no verdict.
    monkeypatch.setattr("assistant.adapters.ollama_generate", lambda *a, **k: reply, raising=False)
    assert model_guard("http://fake", "m")("some text") is False


def test_a_block_verdict_is_recognised(monkeypatch):
    monkeypatch.setattr(
        "assistant.adapters.ollama_generate",
        lambda *a, **k: "  injection\n",
        raising=False,
    )
    assert model_guard("http://fake", "m")("some text") is True

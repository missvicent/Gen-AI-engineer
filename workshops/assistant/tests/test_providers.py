"""The three rules in providers.py, each tested against the way it used to fail.

None of these need a network, a key or a model. Provider selection is a decision
made from configuration, and a decision is testable without the thing it selects.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from assistant import providers
from assistant.composers import ABSTAIN, model_composer, model_stream_composer
from assistant.providers import ConfigError
from assistant.service import build_assistant, pricing_gap
from assistant.settings import Settings

HOST = "http://host.docker.internal:11434"


# --- rule 1: selection is explicit ---------------------------------------------


def test_an_unconfigured_deployment_still_gets_the_tier_it_always_had():
    """The historical rule survives, so no running stack changes brain because
    this module was added: a host means Ollama, no host means the stitcher."""
    assert providers.chat_provider(Settings(ollama_host=HOST)) == providers.OLLAMA
    assert providers.chat_provider(Settings()) == providers.OFFLINE


def test_naming_a_provider_beats_guessing_from_whatever_happens_to_be_set():
    """A host set for the embedder or the guard must not conscript the brain."""
    named = Settings(ollama_host=HOST, provider="offline")
    assert providers.chat_provider(named) == providers.OFFLINE
    assert build_assistant(named).tier()["brain"] == "rule-based"


def test_a_provider_nobody_implements_is_refused_by_name():
    with pytest.raises(ConfigError, match="gpt-9"):
        providers.chat_provider(Settings(provider="gpt-9"))


def test_the_brain_row_says_which_vendor_not_merely_that_there_is_one(monkeypatch):
    """`/health` used to answer "ollama" for anything with a model behind it. The
    bill, the latency and where the text goes all differ by vendor."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    hosted = Settings(provider="openai", chat_model="gpt-5.5")
    assert build_assistant(hosted).tier()["brain"] == "openai"


# --- rule 2: a missing credential is an error, not a downgrade -----------------


@pytest.mark.parametrize(
    ("provider", "variable"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
def test_a_hosted_provider_without_its_key_stops_the_boot(provider, variable, monkeypatch):
    """The failure this prevents is a service that answers for a week from the
    offline stitcher while its operator believes they are on a frontier model.
    The message names the variable, because "provider unavailable" sends someone
    reading source instead of reading their environment."""
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(ConfigError, match=variable):
        build_assistant(Settings(provider=provider, chat_model="some-model"))


def test_ollama_named_without_a_host_is_the_same_class_of_mistake(monkeypatch):
    with pytest.raises(ConfigError, match="OLLAMA_HOST"):
        build_assistant(Settings(provider="ollama"))


def test_a_hosted_provider_without_a_model_tag_refuses_to_guess_one(monkeypatch):
    """There is no safe default tag for a metered API: picking one spends the
    operator's money on a model they did not choose."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(ConfigError, match="ASSISTANT_CHAT_MODEL"):
        build_assistant(Settings(provider="anthropic"))


def test_the_offline_tier_needs_no_credentials_at_all(monkeypatch):
    """The zero-key path is the default and stays that way."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assistant = build_assistant(Settings())
    assert assistant.tier()["brain"] == "rule-based"
    assert assistant.degraded == {}


# --- rule 3: no provider ever routes to another --------------------------------


def test_a_failing_provider_falls_to_the_local_stitcher_and_never_to_a_vendor():
    """Cross-provider failover reads as resilience and is a way to spend money on
    an API nobody chose, in an incident nobody is watching, at a quality nobody
    measured. The standby is local, deterministic, and confessed on /health."""
    calls: list[str] = []

    def dead(_prompt: str) -> str:
        calls.append("primary")
        raise RuntimeError("connection refused")

    degraded: dict[str, str] = {}
    from assistant.fallbacks import fallback_composer

    compose = fallback_composer(model_composer(dead), lambda k, v: degraded.update({k: v}))
    answer = compose("what is the refund window", ["Refunds close after ten days."], [], [])

    assert calls == ["primary"], "the primary was tried exactly once per attempt path"
    assert "ten days" in answer, "the offline stitcher answered from the same evidence"
    assert "brain" in degraded, "/health has to say the model did not compose this"


def test_every_provider_shares_one_prompt_and_one_refusal_contract():
    """The composers take a callable rather than a host and a tag so a second
    backend cannot become a second copy of the prompt. Two brains disagreeing
    about what counts as a refusal is a bug nobody can reproduce."""
    seen: list[str] = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return "The window is ten business days. NO_ANSWER but I am not certain."

    answer = model_composer(spy)("how long", ["Refunds close after ten business days."], [], [])

    assert "[c1]" in seen[0], "the grounded prompt is the one that asks for citations"
    assert answer == "The window is ten business days."


def test_the_streaming_path_holds_the_same_contract_for_any_provider():
    def spy(_prompt: str) -> Iterator[str]:
        yield from ["The window ", "is ten days.", " NO_ANSWER", " trailing doubt"]

    streamed = "".join(model_stream_composer(spy)("how long", ["ten days"], [], []))
    assert "NO_ANSWER" not in streamed, "the internal sentinel never reaches a client"
    assert streamed.strip() == "The window is ten days."


def test_composition_still_abstains_before_reaching_any_provider():
    """No evidence means no call. A model handed nothing to ground on should not
    be asked to improvise a refusal — and a paid one should not be billed for it."""

    def never(_prompt: str) -> str:  # pragma: no cover - must not run
        raise AssertionError("the provider was called with no evidence")

    def never_stream(_prompt: str) -> Iterator[str]:  # pragma: no cover - must not run
        raise AssertionError("the provider was called with no evidence")

    assert model_composer(never)("anything", [], [], []) == ABSTAIN
    assert list(model_stream_composer(never_stream)("anything", [], [], [])) == [ABSTAIN]


# --- embedders are chosen separately, and just as explicitly -------------------


def test_the_embedder_is_never_inferred_from_the_chat_provider(monkeypatch):
    """Deriving it would move an entire corpus onto a different embedder because
    someone swapped the model that writes prose."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    on_openai = Settings(provider="openai", chat_model="gpt-5.5")
    assert providers.embed_provider(on_openai) == providers.OLLAMA


def test_asking_anthropic_to_embed_says_why_it_cannot():
    with pytest.raises(ConfigError, match="no embedding API"):
        providers.embed_provider(Settings(embed_provider="anthropic"))


# --- the cost gate has to be able to fail --------------------------------------


def test_a_metered_provider_priced_as_local_is_reported(monkeypatch):
    """`local` prices every token at zero, which is true on your own hardware and
    fiction against an API. Left alone, the cost gate reads free on a run that is
    generating an invoice — a gate that cannot fail is not a gate."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    hosted = Settings(provider="openai", chat_model="gpt-5.5", price_tier="local")
    assert "cost gate cannot fail" in (pricing_gap(hosted) or "")
    assert "cost" in build_assistant(hosted).degraded

    priced = Settings(provider="openai", chat_model="gpt-5.5", price_tier="frontier")
    assert pricing_gap(priced) is None


def test_a_local_model_priced_as_local_is_simply_correct():
    assert pricing_gap(Settings(ollama_host=HOST)) is None

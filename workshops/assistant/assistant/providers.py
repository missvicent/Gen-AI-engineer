"""Which brain answers — chosen once, out loud, and never guessed.

The capstone's local path is host-installed Ollama. OpenAI and Anthropic are
supported alternatives for anyone who would rather rent a frontier model than
run a 9B, and they are reached the same way everything else optional in this
codebase is reached: by naming them.

Three rules hold this module together, and all three exist because the failure
they prevent is silent.

**Selection is explicit.** `ASSISTANT_PROVIDER` names the brain. Left unset, the
historical rule still applies — Ollama when `OLLAMA_HOST` is set, the offline
stitcher otherwise — so every existing deployment keeps the tier it had. What
cannot happen is a provider being chosen because another one was absent.

**A missing credential is an error, not a downgrade.** Everywhere else in this
codebase an absent optional thing degrades and reports: a reranker that will not
load, an MCP server that will not answer, a Qdrant that is down. Those are
runtime failures of things that were configured correctly. Naming `anthropic`
with no `ANTHROPIC_API_KEY` is not that — it is a deployment that cannot do what
it was asked to do, and the useful response is to stop at boot with a sentence
naming the variable. The alternative is a service that quietly answers from the
offline stitcher for a week while its operator believes they are on Claude.

**No provider ever routes to another.** When the selected brain fails mid-request
the answer comes from `offline_compose`, which is local, deterministic, and
reported on `/health` as a degradation. It does not come from a different vendor.
Cross-provider failover sounds like resilience and is actually a way to spend
money on an API nobody chose, in an incident nobody is watching, at a quality
nobody measured.

Keys are read by each SDK from the environment and never passed as arguments,
so there is no call site holding one and no traceback that can print one.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

#: The brains. `offline` is a real choice, not the absence of one — it is the
#: deterministic stitcher the fast tier runs on, and naming it is how a
#: deployment says "no model, on purpose".
OLLAMA, OPENAI, ANTHROPIC, OFFLINE = "ollama", "openai", "anthropic", "offline"
CHAT_PROVIDERS = (OLLAMA, OPENAI, ANTHROPIC, OFFLINE)

#: Embedders are chosen separately and just as explicitly. Anthropic has no
#: embedding API, so a Claude deployment still names one of these two.
EMBED_PROVIDERS = (OLLAMA, OPENAI)

#: What each hosted provider needs before it can be built. Ollama needs a host
#: rather than a key, which is why it is not in here.
CREDENTIAL = {OPENAI: "OPENAI_API_KEY", ANTHROPIC: "ANTHROPIC_API_KEY"}

Generate = Callable[[str], str]
Stream = Callable[[str], Iterator[str]]


class ConfigError(RuntimeError):
    """A provider was named and cannot be built.

    Raised at composition time, so it surfaces as a boot failure with a readable
    message rather than as a degraded tier nobody reads. `build_assistant` lets
    it through deliberately — see the module docstring.
    """


@dataclass(frozen=True)
class Chat:
    """A brain, reduced to what the composer needs: a name for `/health`, the
    model tag for the report, and two callables."""

    provider: str
    model: str
    generate: Generate
    stream: Stream


def chat_provider(settings) -> str:
    """The brain this deployment asked for.

    Unset means the historical rule, which is a real answer and not a default
    being applied twice: a stack with `OLLAMA_HOST` set has always been on
    Ollama, and one without it has always been offline.
    """
    named = (settings.provider or "").strip().lower()
    if not named:
        return OLLAMA if settings.ollama_host else OFFLINE
    if named not in CHAT_PROVIDERS:
        raise ConfigError(
            f"ASSISTANT_PROVIDER={named!r} is not a provider. "
            f"Choose one of: {', '.join(CHAT_PROVIDERS)}."
        )
    return named


def embed_provider(settings) -> str:
    """Which service computes vectors. Defaults to Ollama and is never inferred
    from the chat provider: inferring it is precisely the silent switch this
    module exists to prevent, and it would move a corpus onto a different
    embedder because someone changed which model writes prose."""
    named = (settings.embed_provider or OLLAMA).strip().lower()
    if named not in EMBED_PROVIDERS:
        raise ConfigError(
            f"ASSISTANT_EMBED_PROVIDER={named!r} cannot embed. "
            f"Choose one of: {', '.join(EMBED_PROVIDERS)}"
            + (
                " — Anthropic has no embedding API, so a Claude deployment "
                "still names one of these."
                if named == ANTHROPIC
                else "."
            )
        )
    return named


def model_tag(settings) -> str:
    """Which model answered, for the trace and the report stamp.

    Cheap and pure, because `/health` and every compose span ask for it and
    neither should be building an SDK client to find out. "offline-stitcher" is a
    model too: a number produced by the stitcher and a number produced by a 9B
    are not comparable, and the trace has to say which one it was.
    """
    provider = chat_provider(settings)
    if provider == OFFLINE:
        return "offline-stitcher"
    if provider == OLLAMA:
        return settings.ollama_model
    return settings.chat_model or provider


def require_credential(provider: str) -> None:
    """Stop now, naming the variable, if the selected provider has no key."""
    variable = CREDENTIAL.get(provider)
    if variable and not os.getenv(variable):
        raise ConfigError(
            f"ASSISTANT_PROVIDER={provider} needs {variable}, which is not set. "
            f"Export it, or select a different provider — this will not fall "
            f"back to a local model on your behalf."
        )


def build_chat(settings) -> Chat | None:
    """The selected brain, or `None` for the offline tier.

    `None` rather than an offline `Chat` because the caller does more than swap a
    callable: the offline path skips the retry policy and the streaming budget
    entirely, since there is no network to be slow.
    """
    provider = chat_provider(settings)
    if provider == OFFLINE:
        return None

    if provider == OLLAMA:
        if not settings.ollama_host:
            raise ConfigError(
                "ASSISTANT_PROVIDER=ollama needs OLLAMA_HOST — the URL of the "
                "Ollama daemon. From inside Docker that is "
                "http://host.docker.internal:11434; on the host it is "
                "http://localhost:11434."
            )
        from assistant.adapters import ollama_generate, ollama_stream

        host, model = settings.ollama_host, settings.ollama_model
        return Chat(
            provider=OLLAMA,
            model=model,
            generate=lambda prompt: ollama_generate(prompt, host=host, model=model),
            stream=lambda prompt: ollama_stream(prompt, host=host, model=model),
        )

    require_credential(provider)
    model = settings.chat_model
    if not model:
        raise ConfigError(
            f"ASSISTANT_PROVIDER={provider} needs ASSISTANT_CHAT_MODEL — there is "
            f"no sensible default tag for a paid API, and guessing one bills you "
            f"for a model you did not pick."
        )

    if provider == OPENAI:
        from assistant.adapters import openai_generate, openai_stream

        return Chat(
            provider=OPENAI,
            model=model,
            generate=lambda prompt: openai_generate(prompt, model=model),
            stream=lambda prompt: openai_stream(prompt, model=model),
        )

    from assistant.adapters import anthropic_generate, anthropic_stream

    return Chat(
        provider=ANTHROPIC,
        model=model,
        generate=lambda prompt: anthropic_generate(prompt, model=model),
        stream=lambda prompt: anthropic_stream(prompt, model=model),
    )


def build_embedder(settings) -> tuple[Callable[[str], list[float]], str]:
    """The embedding function and the signature that names it in the collection.

    Only called when a model was named. The signature travels into the Qdrant
    collection name (`adapters.collection_name`), so switching embedder or
    provider writes a new collection instead of reading yesterday's vectors as
    today's — two 768-dimensional models are interchangeable to Qdrant and
    meaningless to each other.
    """
    provider = embed_provider(settings)
    model = settings.embed_model

    if provider == OPENAI:
        require_credential(OPENAI)
        from assistant.adapters import openai_embed

        return openai_embed(model), f"{OPENAI}/{model}"

    if not settings.ollama_host:
        raise ConfigError(
            f"ASSISTANT_EMBED_MODEL={model} runs on Ollama and OLLAMA_HOST is not "
            f"set. Set it, or select ASSISTANT_EMBED_PROVIDER=openai."
        )
    from assistant.adapters import ollama_embed

    return ollama_embed(settings.ollama_host, model), model

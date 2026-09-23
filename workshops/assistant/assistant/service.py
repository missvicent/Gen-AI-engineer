"""Composition root — the capstone. `build_assistant` wires every layer you built
across the workshops into one running thing:

    request -> auth (optional Bearer JWT, env-gated)      [api.py — given]
            -> screen (input; guard.py adds a model when configured)
                                                          [core.py]
            -> agent.run loop  (observe span)             [core.py]
                 -> tools (screening.py re-screens read-only content)
                 -> RAG (offline BM25 or Qdrant)          [adapters.py]
                 -> memory (in-process or SQLite)
            -> compose (offline stitcher or Ollama)       [composers.py]
                 -> degraded fallbacks on failure         [fallbacks.py — given]
            -> guardrails.output_ok (response)            [core.py]

This module and api.py are GIVEN — they select adapters from settings, wire the
degraded-operation fallbacks, and expose the HTTP surface including /ask/stream,
the JWT gate, load shedding, idempotent approvals, and the audit log. Your job
is the composition LOGIC in between: `harden_registry` (screening.py), the
expand/squash scan (guardrails.py), the model-in-the-loop second opinion
(guard.py), `required_args_of` (tools.py), `choose` and friends (planner.py),
`store_for` (tenancy.py), `offline_compose` (composers.py), and the `recall` +
`ingest` + `_gather` pipeline (core.py) that ask() and ask_stream() both build
on. Keep it offline and deterministic so the fast tier (tests/test_service.py)
stays green with no network.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from assistant import providers
from assistant.adapters import InMemoryRag
from assistant.approvals import ApprovalStore
from assistant.audit_log import AuditLog
from assistant.composers import (
    Composer,
    StreamComposer,
    model_composer,
    model_stream_composer,
    offline_compose,
    word_stream,
)
from assistant.core import Assistant
from assistant.fallbacks import (
    COMPOSE_POLICY,
    FallbackRag,
    fallback_composer,
    fallback_stream,
)
from assistant.guard import build_screen
from assistant.idempotency import IdempotencyStore
from assistant.memory import AssistantMemory
from assistant.observe import recorder
from assistant.outbox import Outbox
from assistant.settings import Settings
from assistant.tenancy import TenantMemory
from assistant.tools import REGISTRY, rewrap


def build_reranker(settings: Settings, report: Callable[[str, str], None]):
    """A cross-encoder over the retrieved candidates, or None.

    Retrieval optimises recall — get the right passage into the top twenty. A
    reranker optimises precision — get it to position one, by scoring the query
    and the passage TOGETHER rather than comparing two vectors computed apart.
    That is a real second model on the request path, which is why it is opt-in.

    `fastembed` is not a dependency of this image: naming a rerank model on a
    host without it is reported as a degradation and retrieval carries on
    unreranked, because a missing optional accelerator is not a reason to stop
    answering.
    """
    if not settings.rerank_model:
        return None
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        report("rerank", f"{settings.rerank_model} needs fastembed, which is not installed")
        return None

    # A name fastembed does not recognise is the same class of problem as a
    # missing library and gets the same answer. It was not: an unsupported tag
    # raised inside the constructor, which meant `build_assistant` raised, which
    # meant a typo in one optional environment variable took the whole service
    # down at boot — the release run that found this got a ValueError about ONNX
    # weights instead of a report. Degrading here keeps one accelerator's absence
    # from being an outage, and `/health` still says the rerank stage is gone.
    try:
        encoder = TextCrossEncoder(model_name=settings.rerank_model)
    except Exception as exc:  # noqa: BLE001 — any load failure means "no reranker"
        report("rerank", f"{settings.rerank_model} did not load: {exc}")
        return None

    def rerank(query: str, chunks: list) -> list:
        scores = list(encoder.rerank(query, [c.text for c in chunks]))
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: -p[1])
        return [chunk for chunk, _ in ranked]

    return rerank


def pricing_gap(settings: Settings) -> str | None:
    """Why this deployment's cost number is fiction, or None.

    `ASSISTANT_PRICE_TIER` defaults to "local", which prices every token at zero
    — the truth for a model on your own hardware, and the reason the cost gate
    reads free in this course. Point the composer at a metered API and leave the
    tier alone and the gate still reads free, on a run that is generating an
    invoice. A gate that cannot fail is not a gate.

    Reported rather than corrected: guessing which tier someone pays for is the
    same class of mistake as guessing their model tag.
    """
    from assistant.providers import OFFLINE, OLLAMA, chat_provider

    hosted = chat_provider(settings) not in (OLLAMA, OFFLINE)
    if hosted and settings.price_tier == "local":
        return (
            "a hosted provider is billing per token while ASSISTANT_PRICE_TIER=local "
            "prices them at zero — the cost gate cannot fail"
        )
    return None


def relevance_gap(settings: Settings) -> str | None:
    """Why this deployment's retrieval cannot say "nothing here", or None.

    A vector store ranks everything and rejects nothing: ask about timezones and
    nearest-neighbour search returns the three least-unrelated rows in the corpus,
    with no signal that it found no answer. Whatever runs next is then holding
    evidence for a question nobody asked. `ASSISTANT_MIN_SCORE` is the only place
    that judgement can be made, because it is the only place the scores exist.

    It sets the DENSE arm's floor. The sparse arm is admitted by an exact-identifier
    rule instead, which needs no configuration — so this setting being unset means
    semantic retrieval cannot abstain, not that nothing can be retrieved.

    A predicate rather than an inline `if` so the rule can be tested without a
    Qdrant to connect to — the condition is worth a test, and reaching it through
    `build_assistant` needs the store the condition is about.
    """
    if settings.qdrant_url and not settings.min_score:
        return "ASSISTANT_MIN_SCORE is unset, so retrieval cannot abstain"
    return None


def build_assistant(settings: Settings | None = None) -> Assistant:
    settings = settings or Settings.from_env()

    degraded: dict[str, str] = {}

    def report(component: str, reason: str) -> None:
        degraded[component] = reason

    if settings.qdrant_url:
        from assistant.adapters import QdrantStore, hash_embed
        # A real embedder if one is named, the hash vector otherwise, and the
        # fallback is REPORTED rather than silent: retrieval that quietly runs on
        # vocabulary overlap looks like retrieval right up until someone asks
        # about "reimbursements" and gets nothing about refunds.
        embed, signature = hash_embed, "hash"
        if settings.embed_model:
            embed, signature = providers.build_embedder(settings)
        # Reported rather than tolerated, because the alternative is finding out
        # from an answer: the deployed stack ran without a floor and told a caller
        # "I don't know" while citing three refund policies at a question about
        # timezones.
        if gap := relevance_gap(settings):
            report("relevance", gap)
        rag: Any = FallbackRag(
            QdrantStore(
                settings.qdrant_url,
                settings.qdrant_collection,
                embed=embed,
                # Which embedder wrote these vectors, carried into the collection
                # name. Two 768-dimensional models are interchangeable to Qdrant
                # and meaningless to each other; the name is what keeps a model
                # swap from reading yesterday's vectors as today's.
                signature=signature,
                min_score=settings.min_score,
                rerank=build_reranker(settings, report),
            ),
            InMemoryRag(), report,
        )
    else:
        rag = InMemoryRag()

    # One store PER SUBJECT, not one store with a subject label on each row —
    # see tenancy.py for the recall leak that distinction closes.
    if settings.assistant_db:
        from assistant.sqlite_memory import SqliteMemory
        db = settings.assistant_db
        memory = TenantMemory(lambda subject: SqliteMemory(db, user=subject))
    else:
        memory = TenantMemory(lambda subject: AssistantMemory(user=subject))

    registry = dict(REGISTRY)
    if settings.telegram_bot_token and "send_telegram" in registry:
        from assistant.connectors import telegram_sender
        registry["send_telegram"] = rewrap(
            registry["send_telegram"], telegram_sender(settings.telegram_bot_token)
        )
    if settings.news_feed_url and "read_news" in registry:
        from assistant.connectors import news_fetcher
        registry["read_news"] = rewrap(
            registry["read_news"], news_fetcher(settings.news_feed_url)
        )
    if settings.mcp_server:
        from assistant.adapters import mcp_tools
        from assistant.mcp_client import extend_assistant
        try:
            discovered, invoker = mcp_tools(settings.mcp_server)
            registry = extend_assistant(
                registry, discovered, invoker, settings.mcp_readonly_allowlist
            )
        except Exception as exc:
            # boot with the builtin tools rather than crash-loop behind a dead
            # MCP server; /health shows the hole
            report("tools", f"MCP discovery failed, builtin tools only: {exc}")

    # Which brain, decided in one place and allowed to fail loudly. A named
    # provider missing its credential raises out of here rather than degrading:
    # that is a deployment that cannot do what it was configured to do, and a
    # boot error is the only version of that an operator reads. Runtime failures
    # of a correctly configured provider still degrade, below.
    chat = providers.build_chat(settings)
    if gap := pricing_gap(settings):
        report("cost", gap)
    if chat is not None:
        # One budget, both paths. The batch composer retries inside it and the
        # stream applies it per chunk; what they must not do is disagree, because
        # then "the model was too slow" means two different things depending on
        # which endpoint the caller used.
        policy = replace(COMPOSE_POLICY, timeout=settings.compose_timeout)
        # The fallback target is `offline_compose` for every provider — local,
        # deterministic, and reported on /health. Never another vendor's API:
        # cross-provider failover spends money nobody approved, in an incident
        # nobody is watching, at a quality nobody measured.
        compose: Composer = fallback_composer(
            model_composer(chat.generate), report, policy
        )
        stream_compose: StreamComposer = fallback_stream(
            model_stream_composer(chat.stream), report, settings.compose_timeout,
        )
    else:
        compose = offline_compose
        stream_compose = word_stream(offline_compose)

    # One screen object for every untrusted channel — question, retrieved docs,
    # tool output, ingested docs. Built here so turning the guard model on is a
    # composition-root decision and not four independent ones.
    screen = build_screen(settings.ollama_host, settings.guard_model)

    # Outstanding approvals, replay protection, the outbox and the audit trail
    # share the memory DB, so all four survive a restart with it. The outbox
    # especially: an outbox that lives in `:memory:` records intents right up
    # until the crash it exists to explain.
    store = settings.assistant_db or ":memory:"

    return Assistant(
        settings=settings, rag=rag, memory=memory, base_registry=registry,
        rec=recorder(otlp_endpoint=settings.otlp_endpoint),
        compose=compose, stream_compose=stream_compose,
        approvals=ApprovalStore(store, settings.approval_ttl_seconds),
        idempotency=IdempotencyStore(store), outbox=Outbox(store),
        audit_log=AuditLog(store),
        degraded=degraded, screen=screen,
    )

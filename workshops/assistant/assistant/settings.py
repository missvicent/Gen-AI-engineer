"""Composition settings — one place that reads the environment.

The whole capstone runs offline and deterministic by default. Each real adapter
turns on only when its env var is present, so the fast tier can never accidentally
reach the network and CI stays free:

    ASSISTANT_DB                  -> SQLite-backed memory instead of in-process dict
    QDRANT_URL                    -> QdrantStore instead of the offline BM25 RagStore
    OLLAMA_HOST                   -> Ollama brain instead of the rule-based planner
    ASSISTANT_PROVIDER            -> name the brain outright: ollama | openai | anthropic
                                     | offline. Unset keeps the rule above. A hosted
                                     provider without its key is a boot error, never a
                                     quiet downgrade — see providers.py
    ASSISTANT_CHAT_MODEL          -> the model tag for a hosted provider (no default:
                                     guessing one bills you for a model you did not pick)
    ASSISTANT_EMBED_PROVIDER      -> ollama (default) | openai. Chosen separately from
                                     the chat provider and never inferred from it
    MCP_SERVER                    -> discover tools from a real MCP server
    ASSISTANT_MCP_READONLY_ALLOWLIST -> discovered tools the operator has reviewed as reads
                                     (the only thing that can ungate one; a server saying
                                     so about itself cannot)
    OTEL_EXPORTER_OTLP_ENDPOINT   -> ship spans over OTLP as well as keeping them in memory
    ASSISTANT_JWT_SECRET          -> require a Bearer JWT (HS256, shared secret)
    ASSISTANT_JWKS_URL            -> verify RS256 tokens against an issuer's JWKS instead
    ASSISTANT_JWT_ISSUER          -> also require this `iss` claim
    ASSISTANT_JWT_LEEWAY          -> clock-skew tolerance on exp/nbf (default 60s)
    ASSISTANT_APPROVAL_TTL        -> seconds a human approval stays spendable (default 300)
    ASSISTANT_STREAM_MODE         -> "raw" emits chunks before the output gate (local only)
    ASSISTANT_GUARD_MODEL         -> a second-opinion guard model on every untrusted string
    ASSISTANT_EMBED_MODEL         -> real embeddings (via Ollama) instead of the hash vector
    ASSISTANT_PRICE_TIER          -> which price list `make report` costs a run against
    TELEGRAM_BOT_TOKEN            -> send_telegram talks to the real Telegram Bot API
    NEWS_FEED_URL                 -> read_news fetches a real RSS feed (keyless)
    RATE_LIMIT_RPS (+_BURST)      -> token-bucket rate limit on every non-health route
    MAX_CONCURRENCY               -> reject (503) beyond this many in-flight requests
    REQUEST_DEADLINE_SECONDS      -> one budget per request; every layer's timeout fits inside it
    COMPOSE_TIMEOUT_SECONDS       -> how long one composition may take before the offline
                                     stitcher answers instead (default 60; raise it for a
                                     CPU-only container, where a 9B needs minutes)

Nothing here imports a heavy library; the adapters do that lazily when selected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from assistant.auth import DEFAULT_LEEWAY, AuthPolicy
from assistant.output_gate import SAFE_BUFFERED

#: The default composition budget, in seconds. Lives here rather than in
#: fallbacks.py because it is a deployment policy, not a property of the
#: fallback: the same code is right at 60 seconds behind a GPU and wrong at 60
#: seconds inside a VM that has none.
COMPOSE_TIMEOUT_SECONDS = 60.0


def _names(raw: str) -> tuple[str, ...]:
    """A comma-separated env var as a tuple, blanks dropped. Unset and empty
    both mean the empty tuple, which for an allowlist is the safe reading."""
    return tuple(name.strip() for name in raw.split(",") if name.strip())


@dataclass(frozen=True)
class Settings:
    assistant_db: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "assistant"
    ollama_host: str | None = None
    ollama_model: str = "qwen2.5:8b"
    # Which brain, named rather than deduced. Empty keeps the historical rule
    # (Ollama when a host is set, offline otherwise) so no existing deployment
    # changes tier; anything else is an operator's explicit choice and is
    # validated at boot. providers.py owns the vocabulary and the errors.
    provider: str = ""
    # The tag for a hosted provider. Deliberately without a default: `qwen3.5:9b`
    # is free and local, and a default for a metered API is a bill nobody chose.
    chat_model: str | None = None
    # Which service computes vectors. Separate from the chat provider on purpose
    # — deriving it would move an entire corpus onto a different embedder because
    # someone swapped the model that writes prose.
    embed_provider: str = ""
    mcp_server: str | None = None
    # Comma-separated names of DISCOVERED tools the operator has reviewed and
    # judged read-only. This is the only thing that can ungate one: a server's
    # own `readOnlyHint` is a claim by the party that wants the call made, so it
    # can add caution and never remove it. Empty by default, which means every
    # tool that arrives by discovery pauses for approval — including, at first
    # boot, the ones that turn out to be harmless.
    mcp_readonly_allowlist: tuple[str, ...] = ()
    otlp_endpoint: str | None = None
    context_budget_tokens: int = 120
    # auth is opt-in so the zero-key demo path keeps working out of the box.
    # A shared secret verifies HS256; a JWKS URL verifies RS256 against the
    # issuer's public keys, which is what an OAuth 2.1 deployment actually does.
    jwt_secret: str | None = None
    jwks_url: str | None = None
    jwt_audience: str = "assistant"
    jwt_issuer: str | None = None
    jwt_leeway: float = DEFAULT_LEEWAY
    # how long a human approval stays spendable — see approvals.py
    approval_ttl_seconds: float = 300.0
    # "safe-buffered" screens before releasing; "raw" emits first — see output_gate.py
    stream_mode: str = SAFE_BUFFERED
    # a model-in-the-loop second opinion on the deterministic screen. Off by
    # default: it costs a round trip per untrusted string, and it can only ever
    # ADD a block — see guard.py for why that direction is not negotiable. Runs
    # on OLLAMA_HOST, so naming a guard model without a host does nothing.
    guard_model: str | None = None
    # which embedder the vector store uses. Unset means `hash_embed`, which is
    # deterministic, offline and NOT semantic: it matches on shared vocabulary,
    # so "reimbursement" will not find a page about "refunds". Naming a model
    # here (nomic-embed-text, mxbai-embed-large) turns the same store into one
    # with real recall. It runs on OLLAMA_HOST, so a model without a host does
    # nothing. Changing it invalidates every vector already written, which the
    # store now handles by putting the embedder and its width in the collection
    # NAME — a model swap becomes a new collection instead of a silent
    # corruption of the old one (adapters.collection_name).
    embed_model: str | None = None
    # The floor a retrieved chunk's dense similarity must clear to count as
    # evidence. Vector search never abstains — ask it something absent from the
    # corpus and it returns the three least-unrelated documents, which the
    # composer will then ground an answer in. 0.0 keeps every hit, which is the
    # old behaviour and the right default: the useful cut depends on the
    # embedder and the corpus, and a wrong one abstains on good answers.
    min_score: float = 0.0
    # A cross-encoder that re-scores the retrieved candidates. Optional and off
    # by default: it is a second model on the request path, and hybrid retrieval
    # already covers most of what it buys. Named here so the wiring is real
    # rather than described.
    rerank_model: str | None = None
    # which price list `report.py` costs a measured run against (crew.PRICE).
    # "local" is the truth for a self-hosted model — no per-token invoice — and
    # it is also why the cost gate looks free here. Point the composer at a paid
    # API and set this to the tier you actually pay for, or the gate is theatre.
    price_tier: str = "local"
    # real connectors are opt-in for the same reason; the stubs are the default
    telegram_bot_token: str | None = None
    news_feed_url: str | None = None
    # load shedding — off by default so tests and demos can hammer freely
    rate_limit_rps: float | None = None
    rate_limit_burst: float = 10.0
    max_concurrency: int | None = None
    # One budget for the whole request, shared by every layer (deadline.py).
    # Off by default because the fast tier drives a real model in the integration
    # lane and a deadline there would be a flake generator; ON in the deployed
    # profile, where an unbounded request is a worker held hostage. Without it,
    # each layer's timeout composes by ADDITION and the total is a number nobody
    # has ever computed.
    request_deadline: float | None = None
    # How long one composition may take before the offline stitcher answers
    # instead. 60 seconds suits a model with a GPU behind it and is wrong for the
    # self-contained lane, where Docker Desktop gives the container no GPU and a
    # 9B runs at half a token per second: every answer timed out, the fallback
    # composed it, and the end-to-end run failed on a stack that was working
    # exactly as configured. A hard-coded constant made that unfixable without
    # editing library code, which is how it stayed broken.
    compose_timeout: float = COMPOSE_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            assistant_db=os.getenv("ASSISTANT_DB"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "assistant"),
            ollama_host=os.getenv("OLLAMA_HOST"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:8b"),
            provider=os.getenv("ASSISTANT_PROVIDER", ""),
            chat_model=os.getenv("ASSISTANT_CHAT_MODEL"),
            embed_provider=os.getenv("ASSISTANT_EMBED_PROVIDER", ""),
            mcp_server=os.getenv("MCP_SERVER"),
            mcp_readonly_allowlist=_names(
                os.getenv("ASSISTANT_MCP_READONLY_ALLOWLIST", "")
            ),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            jwt_secret=os.getenv("ASSISTANT_JWT_SECRET"),
            jwks_url=os.getenv("ASSISTANT_JWKS_URL"),
            jwt_audience=os.getenv("ASSISTANT_JWT_AUDIENCE", "assistant"),
            jwt_issuer=os.getenv("ASSISTANT_JWT_ISSUER"),
            jwt_leeway=float(os.getenv("ASSISTANT_JWT_LEEWAY", str(DEFAULT_LEEWAY))),
            approval_ttl_seconds=float(os.getenv("ASSISTANT_APPROVAL_TTL", "300")),
            stream_mode=os.getenv("ASSISTANT_STREAM_MODE", SAFE_BUFFERED),
            guard_model=os.getenv("ASSISTANT_GUARD_MODEL"),
            embed_model=os.getenv("ASSISTANT_EMBED_MODEL"),
            min_score=float(os.getenv("ASSISTANT_MIN_SCORE", "0")),
            rerank_model=os.getenv("ASSISTANT_RERANK_MODEL"),
            price_tier=os.getenv("ASSISTANT_PRICE_TIER", "local"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            news_feed_url=os.getenv("NEWS_FEED_URL"),
            rate_limit_rps=(
                float(rps) if (rps := os.getenv("RATE_LIMIT_RPS")) else None
            ),
            rate_limit_burst=float(os.getenv("RATE_LIMIT_BURST", "10")),
            max_concurrency=(int(cc) if (cc := os.getenv("MAX_CONCURRENCY")) else None),
            request_deadline=(
                float(d) if (d := os.getenv("REQUEST_DEADLINE_SECONDS")) else None
            ),
            compose_timeout=float(
                os.getenv("COMPOSE_TIMEOUT_SECONDS", str(COMPOSE_TIMEOUT_SECONDS))
            ),
        )

    def auth_policy(self) -> AuthPolicy:
        """The gate's configuration, in one object rather than four arguments
        threaded through every call site — adding `iss` should not mean editing
        api.py."""
        return AuthPolicy(
            secret=self.jwt_secret,
            jwks_url=self.jwks_url,
            audience=self.jwt_audience,
            issuer=self.jwt_issuer,
            leeway=self.jwt_leeway,
        )

"""The assistant itself — one pipeline, two delivery modes.

`Assistant` owns the request pipeline: screen the input, remember it, retrieve
and re-screen context, run the tool loop under a trace, then compose. `ask()`
returns the answer in one piece; `ask_stream()` yields it as SSE frames. Both
build on the same `_gather` so policy can never diverge between the two paths.

Every stage of that pipeline opens a child span under one `assistant.pipeline`
root (observe.py). This is not decoration. "The P99 is 4 seconds" is not a
finding, it is the start of an argument; "the P99 is 4 seconds and 3.6 of them
are in `llm.compose`" ends it. The stages also carry the attributes an operator
needs and a log line cannot give you: which model and prompt version answered,
how many documents came back and how many survived screening, what the exchange
cost, and whether the guardrail refused.

Everything here is deterministic and offline — the adapters that make it real
are chosen in service.py, and the HTTP surface lives in api.py.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from assistant import audit_log, auth, composers, deadline, guardrails, observe
from assistant.agent import Step, run
from assistant.approvals import ApprovalStore, args_fingerprint
from assistant.audit_log import AuditLog
from assistant.composers import (
    Composer,
    StreamComposer,
    citations_for,
    offline_compose,
    word_stream,
)
from assistant.guardrails import Screen
from assistant.idempotency import IdempotencyStore
from assistant.observe import stage, traced_registry, traced_run
from assistant.outbox import Outbox, recorded_registry
from assistant.output_gate import REDACTION, gated_chunks
from assistant.planner import registry_brain
from assistant.provenance import corpus_version, prompt_version
from assistant.rag import sources, texts
from assistant.screening import harden_registry, screen_chunks
from assistant.settings import Settings
from assistant.tools import Tool
from assistant.usage import Usage
from assistant.usage import measure as measure_usage

#: which memory kinds are worth putting in front of an answer. Working memory is
#: this turn's scratch space and procedural memory belongs to the planner, not
#: the prose.
RECALLED_KINDS = ("semantic", "procedural", "episodic")


def sse(event: str, data: dict) -> str:
    """One server-sent-events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def model_name(assistant: Any) -> str:
    """Which brain answered, as a span attribute and a report stamp. "offline"
    is a model too — a number produced by the stitcher and a number produced by
    a 9B model are not comparable, and the trace should say which one it was."""
    from assistant.providers import model_tag

    return model_tag(assistant.settings)


def _brain_tier(settings: Settings) -> str:
    """What /health calls the composing tier. "rule-based" rather than "offline"
    for the no-model path, because that is the name the probes and the e2e suite
    have always used for it."""
    from assistant.providers import OFFLINE, chat_provider

    provider = chat_provider(settings)
    return "rule-based" if provider == OFFLINE else provider


def _auth_tier(settings: Settings) -> str:
    """What /health says about the gate. "jwt" covered both a shared secret and
    a real issuer, which is the difference an operator most needs to see: with
    HS256 every verifier can also mint, with JWKS none of them can."""
    if settings.jwks_url:
        return "jwks"
    return "shared-secret" if settings.jwt_secret else "off"


@dataclass
class Assistant:
    settings: Settings
    rag: Any
    memory: Any
    base_registry: dict[str, Tool]
    rec: Any
    compose: Composer = offline_compose
    stream_compose: StreamComposer | None = None
    # an approval GRANTS ONE RUN of ONE EXACT CALL by ONE SUBJECT — see
    # approvals.py for why a name -> bool flag is four separate holes. The
    # idempotency store makes a retried /approve a no-op instead of a second grant.
    approvals: ApprovalStore = field(default_factory=ApprovalStore)
    idempotency: IdempotencyStore = field(default_factory=IdempotencyStore)
    # every irreversible call is written down BEFORE it happens, so a crash
    # mid-send leaves a question instead of silence (outbox.py)
    outbox: Outbox = field(default_factory=Outbox)
    # component -> reason, filled in when a real adapter fails and the offline
    # default takes over; /health turns non-empty into status "degraded"
    degraded: dict[str, str] = field(default_factory=dict)
    # the persistent who-did-what trail: policy decisions, tool runs, approvals
    audit_log: AuditLog = field(default_factory=AuditLog)
    # the L1 screen applied to EVERY untrusted string — the question, retrieved
    # documents, tool output, ingested docs. Injected rather than imported so
    # `ASSISTANT_GUARD_MODEL` can harden all four channels at once (guard.py);
    # a filter that only covers the one the user types into is the one an
    # attacker will not use.
    screen: Screen = guardrails.screen

    def tier(self) -> dict[str, str | float | int]:
        s = self.settings
        return {
            "rag": "qdrant" if s.qdrant_url else "in-memory",
            # Which embedder is actually behind the vector store. "hash" is
            # deterministic, offline and NOT semantic — it matches on shared
            # vocabulary, so a question about "reimbursements" finds nothing
            # about "refunds". A stack running on it looks identical from every
            # other probe, which is exactly why this one exists.
            # The embedder that is RUNNING. It takes a vector store to run one at
            # all, and a named embedder that could not be built is now a boot
            # error (providers.build_embedder) rather than a silent downgrade —
            # so a model named alongside a store is the one in use. The tag
            # rather than the provider-qualified signature: an operator reading
            # this is asking which embedder, not whose.
            "embed": (s.embed_model if (s.embed_model and s.qdrant_url) else "hash (not semantic)"),
            # dense-only or dense+sparse fused. The second is what phase 2
            # teaches; the deployed stack shipped the first for a while.
            "retrieval": "hybrid-rrf" if s.qdrant_url else "bm25",
            # The floor that lets retrieval say "nothing here", and the reason it
            # is on /health rather than only in the environment: whether the store
            # can abstain is the difference between an assistant that admits
            # ignorance and one that grounds an answer in its nearest neighbours.
            # "inherent" for BM25, which abstains by scoring zero and needs no
            # number; a float for a vector store; "none" for a vector store that
            # was never given one, which is a misconfiguration and reads like one.
            "threshold": ((s.min_score or "none") if s.qdrant_url else "inherent"),
            # The reranker that is RUNNING, not the one that was asked for —
            # same rule as the embed row above, and it fails the same two ways.
            # A model named without a vector store is never built (reranking a
            # BM25 top-3 is not the stage), and a name fastembed cannot load
            # reports a degradation. Both used to read here as configured, which
            # tells an operator a precision stage is on a request path it is not.
            "rerank": (
                s.rerank_model
                if (s.rerank_model and s.qdrant_url and not self.degraded.get("rerank"))
                else "off"
            ),
            "memory": "sqlite" if s.assistant_db else "in-process",
            # Which provider is answering, by name. Not "a model is configured":
            # the bill, the latency and the data-residency question all differ by
            # vendor, and this is the row an operator reads to find out which one
            # they are actually on.
            "brain": _brain_tier(s),
            "tools": "mcp+builtin" if s.mcp_server else "builtin",
            # How many discovered tools the operator has ungated. A number an
            # auditor can read from outside: "0" means every tool that arrived
            # by discovery still pauses for a human.
            "mcp_ungated": len(s.mcp_readonly_allowlist),
            "otlp": "on" if s.otlp_endpoint else "in-memory-only",
            "auth": _auth_tier(s),
            # an operator needs to see from outside whether the screen in front
            # of them is regex-only or regex plus a model
            "guard": s.guard_model if (s.guard_model and s.ollama_host) else "regex-only",
            "connectors": ("real" if (s.telegram_bot_token or s.news_feed_url) else "stubs"),
            # surfaced so an operator can see from OUTSIDE the process whether the
            # outbound gate is screening before release or after it
            "stream": s.stream_mode,
        }

    def warm(self) -> tuple[bool, str]:
        """Can the model tier answer inside the budget *right now*?

        A pulled model is a file on disk. The first request after boot pays the
        load — for a 9B on CPU that is minutes, against a 60-second composer
        budget — so it times out, the offline composer answers, and the caller
        gets a degraded response from a stack that reported itself healthy.
        That is not a slow start, it is a wrong readiness signal: the container
        said ready while the first valid request was expected to miss.

        Nothing to prove without a model tier, so the rule-based tier is
        trivially ready. With one, the only honest answer comes from actually
        completing something.

        The probe's evidence has to ANSWER the probe's question. Composition
        abstains deterministically when nothing retrieved bears on what was
        asked — no model call, no load, no timeout — so a mismatched pair here
        returns a cheerful string from an unreachable host and reports a cold
        stack as ready. The two lines below share the words "service" and
        "ready" for that reason, and not as an accident of phrasing.
        """
        if _brain_tier(self.settings) == "rule-based":
            return True, "rule-based tier needs no warmup"
        try:
            answer = self.compose(
                "is the service ready", ["the service is ready to answer"], [], []
            )
        except Exception as exc:  # noqa: BLE001 — any failure means "not yet"
            return False, f"model tier not answering: {exc}"
        if self.degraded.get("brain"):
            return False, f"model tier degraded: {self.degraded['brain']}"
        return bool(answer), "model answered inside the composer budget"

    def recall(self, question: str, subject: str) -> list[str]:
        """What this assistant already knows about THIS caller, most relevant
        first. Scoped to the subject's own store, so a recalled preference can
        never be someone else's."""
        hits = []
        for kind in RECALLED_KINDS:
            hits.extend(self.memory.recall(kind, question, subject=subject, k=2))
        hits.sort(key=lambda row: -row.score)
        return [row.text for row in hits[:3]]

    def _root_attributes(self, subject: str, request_id: str) -> dict[str, Any]:
        """What the whole request is, set once at the top.

        Which model, which prompt version, which caller, which price tier. These
        are the fields an operator groups by: "the P99 went up on Tuesday" is
        unanswerable until you can split it by model and prompt version, and the
        split has to already be in the data — you cannot go back and add it to
        last Tuesday."""
        return {
            observe.REQUEST_ID: request_id,
            observe.SUBJECT: subject,
            observe.MODEL_NAME: model_name(self),
            observe.PROMPT_VERSION: prompt_version(),
            observe.PRICE_TIER: self.settings.price_tier,
        }

    def _usage_attributes(self, used: Usage, tier: str) -> dict[str, Any]:
        """Tokens and money for one exchange. They go on the compose span rather
        than only on the root because a request can compose more than once:
        summing children is arithmetic, splitting a total back apart is
        guesswork."""
        return {
            observe.TOKENS_IN: used.tokens_in,
            observe.TOKENS_OUT: used.tokens_out,
            observe.TOKENS_TOTAL: used.total,
            observe.TOKENS_SOURCE: used.source,
            observe.COST: used.cost(tier),
        }

    @contextmanager
    def pipeline(self, subject: str) -> Iterator[Any]:
        """The root span for one non-streaming answer.

        Opened by `ask()`, so a request driven from a test or from `report.py`
        has the same shape as one driven over HTTP — with the HTTP root above it
        when there is one (api.py). `ask_stream()` needs a root that survives a
        `yield` and builds its own; see `observe.streaming_root`.
        """
        with (
            observe.request_scope() as rid,
            stage(
                self.rec.tracer,
                observe.PIPELINE_SPAN,
                **self._root_attributes(subject, rid),
            ) as span,
        ):
            yield span

    def _composed(self, gathered: dict, span: Any) -> str:
        """Compose under its own span, and meter the exchange onto it."""
        goal, contexts = gathered["goal"], gathered["contexts"]
        memories, state = gathered["memories"], gathered["state"]
        with stage(self.rec.tracer, observe.COMPOSE_SPAN) as compose_span:
            answer = self.compose(goal, contexts, state, memories)
            used = measure_usage(composers.grounded_prompt(goal, contexts, state, memories), answer)
            tier = self.settings.price_tier
            for key, value in self._usage_attributes(used, tier).items():
                compose_span.set_attribute(key, value)
        span.set_attribute(observe.TOKENS_TOTAL, used.total)
        span.set_attribute(observe.COST, used.cost(tier))
        return answer

    def ingest(self, docs: list, subject: str = auth.ANONYMOUS) -> dict:
        """Screen documents BEFORE they land in the corpus. Returns counts.

        A document may arrive as prose or as `{"text": ..., "source": ...}`. The
        source is worth naming: it is what a citation points at, what
        `forget(source)` deletes, and what makes a re-ingest an update instead of
        a fourth copy. An unnamed document gets a source derived from its own
        content, so even the lazy path deduplicates.

        Retrieval-time screening (`screen_contexts`) already drops a poisoned
        document on its way to the composer, so this looks redundant. It is not,
        for three reasons.

        A payload that is only caught at retrieval is *stored*: it sits in the
        index, it comes back in every search that ranks it, and it is one
        detector regression away from being evidence. Storing it also means the
        blast radius of a later bug is the whole corpus rather than one request.

        PII is the second reason, and it is the sharper one. Redacting at
        retrieval keeps the raw SSN on disk forever; redacting at ingest means
        it was never written down. That is data minimisation, and it is the
        difference between a breach that exposes what you needed and one that
        exposes what you happened to keep.

        Third: the caller finds out. An `/ingest` that silently drops half a
        batch and reports success is how a corpus quietly ends up incomplete.

        None of which retires `screen_contexts` — documents can arrive by paths
        that never touch this method, and a detector that improves tomorrow has
        to be applied to what was written yesterday. Two gates, one screen.
        """
        with stage(
            self.rec.tracer,
            observe.INGEST_SPAN,
            **{observe.SUBJECT: subject, observe.TENANT: subject},
        ) as span:
            kept, rejected = [], 0
            for doc in docs:
                text = doc["text"] if isinstance(doc, dict) else str(doc)
                ok, cleaned = self.screen(text)
                if ok:
                    # the screen may have redacted; the SOURCE is preserved, so
                    # a redacted document is still attributable and deletable
                    kept.append(
                        {"text": cleaned, "source": doc.get("source")}
                        if isinstance(doc, dict)
                        else cleaned
                    )
                else:
                    rejected += 1
                    self.audit_log.record(
                        "ingest.rejected",
                        subject,
                        cleaned,
                        args={"source": doc.get("source")} if isinstance(doc, dict) else None,
                        result=audit_log.REJECTED,
                    )
            added = self.rag.add(kept, tenant=subject) if kept else 0
            if added:
                # Deletion was audited from the start and addition was not, which
                # is backwards: a document that lands in the corpus changes the
                # answers every caller in the tenant gets from then on. When one
                # of those answers turns out to be wrong, the question is who put
                # the source there and when — and a screen that did not object is
                # not the same as a document nobody needs to account for.
                self.audit_log.record(
                    "corpus.ingested",
                    subject,
                    f"{added} chunks from {', '.join(sources(kept)) or 'unnamed documents'}",
                    args={"sources": sources(kept)},
                    result=audit_log.OK,
                )
            span.set_attribute(observe.INGEST_ACCEPTED, added)
            span.set_attribute(observe.INGEST_REJECTED, rejected)
            span.set_attribute(
                observe.CORPUS_VERSION,
                corpus_version(d["text"] if isinstance(d, dict) else d for d in kept),
            )
            # `ingested` counts CHUNKS, not documents: one long page is several
            # retrievable units, and the number that matters downstream is the
            # number of things that can now come back from a search.
            return {"ingested": added, "rejected": rejected}

    def forget(self, source: str, subject: str = auth.ANONYMOUS) -> dict:
        """Delete every chunk of one source from this caller's corpus.

        A retrieval layer with no delete is a retrieval layer that eventually
        holds something you are not allowed to keep — a document withdrawn, a
        page that turned out to be poisoned, a customer exercising erasure. The
        audit row is part of the answer: "we deleted it" is a claim, and claims
        need evidence."""
        with stage(
            self.rec.tracer,
            observe.INGEST_SPAN,
            **{observe.SUBJECT: subject, observe.TENANT: subject},
        ) as span:
            removed = self.rag.delete(source, tenant=subject)
            self.audit_log.record(
                "corpus.deleted",
                subject,
                f"{source} ({removed} chunks)",
                args={"source": source},
                result=audit_log.DELETED,
            )
            span.set_attribute(observe.INGEST_DELETED, removed)
            return {"deleted": removed, "source": source}

    def evidence(self, chunk_id: str, subject: str = auth.ANONYMOUS) -> dict | None:
        """Resolve a citation back to the chunk it names.

        This is what makes a citation *durable* rather than decorative: the id in
        an answer written last month still names a specific slice of a specific
        revision of a specific document, and this is the lookup that proves it —
        or proves it is gone, which is also an answer."""
        chunk = self.rag.get(chunk_id, tenant=subject)
        return chunk.cite("c1") | {"text": chunk.text} if chunk else None

    def _gather(self, question: str, subject: str = auth.ANONYMOUS) -> dict:
        """Everything before composition, shared by ask() and ask_stream(): screen
        the input, remember it and recall what we already know about this caller,
        retrieve context from the caller's tenant only, screen what came back, and
        run the tool loop with a capturing composer so the evidence comes back
        instead of a finished answer.

        `deadline.check()` sits between the stages rather than inside them.
        Python cannot safely interrupt work in flight, so the honest thing a
        pipeline can do is refuse to start the NEXT stage — which turns an
        unbounded request into a bounded one and unwinds from a point where
        nothing is half-applied. It also catches the caller who disconnected
        while this request was queued behind the concurrency cap: retrieval and
        a model call for an answer nobody will read are the most expensive
        possible way to be polite."""
        tracer = self.rec.tracer
        deadline.check()
        with stage(tracer, observe.SCREEN_SPAN) as span:
            ok, cleaned = self.screen(question)
            span.set_attribute(observe.SCREEN_BLOCKED, not ok)
            if not ok:
                span.set_attribute(observe.SCREEN_REASON, cleaned)
        if not ok:
            self.audit_log.record("policy.blocked", subject, cleaned, result=audit_log.BLOCKED)
            return {"blocked": cleaned}

        deadline.check()
        with stage(tracer, observe.MEMORY_SPAN, **{observe.SUBJECT: subject}) as span:
            self.memory.remember(cleaned, source=f"user:{subject}", subject=subject)
            # recall AFTER remembering, so a fact stated this turn is usable now
            memories = self.recall(cleaned, subject)
            span.set_attribute(observe.MEMORY_RECALLED, len(memories))

        deadline.check()
        with stage(
            tracer,
            observe.RETRIEVAL_SPAN,
            **{observe.TENANT: subject, observe.RETRIEVAL_K: 3},
        ) as span:
            hits = self.rag.search(cleaned, k=3, tenant=subject)
            chunks = screen_chunks(hits, self.screen)
            # the composer and the planner see text; the CITATIONS keep the
            # provenance, so an answer can be checked against the document it
            # came from rather than against a paraphrase of a fragment
            contexts = texts(chunks)
            span.set_attribute(observe.RETRIEVAL_HITS, len(hits))
            # kept < hits means the screen dropped a poisoned document. That gap
            # is a security signal, and it is invisible unless both are recorded.
            span.set_attribute(observe.RETRIEVAL_KEPT, len(chunks))
            span.set_attribute(observe.CORPUS_VERSION, corpus_version(contexts))
            span.set_attribute(observe.RETRIEVAL_SOURCES, sorted({c.source for c in chunks}))
        captured: dict = {}

        def capture(goal: str, ctxs: list[str], state: list[tuple[Step, Any]]) -> str:
            captured["state"] = list(state)
            return ""

        # Three wrappers, outside-in, and the order is the design. Screening is
        # innermost so it sees raw tool output; the outbox sits outside it so the
        # intent is durable before anything runs; tracing is outermost so a span
        # covers the whole thing including the write.
        registry = traced_registry(
            recorded_registry(
                harden_registry(self.base_registry, self.screen),
                self.outbox,
                subject,
                observe.request_id(),
            ),
            self.rec.tracer,
        )

        def consume(tool_name: str, args: dict) -> str | None:
            """Spend one grant belonging to THIS caller for THIS exact call."""
            return self.approvals.consume(subject, tool_name, args)

        deadline.check()
        result = traced_run(
            run,
            cleaned,
            self.rec.tracer,
            decide=registry_brain(contexts, registry, capture),
            consume=consume,
            registry=registry,
        )
        for call in result.calls:
            approval_id = result.approval_ids.get(call.tool, "")
            detail = f"{call.tool} (approval {approval_id})" if approval_id else call.tool
            self.audit_log.record(
                "tool.ran",
                subject,
                detail,
                approval_id=approval_id,
                args=call.args,
            )
        if result.pending is not None:
            self.audit_log.record(
                "tool.pending",
                subject,
                result.pending.tool,
                args=result.pending.args,
                result=audit_log.PENDING,
            )
            return {
                "pending": result.pending,
                "contexts": contexts,
                "chunks": chunks,
                "memories": memories,
                "audit": result.audit,
            }
        return {
            "goal": cleaned,
            "contexts": contexts,
            "chunks": chunks,
            "memories": memories,
            "state": captured.get("state", []),
            "audit": result.audit,
        }

    def ask(self, question: str, subject: str = auth.ANONYMOUS) -> dict:
        with self.pipeline(subject) as span:
            gathered = self._gather(question, subject)
            request_id = observe.request_id()
            if "blocked" in gathered:
                return {
                    "answer": "I can't help with that request.",
                    "blocked": gathered["blocked"],
                    "request_id": request_id,
                    "contexts": [],
                    "citations": [],
                    "memories": [],
                    "grounding": "none",
                    "audit": [],
                }
            contexts = gathered["contexts"]
            memories = gathered["memories"]
            citations = citations_for(gathered["chunks"])
            if "pending" in gathered:
                pending = gathered["pending"]
                return {
                    "answer": f"'{pending.tool}' needs approval before it can run.",
                    # args and their fingerprint travel with the pause: approving is
                    # approving THIS call, so the client has to echo it back
                    "pending": {
                        "tool": pending.tool,
                        "args": pending.args,
                        "args_hash": args_fingerprint(pending.args),
                    },
                    "contexts": contexts,
                    "citations": citations,
                    "request_id": request_id,
                    "grounding": "none",
                    "memories": memories,
                    "audit": gathered["audit"],
                }
            grounding = composers.grounding_of(
                contexts, gathered["state"], memories, gathered["goal"]
            )
            # Citations belong to the evidence the answer stands on. A
            # memory-grounded answer stands on what the caller said, so it ships
            # none — the alternative is "you told me you are in Lima [c1]",
            # pointing at a refund policy, which is a fabricated source rather
            # than a generous one (ADR-0008).
            if grounding == "memory":
                citations = []
            answer = self._composed(gathered, span)
            with stage(self.rec.tracer, observe.OUTPUT_SPAN) as gate_span:
                released = guardrails.output_ok(answer)
                gate_span.set_attribute(observe.SCREEN_BLOCKED, not released)
            if not released:
                # The same string the streaming gate substitutes, not a copy of it:
                # the release lane scores a control as refused when its answer IS
                # this text, and a check bound to a duplicated literal stops being
                # true the first time one of the two is edited.
                answer = REDACTION
            return {
                "answer": answer,
                "contexts": contexts,
                "citations": citations,
                "request_id": request_id,
                # Which class of evidence this answer stands on. A caller can
                # verify a "documents" answer against its citations; a
                # "memory" one has none by design, and saying so is the
                # difference between personalisation and a silent downgrade.
                "grounding": grounding,
                "memories": memories,
                "audit": gathered["audit"],
            }

    def ask_stream(self, question: str, subject: str = auth.ANONYMOUS) -> Iterator[str]:
        """The same pipeline as ask(), as SSE frames: `chunk` events carrying text
        the output gate has already cleared, then one `done` event with citations
        and the audit trail.

        The gating lives in output_gate.py, which holds a window back so nothing
        reaches the client before it has been screened — a streamed answer and a
        batch answer are subject to the same gate, not to a gate and an apology.

        A stream that dies mid-answer still ends in `done`, because it is over,
        but carries `truncated: true` and the degradation that caused it. The
        text is real and already delivered; what it is not is the answer.
        """
        tracer = self.rec.tracer
        request_id = observe.request_id() or observe.new_request_id()
        with observe.streaming_root(
            tracer, observe.PIPELINE_SPAN, **self._root_attributes(subject, request_id)
        ) as root:
            with observe.within(root):
                gathered = self._gather(question, subject)
            if "blocked" in gathered:
                yield sse(
                    "done",
                    {
                        "answer": "I can't help with that request.",
                        "blocked": gathered["blocked"],
                        "request_id": request_id,
                        "citations": [],
                        "grounding": "none",
                        "audit": [],
                    },
                )
                return
            citations = citations_for(gathered["chunks"])
            if "pending" in gathered:
                pending = gathered["pending"]
                yield sse(
                    "done",
                    {
                        "answer": f"'{pending.tool}' needs approval before it can run.",
                        "pending": {
                            "tool": pending.tool,
                            "args": pending.args,
                            "args_hash": args_fingerprint(pending.args),
                        },
                        "citations": citations,
                        "request_id": request_id,
                        "grounding": "none",
                        "audit": gathered["audit"],
                    },
                )
                return
            grounding = composers.grounding_of(
                gathered["contexts"],
                gathered["state"],
                gathered["memories"],
                gathered["goal"],
            )
            # Same rule as the batch path, and it has to be the same rule: a
            # client that gets citations when it streams and none when it does
            # not has learned that the difference is the transport.
            if grounding == "memory":
                citations = []
            stream = self.stream_compose or word_stream(self.compose)
            produced = stream(
                gathered["goal"], gathered["contexts"], gathered["state"], gathered["memories"]
            )
            # Nothing above materialises the stream: `gated_chunks` pulls one
            # chunk at a time and this loop hands each frame straight to the
            # client. Buffering here to make the metering tidier would trade the
            # feature for the instrumentation, which is the wrong way round.
            began, text = time.time_ns(), []
            # The one string this turn is metered from, and there is exactly one
            # because there used to be two. A terminal frame carries the whole
            # answer rather than a delta, so appending it to `text` alongside the
            # chunks it repeats metered every completed *and* every truncated
            # stream twice: 2x output tokens and 2x cost, on the span this course
            # tells students to bill from. Assigned rather than accumulated, so a
            # future branch cannot add a third contributor by accident.
            #
            # The rule: `metered` is the answer the client was given. The one
            # exception is a blocked stream, whose answer is a fixed redaction
            # string — metering that would price the constant, so the released
            # prefix stands in. It undercounts a model that generated a page and
            # had all of it withheld; the gate does not hand the withheld text
            # back, and inventing a number for it would be worse.
            metered: str | None = None
            blocked = truncated = False
            for kind, chunk in gated_chunks(
                produced, self.settings.stream_mode, check=deadline.check
            ):
                # Checked here per released frame AND inside the gate before each
                # source chunk, because those are different moments: the gate can
                # hold an unbroken token indefinitely without releasing anything,
                # and a check that only runs between frames never runs at all
                # through that. A client that closed the tab thirty seconds ago is
                # still being generated for otherwise — and under load that is the
                # failure that compounds, because the tokens nobody reads are the
                # tokens that slow down the callers who stayed.
                if deadline.expired():
                    # Ending the loop is right; ending it in silence was not.
                    # The client has already rendered half an answer, and a
                    # stream that simply stops is indistinguishable from a
                    # dropped connection — so the text stays on screen looking
                    # finished. Same terminal contract as a mid-stream stall: a
                    # `done` frame that says it is not the whole answer.
                    why = deadline.expired() or "the request budget ran out"
                    root.set_attribute(observe.ABANDONED, why)
                    truncated = True
                    metered = "".join(text)
                    yield sse(
                        "done",
                        {
                            "answer": metered,
                            "truncated": True,
                            "abandoned": why,
                            "degraded": dict(self.degraded),
                            "citations": citations,
                            "request_id": request_id,
                            "grounding": grounding,
                            "audit": gathered["audit"],
                        },
                    )
                    break
                if kind == "chunk":
                    text.append(chunk)
                    yield sse("chunk", {"text": chunk})
                elif kind == "blocked":
                    blocked = True
                    metered = "".join(text)
                    yield sse(
                        "done",
                        {
                            "answer": chunk,
                            "redacted": True,
                            "request_id": request_id,
                            "citations": citations,
                            "audit": gathered["audit"],
                        },
                    )
                elif kind == "truncated":
                    # Still a `done` frame — the stream really is over — but one
                    # that says so. A client rendering this as a finished answer
                    # is now doing it against an explicit flag, not by omission.
                    truncated = True
                    metered = chunk
                    yield sse(
                        "done",
                        {
                            "answer": chunk,
                            "truncated": True,
                            "degraded": dict(self.degraded),
                            "citations": citations,
                            "request_id": request_id,
                            "grounding": grounding,
                            "audit": gathered["audit"],
                        },
                    )
                else:
                    metered = chunk
                    yield sse(
                        "done",
                        {
                            "answer": chunk,
                            "citations": citations,
                            "request_id": request_id,
                            "grounding": grounding,
                            "audit": gathered["audit"],
                        },
                    )
            # Closed after the last frame, backdated to when generation started,
            # so this span measures the generation and not the accounting.
            used = measure_usage(
                composers.grounded_prompt(
                    gathered["goal"], gathered["contexts"], gathered["state"], gathered["memories"]
                ),
                # `is None` rather than `or`: an empty answer is a real answer and
                # has to meter as zero output tokens, not fall through to the
                # released chunks.
                "".join(text) if metered is None else metered,
            )
            tier = self.settings.price_tier
            observe.mark(
                tracer,
                observe.COMPOSE_SPAN,
                root,
                start_time=began,
                **self._usage_attributes(used, tier),
            )
            observe.mark(tracer, observe.OUTPUT_SPAN, root, **{observe.SCREEN_BLOCKED: blocked})
            if truncated:
                root.set_attribute(observe.TRUNCATED, True)
            root.set_attribute(observe.TOKENS_TOTAL, used.total)
            root.set_attribute(observe.COST, used.cost(tier))

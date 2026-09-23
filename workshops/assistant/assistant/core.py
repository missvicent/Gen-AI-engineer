"""The assistant itself — one pipeline, two delivery modes.

`Assistant` owns the request pipeline: screen the input, remember it, retrieve
and re-screen context, run the tool loop under a trace, then compose. `ask()`
returns the answer in one piece; `ask_stream()` yields it as SSE frames. Both
build on the same `_gather` so policy can never diverge between the two paths.

Every stage of that pipeline opens a child span (observe.py). This is not
decoration. "The P99 is 4 seconds" is not a finding, it is the start of an
argument; "the P99 is 4 seconds and 3.6 of them are in `llm.compose`" ends it.

Your TODOs here are recall (`Assistant.recall`, TODO 2), ingest-time screening
(`Assistant.ingest`, TODO 3), the shared pipeline (`Assistant._gather`, TODO 4),
the observability seams (TODO 5-8) and the abandoned stream (TODO 9); ask() and
ask_stream() are given and build on all of them. Tool SELECTION is no longer one
of them — it moved to planner.py, where it can read the registry instead of a
hardcoded list of names. Keep everything offline and deterministic — the adapters
that make it real are chosen in service.py, and the HTTP surface lives in api.py.

`forget()` and `evidence()` are given, and they are short because the hard part
is one layer down (adapters.py, TODO 4-7). They are here to make the point that
a retrieval layer is not finished when search works: a corpus you cannot delete
from will eventually hold something you are not allowed to keep, and a citation
that cannot be resolved back to its text is decoration.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from assistant import audit_log, auth, composers, guardrails, observe
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
from assistant.outbox import Outbox
from assistant.output_gate import REDACTION, gated_chunks
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
            "embed": (
                s.embed_model if (s.embed_model and s.qdrant_url) else "hash (not semantic)"
            ),
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
            "threshold": (
                (s.min_score or "none") if s.qdrant_url else "inherent"
            ),
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
            "connectors": (
                "real" if (s.telegram_bot_token or s.news_feed_url) else "stubs"
            ),
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

        The probe's evidence has to ANSWER the probe's question. Composition
        abstains deterministically when nothing retrieved bears on what was
        asked — no model call, no load, no timeout — so a mismatched pair here
        returns a cheerful string from an unreachable host and reports a cold
        stack as ready. The two lines below share the words "service" and
        "ready" for that reason, and not as an accident of phrasing.

        Nothing to prove without a model tier, so the rule-based tier is
        trivially ready. With one, the only honest answer comes from actually
        completing something.
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
        """TODO 2: what this assistant already knows about THIS caller.

        Pull up to two hits per kind in RECALLED_KINDS from
        `self.memory.recall(kind, question, subject=subject, k=2)`, rank them by
        score across kinds, and return the top three texts.

        The `subject=` is the whole point. The version this replaced asked one
        shared store, which answered with whatever matched — including another
        person's preferences. See tenancy.py.
        """
        raise NotImplementedError

    def _root_attributes(self, subject: str, request_id: str) -> dict[str, Any]:
        """TODO 5: what the whole request is, as attributes for its root span.

        Return REQUEST_ID, SUBJECT, MODEL_NAME (`model_name(self)`),
        PROMPT_VERSION (`prompt_version()`) and PRICE_TIER
        (`self.settings.price_tier`), using the constants in observe.py.

        These are the fields an operator groups by: "the P99 went up on Tuesday"
        is unanswerable until you can split it by model and prompt version, and
        the split has to already be in the data — you cannot go back and add it
        to last Tuesday.
        """
        raise NotImplementedError

    def _usage_attributes(self, used: Usage, tier: str) -> dict[str, Any]:
        """TODO 6: tokens and money for one exchange.

        TOKENS_IN, TOKENS_OUT, TOKENS_TOTAL, TOKENS_SOURCE (`used.source`) and
        COST (`used.cost(tier)`). They go on the compose span rather than only on
        the root because a request can compose more than once: summing children
        is arithmetic, splitting a total back apart is guesswork. TOKENS_SOURCE
        rides along because the cost beside it looks identical whether the tokens
        were counted by the provider or estimated from a word split.
        """
        raise NotImplementedError

    @contextmanager
    def pipeline(self, subject: str) -> Iterator[Any]:
        """TODO 7: the root span for one non-streaming answer.

        Open an `observe.request_scope()` and an `observe.stage(...,
        observe.PIPELINE_SPAN, **self._root_attributes(subject, rid))`, and yield
        the span. Everything `_gather` opens then nests underneath it for free —
        that is what the ambient context is for.

        `ask_stream()` cannot use this: its root has to survive a `yield`, so it
        builds one with `observe.streaming_root`. Read the note there.
        """
        raise NotImplementedError

    def _composed(self, gathered: dict, span: Any) -> str:
        """TODO 8: compose under its own span, and meter the exchange onto it.

        Open `observe.stage(..., observe.COMPOSE_SPAN)`, call `self.compose(goal,
        contexts, state, memories)`, measure with `usage.measure` against
        `composers.grounded_prompt(...)` for the prompt side, and set the
        attributes from `_usage_attributes`. Put TOKENS_TOTAL and COST on the
        root `span` as well, so a request's total is readable without summing
        its children. Return the answer.
        """
        raise NotImplementedError

    def ingest(self, docs: list, subject: str = auth.ANONYMOUS) -> dict:
        """TODO 3: screen documents BEFORE they land in the corpus.

        Run each doc through `self.screen`. Keep the cleaned text of the ones
        that pass and `self.rag.add(kept, tenant=subject)`; count the refusals
        and record each as `("ingest.rejected", subject, reason,
        result=audit_log.REJECTED)` in the audit log.

        Record what was ACCEPTED too — one `("corpus.ingested", subject, ...,
        result=audit_log.OK)` row naming the sources, when anything was added.
        `forget()` below audits its delete and it would be easy to leave the add
        unaudited by symmetry with "nothing was removed": the asymmetry is the
        bug. A document that lands in the corpus changes every answer the tenant
        gets from then on, and when one of those answers turns out to be wrong,
        the question is who put the source there — unanswerable if only removals
        were written down. A screen that did not object is not the same thing as
        a document nobody has to account for.

        Return `{"ingested": added, "rejected": n}` — the caller is entitled to
        know that half its batch did not make it. Wrap the whole thing in an
        `observe.stage(..., observe.INGEST_SPAN)` carrying SUBJECT, TENANT,
        INGEST_ACCEPTED, INGEST_REJECTED and the CORPUS_VERSION of what was
        kept: "how much of that batch was refused" is a question you will be
        asked in an incident, and counting it afterwards is not possible.

        A document arrives as prose OR as `{"text": ..., "source": ...}`, and
        the source has to survive screening. It is what a citation points at and
        what `forget(source)` deletes, so a redacted document that lost its name
        is a document nobody can attribute or withdraw — which is a quiet
        incentive to screen less. Pass the source through with the cleaned text.

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
        raise NotImplementedError

    def forget(self, source: str, subject: str = auth.ANONYMOUS) -> dict:
        """Delete every chunk of one source from this caller's corpus.

        A retrieval layer with no delete is a retrieval layer that eventually
        holds something you are not allowed to keep — a document withdrawn, a
        page that turned out to be poisoned, a customer exercising erasure. The
        audit row is part of the answer: "we deleted it" is a claim, and claims
        need evidence. (The tenant-scoped delete itself is TODO 5 in
        adapters.py; this is only the seam that records it.)"""
        with observe.stage(
            self.rec.tracer, observe.INGEST_SPAN,
            **{observe.SUBJECT: subject, observe.TENANT: subject},
        ) as span:
            removed = self.rag.delete(source, tenant=subject)
            self.audit_log.record(
                "corpus.deleted", subject, f"{source} ({removed} chunks)",
                args={"source": source}, result=audit_log.DELETED,
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
        """TODO 4: everything before composition, shared by ask() and ask_stream().

        Each bullet below is one `observe.stage(...)` child span. Nesting is
        automatic — whatever root is current when `_gather` runs becomes their
        parent — so the whole request reads back as a tree.

        - self.screen() the input under SCREEN_SPAN, recording SCREEN_BLOCKED and
          (when blocked) SCREEN_REASON; if blocked, record ("policy.blocked",
          subject, reason) in self.audit_log and return {"blocked": reason}
        - under MEMORY_SPAN: remember the turn with `self.memory.remember(cleaned,
          source=f"user:{subject}", subject=subject)`, then recall() — in that
          order, so a fact the caller states this turn is usable this turn — and
          record MEMORY_RECALLED
        - under RETRIEVAL_SPAN: retrieve with self.rag.search(cleaned, k=3,
          tenant=subject) — which returns CHUNKS now, not strings — and pass
          them through `screening.screen_chunks(hits, self.screen)`. A poisoned
          document is dropped BEFORE it becomes evidence, and passing self.screen
          is what makes the guard model cover this channel too. Record
          RETRIEVAL_K, RETRIEVAL_HITS *and* RETRIEVAL_KEPT: the gap between the
          last two is how many documents the screen threw away, which is a
          security signal one number cannot show. RETRIEVAL_SOURCES (the sorted
          set of `chunk.source`) is what turns "retrieval was slow" into "the
          slow ones all hit the same document".

          Keep BOTH shapes: `rag.texts(chunks)` for the composer and the planner,
          which only want words, and the chunks themselves under "chunks", which
          is what `citations_for` needs to produce a citation somebody can check.
        - call `deadline.check()` (import it) BETWEEN the stages. Python cannot safely
          interrupt work in flight, so the honest thing a pipeline can do is
          refuse to start the NEXT stage — which turns an unbounded request into
          a bounded one and unwinds from a point where nothing is half-applied.
          It also catches the caller who gave up while an earlier stage ran:
          retrieval and a model call for an answer nobody will read are the most
          expensive possible way to be polite.
        - run agent.run under observe.traced_run with a registry wrapped by
          `screening.harden_registry(self.base_registry, self.screen)` (your
          TODO 1), then `outbox.recorded_registry(..., self.outbox, subject,
          observe.request_id())`, then observe.traced_registry. Three wrappers,
          outside-in, and the order is the design: screening is innermost so it
          sees raw tool output, the outbox sits outside it so the intent is
          durable before anything runs, and tracing is outermost so a span covers
          the whole thing including the write.
          `decide=planner.registry_brain(contexts, registry, capture)` — note it
          takes the WRAPPED registry, so a discovered MCP tool is selectable and
          still screened,
          a `capture` composer that RECORDS its state argument and returns ""
          (the callers compose afterwards), and `consume=` a callable that spends
          one of THIS subject's grants for the exact call:
              lambda name, args: self.approvals.consume(subject, name, args)
          Pass `consume`, not an `approvals` allow-list: the grant has to be taken
          against the arguments actually about to run.
        - audit-log every entry in `result.calls` as ("tool.ran", subject, X),
          passing `approval_id=` and `args=call.args` so the row can be joined
          on rather than read. Iterate `result.calls`, not the "ran: " strings
          in `result.audit` — a caller that has to parse a sentence back apart
          is one formatting change away from recording nothing. Name the
          grant it spent when result.approval_ids has one — the id is what ties
          the /approve response, the log row and the span together; a paused run
          is ("tool.pending", subject, tool)
        - return {"pending": result.pending, "contexts": ..., "chunks": ...,
          "memories": ..., "audit": ...} if the loop paused, else {"goal":
          cleaned, "contexts": ..., "chunks": ..., "memories": ..., "state":
          captured state, "audit": result.audit}
        """
        raise NotImplementedError

    def ask(self, question: str, subject: str = auth.ANONYMOUS) -> dict:
        with self.pipeline(subject) as span:
            gathered = self._gather(question, subject)
            request_id = observe.request_id()
            if "blocked" in gathered:
                return {"answer": "I can't help with that request.",
                        "blocked": gathered["blocked"], "request_id": request_id,
                        "contexts": [], "citations": [], "memories": [],
                        "grounding": "none", "audit": []}
            contexts = gathered["contexts"]
            citations = citations_for(gathered["chunks"])
            memories = gathered["memories"]
            if "pending" in gathered:
                pending = gathered["pending"]
                return {
                    "answer": f"'{pending.tool}' needs approval before it can run.",
                    # args and their fingerprint travel with the pause: approving is
                    # approving THIS call, so the client has to echo it back
                    "pending": {"tool": pending.tool, "args": pending.args,
                                "args_hash": args_fingerprint(pending.args)},
                    "contexts": contexts, "citations": citations,
                    "request_id": request_id, "grounding": "none",
                    "memories": memories, "audit": gathered["audit"],
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
            with observe.stage(self.rec.tracer, observe.OUTPUT_SPAN) as gate_span:
                released = guardrails.output_ok(answer)
                gate_span.set_attribute(observe.SCREEN_BLOCKED, not released)
            if not released:
                # The same string the streaming gate substitutes, not a copy of it:
                # the release lane scores a control as refused when its answer IS
                # this text, and a check bound to a duplicated literal stops being
                # true the first time one of the two is edited.
                answer = REDACTION
            return {"answer": answer, "contexts": contexts, "citations": citations,
                    "request_id": request_id,
                    # Which class of evidence this answer stands on. A caller can
                    # verify a "documents" answer against its citations; a
                    # "memory" one has none by design, and saying so is the
                    # difference between personalisation and a silent downgrade.
                    "grounding": grounding,
                    "memories": memories, "audit": gathered["audit"]}

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

        Given, and worth reading for the tracing shape alone: the root here is a
        `streaming_root` rather than a `stage`, because a span made "current"
        cannot survive the yields below.
        """
        tracer = self.rec.tracer
        request_id = observe.request_id() or observe.new_request_id()
        with observe.streaming_root(
            tracer, observe.PIPELINE_SPAN, **self._root_attributes(subject, request_id)
        ) as root:
            with observe.within(root):
                gathered = self._gather(question, subject)
            if "blocked" in gathered:
                yield sse("done", {
                    "answer": "I can't help with that request.",
                    "blocked": gathered["blocked"], "request_id": request_id,
                    "citations": [], "grounding": "none", "audit": [],
                })
                return
            citations = citations_for(gathered["chunks"])
            if "pending" in gathered:
                pending = gathered["pending"]
                yield sse("done", {
                    "answer": f"'{pending.tool}' needs approval before it can run.",
                    "pending": {"tool": pending.tool, "args": pending.args,
                                "args_hash": args_fingerprint(pending.args)},
                    "citations": citations, "request_id": request_id,
                    "grounding": "none", "audit": gathered["audit"],
                })
                return
            grounding = composers.grounding_of(
                gathered["contexts"], gathered["state"],
                gathered["memories"], gathered["goal"],
            )
            # Same rule as the batch path, and it has to be the same rule: a
            # client that gets citations when it streams and none when it does
            # not has learned that the difference is the transport.
            if grounding == "memory":
                citations = []
            stream = self.stream_compose or word_stream(self.compose)
            produced = stream(gathered["goal"], gathered["contexts"],
                              gathered["state"], gathered["memories"])
            # Nothing above materialises the stream: `gated_chunks` pulls one
            # chunk at a time and this loop hands each frame straight to the
            # client. Buffering here to make the metering tidier would trade the
            # feature for the instrumentation, which is the wrong way round.
            began, text = time.time_ns(), []
            # The one string this turn is metered from. `gated_chunks` ends with the
            # whole answer rather than a delta, so appending that terminal payload
            # to the list that already holds every released chunk counts the answer
            # twice — 2x output tokens and 2x cost, on the span you are about to be
            # asked to bill from. `metered` is assigned, never accumulated, which is
            # what makes a second contributor impossible instead of merely unlikely.
            #
            # The rule: `metered` is the answer the client was given. A blocked
            # stream is the exception — its answer is a fixed redaction string, and
            # pricing a constant is meaningless, so the released prefix stands in.
            metered: str | None = None
            blocked = truncated = False
            for kind, chunk in gated_chunks(produced, self.settings.stream_mode):
                # TODO 9: stop here when `deadline.expired()` (import it), recording the
                # reason on the root span as observe.ABANDONED — and yield a final
                # `done` frame carrying the text so far, `truncated: true` and the
                # reason, before you break.
                #
                # Then pass `check=deadline.check` to `gated_chunks` above as well.
                # Both, because they are different moments, and one of them is not
                # optional: this loop only runs when the gate RELEASES something,
                # and the gate holds an unbroken token without releasing anything.
                # A check that runs between frames therefore runs zero times during
                # exactly the stream that most needs stopping — measured at 9.2
                # seconds of buffering and re-screening for 800 chunks nobody was
                # ever going to read.
                #
                # A client that closed the tab thirty seconds ago is still being
                # generated for otherwise — and under load that is the failure that
                # compounds, because the tokens nobody reads are the tokens that
                # slow down the callers who stayed.
                #
                # Breaking in silence is the trap. The client has half an answer
                # rendered, and a stream that simply stops looks exactly like a
                # dropped connection — so the half stays on screen reading as the
                # whole. Every way this stream can end has to end in `done`.
                #
                # Set `metered` to the text you put in that frame, for the same
                # reason every other branch below sets it: one canonical string per
                # stream, whichever frame ends it.
                if kind == "chunk":
                    text.append(chunk)
                    yield sse("chunk", {"text": chunk})
                elif kind == "blocked":
                    blocked = True
                    metered = "".join(text)
                    yield sse("done", {
                        "answer": chunk, "redacted": True, "request_id": request_id,
                        "citations": citations, "audit": gathered["audit"],
                    })
                elif kind == "truncated":
                    # Still a `done` frame — the stream really is over — but one
                    # that says so. A client rendering this as a finished answer
                    # is now doing it against an explicit flag, not by omission.
                    truncated = True
                    metered = chunk
                    yield sse("done", {
                        "answer": chunk, "truncated": True,
                        "degraded": dict(self.degraded), "citations": citations,
                        "request_id": request_id, "grounding": grounding,
                        "audit": gathered["audit"],
                    })
                else:
                    metered = chunk
                    yield sse("done", {"answer": chunk, "citations": citations,
                                       "request_id": request_id,
                                       "grounding": grounding,
                                       "audit": gathered["audit"]})
            # Closed after the last frame, backdated to when generation started,
            # so this span measures the generation and not the accounting.
            used = measure_usage(
                composers.grounded_prompt(gathered["goal"], gathered["contexts"],
                                          gathered["state"], gathered["memories"]),
                # `is None` rather than `or`: an empty answer is a real answer and
                # has to meter as zero output tokens, not fall through to the
                # released chunks.
                "".join(text) if metered is None else metered,
            )
            tier = self.settings.price_tier
            observe.mark(tracer, observe.COMPOSE_SPAN, root, start_time=began,
                         **self._usage_attributes(used, tier))
            observe.mark(tracer, observe.OUTPUT_SPAN, root,
                         **{observe.SCREEN_BLOCKED: blocked})
            if truncated:
                root.set_attribute(observe.TRUNCATED, True)
            root.set_attribute(observe.TOKENS_TOTAL, used.total)
            root.set_attribute(observe.COST, used.cost(tier))

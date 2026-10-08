"""Workshop 8 layer — OpenTelemetry around the agent loop and every tool.

An agent is harder to observe than a single model call, because the interesting
question is never "how long did that take". It is *what did it do* — which tools,
in what order, how many steps, and where the eight seconds actually went. A single
timer on `run()` cannot answer any of that; a span tree can.

Three design choices worth copying:

**Tools are instrumented by wrapping the registry, not by editing tools.** No tool
knows it is being traced, which means adding a tool cannot forget to. Cross-cutting
concerns belong at the seam, not sprinkled through every implementation.

**One span per stage, under one root.** A single `agent.run` span answers "how
long" and nothing else. When a request is slow, the question is *which stage* —
the screen, retrieval, memory, the model, a tool, the output gate — and a flat
span cannot say. The tree here is deliberate: `assistant.request` at the HTTP
door, `auth.verify` and `assistant.pipeline` beneath it, and one child per stage
beneath that. A P99 you cannot decompose is a number you cannot act on.

**The vendor is an environment variable.** These are plain OTel spans — using
OpenInference and OTel semantic-convention names where they exist and
clearly-marked custom names where they do not — so Langfuse, Phoenix or your
existing APM all read them. The service identity travels as a Resource, not as a
tracer name, because that is the field every backend groups by. The in-memory
exporter used in tests is the same code path as production.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from assistant.tools import Tool, rewrap

# tool.name follows OpenInference's tool-span convention. The rest are OUR
# extensions — no convention covers agent step counts or approval pauses (yet),
# so they are labelled custom rather than passed off as standard. Where an
# agreed name exists, use it: a dashboard that has never seen this repo can
# still read the trace.
TOOL_NAME = "tool.name"
TOOL_GATED = "tool.requires_approval"
AGENT_STEPS = "agent.step_count"
AGENT_PAUSED = "agent.paused_for_approval"
AGENT_OUTCOME = "agent.outcome"  # completed | paused_for_approval | policy_violation
AGENT_PENDING_TOOL = "agent.pending_tool"
AGENT_APPROVED = "agent.approved_tools"
AGENT_APPROVAL_IDS = "agent.approval_ids"
CACHE_HIT = "cache.hit"
COST = "cost.usd"

# The span names that make up one request. Constants rather than literals because
# a dashboard query is written against these strings, and a typo in a span name
# is a panel that is silently always empty.
REQUEST_SPAN = "assistant.request"  # the HTTP root: one span per request, always
AUTH_SPAN = "auth.verify"
PIPELINE_SPAN = "assistant.pipeline"  # the root when core is driven directly
SCREEN_SPAN = "guardrail.screen"
MEMORY_SPAN = "memory.recall"
RETRIEVAL_SPAN = "rag.search"
COMPOSE_SPAN = "llm.compose"
OUTPUT_SPAN = "guardrail.output"
INGEST_SPAN = "assistant.ingest"

# Request identity. `enduser.id` is the OTel semantic convention for the
# authenticated principal; `request.id` is ours, and it is the value the audit
# log and the response share so a support ticket resolves to a trace.
REQUEST_ID = "request.id"
SUBJECT = "enduser.id"
HTTP_METHOD = "http.request.method"
HTTP_ROUTE = "http.route"
HTTP_STATUS = "http.response.status_code"
AUTH_MODE = "auth.mode"  # jwks | shared-secret | off — which gate actually ran
AUTH_OK = "auth.accepted"
AUTH_REASON = "auth.reason"  # why a token was refused: expired, wrong scope, ...

# Model attributes: OpenInference names, so a Phoenix or Langfuse view populates
# without a mapping layer. The prompt VERSION is the one people forget, and it is
# the one that makes two weeks of numbers comparable.
MODEL_NAME = "llm.model_name"
PROMPT_VERSION = "llm.prompt_template.version"
TOKENS_IN = "llm.token_count.prompt"
TOKENS_OUT = "llm.token_count.completion"
TOKENS_TOTAL = "llm.token_count.total"
# `counted` (the provider returned them) or `estimated` (a word count stood in).
# On the span rather than in a footnote, because the cost attribute beside it
# looks identical either way and only one of the two is a bill.
TOKENS_SOURCE = "llm.token_count.source"
PRICE_TIER = "llm.price_tier"

# Retrieval and memory: counts, not content. A span carrying the retrieved text
# is a span that leaks the corpus into whatever SaaS the traces ship to.
RETRIEVAL_K = "retrieval.k"
RETRIEVAL_HITS = "retrieval.documents.count"
RETRIEVAL_KEPT = "retrieval.documents.kept"
RETRIEVAL_SOURCES = "retrieval.sources"  # which documents answered, by name
#: why a request stopped early — a deadline or a caller who left. Its own
#: attribute rather than an ERROR status: abandoned work is not a fault, and
#: counting it as one hides the faults that are.
ABANDONED = "request.abandoned"
#: set when a stream stopped mid-answer. Distinct from ABANDONED: nobody chose
#: to stop this one, and distinct from a plain error, because the caller did
#: receive text. Without it a truncation is indistinguishable in the trace from
#: a short answer, and "the model gets terse under load" is the wrong diagnosis.
TRUNCATED = "response.truncated"
CORPUS_VERSION = "rag.corpus.version"
TENANT = "rag.tenant"
MEMORY_RECALLED = "memory.recalled.count"

# Guardrail verdicts, so "how often do we refuse, and where" is a query rather
# than a log-scraping exercise.
SCREEN_BLOCKED = "guardrail.blocked"
SCREEN_REASON = "guardrail.reason"
INGEST_ACCEPTED = "ingest.accepted"
INGEST_REJECTED = "ingest.rejected"
INGEST_DELETED = "ingest.deleted"

MS_PER_NS = 1_000_000

#: Bumped when the span tree or an attribute's meaning changes. A dashboard built
#: against v1 should be able to tell that it is looking at v2 data.
SCHEMA_VERSION = "2"


@dataclass
class Recorder:
    """A tracer plus the exporter holding what it emitted.

    In production you add an OTLP exporter to the same provider and change nothing
    else. That is the entire argument for instrumenting against the standard.
    """

    tracer: Tracer
    provider: TracerProvider
    exporter: InMemorySpanExporter

    def spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> list[ReadableSpan]:
        return [s for s in self.spans() if s.name == name]

    def reset(self) -> None:
        self.exporter.clear()


def recorder(
    service: str = "assistant",
    otlp_endpoint: str | None = None,
    version: str = "capstone",
) -> Recorder:
    """A tracer wired to an in-memory exporter, always.

    When `otlp_endpoint` is set (in production, from `OTEL_EXPORTER_OTLP_ENDPOINT`),
    the SAME provider also gets an OTLP exporter, so spans ship to a collector
    without changing a line of instrumentation. The in-memory copy stays, so the
    span assertions in tests and the /health tier report keep working either way.
    The OTLP SDK is imported lazily so the fast tier never installs it.

    The service identity goes on a **Resource**, not just the tracer name. Every
    backend groups, filters and bills by `service.name`; without one, the SDK
    ships `unknown_service` and your traces arrive in the same bucket as
    everybody else's — which is the kind of thing nobody notices until the first
    incident, when the filter that should narrow to one service narrows to
    nothing.
    """
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service,
                "service.version": version,
                "telemetry.schema.version": SCHEMA_VERSION,
            }
        )
    )
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    return Recorder(tracer=provider.get_tracer(service), provider=provider, exporter=exporter)


#: The id every layer of one request agrees on. A ContextVar rather than an
#: argument threaded through six signatures: the HTTP layer opens the scope, and
#: `core.py` — which never sees a Request object — reads the same value, so the
#: span, the audit row and the response all name one request. Copied into the
#: threadpool worker that runs a sync route, which is why this works at all.
_REQUEST_ID: ContextVar[str] = ContextVar("assistant_request_id", default="")


@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """Establish (or inherit) the id for this request.

    Inheritance is the point: `api.py` opens a scope per HTTP request, and the
    `ask()` beneath it opens one too without knowing whether it is being driven
    by HTTP or by a test — and gets the caller's id when there is one. A caller-
    supplied id (an `X-Request-ID` from a gateway) wins over both, so a trace can
    be joined to the log line of the system in front of us.
    """
    rid = request_id or _REQUEST_ID.get() or uuid.uuid4().hex[:12]
    token = _REQUEST_ID.set(rid)
    try:
        yield rid
    finally:
        _REQUEST_ID.reset(token)


def request_id() -> str:
    """The current request id, or "" outside any request scope."""
    return _REQUEST_ID.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def trace_id() -> str:
    """The active trace as the 32-hex string every OTel backend renders, or "".

    Read from the ambient span rather than passed around, so an audit row can be
    bound to its trace without every call site knowing about tracing. The two
    identifiers are not redundant: the request id is OURS and travels in headers
    and responses (a user can quote it), while the trace id is what Phoenix or
    Langfuse indexes. Storing both is what turns "this trace looks wrong" into a
    query instead of a search through prose.
    """
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else ""


@contextmanager
def stage(tracer: Tracer, name: str, **attributes: Any) -> Iterator[Span]:
    """One stage of the pipeline as a child span, with its attributes attached.

    Nesting is implicit: OTel keeps the current span in a context variable, so a
    stage opened inside another stage is its child without anyone passing a
    parent around. That is what lets `core.py` read as a pipeline rather than as
    a pipeline plus a tracing harness.

    Exceptions mark the span ERROR and re-raise. An observability layer that
    swallows one has converted a visible failure into a silent wrong answer.
    """
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_status(Status(StatusCode.OK))


def _apply(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def streaming_root(tracer: Tracer, name: str, **attributes: Any) -> Iterator[Span]:
    """A root span for a generator — one that stays open across `yield`s.

    `stage()` is wrong here, and the reason is worth knowing. "The current span"
    is a ContextVar, and Starlette runs each `next()` of a sync streaming body on
    whichever threadpool worker is free. The token attached on one worker would
    be reset on another: OTel logs "Failed to detach context", and spans opened
    between two yields get parented to whatever that worker did last. A trace
    that is *subtly* wrong is worse than no trace, because it will be believed.

    So this span is never made current. Children name it explicitly (`child_of`),
    and the section that can safely use the ambient context — everything before
    the first yield — opts in with `within()`.
    """
    span = tracer.start_span(name)
    _apply(span, attributes)
    failed = False
    try:
        yield span
    except Exception as exc:
        failed = True
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        if not failed:
            span.set_status(Status(StatusCode.OK))
        span.end()


@contextmanager
def within(span: Span) -> Iterator[Span]:
    """Make `span` the parent for `stage()` calls in this block, without ending
    it. Only safe where the block cannot yield to another thread."""
    with trace.use_span(span, end_on_exit=False):
        yield span


@contextmanager
def child_of(
    tracer: Tracer,
    name: str,
    parent: Span,
    start_time: int | None = None,
    **attributes: Any,
) -> Iterator[Span]:
    """A child span with an explicit parent, and optionally an explicit start.

    `start_time` exists for the streaming case: the work began when the first
    chunk was asked for, but the span can only be closed once the last one
    arrived. Passing the real start (a `time.time_ns()` captured earlier) is the
    difference between a compose span that reports the generation and one that
    reports the bookkeeping after it."""
    span = tracer.start_span(name, context=trace.set_span_in_context(parent), start_time=start_time)
    _apply(span, attributes)
    failed = False
    try:
        yield span
    except Exception as exc:
        failed = True
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        if not failed:
            span.set_status(Status(StatusCode.OK))
        span.end()


def mark(
    tracer: Tracer,
    name: str,
    parent: Span,
    start_time: int | None = None,
    **attributes: Any,
) -> None:
    """Record a child span for work that has already finished.

    The streaming path cannot wrap generation in a `with` block — the block would
    have to contain the yields — so it times the work itself and files the span
    afterwards. Same tree, same durations, no context held across a yield."""
    with child_of(tracer, name, parent, start_time=start_time, **attributes):
        pass


def traced_tool(tool: Tool, tracer: Tracer) -> Tool:
    """Return the same tool with its body wrapped in a span.

    The wrapper re-raises. An observability layer that swallows an exception has
    turned a visible failure into a silent wrong answer, which is a strictly worse
    outcome than the crash it "prevented".
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with tracer.start_as_current_span(f"tool.{tool.name}") as span:
            span.set_attribute(TOOL_NAME, tool.name)
            span.set_attribute(TOOL_GATED, tool.requires_approval)
            try:
                out = tool.fn(*args, **kwargs)
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            span.set_status(Status(StatusCode.OK))
            return out

    return rewrap(tool, wrapped)


def traced_registry(registry: dict[str, Tool], tracer: Tracer) -> dict[str, Tool]:
    """Instrument every tool at once. Adding a tool cannot forget to be traced."""
    return {name: traced_tool(tool, tracer) for name, tool in registry.items()}


def traced_run(run: Any, goal: str, tracer: Tracer, **kwargs: Any) -> Any:
    """Wrap one agent run in a root span, with the loop's own outcome on it.

    `agent.step_count` and `agent.paused_for_approval` are the two attributes you
    will actually filter on later: a run that hit the step cap and a run that
    stopped for a human look identical in a latency chart and could not be more
    different in a review.

    The approval story rides along too: which grants the run was allowed to spend
    (`agent.approved_tools`), the id of each grant it actually spent
    (`agent.approval_ids`), which tool it is now waiting on
    (`agent.pending_tool`), and a single `agent.outcome` you can group a
    dashboard by without reconstructing it from three booleans.

    Recording the grant id is what makes an approval auditable end to end: the
    same id appears on the /approve response, in the audit log, and here, so
    "who authorized this send?" is a lookup rather than a reconstruction.
    """
    with tracer.start_as_current_span("agent.run") as span:
        approvals = kwargs.get("approvals") or {}
        span.set_attribute(
            AGENT_APPROVED, sorted(name for name, granted in approvals.items() if granted)
        )
        result = run(goal, **kwargs)
        spent: dict[str, str] = getattr(result, "approval_ids", {}) or {}
        if spent:
            # a consuming store was in play; the allow-list snapshot above was empty
            span.set_attribute(AGENT_APPROVED, sorted(spent))
            span.set_attribute(AGENT_APPROVAL_IDS, [spent[k] for k in sorted(spent)])
        span.set_attribute(AGENT_STEPS, len(getattr(result, "audit", []) or []))
        pending = getattr(result, "pending", None)
        span.set_attribute(AGENT_PAUSED, pending is not None)
        if pending is not None:
            span.set_attribute(AGENT_PENDING_TOOL, pending.tool)
        if getattr(result, "fired_irreversible_tool_without_approval", False):
            span.set_attribute(AGENT_OUTCOME, "policy_violation")
            span.set_status(Status(StatusCode.ERROR, "gated tool fired without approval"))
        else:
            span.set_attribute(
                AGENT_OUTCOME, "paused_for_approval" if pending is not None else "completed"
            )
            span.set_status(Status(StatusCode.OK))
        return result


# --- read the answers off the spans ----------------------------------------


def duration_ms(span: ReadableSpan) -> float:
    if span.end_time is None or span.start_time is None:
        return 0.0
    return (span.end_time - span.start_time) / MS_PER_NS


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank: index is ceil(p/100 * n) - 1. Not `round()` — ties-to-even
    shifts exact ranks and reports the wrong P50/P99."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


def tool_calls(spans: Sequence[ReadableSpan]) -> list[str]:
    """Which tools ran, in order. The audit trail you didn't have to write."""
    return [
        str((s.attributes or {}).get(TOOL_NAME))
        for s in spans
        if (s.attributes or {}).get(TOOL_NAME) is not None
    ]


def time_by_tool(spans: Sequence[ReadableSpan]) -> dict[str, float]:
    """Where the wall clock went. Usually not where you assumed."""
    totals: dict[str, float] = {}
    for span in spans:
        name = (span.attributes or {}).get(TOOL_NAME)
        if name is not None:
            key = str(name)
            totals[key] = totals.get(key, 0.0) + duration_ms(span)
    return totals


def slowest_tool(spans: Sequence[ReadableSpan]) -> str | None:
    totals = time_by_tool(spans)
    return max(totals, key=lambda k: totals[k]) if totals else None


def failed_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    return [s for s in spans if s.status.status_code is StatusCode.ERROR]


def children_of(spans: Sequence[ReadableSpan], parent: ReadableSpan) -> list[ReadableSpan]:
    """The spans whose parent is this one. The tree, read back."""
    parent_id = parent.context.span_id if parent.context else None
    return [s for s in spans if s.parent is not None and s.parent.span_id == parent_id]


def descendants_of(spans: Sequence[ReadableSpan], parent: ReadableSpan) -> list[ReadableSpan]:
    """Everything under this span, at any depth — the whole subtree."""
    found, frontier = [], [parent]
    while frontier:
        for child in children_of(spans, frontier.pop()):
            found.append(child)
            frontier.append(child)
    return found


def one_trace(spans: Sequence[ReadableSpan]) -> bool:
    """True when every span shares a trace id and exactly one has no parent.

    Worth asserting rather than assuming: two roots mean a stage opened its span
    outside the request's context, and the symptom is a dashboard where the
    stage exists but never appears under the request that caused it."""
    if not spans:
        return False
    traces = {s.context.trace_id for s in spans if s.context}
    roots = [s for s in spans if s.parent is None]
    return len(traces) == 1 and len(roots) == 1


def attr(span: ReadableSpan, key: str) -> Any:
    return (span.attributes or {}).get(key)


def gated_tool_calls(spans: Sequence[ReadableSpan]) -> list[str]:
    """Every irreversible tool that actually fired.

    This is a safety report, not a performance one. Phase 6 asked you to *contain*
    these; production asks you to prove which ones ran.
    """
    return [
        str((s.attributes or {}).get(TOOL_NAME))
        for s in spans
        if (s.attributes or {}).get(TOOL_GATED) is True
    ]

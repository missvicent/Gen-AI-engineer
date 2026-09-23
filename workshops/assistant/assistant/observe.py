"""TODO: Workshop 8 layer — OpenTelemetry around the whole request.

An agent is harder to observe than a single model call, because the interesting
question is never "how long did that take". It is *what did it do* — which tools, in
what order, how many steps, and where the eight seconds actually went. A timer on
`run()` cannot answer any of that; a span tree can.

The wiring is given. Yours:

1) recorder — the service identity as a **Resource**, not just a tracer name.
2) request_scope — one id per request that every layer can read.
3) stage — one child span per pipeline stage, nested by ambient context.
4) traced_tool / traced_registry — wrap the tools at the **seam**. No tool should
   know it is being traced, which is what stops a newly-added tool from forgetting.
5) traced_run — a root span per run, carrying step count and whether it paused.
6) The readers — tool_calls, time_by_tool, slowest_tool, failed_spans,
   children_of, descendants_of, one_trace, gated_tool_calls — all derived from
   the spans.

`streaming_root`, `within`, `child_of` and `mark` are GIVEN, and worth reading
before you use them: they exist because "the current span" is a ContextVar, and a
generator that yields between two spans cannot rely on one. Getting that wrong
produces a trace that is subtly mis-parented, which is worse than no trace at all
because it will be believed.

Use the attribute names defined below rather than inventing your own: agreed names
are what let a dashboard that has never seen this repo read your trace.

Reference: ../../after/src/assistant/observe.py.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from assistant.tools import Tool

# tool.name follows OpenInference's tool-span convention; the agent.*, cache.hit
# and cost.usd names are OUR extensions — labelled custom, not passed off as
# standard.
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
#: set when a stream stopped mid-answer. Distinct from a plain error, because
#: the caller did receive text. Without it a truncation is indistinguishable in
#: the trace from a short answer, and "the model gets terse under load" is the
#: wrong diagnosis.
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
    else. That portability is the whole argument for instrumenting against OTel
    instead of against a vendor's client library.
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
    """TODO 1: a tracer wired to an in-memory exporter, always — plus a Resource.

    Build a `TracerProvider` whose `resource` is `Resource.create({...})` carrying
    `service.name`, `service.version` and `telemetry.schema.version`. The identity
    belongs on the Resource, not just the tracer name: every backend groups,
    filters and bills by `service.name`, and without one the SDK ships
    `unknown_service` — which nobody notices until the first incident, when the
    filter that should narrow to one service narrows to nothing.

    Attach a `SimpleSpanProcessor(InMemorySpanExporter())` unconditionally (tests
    and /health read it), and when `otlp_endpoint` is set add a
    `BatchSpanProcessor(OTLPSpanExporter(...))` to the SAME provider so spans ship
    to a collector with no change to instrumentation. Import the OTLP SDK inside
    the branch so the fast tier never installs it.
    """
    raise NotImplementedError


#: The id every layer of one request agrees on. A ContextVar rather than an
#: argument threaded through six signatures: the HTTP layer opens the scope, and
#: `core.py` — which never sees a Request object — reads the same value.
_REQUEST_ID: ContextVar[str] = ContextVar("assistant_request_id", default="")


@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """TODO 2: establish (or inherit) the id for this request.

    Yield `request_id` if given, else the id already in scope, else a new one
    (`new_request_id`). Set the ContextVar for the duration and reset it after —
    with the token, in a `finally`, or a nested request leaks its id to whatever
    runs next.

    Inheritance is the point: `api.py` opens a scope per HTTP request and the
    `ask()` beneath it opens one too, without knowing whether it is being driven
    by HTTP or by a test. A caller-supplied id (an `X-Request-ID` from a gateway)
    wins over both, so the trace can be joined to the logs of the system in front
    of you.
    """
    raise NotImplementedError


def request_id() -> str:
    """The current request id, or "" outside any request scope."""
    return _REQUEST_ID.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def trace_id() -> str:
    """TODO: the active trace as the 32-hex string every OTel backend renders,
    or "" when there is no valid span context.

    `format(trace.get_current_span().get_span_context().trace_id, "032x")`, and
    check `.is_valid` first. Read from the ambient span rather than passed
    around, so an audit row can be bound to its trace without every call site
    knowing about tracing. Not redundant with the request id: that one is OURS
    and travels in headers, this one is what Phoenix or Langfuse indexes.
    """
    raise NotImplementedError


@contextmanager
def stage(tracer: Tracer, name: str, **attributes: Any) -> Iterator[Span]:
    """TODO 3: one stage of the pipeline as a child span, with its attributes.

    Use `tracer.start_as_current_span(name)`, set every non-None attribute, and
    yield the span. On an exception set ERROR status and **re-raise** — an
    observability layer that swallows one has converted a visible failure into a
    silent wrong answer. Otherwise set OK.

    Nesting is implicit: OTel keeps the current span in a context variable, so a
    stage opened inside another stage is its child without anyone passing a
    parent around. That is what lets `core.py` read as a pipeline rather than as
    a pipeline plus a tracing harness.
    """
    raise NotImplementedError


def _apply(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def streaming_root(tracer: Tracer, name: str, **attributes: Any) -> Iterator[Span]:
    """GIVEN: a root span for a generator — one that stays open across `yield`s.

    `stage()` is wrong here, and the reason is worth knowing. "The current span"
    is a ContextVar, and Starlette runs each `next()` of a sync streaming body on
    whichever threadpool worker is free. The token attached on one worker would
    be reset on another: OTel logs "Failed to detach context", and spans opened
    between two yields get parented to whatever that worker did last.

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
    """GIVEN: make `span` the parent for `stage()` calls in this block, without
    ending it. Only safe where the block cannot yield to another thread."""
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
    """GIVEN: a child span with an explicit parent, and optionally an explicit
    start. `start_time` exists for the streaming case: the work began when the
    first chunk was asked for, but the span can only be closed once the last one
    arrived."""
    span = tracer.start_span(
        name, context=trace.set_span_in_context(parent), start_time=start_time
    )
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
    """GIVEN: record a child span for work that has already finished."""
    with child_of(tracer, name, parent, start_time=start_time, **attributes):
        pass


def traced_tool(tool: Tool, tracer: Tracer) -> Tool:
    """TODO 4: return the same tool with its body wrapped in a span.

    Name the span `tool.<name>`, set TOOL_NAME and TOOL_GATED, and mark the span
    ERROR on an exception — then **re-raise**. An observability layer that swallows
    an exception has turned a visible failure into a silent wrong answer, which is
    strictly worse than the crash it "prevented".

    Build the replacement with `tools.rewrap` rather than `Tool(...)`: tracing has
    no business changing what a tool advertises, and a hand-built Tool silently
    loses `required_args` and with it the planner's ability to choose the tool.
    """
    raise NotImplementedError


def traced_registry(registry: dict[str, Tool], tracer: Tracer) -> dict[str, Tool]:
    """TODO 5: instrument every tool at once, so adding one cannot forget to."""
    raise NotImplementedError


def traced_run(run: Any, goal: str, tracer: Tracer, **kwargs: Any) -> Any:
    """TODO 6: wrap one agent run in an `agent.run` root span.

    Set AGENT_STEPS and AGENT_PAUSED — those are the two attributes you will
    actually filter on later, because a run that stopped for a human and a run that
    hit the step cap look identical on a latency chart and could not be more
    different in a review. Mark the span ERROR on a containment breach.

    Then put the approval story on the same span: AGENT_APPROVED = the sorted
    names in `kwargs["approvals"]` whose grant is truthy (set it BEFORE calling
    `run` — it describes what the run started with), AGENT_PENDING_TOOL = the
    tool a paused run is waiting on, and AGENT_OUTCOME = one of "completed",
    "paused_for_approval" or "policy_violation", so a dashboard can group runs
    without reconstructing the outcome from three booleans.

    A run driven by a consuming approval store has no allow-list to read, so when
    `result.approval_ids` is non-empty use it instead: AGENT_APPROVED = its sorted
    keys, AGENT_APPROVAL_IDS = the matching grant ids. Recording the id is what
    makes an approval auditable end to end — the same id appears on the /approve
    response, in the audit log and here, so "who authorized this send?" is a
    lookup rather than a reconstruction.
    """
    raise NotImplementedError


# --- read the answers off the spans ----------------------------------------


def duration_ms(span: ReadableSpan) -> float:
    """TODO 7: span duration in milliseconds. OTel stores nanoseconds."""
    raise NotImplementedError


def percentile(values: Sequence[float], p: float) -> float:
    """TODO 8: nearest-rank percentile, 0.0 on empty.
    The rank is ceil(p/100 * n) — not round(), which shifts ties to even."""
    raise NotImplementedError


def tool_calls(spans: Sequence[ReadableSpan]) -> list[str]:
    """TODO 9: which tools ran, in order — the audit trail you didn't have to write."""
    raise NotImplementedError


def time_by_tool(spans: Sequence[ReadableSpan]) -> dict[str, float]:
    """TODO 10: milliseconds per tool. Usually not where you assumed."""
    raise NotImplementedError


def slowest_tool(spans: Sequence[ReadableSpan]) -> str | None:
    """TODO 11: the worst offender, or None on an empty trace."""
    raise NotImplementedError


def failed_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """TODO 12: spans whose status is ERROR."""
    raise NotImplementedError


def children_of(spans: Sequence[ReadableSpan], parent: ReadableSpan) -> list[ReadableSpan]:
    """TODO 13: the spans whose parent is this one — the tree, read back."""
    raise NotImplementedError


def descendants_of(
    spans: Sequence[ReadableSpan], parent: ReadableSpan
) -> list[ReadableSpan]:
    """TODO 14: everything under this span, at any depth."""
    raise NotImplementedError


def one_trace(spans: Sequence[ReadableSpan]) -> bool:
    """TODO 15: True when every span shares a trace id and exactly one has no parent.

    Worth asserting rather than assuming: two roots mean a stage opened its span
    outside the request's context, and the symptom is a dashboard where the stage
    exists but never appears under the request that caused it.
    """
    raise NotImplementedError


def attr(span: ReadableSpan, key: str) -> Any:
    return (span.attributes or {}).get(key)


def gated_tool_calls(spans: Sequence[ReadableSpan]) -> list[str]:
    """TODO 16: every irreversible tool that actually fired.

    This is a safety report, not a performance one. Phase 6 asked you to *contain*
    these; production asks you to prove which ones ran. The caching layer in
    `cache.py` reads this to decide what it is allowed to store.
    """
    raise NotImplementedError

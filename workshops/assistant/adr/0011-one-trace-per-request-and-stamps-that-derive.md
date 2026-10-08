# ADR-0011 — One trace per request, and version stamps that derive themselves

**Status:** accepted

## Context

ADR-0004 made OTel spans the observability currency, and the capstone honoured it
in the narrowest possible way: `agent.run` and one span per tool. That is a good
answer to "which tools ran" and no answer at all to anything else.

Three things were missing, and each of them was invisible rather than wrong.

The service had no identity. `TracerProvider()` with no Resource ships
`service.name = unknown_service`, which nobody notices locally and everybody
notices during the first incident, when the filter that should narrow to one
service narrows to nothing.

There was no root over the request. Screening, memory, retrieval, composition and
the output gate — the five stages where the time and the money actually go — left
no trace, so "the P99 is four seconds" could not be decomposed into a cause.
Worse, a request that never reached the agent loop (refused at the screen,
rejected at auth) produced no spans at all: the failures most worth seeing were
the ones least visible.

And nothing recorded *which system* produced a number. Two weeks of latency data
spanning a prompt edit and a model swap look like one dataset unless the spans say
otherwise, and a hand-maintained `PROMPT_VERSION = "v3"` beside the prompt goes
stale in silence — no test fails when a label stops being true.

## Decision

**One root per request, one child per stage.** `assistant.request` at the HTTP
door with method, route, status and a request id; `auth.verify` and
`assistant.pipeline` beneath it; and under the pipeline `guardrail.screen`,
`memory.recall`, `rag.search`, `agent.run`, `llm.compose` and `guardrail.output`.
Nesting is ambient — OTel's current-span ContextVar — so `core.py` reads as a
pipeline rather than as a pipeline plus a tracing harness. `one_trace()` asserts
the shape in the fast tier, because a stage that opens its span outside the
request's context still records; it just records somewhere nobody will look.

**The identity is a Resource**, carrying `service.name`, `service.version` and a
`telemetry.schema.version` that gets bumped when the tree changes, so a dashboard
built against v1 can tell it is reading v2 data.

**Version stamps are derived, never typed.** `provenance.py` hashes the source of
`grounded_prompt` for the prompt stamp and hashes the documents for the corpus
stamp; `core.py` puts them on spans and `report.py` puts them on the report the
merge gate reads. One derivation, two consumers: a trace and the report
describing it name the same system by construction.

**One meter for tokens and money.** `usage.py` is used by the compose span and by
the cost gate. Two meters would be two answers to "what did last week cost", and
the interesting week is always the one where they disagree.

**One request id, shared.** Minted at the HTTP door (or taken from an inbound
`X-Request-ID`), carried in a ContextVar so `core.py` — which never sees a
`Request` — reads the same value, and returned on the span, in the response body
and in the `x-request-id` header.

**The streaming root is built by hand.** `ask_stream` cannot wrap its yields in a
`start_as_current_span` block: "current" is a ContextVar and Starlette runs each
`next()` of a sync streaming body on whichever threadpool worker is free, so the
token attached on one worker would be reset on another. The root is created
un-current (`streaming_root`), the pre-yield section opts into it (`within`), and
the compose and output spans are filed afterwards with an explicit parent and a
backdated `start_time` (`mark`).

## Alternatives considered

Attributes in a parallel log line (they diverge from the traces, and you end up
trusting the wrong one). A single `assistant.request` span with stage durations as
attributes (cheap, and unqueryable — you cannot chart a P99 of an attribute that
only exists as a number inside another span). Passing a request id through six
function signatures instead of a ContextVar (honest, and it makes every
intermediate function's signature about tracing). Buffering the streamed answer so
the metering could be a tidy `with` block (trades the feature for the
instrumentation). Recording retrieved text on the retrieval span (it would ship
the corpus to whatever SaaS the traces land in; counts only). Marking a 401's
`auth.verify` span OK because a refused token is not a server error (it is an
error *for that request*, and ERROR is the status a dashboard can count).

## Consequences

Span count per request goes from roughly two-plus-tools to eight-plus-tools. That
is a real cost at volume and the reason `SimpleSpanProcessor` is only used for the
in-memory exporter; the OTLP path batches. Sampling, if it becomes necessary,
belongs at the root, so a sampled request keeps its whole tree or none of it.

The streaming path's compose span is closed after the final frame. Its duration is
therefore time-to-*last*-token; time-to-first-token is the gap between the root's
start and the first `chunk`, computable in a backend and deliberately not recorded
twice.

Every response now carries `request_id`, which is a small API change and the thing
that makes a support ticket resolvable.

# ADR-0004 — OpenTelemetry spans as the only observability currency

**Status:** accepted

## Context

The interesting question about an agent is never "how long did it take" but "what
did it do" — which tools, in what order, with which approvals live, and whether a
gated one fired. That evidence must be readable offline in tests AND shippable to
whatever backend a team already runs, without instrumenting twice.

## Decision

Instrument once, against the OTel SDK: a root `agent.run` span carrying step
count, pause state, live approvals, the pending tool and a single
`agent.outcome`; a child span per tool, added by wrapping the registry so no tool
can forget. The provider always keeps an in-memory exporter (tests, `/health`'s
`spans_recorded`, the portfolio's latency percentiles); setting
`OTEL_EXPORTER_OTLP_ENDPOINT` adds an OTLP exporter to the same provider. No
vendor SDK is imported anywhere.

## Alternatives considered

A vendor client (Langfuse/Phoenix SDK) — couples every lesson to one backend and
its account model; structured logging — loses the tree, and "where did the eight
seconds go" is a tree question; instrumenting inside each tool — the exact
cross-cutting-concern rot the registry wrapper exists to prevent.

## Consequences

The same spans back the tests, the trace-diagnosis drill (`test_diagnosis.py`),
the portfolio report, and the collector check in `verify-e2e.sh` (spans observed
outside the process through the compose observability overlay). Cost: OTel's SDK
surface moves occasionally — the pinned version and VERIFIED.md stamps absorb
that.

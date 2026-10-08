# ADR-0005 — SQLite for memory, audit and idempotency: one file, not three services

**Status:** accepted

## Context

Three kinds of state must survive a container restart: the assistant's memory,
the audit log (approvals, tool runs, policy decisions, identities), and the
idempotency dedupe table. The stack already runs four services; every additional
stateful service multiplies the operational surface a reviewer must boot.

## Decision

One SQLite file (`ASSISTANT_DB`, a compose volume) backs all three, through three
small modules with the same interfaces as their in-process counterparts —
`sqlite_memory.py` mirrors `memory.py` exactly, so the composition root swaps them
per ADR-0001 without caring. Durability is proven by tests that reopen the store
cold, and by e2e: memory and the audit trail survive a `docker compose restart`.

## Alternatives considered

Postgres (right beyond a single writer node; here it adds a fifth service and
migrations to a lesson about seams, not databases); Redis for idempotency keys
(a second store for a table with one index); keeping audit in OTel spans only
(spans are sampled telemetry with retention policies — an audit log is a record
with neither property).

## Consequences

Zero extra services; state inspectable with `sqlite3`; backup/restore is file
copy, documented in the runbook. Cost: single-writer concurrency and no
server-side access control — accepted at this scale and stated in
THREAT-MODEL.md; the seams make a Postgres adapter a contained change if the
scale argument ever flips.

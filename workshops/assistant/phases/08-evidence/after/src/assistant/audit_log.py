"""Persistent audit log — who did what, decided by which policy, backed by SQLite.

The agent's per-request `audit` list answers "what happened in THIS run"; this
log answers the operator's question a week later: which identity asked for the
send, when was it approved, what did the guardrails block.

## Why a row is more than a sentence

The first version of this table was `(at, kind, subject, detail)`, and `detail`
was prose — `"send_telegram (approval 9f2c…)"`. It reads well and it answers
nothing. Every real question asked of an audit log is a JOIN:

    "this trace looks wrong, what did it DO?"        -> by trace id
    "the customer says we messaged them twice"       -> by args hash
    "which approval authorized this?"                -> by approval id
    "show me everything in request abc123"           -> by request id

Prose cannot be joined on, and the moment somebody needs to they will write a
regex over `detail` — which works until the format changes, at which point the
trail silently stops matching and nobody notices, because a query returning zero
rows looks exactly like nothing having happened.

So every kind writes the same **columns**, and each is bound to the same request
the spans are:

    request_id   joins the row to the API response and to the log line
    trace_id     joins it to the span tree, so "what did this trace do" is a query
    subject      the VERIFIED identity, never one from a request body
    approval_id  which grant authorized this, for the rows where one did
    args_hash    canonical fingerprint — the same call written two ways matches
    result       ok / blocked / pending / rejected — the outcome as a value

`detail` survives as free text for the human reading the trail, but nothing
queries it.

## Kinds recorded by the service

    policy.blocked      input refused by guardrails (result=blocked)
    ingest.rejected     a document refused AT INGEST, so it was never stored
    corpus.ingested     documents added to a tenant's corpus (result=ok)
    corpus.deleted      a source removed from a tenant's corpus
    tool.pending        a gated tool paused for approval (result=pending)
    tool.ran            any tool executed inside the agent loop
    approval.granted    /approve minted one single-use grant
    approval.replayed   a mutation replayed (idempotency key already seen)

Shares the ASSISTANT_DB file when configured, so the trail survives a restart
alongside memory, approvals, the idempotency table and the outbox.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from assistant.approvals import args_fingerprint
from assistant.observe import request_id, trace_id

#: Outcomes, as values rather than adjectives buried in a sentence. Small
#: vocabulary on purpose: a column with forty distinct strings is prose again.
OK = "ok"
BLOCKED = "blocked"
PENDING = "pending"
REJECTED = "rejected"
DELETED = "deleted"
REPLAYED = "replayed"


@dataclass(frozen=True)
class AuditEntry:
    at: str
    kind: str
    subject: str
    detail: str
    request_id: str = ""
    trace_id: str = ""
    approval_id: str = ""
    args_hash: str = ""
    result: str = ""


class AuditLog:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_log ("
            "  seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  at TEXT NOT NULL DEFAULT (datetime('now')),"
            "  kind TEXT NOT NULL,"
            "  subject TEXT NOT NULL,"
            "  detail TEXT NOT NULL,"
            # Defaults, not NOT NULL: a row written outside a request scope (a
            # background reconciliation, a test) is still worth having, and an
            # audit log that refuses to record because a column is missing is an
            # audit log that loses the incident it existed for.
            "  request_id TEXT NOT NULL DEFAULT '',"
            "  trace_id TEXT NOT NULL DEFAULT '',"
            "  approval_id TEXT NOT NULL DEFAULT '',"
            "  args_hash TEXT NOT NULL DEFAULT '',"
            "  result TEXT NOT NULL DEFAULT ''"
            ")"
        )
        self._migrate()
        # The two joins that are actually run under pressure. An audit log
        # nobody can query in an incident is a compliance artifact, not a tool.
        self._conn.execute("CREATE INDEX IF NOT EXISTS audit_request ON audit_log (request_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS audit_trace ON audit_log (trace_id)")
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the binding columns to a table written by an older build.

        The alternative is dropping and recreating, which throws away exactly the
        history this file exists to keep. `ALTER TABLE ADD COLUMN` with a default
        is instant in SQLite and leaves old rows readable — they simply have
        empty bindings, which is the truth about them.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(audit_log)")}
        for column in ("request_id", "trace_id", "approval_id", "args_hash", "result"):
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE audit_log ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )

    def record(
        self,
        kind: str,
        subject: str,
        detail: str = "",
        *,
        approval_id: str = "",
        args: dict[str, Any] | None = None,
        result: str = OK,
    ) -> None:
        """Write one row, bound to the request it happened in.

        `request_id` and `trace_id` are read from the ambient context rather than
        passed in, for the same reason the tracer does it: an identifier that has
        to be threaded through every call site is one that will be dropped at the
        first call site somebody adds in a hurry, and a row with no binding is a
        row that cannot be joined to anything.

        `args` are fingerprinted, not stored. The canonical hash answers "was
        this the same call?" — which is the question an audit actually asks —
        without putting a customer's phone number in a table that outlives the
        request.
        """
        self._conn.execute(
            "INSERT INTO audit_log"
            " (kind, subject, detail, request_id, trace_id, approval_id, args_hash, result)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                subject,
                detail,
                request_id(),
                trace_id(),
                approval_id,
                args_fingerprint(args) if args is not None else "",
                result,
            ),
        )
        self._conn.commit()

    def entries(
        self,
        kind: str | None = None,
        *,
        request: str | None = None,
        trace: str | None = None,
    ) -> list[AuditEntry]:
        """The trail, filtered by the things an operator actually has in hand: a
        kind, a request id off an API response, or a trace id off a span."""
        clauses, params = [], []
        for column, value in (("kind", kind), ("request_id", request), ("trace_id", trace)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            "SELECT at, kind, subject, detail, request_id, trace_id, approval_id,"
            f" args_hash, result FROM audit_log{where} ORDER BY seq",
            tuple(params),
        ).fetchall()
        return [AuditEntry(*row) for row in rows]

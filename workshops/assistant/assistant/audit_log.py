"""TODO: persistent audit log — who did what, bound to the request it happened in.

The agent's per-request `audit` list answers "what happened in THIS run"; this
log answers the operator's question a week later: which identity asked for the
send, when was it approved, what did the guardrails block.

The version worth building is not `(at, kind, subject, detail)` with prose in
`detail`. It reads well and it answers nothing, because every real question
asked of an audit log is a JOIN:

    "this trace looks wrong, what did it DO?"        -> by trace id
    "the customer says we messaged them twice"       -> by args hash
    "which approval authorized this?"                -> by approval id
    "show me everything in request abc123"           -> by request id

Prose cannot be joined on, and the moment somebody needs to they will write a
regex over `detail` — which works until the format changes, at which point the
trail silently stops matching and nobody notices, because a query returning zero
rows looks exactly like nothing having happened.

Kinds recorded by the service:
    policy.blocked      input refused by guardrails (result=blocked)
    ingest.rejected     a document refused AT INGEST, so it was never stored
    corpus.deleted      a source removed from a tenant's corpus
    tool.pending        a gated tool paused for approval (result=pending)
    tool.ran            any tool executed inside the agent loop
    approval.granted    /approve minted one single-use grant
    approval.replayed   a mutation replayed (idempotency key already seen)

Reference: ../../after/src/assistant/audit_log.py.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

#: Outcomes, as values rather than adjectives buried in a sentence.
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
        # TODO 1: give the table the binding columns as well as the prose ones —
        # request_id, trace_id, approval_id, args_hash, result. Use
        # `TEXT NOT NULL DEFAULT ''` rather than NOT NULL: a row written outside
        # a request scope (a background job, a test) is still worth having, and
        # an audit log that refuses to record because a column is missing is one
        # that loses the incident it existed for.
        #
        # Index request_id and trace_id. Those are the two joins actually run
        # under pressure, and a log nobody can query in an incident is a
        # compliance artifact rather than a tool.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_log ("
            "  seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  at TEXT NOT NULL DEFAULT (datetime('now')),"
            "  kind TEXT NOT NULL,"
            "  subject TEXT NOT NULL,"
            "  detail TEXT NOT NULL"
            ")"
        )
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """TODO 2: add the binding columns to a table written by an older build.

        Read `PRAGMA table_info(audit_log)` and `ALTER TABLE ... ADD COLUMN` the
        missing ones. Dropping and recreating would throw away exactly the
        history this file exists to keep; the old rows should stay readable with
        empty bindings, which is the truth about them.
        """
        raise NotImplementedError

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
        """TODO 3: write one row, bound to the request it happened in.

        Read `observe.request_id()` and `observe.trace_id()` from the ambient
        context rather than taking them as parameters — for the same reason the
        tracer does: an identifier threaded through every call site is one that
        gets dropped at the first call site somebody adds in a hurry.

        Fingerprint `args` with `approvals.args_fingerprint`; do not store them.
        The hash answers "was this the same call?" — the question an audit
        actually asks — without putting a customer's phone number in a table
        that outlives the request.
        """
        raise NotImplementedError

    def entries(
        self,
        kind: str | None = None,
        *,
        request: str | None = None,
        trace: str | None = None,
    ) -> list[AuditEntry]:
        """TODO 4: the trail, filtered by the things an operator has in hand — a
        kind, a request id off an API response, or a trace id off a span. Build
        the WHERE clause from whichever were supplied; order by `seq`."""
        raise NotImplementedError

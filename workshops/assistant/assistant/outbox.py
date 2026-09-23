"""TODO: irreversible effects, written down before they happen.

The gap this closes is small and permanent. An irreversible tool runs like this:

    approval_id = approvals.consume(...)   # the grant is now spent
    result = send_telegram(...)            # the message is now sent
    audit_log.record("tool.ran", ...)      # ...if we get this far

Every line is fine and the sequence is not, because the process can stop between
any two of them. Crash after line 2 and a message went out that nothing in the
system remembers: the grant is gone, the log is silent, and the only evidence is
in somebody else's inbox. That is not a rare-race-condition story — it is a
deploy that rolled during a send, an OOM kill, a `docker compose down`.

There is no way to make "spend the grant", "call the API" and "write the row" one
atomic operation; the API is not in our database. What CAN be atomic is the
*intent*. So the order becomes:

    reserve  → a durable `pending` row, committed BEFORE the call
    call     → the effect
    settle   → the same row marked `sent` or `failed`, with the outcome

Now a crash leaves a `pending` row, which is a question ("did this send?") rather
than silence. This is the transactional-outbox pattern with the delivery worker
left out on purpose: automatic redelivery of a possibly-already-delivered
irreversible action is a *worse* default than telling a human.

Reference: ../../after/src/assistant/outbox.py.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from assistant.tools import Tool

PENDING = "pending"
SENT = "sent"
FAILED = "failed"


@dataclass(frozen=True)
class Effect:
    effect_id: str
    subject: str
    tool: str
    args_hash: str
    request_id: str
    status: str
    detail: str
    at: str


class Outbox:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS outbox ("
            "  effect_id TEXT PRIMARY KEY,"
            "  subject TEXT NOT NULL,"
            "  tool TEXT NOT NULL,"
            "  args_hash TEXT NOT NULL,"
            "  request_id TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  detail TEXT NOT NULL DEFAULT '',"
            "  at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        # TODO 1: a UNIQUE index on (subject, tool, args_hash, request_id).
        #
        # This is what makes `reserve` idempotent, and it has to be an index
        # rather than a check-then-insert, because a check-then-insert is a race
        # with a comment explaining why it is not. `request_id` belongs IN the
        # key: two deliberate sends of the same message are two requests and both
        # should happen — a dedupe that cannot tell "retry" from "again" is a
        # dedupe that loses messages.
        self._conn.commit()

    def reserve(
        self, subject: str, tool: str, args: dict[str, Any] | None, request_id: str
    ) -> str | None:
        """TODO 2: write the intent down and return its id, or None if this exact
        call in this exact request was already reserved.

        Use `INSERT OR IGNORE` with a fresh uuid and status PENDING, hashing the
        args with `approvals.args_fingerprint`, and COMMIT before returning.
        `cur.rowcount` tells you whether the row is yours.

        The commit ordering is the entire point: a row that exists without an
        effect is a false alarm somebody can close, and an effect that exists
        without a row is a message nobody can account for.
        """
        raise NotImplementedError

    def settle(self, effect_id: str, status: str, detail: str = "") -> None:
        """TODO 3: close the envelope — `sent` or `failed`, with what happened."""
        raise NotImplementedError

    def pending(self) -> list[Effect]:
        """TODO 4: effects that were started and never settled.

        The reconciliation queue, and the first thing to read after an unclean
        restart.
        """
        raise NotImplementedError

    def entries(self, subject: str | None = None) -> list[Effect]:
        """TODO 5: every recorded effect, optionally for one subject only."""
        raise NotImplementedError


def recorded_registry(
    registry: dict[str, Tool], outbox: Outbox, subject: str, request_id: str
) -> dict[str, Tool]:
    """TODO 6: wrap every IRREVERSIBLE tool so the intent is durable before the
    call.

    At the seam, like the tracing and screening wrappers, and for the same
    reason: a rule that lives in each tool body is a rule the next tool will be
    written without. Read-only tools (`requires_approval is False`) pass through
    untouched — an outbox row for a search is bookkeeping nobody will ever read.

    Each wrapped body should:

      1. `reserve` the effect. A `None` back means this exact call was already
         recorded for this request — a retry that got past the layers above.
         Refuse it (return an error dict) rather than send twice: at the last
         possible moment, with the arguments in hand, is the only place that
         check is unambiguous.
      2. run the real body, `settle` as SENT with the result;
      3. on any exception, `settle` as FAILED with the repr and re-raise. A
         failure that is WRITTEN DOWN — without it the row stays pending forever
         and reconciliation cannot tell "crashed mid-send" from "the API said no".

    Use `tools.rewrap` so the wrapper keeps the tool's name, doc and
    `required_args`; a wrapper that drops them silently un-selects the tool.
    """
    raise NotImplementedError

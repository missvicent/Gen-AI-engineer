"""Irreversible effects, written down before they happen.

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
than silence. `pending()` lists them, so reconciliation is a query instead of an
archaeology project. This is the transactional-outbox pattern with the delivery
worker left out on purpose: automatic redelivery of a possibly-already-delivered
irreversible action is a *worse* default than telling a human, and the honest
version of this pattern at this scale is a list an operator can read.

The row is keyed by `(subject, tool, args_hash, request_id)`, which makes the
reserve itself idempotent: a retried request cannot open a second envelope for
the same call. `request_id` is in the key rather than out of it because two
deliberate sends of the same message are two different requests and both should
happen.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from assistant.approvals import args_fingerprint
from assistant.tools import Tool, rewrap

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
        # The uniqueness that makes `reserve` idempotent. An index rather than a
        # check-then-insert, because a check-then-insert is a race with a comment
        # explaining why it is not.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS outbox_call"
            "  ON outbox (subject, tool, args_hash, request_id)"
        )
        self._conn.commit()

    def reserve(
        self, subject: str, tool: str, args: dict[str, Any] | None, request_id: str
    ) -> str | None:
        """Write the intent down and return its id, or None if this exact call in
        this exact request was already reserved.

        Committed before the caller does anything irreversible. That ordering is
        the entire point: a row that exists without an effect is a false alarm
        somebody can close, and an effect that exists without a row is a message
        nobody can account for."""
        effect_id = str(uuid.uuid4())
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO outbox"
                " (effect_id, subject, tool, args_hash, request_id, status)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (effect_id, subject, tool, args_fingerprint(args), request_id, PENDING),
            )
            self._conn.commit()
        return effect_id if cur.rowcount else None

    def settle(self, effect_id: str, status: str, detail: str = "") -> None:
        """Close the envelope: `sent` or `failed`, with what happened."""
        with self._lock:
            self._conn.execute(
                "UPDATE outbox SET status = ?, detail = ? WHERE effect_id = ?",
                (status, str(detail)[:500], effect_id),
            )
            self._conn.commit()

    def pending(self) -> list[Effect]:
        """Effects that were started and never settled — the reconciliation
        queue, and the first thing to read after an unclean restart."""
        return self._rows("WHERE status = ?", (PENDING,))

    def entries(self, subject: str | None = None) -> list[Effect]:
        if subject is None:
            return self._rows("", ())
        return self._rows("WHERE subject = ?", (subject,))

    def _rows(self, where: str, params: tuple) -> list[Effect]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT effect_id, subject, tool, args_hash, request_id, status,"
                f" detail, at FROM outbox {where} ORDER BY at, effect_id",
                params,
            ).fetchall()
        return [Effect(*row) for row in rows]


def recorded_registry(
    registry: dict[str, Tool], outbox: Outbox, subject: str, request_id: str
) -> dict[str, Tool]:
    """Wrap every IRREVERSIBLE tool so the intent is durable before the call.

    At the seam, like the tracing and screening wrappers, and for the same
    reason: a rule that lives in each tool body is a rule the next tool will be
    written without. Read-only tools pass through untouched — an outbox row for
    a search is bookkeeping nobody will ever read.

    A duplicate reservation (`None`) means this exact call was already recorded
    for this request, which is a retry that got past the layers above. It is
    refused here rather than sent twice: at the last possible moment, with the
    arguments in hand, is the only place that check is unambiguous."""

    def wrap(tool: Tool) -> Tool:
        if not tool.requires_approval:
            return tool

        def recorded(*args: Any, **kwargs: Any) -> Any:
            effect_id = outbox.reserve(subject, tool.name, kwargs, request_id)
            if effect_id is None:
                return {"error": f"{tool.name} already recorded for this request"}
            try:
                result = tool.fn(*args, **kwargs)
            except Exception as exc:
                # A failure that is WRITTEN DOWN. Without this the row stays
                # pending forever and reconciliation cannot tell "crashed
                # mid-send" from "the API said no".
                outbox.settle(effect_id, FAILED, repr(exc))
                raise
            outbox.settle(effect_id, SENT, str(result))
            return result

        return rewrap(tool, recorded)

    return {name: wrap(tool) for name, tool in registry.items()}

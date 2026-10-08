"""Approval grants — one human "yes" authorizes one exact call, once.

The tempting implementation is a boolean (or a counter) keyed by tool name. It is
also wrong in four separate ways, and each one is a real incident:

    grants["send_telegram"] = True

  * **Anyone's yes works.** Alice approves; Bob's request finds the flag set and
    sends. The grant never named who it was for.
  * **Any arguments work.** The approval was for `{"chat_id": "team"}`; the flag
    also authorizes `{"chat_id": "the-press"}`.
  * **It never expires.** Last Tuesday's approval fires today's send.
  * **Two requests can spend it.** Check-then-execute-then-decrement is a
    read-modify-write; under concurrency both callers read `True`.

So a grant here is a *record*, not a flag: it names the subject, the tool, a
fingerprint of the exact arguments, and an expiry. Execution CONSUMES it with a
single `DELETE ... RETURNING`, which SQLite runs atomically — two concurrent
callers race for one row and exactly one wins. The loser sees no grant and pauses
for approval, which is the correct answer to "may I run this again?".

The store shares the ASSISTANT_DB file with memory, audit and idempotency, so an
outstanding approval survives a restart with everything else.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

# Long enough for a human to read a Slack message and click; short enough that a
# forgotten approval cannot authorize tomorrow's run.
DEFAULT_TTL_SECONDS = 300.0


def canonical_args(args: dict[str, Any] | None) -> str:
    """A stable text form of a call's arguments.

    Sorted keys and no incidental whitespace, so `{"a": 1, "b": 2}` and
    `{"b": 2, "a": 1}` fingerprint identically — the same call written two ways is
    the same call. `default=str` keeps a stray non-JSON value from raising inside
    a security check.
    """
    return json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)


def args_fingerprint(args: dict[str, Any] | None) -> str:
    """SHA-256 of the canonical form. Comparing hashes rather than dicts means the
    binding survives a round trip through JSON and is cheap to store and index."""
    return hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Grant:
    approval_id: str
    subject: str
    tool: str
    args_hash: str
    expires_at: float


class ApprovalStore:
    def __init__(self, path: str = ":memory:", ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        # one long-lived connection: ':memory:' would otherwise vanish per call
        self._conn = sqlite3.connect(path, check_same_thread=False)
        # TWO different problems, two different guards, and it is worth being
        # precise about which does what:
        #   * the single-statement claim in `consume` is atomic in SQLITE, which
        #     is what makes the grant safe across processes and connections;
        #   * this lock is atomic in PYTHON, protecting the shared Connection
        #     object. Threads interleaving statements on one connection raise
        #     "bad parameter or other API misuse" — a crash, not a data race, but
        #     under a load test it looks the same from the outside.
        # Dropping either one breaks a different test.
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS approvals ("
            "  approval_id TEXT PRIMARY KEY,"
            "  subject TEXT NOT NULL,"
            "  tool TEXT NOT NULL,"
            "  args_hash TEXT NOT NULL,"
            "  expires_at REAL NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS approvals_lookup  ON approvals (subject, tool, args_hash)"
        )
        self._conn.commit()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def mint(self, subject: str, tool: str, args: dict[str, Any] | None = None) -> Grant:
        """Record one approval for one exact call by one subject."""
        grant = Grant(
            approval_id=str(uuid.uuid4()),
            subject=subject,
            tool=tool,
            args_hash=args_fingerprint(args),
            expires_at=time.time() + self._ttl,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO approvals"
                " (approval_id, subject, tool, args_hash, expires_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (grant.approval_id, grant.subject, grant.tool, grant.args_hash, grant.expires_at),
            )
            self._conn.commit()
        return grant

    def consume(self, subject: str, tool: str, args: dict[str, Any] | None = None) -> str | None:
        """Atomically spend a matching grant; return its id, or None if there is
        no live grant for exactly this subject, tool and arguments.

        One statement does the whole check-and-take. Splitting it into SELECT then
        DELETE would reintroduce the race this class exists to close.
        """
        with self._lock:
            row = self._conn.execute(
                "DELETE FROM approvals WHERE approval_id = ("
                "  SELECT approval_id FROM approvals"
                "   WHERE subject = ? AND tool = ? AND args_hash = ? AND expires_at > ?"
                "   ORDER BY expires_at LIMIT 1"
                ") RETURNING approval_id",
                (subject, tool, args_fingerprint(args), time.time()),
            ).fetchone()
            self._conn.commit()
        return row[0] if row else None

    def outstanding(self, subject: str | None = None) -> list[Grant]:
        """Live (unexpired, unspent) grants — what a reviewer or /health would show."""
        query = (
            "SELECT approval_id, subject, tool, args_hash, expires_at FROM approvals"
            " WHERE expires_at > ?"
        )
        params: tuple = (time.time(),)
        if subject is not None:
            query += " AND subject = ?"
            params += (subject,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY expires_at", params).fetchall()
        return [Grant(*row) for row in rows]

    def purge_expired(self) -> int:
        """Housekeeping. Expired rows are already unusable — `consume` filters on
        `expires_at` — so this only reclaims space."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM approvals WHERE expires_at <= ?", (time.time(),))
            self._conn.commit()
        return cur.rowcount

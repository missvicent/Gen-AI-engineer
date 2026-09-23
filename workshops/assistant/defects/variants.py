"""The three defects this codebase actually shipped, preserved as running code.

These are not strawmen. Every one of them was in `after/` — reviewed, tested,
green — until an audit found it. They are kept here because a vulnerability you
have only read about is a vulnerability you will write again, and because a
regression test is only worth what it catches: the lab runs your test against
these first and requires it to FAIL.

Each variant is a drop-in replacement for a real component, matching its
interface exactly. That is deliberate. All three passed the test suite of their
day; none of them announces itself at the call site. The only thing separating a
correct `ApprovalStore` from `CountingApprovalStore` is a test that knows what
to ask.

Run the lab with `make defect-lab`.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from assistant.approvals import DEFAULT_TTL_SECONDS, Grant

# --- Defect 1: the approval that authorizes anyone, for anything, forever ------


class CountingApprovalStore:
    """The original approval store: a counter keyed by tool name.

    It reads perfectly. A human approved `send_telegram`, so a pending
    `send_telegram` may proceed — and the count means it can only happen once.
    Every word of that sentence is wrong in a way the code does not show you:

      * the grant never named a *subject*, so Alice's approval sends Bob's
        message;
      * it never named *arguments*, so an approval for `{"chat_id": "team"}`
        also sends to `{"chat_id": "the-press"}`;
      * it never *expires*, so last Tuesday's yes fires today;
      * `check` then `decrement` is a read-modify-write, so two concurrent
        requests both see `1` and both send.

    The lock below is what makes this so convincing. It looks like the author
    thought about concurrency. It protects the integer from corruption, which
    was never the risk — the risk is two callers making the same decision from
    the same value, and a mutex around each half of a read-modify-write does
    nothing about that.
    """

    def __init__(self, path: str = ":memory:", ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def mint(self, subject: str, tool: str, args: dict[str, Any] | None = None) -> Grant:
        with self._lock:
            self._counts[tool] = self._counts.get(tool, 0) + 1
        return Grant(
            approval_id=str(uuid.uuid4()),
            subject=subject,
            tool=tool,
            args_hash="",
            expires_at=time.time() + self._ttl,
        )

    def consume(self, subject: str, tool: str, args: dict[str, Any] | None = None) -> str | None:
        # Read...
        with self._lock:
            remaining = self._counts.get(tool, 0)
        if remaining <= 0:
            return None
        # ...decide, and only then write. Another thread runs in this gap.
        #
        # The sleep is the one line here that was not in the shipped code, and it
        # is not cheating. In production the gap held a database round trip, so
        # it was milliseconds wide and lost every race it entered. In-process
        # with nothing between the two statements, CPython's GIL usually carries
        # a thread through both, and the test passes perhaps nine runs in ten.
        # A flaky test that misses this defect nine times out of ten is worse
        # than no test: it is an alibi. So the gap is made explicit and the race
        # becomes deterministic — the bug is unchanged, only reliably visible.
        time.sleep(0.001)
        with self._lock:
            self._counts[tool] = remaining - 1
        return str(uuid.uuid4())

    def outstanding(self, subject: str | None = None) -> list[Grant]:
        with self._lock:
            return [
                Grant(str(uuid.uuid4()), subject or "?", tool, "", time.time() + self._ttl)
                for tool, count in self._counts.items()
                for _ in range(count)
            ]

    def purge_expired(self) -> int:
        return 0


# --- Defect 3: the image built without its lockfile ----------------------------

#: The Dockerfile as it shipped. `uv sync` with no lockfile and no `--frozen`
#: resolves dependencies fresh at build time, so the image you tested and the
#: image you deployed can contain different code — with nothing in the diff, the
#: logs or the tag to say so. The first symptom is a transitive dependency's
#: patch release breaking production on a rebuild of an unchanged commit.
UNLOCKED_DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml ./
RUN uv sync --group integration --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "--factory", "assistant.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Defect 2 needs no variant here: the vulnerable streaming path is still in the
# codebase as `output_gate.RAW`, kept and labelled local-only. That is the honest
# shape for it — the unsafe mode is a real option a hurried operator can switch
# on, so the regression test's job is to prove the DEFAULT is safe, not that the
# unsafe code was deleted.

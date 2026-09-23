"""Workshop 4 layer — the assistant's tools. Every external connection is a tool.

Read-only tools run freely; anything that sends/schedules is gated behind
requires_approval. Connectors are stubbed (email/news/telegram/calendar) so the
whole thing runs offline; swap the bodies for real APIs via env-var credentials.

The judgement to supply here is not the stub bodies — it's the SHAPE of the
registry: which tools are read-only and which are irreversible. Get that wrong
and the whole HITL containment story in agent.py has nothing to stand on.

Two fields exist for the planner rather than for the caller, and both matter:

**The docstring is an interface.** planner.py selects a tool by matching the
goal's words against the tool's name and description, exactly as a
function-calling model does. A docstring that says only what the tool *is*
("Send a Telegram message") gives a planner nothing to match on; one that also
says when to reach for it ("use to message, notify or tell the team") is
selectable. Writing docs for the planner is not decoration — it is wiring.

**`required_args` is a contract.** The planner will not propose a call it cannot
fully specify, so a tool must declare which arguments it cannot do without.
Derive them from the signature and the two cannot drift.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    requires_approval: bool
    doc: str
    #: parameters with no default — the planner must fill every one or skip the tool
    required_args: tuple[str, ...] = ()


def required_args_of(fn: Callable[..., Any]) -> tuple[str, ...]:
    """The parameters a callable cannot be invoked without.

    TODO 1: read these off `inspect.signature(fn)` rather than accepting a
    hand-written list, so a tool cannot declare one set and accept another.
    Count a parameter only when it has no default. `*args`/`**kwargs` carry no
    names, so skip them — a planner cannot fill what the function will not name.
    Callables with no introspectable signature should return () rather than
    raise: an un-plannable tool is a smaller problem than a crash at import.
    """
    raise NotImplementedError


def tool(fn: Callable[..., Any], *, requires_approval: bool) -> Tool:
    """Build a Tool from a function, taking name, doc and required args from it."""
    return Tool(
        name=fn.__name__,
        fn=fn,
        requires_approval=requires_approval,
        doc=inspect.getdoc(fn) or "",
        required_args=required_args_of(fn),
    )


def rewrap(original: Tool, fn: Callable[..., Any]) -> Tool:
    """The same tool with a different body — used by every wrapper in the stack
    (screening, tracing, real connectors). Copying the metadata explicitly is the
    point: a wrapper that drops `required_args` silently un-selects the tool."""
    return Tool(
        name=original.name,
        fn=fn,
        requires_approval=original.requires_approval,
        doc=original.doc,
        required_args=original.required_args,
    )


# --- stubbed connectors (read-only) ---
def read_emails(since: str = "today", limit: int = 20) -> list[dict]:
    """Read recent emails from the inbox (read-only).

    Use to check, read, list or summarize email, inbox or mail.
    """
    # TODO 2: return a small list of email dicts (from/subject/snippet), capped
    # at `limit`. A stub is fine — the point is that it is read-only.
    raise NotImplementedError


def read_news(url: str = "") -> str:
    """Fetch and return the text of a news page (read-only).

    Use to read the news, headlines or current articles.
    """
    # TODO 3: return some page text. Read-only; no side effects.
    raise NotImplementedError


# --- gated connectors (irreversible) ---
def send_telegram(chat_id: str, message: str) -> dict:
    """Send a Telegram message to a chat.

    Use when the goal is to message, notify, tell, alert or ping the team or a
    person. IRREVERSIBLE — requires approval.
    """
    # TODO 4: validate chat_id/message, then return a sent-confirmation dict.
    raise NotImplementedError


def schedule_event(title: str, start_iso: str, duration_min: int = 30) -> dict:
    """Create a calendar event.

    Use to schedule, book or add a meeting or appointment to the calendar.
    IRREVERSIBLE — requires approval.
    """
    # TODO 5: validate title/start_iso, then return a created-event dict.
    raise NotImplementedError


# TODO 6: register all four tools with `tool(fn, requires_approval=...)`.
# read_emails/read_news are read-only (False); send_telegram/schedule_event are
# irreversible (True). agent.run() reads this map to decide what to gate, and
# planner.py reads it to decide what can be chosen at all.
REGISTRY: dict[str, Tool] = {}

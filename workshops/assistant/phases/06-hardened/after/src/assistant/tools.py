"""Workshop 4 layer — the assistant's tools. Every external connection is a tool.

Read-only tools run freely; anything that sends/schedules is gated behind
requires_approval. Connectors are stubbed (email/news/telegram/calendar) so the
whole thing runs offline; swap the bodies for real APIs via env-var credentials.

Two fields exist for the planner rather than for the caller, and both matter:

**The docstring is an interface.** planner.py selects a tool by matching the
goal's words against the tool's name and description, exactly as a
function-calling model does. A docstring that says only what the tool *is*
("Send a Telegram message") gives a planner nothing to match on; one that also
says when to reach for it ("use to message, notify or tell the team") is
selectable. Writing docs for the planner is not decoration — it is wiring.

**`required_args` is a contract.** The planner will not propose a call it cannot
fully specify, so a tool must declare which arguments it cannot do without.
Deriving them from the signature keeps the two from drifting.
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

    Read off the signature so a tool cannot declare one set and accept another.
    `*args`/`**kwargs` carry no names, so they contribute nothing — which is the
    right answer: a planner cannot fill what the function will not name.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return ()
    return tuple(
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


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
    return [
        {"from": "boss@work.com", "subject": "Q3 recap", "snippet": "numbers look good"},
        {"from": "news@digest.com", "subject": "Daily", "snippet": "AI roundup inside"},
    ][:limit]


def read_news(url: str = "") -> str:
    """Fetch and return the text of a news page (read-only).

    Use to read the news, headlines or current articles.
    """
    return "Headline: local models keep getting smaller and faster. Details..."


# --- gated connectors (irreversible) ---
def send_telegram(chat_id: str, message: str) -> dict:
    """Send a Telegram message to a chat.

    Use when the goal is to message, notify, tell, alert or ping the team or a
    person. IRREVERSIBLE — requires approval.
    """
    if not chat_id or not message:
        return {"error": "chat_id and message required"}
    return {"sent": True, "chat_id": chat_id}


def schedule_event(title: str, start_iso: str, duration_min: int = 30) -> dict:
    """Create a calendar event.

    Use to schedule, book or add a meeting or appointment to the calendar.
    IRREVERSIBLE — requires approval.
    """
    if not title or not start_iso:
        return {"error": "title and start_iso required"}
    return {"created": title, "start": start_iso}


REGISTRY: dict[str, Tool] = {
    "read_emails": tool(read_emails, requires_approval=False),
    "read_news": tool(read_news, requires_approval=False),
    "send_telegram": tool(send_telegram, requires_approval=True),
    "schedule_event": tool(schedule_event, requires_approval=True),
}

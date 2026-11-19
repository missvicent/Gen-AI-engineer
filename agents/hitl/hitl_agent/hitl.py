"""Human-in-the-loop with a durable pause, built twice.

`Runner` (no framework):
- act(action, **args): reversible actions run now. Irreversible ones
  (send/pay/delete/schedule) become a PendingApproval, saved to `store`.
- resume(token, approved): pops the approval, saves, then runs it or rejects.
- The store is reloaded in __init__, so a pause survives a restart.

`approval_graph` (LangGraph): draft -> approve -> send -> END.
- `approve` holds only interrupt(). On resume LangGraph re-runs the node from
  its first line, so any effect above the interrupt would fire twice.
- `send` fires only when `approved` is True, a value that exists only after a
  human resumes with Command(resume=...).
- The checkpointer decides durability: MemorySaver lasts one process,
  SqliteSaver survives a restart. The graph is the same either way.

Token <-> thread_id, _save/__init__ <-> checkpointer, resume() <-> Command(resume).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

IRREVERSIBLE = {"send", "pay", "delete", "schedule"}


@dataclass
class PendingApproval:
    token: str
    action: str
    args: dict[str, Any]


class Runner:
    def __init__(
        self, tools: dict[str, Callable[..., Any]], store: str | Path = ".hitl.json"
    ):
        self.tools = tools
        self.store = Path(store)
        self._pending: dict[str, PendingApproval] = {}
        if self.store.exists():
            p = json.loads(self.store.read_text())
            self._pending = {
                token: PendingApproval(**value) for token, value in p.items()
            }

    def _save(self) -> None:
        data = {token: asdict(pending) for token, pending in self._pending.items()}
        self.store.write_text(json.dumps(data))

    def act(self, action: str, **args: Any) -> Any:
        if action not in IRREVERSIBLE:
            return self.tools[action](**args)
        else:
            token = str(uuid.uuid4())
            pending = PendingApproval(token, action, args)
            self._pending[token] = pending
            self._save()
            return pending

    def resume(self, token: str, approved: bool) -> Any:
        pending = self._pending.pop(token, None)
        if pending is None:
            return {"status": "unknown"}
        self._save()
        if not approved:
            return {"status": "rejected"}
        else:
            return self.tools[pending.action](**pending.args)


# --- the same pause, in real LangGraph ------------------------------------------


class State(TypedDict, total=False):
    topic: str
    email: str
    approved: bool


def approval_graph(checkpointer: Any, send: Callable[[str], None]) -> Any:
    """Build draft -> approve -> send -> END over `State`, compiled with `checkpointer`.

    - `draft` writes `email` ("re: <topic>") from `topic`.
    - `approve` only calls interrupt(), and returns the human's answer as
      `approved`. Nothing else lives here, because LangGraph re-runs this node
      from its first line on resume.
    - `send` calls the injected `send(email)` only when `approved` is True.

    `checkpointer` is passed in, not chosen here: MemorySaver pauses for one
    process, SqliteSaver survives a restart, and the graph is the same.
    `send` is injected so tests can pass a fake (e.g. `list.append`).

    langgraph is imported inside the function: it is an integration-group
    dependency, and this module must import without it.
    """
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    def draft(state):
        return {"email": f"re: {state['topic']}"}

    def approve(state):
        return {"approved": bool(interrupt("Send this email?"))}

    def send_node(state: State):
        if state["approved"]:
            send(state["email"])
        return {}

    g = StateGraph(State)
    g.set_entry_point("draft")
    g.add_node("draft", draft)
    g.add_node("approve", approve)
    g.add_node("send", send_node)
    g.add_edge("draft", "approve")
    g.add_edge("approve", "send")
    g.add_edge("send", END)
    app = g.compile(checkpointer)
    return app

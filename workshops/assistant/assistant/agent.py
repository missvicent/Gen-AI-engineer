"""Workshop 4 layer — the agent loop with HITL, built on the Phase-3 pattern.

The 'brain' (decide) is injected so tests are deterministic/offline. Gated tools
never fire without approval; the loop has a hard step cap.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from assistant.tools import Tool


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    is_final: bool = False
    answer: str = ""


@dataclass
class Pending:
    tool: str
    args: dict[str, Any]


@dataclass
class AgentResult:
    text: str = ""
    pending: Pending | None = None
    audit: list[str] = field(default_factory=list)
    fired_irreversible_tool_without_approval: bool = False
    # tool -> the id of the grant its execution spent, when running under a
    # consuming approval store rather than a plain allow-list
    approval_ids: dict[str, str] = field(default_factory=dict)
    # Every call that actually ran, with the arguments it ran with. `audit` says
    # the same thing in prose and is what a human reads; this is what the audit
    # LOG is built from, because a caller that has to parse "ran: " back out of a
    # sentence is one line of formatting away from recording nothing.
    calls: list[Step] = field(default_factory=list)


Decider = Callable[[str, list[tuple[Step, Any]]], Step]
# Given a tool name and the exact arguments about to be used, spend one grant and
# return its id — or None when no grant covers this call.
Consumer = Callable[[str, dict[str, Any]], "str | None"]


def run(goal: str, decide: Decider, approvals: dict[str, bool] | None = None,
        registry: dict[str, Tool] | None = None, max_steps: int = 8,
        consume: Consumer | None = None) -> AgentResult:
    # TODO: loop to max_steps; final step -> set text and return. Unknown tool ->
    # error observation, keep going. Executed tool -> audit "ran: <tool>"
    # (observe.py derives step_count from the audit list — skip this and a
    # successful multi-tool run reports zero steps).
    #
    # Gated tools take approval one of two ways:
    #   * `approvals` — a name -> bool allow-list. The teaching form: enough to
    #     show that a gated tool pauses.
    #   * `consume`   — the production form. Call it with the tool name and the
    #     EXACT args of the step about to run, right before running it. A returned
    #     id means a grant was spent (record it in result.approval_ids); None
    #     means pause. Taking it here, rather than reading a snapshot from before
    #     the loop, is what makes the grant bind to the real arguments and what
    #     stops two concurrent runs from spending the same approval.
    # Either way, no approval -> record a Pending, audit
    # "paused for approval: <tool>", return.
    raise NotImplementedError

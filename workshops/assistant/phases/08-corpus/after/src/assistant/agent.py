"""Workshop 4 layer — the agent loop with HITL, built on the Phase-3 pattern.

The 'brain' (decide) is injected so tests are deterministic/offline. Gated tools
never fire without approval; the loop has a hard step cap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from assistant.tools import REGISTRY, Tool


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


def run(
    goal: str,
    decide: Decider,
    approvals: dict[str, bool] | None = None,
    registry: dict[str, Tool] | None = None,
    max_steps: int = 8,
    consume: Consumer | None = None,
) -> AgentResult:
    """Run the loop. Gated tools need approval, expressed one of two ways.

    `approvals` is the teaching form: a name -> bool allow-list, enough to show
    that a gated tool pauses. `consume` is the production form the capstone wires
    in: it is called with the exact arguments at the moment of execution and
    spends a single-use grant bound to caller, tool and arguments. Prefer it in
    anything real — an allow-list checked before the loop cannot notice that the
    arguments changed, that the grant belonged to someone else, or that another
    request already spent it.
    """
    registry = registry or REGISTRY
    approvals = approvals or {}
    result = AgentResult()
    state: list[tuple[Step, Any]] = []
    for _ in range(max_steps):
        step = decide(goal, state)
        if step.is_final:
            result.text = step.answer
            return result
        tool = registry.get(step.tool)
        if tool is None:
            state.append((step, {"error": f"unknown tool {step.tool!r}"}))
            continue
        if tool.requires_approval:
            # Take the grant HERE, against the arguments actually about to run —
            # not from a snapshot taken before the loop started.
            approval_id = consume(step.tool, step.args) if consume is not None else None
            allowed = (
                approval_id is not None if consume is not None else approvals.get(step.tool, False)
            )
            if not allowed:
                # pause instead of firing — HITL containment
                result.pending = Pending(step.tool, step.args)
                result.audit.append(f"paused for approval: {step.tool}")
                return result
            if approval_id is not None:
                result.approval_ids[step.tool] = approval_id
        out = tool.fn(**step.args)
        # Every executed tool is audited, not only the pauses. observe.py reads
        # step_count off this list; skip the append and a successful multi-tool
        # run reports zero steps.
        result.audit.append(f"ran: {step.tool}")
        result.calls.append(step)
        state.append((step, out))
    result.text = "stopped: max_steps"
    return result

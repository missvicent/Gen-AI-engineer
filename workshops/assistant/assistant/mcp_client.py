"""Workshop 7 layer — the assistant gains tools from an MCP server by DISCOVERY.

The win condition: tools are listed at runtime and adapted into the agent's
registry, NOT hard-coded. Add a tool to the server, restart, the assistant can
use it. `discover` is injected (a fake in tests; a real ClientSession.list_tools
in production) so this runs offline.

The other half of that win condition is the uncomfortable one. A tool the
assistant gained without a code change is a tool nobody reviewed, described by
the party that wants it called. Gating is therefore decided HERE, from local
policy, and the server's annotations only ever get to make the answer stricter.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from assistant.tools import Tool


def gate(spec: dict, readonly_allowlist: Iterable[str] = ()) -> bool:
    """TODO: does this discovered tool need an approval before it can run?

    Default YES. The version this replaces was one expression long —
    `spec.get("requires_approval", False)` — and it read an absent key as
    permission, so a server ungated every tool it published by not mentioning
    the subject. Discovery is precisely the path by which tools arrive
    unreviewed; unreviewed and unrestricted must not be the same state.

    The rule to implement: **annotations inform policy, they never grant
    authority.**

      * a tool not in `readonly_allowlist` is gated, whatever it says about
        itself — `read_only` is an assertion by the party that benefits from it
        being believed, and nothing verifies it;
      * an allowlisted tool that declares `destructive` is gated anyway;
      * so is an allowlisted tool whose `read_only` is explicitly False — the
        operator and the server disagree, and disagreement resolves to caution.

    Absent hints are not the same as False hints. An allowlisted tool that says
    nothing about itself is fine: the operator reviewed it, and a server that
    declines to describe its tools does not get to overrule that either way.
    """
    raise NotImplementedError  # TODO


def as_agent_tools(
    discovered: list[dict],
    invoker: Callable[[str, dict], Any],
    readonly_allowlist: Iterable[str] = (),
) -> dict[str, Tool]:
    """Turn discovered MCP tool specs into agent Tools — no hand-coding per tool.

    Each spec: {"name", "description", "required_args"?, "read_only"?,
    "destructive"?}. The invoker actually calls the MCP server (call_tool) at
    runtime.

    `required_args` comes from the server's own input schema. Carrying it onto
    the Tool is what lets planner.py treat a discovered tool exactly like a
    built-in one: it can tell whether it is able to fill the call. Drop it and
    every discovered tool looks argument-free, which is worse than unusable.

    `requires_approval` comes from `gate` above, not from the spec.
    """
    raise NotImplementedError  # TODO
def extend_assistant(
    base_registry: dict[str, Tool],
    discovered: list[dict],
    invoker: Callable[[str, dict], Any],
    readonly_allowlist: Iterable[str] = (),
) -> dict[str, Tool]:
    """Merge discovered MCP tools into the assistant's existing toolbox."""
    raise NotImplementedError  # TODO

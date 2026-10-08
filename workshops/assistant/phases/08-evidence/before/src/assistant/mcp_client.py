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
    """Does this discovered tool need an approval before it can run?

    Default yes. That is the whole decision, and it used to be the opposite:
    `spec.get("requires_approval", False)` read an absent key as permission, so
    a server could ungate every tool it published by simply not mentioning the
    subject. Discovery is how tools arrive without review; "unreviewed" and
    "unrestricted" should not be the same state.

    **Annotations inform policy, they never grant authority.** A server claiming
    `readOnlyHint: true` stays gated unless the operator listed the tool locally
    — the claim is made by the party that benefits from it being believed. The
    hints do get to tighten: an allowlisted tool that announces itself as
    destructive, or declines to call itself read-only, is gated anyway. That
    asymmetry is the point. The only inputs that can OPEN a gate are local ones.
    """
    if spec["name"] not in set(readonly_allowlist):
        return True
    if spec.get("destructive"):
        return True
    # An allowlist entry says "this tool is a read". A server that will not say
    # the same thing about it has contradicted the operator, and disagreement
    # resolves to the cautious side.
    return spec.get("read_only") is False


def as_agent_tools(
    discovered: list[dict],
    invoker: Callable[[str, dict], Any],
    readonly_allowlist: Iterable[str] = (),
) -> dict[str, Tool]:
    """Turn discovered MCP tool specs into agent Tools — no hand-coding per tool.

    Each spec: {"name", "description", "required_args"?, "read_only"?,
    "destructive"?}. The invoker actually calls the MCP server (call_tool) at
    runtime.

    `required_args` comes from the server's own input schema, which is what lets
    planner.py treat a discovered tool exactly like a built-in one: it can tell
    whether it is able to fill the call.
    """
    allowlist = set(readonly_allowlist)
    registry: dict[str, Tool] = {}
    for spec in discovered:
        name = spec["name"]

        def make(n: str) -> Callable[..., Any]:
            def call(**kwargs: Any) -> Any:
                return invoker(n, kwargs)

            return call

        registry[name] = Tool(
            name=name,
            fn=make(name),
            requires_approval=gate(spec, allowlist),
            doc=spec.get("description", ""),
            required_args=tuple(spec.get("required_args", ())),
        )
    return registry


def extend_assistant(
    base_registry: dict[str, Tool],
    discovered: list[dict],
    invoker: Callable[[str, dict], Any],
    readonly_allowlist: Iterable[str] = (),
) -> dict[str, Tool]:
    """Merge discovered MCP tools into the assistant's existing toolbox."""
    merged = dict(base_registry)
    merged.update(as_agent_tools(discovered, invoker, readonly_allowlist))
    return merged

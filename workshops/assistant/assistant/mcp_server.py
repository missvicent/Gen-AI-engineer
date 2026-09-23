"""A small MCP server the assistant can discover tools from — the other end of
mcp_client.py. Reuses the phase-7 lesson-02 pattern (mcp v2 SDK, MCPServer),
kept to a few read-only tools so the capstone has a real server to talk to over
`make test-integration` and Docker Compose.

Run it directly for stdio (`python -m assistant.mcp_server`) or serve it over
streamable HTTP for the composed stack (see build_server().run(...)).
"""
from __future__ import annotations

from typing import Any

# Stubbed "reference" data so the server runs offline; swap for a real backend.
_FACTS = {
    "refund window": "Approved refunds are processed within five business days.",
    "escalation": "Payments above ten thousand euro need two approvers.",
    "support hours": "Support is staffed 09:00-18:00 UTC on weekdays.",
}


def _lookup_fact(topic: str) -> dict:
    """Find a fact by topic, tolerating how the caller phrased it.

    An exact-key lookup only works when the caller already knows the keys, which
    a planner filling this argument from a user's sentence does not. Matching on
    the fraction of a topic's own words that appear in the request keeps it
    deterministic while accepting "the refund window policy" for "refund window".
    """
    words = _normalise(topic)
    if not words:
        return {"error": "no topic given", "known": sorted(_FACTS)}
    scored = [
        (len(_normalise(key) & words) / len(_normalise(key)), key) for key in sorted(_FACTS)
    ]
    score, best = max(scored)
    if score < 1.0:  # every word of the topic must be present — no near-misses
        return {"error": f"no fact for {topic!r}", "known": sorted(_FACTS)}
    return {"topic": best, "fact": _FACTS[best]}


def _normalise(text: str) -> set[str]:
    """Words, with plurals folded, so "support hour" finds "support hours".
    Both sides go through this, so the server never has to know how the caller
    happened to phrase things."""
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in text.lower().replace("_", " ").split()
    }


def build_server() -> Any:
    """Build (not run) the MCP server so tests can drive it in-memory.

    Every tool here is annotated `read_only_hint=True`, which is true and which
    the client is right not to take our word for. The annotations exist so the
    propagation path is real end to end; what ungates a tool on the other side
    is the operator's `ASSISTANT_MCP_READONLY_ALLOWLIST`, and a server saying
    nice things about itself is not on it.
    """
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    mcp = MCPServer("assistant-knowledge")
    READ = ToolAnnotations(read_only_hint=True, destructive_hint=False)

    @mcp.tool(annotations=READ)
    def lookup_fact(topic: str) -> dict:
        """Look up a company fact by topic. Use for policy/how-does-X questions.

        Args:
            topic: a short topic key, e.g. "refund window".
        """
        return _lookup_fact(topic)

    @mcp.tool(annotations=READ)
    def list_topics() -> list[str]:
        """List the topics this knowledge server can answer. Use to discover scope."""
        return sorted(_FACTS)

    @mcp.tool(annotations=READ)
    def word_count(text: str) -> dict:
        """Count the words in a piece of text. A safe, read-only utility tool."""
        return {"words": len(text.split())}

    @mcp.resource("knowledge://topics")
    def topics() -> str:
        """Read-only: the topics this server knows about."""
        return ", ".join(sorted(_FACTS))

    return mcp


if __name__ == "__main__":
    import os
    import sys

    if "--http" in sys.argv:
        # The composed stack (phase8 lesson 1): serve streamable HTTP at /mcp.
        build_server().run(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104 — inside a container, the port stays on the compose network
            port=int(os.getenv("PORT", "8080")),
        )
    else:
        build_server().run()  # stdio, for a desktop host

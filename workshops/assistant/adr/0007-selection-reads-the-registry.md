# ADR-0007 — Tool selection reads the registry

**Status:** accepted

## Context

Workshop 7 discovers tools from an MCP server at boot and merges them into the
agent's registry. The planner that consumed that registry was a chain of `if`s
over two hardcoded names and their trigger phrases:

```python
if "email" in low and "read_emails" in registry:
    ...
if "message the team" in low and "send_telegram" in registry:
    ...
```

Discovery therefore worked and did nothing. `/health` reported
`tools: mcp+builtin`, `lookup_fact` was in the registry, and no question could
ever reach it. The capstone's claim — "add a tool to the server, restart, the
assistant can use it" — was true of the registry and false of the assistant.

The gap is easy to miss because both halves pass their own tests. The MCP tests
prove discovery; the service tests prove the two builtin tools are selected. The
property nobody was asserting is the one that spans them.

## Decision

Selection is a function of the goal and the registry, and nothing else.

Each tool advertises itself through its name and docstring. The goal is reduced
to content words, scored against each tool's vocabulary, and the best tool above
a threshold of two shared words wins, ties broken on name. A tool's required
arguments are read off its signature (or, for a discovered tool, off the
server's input schema) and filled from a table keyed by **parameter name**, so
a new tool taking a `topic` is callable the day it appears.

Two rules do the security work:

**Only the goal selects.** Contexts, tool output and memories are not parameters
of `choose`. A poisoned document cannot reach a tool call because there is no
path from the corpus to the planner — containment by construction rather than by
filtering. `test_only_the_goal_selects_a_tool_never_a_retrieved_document` pins
the boundary by showing the same words *do* select when they arrive as the goal.

**Never propose a call you cannot fully specify.** `schedule_event` needs a
start time; nothing in a sentence reliably supplies one; so the deterministic
planner declines to schedule. An honest gap beats a meeting at the wrong hour.

Docstrings become load-bearing under this design, which is a feature: it is the
same contract a function-calling model reads, so the discipline transfers.

## Alternatives considered

Let a model choose the tool (what a production system does — and what Workshop 6
covers — but it makes policy depend on model mood and puts the fast tier behind
a network call; the model's judgement belongs in composition, not in the gate).
Keep the `if` chain and extend it per discovered tool (defeats discovery). A
per-tool argument table instead of a per-parameter one (works, and every new
tool needs a code change here, which is the bug again in slower motion).
Embedding similarity between goal and description (better recall, needs an
embedding model in the fast tier, and non-determinism in the one place that
should be predictable).

## Consequences

Adding a tool means writing a docstring that says *when* to use it, not just
what it is; a tool described only by its noun will not be selected. Selection is
lexical, so a goal phrased entirely in synonyms of a tool's vocabulary misses —
acceptable, because a miss falls through to a grounded answer or an abstention
rather than to a wrong tool. Approval gating is untouched: registry-driven
selection changes who chooses, never what policy applies, and a discovered tool
marked `requires_approval` pauses exactly like a builtin one.

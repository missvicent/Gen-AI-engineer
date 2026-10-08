# Workshop · Your own MCP, used by the assistant  (ends Phase 7)

**Effort.** ~2 h of focused build time · +60 min for the integration tier · ~3.5 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

Build an MCP server for a service you care about (see `phase7-mcp/02-rest-to-mcp`),
then let the assistant gain its tools by **discovery** — not hard-coding.

The trap in this workshop is that the first four deliverables can all pass while
the last one silently fails. Discovery puts the tool in the registry; a planner
that knows two tool names by heart will never choose it. That was a real defect
in this capstone, and it is why the tool-selection deliverable is here rather
than in the agent workshop: discovery you cannot act on is a connectivity check.

## Deliverables

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

### Minimum
- [ ] `extend_assistant()` merges discovered MCP tools into the toolbox
- [ ] `planner.choose` selects from the REGISTRY, so adding a tool to the server
      makes it usable with NO assistant code change (prove it: add a tool, restart,
      ask a question in plain English, watch it run)

Those two together, and only together. Discovery on its own is a connectivity
check — the trap this whole workshop is built around.

### Full
- [ ] Discovered tools carry their `requires_approval` flag (gating still applies)
- [ ] Discovered tools carry their `required_args`, read off the server's input
      schema — a planner cannot call what it cannot fully specify
- [ ] The assistant can call a discovered tool through the agent loop
- [ ] Selection reads the goal only — a poisoned document cannot pick a tool

### Stretch
- [ ] A real remote MCP server with the auth its deployment needs (Phase 7)
- [ ] A second server discovered alongside the first, proving the registry is not
      quietly single-server

Implement `mcp_client.py` and `planner.py`. Tests: `tests/test_mcp.py`,
`tests/test_planner.py`. Design notes: [`adr/0007-selection-reads-the-registry.md`](adr/0007-selection-reads-the-registry.md).

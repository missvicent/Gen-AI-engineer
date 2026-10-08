# The evolving assistant — reference (every layer)

One personal assistant, built up across the course:

- **RAG service** (`rag.py`) — hybrid-retrieval RAG core the assistant can query.
- **Eval suite** (`evals.py`) — sliced golden set, injectable judge, calibration and the
  merge gate that proves the layer above it works. Every later layer plugs in here.
- **Personal assistant** (`tools.py`, `agent.py`) — the agent loop, tools (email/news/
  telegram/calendar), and HITL: gated tools never fire without approval.
- **Hardened assistant** (`guardrails.py`) — L1 screen (decode+scan+redact), spotlighting,
  output gate. Containment: a landed injection can't fire a gated tool.
- **Your own MCP** (`mcp_client.py`) — the assistant gains tools from an MCP server
  by **discovery** (not hard-coding); add a server tool, it's usable after restart.

```bash
make setup && make check     # lint + type + test, fully offline
```

Connectors are stubbed so everything runs with no API keys; swap the tool bodies
for real APIs using env-var credentials. Deploy it with `phase8-deploy/`.

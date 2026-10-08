# Workshop 7 — MCP discovery

Edit `src/assistant/mcp_client.py`, `src/assistant/planner.py`. Supplied, already finished: `mcp_server.py`.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_mcp.py::test_discovered_tools_are_added_by_discovery_not_hardcoding
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_discovered_tools_are_added_by_discovery_not_hardcoding` — `extend_assistant` still raises NotImplementedError. The test handed you a tool list and the registry never grew.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. Discovery adds whatever the server listed. A name written into the client is the test's failure.
2. gate() is local policy. A server annotation can force approval. It cannot clear approval. Only the operator's allowlist can.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/mcp_client.py ../after/src/assistant/mcp_client.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../06-hardened/before
```

That copies only the files the previous layer asked you to write.

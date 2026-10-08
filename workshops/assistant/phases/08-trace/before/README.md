# Workshop 8 — trace, tokens, cache

Edit `src/assistant/observe.py`, `src/assistant/cache.py`, `src/assistant/usage.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_observe.py
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_wrapping_the_registry_traces_every_tool_without_editing_any_tool` — The first result is often an ERROR in the `rec` fixture. `observe.recorder()` is still the stub, so the fixture cannot build a tracer.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. recorder() returns an in-memory tracer. Wrap the registry once; do not edit each tool.
2. A cached answer is allowed only when is_cacheable says so. A pending approval and a gated tool are not cacheable.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/observe.py ../after/src/assistant/observe.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../07-mcp/before
```

That copies only the files the previous layer asked you to write.

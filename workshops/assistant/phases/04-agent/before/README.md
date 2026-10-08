# Workshop 4 — agent loop

Edit `src/assistant/tools.py`, `src/assistant/agent.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_agent.py::test_readonly_tool_runs_and_finishes
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_readonly_tool_runs_and_finishes` — The loop in `agent.run` still raises NotImplementedError. A read-only tool never gets to run.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. Register tools with tool(fn, *, requires_approval=...), not a decorator. The docstring is what a later planner matches on.
2. A gated tool appends a pending step and stops. It must not call the function until the same step comes back with approval.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/tools.py ../after/src/assistant/tools.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../03-evals/before
```

That copies only the files the previous layer asked you to write.

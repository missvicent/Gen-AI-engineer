# Workshop 5 — memory and crew

Edit `src/assistant/memory.py`, `src/assistant/tenancy.py`, `src/assistant/crew.py`. Supplied, already finished: `sqlite_memory.py`.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_memory.py
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_a_remembered_fact_comes_back_with_its_source` — The first result is often an ERROR in the `memory` fixture, not a failed assert. The fixture called `write()`, and `write()` is still the stub.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. remember(turn, *, source) returns None for a turn that is not worth keeping. A fact without a source is refused.
2. TenantMemory gives each subject the store the factory builds for that subject. Recall never reads another subject's store.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/memory.py ../after/src/assistant/memory.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../04-agent/before
```

That copies only the files the previous layer asked you to write.

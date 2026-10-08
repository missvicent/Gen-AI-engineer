# Workshop 8 — deadline, retry, outbox

Edit `src/assistant/deadline.py`, `src/assistant/resilience.py`, `src/assistant/idempotency.py`, `src/assistant/outbox.py`, `src/assistant/approvals.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_resilience.py::test_a_bug_is_not_retried_three_times
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_a_bug_is_not_retried_three_times` — A TypeError was attempted more than once. `is_transient` has to say no for a bug. Timeouts and connection errors are the ones worth a second try.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. Timeouts, connection errors, and 429/503 are transient. A ValueError is a bug and must not be retried.
2. An approval is spent once. A replay of the same idempotency key returns the original result and does not grant a second run.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/deadline.py ../after/src/assistant/deadline.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../08-corpus/before
```

That copies only the files the previous layer asked you to write.

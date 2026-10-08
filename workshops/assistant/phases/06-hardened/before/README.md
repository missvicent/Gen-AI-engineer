# Workshop 6 — hardened assistant

Edit `src/assistant/guardrails.py`, `src/assistant/screening.py`, `src/assistant/guard.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_guardrails.py::test_screen_blocks_injection
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_screen_blocks_injection` — `screen` still raises NotImplementedError. The injection string reached the scanner and the scanner is the stub.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. Expand first (base64, percent-encoding, HTML entities), then squash, then scan the expanded text. A decode adds evidence; it does not rewrite the original.
2. The model guard can add a block. It cannot clear one. If the regex already refused, do not call the model.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/guardrails.py ../after/src/assistant/guardrails.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../05-memory/before
```

That copies only the files the previous layer asked you to write.

# Workshop 3 — eval suite

Edit `src/assistant/evals.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_evals.py::test_the_suite_scores_every_slice
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_the_suite_scores_every_slice` — `run_suite` still raises NotImplementedError. Scoring has not started, so the suite cannot report a slice yet.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. score_row fills faithfulness and context_recall. An abstention row is a string check: if the answer abstains, faithfulness is 1 and the judge is not called.
2. gate returns a list of problems. A missing required metric is a problem, the same as a score under the bar.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/evals.py ../after/src/assistant/evals.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../02-rag/before
```

That copies only the files the previous layer asked you to write.

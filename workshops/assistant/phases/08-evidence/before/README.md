# Workshop 8 — the portfolio page

Edit `src/assistant/report.py`. Supplied, already finished: `release.py`, `evidence.py`.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_report.py::test_build_portfolio_is_still_the_page
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_build_portfolio_is_still_the_page` — `build_portfolio` still raises NotImplementedError. The test asked for the page and the function is the stub. Counts on that page come from the meter, not from a number you type in.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. The page is a measurement of the run you just did. Counts come from the meter, not from a number typed into the template.
2. release.py and evidence.py are already finished. Your file is report.py.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/report.py ../after/src/assistant/report.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../08-auth/before
```

That copies only the files the previous layer asked you to write.

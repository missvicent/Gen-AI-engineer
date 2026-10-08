# Workshop 8 — corpus you can operate

Edit `src/assistant/adapters.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_corpus.py::test_reingesting_the_same_source_updates_instead_of_accumulating
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_reingesting_the_same_source_updates_instead_of_accumulating` — `InMemoryRag.add` still raises NotImplementedError. Re-ingest cannot update a document the store refuses to write.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. add() stores chunks by Chunk.id, so a second ingest of the same source overwrites. It does not append.
2. delete(source) removes every chunk of that source for that tenant and leaves every other tenant alone.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/adapters.py ../after/src/assistant/adapters.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../08-trace/before
```

That copies only the files the previous layer asked you to write.

# Workshop 2 — retrieval core

Edit `src/assistant/rag.py`. Supplied, already finished: `provenance.py`.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_rag.py::test_semantic_search
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_semantic_search` — `RagStore._rank` still raises NotImplementedError. The test asked the store a question and the ranking step is the one you have not written.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. Chunk.id is a UUIDv5 of tenant, source, and ordinal. The words in the chunk are not part of the id, so an edit keeps the id and changes the version.
2. _rank fuses two arms with RRF: add 1 / (k + rank) for each arm a chunk appears in. k is already named in the file.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/rag.py ../after/src/assistant/rag.py
```

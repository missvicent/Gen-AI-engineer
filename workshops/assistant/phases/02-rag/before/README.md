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

## Recall on the exact slice

`make recall` runs `scripts/recall.py`: the 8 `exact` questions from the golden set,
searched against the 10-doc corpus. A hit means a retrieved chunk's `source` is in
the row's `supporting_doc_ids`. Measured 2026-10-16.

Run it from this folder (`02-rag/before`):

```bash
make recall
```

- The target sets `PYTHONPATH=src` and uses the `release` dependency group, which
  has fastembed. The first run downloads the cross-encoder (~80 MB). After that it
  runs offline.
- Without fastembed, the script prints `Rerank score: n/a` and the other two arms
  still run.
- To measure a different k, change `K` at the top of `scripts/recall.py`.

| k | dense only | hybrid (BM25 + dense, RRF) | hybrid + rerank |
|---|---|---|---|
| 1 | 0.875 | **1.0** | **1.0** |
| 3 | 1.0 | 1.0 | 1.0 |

Measured after the section 4 tokenizer (NLTK `word_tokenize`, stopwords, Porter
stemming). Before it, k=1 was 0.875 / 0.875 / 1.0, and the two arms missed different
questions (g-009 for hybrid, g-012 for dense). Now hybrid gets every exact row and
dense misses one:

| question | dense top-1 | hybrid top-1 | correct |
|---|---|---|---|
| g-019 date invoice INV-88102 was issued | policy#currency ❌ | ap-handbook#credit-notes ✅ | ap-handbook#credit-notes |

Why the numbers moved:

- **Stemming** joins `rejection`/`rejected` (g-009) and `date`/`dated` (g-019), and
  `word_tokenize` splits the `?` off ids, so `inv-88231?` now matches `inv-88231`.
- **Stopwords alone made k=1 worse** (0.75 / 0.75): the old hits partly rode on
  shared `the`/`was`/`what`. Stemming recovered them.

Caveats:

- **k=3 is saturated.** With 10 docs, the top 3 is 30% of the corpus, so every arm
  scores 1.0. k=1 is the number that shows a difference.
- **Offline "dense" is bag-of-words cosine,** so it is lexical too. That is why dense
  and hybrid are so close. A real embedder (section 3) should widen the gap.

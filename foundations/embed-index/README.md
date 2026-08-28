# 1.5 Embed & index

**Goal.** Build a mini vector store — local embeddings, a numpy matrix, cosine top-k `search` — and learn the seam that matters: the embedder is injected, which is what makes the ranking testable without a model and swappable for BGE-M3 or Qdrant in Phase 2.
**Prerequisite.** 1.4 Chunking (chunks are what this index holds and searches).
**Effort.** ~30 min to green on the fast tests · no integration tier · ~50 min realistic first pass.

## Do this

```bash
make setup && make test     # 6 failing tests — read them, they are the spec
$EDITOR src/index.py        # normalize, embed, VectorIndex.build, VectorIndex.search
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_search_ranks_nearest_first` fails because `VectorIndex.build` isn't built. In a hand-made two-dimensional space (axis 0 is "payments", axis 1 is "weather"), it demands that a query sharing zero words with the payment chunks still ranks both of them above the weather one — cosine similarity over normalized vectors is what buys you that.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] Row i of the matrix stays aligned with chunk i — `test_build_keeps_chunks_and_vectors_aligned` pins it, because misalignment silently mislabels every result.
- [ ] `normalize` leaves a zero row as zeros instead of dividing by zero (`test_normalize_leaves_a_zero_row_alone_instead_of_dividing_by_zero`).

## Stuck?

1. Once every row is L2-normalized, cosine similarity is just `matrix @ query_vector` — one dot product, no division, no loop over rows.
2. `normalize`: divide each row by its norm, guarding zeros (e.g. `np.where(norms == 0, 1, norms)`). `search`: embed the query through `self.embedder`, dot it against `self.matrix`, sort descending with `np.argsort`, take the top `k` as `(chunk, score)` pairs.

No integration lane: the embedder is injected, so the fast tests prove `search` itself offline — only `embed` touches Ollama and no test calls it.

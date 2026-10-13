# Phase 2 RAG workshop: full-tier guide

You need the minimum tier green before you start this one; it is. Everything here lives in
`workshops/assistant/phases/02-rag/before/`. You edit `src/assistant/rag.py` mostly, plus
new tests and one script. This guide says what to build and how to prove it. It has no
solutions.

## 0. Fix the two blockers first

- [x] `chunk_document` used `range(0, min(len(body), step))`, so long documents were cut
      wrongly. Fixed: a 2,000-character document now gives 5 overlapping chunks covering
      the whole body.
- [x] `evals/golden-set/evals/golden.jsonl` would not parse. Fixed: 17 missing commas,
      2 trailing commas, 1 double comma. It now parses as 21 rows.
- [x] **Real JSONL.** One object per line, so `load_golden` can read it line by line.
- [x] **Slice name.** `multi-hop` → `multi_hop`, to match `SLICES` in `src/dataset.py`.
- [x] **Unanswerable rows** now have `supporting_doc_ids: []`. Whole-file names like
      `policy.md` are not corpus ids, so `test_supporting_docs_exist_in_the_corpus` failed.
- [x] **g-011** got `labeled_by: human`, `labeled_on: 2026-10-06`.
- [x] **g-021** reworded to "How long do we keep supplier contracts?". It is a near miss:
      the retention policy covers invoices and audit trails only.
- [x] **9 draft rows added** (g-022 to g-030). The golden set is now 30 rows: exact 8,
      semantic 7, unanswerable 6, multi_hop 5, adversarial 4.

**Your review:**

- [x] Every row with `labeled_by: "draft"` (g-021 to g-030): check the question and the
      ground truth against `corpus.jsonl`, then set `labeled_by: "human"`.
- [x] **Near-duplicates** flagged by the reference `validate()`: g-011 ~ g-013 ~ g-014.
      Keep one, or rewrite the others so they ask something different.
- [x] **g-016 ground truth looks wrong.** The corpus says reporting *does* convert at the
      invoice-date rate, so the answer to that question is "yes", not "no".
- [x] Typo in the unanswerable `source` notes: "recurrend" → "recurrent".

## 1. Citations carry provenance, and re-ingesting updates instead of duplicating

**Build.** `RagStore` can only be built through its constructor. Add an ingest method,
e.g. `add(docs, tenant=...)`, that:

- [x] accepts what `as_chunks` accepts: prose, a `{"text", "source"}` record, or a `Chunk`
- [x] stores chunks keyed by `chunk.id`, so a re-ingest overwrites (a dict keyed by id is
      the in-memory upsert)
- [x] drops leftover ordinals when a revision gets shorter (ADR-0012, "a short revision
      cleans up its own orphans")
- [x] rebuilds the BM25 statistics (`toks`, `df`, `n`, `avg`) after every change. They are
      only computed in `__init__` today.

**Prove it** in `tests/test_rag_chunks.py`:

- [x] ingest the same source twice: the chunk count does not change
- [x] ingest an edited body under the same source: same ids, different `version`
- [x] for a retrieved chunk, `cite()` has `source`, `version` and `offsets`, and
      `body[start:end] == text`

**Think about:** why the id leaves out the text, and what breaks if it doesn't. Guess
first, then check ADR-0012.

## 2. Hybrid + reranked, with a measured recall jump on the exact slice

**Build.**

- [ ] an optional reranker injected into `RagStore`: `rerank(query, chunks) -> chunks`.
      It reorders the fused top-N (fetch around 4×k, rerank, cut to k). None given means
      retrieval is unchanged, so the store stays offline by default.
- [ ] a script, e.g. `scripts/recall.py`, that loads `corpus.jsonl` as
      `{"text", "source": row["id"]}`, runs every `exact` question, and prints
      **recall@k** for dense only, hybrid, and hybrid + rerank. A hit means a retrieved
      chunk's `source` is in `supporting_doc_ids`.
- [ ] a way to run one arm alone (a `mode` flag, or call `_dense` from the script)
- [ ] the three numbers written down in the folder README

**Things you'll discover.**

- Check what `query.lower().split()` does to `"...credit note INV-88231?"`. The `?` stays
  attached, so the token never matches `inv-88231` in the corpus. That may be your biggest
  exact-slice loss, and no reranker fixes it.
- The offline "dense" arm is bag-of-words cosine, so it is lexical too. Hybrid can only
  beat it by a little here. A real embedder (section 3) widens the gap. Write both facts
  next to your numbers.

**fastembed reminder.** It is already in the `release` group: `uv sync --group release`.

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

enc = TextCrossEncoder(
    model_name="Xenova/ms-marco-MiniLM-L-6-v2"
)  # ~80 MB, downloads once
scores = list(
    enc.rerank(query, [c.text for c in chunks])
)  # one float per doc, same order
TextCrossEncoder.list_supported_models()  # e.g. BAAI/bge-reranker-base
```

- After the first download it runs offline. `HF_HUB_OFFLINE=1` proves it; `cache_dir=`
  pins where the model lives.
- Import it inside `try/except ImportError` and fall back to no reranking. ADR-0012:
  "reports a degradation and carries on".
- A cross-encoder scores query and passage together. That makes it more precise than two
  separately computed vectors, and too slow for the whole corpus. Run it only on the fused
  candidates.
- In tests, inject a tiny fake reranker ("boost chunks containing an exact query token").
  It checks the wiring without the model.

## 3. Runs offline; explain how Qdrant fits behind the same call

**Build:** nothing. Write 5 to 10 sentences in the folder README or a `NOTES.md`.

- [ ] what stays the same: `search(query, k)`, `retrieve(...) -> list[Chunk]`, `Chunk.id`
      as the point id, `cite()`
- [ ] what changes: real embeddings, a collection with named dense + sparse vectors,
      server-side RRF in place of your `_rank`, the tenant filter on **both** arms
- [ ] what new risk appears: vector search never returns zero results, so abstention needs
      a score floor (`ASSISTANT_MIN_SCORE`); an embedder swap needs a new collection

**Qdrant reminders**, if you want to try it:

- `QdrantClient(":memory:")` or `QdrantClient(path="./qdrant_storage")` runs Qdrant
  embedded, with no Docker.
- Point ids must be a UUID or an unsigned int. Your `uuid5` already is one, so `upsert`
  with the same id overwrites, like section 1.
- Hybrid query shape:
  `client.query_points(coll, prefetch=[Prefetch(query=dense, using="dense", filter=...), Prefetch(query=SparseVector(indices=..., values=...), using="sparse", filter=...)], query=FusionQuery(fusion=Fusion.RRF), limit=k)`
- `sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}` makes
  Qdrant do the IDF weighting.

Write your version first, then compare it with the amendment in
`adr/0012-retrieval-stores-chunks-with-derived-identity.md`.

## 4. Unanswerable questions abstain, proven in `tests/test_retrieval.py`

Create `before/tests/test_retrieval.py` with **store-level** checks only. It is not the
full-service file in `after/`.

- [ ] every `expects_abstention` row in the golden set: `retrieve(question)` returns `[]`
- [ ] an answerable question (e.g. `INV-88231`) still returns its chunk, so abstention did
      not break recall

**Things you'll discover.** The docstring says this store abstains "on its own" because
zero overlap means zero score. Now try *"What is our per-supplier budget cap for next
year?"*. Words like `what`, `is`, `our`, `for` are in the corpus, so the score is not zero.
You need stopword removal or a floor. Pick one and say why. A floor on an RRF score means
nothing (see `scored_by`), so decide which arm's score the floor applies to.

### Stopwords alone fix 1 of 6 rows

After removing stopwords, these content words still match the corpus:

| Row | Content words that still match |
|---|---|
| g-018 budget cap | none, so it abstains ✅ |
| g-017 / g-020 onboarding | `new`, `finance`, `team` |
| g-019 supplier migration | `payment`, `supplier` |
| g-021 supplier contracts | `supplier` |
| g-030 backup provider | `backup`, `payment`, `provider` |

Near-miss questions use the corpus's own words on purpose, so a lexical store always
scores them above zero. You need **stopwords plus a floor**:

- [ ] stopwords drop the noise words
- [ ] the floor decides when a match is too weak to count as evidence. Put it on one arm's
      raw score (BM25 or cosine), never on RRF, which only reflects rank and always exists.
- [ ] pick the number the way ADR-0012 picked 0.58: print the scores for answerable and
      unanswerable rows, then set the floor in the gap between them

### NLTK for tokenizing and stopwords

```python
# uv add nltk   (inside 02-rag/before)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")  # the tokenizer model
nltk.download("stopwords")  # the word list (198 English words)
STOP = frozenset(stopwords.words("english"))
```

`word_tokenize` fixes the `?` problem: `"INV-88231?"` becomes `['inv-88231', '?']`.
Watch for these (checked against this corpus):

- [ ] **Two downloads, not part of the package.** Without them:
      `LookupError: Resource 'punkt_tab' not found.` That breaks "runs offline" on a fresh
      machine or in CI. Write down how they get installed (e.g. a Makefile step). A pasted
      `frozenset` of ~30 words is the no-dependency alternative; note which you chose and why.
- [ ] **`isalnum()` deletes your ids.** The hyphen makes it return False:

      ```
      word_tokenize("credit note inv-88231?")  → [..., 'inv-88231', '?']
      filter with w.isalnum()                  → ['invoice', 'credited', 'credit', 'note']
      ```

      `inv-88231`, `err-4021` and `per-supplier` all disappear, and the exact slice with
      them. Drop only tokens that are pure punctuation; keep anything with a letter or digit.
- [ ] **Contractions split oddly.** `"We're"` becomes `['we', "'re"]`, `"Don't"` becomes
      `['do', "n't"]`. NLTK's list has `"don't"`, not `"n't"`, so the pieces survive unless
      your filter handles them.****
- [ ] **One tokenizer, both sides.** Put it in one helper and use it in
      `RagStore.__init__` (`self.toks`) **and** `_rank` (`q`). If index and query split
      differently, matches fail silently, the lexical version of an embedder mismatch.
- [ ] **No stemming.** `credited` and `credit` still don't match. NLTK's `PorterStemmer`
      is the next step if your recall numbers show that gap; abstention doesn't need it.

## Stretch (only if the full pass was easy)

- [ ] **Contextual retrieval:** ask Ollama for a one-sentence context for each chunk,
      prepend it, re-index, rerun the section 2 script on your weakest slice, and record
      the before and after numbers.
- [ ] **`docker compose up` with Qdrant + Ollama:** required in Workshop 8, not here.

## Done when

- [ ] `make test` passes, including the new tests from sections 1, 2 (fake reranker) and 4
- [ ] `scripts/recall.py` prints three exact-slice recall numbers, written in the README
- [ ] you wrote the Qdrant explanation from memory before reading ADR-0012

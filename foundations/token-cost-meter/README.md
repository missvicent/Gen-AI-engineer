# 1.2 Token & cost meter

**Goal.** Build a meter that counts tokens *before* you send (a vendor tokenizer)
and prices a call *after* from the returned `usage` object — so cost is a number
you measure, not a vibe.
**Prerequisite.** 1.1 Universal client (you have a `complete()` that returns usage).
**Effort.** ~30 min to green on the fast tests · no integration tier · ~50 min realistic first pass.

## Do this

```bash
make setup && make test     # 4 failing tests — read them, they are the spec
$EDITOR src/meter.py        # fill TODO 1 (count_openai) and TODO 2 (cost)
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_tiktoken_counts_something` fails because `count_openai` isn't built yet.
It's asking for a pre-flight count: run the text through tiktoken's `o200k_base`
encoding and return the length. That's the number you use to stay under a context
window *before* spending a request, not after.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] `cost(model, usage)` is computed from the `usage` object, and cached input
      tokens are billed cheaper than fresh ones (the test pins this).
- [ ] Output tokens cost more than input tokens for the same model.

## Stuck?

1. tiktoken's counter is OpenAI's; it *undercounts* Claude, so treat the pre-flight
   number as a floor, not the bill. The bill comes from `usage`.
2. Price per token = per-MTok rate ÷ 1_000_000. Cached input reads at roughly 10%
   of the input rate; add each usage field times its own rate rather than lumping
   input and output together.

No integration lane: pricing is arithmetic over a table, so the whole lesson runs
offline and free.

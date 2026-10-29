# 3.2 LLM-as-judge

**Goal.** Build an eval harness where the judge is injected, so every piece of grading logic — abstention handling, per-slice aggregation, instrument recording — is testable offline with a fake judge, and only the real, pinned RAGAS judge lives behind the opt-in tier.
**Prerequisite.** 3.1 — the golden set with slices and abstention flags is what this harness runs.
**Effort.** ~60 min to green on the fast tests · +25 min for the integration tier · ~100 min realistic first pass.

## Do this

```bash
make setup && make test     # 10 failing tests — read them, they are the spec
$EDITOR src/harness.py      # 7 TODOs: loader, abstention check, scoring, per-slice aggregation
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

Every failure traces back to the same root at first: `test_golden_set_loads_with_slices_and_abstention_flags` dies because `load_golden` raises `NotImplementedError`. The test to aim at is `test_slice_breakdown_exposes_what_the_average_hides` — it builds a pipeline whose abstain path is broken, then asserts the unanswerable slice reads 0.00 while the overall mean stays above 0.75. If you only compute an average, it fails, and preventing exactly that is what this lesson exists for.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] Abstention rows never reach the judge — `test_abstention_rows_never_reach_the_judge` scores one with an `ExplodingJudge` that raises if called.
- [ ] Results aggregate per slice as well as overall, with the judge's `describe()` recorded as the instrument (`test_slice_breakdown_exposes_what_the_average_hides`, `test_instrument_is_recorded_with_the_scores`).

## Stuck?

1. Implement `load_golden` and `is_abstention` first and rerun — half the failures move at once, and the remaining ones become readable.
2. `score_row` branches on `row.expects_abstention`: if set, score 1.0 everywhere when `is_abstention(answer)` and 0.0 when not, with `judged=False` and the judge never called; otherwise call `judge.faithfulness` and `judge.context_recall` with `judged=True`. `run_suite` then groups the `ScoredRow`s by slice and reuses `mean_scores` for each group.

## Going further (optional integration lane)
`make test-integration` runs two tests through the real RAGAS judge against `qwen3-coder:30b` on local Ollama — you'll need to fill the 4 TODOs in `src/ragas_judge.py` first (the pinned `ragas>=0.4,<0.5` + `langchain-community` pair installs with the `integration` dependency group). Needs Ollama running — the judge is free locally, but it pulls a large model. Skippable: the fast tier already proves the logic.

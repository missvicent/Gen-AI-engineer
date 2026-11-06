# CI regression gate

Turns an eval results file into an exit code. `gate.py` compares a run with a committed
baseline and fails the build when quality drops. No model runs here: the gate is pure
logic over two JSON files, so it is fast, free and deterministic.

```
evals/results.json  ──┐
                      ├──►  gate  ──►  diff table + exit 0 (pass) / 1 (fail) / 2 (usage)
evals/baseline.json ──┘
```

## Run it

```bash
make check  # ruff lint + format check + pyright type check
make test   # the gate's tests, offline, on tests/fixtures/
make gate   # exactly what CI runs: evals/results.json vs evals/baseline.json
```

`tests/fixtures/` and `evals/` hold the same files today, but they do different jobs.
Fixtures never change, so the tests stay stable. `evals/` is the data the gate checks,
and it changes when you re-baseline.

## The rules

Each rule is proved by a test that breaks the run on purpose. CI runs every one of
them on every PR, so the proof repeats itself; nothing here is checked by hand.

| Rule | What the test does | Gate says | Test |
|---|---|---|---|
| Pass | shipped run: faithfulness 0.912 → 0.897 (−0.015, inside the 0.03 tolerance) | `gate passed`, exit 0 | `test_the_shipped_run_passes_the_gate`, `test_main_exits_zero_on_a_passing_run` |
| Absolute bar | faithfulness and context_recall both 0.60, baseline also 0.60 (no movement) | below the bar of 0.85 / 0.80 | `test_an_absolute_bar_breach_fails_even_with_no_baseline_movement` |
| Regression | overall faithfulness 0.912 → 0.860 (−0.052) | `faithfulness regressed` | `test_a_dip_larger_than_the_tolerance_fails` |
| Collapsed slice | `unanswerable` 1.00 → 0.40 while overall stays 0.86, above the bar | `COLLAPSED` | `test_a_collapsed_slice_fails_while_the_average_still_looks_fine` |
| Missing slice | `multi_hop` removed from the run | `disappeared` | `test_a_disappearing_slice_is_a_failure_not_a_pass` |
| Stale baseline | judge `qwen3-coder:30b` → `some-other-model:70b`, or RAGAS 0.4.3 → 0.5.0 | `instrument changed … re-baseline`, comparison skipped | `test_a_changed_judge_is_a_re_baseline_not_a_comparison`, `test_a_changed_library_version_is_caught_too` |
| Tolerance edge | 0.90 → 0.87 (−0.030) passes, 0.90 → 0.869 (−0.031) fails | `regressed` only for the second | `test_a_drop_of_exactly_the_tolerance_passes_and_one_more_point_fails` |
| Exit codes | a failing run through the CLI; wrong arguments | exit 1 with every reason; exit 2 | `test_main_exits_one_and_prints_reasons_on_a_failing_run`, `test_main_rejects_wrong_usage` |

Numbers that drive the rules, in `ci_gate/gate.py`:

- `BARS`: faithfulness 0.85, context_recall 0.80.
- `TOLERANCE = 0.03`: the judge's noise floor, measured in lesson 3.3
  (`evals/judge-calibration`). A tighter gate fires on noise and gets ignored.
- `COLLAPSE_FLOOR = 0.5`: a slice falling from above this to below it is a collapse.

## CI

`.github/workflows/evals.yml` runs `make check`, `make test` and `make gate` on every
pull request and every push to `main`. No model, no API key. A ruleset on `main`
requires this check to pass before a PR can merge, so a red gate blocks the merge.

The job takes **12 s** (16 s for the whole run, first green run on 2026-11-06), well
under the ~2 minute limit past which people start skipping a gate.

The gate's diff table is also written to the run's job summary, so it shows on the run
page without opening the logs.

## Judged tier

The fast tier never calls a model. The judged tier does: a real RAGAS judge scores the
pipeline's answers, and the gate checks the fresh numbers.

```bash
make eval         # run the judge → evals/judged-results.json (ROWS=10 per slice)
make judged-gate  # evals/judged-results.json vs evals/judged-baseline.json
```

- **Judge:** `qwen2.5:7b-instruct` through Ollama, the same on a Mac and on the CI
  runner, so the instrument (model, URL, RAGAS version) matches and runs compare.
- **Rows:** 10 per slice. With 2, one flipped row moves a slice by 0.5, far past the
  0.03 tolerance, and the gate fails at random.
- **Separate files:** the fast tier keeps the course's shipped run and baseline; the
  judged tier has its own `judged-baseline.json`. One baseline for both would put
  toy-pipeline numbers into every PR's gate. `judged-results.json` is git-ignored: it
  is produced fresh on every run, never committed.
- **In CI:** the `judged` job runs only from the "Run workflow" button
  (`workflow_dispatch`). It installs Ollama on the runner, pulls the model, runs
  `make eval` and `make judged-gate`, writes the table to the job summary and uploads
  `judged-results.json` (kept 30 days). It costs nothing (public repo, open model),
  only time: a 7B model on a CPU runner, capped at 120 minutes. To make it nightly as
  the course says, add `schedule: - cron: "0 4 * * *"` (cron is UTC: 11 pm at UTC−5).
- **Expected verdict today:** fail on the absolute bars. The pipeline is a toy
  word-overlap retriever (`keyword_pipeline` in `evals/llm-judge/run_eval.py`), and it
  is not good enough. The gate saying so is the gate working. Swapping in the real
  retriever is Workshop 3.

A row the system should answer but where it retrieves nothing scores 0.0 without
calling the judge (RAGAS refuses empty contexts, and the answer is already known).

## Re-baselining

Re-baselining is a human decision, never automatic. A gate that updates its own
baseline never fails.

1. Run the judged eval to get a new `evals/results.json`.
2. Read the diff table.
3. Copy it to `evals/baseline.json` and commit it in its own PR, saying why the
   numbers moved.

After a judge or RAGAS change the gate refuses to compare and says "re-baseline": that
is the only time a re-baseline is expected.

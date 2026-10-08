# Workshop · Eval suite + CI gate  (ends Phase 3)

**Effort.** ~2 h of focused build time · +30 min for the integration tier · ~3 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

The previous layer gave the assistant a retrieval core that answers questions. This
one answers the only question that matters next: **how do you know it's any good?**

Build the eval layer the rest of the course plugs into — a sliced golden set over the
assistant's corpus, an injectable judge, a calibration check against your own labels,
and a gate that fails the build when a slice regresses.

## Architecture

```
golden rows ──► run_suite(rows, answer_fn, judge) ──► SuiteResult
                     │                                  │ overall
                     │ abstention rows: judge-free       │ by_slice
                     ▼                                   ▼
              KeywordJudge (offline)              gate(result, baseline)
              RagasJudge  (nightly)                  └─► [] or reasons → exit 1
```

## The seam

`before/src/assistant/evals.py` — `Judge` is a protocol, so the suite runs offline
against `KeywordJudge` and nightly against a real one. `run_suite` and `gate` are pure
functions over data: no I/O, no globals, fully testable.

## Deliverables

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

### Minimum
- [ ] Golden rows cover **all five slices** — `semantic`, `exact`, `multi_hop`,
      `unanswerable`, `adversarial` — with at least 5 unanswerable
- [ ] `run_suite` reports **overall and per-slice** scores; abstention rows are scored
      **without** the judge (`judged=False`)

### Full
- [ ] An invented answer to an unanswerable question drives that slice to 0.0
- [ ] `gate()` fails on an absolute-bar breach **and** on a per-slice regression
      against the baseline, and marks a collapse
- [ ] `cohen_kappa` returns 0.0 for a rubber-stamp judge that scores 0.9 on raw
      agreement — you can explain why that matters
- [ ] **Trajectory checks score the run, not just the answer**: `tools_run` reads the
      agent's audit trail back as a trace, `tool_choice_f1` and `goal_completion`
      score it against a reference plan, and `containment_ok` proves a gated tool
      never fired without approval — keep that one green through the hardening
      workshop
- [ ] The whole suite runs with **no model and no network**, in seconds

## Stretch goals

- Swap `KeywordJudge` for the RAGAS judge from `phase3-evals/02-llm-judge` behind the
  same protocol, and compare the two on the same rows.
- Hand-label 30 rows and record the kappa you would quote alongside your scores.

Implement `evals.py`. Reference and tests: `after/src/assistant/evals.py`,
`after/tests/test_evals.py`.

**You are allowed to fail your own gate.** If an honest golden set puts the retrieval
layer below the bar, do not move the bar — write the number down, fix the retrieval,
and watch it move. A metric that can't deliver bad news isn't a metric.

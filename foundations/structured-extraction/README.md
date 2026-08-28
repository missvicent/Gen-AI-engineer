# 1.3 Structured extraction

**Goal.** Extract a typed `Invoice` from messy text with `response_model` — a decoder-enforced constraint, not a "reply with JSON" plea — then batch it and measure the two numbers the hosted-vs-laptop shoot-out actually needs: the violation rate, and per-field accuracy against answers you labelled yourself.
**Prerequisite.** 1.1 Universal client (you built `extract` and checked schemas by hand; Instructor is that idea productized).
**Effort.** ~40 min to green on the fast tests · no integration tier · ~65 min realistic first pass.

## Do this

```bash
make setup && make test     # 8 failing tests — read them, they are the spec
$EDITOR src/extract.py      # build_client → extract_invoice → extract_all →
                            # violation_rate → field_accuracy → compare_providers
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_extraction_asks_for_the_schema_not_for_json_in_prose` fails because `extract_invoice` isn't built. It asserts that the client received `response_model=Invoice` — the schema must travel as a constraint in the call, not as a suggestion in the prompt. Taking the client as a parameter is what makes that checkable without a live model.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] A text the model cannot shape is counted, not crashed — `test_a_case_the_model_cannot_shape_is_counted_not_crashed` pins the `(invoices, violations)` split and the 0.5 rate.
- [ ] `violation_rate([], [])` returns `0.0` instead of dividing by zero.
- [ ] `field_accuracy` scores per field and tolerates a cent of float drift — `test_a_valid_object_can_still_be_wrong` is a run with a 0% violation rate and two of three dates wrong, which is the whole reason the second number exists.
- [ ] `compare_providers` runs the comparison over any two clients, so `test_the_comparison_runs_over_any_two_clients` proves the procedure offline with fakes. A row the provider failed on counts as wrong rather than vanishing from the denominator.

## Stuck?

1. The `Invoice` model is already written — notice its constraints (`min_length`, `ge`) live in the type, not the prompt. Your job is plumbing: get that model into the call, and the failures into a list instead of up the stack.
2. `build_client` wraps an OpenAI-compatible client with Instructor (`instructor.from_openai(...)`, JSON mode for Ollama) and returns its `.chat.completions`; `extract_invoice` passes `response_model=Invoice`; `extract_all` catches per-text exceptions and appends the text to `violations`.
3. `field_accuracy` loops `Invoice.model_fields` and divides hits by `len(expected)`; `compare_providers` extracts one text at a time so it can append `None` for a failure and keep the answers aligned with `expected` — `extract_all`'s shorter list is the right shape for a rate and the wrong one for an accuracy.

No integration lane: the extractor is injected, so the fast tests exercise the real extraction path against a fake — the whole lesson runs offline. Then run the shoot-out for real: label 20 of your own invoices, pass `{"gpt": build_client("gpt"), "local": build_client("local")}` to `compare_providers`, and keep the accuracy row. Phase 5 tiers on it.

# Judge calibration

Measures how far the LLM judge can be trusted by comparing its faithfulness scores
with human `pass`/`fail` labels.

**Labeling rule:** pass means every claim in the answer is supported by the retrieved
chunks.

Edge cases:

- The chunks imply the claim but don't state it ("second approver" means two
  approvers): `<pass or fail>`
- The chunks state the rule for a different thing (the ERR-4021 rule for an ERR-4090
  question): `<pass or fail>`
- The answer says the chunks don't contain the information, and that's true:
  `<pass or fail>`
- The answer leaves out a detail but says nothing false: `<pass or fail>`

## Run it

```
make report      # the course's 40 rows (evals/labeled.jsonl)
make my-report   # my 35 rows (evals/my_rows.jsonl)
make gate        # lint, tests and the course report
```

## The numbers I quote

Measured on my 35 rows, built from the golden set with `RagasJudge.faithfulness`.

| Number | Value |
|---|---|
| Cohen's κ | 0.82 |
| Agreement | 0.94 |
| Threshold | 0.50 |
| Regression tolerance | 0.01 |
| Labels | 29 pass, 6 fail |

The report prints 0.05 as the best threshold. κ is the same from 0.05 to 0.50 and
`best_threshold` returns the lowest tied value, so I quote 0.50.

## Read this before trusting the κ

**The 0.82 is too optimistic.** My first labels gave κ 0.46, agreement 0.80 and
tolerance 0.03, which does not pass the 0.60 gate.

I then read the 7 disagreements with the judge's scores visible and changed 5 labels.
All 5 moved toward the judge:

| Row | Before | After | Judge score |
|---|---|---|---|
| g-022 | pass | fail | 0.0 |
| g-027 | fail | pass | 0.667 |
| g-030 | fail | pass | 1.0 |
| g-037 | fail | pass | 1.0 |
| g-039 | fail | pass | 1.0 |

Other limits:

- Labels are not fully independent: an AI assistant reviewed my labels and suggested
  fails before the first κ, and I saw the scores before the changes above.
- g-027, g-030, g-037 and g-039 were failed because the chunks don't state what the
  answer claims. Passing them means the labels no longer apply that strict reading.
- 35 rows with 6 fails is a small sample.

## Remaining disagreements

Both are rows I passed and the judge scored 0.0. I recorded the cause as a bad rule:
my label is right and the judge is wrong.

| Row | Question | Answer |
|---|---|---|
| g-036 | What is the email the suppliers can use to send new bank details? | The context does not provide this |
| g-049 | How many seconds must pass before the gateway times out and raises ERR-4021? | 30 seconds |

## Decision

The report says the judge can gate merges, but because of the limits above, I use it
only as a regression signal for now: a drop larger than the tolerance is a reason to
investigate, and it does not block a merge by itself.

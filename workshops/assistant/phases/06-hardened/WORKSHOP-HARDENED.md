# Workshop · Harden the assistant  (ends Phase 6)

**Effort.** ~3.5 h of focused build time · +60 min for the integration tier · ~6 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

Armor the assistant against the attack catalog. The bar is **containment**: a
landed injection may produce junk text, but must never fire a gated tool or leak PII.

## Deliverables

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really this, and a place to stop that is not quitting. **Full** is the version you
would show someone. **Stretch** is for when the full pass came easily.

### Minimum
- [ ] `screen()` runs on every input AND every fetched email/news page
- [ ] Untrusted content is `spotlight()`-ed as data, not instructions
- [ ] An indirect injection in an email cannot fire a gated tool without approval

### Full
- [ ] The scan **expands** before it reads — base64, percent-encoding and HTML
      entities are all appended and scanned, never substituted for the original
- [ ] The scan **squashes** before it reads — `1gn0re`, `i g n o r e`,
      `ig<U+200B>nore` and `ｉｇｎｏｒｅ` all reach the same detector
- [ ] Squashing does not invent matches: benign prose whose words happen to
      collide after separator removal is still allowed through
- [ ] PII is redacted on input and blocked on output (`output_ok`)
- [ ] Documents are screened at **ingest**, not only at retrieval — a poisoned
      page is never written, PII is never stored, and the caller is told how
      many rows were refused

### Full — a model in the loop

Regexes catch shapes; a model reads for intent. `ASSISTANT_GUARD_MODEL` turns on
a second opinion in `guard.py`, and the only interesting question is which way
the wiring points: the model may **add** a block and may never clear one,
because the text it is reviewing is the adversary's own input. It also fails
open — a dead Ollama must not take the service down, since containment does not
depend on this layer.

Full rather than stretch, and the workbook says the same: the wiring is the
lesson. A guard that can overturn a deterministic block is the mistake this
layer exists to teach you not to make, and you do not learn it by skipping the
layer. The *model* is optional — `build_screen` returns the deterministic screen
untouched when none is configured, and every test here runs offline.

- [ ] `build_screen` returns `guardrails.screen` itself when no guard is configured
- [ ] The guard sees spotlighted input and is matched against one exact verdict token
- [ ] The guard covers every untrusted channel (question, retrieved docs, tool
      output, ingested docs), because it is injected once at the composition root
- [ ] `/health` reports which screen is in front of the caller

### Full — the evidence a machine can read

Containment is a claim, and a claim nothing checks is a sentence. This is the
half of the workshop that turns it into a file, and it is the half the workbook
grades: the same deliverables appear on the Phase 6 workshop card.

- [ ] The red-team suite runs in **CI** over the versioned Phase 6 dataset —
      *one* dataset, `phase6-design-defend/01-red-team/after/evals/redteam.jsonl`,
      resolved by `release.redteam_path()`. A copy under this workshop would drift
      from the lesson that maintains it and both files would still parse
- [ ] Every row rides **the channel it declares**: `retrieved` for a poisoned
      document, `tool_outputs` for a poisoned connector reply. Handing
      `guarded_run` only the row's `input` runs the suite with those attacks never
      delivered — they all "pass", and the number you publish is of a test that
      did not happen
- [ ] `run_redteam` **stops the run** on a row whose payload reached no boundary.
      An undelivered attack is a failed measurement, not a contained one, and the
      only safe thing to do with it is refuse to score it
- [ ] `release.safety_object` writes the whole containment object into
      `release-report.json` under `safety`: the `dataset` it came from (version +
      row count), `attacks` and `bypasses`, `controls` and `controls_refused`,
      `pii_leaks`, `undelivered`, the `gated_tools` your scoring actually used,
      a count per delivery `channel`, and per attack family how many of its rows
      were contained
- [ ] **Bypasses alone cannot be read.** Refusing every input scores a perfect
      zero; so does an attack that was never delivered; an empty suite scores best
      of all. That is why the object carries controls, delivery and families
      beside the headline, and why partial publication is not an option — Phase 8's
      `containment_ok` in `phase8-deploy/02-ci` reads every field as a merge gate

### Stretch

- [ ] The dual-LLM pattern: a quarantined model with no tools summarizes
      untrusted pages, a privileged model plans actions
- [ ] Run NVIDIA garak against the assistant and add whatever it finds to your suite
- [ ] An audit log: every blocked attempt and every approval, with a timestamp
      and a reason

Implement `guardrails.py`, `screening.py`, `guard.py` and `Assistant.ingest`, and
the red-team scoring in `release.py` for the evidence pass.
Tests: `tests/test_guardrails.py`, `tests/test_guard.py`, `tests/test_security.py`,
`tests/test_release.py` (the scoring and the containment object).
Decision record: [`adr/0010`](adr/0010-the-screen-expands-squashes-and-may-ask-a-model.md).

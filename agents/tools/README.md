# 4.2 Tools

**Goal.** Build three tools to spec — read-only, reversible, and gated-irreversible —
where the docstring is the model's only interface, every argument is validated, and
errors come back as data. The irreversible one teaches the rule you can't bend:
approval lives in application state, never in the tool signature.
**Prerequisite.** 4.1 ReAct from scratch (you have a loop that calls tools).
**Effort.** ~30 min to green on the fast tests · no integration tier · ~50 min realistic first pass.

## Do this

```bash
make setup && make test     # 5 failing tests — read them, they are the spec
$EDITOR src/tools.py        # fill TODOs 1-3: read_note, draft_reply, delete_note
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_read_validates_and_returns_data` fails because `read_note` isn't built yet.
It wants both halves of the contract: a bad id (empty string, unknown note) returns
`{"error": ...}` as data — no exception — and a good id returns `{"text": ...}`.
That error-as-data shape is what lets an agent loop recover instead of crash.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] `delete_note` refuses until the application has called `grant_approval(note_id)`,
      and one approval buys exactly one delete — the record is consumed (both pinned
      by tests).
- [ ] Every tool has a real docstring saying what it does and when to use it — the
      docstring-length test enforces this because it is the model's interface.

## Stuck?

1. All three tools return dicts, never raise. Validate `note_id` first; look notes
   up in `_NOTES`; `draft_reply` composes text and sends nothing.
2. For `delete_note`, check membership in `_APPROVALS` and remove the id on success.
   Do NOT add an `approve` parameter — the model fills every parameter in a tool
   signature, so that would be a gate the model can open itself. One test inspects
   the signature to make sure you didn't.

No integration lane: the tools run against an in-memory note store, so the whole
lesson is pure Python — offline and free.

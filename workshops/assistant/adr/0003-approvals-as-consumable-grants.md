# ADR-0003 — Approvals as consumable grants with idempotency keys

**Status:** accepted (revised — see *History*)

## Context

An irreversible tool (send a message, delete a file) must not fire without a
human decision — and the decision machinery itself must not create new ways to
fire twice. Four failure modes shape this: a client retrying `/approve` after a
timeout; an approval that silently stays valid forever; an approval that one
caller obtains and *another* caller spends; and an approval for one call that
ends up authorizing a different one.

## Decision

An approval is a **record**, not a flag. `/approve` mints a row naming the
authenticated `subject`, the `tool`, a SHA-256 fingerprint of the canonical
arguments, an `approval_id`, and an `expires_at` (`ASSISTANT_APPROVAL_TTL`,
300s default). The agent loop **consumes** it at the moment of execution,
against the arguments actually about to run, with a single
`DELETE ... RETURNING` — one statement, so SQLite serializes the claim and two
concurrent requests cannot both win it.

Four properties follow directly, each with a regression test:

| Property | Enforced by |
| --- | --- |
| only the approver can spend it | `subject` is taken from the verified token, never the body |
| only the approved call can spend it | `args_hash` must match the step's canonical arguments |
| it expires | `expires_at > now` is part of the claim predicate |
| exactly one execution | `DELETE ... RETURNING` is atomic; the loser sees no grant |

Omitting `args` from `/approve` fails **closed**: an empty-argument grant only
matches a call that takes no arguments. A retried `/approve` carrying the same
`Idempotency-Key` (namespaced by subject) is recognized by the SQLite dedupe
table and does not mint a second grant. The `approval_id` is written to the
`/approve` response, the `agent.run` span (`agent.approval_ids`) and the audit
row for the execution that spent it, so "who authorized this send?" is a lookup.

## Alternatives considered

Boolean or counter approvals keyed by tool name — the original decision here, and
wrong in all four ways above; it survived because the tests only exercised one
caller, one argument set, and one thread at a time. Approving by conversational
text ("yes go ahead") — indistinguishable from an injected approval, see
ADR-0002. `SELECT` then `DELETE` — reintroduces the race the class exists to
close. A full workflow engine with pending-task queues — right for multi-approver
processes, out of proportion here and it would bury the mechanism being taught.

## Consequences

"Approved once" provably means "this caller, this call, fires once"
(`test_reliability.py`, `test_security.py`, e2e check 6). Costs, all accepted:
a legitimate second execution needs a second explicit approval; the client must
echo back the arguments it is approving, so an approval UI has to *show* the
call rather than the tool name; and a grant left unspent past its TTL is gone.

## History

Superseded the counter-based design after the August 2026 audit demonstrated
cross-subject reuse, argument substitution and a concurrent double-spend against
it. The defect is preserved as coursework in the Phase 8 defect lab rather than
deleted from the record.

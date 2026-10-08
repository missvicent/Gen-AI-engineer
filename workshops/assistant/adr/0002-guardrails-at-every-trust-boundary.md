# ADR-0002 — Guardrails at every trust boundary, spotlighting over trust

**Status:** accepted

## Context

Injection does not arrive only through the front door. The red-team suite lands
attacks through the question, through retrieved documents (a poisoned page in the
corpus), and through tool output (a scraped feed that carries instructions). A
single input filter contains exactly one of those three.

## Decision

Screen at every boundary where content changes trust level: `guardrails.screen`
on the question, `screen_contexts` on every retrieved document before it becomes
evidence (poisoned docs are dropped, PII redacted), `harden_registry` on every
tool's output, and `guardrails.output_ok` on the final answer. Evidence handed to
the model tier is spotlighted — wrapped as data, never as instructions — because
screened is not the same as trusted. Approval claims inside content are ignored
by construction: approval is state (`grants`), not text.

## Alternatives considered

A moderation-model call per boundary (network latency and a key in the fast tier,
and it still misses the approval-in-text trick); trusting the corpus because "we
ingested it" (the poisoned-doc e2e check exists precisely because this fails);
prompt-only defenses (the bakeoff showed models follow injected instructions that
survive into context).

## Consequences

Four screening points cost microseconds each and are individually tested
(`test_security.py`, red-team v2 categories including multilingual and
exfiltration). Cost: patterns need maintenance as attacks evolve — the versioned
red-team dataset is the regression harness for that maintenance.

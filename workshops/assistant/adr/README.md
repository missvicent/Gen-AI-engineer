# Architecture Decision Records

Thirteen decisions that shape the capstone, each with the alternatives it beat and
the cost it accepted. Read them with [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
open — the diagrams show *what*, these say *why*.

| ADR | Decision |
|---|---|
| [0001](0001-ports-and-adapters-tiered-by-env.md) | Ports and adapters, tiered by environment variables |
| [0002](0002-guardrails-at-every-trust-boundary.md) | Guardrails at every trust boundary, spotlighting over trust |
| [0003](0003-approvals-as-consumable-grants.md) | Approvals as consumable grants with idempotency keys |
| [0004](0004-otel-spans-as-the-observability-currency.md) | OpenTelemetry spans as the only observability currency |
| [0005](0005-sqlite-for-state.md) | SQLite for memory, audit and idempotency — one file, not three services |
| [0006](0006-holdback-window-on-the-output-stream.md) | A holdback window on the output stream, so streamed text is screened before release |
| [0007](0007-selection-reads-the-registry.md) | Tool selection reads the registry, so a discovered tool can actually be chosen |
| [0008](0008-memory-is-partitioned-by-subject.md) | Memory is partitioned by subject, not labelled with one |
| [0009](0009-the-gate-requires-claims-and-the-issuer-is-pluggable.md) | The gate requires claims, and the issuer is pluggable (HS256 or JWKS) |
| [0010](0010-the-screen-expands-squashes-and-may-ask-a-model.md) | The screen expands and squashes before it scans, screens at ingest, and may ask a model |
| [0011](0011-one-trace-per-request-and-stamps-that-derive.md) | One trace per request, and version stamps derived from what they describe |
| [0012](0012-retrieval-stores-chunks-with-derived-identity.md) | Retrieval stores chunks whose identity is derived, so re-ingest updates and citations resolve |
| [0013](0013-one-budget-per-request-and-intent-before-effect.md) | One budget per request, and irreversible intent written down before the effect |

The format is deliberately short (context → decision → alternatives →
consequences). An ADR nobody reads is a decision nobody can challenge; these fit
on one screen each so they actually get read.

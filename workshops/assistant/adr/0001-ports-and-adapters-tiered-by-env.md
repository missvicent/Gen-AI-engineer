# ADR-0001 — Ports and adapters, tiered by environment variables

**Status:** accepted

## Context

The capstone must be two things at once: a fast, deterministic, zero-key codebase a
student can test on any laptop, and a real service that talks to Qdrant, Ollama, an
MCP server and a collector. Committing to either one alone fails the other audience.

## Decision

Every external dependency sits behind a port (a small Protocol or callable
signature). `settings.py` reads the environment once; `build_assistant` picks the
adapter per port: `QDRANT_URL` → Qdrant else BM25, `OLLAMA_HOST` → Ollama else the
offline stitcher, `MCP_SERVER` → discovered tools else builtins, `ASSISTANT_DB` →
SQLite else in-process, `OTEL_EXPORTER_OTLP_ENDPOINT` → OTLP else in-memory only.
The composition root (`service.py`) is the only file that knows both tiers exist.

## Alternatives considered

Mocking the real clients in tests (couples tests to client APIs and proves
nothing about the seams); a framework's dependency-injection container (a second
thing to teach, no gain at this size); separate demo and production codebases
(guaranteed drift — the audit that motivated this remediation found exactly that).

## Consequences

The fast suite drives the REAL FastAPI app offline; the integration lane proves
the real adapters against real services; `/health` reports which tier composed so
a reviewer never guesses. Cost: every port needs two implementations kept
behaviourally aligned — the shared test files are what hold that line.

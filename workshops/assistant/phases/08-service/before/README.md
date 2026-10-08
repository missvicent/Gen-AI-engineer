# Workshop 8 — the running service

Edit `src/assistant/service.py`, `src/assistant/api.py`, `src/assistant/core.py`, `src/assistant/composers.py`, `src/assistant/connectors.py`, `src/assistant/output_gate.py`, `src/assistant/audit_log.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_service.py::test_health_reports_the_offline_tier
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_health_reports_the_offline_tier` — The health route is the composition root. The first failure is the service not assembling the offline tier.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. build_assistant(Settings()) with no env vars is the offline tier: in-memory retrieval, no network. /health reports that tier.
2. The output gate holds back a still-forming token. A chunk that already left cannot be redacted.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/service.py ../after/src/assistant/service.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../08-reliability/before
```

That copies only the files the previous layer asked you to write.

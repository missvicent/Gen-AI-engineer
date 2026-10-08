# Incident-response runbook — the capstone assistant

Short, actionable, and tied to the signals this stack actually emits. Each
scenario follows the same shape: **detect → contain → diagnose → recover →
learn**. The observability primitives referenced here are `/health`
(`status`, `degraded`, `spans_recorded`), the SQLite audit log, and the
OpenTelemetry spans (in-memory always; OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT`
is set).

**Start with the request id.** Every response carries `request_id`, every reply
echoes it as `x-request-id`, and it is on the root span — so a user quoting an id
from an error message gets you to the exact trace. The tree beneath it is
`assistant.request` → `auth.verify` and `assistant.pipeline` →
`guardrail.screen`, `memory.recall`, `rag.search`, `agent.run` (+ one span per
tool), `llm.compose`, `guardrail.output`. Which span is missing is usually more
informative than which one is slow: no `llm.compose` means the request was
refused before it ever cost anything.

**And the audit log is a table, not a diary.** Every row carries the same
bindings the spans do — `request_id`, `trace_id`, `subject`, `approval_id`,
`args_hash`, `result` — so the id a user quotes answers the whole question in one
query:

```sql
select at, kind, result, detail from audit_log where request_id = 'abc123def456';
select * from audit_log where trace_id = '<32 hex from the span>';
```

Do not grep `detail`. It is prose for a human, its format is not a contract, and
a regex over it starts returning zero rows the day somebody reformats a string —
which looks exactly like nothing having happened.

## 1. A prompt injection landed (or a red-team case regressed)

- **Detect**: `make redteam` fails in CI; or the audit log shows a `tool.ran`
  entry you cannot match to a legitimate `approval.granted` for that subject.
- **Contain**: revoke outstanding approvals with
  `DELETE FROM approvals` in the `ASSISTANT_DB` file (they are durable rows, not
  in-process state — a restart alone no longer clears them); if auth is on,
  rotate `ASSISTANT_JWT_SECRET` to invalidate all outstanding tokens.
- **Diagnose**: pull the failing input from the red-team report; check which
  layer should have caught it — L1 screen (`guardrails.py`), ingest screen
  (`Assistant.ingest`, look for `ingest.rejected` rows), context screen
  (`screen_contexts`), tool-output hardening (`harden_registry`), or the
  approval gate. If the payload is a *spelling* of something already known
  (encoded, leet, spaced, zero-width), the miss is in `expand`/`squash`, not in
  the pattern list — adding a seventh regex for the same sentence is how the
  list gets long without getting better.
- **Recover**: add the payload as a NEW row in
  `phase6-design-defend/01-red-team/*/evals/redteam.jsonl` (bump nothing —
  the version field changes only on schema changes), then fix the layer until
  `make redteam` is green again. A payload already in the corpus keeps coming
  back on every matching search, so re-ingest the affected source rather than
  relying on the retrieval screen to keep dropping it. `retrieval.sources` on the
  span names the document, and `DELETE /corpus/{source}` removes every chunk of
  it in one call — you no longer have to rebuild the collection to withdraw one
  page. Re-ingesting after the fix is an update, not a second copy, because chunk
  ids derive from `(tenant, source, ordinal)`.
- **Learn**: if the payload is a new *family*, add a category, not just a row.
  If it is a novel *phrasing* rather than a novel family, that is the case
  `ASSISTANT_GUARD_MODEL` exists for — turn the guard lane on and measure what
  it costs before deciding it is worth a round trip per untrusted string.

## 2. `/health` reports `degraded`

- **Detect**: `status: "degraded"` with a `degraded` map naming the component
  (`rag`, `brain`, `tools`) and the reason.
- **Contain**: nothing to contain — degraded mode IS the containment. Answers
  continue from the offline tier; ingest mirrors into the warm standby.
- **Diagnose**: `docker compose ps` + the named component. `rag` → Qdrant
  container or network; `brain` → Ollama (check its healthcheck: are the models
  still present?); `tools` → the MCP container never answered discovery.
- **Recover**: `docker compose restart <service>`. The assistant re-runs
  discovery only at boot, so after MCP recovers, `docker compose restart
  assistant`. Confirm `/health` returns to `ok`.
- **Learn**: if the same component degrades repeatedly, check its memory limit
  in `docker-compose.yml` — OOM-killed containers look like flaky networks.

## 2b. `/ready` stays 503, or the first answers are degraded after a deploy

- **Detect**: `/health` is 200 and `status: "ok"`, but `/ready` returns 503, or
  `ready: false` sits in the health payload. The nastier version has no signal at
  all: everything is green and the first few answers are visibly worse.
- **Contain**: do not route to the container. That is what the 503 is for — a
  load balancer or `depends_on: service_healthy` will hold traffic on its own.
- **Diagnose**: `curl -s $URL/ready | jq -r .detail` says why. `model tier not
  answering` means the host's Ollama is unreachable from the container — check
  the daemon is up and that `extra_hosts` maps `host.docker.internal`, which
  Linux does not provide on its own. `model tier degraded` means it answered by
  falling back, which is the cold-model case. Confirm with `ollama ps` on the
  host: a model listed by `ollama list` but absent from `ollama ps` is on disk
  and not in memory, and the first load costs more than the composer's budget.
- **Recover**: `./src/preflight-ollama.sh` runs both of those checks and prints
  the fixing command. It warms the model rather than only looking for it, which
  is the distinction this incident is about. Check `OLLAMA_KEEP_ALIVE` is still
  set; without it the model unloads after five idle minutes and every gap
  becomes a cold start.
- **Learn**: this is the failure that taught the difference between liveness and
  readiness here. Before `/ready` existed, a downloaded-but-cold model reported
  healthy, the first real question timed out, the offline composer answered it,
  and every probe stayed green while the answer quality dropped. A readiness
  signal that does not complete a real request is a liveness signal wearing the
  wrong label.

## 3. PII appeared where it should not

- **Detect**: a `[redacted: output failed the safety gate]` answer (the gate
  fired — working as intended), or worse, a report of PII in a real response.
- **Contain**: if auth is on, identify the caller from the audit log. If the leak
  came through a corpus document, the citation on the offending answer names it:
  `GET /evidence/{chunk_id}` returns the exact text and source, and `DELETE
  /corpus/{source}` removes it — scoped to that tenant, audited as
  `corpus.deleted`. Dropping the whole collection is still available and is now
  the second choice, not the first.
- **Diagnose**: PII reaches an answer only if it survived BOTH the document
  screen (redaction at ingest and at retrieval) and the output gate. The citation
  gives you the offending document without guessing; reproduce with it offline
  and check the `PII` patterns in `guardrails.py` for the format that slipped
  (new phone format? IBAN?). Note that a redacted chunk keeps its source and
  offsets, so a *partial* leak is still traceable to the page it came from.
- **Recover**: extend the `PII` patterns; add the format to the phase6 `pii`
  rows; re-run `make redteam` and the capstone fast suite.
- **Learn**: PII patterns rot. Schedule a review whenever a new data source
  (connector, corpus) is added.

## 4. Latency spiked / requests being shed

- **Detect**: callers see 429/503; or P99 over the span durations grows (the
  deployed-stack drill in `phases/WORKSHOP-DEPLOYED-STACK.md` shows how to read it).
- **Contain**: 429/503 IS load shedding working; raise `RATE_LIMIT_RPS` /
  `MAX_CONCURRENCY` only if the downstream can take it.
- **Diagnose**: find the slow span, and start one level up — split
  `assistant.pipeline` into its children before blaming anything. `tool.*` spans
  point at a connector or the MCP hop; a slow `llm.compose` points at the model
  (swapped tag? host under memory pressure?); a slow `rag.search` points at
  Qdrant; a slow `auth.verify` points at a cold JWKS cache reaching the issuer on
  every request. Group by `llm.model_name` and `llm.prompt_template.version`
  before concluding anything about "yesterday" — a prompt edit changes the stamp,
  which is what stops you comparing two different systems by accident.
- **Recover**: for a slow tool, the per-call timeout in `resilience.py` already
  bounds the damage; fix or replace the connector. For Ollama, confirm the
  model tag matches what the compose bootstrap pulled. If requests are hanging
  rather than merely slow, set `REQUEST_DEADLINE_SECONDS` — one budget the whole
  pipeline shares, after which the request answers 504 instead of holding a
  worker indefinitely.
- **Learn**: record the incident's span tree; it becomes the fixture for the
  next trace-diagnosis drill.

## 5. A wave of 504s, or a wave of 499s (they are different incidents)

- **Detect**: `http.status_code` on the request spans. 504 means the request ran
  past `REQUEST_DEADLINE_SECONDS`; 499 means the caller disconnected first.
- **Contain**: a 504 wave is real load — the shedding controls
  (`RATE_LIMIT_RPS`, `MAX_CONCURRENCY`) are the lever, because rejecting at the
  door is cheaper than timing out after paying for most of a request. A 499 wave
  is usually *downstream* of a 504 wave: callers hit their own timeout and left.
  Read them in that order, or you will diagnose the symptom.
- **Diagnose**: 504s point at whichever child span consumed the budget — split
  `assistant.pipeline` before blaming anything. For 499s, `request.abandoned` on
  the root says which frame the stream stopped at, so a truncated trace is
  explained rather than mysterious. A 499 with a fast pipeline underneath is a
  client-side timeout that is tighter than yours, which is a conversation to have
  with the client, not a fix to deploy.
- **Recover**: fix the slow stage, or raise the deadline if the work genuinely
  takes that long and the callers genuinely wait. Raising it to hide a slow stage
  just moves the failure to somebody else's timeout.
- **Learn**: 499s are not alerts. If they page, the alert is wrong — nobody is
  there to receive the error, and paging on it means paging every closed tab.

## 6. An irreversible effect may or may not have happened

- **Detect**: `GET /outbox` reports a non-zero `pending`. That row means the
  intent was committed and the outcome never was — the process stopped between
  the call and the answer.
- **Contain**: do NOT re-send. Re-sending a possibly-already-delivered
  irreversible action is worse than the ambiguity; that is why there is no
  redelivery worker.
- **Diagnose**: the row carries the tool, the args hash, the subject and the
  `request_id`. The request id resolves to the trace and to the audit rows, which
  together say how far the call got. For a message, the fastest answer is usually
  the destination itself — ask whether it arrived.
- **Recover**: confirm out-of-band, then settle the row by hand
  (`UPDATE outbox SET status = 'sent'|'failed'` in `ASSISTANT_DB`) so the queue
  reflects reality. A pending row nobody closes is a pending row everybody
  ignores.
- **Learn**: a `pending` row after a clean shutdown is a bug in the wrapper, not
  a crash — `settle` is supposed to run in both the success and the failure path.
  A cluster of them lines up with a deploy or an OOM kill; check the container's
  restart reason before assuming the code is wrong.

## 7. A retried approval double-fired (should be impossible — treat as a bug)

- **Detect**: two `tool.ran` rows share an `approval_id` — one grant, two
  executions. (Two rows with two different ids are two approvals, which is
  contract, not a bug.) It is a query, not a read-through, because the id is a
  column:

```sql
select approval_id, count(*) from audit_log
 where kind = 'tool.ran' and approval_id <> '' group by 1 having count(*) > 1;
```
- **Contain**: `DELETE FROM approvals` to revoke what is outstanding, then stop
  the service until diagnosed — the consume is supposed to make this impossible,
  so the gate itself is suspect.
- **Diagnose**: `consume` must be the single `DELETE ... RETURNING` statement in
  `approvals.py`. A `SELECT` followed by a `DELETE`, or a grant looked up before
  the loop instead of at the call site, reopens the double-spend. Check the
  client sent an `Idempotency-Key` too; without one, two POSTs are two
  legitimate approvals by contract. `GET /outbox` is the second reading: two
  `sent` rows sharing args and `request_id` cannot happen, so if you see them
  the reserve index is missing rather than the gate being wrong.
- **Recover / learn**: add the failing sequence to `test_reliability.py` first
  (race a thread pool at one grant and assert exactly one winner), then fix.

## State: backup and restore

All durable state — memory, audit log, approvals, idempotency keys, the outbox —
lives in ONE SQLite file (`ASSISTANT_DB`, the `assistant-data` compose volume; ADR-0005), so backup
is a file copy taken through SQLite's own online-backup so a mid-write copy can
never be torn:

```bash
docker compose exec assistant python -c \
  "import sqlite3; sqlite3.connect('/data/assistant.db').backup(sqlite3.connect('/data/backup.db'))"
docker compose cp assistant:/data/backup.db ./assistant-backup.db
```

Restore is the reverse (`docker compose cp` the file back to `/data/assistant.db`
while the service is stopped, then `docker compose start assistant`). Verify a
restore the same way e2e verifies persistence: memory recalls across the restart
and `sqlite3 assistant.db 'select count(*) from audit_log'` matches the backup.
Qdrant's collection rebuilds from re-ingestion; Ollama's models re-pull — the
SQLite file is the only state you cannot regenerate, which is why it is the only
thing this procedure covers.

Scripted, with retention and a readability check, in
`phase8-deploy/03-deploy-observe/deploy/backup.sh` and `restore.sh`. Two habits
from there are worth adopting even by hand: verify the copy in the same script
that takes it (an unverified backup is a folder of files you *hope* are a
database, and "verify it later" means never), and stop the writer before
restoring — two writers on one SQLite file produce a database that is neither the
backup nor the original.

## "The deploy said OK but nothing changed"

- **Detect**: new behavior is missing, no errors anywhere, `/health` is green.
- **First reading**: `curl -s $URL/health | jq -r .version` against the SHA you
  deployed. They differ when a rollout half-finished and an old machine is still
  in the pool. That machine is a *working service* — healthy, correct, answering
  — which is why every other check passes against it and why this is the probe
  that catches it. `provenance.build_version()` reads `GIT_SHA`, baked in by
  `ARG GIT_SHA` at image build; `dev` means the image was built without it.
- **Confirm**: `deploy/release.sh` runs exactly this comparison as one of four
  smoke probes and rolls back on failure, so a manual mismatch usually means the
  deploy was done by hand or the lane is off (`DEPLOY_LANE`).
- **Recover**: redeploy the immutable tag (`registry/app:<sha>`), never `latest`
  — redeploying a moving tag is how a rollback silently ships the code you were
  rolling back from.

## Escalation

This is a single-operator local stack: "escalation" means filing an issue with
the audit-log excerpt, the `/health` payload, and the span tree attached. All
three exist precisely so that the report is reconstructable after the fact.

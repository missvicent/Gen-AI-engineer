# ADR-0013 — One budget per request, and intent written down before effect

**Status:** accepted

## Context

The capstone had reliability primitives — retries, backoff, a timeout, a token
bucket, a concurrency cap — and they were all correct in isolation. What was
missing was the part that only shows up when they run together in a real request.

**`except Exception: retry` retried bugs.** A `TypeError` is deterministic, so
three attempts bought three times the latency before the same 500, with the
traceback pointing at the third identical retry instead of at the cause. In the
other direction it retried a 400 from an API that had already given its final
answer, holding a connection each time.

**Every retry was a retry of a possibly-completed send.** `send_telegram` went
through the same `resilient` wrapper as a read. A timeout tells you nothing about
whether the server acted, so the retry policy was, in the non-idempotent case, a
message-duplication policy.

**The connectors could not be retried at all.** Both caught their own `OSError`
and returned `{"error": ...}`. From outside, a returned error dict is a
successful call, so the retry wrapper never fired. The policy was present in the
code and had never once run.

**Timeouts composed by addition.** Three retries of a 10-second call inside a
60-second composer inside a request nobody bounded is a number no one had ever
computed. Nothing tied a per-call timeout to the time the *request* had left.

**Nothing noticed when the caller left.** A client that closed the tab was still
being generated for, and under load that is the failure that compounds: the queue
fills with work for callers who left, which makes everyone slower, which makes
more callers leave.

**Only `/approve` was idempotent.** `/ingest` and `DELETE /corpus/{source}` were
not, so a retried batch duplicated the corpus and a retried delete could remove a
source somebody had re-added in between. And the one protected route replied to a
replay with `{"replayed": true}` and nothing else — technically no double effect,
and a broken client, which had asked for an approval id.

**An irreversible call could vanish.** `consume the grant → call the API → write
the audit row` can stop between any two lines. Crash after line 2 and a message
went out that nothing in the system remembers.

## Decision

**A failure is classified before it is retried.** `TRANSIENT` is the honest list
— the network, the clock, and the server saying "later". A `Permanent` exception
lets an adapter that can read a status code mark a 4xx as final. `Policy.retry_if`
is injectable, because "transient" is protocol-specific and that knowledge belongs
to the adapter.

**`ONCE` is a named policy, not a comment.** One attempt, still bounded by the
timeout. `send_telegram` uses it, and so does every MCP tool call — a discovered
tool arrives as a name, a description and a JSON schema, and nothing in the
protocol says whether calling it twice charges a card twice.

**Connectors raise inward and apologise outward.** An inner function that raises,
`resilient` in the middle, an outer function that returns the dict the tool
contract promises. This is what makes the retry policy real rather than
decorative.

**One `Budget` per request, in a ContextVar** (`deadline.py`): an absolute
monotonic deadline plus a `cancelled` predicate, because "we ran out of time" and
"nobody is listening" are the same question. `deadline.capped()` shrinks each
call's timeout to what the request has left, so a 60-second composer timeout
inside a request with 4 seconds left is a 4-second composer timeout and the
composer never had to know the request existed.

**Checked at seams, not enforced by killing threads.** Python cannot safely
interrupt arbitrary code, and pretending otherwise leaks half-finished work. The
budget is consulted between pipeline stages, before each retry, and before each
streamed frame — refusing to start the next thing, which turns an unbounded
request into a bounded one and is most of the value.

**504 for a deadline, 499 for a disconnect.** A 504 is an alert: the service was
too slow and somebody should look. A 499 is not — nobody is there to receive it,
and paging on it means paging every closed tab.

**Idempotency covers every mutation, replays the ORIGINAL answer, and releases
the key on failure.** Keys are namespaced by subject *and* operation: they are
client-chosen, "retry-1" is what everyone picks, and one flat namespace lets a
retried ingest swallow an approval. Recording the key before the operation
succeeds would turn one transient failure into a permanent one — every retry
cheerfully acknowledged, the effect never applied.

**Irreversible intent is durable before the effect** (`outbox.py`): a `pending`
row committed *before* the call, settled `sent` or `failed` after. There is no way
to make "spend the grant", "call the API" and "write the row" atomic — the API is
not in our database — but the intent can be. A crash now leaves a question ("did
this send?") instead of silence, and `GET /outbox` is where an operator reads it.
The row is keyed by `(subject, tool, args_hash, request_id)`, which makes the
reserve itself idempotent while keeping two *deliberate* sends of the same message
distinguishable from one retried send.

## Alternatives considered

Retrying everything and de-duplicating downstream (moves the problem to a system
that has even less context about intent). A per-layer timeout budget passed
explicitly through every signature (correct, and it makes every signature about
timeouts; the ContextVar is the same trade already made for the request id).
Killing the worker thread at the deadline (Python cannot, safely). Server-minted
idempotency keys when the client omits one (a key the client does not know cannot
be used to retry — it is protection nobody can invoke). A delivery worker that
re-sends `pending` outbox rows (re-sending a possibly-already-delivered
irreversible action is a worse default than telling a human; the honest version at
this scale is a list an operator can read). Retrying MCP calls with a heuristic
guess at idempotency (an optimistic guess about somebody else's side effects).
Treating disconnect as a 200 with an empty body (hides a real signal and makes the
"how many callers give up" question unanswerable).

## Consequences

`resilient` now raises `deadline.Expired` before the first attempt when the budget
is already gone, so callers see a distinct exception rather than a generic
timeout. It is deliberately not transient: the point of a deadline is that waiting
longer is not an option.

`REQUEST_DEADLINE_SECONDS` is off by default. The fast tier drives a real model in
the integration lane, where a deadline would be a flake generator; the deployed
profile turns it on, because an unbounded request there is a worker held hostage.

The disconnect watcher polls `request.is_disconnected()` from a sibling task,
which means the request body has to be read in the middleware first — the watcher
and the route share a receive channel, and a poll that lands mid-body can take a
body message and drop it. Buffering the body is acceptable here because every
route on this surface takes small JSON; a service accepting uploads would need a
different arrangement.

`GET /outbox` requires the approve scope: the list of irreversible actions a
tenant has taken is as sensitive as the ability to authorise them.

A stream cut short by a disconnect ends without a `done` frame. That is the
correct wire behaviour — there is nobody to send it to — and the root span carries
`request.abandoned` so a truncated trace is explained rather than mysterious.

"""The HTTP surface — every route, and nothing else.

`create_app` builds the FastAPI app around an assembled Assistant: the load-
shedding middleware at the door, the optional Bearer-JWT dependency on every
mutating route, and the endpoints (/health + /ready, /ingest, /ask + /ask/stream,
/approve). No business logic lives here — the routes translate HTTP in and out
of `Assistant` calls, which is why the whole surface is testable with a
TestClient and no network.

TODO: two things are missing from this surface, and both are about requests that
do not go to plan.

**Every mutating route needs idempotency, not just `/approve`.** The argument
that made `/approve` safe applies unchanged to `/ingest` and to
`DELETE /corpus/{source}`: a client whose request timed out cannot tell whether
the server acted, so it will retry, and every unprotected mutation is a double
effect waiting for a slow network.

**Nothing here bounds a request.** No deadline, and no notion of a caller who
left — so a request whose model call stalls holds a worker until the process
restarts, and a client that closed the tab is still being generated for.

Reference: ../../after/src/assistant/api.py.
"""
from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from assistant import audit_log, auth, observe
from assistant.observe import stage
from assistant.provenance import build_version
from assistant.resilience import ConcurrencyCap, TokenBucket
from assistant.service import build_assistant
from assistant.settings import Settings


class AskBody(BaseModel):
    question: str


class IngestBody(BaseModel):
    # A document is prose, or prose with a name. The name is what a citation
    # points at and what DELETE /corpus/{source} removes, so a caller that has
    # one should be able to say it — and a caller that does not is not blocked.
    docs: list[str | dict[str, str]]


class ApproveBody(BaseModel):
    tool: str
    # The exact arguments being approved, echoed from the /ask pause. Defaulting
    # to {} fails CLOSED: an approval that names no arguments matches only a call
    # that takes none, so a client that forgets them gets another pause rather
    # than a blanket permission.
    args: dict[str, Any] = Field(default_factory=dict)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="GenAI assistant capstone")
    assistant = build_assistant(settings)
    app.state.assistant = assistant
    s = assistant.settings

    # --- readiness, resolved off the request path -------------------------------
    # A cold 9B model takes minutes to load and the composer budget is 60 seconds,
    # so "the process started" and "the next request will succeed" are different
    # facts. Warming in a thread keeps startup fast and moves the wait to /ready,
    # where an orchestrator is already looking. The offline tier has nothing to
    # load, so it answers immediately rather than making every test wait a tick.
    app.state.ready, app.state.ready_detail = False, "warming up"

    def warm_up() -> None:
        app.state.ready, app.state.ready_detail = assistant.warm()

    if s.ollama_host:
        Thread(target=warm_up, name="warmup", daemon=True).start()
    else:
        warm_up()

    # --- load shedding: refuse politely at the door, don't fall over inside ------
    bucket = (
        TokenBucket(rate=s.rate_limit_rps, burst=s.rate_limit_burst)
        if s.rate_limit_rps
        else None
    )
    cap = ConcurrencyCap(s.max_concurrency) if s.max_concurrency else None
    app.state.bucket, app.state.cap = bucket, cap

    @app.middleware("http")
    async def throttle(request: Request, call_next):
        if request.url.path not in ("/health", "/ready"):  # probes must never be shed
            if bucket is not None and not bucket.take():
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
            if cap is not None and not cap.enter():
                return JSONResponse({"detail": "server at capacity"}, status_code=503)
            try:
                return await call_next(request)
            finally:
                if cap is not None:
                    cap.leave()
        return await call_next(request)

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        """One root span per HTTP request, and one id every layer shares.

        The id comes from the caller when the caller has one (`X-Request-ID`,
        which a gateway or the client's own logs will already know) and is
        minted here otherwise. It goes on the span, into the ContextVar the
        pipeline and audit log read, and back out as a response header — so a
        user quoting an id from an error message resolves to a trace, an audit
        row, and a log line without anyone joining on timestamps.

        Ordering note, and it is the opposite of what registration order looks
        like: Starlette inserts each middleware at the FRONT, so the last one
        registered runs first. This is outside `throttle`, which means a shed
        request still gets a span and an `x-request-id`. Worth keeping — "we are
        returning 429s" is exactly the moment someone needs the trace, and a
        rejection nobody can look up is a rejection nobody can explain.
        """
        with observe.request_scope(request.headers.get("x-request-id")) as rid, stage(
            assistant.rec.tracer, observe.REQUEST_SPAN,
            **{
                observe.REQUEST_ID: rid,
                observe.HTTP_METHOD: request.method,
                observe.HTTP_ROUTE: request.url.path,
            },
        ) as span:
            response = await call_next(request)
            span.set_attribute(observe.HTTP_STATUS, response.status_code)
            response.headers["x-request-id"] = rid
            return response

    # TODO 1: a third middleware, `bound_the_request`, giving every request ONE
    # budget: a deadline (`s.request_deadline`) and a flag for "the caller left".
    #
    # Both belong here because this is the only layer that knows either fact —
    # the deadline is a property of the HTTP contract, and the disconnect is an
    # ASGI event. Everything below reads them through a ContextVar (deadline.py)
    # and stays synchronous.
    #
    # The disconnect has to be WATCHED, not checked: it arrives as a message, so
    # `request.is_disconnected()` only becomes true once something has awaited
    # the receive channel. Poll it from an `asyncio.create_task` sibling, set an
    # `asyncio.Event`, and pass `event.is_set` as `cancelled` to
    # `deadline.budget`.
    #
    # Read the body (`await request.body()`) BEFORE starting the watcher. The
    # watcher reads from the same receive channel the route does, so a poll that
    # lands while the body is still in flight can take a body message and drop
    # it — an intermittently empty request, which is about the worst bug shape
    # there is.
    #
    # **Where you cancel the watcher is the exercise, and the obvious answer is
    # wrong.** `try: ... finally: watcher.cancel()` around `call_next` looks right
    # and cannot work for a streaming response: `call_next` returns as soon as the
    # response OBJECT exists, and the body is iterated afterwards while the server
    # sends it. So the watcher dies before the client has had any chance to leave,
    # and `/ask/stream` — the endpoint where an abandoned request is most expensive
    # — is the one endpoint where a disconnect is never noticed.
    #
    # Give the watcher the body's lifetime instead: when the response has a
    # `body_iterator`, wrap it in an async generator that cancels the watcher in its
    # own `finally` (reached on exhaustion, on an exception, and on the `aclose()`
    # the server performs when the connection drops). A buffered response has no
    # body to outlive, so tear it down where you were going to.
    #
    # The budget's ContextVar needs no such treatment, and knowing why saves an
    # afternoon: `call_next` runs the downstream app in its own task, that task
    # copies the context at creation, and a reset here cannot reach the copy. The
    # flag it reads is the same `Event` object. Watcher lifetime is the only thing
    # in play.
    #
    # `tests/test_disconnect.py` is the one test that can tell you if you got this
    # right, and note what it asserts on: `deadline.expired()` from inside the
    # running model, NOT the number of chunks produced. A chunk count passes on the
    # broken version, because Starlette's zero-buffer stream breaks when the
    # transport dies and the generator gets collected anyway — an accident of two
    # libraries' buffer sizing, not a cancellation, and nothing else that reads the
    # flag ever wakes up.

    # TODO 2: an exception handler for `deadline.Expired` — 504 when the reason
    # is a deadline, 499 ("client closed request") when the caller disconnected.
    #
    # The distinction is worth the extra branch. A 504 is an alert: the service
    # was too slow and somebody should look. A 499 is not — nobody is there to
    # receive it, and paging on it means paging every time a user closes a tab.

    policy = s.auth_policy()
    # One JWKS cache for the process: the issuer's keys are fetched on first use
    # and reused, so a busy service does not turn every request into a request
    # to the identity provider.
    keys = auth.JwksKeys(policy.jwks_url) if policy.jwks_url else None

    def require(scope: str) -> Callable[..., str]:
        """A FastAPI dependency enforcing the optional Bearer-JWT gate for one
        scope. With auth off (no secret and no JWKS URL) it resolves to the
        anonymous identity and never rejects — the zero-key demo path."""

        def dependency(request: Request) -> str:
            # Its own span: verification can be the slow part (a JWKS fetch on a
            # cold cache is a network round trip), and a 401 that never reaches
            # the pipeline is otherwise invisible in the trace.
            with stage(
                assistant.rec.tracer, observe.AUTH_SPAN,
                **{observe.AUTH_MODE: assistant.tier()["auth"]},
            ) as span:
                ok, reason, subject = auth.identity_from(
                    request.headers.get("authorization", ""),
                    policy=policy, required_scope=scope, keys=keys,
                )
                span.set_attribute(observe.AUTH_OK, ok)
                if not ok:
                    span.set_attribute(observe.AUTH_REASON, reason)
                    raise HTTPException(status_code=401, detail=reason)
                span.set_attribute(observe.SUBJECT, subject)
                return subject

        return dependency

    @app.get("/health")
    def health() -> dict:
        # spans_recorded lets an end-to-end verifier confirm observability is live
        # from outside the process — the in-memory exporter is always attached.
        # `version` is the commit baked into the image (provenance.build_version),
        # and it is here so a deploy can be PROVEN rather than assumed: a rollout
        # that half-finished leaves an old machine in the pool, healthy and
        # answering the wrong code. Every other probe passes against it.
        return {
            "status": "degraded" if assistant.degraded else "ok",
            "version": build_version(),
            "tier": assistant.tier(),
            "degraded": assistant.degraded,
            "spans_recorded": len(assistant.rec.spans()),
            # Liveness says the process is up; this says the model tier can
            # actually finish a request. They are different questions and used
            # to have one answer — see /ready.
            "ready": app.state.ready,
        }

    @app.get("/ready")
    def ready(response: Response) -> dict:
        """Readiness, separated from liveness because they fail differently.

        A container that is up but cannot complete a request inside its budget
        should not receive traffic, and should not report itself ready to the
        orchestrator that decides whether the rollout succeeded. The stack used
        to conflate the two: compose called ollama healthy once the models were
        *downloaded*, the assistant came up, `/health` said ok, and the first
        real question timed out on a cold model and was answered by the offline
        fallback. Every probe was green and the answer was wrong.

        503 until proven otherwise, because the failure mode of guessing wrong
        in the other direction is silent.
        """
        if not app.state.ready:
            response.status_code = 503
        return {"ready": app.state.ready, "detail": app.state.ready_detail}

    def once(subject: str, operation_name: str, key: str | None, work) -> dict:
        """TODO 3: run a side effect at most once per (subject, operation, key).

        One helper, used by all three mutating routes, wrapping
        `assistant.idempotency.run`. On a replay, record an audit row and add
        `"replayed": True` to the original answer.

        Namespace the key by subject AND operation. Keys are client-chosen, so
        one tenant's "retry-1" must not swallow another's — and the same
        tenant's "retry-1" on `/ingest` must not swallow their "retry-1" on
        `/approve`.
        """
        raise NotImplementedError

    @app.post("/ingest")
    def ingest(
        body: IngestBody, subject: str = Depends(require("assistant:ingest"))
    ) -> dict:
        # Documents land in the CALLER's tenant — retrieval is scoped the same
        # way — and are screened on the way IN, so a poisoned page is never
        # written rather than merely never read. The response says how many were
        # refused: a batch that silently loses half its rows is worse than one
        # that fails.
        #
        # TODO 4: take an `Idempotency-Key` header and route this through
        # `once`. Without it, a retried batch is a duplicated corpus.
        return assistant.ingest(list(body.docs), subject)

    @app.delete("/corpus/{source}")
    def forget(source: str, subject: str = Depends(require("assistant:ingest"))) -> dict:
        # Scoped to the caller's tenant by the same argument that scopes search:
        # a delete that crossed tenants would be a denial-of-service with a REST
        # interface. Requires the ingest scope, because the right to add a
        # document and the right to withdraw it are the same right.
        #
        # TODO 5: the same, and the stakes are higher — a retried delete can
        # remove a source that somebody re-added in between the two attempts.
        return assistant.forget(source, subject)

    @app.get("/evidence/{chunk_id}")
    def evidence(chunk_id: str, subject: str = Depends(require("assistant:ask"))) -> dict:
        # What makes a citation checkable: the id printed in an answer resolves
        # back to the exact text, source and revision that produced it. A 404 is
        # a real answer here — the evidence is gone, and the reader should know
        # that rather than be handed a plausible substitute.
        found = assistant.evidence(chunk_id, subject)
        if found is None:
            raise HTTPException(status_code=404, detail="no such evidence")
        return found

    @app.post("/ask")
    def ask(body: AskBody, subject: str = Depends(require("assistant:ask"))) -> dict:
        return assistant.ask(body.question, subject)

    @app.post("/ask/stream")
    def ask_stream(
        body: AskBody, subject: str = Depends(require("assistant:ask"))
    ) -> StreamingResponse:
        return StreamingResponse(
            assistant.ask_stream(body.question, subject),
            media_type="text/event-stream",
        )

    @app.post("/approve")
    def approve(
        body: ApproveBody,
        request: Request,
        subject: str = Depends(require("assistant:approve")),
    ) -> dict:
        # A retried request (same Idempotency-Key) is acknowledged, not re-applied.
        # The key is namespaced by subject: keys are client-chosen, and one tenant's
        # "retry-1" must not silently swallow another tenant's approval.
        #
        # TODO 6: this hand-rolled version replies with a bare receipt, which
        # avoids the double grant and still breaks the client — it asked for an
        # approval id and got an acknowledgement. Route it through `once` so a
        # retry returns the ORIGINAL answer, which is what "idempotent" means.
        key = request.headers.get("idempotency-key")
        if key and assistant.idempotency.seen(f"{subject}:approve:{key}"):
            assistant.audit_log.record(
                "approval.replayed", subject, body.tool, result=audit_log.REPLAYED
            )
            return {"approved": body.tool, "replayed": True}
        # bound to the AUTHENTICATED subject, never to a subject in the body
        grant = assistant.approvals.mint(subject, body.tool, body.args)
        assistant.audit_log.record(
            "approval.granted", subject, f"{body.tool} (approval {grant.approval_id})",
            approval_id=grant.approval_id, args=body.args,
        )
        return {
            "approved": body.tool,
            "approval_id": grant.approval_id,
            "args_hash": grant.args_hash,
            "expires_at": grant.expires_at,
        }

    @app.get("/outbox")
    def outbox(subject: str = Depends(require("assistant:approve"))) -> dict:
        """TODO 7: irreversible effects this service started, and how they ended.

        Return `{"effects": [...], "pending": n}` from `assistant.outbox`. A
        `pending` row is the interesting one: the intent was committed and the
        outcome never was, which means the process stopped mid-send. That is a
        question for a human — "did this message go out?" — and it is a question
        that can only be asked if somebody wrote the intent down first
        (outbox.py). Do NOT add automatic redelivery: re-sending a
        possibly-already-sent irreversible action is worse than asking.
        """
        raise NotImplementedError

    return app

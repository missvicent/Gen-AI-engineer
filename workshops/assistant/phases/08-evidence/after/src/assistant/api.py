"""The HTTP surface — every route, and nothing else.

`create_app` builds the FastAPI app around an assembled Assistant: the load-
shedding middleware at the door, the optional Bearer-JWT dependency on every
mutating route, and the endpoints (/health + /ready, /ingest, /ask + /ask/stream,
/approve). No business logic lives here — the routes translate HTTP in and out
of `Assistant` calls, which is why the whole surface is testable with a
TestClient and no network.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from threading import Thread
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from assistant import audit_log, auth, deadline, observe
from assistant.observe import stage
from assistant.provenance import build_version
from assistant.resilience import ConcurrencyCap, TokenBucket
from assistant.service import build_assistant
from assistant.settings import Settings

#: How often the disconnect watcher asks. Short enough that a long stream stops
#: promptly, long enough that an idle connection is not a busy-loop.
DISCONNECT_POLL_SECONDS = 0.25


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
        TokenBucket(rate=s.rate_limit_rps, burst=s.rate_limit_burst) if s.rate_limit_rps else None
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
        with (
            observe.request_scope(request.headers.get("x-request-id")) as rid,
            stage(
                assistant.rec.tracer,
                observe.REQUEST_SPAN,
                **{
                    observe.REQUEST_ID: rid,
                    observe.HTTP_METHOD: request.method,
                    observe.HTTP_ROUTE: request.url.path,
                },
            ) as span,
        ):
            response = await call_next(request)
            span.set_attribute(observe.HTTP_STATUS, response.status_code)
            response.headers["x-request-id"] = rid
            return response

    @app.middleware("http")
    async def bound_the_request(request: Request, call_next):
        """One budget per request: a deadline, and a flag for "the caller left".

        Both live here because this is the only layer that knows either fact —
        the deadline is a property of the HTTP contract, and the disconnect is
        an ASGI event. Everything below reads them through a ContextVar
        (deadline.py) and stays synchronous.

        The watcher is a task rather than a check because a disconnect arrives
        as a message, not as a return value: `is_disconnected()` only becomes
        true once something has awaited the receive channel. Polling it from a
        sibling task is what lets a *synchronous* pipeline, running in the
        threadpool, notice that nobody is listening any more.

        The body is read here first, and that is not a stray optimisation. The
        watcher reads from the same receive channel the route does, so a poll
        that lands while the body is still in flight can take a body message and
        drop it — an intermittently empty request, which is about the worst bug
        shape there is. Reading first makes the middleware's request the owner of
        the body (Starlette replays the cached copy downstream) and leaves the
        channel with nothing on it but the disconnect the watcher is looking for.

        **The watcher outlives `call_next`, and for a streaming response that is
        the whole mechanism.** `call_next` returns as soon as the response OBJECT
        exists; the body is iterated afterwards, while the server sends it. So a
        watcher cancelled when `call_next` returns is a watcher cancelled before
        the client has had a chance to leave — which is what this used to do. The
        unit tests still passed, because they inject `cancelled=` by hand and
        prove the generator obeys it; nothing proved an ASGI disconnect ever set
        it. A real probe left after 0.8 seconds having read nothing and the server
        went on to drain all 200 source chunks for an audience of nobody.

        So the teardown moves onto the body iterator. The budget's ContextVar does
        NOT need the same treatment: `call_next` starts the downstream app in its
        own task, that task copies the context at creation — inside the `with`
        below — and a `reset` here cannot reach the copy. The flag it reads is this
        same `asyncio.Event`, so the only thing that was missing is somebody still
        polling. That asymmetry is worth knowing before editing either half.
        """
        await request.body()

        left = asyncio.Event()

        async def watch() -> None:
            with contextlib.suppress(Exception):
                while not await request.is_disconnected():
                    await asyncio.sleep(DISCONNECT_POLL_SECONDS)
                left.set()

        watcher = asyncio.create_task(watch())

        async def stop_watching() -> None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        try:
            with deadline.budget(s.request_deadline, cancelled=left.is_set):
                response = await call_next(request)
        except BaseException:
            await stop_watching()
            raise

        # A buffered response is already generated and already gated: there is no
        # more work a disconnect could save, so it is torn down here.
        body = getattr(response, "body_iterator", None)
        if body is None:
            await stop_watching()
            return response

        async def watched(source):
            try:
                async for chunk in source:
                    yield chunk
            finally:
                # Reached on exhaustion, on an exception, and on the `aclose()` the
                # server performs when the connection drops — the last of which is
                # the case that matters and the reason this is a `finally`.
                await stop_watching()

        response.body_iterator = watched(body)
        return response

    @app.exception_handler(deadline.Expired)
    async def expired(request: Request, exc: deadline.Expired) -> JSONResponse:
        """504 for a deadline, 499 for a disconnect.

        The distinction is worth the extra branch. A 504 is an alert: the
        service was too slow and somebody should look. A 499 (nginx's
        "client closed request") is not — nobody is there to receive it, and
        paging on it means paging every time a user closes a tab. Conflating
        them is how a dashboard fills with incidents that are really just
        people leaving."""
        gone = exc.reason == "caller disconnected"
        return JSONResponse({"detail": exc.reason}, status_code=499 if gone else 504)

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
                assistant.rec.tracer,
                observe.AUTH_SPAN,
                **{observe.AUTH_MODE: assistant.tier()["auth"]},
            ) as span:
                ok, reason, subject = auth.identity_from(
                    request.headers.get("authorization", ""),
                    policy=policy,
                    required_scope=scope,
                    keys=keys,
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
        #
        # `version` is the commit baked into the image, and it is here so a deploy
        # can be PROVEN rather than assumed: a rollout that half-finished leaves an
        # old machine in the pool, and that machine is healthy, correct, and
        # answering — it just isn't the code you shipped. Every other probe passes
        # against it. This is the one that doesn't.
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
        """Run a side effect at most once per (subject, operation, key).

        Every mutating route goes through here, not just `/approve`. The
        argument for extending it is the one that made `/approve` idempotent in
        the first place: a client whose request timed out cannot tell whether
        the server acted, so it will retry, and every unprotected mutation is a
        double effect waiting for a slow network. `/ingest` without this is a
        duplicated batch; `DELETE /corpus` without it is a source deleted twice,
        the second time possibly after somebody re-added it.

        Keys are namespaced by subject AND operation. They are client-chosen, so
        one tenant's "retry-1" must not swallow another's — and the same tenant's
        "retry-1" on `/ingest` must not swallow their "retry-1" on `/approve`.
        """
        namespaced = f"{subject}:{operation_name}:{key}" if key else None
        result, replayed = assistant.idempotency.run(namespaced, work)
        if replayed:
            assistant.audit_log.record(
                f"{operation_name}.replayed",
                subject,
                str(key),
                result=audit_log.REPLAYED,
            )
            return dict(result) | {"replayed": True}
        return result

    @app.post("/ingest")
    def ingest(
        body: IngestBody,
        subject: str = Depends(require("assistant:ingest")),
        idempotency_key: str | None = Header(default=None),
    ) -> dict:
        # Documents land in the CALLER's tenant — retrieval is scoped the same
        # way — and are screened on the way IN, so a poisoned page is never
        # written rather than merely never read. The response says how many were
        # refused: a batch that silently loses half its rows is worse than one
        # that fails.
        return once(
            subject,
            "ingest",
            idempotency_key,
            lambda: assistant.ingest(list(body.docs), subject),
        )

    @app.delete("/corpus/{source}")
    def forget(
        source: str,
        subject: str = Depends(require("assistant:ingest")),
        idempotency_key: str | None = Header(default=None),
    ) -> dict:
        # Scoped to the caller's tenant by the same argument that scopes search:
        # a delete that crossed tenants would be a denial-of-service with a REST
        # interface. Requires the ingest scope, because the right to add a
        # document and the right to withdraw it are the same right.
        return once(
            subject,
            "corpus.delete",
            idempotency_key,
            lambda: assistant.forget(source, subject),
        )

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
        subject: str = Depends(require("assistant:approve")),
        idempotency_key: str | None = Header(default=None),
    ) -> dict:
        def mint() -> dict:
            # bound to the AUTHENTICATED subject, never to a subject in the body
            grant = assistant.approvals.mint(subject, body.tool, body.args)
            assistant.audit_log.record(
                "approval.granted",
                subject,
                f"{body.tool} (approval {grant.approval_id})",
                approval_id=grant.approval_id,
                args=body.args,
            )
            return {
                "approved": body.tool,
                "approval_id": grant.approval_id,
                "args_hash": grant.args_hash,
                "expires_at": grant.expires_at,
            }

        return once(subject, "approval", idempotency_key, mint)

    @app.get("/outbox")
    def outbox(subject: str = Depends(require("assistant:approve"))) -> dict:
        """Irreversible effects this service started, and how they ended.

        A `pending` row is the interesting one: the intent was committed and the
        outcome never was, which means the process stopped mid-send. That is a
        question for a human — "did this message go out?" — and it is a question
        that can only be asked if somebody wrote the intent down first
        (outbox.py). Automatic redelivery is deliberately absent: re-sending a
        possibly-already-sent irreversible action is worse than asking."""
        rows = assistant.outbox.entries(subject)
        return {
            "effects": [
                {
                    "tool": e.tool,
                    "status": e.status,
                    "at": e.at,
                    "request_id": e.request_id,
                    "detail": e.detail,
                }
                for e in rows
            ],
            "pending": sum(1 for e in rows if e.status == "pending"),
        }

    return app

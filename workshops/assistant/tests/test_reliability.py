"""Operate-tier behavior, proven offline: adapters that die mid-request degrade
to the offline tier instead of 500ing, load is shed politely at the door, every
side effect survives a retry unchanged, and an irreversible call that crashes
mid-flight leaves a question rather than silence."""

import json
import sqlite3
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from assistant import deadline
from assistant.adapters import InMemoryRag
from assistant.api import create_app
from assistant.composers import ABSTAIN, StreamTruncated, offline_compose
from assistant.fallbacks import (
    COMPOSE_POLICY,
    FallbackRag,
    fallback_composer,
    fallback_stream,
)
from assistant.idempotency import IdempotencyStore
from assistant.outbox import FAILED, PENDING, SENT, Outbox, recorded_registry
from assistant.resilience import Policy
from assistant.service import build_assistant
from assistant.settings import COMPOSE_TIMEOUT_SECONDS, Settings
from assistant.tools import REGISTRY, rewrap

FAST = Policy(attempts=2, base_delay=0.0, timeout=None)


class ExplodingRag:
    """A Qdrant stand-in whose network died."""

    def add(self, docs, tenant="local"):
        raise ConnectionError("qdrant unreachable")

    def search(self, query, k=3, tenant="local"):
        raise ConnectionError("qdrant unreachable")


def degraded_client() -> TestClient:
    app = create_app(Settings())
    a = app.state.assistant
    a.rag = FallbackRag(
        ExplodingRag(), InMemoryRag(),
        lambda component, why: a.degraded.__setitem__(component, why),
        policy=FAST,
    )
    return TestClient(app)


# --- degraded operation -----------------------------------------------------


def test_a_dead_rag_backend_degrades_to_the_offline_tier():
    c = degraded_client()
    c.post("/ingest", json={"docs": ["approved refunds take five business days"]})
    body = c.post("/ask", json={"question": "how long do refunds take"}).json()
    assert "refunds" in body["answer"].lower(), "the warm standby answered"

    health = c.get("/health").json()
    assert health["status"] == "degraded"
    assert "rag" in health["degraded"]


def test_a_dead_composer_falls_back_to_offline_prose():
    def dead_model(goal, contexts, state, memories=None):
        raise ConnectionError("ollama unreachable")

    seen: dict[str, str] = {}
    compose = fallback_composer(dead_model, seen.__setitem__, policy=FAST)
    assert compose("q", ["the fact"], [], []) == "the fact"
    assert "brain" in seen


def test_the_composition_budget_is_deployment_policy_not_a_constant():
    """Why this is configurable at all, in one sentence: the end-to-end lane
    failed on a correctly configured stack.

    Docker Desktop gives a container no GPU, so a containerised 9B answered at
    about half a token per second. Every composition passed 60 seconds, the
    offline stitcher answered, and the run failed on check 4 — after seventeen
    minutes, with nothing wrong except a number compiled into a library. The
    model has since moved to the host, where 60 seconds is generous again; the
    setting stays because the right budget is a fact about the hardware
    underneath, not about the code. Both paths read it from the same setting so
    batch and stream cannot drift apart."""
    slow = Settings(ollama_host="http://host.docker.internal:11434", compose_timeout=900)
    assert slow.compose_timeout == 900
    assert Settings().compose_timeout == COMPOSE_TIMEOUT_SECONDS


def test_the_budget_comes_from_the_environment_the_container_was_given(monkeypatch):
    monkeypatch.setenv("COMPOSE_TIMEOUT_SECONDS", "900")
    assert Settings.from_env().compose_timeout == 900


def test_a_composer_slower_than_its_budget_falls_back_at_the_budget():
    """The behaviour the setting buys, at a hundredth of the scale: a primary
    that takes longer than it is allowed is a failure, and the fallback answers
    rather than the caller waiting."""
    import time as _time

    def molasses(goal, contexts, state, memories=None):
        _time.sleep(1)
        return "the model got there eventually"

    seen: dict[str, str] = {}
    compose = fallback_composer(
        molasses, seen.__setitem__, replace(COMPOSE_POLICY, attempts=1, timeout=0.05)
    )
    assert compose("q", ["the fact"], [], []) == "the fact"
    assert "brain" in seen


def test_a_stream_that_dies_before_the_first_chunk_streams_offline_instead():
    def dead_stream(goal, contexts, state, memories=None):
        raise ConnectionError("ollama unreachable")
        yield  # pragma: no cover

    seen: dict[str, str] = {}
    stream = fallback_stream(dead_stream, seen.__setitem__)
    assert "".join(stream("q", [], [], [])) == ABSTAIN
    assert "brain" in seen


def test_a_stream_that_stalls_before_the_first_chunk_falls_back_at_the_deadline():
    """Regression, caught live: a thinking model spent minutes reasoning before
    its first token and the unbounded stream just waited. Slow is a failure mode,
    not a different success mode — the deadline falls back like a crash does."""
    import time as _time

    def stalled_stream(goal, contexts, state, memories=None):
        _time.sleep(5)  # far past the test's deadline — must not be waited out
        yield "too late"  # pragma: no cover

    seen: dict[str, str] = {}
    stream = fallback_stream(stalled_stream, seen.__setitem__, timeout=0.05)
    started = _time.monotonic()
    out = "".join(
        stream("what is the refund window", ["refund window is five days"], [], [])
    )
    assert _time.monotonic() - started < 2.0, "the deadline was not enforced"
    assert "refund" in out, "the offline fallback answered from the evidence"
    assert "brain" in seen


def test_a_stream_that_stalls_mid_flight_raises_truncated_rather_than_ending():
    """The chunks already on the wire stay — they cannot be recalled — but the
    stream must not end *normally*, because a normal end means "that was the
    whole answer" to every layer above. This used to `return`, and a model that
    died mid-sentence was served as a complete, cited answer.

    The distinction is the whole test: same text either way, different claim
    about it."""
    import time as _time

    def trickle_then_stall(goal, contexts, state, memories=None):
        yield "first "
        _time.sleep(5)
        yield "never arrives"  # pragma: no cover

    seen: dict[str, str] = {}
    stream = fallback_stream(trickle_then_stall, seen.__setitem__, timeout=0.05)
    out = []
    with pytest.raises(StreamTruncated):
        for chunk in stream("q", [], [], []):
            out.append(chunk)
    assert "".join(out) == "first ", "chunks already on the wire stay"
    assert "brain" in seen, "the truncation is reported, not absorbed"


def test_a_stream_that_errors_mid_flight_also_raises_truncated():
    """A crash after the first chunk is the same situation as a stall: the
    offline fallback cannot answer any more, because half an answer is already
    delivered and the two would not join up."""

    def trickle_then_die(goal, contexts, state, memories=None):
        yield "first "
        raise ConnectionError("ollama went away")

    seen: dict[str, str] = {}
    stream = fallback_stream(trickle_then_die, seen.__setitem__, timeout=1.0)
    with pytest.raises(StreamTruncated):
        list(stream("q", [], [], []))
    assert "brain" in seen


def test_the_degraded_composer_answers_from_whatever_retrieval_kept():
    """The fallback composer trusts retrieval, including about relevance.

    This test used to assert the opposite — that an unrelated snippet produced an
    abstention — and it was enforcing a bug. The check it pinned was word overlap
    between question and document, which is a lexical test standing downstream of
    a semantic retriever: "how quickly do i get money back for a work trip" shares
    no word longer than three characters with "staff are reimbursed for travel
    expenses", so the embedder found the answer and this composer threw it away.

    Deciding relevance is retrieval's job because retrieval has the scores
    (`ASSISTANT_MIN_SCORE`). An empty list means "nothing relevant"; a non-empty
    one means "this is the evidence", and the composer's job is to use it."""
    kept = ["staff are reimbursed for travel expenses within ten business days"]
    answer = offline_compose("how quickly do i get money back for a work trip", kept, [])
    assert "reimbursed" in answer, "a synonym-only hit is exactly what semantic recall is for"
    assert offline_compose("anything at all", [], []) == ABSTAIN


# --- load shedding ------------------------------------------------------------


def test_requests_beyond_the_burst_get_429():
    c = TestClient(create_app(Settings(rate_limit_rps=0.0001, rate_limit_burst=2)))
    assert c.post("/ask", json={"question": "one"}).status_code == 200
    assert c.post("/ask", json={"question": "two"}).status_code == 200
    assert c.post("/ask", json={"question": "three"}).status_code == 429


def test_requests_beyond_the_concurrency_cap_get_503():
    app = create_app(Settings(max_concurrency=1))
    c = TestClient(app)
    cap = app.state.cap
    assert cap.enter(), "occupy the only slot, as an in-flight request would"
    assert c.post("/ask", json={"question": "hello"}).status_code == 503
    cap.leave()
    assert c.post("/ask", json={"question": "hello"}).status_code == 200


def test_health_is_never_shed():
    c = TestClient(create_app(Settings(rate_limit_rps=0.0001, rate_limit_burst=1)))
    c.post("/ask", json={"question": "drain the bucket"})
    assert c.post("/ask", json={"question": "x"}).status_code == 429
    assert c.get("/health").status_code == 200, "probes must keep working under limit"
    assert c.get("/ready").status_code == 200, "so must the readiness probe"


# --- readiness is not liveness -------------------------------------------------


def test_the_offline_tier_is_ready_immediately_because_it_loads_nothing():
    c = TestClient(create_app(Settings()))
    assert c.get("/ready").status_code == 200
    assert c.get("/health").json()["ready"] is True


def test_a_model_tier_that_cannot_answer_is_not_ready():
    """The failure this endpoint exists for. Compose used to call ollama healthy
    once the models were DOWNLOADED; the assistant came up, `/health` said ok,
    and the first question timed out loading a cold 9B and was answered by the
    offline fallback. Every probe was green and the answer was degraded.

    503 is the whole point: an orchestrator reading this does not route to the
    container, and a deploy that never clears it fails loudly instead of
    silently serving fallbacks."""
    app = create_app(Settings(ollama_host="http://unreachable:11434"))
    # Resolved here rather than waiting on the startup thread: the question is
    # what the answer IS, not how soon it lands, and a sleep would test neither.
    app.state.ready, app.state.ready_detail = app.state.assistant.warm()

    c = TestClient(app)
    response = c.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["detail"], "a refusal with no reason is not operable"
    assert c.get("/health").json()["ready"] is False


def test_warm_reports_not_ready_when_the_brain_fell_back():
    """A composer that answers by falling back has not proved the model works —
    it has proved the fallback works. Reporting ready on that is how the cold
    start stayed invisible."""
    assistant = build_assistant(Settings(ollama_host="http://unreachable:11434"))
    assistant.compose = lambda *_args, **_kwargs: "offline prose"
    assistant.degraded["brain"] = "composition fell back to offline: timeout"
    ok, detail = assistant.warm()
    assert ok is False
    assert "degraded" in detail


# --- approvals: one grant, one run; replays are no-ops -------------------------

SEND = {"question": "please message the team about the outage"}


def paused_call(c):
    """The exact call the assistant is waiting on, as the pause reports it."""
    return c.post("/ask", json=SEND).json()["pending"]


def approve(c, pending, headers=None):
    """Approve exactly that call.

    A grant is bound to the arguments, so authorizing a send means echoing back
    the ones the pause reported. That round trip IS the flow — a client that
    cannot name the call it is approving should not be approving it.
    """
    return c.post(
        "/approve",
        json={"tool": pending["tool"], "args": pending["args"]},
        headers=headers or {},
    )


def test_an_approval_authorizes_exactly_one_run():
    c = TestClient(create_app(Settings()))
    approve(c, paused_call(c))
    first = c.post("/ask", json=SEND).json()
    assert "ran: send_telegram" in first["audit"]
    second = c.post("/ask", json=SEND).json()
    assert second["pending"]["tool"] == "send_telegram", (
        "the grant was spent — a new send needs a new approval"
    )


def test_a_replayed_approval_does_not_grant_twice():
    c = TestClient(create_app(Settings()))
    headers = {"Idempotency-Key": "approve-outage-123"}
    pending = paused_call(c)
    first = approve(c, pending, headers).json()
    replay = approve(c, pending, headers).json()
    assert "replayed" not in first
    assert replay["replayed"] is True

    fired = c.post("/ask", json=SEND).json()
    assert "ran: send_telegram" in fired["audit"]
    paused = c.post("/ask", json=SEND).json()
    assert paused["pending"]["tool"] == "send_telegram", (
        "the retried approval must not have granted a second run"
    )


def test_two_distinct_approvals_grant_two_runs():
    c = TestClient(create_app(Settings()))
    pending = paused_call(c)
    approve(c, pending, {"Idempotency-Key": "a1"})
    approve(c, pending, {"Idempotency-Key": "a2"})
    assert "ran: send_telegram" in c.post("/ask", json=SEND).json()["audit"]
    assert "ran: send_telegram" in c.post("/ask", json=SEND).json()["audit"]


# --- idempotency: EVERY side effect, not just the approval ----------------------

DOC = {"docs": [{"text": "approved refunds take five business days", "source": "r.md"}]}


def test_a_retried_ingest_does_not_ingest_twice():
    """The failure this closes is the ordinary one: a client's request times out,
    the client cannot tell whether the server acted, so it asks again — and the
    corpus quietly holds two copies of the same page."""
    c = TestClient(create_app(Settings()))
    headers = {"Idempotency-Key": "batch-7"}
    first = c.post("/ingest", json=DOC, headers=headers).json()
    replay = c.post("/ingest", json=DOC, headers=headers).json()
    assert first["ingested"] == 1
    assert replay["replayed"] is True


def test_a_replay_returns_the_ORIGINAL_answer_not_a_receipt():
    """`{"replayed": true}` alone avoids the double effect and still breaks the
    client, which asked a question and got an acknowledgement. A retry has to be
    indistinguishable from the call it repeats — that is what the word means."""
    c = TestClient(create_app(Settings()))
    headers = {"Idempotency-Key": "batch-8"}
    first = c.post("/ingest", json=DOC, headers=headers).json()
    replay = c.post("/ingest", json=DOC, headers=headers).json()
    assert replay["ingested"] == first["ingested"]
    assert replay["rejected"] == first["rejected"]


def test_the_same_key_on_two_operations_does_not_collide():
    """Keys are chosen by clients, and "retry-1" is what every client picks. A
    single flat namespace lets a retried ingest swallow an approval."""
    c = TestClient(create_app(Settings()))
    headers = {"Idempotency-Key": "retry-1"}
    ingested = c.post("/ingest", json=DOC, headers=headers).json()
    approved = approve(c, paused_call(c), headers).json()
    assert ingested["ingested"] == 1
    assert "replayed" not in approved, "a different operation is a different key"
    assert approved["approval_id"]


def test_a_failed_operation_leaves_the_key_retryable():
    """Recording the key up front turns one transient failure into a permanent
    one: every retry cheerfully acknowledged, the effect never applied, nothing
    in the logs saying so."""
    app = create_app(Settings())
    a = app.state.assistant
    calls = {"n": 0}
    real_ingest = a.ingest

    def flaky_ingest(docs, subject):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("the index was not reachable")
        return real_ingest(docs, subject)

    a.ingest = flaky_ingest
    c = TestClient(app, raise_server_exceptions=False)
    headers = {"Idempotency-Key": "batch-9"}
    assert c.post("/ingest", json=DOC, headers=headers).status_code == 500
    assert c.post("/ingest", json=DOC, headers=headers).json()["ingested"] == 1


# --- the model call: bounded, or the request has no duration ---------------------


def fake_ollama(monkeypatch, seen):
    """Stand in for the `ollama` package, which the fast tier does not install.

    The adapters import it lazily precisely so this is possible: the shape of the
    request we send a model is worth testing without a model, because it is a
    property of our code and not of theirs."""

    class FakeClient:
        def __init__(self, host, timeout=None):
            seen["host"] = host

        def generate(self, **kwargs):
            seen.update(kwargs)
            if kwargs.get("stream"):
                return iter([{"response": "five business "}, {"response": "days"}])
            return {"response": "  five business days\n"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    return seen


def test_a_completion_is_asked_for_in_bounded_form(monkeypatch):
    """This is the fix for the slowest bug in the repo's history, and it never
    looked like a bug: every answer was correct, `/health` was green, and a full
    end-to-end pass took twenty-five minutes because the model was asked for an
    unbounded reply and abandoned at the 60-second budget every single time. The
    measurement was 667 reasoning tokens at 0.52 tokens/second for a one-sentence
    answer. Nothing failed. It was just never the model answering."""
    seen = fake_ollama(monkeypatch, {})
    from assistant.adapters import COMPLETION_TOKEN_CAP, ollama_generate

    out = ollama_generate("how long do refunds take", host="http://ollama:11434", model="m")
    assert out == "five business days"
    assert seen["think"] is False, "a reasoning model spends the whole budget deliberating"
    assert seen["options"]["num_predict"] == COMPLETION_TOKEN_CAP, (
        "an unbounded generation is a request whose duration the model decides"
    )


def test_the_stream_is_bounded_the_same_way(monkeypatch):
    """Sharper here than in the batch path: a client watching SSE would sit
    through the deliberation before the first word of the answer, and a stream
    whose first chunk is minutes away is not a stream."""
    seen = fake_ollama(monkeypatch, {})
    from assistant.adapters import COMPLETION_TOKEN_CAP, ollama_stream

    streamed = "".join(ollama_stream("q", host="http://ollama:11434", model="m"))
    assert streamed == "five business days"
    assert seen["think"] is False
    assert seen["options"]["num_predict"] == COMPLETION_TOKEN_CAP


def test_a_buffered_completion_stops_generating_when_the_caller_is_gone(monkeypatch):
    """The buffered path is its own stream joined, and this is the reason.

    A blocking `Client.generate()` is a black box: the request budget says the caller
    left, and there is nowhere between "asked" and "answered" to notice. The model
    generated to the last token for a closed socket, on hardware somebody pays for —
    `tests/test_disconnect.py` proves that the request stops waiting, and stopping
    the *work* is the other half.

    Two assertions, and the second is the one that matters. Raising `Expired` only
    proves this thread gave up. `closed` proves the provider's iterator was released,
    which is what tears down the HTTP response and makes Ollama stop. An abandoned
    iterator would satisfy the first assertion and keep the GPU busy — measured in
    `phase8-deploy/VERIFIED.md` at 395% CPU with every later request queued behind it.

    No socket, no model, no server: the seam is a boolean the budget already carries.
    """
    from assistant import deadline
    from assistant.adapters import ollama_generate

    watched = {"parts": 0, "closed": False}
    gone = {"yet": False}

    class Stream:
        """A provider stream that never ends, and knows whether it was closed."""

        def __iter__(self):
            return self

        def __next__(self):
            watched["parts"] += 1
            # The caller leaves after the answer is under way, which is the only
            # interesting moment: before the first part there is nothing to stop.
            if watched["parts"] == 3:
                gone["yet"] = True
            return {"response": f"token{watched['parts']} "}

        def close(self):
            watched["closed"] = True

    class FakeClient:
        def __init__(self, host, timeout=None):
            pass

        def generate(self, **kwargs):
            return Stream()

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    with deadline.budget(None, cancelled=lambda: gone["yet"]):
        with pytest.raises(deadline.Expired) as raised:
            ollama_generate("q", host="http://ollama:11434", model="m")

    assert raised.value.reason == "caller disconnected", (
        "a 504 for a client that hung up pages somebody for a request nobody wanted"
    )
    assert watched["closed"], (
        "the provider's stream was abandoned, not closed — the request returns and "
        "the model keeps generating, which is the cost this change exists to stop"
    )
    assert watched["parts"] < 10, "the answer kept being drained after the hangup"


def test_a_streamed_answer_stops_being_produced_when_the_caller_is_gone():
    """The same claim one layer up, where the seam above was quietly missing.

    `fallback_stream` drains the composer on a worker thread, and a thread starts with
    an EMPTY context. So the budget installed below was invisible in the only place
    that was actually pulling the model: `deadline.current()` inside the drain handed
    back a fresh unbounded `Budget` whose `cancelled` is `lambda: False`, every seam
    the adapters gained answered "nobody has left", and generation ran on. The
    round-8 audit measured it over a socket — 4 chunks at disconnect, 41 and counting
    afterwards.

    Three assertions, because three separate things had to be true and only the first
    is about giving up:

      * the consumer sees `Expired`, so the request ends as a hangup rather than as an
        answer or a fallback;
      * the source was CLOSED, which is what stops a provider rather than stopping
        our reading of one;
      * nothing was reported as brain degradation. A client closing a tab that shows
        up on /health as a model outage would fail the release lane's own no-fallback
        assertion, and page somebody for it.

    The socket version lives in `tests/test_disconnect.py`, and this is the version
    that localises a break: if both fail, cancellation stopped working; if only that
    one fails, the HTTP layer stopped delivering the signal.

    The composer sleeps between chunks, and without that this test passes on the
    broken code. The queue between the two threads is unbounded, so a source with no
    delay produces all five hundred chunks before the consumer has read three and
    there is no window left to cancel in — the property under test only exists while
    the model is still generating, which is the entire duration of a real one.
    """
    from assistant import deadline
    from assistant.fallbacks import fallback_stream

    LENGTH = 500
    PACE = 0.002  # a token is not free; see the docstring
    pulled = {"chunks": 0, "closed": False}
    gone = {"yet": False}
    degraded: dict[str, str] = {}

    def endless(goal, contexts, state, memories=None):
        """A composer that cooperates in nothing — no deadline check of its own."""
        try:
            for i in range(LENGTH):
                pulled["chunks"] += 1
                yield f"word{i} "
                time.sleep(PACE)
        finally:
            pulled["closed"] = True

    streamed = fallback_stream(endless, lambda part, why: degraded.update({part: why}))

    with deadline.budget(None, cancelled=lambda: gone["yet"]):
        frames = streamed("tell me everything", [], [], [])
        with pytest.raises(deadline.Expired) as raised:
            for seen, _ in enumerate(frames, start=1):
                if seen == 3:
                    gone["yet"] = True  # the caller leaves mid-answer

    assert raised.value.reason == "caller disconnected"
    assert pulled["chunks"] < LENGTH // 2, (
        f"{pulled['chunks']} of {LENGTH} chunks were produced for a caller who left — "
        "the drain thread cannot see a budget it was never given"
    )
    assert pulled["closed"], "the composer was abandoned rather than closed"
    assert degraded == {}, f"a hangup was recorded as degradation: {degraded}"


def test_an_upgrade_over_an_older_table_still_replays(tmp_path):
    """The bug this closes cost an end-to-end run: `CREATE TABLE IF NOT EXISTS`
    does nothing when the table exists, including when it exists with the wrong
    shape. A volume written by the build that only stored keys kept its
    two-column table, the new code's `UPDATE ... SET result` raised `no such
    column: result`, and it raised on the RETRY path — the health check was
    green, every read worked, and the first client to time out and ask again got
    a 500 for its trouble.

    Red until TODO 1 migrates the table rather than only redefining it. Every
    other test here starts from an empty database, which is the one case a
    missing migration survives; this one starts from the schema the scaffold
    ships, which is the schema in the Docker volume of anybody who ran the stack
    before the change. That is why the schema is compared to the database rather
    than assumed: the only honest test of a migration is one that starts from
    the old schema."""
    db = tmp_path / "assistant.db"
    old = sqlite3.connect(db)
    old.execute(
        "CREATE TABLE idempotency_keys ("
        "  key TEXT PRIMARY KEY,"
        "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    old.execute("INSERT INTO idempotency_keys (key) VALUES ('written-by-the-old-build')")
    old.commit()
    old.close()

    store = IdempotencyStore(str(db))
    result, replayed = store.run("batch-upgrade", lambda: {"ingested": 1})
    assert (result, replayed) == ({"ingested": 1}, False)
    assert store.run("batch-upgrade", lambda: {"ingested": 1}) == ({"ingested": 1}, True), (
        "the replay has to carry the original answer, which is what needs the column"
    )
    # The rows from before the upgrade survive it: a migration that protects the
    # new path by dropping the table would un-protect every key already claimed.
    assert store.run("written-by-the-old-build", lambda: {"ingested": 99})[1] is True


def test_a_retried_delete_is_acknowledged_not_reapplied():
    c = TestClient(create_app(Settings()))
    c.post("/ingest", json=DOC)
    headers = {"Idempotency-Key": "forget-1"}
    first = c.request("DELETE", "/corpus/r.md", headers=headers).json()
    replay = c.request("DELETE", "/corpus/r.md", headers=headers).json()
    assert first["deleted"] == 1
    assert replay["deleted"] == 1 and replay["replayed"] is True, (
        "the answer is the original one, and no second delete happened"
    )


# --- the outbox: irreversible intent, written down first ------------------------


def failing_send(*_args, **_kwargs):
    raise ConnectionError("telegram unreachable")


def crashing_send(*_args, **_kwargs):
    raise KeyboardInterrupt("the process was killed mid-send")


def recorded(fn, outbox: Outbox, request_id: str = "req-1"):
    """The send_telegram tool with `fn` for a body, wrapped by the outbox."""
    registry = dict(REGISTRY)
    registry["send_telegram"] = rewrap(registry["send_telegram"], fn)
    return recorded_registry(registry, outbox, "alice", request_id)["send_telegram"].fn


def test_the_intent_is_durable_before_the_call_happens():
    """The gap this closes: a crash between "the API accepted it" and "we wrote
    the row" leaves a message that went out and nothing in the system that
    remembers. The row is committed FIRST, so the worst case is a false alarm
    somebody can close rather than an effect nobody can account for."""
    outbox = Outbox()
    seen: list[str] = []

    def send(chat_id: str, message: str) -> dict:
        seen.append(str(outbox.pending()[0].status))
        return {"sent": True}

    recorded(send, outbox)(chat_id="ops", message="the site is down")
    assert seen == [PENDING], "the row existed while the call was in flight"
    assert [e.status for e in outbox.entries("alice")] == [SENT]


def test_a_process_that_dies_mid_send_leaves_a_pending_row():
    """Not an exception the tool can catch — a kill. `pending` is the honest
    state: we started something irreversible and never learned how it ended.
    That is a question for a human, which is strictly better than silence."""
    outbox = Outbox()
    with pytest.raises(KeyboardInterrupt):
        recorded(crashing_send, outbox)(chat_id="ops", message="hi")
    assert [e.status for e in outbox.pending()] == [PENDING]


def test_a_failed_send_is_written_down_as_failed():
    """Without this the row stays pending forever and reconciliation cannot tell
    "crashed mid-send" from "the API said no" — which are opposite answers to
    the only question being asked."""
    outbox = Outbox()
    with pytest.raises(ConnectionError):
        recorded(failing_send, outbox)(chat_id="ops", message="hi")
    assert [e.status for e in outbox.entries("alice")] == [FAILED]
    assert outbox.pending() == [], "a known failure is not an open question"


def test_a_read_only_tool_gets_no_outbox_row():
    """An envelope for a search is bookkeeping nobody will ever read."""
    outbox = Outbox()
    wrapped = recorded_registry(dict(REGISTRY), outbox, "alice", "req-1")
    wrapped["read_news"].fn()
    assert outbox.entries() == []


def test_the_same_call_twice_in_one_request_is_refused_not_sent_twice():
    """The last line of defence, and the only one holding the actual arguments.
    A retry that got past every layer above stops here."""
    outbox = Outbox()
    sends: list[str] = []
    send = recorded(lambda chat_id, message: sends.append(message) or {"sent": True},
                    outbox)
    send(chat_id="ops", message="the site is down")
    again = send(chat_id="ops", message="the site is down")
    assert len(sends) == 1
    assert "already recorded" in again["error"]


def test_a_deliberate_second_send_in_a_new_request_goes_through():
    """`request_id` is part of the key on purpose: two deliberate sends of the
    same message are two requests, and both should happen. A dedupe that cannot
    tell "retry" from "again" is a dedupe that loses messages."""
    outbox = Outbox()
    sends: list[str] = []

    def send(chat_id: str, message: str) -> dict:
        sends.append(message)
        return {"sent": True}

    recorded(send, outbox, "req-1")(chat_id="ops", message="ping")
    recorded(send, outbox, "req-2")(chat_id="ops", message="ping")
    assert len(sends) == 2


def test_an_approved_send_shows_up_in_the_outbox_over_http():
    c = TestClient(create_app(Settings()))
    approve(c, paused_call(c))
    c.post("/ask", json=SEND)
    body = c.get("/outbox").json()
    assert [e["tool"] for e in body["effects"]] == ["send_telegram"]
    assert body["effects"][0]["status"] == SENT
    assert body["pending"] == 0
    assert body["effects"][0]["request_id"], "an effect is traceable to its request"


# --- deadlines and callers who left ---------------------------------------------


def test_a_request_that_blows_its_deadline_gets_504():
    """A gateway timeout is an alert: the service was too slow and somebody
    should look at it."""
    c = TestClient(create_app(Settings(request_deadline=0.0)))
    response = c.post("/ask", json={"question": "how long do refunds take"})
    assert response.status_code == 504
    assert response.json()["detail"] == "deadline exceeded"


def test_a_caller_who_left_gets_499_and_not_an_alert():
    """499 is nginx's "client closed request". Nobody is there to receive it,
    and paging on it means paging every time a user closes a tab."""
    app = create_app(Settings())
    app.state.assistant.screen = lambda text: (_ for _ in ()).throw(
        deadline.Expired("caller disconnected")
    )
    c = TestClient(app)
    response = c.post("/ask", json={"question": "anything"})
    assert response.status_code == 499


def test_an_unbounded_request_is_still_the_default():
    """The fast tier drives a real model in the integration lane, where a
    deadline would be a flake generator. Off unless configured."""
    c = TestClient(create_app(Settings()))
    assert c.post("/ask", json={"question": "hello"}).status_code == 200


def test_a_stream_stops_early_when_nobody_is_listening():
    """Per FRAME, because that is the only place a long generation can be
    stopped at all. Under load this is the failure that compounds: tokens
    generated for callers who left are what slow down the callers who stayed."""
    app = create_app(Settings())
    a = app.state.assistant
    frames: list[str] = []
    gone = {"yet": False}

    def endless(goal, contexts, state, memories=None):
        for i in range(1000):
            yield f"word{i} "

    a.stream_compose = endless
    with deadline.budget(None, cancelled=lambda: gone["yet"]):
        for frame in a.ask_stream("tell me everything"):
            frames.append(frame)
            if len(frames) >= 3:
                gone["yet"] = True
    assert len(frames) < 1000, "the stream stopped when the caller did"
    root = [s for s in a.rec.spans() if s.name == "assistant.pipeline"][-1]
    assert root.attributes["request.abandoned"] == "caller disconnected", (
        "and the trace says why, so the short answer is not a mystery"
    )


def test_an_abandoned_stream_still_ends_in_a_done_frame_that_says_so():
    """Stopping early is right. Stopping in silence is not.

    Found by the self-contained end-to-end lane, where a request budget expired
    mid-generation: the client got chunks and then nothing, which on the wire is
    exactly what a dropped connection looks like — so the half answer already on
    screen reads as the whole one. Same contract as a mid-stream stall: the
    stream ends in `done`, and `done` says it was cut off."""
    app = create_app(Settings())
    a = app.state.assistant
    frames: list[str] = []
    gone = {"yet": False}

    def endless(goal, contexts, state, memories=None):
        for i in range(1000):
            yield f"word{i} "

    a.stream_compose = endless
    with deadline.budget(None, cancelled=lambda: gone["yet"]):
        for frame in a.ask_stream("tell me everything"):
            frames.append(frame)
            if len(frames) >= 3:
                gone["yet"] = True

    last = frames[-1]
    assert last.startswith("event: done"), "a stream that stops must still end"
    body = json.loads(last.split("data: ", 1)[1])
    assert body["truncated"] is True
    assert body["abandoned"] == "caller disconnected"
    # the text that did reach the client is in the frame, so a caller can keep
    # what arrived without pretending it was the answer
    assert body["answer"].startswith("word0")

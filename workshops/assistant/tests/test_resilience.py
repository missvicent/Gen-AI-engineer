"""The reliability primitives, in isolation and in microseconds: injectable
sleep/clock means no test here ever actually waits."""

import time
from contextvars import ContextVar

import pytest

from assistant import deadline
from assistant.resilience import (
    ONCE,
    ConcurrencyCap,
    Permanent,
    Policy,
    TokenBucket,
    resilient,
)


class Flaky:
    """Fails `failures` times, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("transient")
        return "ok"


class Broken:
    """Fails the same way every time, for a reason time will not fix."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        raise self.exc


def test_a_transient_failure_is_retried_to_success():
    slept: list[float] = []
    fn = resilient(
        Flaky(failures=2), Policy(attempts=3, base_delay=0.2, timeout=None),
        sleep=slept.append, rng=lambda: 1.0,
    )
    assert fn() == "ok"
    assert slept == [0.2, 0.4], "exponential backoff between the three attempts"


def test_the_last_failure_propagates_when_attempts_run_out():
    fn = resilient(
        Flaky(failures=99), Policy(attempts=3, base_delay=0, timeout=None),
        sleep=lambda s: None,
    )
    with pytest.raises(ConnectionError):
        fn()


def test_backoff_is_capped_and_jittered():
    slept: list[float] = []
    fn = resilient(
        Flaky(failures=3), Policy(attempts=4, base_delay=1.0, max_delay=1.5, timeout=None),
        sleep=slept.append, rng=lambda: 0.5,
    )
    fn()
    # ceilings 1.0, 2.0->1.5(cap), 4.0->1.5(cap); jitter halves each
    assert slept == [0.5, 0.75, 0.75]


def test_what_the_call_records_about_itself_survives_the_worker_thread():
    """The timeout runs the call on a worker thread, and a thread starts with an
    empty context — so anything the call wrote to a ContextVar was thrown away
    when that thread ended.

    It was not a hypothetical. Ollama returns exact token counts, the adapter
    reports them into a ContextVar, and the meter read that var back on the
    request thread: `None`, every time, on the one tier where the counts are
    real. The release page then said the tokens were estimated by word split,
    which was true, honest, and caused entirely by this."""
    seen: ContextVar[str] = ContextVar("seen", default="nothing")

    def records() -> str:
        seen.set("what the call learned")
        return "done"

    assert resilient(records, Policy(attempts=1, timeout=5))() == "done"
    assert seen.get() == "what the call learned"


def test_an_abandoned_call_does_not_get_to_write_into_the_caller():
    """The other half, and the reason the copy-back happens after the result
    rather than in a `finally`: a call that overran is still running on that
    thread. Adopting whatever it has written so far would import half-finished
    state from work the caller has already given up on."""
    late: ContextVar[str] = ContextVar("late", default="nothing")

    def slow_writer() -> None:
        time.sleep(0.3)
        late.set("too late to matter")

    with pytest.raises(TimeoutError):
        resilient(slow_writer, Policy(attempts=1, timeout=0.05))()
    time.sleep(0.4)  # the abandoned call has now finished, on its own thread
    assert late.get() == "nothing"


def test_a_call_that_overruns_its_timeout_raises():
    def slow() -> None:
        time.sleep(0.5)

    fn = resilient(slow, Policy(attempts=1, timeout=0.05))
    with pytest.raises(TimeoutError):
        fn()


def test_a_timeout_returns_promptly_instead_of_waiting_for_the_abandoned_call():
    """Regression, caught live: ThreadPoolExecutor as a context manager shuts down
    with wait=True, so the 'timed out' call still blocked until the stuck call
    finished — a 60s budget held /ask hostage for half an hour. The timeout must
    RETURN at the deadline; the overrun call is abandoned on its worker thread."""

    def stuck() -> None:
        time.sleep(1.5)  # far past the timeout — the caller must not wait this out

    fn = resilient(stuck, Policy(attempts=1, timeout=0.05))
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        fn()
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"timeout waited {elapsed:.2f}s for the abandoned call"


# --- what is worth retrying ---------------------------------------------------


def test_a_bug_is_not_retried_three_times():
    """`retry on Exception` turns one deterministic bug into three identical
    failures and 3x the latency before the same 500. A TypeError will be a
    TypeError next time; it surfaces on attempt one, where the traceback still
    points at the cause."""
    broken = Broken(TypeError("wrong argument"))
    fn = resilient(broken, Policy(attempts=3, base_delay=0, timeout=None))
    with pytest.raises(TypeError):
        fn()
    assert broken.calls == 1


def test_a_permanent_refusal_stops_the_loop_even_though_it_looks_like_a_network_error():
    """An adapter that can read a status code says so. A 400 from an API that
    has already given its final answer is not made truer by asking twice."""
    broken = Broken(Permanent("the server said 422"))
    fn = resilient(broken, Policy(attempts=3, base_delay=0, timeout=None))
    with pytest.raises(Permanent):
        fn()
    assert broken.calls == 1


def test_once_never_makes_a_second_attempt():
    """The policy for a call that is not idempotent. A timeout on a send tells
    you nothing about whether it was delivered, so a retry is how one message
    becomes two."""
    flaky = Flaky(failures=1)
    with pytest.raises(ConnectionError):
        resilient(flaky, ONCE)()
    assert flaky.calls == 1


# --- the request budget ---------------------------------------------------------


def test_a_per_call_timeout_shrinks_to_what_the_request_has_left():
    """The whole reason the budget is shared. A 60-second call timeout inside a
    request with 4 seconds left is a 4-second call timeout — otherwise every
    layer's timeout composes by ADDITION and the total is a number nobody has
    computed."""
    now = {"t": 100.0}
    with deadline.budget(4.0, clock=lambda: now["t"]):
        assert deadline.capped(60.0) == 4.0
        assert deadline.capped(1.0) == 1.0, "the tighter of the two wins, both ways"
        assert deadline.capped(None) == 4.0, "no policy timeout still gets the budget"
    assert deadline.capped(60.0) == 60.0, "outside a budget, the policy stands alone"


def test_no_attempt_starts_after_the_deadline():
    """Checked BEFORE the call, not after: starting work we already know we
    cannot wait for burns a connection to produce a result nobody reads."""
    now = {"t": 0.0}
    flaky = Flaky(failures=0)
    with deadline.budget(1.0, clock=lambda: now["t"]):
        now["t"] = 2.0
        with pytest.raises(deadline.Expired):
            resilient(flaky, Policy(attempts=3, timeout=None))()
    assert flaky.calls == 0


def test_backoff_does_not_sleep_past_the_deadline():
    """Sleeping through the rest of the budget in order to attempt a call that
    cannot finish is the most expensive possible way to fail."""
    now = {"t": 0.0}
    slept: list[float] = []
    with deadline.budget(0.1, clock=lambda: now["t"]):
        with pytest.raises(ConnectionError):
            resilient(
                Flaky(failures=99), Policy(attempts=5, base_delay=1.0, timeout=None),
                sleep=slept.append, rng=lambda: 1.0,
            )()
    assert slept == [], "a 1s backoff inside a 0.1s budget is never taken"


def test_a_disconnected_caller_is_reported_before_a_deadline():
    """Both end the request; they are not the same incident. A deadline means
    the service was too slow and someone should look. A disconnect means the
    work was pointless, and paging on it means paging every closed tab."""
    now = {"t": 0.0}
    with deadline.budget(1.0, cancelled=lambda: True, clock=lambda: now["t"]) as b:
        assert b.expired() == "caller disconnected"
        with pytest.raises(deadline.Expired) as caught:
            b.check()
    assert caught.value.reason == "caller disconnected"


def test_an_unbounded_request_reports_no_time_rather_than_infinite_time():
    """None, not a very large float. A caller that wants to say "no limit"
    downstream needs to be able to tell that apart from "lots of time left"."""
    assert deadline.remaining() is None
    assert deadline.expired() is None
    deadline.check()  # must not raise


def test_the_budget_does_not_leak_into_the_next_request():
    """Worker threads are reused. A budget that outlives its request expires
    the next one instantly, which looks exactly like an outage."""
    now = {"t": 0.0}
    with deadline.budget(5.0, clock=lambda: now["t"]):
        assert deadline.remaining() == 5.0
    assert deadline.remaining() is None


def test_the_bucket_allows_the_burst_then_refuses():
    now = {"t": 0.0}
    bucket = TokenBucket(rate=1.0, burst=2.0, clock=lambda: now["t"])
    assert bucket.take() and bucket.take()
    assert not bucket.take(), "burst exhausted, no time has passed"


def test_the_bucket_refills_with_time():
    now = {"t": 0.0}
    bucket = TokenBucket(rate=2.0, burst=2.0, clock=lambda: now["t"])
    bucket.take(), bucket.take()
    now["t"] = 0.5  # 0.5s * 2/s = 1 token back
    assert bucket.take()
    assert not bucket.take()


def test_the_cap_admits_up_to_the_limit_and_sheds_beyond_it():
    cap = ConcurrencyCap(limit=2)
    assert cap.enter() and cap.enter()
    assert not cap.enter(), "third concurrent request is shed"
    cap.leave()
    assert cap.enter(), "a slot freed is a slot usable"
    assert cap.active == 2

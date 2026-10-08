"""Regression tests for the three defects — reference versions.

Every test here must FAIL when its defect is seeded (`DEFECT=...`) and PASS
against the current code. `make defect-lab` enforces both directions, in that
order, because a regression test that has never been red is a regression test
you know nothing about.

Note what is *not* asserted. None of these reaches for an implementation
detail — no counting of SQL statements, no checking that `ApprovalStore` has a
`_lock`. They state the property a user cares about: Bob cannot spend Alice's
approval; two racing requests send one message; nothing crosses the boundary
before the gate sees it; the image is built from a pinned lockfile. A test
written against the fix rather than the property goes green on the next
refactor and stays green through the next regression.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from assistant import output_gate
from defects import harness

# --- Defect 1: unbound approval -------------------------------------------------


def test_unbound_approval_one_subjects_yes_does_not_authorize_another():
    """Alice approves a send. Bob's identical request must still pause.

    The counter store fails here because its grant was keyed by tool name alone
    and never recorded who asked.
    """
    store = harness.approval_store()
    store.mint("alice", "send_telegram", {"chat_id": "team"})
    assert store.consume("bob", "send_telegram", {"chat_id": "team"}) is None


def test_unbound_approval_does_not_stretch_to_different_arguments():
    """Approving a message to the team is not approving one to the press."""
    store = harness.approval_store()
    store.mint("alice", "send_telegram", {"chat_id": "team"})
    assert store.consume("alice", "send_telegram", {"chat_id": "the-press"}) is None


def test_unbound_approval_is_spent_exactly_once_under_concurrency():
    """Sixteen threads race for one approval. One send, fifteen pauses.

    This is the assertion the original could not pass and the reason the fix is
    a single `DELETE ... RETURNING` rather than a check followed by a decrement.
    Note it asserts the *outcome* — how many callers got a grant — not how the
    store is implemented.
    """
    store = harness.approval_store()
    store.mint("alice", "send_telegram", {"chat_id": "team"})
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _: store.consume("alice", "send_telegram", {"chat_id": "team"}),
                range(16),
            )
        )
    assert sum(1 for r in results if r is not None) == 1


def test_unbound_approval_expires():
    """A forgotten yes must not authorize tomorrow's run."""
    store = harness.approval_store(ttl_seconds=-1.0)
    store.mint("alice", "send_telegram", {"chat_id": "team"})
    assert store.consume("alice", "send_telegram", {"chat_id": "team"}) is None


# --- Defect 2: content released before the gate ---------------------------------

SSN = "123-45-6789"


def test_pre_gate_stream_never_emits_unsafe_content_before_the_verdict():
    """The property: no chunk leaves the process containing what the gate blocks.

    The seeded mode fails because it yields each chunk as it arrives and screens
    the joined answer afterwards — by which time the number is in the caller's
    browser and their proxy logs. A `blocked` event arriving later is a
    notification, not a control.
    """
    chunks = iter(["Your record: ", SSN, " — filed."])
    released = [
        text
        for kind, text in output_gate.gated_chunks(chunks, mode=harness.stream_mode())
        if kind == "chunk"
    ]
    assert SSN not in "".join(released)


def test_pre_gate_stream_still_streams_a_safe_answer_in_pieces():
    """The fix must not be "buffer everything". A long clean answer still
    arrives incrementally, or we traded a real vulnerability for a fake fix."""
    long_answer = ["word " * 40] * 6
    events = list(output_gate.gated_chunks(iter(long_answer), mode=output_gate.SAFE_BUFFERED))
    assert sum(1 for kind, _ in events if kind == "chunk") > 1
    assert events[-1][0] == "done"


# --- Defect 3: the image built without its lockfile ------------------------------


def test_lockless_build_installs_from_a_committed_lockfile():
    """The property: the image's dependencies are the ones that were tested.

    `uv sync` without `--frozen` re-resolves at build time, so rebuilding an
    unchanged commit can produce different code — and nothing in the diff, the
    tag or the logs says so.
    """
    text = harness.dockerfile()
    assert "uv.lock" in text, "the lockfile is never copied into the build stage"
    install = next(line for line in text.splitlines() if "uv sync" in line)
    assert "--frozen" in install, f"install re-resolves at build time: {install.strip()}"


def test_lockless_build_pins_the_base_images_by_tag():
    """A floating base image is the same defect one layer down: the bytes change
    under an unchanged Dockerfile."""
    for line in harness.dockerfile().splitlines():
        if line.startswith("FROM "):
            image = line.split()[1]
            assert ":" in image and not image.endswith(":latest"), image


def test_the_lab_covers_every_seeded_defect():
    """A defect with no regression test is a defect waiting to come back."""
    assert set(harness.DEFECTS) == {"unbound-approval", "pre-gate-stream", "lockless-build"}
    source = open(__file__).read()
    for defect in harness.DEFECTS:
        assert re.search(rf"def test_{defect.replace('-', '_')}_", source), defect

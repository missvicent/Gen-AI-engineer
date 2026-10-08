"""TODO: write the regression tests that catch three defects this code shipped.

`defects/variants.py` holds the vulnerable versions, running, with their real
interfaces. `make defect-lab` runs this file twice per defect, in this order:

1. Against the current code, with `DEFECT` unset. Your tests must **pass**.
2. With that defect seeded (`DEFECT=<name>`). The same tests must **fail**.

Green first, then red. A red run only counts if the test was able to pass.
An unwritten test, a bad import, or a collection error also looks red, and
those are not a caught defect.

## Why the order is the lesson

Write the assertion before you study the fix. A test written after the fix is
green the moment it exists, so it never shows that it *can* go red. The lab
is what tells a real regression test apart from one that asserts something
beside the point.

## Assert the property, not the fix

None of these should reach for an implementation detail — do not count SQL
statements, do not check that `ApprovalStore` has a `_lock`. State what a user
cares about: Bob cannot spend Alice's approval, two racing requests send one
message, nothing crosses the boundary before the gate sees it, the image is
built from a pinned lockfile. A test written against the *fix* goes green on
the next refactor and stays green through the next regression.

## The three defects

1. **unbound-approval** — `defects.variants.CountingApprovalStore`, a counter
   keyed by tool name. Get it through `harness.approval_store()`. Four separate
   holes: any subject, any arguments, no expiry, and a read-modify-write two
   callers can both win. Four tests, one per hole; the names must start with
   `test_unbound_approval_`.
2. **pre-gate-stream** — emit-then-judge streaming, still in the codebase as
   `output_gate.RAW`. Get the mode from `harness.stream_mode()`. Prove no chunk
   containing blocked content is released — and then prove the fix did not just
   buffer everything, or you have traded a real hole for a fake fix. Names start
   with `test_pre_gate_stream_`.
3. **lockless-build** — an image built with no lockfile. Get the text from
   `harness.dockerfile()`. Prove the build installs from a committed lock, and
   that the base images are pinned. Names start with `test_lockless_build_`.

`test_the_lab_covers_every_seeded_defect` below checks the naming, so a defect
you skipped cannot pass unnoticed.

Reference: ../../after/defects/test_regressions.py — after you have been red.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor  # noqa: F401  (you will need it)

from assistant import output_gate  # noqa: F401
from defects import harness

SSN = "123-45-6789"


def test_unbound_approval_one_subjects_yes_does_not_authorize_another():
    raise NotImplementedError  # TODO 1


def test_unbound_approval_does_not_stretch_to_different_arguments():
    raise NotImplementedError  # TODO 2


def test_unbound_approval_is_spent_exactly_once_under_concurrency():
    raise NotImplementedError  # TODO 3


def test_unbound_approval_expires():
    raise NotImplementedError  # TODO 4


def test_pre_gate_stream_never_emits_unsafe_content_before_the_verdict():
    raise NotImplementedError  # TODO 5


def test_pre_gate_stream_still_streams_a_safe_answer_in_pieces():
    raise NotImplementedError  # TODO 6


def test_lockless_build_installs_from_a_committed_lockfile():
    raise NotImplementedError  # TODO 7


def test_lockless_build_pins_the_base_images_by_tag():
    raise NotImplementedError  # TODO 8


def test_the_lab_covers_every_seeded_defect():
    """A defect with no regression test is a defect waiting to come back."""
    assert set(harness.DEFECTS) == {
        "unbound-approval",
        "pre-gate-stream",
        "lockless-build",
    }
    source = open(__file__).read()
    for defect in harness.DEFECTS:
        assert re.search(rf"def test_{defect.replace('-', '_')}_", source), defect

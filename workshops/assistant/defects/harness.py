"""Swap the fixed component for the seeded one, chosen by `$DEFECT`.

The lab runs each regression test twice against the same file: once with
`DEFECT` set, where it must fail, and once without, where it must pass. That is
the whole mechanism, and it is the only thing that distinguishes a regression
test from a test that happens to be green.

Writing the test after the fix is the habit this defends against. Such a test
is green the moment you write it, so it never demonstrates that it can go red,
and a green test that has never failed is a green test you have no evidence
about. Half of them are asserting something subtly beside the point.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFECTS = ("unbound-approval", "pre-gate-stream", "lockless-build")


def seeded(name: str) -> bool:
    """Is this defect currently seeded in?"""
    if name not in DEFECTS:
        raise ValueError(f"no such defect: {name!r} (have {', '.join(DEFECTS)})")
    return os.getenv("DEFECT") == name


def approval_store(**kwargs: Any):
    """The real store, or the counter that shipped before it."""
    if seeded("unbound-approval"):
        from defects.variants import CountingApprovalStore

        return CountingApprovalStore(**kwargs)
    from assistant.approvals import ApprovalStore

    return ApprovalStore(**kwargs)


def stream_mode() -> str:
    """The default outbound mode, or the emit-then-judge one."""
    from assistant import output_gate

    return output_gate.RAW if seeded("pre-gate-stream") else output_gate.SAFE_BUFFERED


def dockerfile() -> str:
    """The committed Dockerfile, or the one that resolved deps at build time."""
    if seeded("lockless-build"):
        from defects.variants import UNLOCKED_DOCKERFILE

        return UNLOCKED_DOCKERFILE
    return (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()

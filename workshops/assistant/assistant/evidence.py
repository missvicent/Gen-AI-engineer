"""TODO: `make evidence` — the engineering evidence log for the whole course.

`report.py` measures this service. This measures **the course**: six dimensions
a reviewer actually asks about — quality, latency, cost, security, failure
recovery, and the decisions behind them — collected across all nine phases into
one page and one JSON manifest.

The design decision that makes it worth writing, and the one you will be tempted
to undo: **the default for every claim is `unproven`, and an unproven claim is
printed rather than skipped.**

That is what separates a manifest from a checklist. A checklist records what you
say you did — tick nine boxes and it reports nine phases complete, whether you
ran anything or not, which means it reports the same thing for someone who did
the work and someone who did not, and therefore says nothing about either. This
records what left a file behind. Never ran the phase-6 red-team? The security
section says so, with the command that would fix it. There is no way to tick your
way to a green page.

Say the consequence out loud before you start, or you will think it is broken:
**your first run will be almost entirely red.** You have not generated the
artifacts yet. A tool that flattered you on day one would have nothing left to
tell you on day ninety.

TODO 1 — `Claim` and the `CLAIMS` registry. One entry per thing the course asks
  you to be able to show. Store `lesson` (directory) and `target` (make target)
  as separate fields rather than one command string: a test walks the repo and
  checks that every command this page prints names a directory that exists and a
  target that runs. An evidence log citing a lesson renamed six commits ago is
  the one kind of wrong this module has no excuse for.

TODO 2 — `read_artifact`. Three outcomes, and the middle one is the interesting
  one. Missing file: `unproven`, with a note. Unreadable or non-object JSON:
  raise. Present but missing the keys the claim promises: **also raise**, louder
  than absent. A file that exists reads as done, so a half-written one is the
  failure mode that actually fools someone; a claim you believe you have is worse
  than one you know you are missing.

TODO 3 — `from_capstone`. The four capstone claims have no artifact. They are
  measured live, in the same pass as `report.py`, so the evidence log and
  PORTFOLIO.md cannot end up describing two different systems.

  The security claim is the one to be careful with. A lone bypass count reads as
  proof and is not: zero bypasses is equally what a run with no attacks reports, or
  one whose gated-tool set was empty, or a filter tuned until it refused the benign
  questions too. Print the attacks thrown, the PII leaks and the refused controls
  beside it when `measured.safety` carries them, and fall back to the bare count
  when it does not — the offline tier has three inline probes and no controls, and a
  fabricated containment object would be worse than a narrow honest one.

TODO 4 — `render`. Six sections. Print unproven rows WITH the command that closes
  them: an unproven claim without its command is a complaint, not a next step.
  Show each artifact's `measured_on` date, and flag anything older than
  `STALE_DAYS`. Old measurements of unchanged code are still true; old
  measurements you cannot date are how a number outlives the system it described.

TODO 5 — `manifest`. The machine-readable half, for the completion manifest.
  Shape it so `complete` can only ever be computed from the findings. If it were
  a field anyone could set, you would have rebuilt the checklist.

TODO 6 — the `decisions` dimension gets a section but no score. Counting ADRs
  would make the one qualitative dimension gradeable by writing more files, which
  is exactly the self-attestation the other five refuse.

Reference: ../../after/src/assistant/evidence.py.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistant.report import DECISIONS, Measured, measure  # noqa: F401

DIMENSIONS = (
    "quality",
    "latency",
    "cost",
    "security",
    "failure-recovery",
    "decisions",
)

PROVEN = "proven"
UNPROVEN = "unproven"
STALE_DAYS = 30


class EvidenceError(Exception):
    """A malformed artifact. Louder than a missing one."""


@dataclass(frozen=True)
class Claim:
    """TODO 1: one thing the course asks you to be able to show."""

    id: str
    dimension: str
    phase: str
    what: str
    lesson: str | None
    target: str
    artifact: str | None
    keys: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        raise NotImplementedError("TODO 1: derive the command from lesson + target")


#: TODO 1: at least one claim per dimension except `decisions`. A dimension with
#: no claims is a heading that quietly promises coverage the course does not have.
CLAIMS: tuple[Claim, ...] = ()


@dataclass
class Finding:
    claim: Claim
    status: str
    values: dict[str, Any] = field(default_factory=dict)
    measured_on: str | None = None
    note: str = ""

    @property
    def proven(self) -> bool:
        return self.status == PROVEN

    def age_days(self, today: dt.date) -> int | None:
        raise NotImplementedError("TODO 4: days between measured_on and today")


def read_artifact(path: Path, claim: Claim) -> Finding:
    """TODO 2: missing -> unproven; malformed OR incomplete -> raise."""
    raise NotImplementedError("TODO 2")


def from_capstone(claim: Claim, measured: Measured) -> Finding:
    """TODO 3: the capstone's own numbers, from the run that just happened."""
    raise NotImplementedError("TODO 3")


def collect(evidence_dir: Path, measured: Measured) -> list[Finding]:
    """TODO: every claim, in registry order, proven or honestly not."""
    raise NotImplementedError("TODO 2 + 3")


def coverage(findings: list[Finding]) -> dict[str, tuple[int, int]]:
    """TODO: proven / total per dimension. `decisions` must not appear."""
    raise NotImplementedError("TODO 6")


def render(findings: list[Finding], today: dt.date | None = None) -> str:
    """TODO 4: the page, unproven rows included, each with its command."""
    raise NotImplementedError("TODO 4")


def manifest(findings: list[Finding], today: dt.date | None = None) -> dict[str, Any]:
    """TODO 5: the completion manifest, with no field a learner can set."""
    raise NotImplementedError("TODO 5")


def build(evidence_dir: Path) -> tuple[str, dict[str, Any]]:
    """TODO: one pass — measure the capstone, read the artifacts, render both."""
    raise NotImplementedError("TODO 1-6")


def main() -> int:
    raise NotImplementedError("TODO: argparse over --evidence-dir/--out/--json")


if __name__ == "__main__":
    raise SystemExit(main())

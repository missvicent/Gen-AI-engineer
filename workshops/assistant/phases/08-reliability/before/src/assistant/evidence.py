"""`make evidence` — the course-wide engineering evidence log.

`report.py` measures the capstone. This measures **the course**: six dimensions
a reviewer actually asks about — quality, latency, cost, security, failure
recovery, and the decisions behind them — collected across all nine phases into
one page and one JSON manifest.

## Why absence is a status, not a gap in a table

The default for every claim is `unproven`, and an unproven claim is *printed*
rather than skipped. This is the whole design, and it is the difference between
a manifest and a checklist.

A checklist records what you say you did. Tick nine boxes and it reports nine
phases complete, whether you ran anything or not — which means it reports the
same thing for someone who did the work and someone who did not, and therefore
carries no information about either. This records what left a file behind. If
you never ran the phase-6 red-team, the security section says so, with the
command that would fix it. You cannot tick your way to a green page.

The consequence is worth stating plainly: **your first run of this will be
almost entirely red**, because you have not generated the artifacts yet. That is
the report working. A tool that flattered you on day one would have nothing left
to tell you on day ninety.

## Where the numbers come from

Capstone claims are measured live, in the same pass as `report.py`, so the
evidence log and the portfolio page cannot disagree about the system they both
describe. Every other claim reads a JSON file that some earlier phase wrote —
named, dated, and reproduced by exactly one command, which the page prints next
to the claim. Nothing here is typed by hand, because a hand-typed number in an
evidence log is the specific thing an evidence log exists to prevent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistant.report import DECISIONS, Measured, measure

#: Ordered by the question a reviewer asks first. Quality before latency because
#: a fast wrong answer is not a trade-off, it is a bug; security before recovery
#: because an attack you never survive has no recovery story.
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
#: An artifact older than this is reported with its age. Not a failure — an old
#: measurement of unchanged code is still true — but a number whose date you
#: cannot see is a number you will quote long after it stopped being about the
#: system you have.
STALE_DAYS = 30


class EvidenceError(Exception):
    """A malformed artifact. Louder than a missing one, because a file that
    exists and cannot be read is a claim you think you have."""


@dataclass(frozen=True)
class Claim:
    """One thing the course asks you to be able to show.

    `lesson` and `target` are stored apart from the prose rather than baked into
    a command string, so a test can walk the repo and check that every command
    this page prints names a directory that exists and a target that runs. An
    evidence log citing a lesson renamed six commits ago is the one kind of wrong
    this module has no excuse for.
    """

    id: str
    dimension: str
    phase: str
    what: str
    #: Lesson directory relative to `src/`. `None` for the capstone's own claims.
    lesson: str | None
    target: str
    #: Filename under the evidence directory holding this claim's numbers.
    #: `None` means the capstone measures it live in this same pass.
    artifact: str | None
    #: Keys to lift out of the artifact and show. Missing keys are reported as
    #: a malformed artifact rather than silently dropped, because a file that
    #: exists reads as done.
    keys: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        return f"make {self.target}" + (f" in {self.lesson}" if self.lesson else "")


CLAIMS: tuple[Claim, ...] = (
    # --- quality ---------------------------------------------------------------
    Claim(
        "p2-retrieval",
        "quality",
        "2",
        "Hybrid + rerank beats either half alone, on your own slices",
        "phase2-retrieval/02-hybrid-rerank/after",
        "check",
        "p2-retrieval.json",
        ("recall_at_5", "baseline_recall_at_5"),
    ),
    Claim(
        "p3-goldenset",
        "quality",
        "3",
        "Faithfulness and context recall on a 50-row golden set, scored by a judge "
        "you calibrated against your own labels",
        "phase3-evals/03-judge-calibration/after",
        "check",
        "p3-goldenset.json",
        ("faithfulness", "context_recall", "kappa"),
    ),
    Claim(
        "capstone-quality",
        "quality",
        "8",
        "The composed service scored against its golden set",
        None,
        "report",
        None,
    ),
    # --- latency ---------------------------------------------------------------
    Claim(
        "p8-ladder",
        "latency",
        "8",
        "P50/P95/P99 before and after each optimization rung — the tail, not the "
        "average, because the average is what hides the incident",
        "phase8-deploy/04-cost-latency/after",
        "check",
        "p8-latency.json",
        ("p50_ms", "p95_ms", "p99_ms"),
    ),
    Claim(
        "capstone-latency",
        "latency",
        "8",
        "P99 of the composed request path, read off the same spans /health counts",
        None,
        "report",
        None,
    ),
    # --- cost ------------------------------------------------------------------
    Claim(
        "p1-meter",
        "cost",
        "1",
        "Cost per call billed from each vendor's own usage field, not estimated "
        "from a token count you guessed",
        "phase1-foundations/02-token-cost-meter/after",
        "check",
        "p1-cost.json",
        ("cost_usd", "tokens_in", "tokens_out"),
    ),
    Claim(
        "p6-cost-model",
        "cost",
        "6",
        "Modelled cost at a stated load, with the assumption that dominates it named",
        "phase6-design-defend/02-cost-model/after",
        "check",
        "p6-cost-model.json",
        ("monthly_usd", "requests_per_day"),
    ),
    Claim(
        "capstone-cost",
        "cost",
        "8",
        "Measured tokens priced at the tier this deployment actually pays for",
        None,
        "report",
        None,
    ),
    # --- security --------------------------------------------------------------
    Claim(
        "p6-redteam",
        "security",
        "6",
        "The versioned red-team dataset run against your assistant: how many "
        "injections landed, and how many of those reached a gated tool",
        "phase6-design-defend/01-red-team/after",
        "check",
        "p6-redteam.json",
        ("cases", "detected", "bypasses"),
    ),
    Claim(
        "capstone-security",
        "security",
        "8",
        "Live containment probes against the running service: no landed injection "
        "fires a gated tool, no response leaks PII, and no benign question is refused",
        None,
        "report",
        None,
    ),
    # --- failure recovery ------------------------------------------------------
    Claim(
        "defect-lab",
        "failure-recovery",
        "8",
        "Your regression tests pass against the fix AND fail against each of the "
        "three vulnerabilities this codebase actually shipped",
        None,
        "defect-lab",
        "defect-lab.json",
        ("defects", "green_against_fix"),
    ),
    Claim(
        "p8-rollback",
        "failure-recovery",
        "8",
        "A failed smoke check rolls back to an immutable tag — and halts, rather "
        "than lying, when the only thing behind you is a moving tag",
        "phase8-deploy/03-deploy-observe/after",
        "check",
        "p8-rollback.json",
        ("smoke_probes", "rollback_halts_on_mutable"),
    ),
    Claim(
        "p8-backup",
        "failure-recovery",
        "8",
        "A backup taken while a writer is running restores with its row counts "
        "verified by the same script that took it",
        "phase8-deploy/03-deploy-observe/after",
        "check",
        "p8-backup.json",
        ("rows_verified",),
    ),
)


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
        if not self.measured_on:
            return None
        try:
            return (today - dt.date.fromisoformat(self.measured_on)).days
        except ValueError:
            return None


def read_artifact(path: Path, claim: Claim) -> Finding:
    """One artifact, or an honest account of why there isn't one."""
    if not path.exists():
        return Finding(claim, UNPROVEN, note="no artifact yet — run the command")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"{path} must hold a JSON object, not {type(data).__name__}")
    missing = [key for key in claim.keys if key not in data]
    if missing:
        # Louder than absent, because a file that exists reads as done. A claim
        # you believe you have is worse than one you know you are missing.
        raise EvidenceError(
            f"{path} is missing {', '.join(missing)} — the artifact exists but does "
            f"not carry the numbers {claim.id} claims"
        )
    return Finding(
        claim,
        PROVEN,
        values={key: data[key] for key in claim.keys},
        measured_on=str(data.get("measured_on") or data.get("date") or ""),
    )


def from_capstone(claim: Claim, measured: Measured) -> Finding:
    """The four capstone claims, lifted out of the run that just happened."""
    today = dt.date.today().isoformat()
    picks: dict[str, dict[str, Any]] = {
        "capstone-quality": {
            "faithfulness": measured.faithfulness,
            "context_recall": measured.recall,
        },
        "capstone-latency": {"p99_ms": measured.p99_ms, "runs": measured.runs},
        "capstone-cost": {
            "cost_usd": measured.cost_usd,
            "tokens_in": measured.tokens_in,
            "tokens_out": measured.tokens_out,
        },
        "capstone-security": security_values(measured),
    }
    return Finding(claim, PROVEN, values=picks[claim.id], measured_on=today)


def security_values(measured: Measured) -> dict[str, Any]:
    """What the security claim is allowed to print.

    A bypass count alone was the whole claim, and it was the shape of the claim that
    made it weak rather than the number. Zero bypasses is equally what you get from a
    run with no attacks in it, or one whose gated set was empty, or a filter tuned
    until it refused the benign questions too. So the page prints the parts a reader
    would otherwise have to assume: how many attacks were actually thrown, whether
    anything leaked, and whether the controls survived.

    Falls back to the bypass count when `safety` is absent, which is the offline lane:
    three inline probes, no controls, and a fabricated containment object would be
    worse than a narrow honest one. The publication gate requires the full object for
    release-class evidence instead, so the absence fails closed where it counts.
    """
    if not measured.safety:
        return {"redteam_bypasses": measured.redteam_bypasses}
    safety = measured.safety
    return {
        "redteam_bypasses": measured.redteam_bypasses,
        "attacks": safety.get("attacks"),
        "pii_leaks": safety.get("pii_leaks"),
        "controls_refused": safety.get("controls_refused"),
    }


def collect(evidence_dir: Path, measured: Measured) -> list[Finding]:
    """Every claim, in registry order, each proven or honestly not."""
    findings = []
    for claim in CLAIMS:
        if claim.artifact is None:
            findings.append(from_capstone(claim, measured))
        else:
            findings.append(read_artifact(evidence_dir / claim.artifact, claim))
    return findings


def coverage(findings: list[Finding]) -> dict[str, tuple[int, int]]:
    """Proven / total per dimension. `decisions` is not in `CLAIMS` — the ADRs
    are text, and counting them as evidence would be exactly the self-attestation
    this module refuses everywhere else."""
    out: dict[str, tuple[int, int]] = {}
    for dimension in DIMENSIONS:
        rows = [f for f in findings if f.claim.dimension == dimension]
        if rows:
            out[dimension] = (sum(1 for f in rows if f.proven), len(rows))
    return out


def _format_values(values: dict[str, Any]) -> str:
    if not values:
        return "—"
    return " · ".join(f"{key} {_scalar(v)}" for key, v in values.items())


def _scalar(value: Any) -> str:
    # `bool` before the numeric branches: it is a subclass of `int`, so the
    # obvious ordering renders a passing check as "1".
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def render(findings: list[Finding], today: dt.date | None = None) -> str:
    """The page. Unproven rows are printed, with the command that closes them."""
    today = today or dt.date.today()
    counts = coverage(findings)
    proven = sum(p for p, _ in counts.values())
    total = sum(t for _, t in counts.values())

    summary = " · ".join(f"{dim} {p}/{t}" for dim, (p, t) in counts.items())
    parts = [
        "# Engineering evidence log — the whole course\n\n"
        f"Generated by `make evidence` on {today.isoformat()} · "
        f"**{proven} of {total} claims proven** · {summary}\n\n"
        "Every row is either backed by a file some phase actually wrote, or "
        "marked unproven with the command that would close it. There is no third "
        "state and no way to tick a box: this reports what you ran, not what you "
        "meant to. Expect a mostly-red page early — that is the report doing its "
        "job rather than flattering you.\n\n"
        "To close a row: run its command, then save the numbers it printed to the "
        "named file under `evidence/`, carrying the listed keys and a "
        "`measured_on` date. A file missing any key is an error, not a partial pass — "
        "half-written evidence reads as done, which makes it worse than none.\n"
    ]

    for dimension in DIMENSIONS:
        rows = [f for f in findings if f.claim.dimension == dimension]
        if not rows:
            continue
        lines = ["| phase | claim | status | measured |", "|---|---|---|---|"]
        for finding in rows:
            age = finding.age_days(today)
            when = ""
            if finding.measured_on:
                when = f" ({finding.measured_on}"
                when += f", {age}d old)" if age is not None and age > STALE_DAYS else ")"
            status = "proven" if finding.proven else "**unproven**"
            lines.append(
                f"| {finding.claim.phase} | {finding.claim.what} | {status}{when} | "
                f"{_format_values(finding.values)} |"
            )
        todo = [f for f in rows if not f.proven]
        follow = ""
        if todo:
            follow = (
                "\nTo close the unproven rows:\n\n"
                + "\n".join(
                    f"- `{f.claim.command}` → `{f.claim.artifact}` ({', '.join(f.claim.keys)})"
                    for f in todo
                )
                + "\n"
            )
        heading = dimension.replace("-", " ").title()
        parts.append(f"## {heading}\n\n" + "\n".join(lines) + "\n" + follow)

    parts.append(_decisions_section())
    return "\n".join(parts)


def _decisions_section() -> str:
    lines = "\n".join(f"- **{adr_id}** — {title}" for adr_id, title in DECISIONS)
    return (
        "## Decisions\n\n"
        f"{lines}\n\n"
        "The one dimension with no proven/unproven column, deliberately. An ADR "
        "is an argument, and an argument is judged by reading it — counting them "
        "would turn the only qualitative section of this page into exactly the "
        "self-attestation the other five refuse. Full context in "
        "`workshops/assistant/adr/`.\n"
    )


def manifest(findings: list[Finding], today: dt.date | None = None) -> dict[str, Any]:
    """The machine-readable half — what a completion manifest consumes.

    Deliberately shaped so `complete` can never be computed from anything but
    the findings: there is no field a learner can set, and no partial credit for
    an artifact that does not exist."""
    today = today or dt.date.today()
    counts = coverage(findings)
    return {
        "generated_on": today.isoformat(),
        "dimensions": {
            dim: {"proven": p, "total": t, "complete": p == t} for dim, (p, t) in counts.items()
        },
        "claims": {
            f.claim.id: {
                "dimension": f.claim.dimension,
                "phase": f.claim.phase,
                "status": f.status,
                "values": f.values,
                "measured_on": f.measured_on or None,
                "command": f.claim.command,
            }
            for f in findings
        },
        "proven": sum(1 for f in findings if f.proven),
        "total": len(findings),
        "complete": all(f.proven for f in findings),
    }


def build(evidence_dir: Path) -> tuple[str, dict[str, Any]]:
    """One pass: measure the capstone, read the phase artifacts, render both."""
    _, measured = measure()
    findings = collect(evidence_dir, measured)
    return render(findings), manifest(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description="write the course-wide evidence log")
    parser.add_argument("--evidence-dir", default="evidence", help="where the phase artifacts live")
    parser.add_argument("--out", default="EVIDENCE.md", help="the page")
    parser.add_argument("--json", default="evidence/manifest.json", help="the completion manifest")
    parser.add_argument(
        "--require-complete", action="store_true", help="exit non-zero unless every claim is proven"
    )
    args = parser.parse_args()

    try:
        page, data = build(Path(args.evidence_dir))
    except EvidenceError as exc:
        print(f"evidence: {exc}")
        return 1

    Path(args.out).write_text(page)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} and {json_path} · {data['proven']}/{data['total']} claims proven")
    if args.require_complete and not data["complete"]:
        print("evidence: not every claim is proven")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The evidence log's only real job is refusing to say something is proven.

So that is what these test. Not the markdown — the four ways a claim can look
finished without being finished: a missing file, a file with the wrong numbers
in it, a stale date, and the tempting shortcut of counting the ADRs.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from assistant.evidence import (
    CLAIMS,
    DIMENSIONS,
    PROVEN,
    UNPROVEN,
    Claim,
    EvidenceError,
    Measured,
    collect,
    coverage,
    manifest,
    read_artifact,
    render,
)

TODAY = dt.date(2026, 8, 1)

MEASURED = Measured(
    faithfulness=0.91,
    recall=0.88,
    redteam_bypasses=0,
    p99_ms=1840.0,
    cost_usd=0.0143,
    tokens_in=9100,
    tokens_out=1200,
    runs=50,
)

CLAIM = Claim(
    "demo", "quality", "3", "something is true", None, "check", "demo.json", ("score",)
)


def test_a_missing_artifact_is_unproven_not_absent(tmp_path):
    """The row still prints. A claim you never ran has to be visible, or the
    page reports the same thing for work done and work skipped."""
    finding = read_artifact(tmp_path / "demo.json", CLAIM)
    assert finding.status == UNPROVEN
    assert "run the command" in finding.note


def test_an_artifact_missing_the_numbers_raises_rather_than_downgrades(tmp_path):
    """Louder than absent, deliberately: a file that exists reads as done, so a
    half-written one is the failure mode that actually fools someone."""
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(EvidenceError, match="does not carry the numbers"):
        read_artifact(path, CLAIM)


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text("{not json")
    with pytest.raises(EvidenceError, match="not valid JSON"):
        read_artifact(path, CLAIM)


def test_a_json_list_is_rejected(tmp_path):
    """`json.loads` is happy with a list; `data.get` on one is an AttributeError
    three frames later. Catch the shape here, where the message names the file."""
    path = tmp_path / "demo.json"
    path.write_text("[1, 2]")
    with pytest.raises(EvidenceError, match="JSON object"):
        read_artifact(path, CLAIM)


def test_a_complete_artifact_carries_its_numbers_and_its_date(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"score": 0.9, "measured_on": "2026-07-30"}))
    finding = read_artifact(path, CLAIM)
    assert finding.status == PROVEN
    assert finding.values == {"score": 0.9}
    assert finding.age_days(TODAY) == 2


def test_capstone_claims_are_proven_by_the_run_that_just_happened(tmp_path):
    """No artifact, no unproven row: these four are measured in the same pass, so
    the evidence log and PORTFOLIO.md cannot disagree about one system."""
    findings = collect(tmp_path, MEASURED)
    live = {f.claim.id: f for f in findings if f.claim.artifact is None}
    assert live and all(f.proven for f in live.values())
    assert live["capstone-latency"].values["p99_ms"] == 1840.0
    assert live["capstone-security"].values["redteam_bypasses"] == 0


def test_the_security_claim_prints_more_than_a_bypass_count(tmp_path):
    """Because a bypass count of zero is also what a suite that measured nothing
    reports. The reader cannot tell "47 attacks, none contained a hole" from "no
    attacks were thrown" unless the page says which — and the same number with a
    leak beside it is not the same claim."""
    from dataclasses import replace

    measured = replace(
        MEASURED,
        safety={"attacks": 47, "controls": 11, "pii_leaks": 0, "controls_refused": 0},
    )
    values = {
        f.claim.id: f.values for f in collect(tmp_path, measured)
    }["capstone-security"]
    assert values == {
        "redteam_bypasses": 0,
        "attacks": 47,
        "pii_leaks": 0,
        "controls_refused": 0,
    }


def test_the_offline_tier_prints_the_narrow_claim_rather_than_a_fabricated_one(tmp_path):
    """`safety` is None off the offline lane: three inline probes, no controls. The
    row stays true by staying small — inventing a containment object for a proxy run
    would put a number on the page that no measurement supports."""
    values = {
        f.claim.id: f.values for f in collect(tmp_path, MEASURED)
    }["capstone-security"]
    assert values == {"redteam_bypasses": 0}


def test_an_empty_evidence_dir_proves_only_the_capstone(tmp_path):
    """The honest first run. Anything green here that you did not run would mean
    the page grades intent."""
    findings = collect(tmp_path, MEASURED)
    proven = {f.claim.id for f in findings if f.proven}
    assert proven == {f.id for f in CLAIMS if f.artifact is None}


def test_complete_requires_every_claim(tmp_path):
    findings = collect(tmp_path, MEASURED)
    data = manifest(findings, TODAY)
    assert data["complete"] is False
    assert data["dimensions"]["quality"]["complete"] is False
    # The live-only dimension is the one that can go green on its own.
    assert data["dimensions"]["security"]["proven"] == 1


def test_the_manifest_exposes_no_field_a_learner_can_set(tmp_path):
    """`complete` is derived, per claim and per dimension. If it were storable,
    the manifest would be a checklist with extra steps."""
    findings = collect(tmp_path, MEASURED)
    data = manifest(findings, TODAY)
    computed = all(c["status"] == PROVEN for c in data["claims"].values())
    assert data["complete"] is computed
    for dim, row in data["dimensions"].items():
        rows = [c for c in data["claims"].values() if c["dimension"] == dim]
        assert row["proven"] == sum(1 for c in rows if c["status"] == PROVEN)


def test_every_unproven_row_prints_the_command_that_closes_it(tmp_path):
    """An unproven claim without its command is a complaint, not a next step."""
    page = render(collect(tmp_path, MEASURED), TODAY)
    for claim in CLAIMS:
        if claim.artifact is not None:
            assert claim.command in page, f"{claim.id} left the reader with no way forward"


def test_the_page_names_the_shortfall_in_its_first_lines(tmp_path):
    page = render(collect(tmp_path, MEASURED), TODAY)
    header = page.split("## ")[0]
    assert "4 of 13 claims proven" in header
    assert "unproven" in page


def test_a_true_flag_renders_as_yes_not_as_one(tmp_path):
    """`bool` is a subclass of `int`, so the obvious formatter ordering prints a
    passing check as "1" — a number, in a table of numbers, meaning nothing."""
    (tmp_path / "defect-lab.json").write_text(
        json.dumps(
            {"defects": ["a", "b"], "green_against_fix": True, "measured_on": "2026-08-01"}
        )
    )
    page = render(collect(tmp_path, MEASURED), TODAY)
    assert "green_against_fix yes" in page
    assert "defects a, b" in page  # a list, not its repr


def test_a_stale_artifact_is_proven_but_shows_its_age(tmp_path):
    """Old measurements of unchanged code are still true. Old measurements you
    cannot date are how a number outlives the system it described."""
    (tmp_path / "p3-goldenset.json").write_text(
        json.dumps(
            {
                "faithfulness": 0.8,
                "context_recall": 0.7,
                "kappa": 0.6,
                "measured_on": "2026-01-01",
            }
        )
    )
    page = render(collect(tmp_path, MEASURED), TODAY)
    assert "212d old" in page


def test_decisions_are_listed_but_never_counted(tmp_path):
    """The one section with no score. Counting ADRs would make the qualitative
    dimension gradeable by writing more files, which is the failure this whole
    module is built against."""
    findings = collect(tmp_path, MEASURED)
    assert "decisions" not in coverage(findings)
    assert "decisions" not in manifest(findings, TODAY)["dimensions"]
    page = render(findings, TODAY)
    assert "## Decisions" in page
    assert "ADR-0001" in page


def test_every_dimension_but_decisions_has_at_least_one_claim():
    """A dimension with no claims is a heading that quietly promises coverage
    the course does not have."""
    covered = {claim.dimension for claim in CLAIMS}
    assert covered == set(DIMENSIONS) - {"decisions"}


def test_claim_ids_are_unique_and_dimensions_are_known():
    ids = [claim.id for claim in CLAIMS]
    assert len(ids) == len(set(ids))
    assert all(claim.dimension in DIMENSIONS for claim in CLAIMS)
    assert all(claim.command for claim in CLAIMS)


COURSE = Path(__file__).resolve().parents[4]


def _course_or_skip() -> Path:
    """Skipped when the capstone is unpacked on its own, which is a supported
    way to use it — the check needs the sibling lessons to check against."""
    if not (COURSE / "phase1-foundations").is_dir():
        pytest.skip("running outside the course tree; no sibling lessons to check")
    return COURSE


def test_every_command_names_a_lesson_that_exists():
    """The one kind of wrong an evidence log has no excuse for: telling you to
    run something in a directory that was renamed six commits ago."""
    course = _course_or_skip()
    for claim in CLAIMS:
        if claim.lesson:
            assert (course / claim.lesson).is_dir(), f"{claim.id} cites a missing lesson"


def test_every_command_names_a_target_that_runs():
    """A path that exists is not enough — `make check` has to be a real target in
    that lesson, whether from its own Makefile or the shared `_lesson.mk`."""
    course = _course_or_skip()
    shared = (course / "_lesson.mk").read_text()
    for claim in CLAIMS:
        home = (course / claim.lesson) if claim.lesson else Path(__file__).parents[1]
        makefile = (home / "Makefile").read_text()
        pattern = rf"^{re.escape(claim.target)}:"
        found = re.search(pattern, makefile, re.M) or re.search(pattern, shared, re.M)
        assert found, f"{claim.id} calls `{claim.command}`, which is not a target"

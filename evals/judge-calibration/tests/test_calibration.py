"""Calibration tests — offline, deterministic, and free. Fixtures, not models."""

from dataclasses import replace
from math import sqrt
from pathlib import Path

import pytest
from judge_calibration.calibration import (
    GATING_KAPPA,
    LabeledRow,
    best_threshold,
    calibrate,
    disagreement_rows,
    interpret,
    load_labeled,
    report,
    verdicts,
)

LABELED = Path(__file__).parents[1] / "evals" / "labeled.jsonl"


def rows():
    return load_labeled(LABELED)


def _synthetic(human_labels: list[str], scores: list[float]) -> list[LabeledRow]:
    return [
        LabeledRow(id=f"s-{i}", question="q", answer="a", judge_score=s, human=h)
        for i, (h, s) in enumerate(zip(human_labels, scores, strict=True))
    ]


def test_the_labeled_set_is_big_enough_and_carries_provenance():
    assert len(rows()) >= 30
    assert all(r.labeled_by and r.labeled_on for r in rows())


def test_a_rubber_stamp_judge_has_high_agreement_and_no_kappa():
    """The whole reason kappa is the number you report."""
    labels = ["pass"] * 36 + ["fail"] * 4
    scores = [0.9] * 40  # says "pass" to everything
    c = calibrate(_synthetic(labels, scores))
    assert c.agreement == 0.9
    assert c.kappa == 0.0
    assert not c.gatable


def test_a_perfect_judge_scores_one():
    labels = ["pass", "fail", "pass", "fail"]
    c = calibrate(_synthetic(labels, [0.9, 0.1, 0.8, 0.2]))
    assert c.agreement == 1.0
    assert c.kappa == 1.0
    assert c.gatable


def test_an_inverted_judge_is_worse_than_chance():
    labels = ["pass", "fail", "pass", "fail"]
    c = calibrate(_synthetic(labels, [0.1, 0.9, 0.2, 0.8]))
    assert c.kappa < 0


def test_sweeping_the_threshold_beats_the_default_of_one_half():
    default = calibrate(rows())
    best = best_threshold(rows())
    assert best.kappa > default.kappa
    assert best.threshold != 0.5
    # Agreement barely moves while kappa moves a lot — read kappa, not agreement.
    assert best.agreement - default.agreement < 0.10
    assert best.kappa - default.kappa > 0.15


def test_the_shipped_judge_only_becomes_gatable_after_calibration():
    assert not calibrate(rows()).gatable
    assert best_threshold(rows()).gatable
    assert best_threshold(rows()).kappa >= GATING_KAPPA


def test_tolerance_comes_from_the_disagreement_rate():
    """Per-row disagreement, averaged over n rows: the noise floor of the suite."""
    best = best_threshold(rows())
    expected = round((1 - best.agreement) / sqrt(best.n), 2)
    assert best.tolerance == pytest.approx(expected)
    assert 0 < best.tolerance < 1 - best.agreement


def test_disagreements_are_returned_as_readable_rows():
    best = best_threshold(rows())
    bad = disagreement_rows(rows(), best)
    assert len(bad) == len(best.disagreements) == round((1 - best.agreement) * best.n)
    assert all(r.question and r.answer for r in bad)


def test_verdicts_use_the_labels_you_labeled_in():
    assert set(verdicts(rows(), 0.5)) <= {"pass", "fail"}


def test_interpretation_bands_are_ordered():
    assert "rubric" in interpret(0.05)
    assert "smoke" in interpret(0.35)
    assert "regressions" in interpret(0.55)
    assert "defend" in interpret(0.85)


def test_report_names_the_gate_decision_and_the_tolerance():
    text = report(rows())
    assert "gate on this judge?" in text
    assert "regression tolerance:" in text
    assert "best kappa" in text


def test_unknown_human_labels_are_rejected_at_load():
    bad = replace(rows()[0], human="maybe")
    with pytest.raises(ValueError, match="unknown human labels"):
        _reload([bad])


def _reload(labeled: list[LabeledRow]):
    """Round-trip through the loader so its validation is what we test."""
    import json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labeled.jsonl"
        path.write_text("\n".join(json.dumps(asdict(r)) for r in labeled))
        return load_labeled(path)

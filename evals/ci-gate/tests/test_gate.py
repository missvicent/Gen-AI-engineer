"""Gate tests — the gate logic is code, so it gets tests like code.

Offline and deterministic: fixtures in, exit code out. This is the tier that runs on
every push, which is the only reason the gate is ever actually enforced.
"""

import json
from dataclasses import replace
from pathlib import Path

from ci_gate.gate import (
    BARS,
    TOLERANCE,
    Run,
    check_bars,
    check_regressions,
    diff_table,
    gate,
    main,
)

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results.json"
BASELINE = FIXTURES / "baseline.json"


def run() -> Run:
    return Run.load(RESULTS)


def baseline() -> Run:
    return Run.load(BASELINE)


def test_the_shipped_run_passes_the_gate():
    """It dipped 0.015 on faithfulness — a real dip, inside the noise floor."""
    assert gate(run(), baseline()) == []
    assert run().overall["faithfulness"] < baseline().overall["faithfulness"]


def test_a_dip_larger_than_the_tolerance_fails():
    dipped = replace(
        run(), overall={**run().overall, "faithfulness": 0.86}
    )  # 0.912 -> 0.860
    problems = gate(dipped, baseline())
    assert any("faithfulness regressed" in p for p in problems)


def test_an_absolute_bar_breach_fails_even_with_no_baseline_movement():
    """A delta check alone would let a permanently bad system stay bad."""
    flat = replace(
        run(),
        overall={"faithfulness": 0.60, "context_recall": 0.60},
    )
    flat_baseline = replace(
        baseline(), overall={"faithfulness": 0.60, "context_recall": 0.60}
    )
    assert check_bars(flat) == [
        f"faithfulness 0.600 is below the bar of {BARS['faithfulness']:.2f}",
        f"context_recall 0.600 is below the bar of {BARS['context_recall']:.2f}",
    ]
    assert gate(flat, flat_baseline)


def test_a_collapsed_slice_fails_while_the_average_still_looks_fine():
    """The failure mode this whole phase exists to catch."""
    broken = replace(
        run(),
        overall={"faithfulness": 0.86, "context_recall": 0.86},
        by_slice={
            **run().by_slice,
            "unanswerable": {"faithfulness": 0.4, "context_recall": 0.4},
        },
    )
    problems = gate(broken, baseline())
    assert any("unanswerable" in p and "COLLAPSED" in p for p in problems)
    assert broken.overall["faithfulness"] >= BARS["faithfulness"]  # the bar is happy


def test_a_disappearing_slice_is_a_failure_not_a_pass():
    without = {k: v for k, v in run().by_slice.items() if k != "multi_hop"}
    problems = gate(replace(run(), by_slice=without), baseline())
    assert any("multi_hop' disappeared" in p for p in problems)


def test_a_changed_judge_is_a_re_baseline_not_a_comparison():
    other_judge = replace(
        run(), instrument={**run().instrument, "judge_model": "some-other-model:70b"}
    )
    problems = gate(other_judge, baseline())
    assert any("instrument changed" in p and "re-baseline" in p for p in problems)


def test_a_changed_library_version_is_caught_too():
    upgraded = replace(run(), instrument={**run().instrument, "ragas_version": "0.5.0"})
    assert any("ragas_version" in p for p in gate(upgraded, baseline()))


def test_improvements_never_fail_the_gate():
    better = replace(
        run(),
        overall={metric: 1.0 for metric in run().overall},
        by_slice={
            name: dict.fromkeys(scores, 1.0) for name, scores in run().by_slice.items()
        },
    )
    assert gate(better, baseline()) == []


def test_the_diff_table_is_readable_and_marks_the_regression():
    dipped = replace(run(), overall={**run().overall, "faithfulness": 0.70})
    table = diff_table(dipped, baseline())
    assert "OVERALL faithfulness" in table
    assert "REGRESSED" in table
    assert "ok (within tolerance)" in diff_table(run(), baseline())


def test_tolerance_is_the_calibrated_noise_floor_not_a_guess():
    """Lesson 3.3 derives 0.03 from 40 hand-labeled rows. Keep them in step."""
    assert TOLERANCE == 0.03


def test_main_exits_zero_on_a_passing_run(capsys):
    assert main([str(RESULTS), str(BASELINE)]) == 0
    assert "gate passed" in capsys.readouterr().out


def test_main_exits_one_and_prints_reasons_on_a_failing_run(tmp_path: Path, capsys):
    raw = json.loads(Path(RESULTS).read_text())
    raw["by_slice"]["unanswerable"] = {"faithfulness": 0.2, "context_recall": 0.2}
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps(raw))

    assert main([str(bad), str(BASELINE)]) == 1
    out = capsys.readouterr().out
    assert "gate FAILED:" in out
    assert "unanswerable" in out


def test_an_empty_report_does_not_pass():
    """Every check used to skip a key that was absent, so nothing was left to fail."""
    empty = Run(overall={}, by_slice={}, instrument={})
    assert gate(empty, empty) != []
    assert any("zero rows" in problem for problem in gate(empty, empty))


def test_a_missing_overall_metric_fails():
    partial = replace(run(), overall={"faithfulness": run().overall["faithfulness"]})
    problems = gate(partial, baseline())
    assert any(
        "context_recall" in problem and "missing" in problem for problem in problems
    )


def test_main_rejects_wrong_usage():
    assert main([]) == 2


def test_only_a_metric_the_baseline_has_can_go_missing():
    """A new metric has no baseline to regress from; a removed one must still fail."""
    slices = baseline().by_slice
    added = replace(
        baseline(),
        overall={**baseline().overall, "new_metric": 0.9},
        by_slice={**slices, "exact": {**slices["exact"], "new_metric": 0.9}},
    )
    assert check_regressions(added, baseline()) == []

    removed = replace(
        baseline(),
        overall={"faithfulness": baseline().overall["faithfulness"]},
        by_slice={**slices, "exact": {"faithfulness": slices["exact"]["faithfulness"]}},
    )
    problems = check_regressions(removed, baseline())
    assert len(problems) == 2
    assert all("context_recall" in p and "missing" in p for p in problems)


def test_a_changed_judge_skips_the_comparison():
    """Scores from a different judge can't be compared, so a drop isn't a reason."""
    other_judge = replace(
        run(),
        instrument={**run().instrument, "judge_model": "some-other-model:70b"},
        overall={**run().overall, "faithfulness": 0.86},  # would regress otherwise
    )
    problems = gate(other_judge, baseline())
    assert len(problems) == 1
    assert "instrument changed" in problems[0]

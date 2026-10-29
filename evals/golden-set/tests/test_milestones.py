"""The milestones are motivation. The interval is the content, so it gets the tests.

Each one pins a claim the module makes in prose, because a docstring that says
"the normal approximation is broken at n=10" and code that quietly uses it is the
most expensive kind of teaching material.
"""

from __future__ import annotations

import pytest
from golden_set.milestones import (
    MILESTONES,
    Interval,
    MilestoneError,
    progress,
    reached,
    render,
    verdict,
    wilson,
)


def test_ten_out_of_ten_does_not_claim_certainty():
    """The headline reason Wilson is here. The textbook interval returns ±0.00 at
    a perfect score — reporting, from ten questions, that the system never fails."""
    low, high = wilson(10, 10)
    assert high == pytest.approx(1.0)
    assert low < 0.75, f"a perfect 10/10 should still admit doubt, got {low:.2f}"


def test_zero_out_of_ten_stays_inside_the_unit_interval():
    low, high = wilson(0, 10)
    assert low == 0.0
    assert 0 < high < 0.35


def test_eight_out_of_ten_is_far_too_wide_to_quote():
    """The number this whole module exists to argue with. 0.80 sounds like a
    measurement; the interval says a mediocre system produces it routinely."""
    low, high = wilson(8, 10)
    assert low < 0.55 and high > 0.90


def test_the_interval_narrows_as_the_milestones_pass():
    """Same score, three sample sizes: the point the learner should feel by row 50."""
    widths = [Interval(round(0.8 * n), n).width for n in MILESTONES]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > 2 * widths[-1]


def test_a_five_point_move_at_ten_rows_is_not_a_move():
    """The failure this prevents: shipping a change because 0.80 became 0.85."""
    before, after = Interval(8, 10), Interval(9, 10)
    assert before.indistinguishable_from(after)


def test_the_same_five_point_move_at_fifty_rows_is_still_not_a_move():
    """Honest about its own limits. Fifty rows makes gating defensible; it does not
    make every five-point difference real, and a module that implied otherwise
    would be selling the same overclaim one milestone later."""
    assert Interval(40, 50).indistinguishable_from(Interval(43, 50))


def test_a_large_move_at_fifty_rows_does_separate():
    assert not Interval(40, 50).indistinguishable_from(Interval(25, 50))


def test_verdict_reads_the_sample_size_and_never_the_score():
    """A rule that could read the score is a loophole: a good number would buy
    conclusions the sample size does not support."""
    assert "quote" in verdict(10)
    assert "slice" in verdict(25)
    assert "gateable" in verdict(50)
    assert verdict(10) == verdict(24)
    assert verdict(50) == verdict(500)


def test_below_the_first_milestone_is_not_a_measurement():
    assert "not yet" in verdict(9)
    assert reached(9) is None


def test_reached_reports_the_highest_milestone_passed():
    assert reached(10) == 10
    assert reached(24) == 10
    assert reached(25) == 25
    assert reached(60) == 50


def test_progress_names_the_distance_to_the_next_sitting():
    assert "6 to the first milestone" in progress(4)
    assert "15 to 25" in progress(10)
    assert "all milestones reached" in progress(50)


def test_render_carries_score_interval_and_licence_together():
    """Inseparable on purpose. An interval with no verdict leaves the reader to
    decide what it licenses, and readers are generous about their own work."""
    line = render(Interval(8, 10))
    assert "0.80" in line and "95% CI" in line and "n=10" in line and "quote" in line


def test_impossible_counts_raise_instead_of_returning_a_number():
    with pytest.raises(MilestoneError):
        wilson(0, 0)
    with pytest.raises(MilestoneError):
        wilson(11, 10)

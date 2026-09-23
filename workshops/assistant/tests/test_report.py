"""The portfolio page is generated evidence, so the generator itself is on trial.

If `make report` can emit a page that hides a containment breach or invents a
number, the portfolio is worse than none. These tests run the real builder against
the real offline assistant and check the page carries measured content — and that
a breach cannot pass silently.

The JSON half is held to a harder standard than the page, because CI acts on it
without a human in the loop: the stamps must be derived rather than typed, the
numbers must move when the system moves, and a breach must show up as a bypass
count the merge gate can block on.
"""
from __future__ import annotations

import pytest

from assistant.observe import PIPELINE_SPAN
from assistant.report import (
    EXCHANGES,
    REDTEAM_PROBES,
    build_portfolio,
    measure,
    metered_ask,
    redteam_section,
    require_every_run_metered,
    run_evals,
    run_probes,
    versions_for,
)
from assistant.service import build_assistant
from assistant.settings import Settings


@pytest.fixture(scope="module")
def measured():
    return measure()


@pytest.fixture(scope="module")
def page(measured) -> str:
    return measured[0]


def test_build_portfolio_is_still_the_page(page):
    """The old entry point kept working, because the workshop brief names it."""
    assert build_portfolio().startswith("# Portfolio")


def test_the_page_carries_every_section(page):
    for heading in (
        "## Eval scores (offline judge)",
        "## Red-team containment",
        "## Latency (from the spans)",
        "## Cost",
        "## Design decisions",
    ):
        assert heading in page


def test_the_eval_table_is_scored_per_slice_and_names_its_judge(page):
    assert "abstention" in page  # the slice the business cares about most
    assert "OVERALL" in page  # format_table's footer — the table is real output
    assert "KeywordJudge" in page  # an unlabelled offline score is a lie


def test_every_probe_was_contained_and_the_verdict_says_so(page):
    assert page.count("| contained |") == len(REDTEAM_PROBES)
    assert "BREACHED" not in page
    assert "all probes contained" in page


def test_latency_percentiles_come_from_actual_runs(page):
    # The golden set + probes ran through the service, so agent.run spans exist
    # and the percentile line carries measured (non-negative) milliseconds.
    assert "P99" in page
    assert "agent.run" in page


def test_the_cost_story_is_honest_about_the_offline_tier(page, measured):
    """Zero, arrived at by counting rather than by assertion — the tokens are on
    the page next to the price, so the reader can see WHY it is free."""
    assert measured[1].cost_usd == 0.0
    assert measured[1].tokens_in > 0 and measured[1].tokens_out > 0
    assert "$0.0000" in page
    assert "no per-token invoice" in page


# --- the cost line covers the work the page reports -------------------------------


def test_every_request_the_page_counts_is_a_request_the_meter_saw(measured):
    """The round-6 defect, stated as the arithmetic that gave it away.

    The release page reported `runs: 63` — every `assistant.pipeline` span — beside a
    token total from the 5 golden rows, because the 58 red-team asks bypassed the
    meter. Both numbers were true about something. Divided by each other, which is
    what a reader does with them, they understated cost per request twelvefold.

    Held here as well as on the release lane because the offline page had the same
    shape at a smaller scale: three probes asking off-meter, and `runs` taken from
    `agent.run`, which a guardrail-blocked request never reaches.
    """
    report = measured[1]
    assert report.runs == len(REDTEAM_PROBES) + 5, "5 golden rows and every probe"
    assert report.tokens_in > 0 and report.tokens_out > 0


def test_a_request_that_skips_the_meter_stops_the_report():
    """The guard, not the intention.

    Re-inlining the accounting into `run_evals` does not fail anything on its own —
    it produces a smaller, entirely plausible cost line. So the two counts are
    compared and the run refuses, which is the only version of this that survives
    somebody adding a caller.
    """
    assistant = build_assistant(Settings())
    meter: dict = {}
    run_evals(assistant, meter)
    asked = len(assistant.rec.named(PIPELINE_SPAN))
    require_every_run_metered(meter, asked, PIPELINE_SPAN)  # in step: fine

    assistant.ask("what is the refund policy")  # the off-meter ask
    with pytest.raises(SystemExit) as refused:
        require_every_run_metered(meter, len(assistant.rec.named(PIPELINE_SPAN)), PIPELINE_SPAN)
    assert "metered_ask" in str(refused.value), "a refusal has to name the way in"
    assert f"{asked} exchange(s)" in str(refused.value)


def test_the_probes_are_billed_like_any_other_request():
    """An attack costs tokens. These three used to be free on the page."""
    assistant = build_assistant(Settings())
    meter: dict = {}
    run_probes(assistant, meter)
    assert meter[EXCHANGES] == len(REDTEAM_PROBES)
    assert meter["in"] > 0


def test_the_meter_takes_the_provider_s_own_counts_when_it_has_them():
    """`metered_ask` prefers `usage.take_last` over re-measuring, and the difference
    is a `counted` claim rather than an `estimated` one. Re-deriving the prompt here
    would quietly overwrite a real number with a reconstruction of it."""
    assistant = build_assistant(Settings())
    meter: dict = {}
    response = metered_ask(assistant, meter, "what is the refund policy")
    assert response["answer"]
    assert meter[EXCHANGES] == 1
    assert meter["in"] > 0 and meter["out"] > 0
    assert meter.get("counted", 0) + meter.get("estimated", 0) == 1, (
        "every exchange has to declare which kind of number it contributed"
    )


def test_the_json_report_carries_the_four_gated_numbers(measured):
    """These are exactly what phase8-deploy/02-ci budgets. A report that omits
    one is a gate that silently stops checking it."""
    report = measured[1]
    assert 0.0 <= report.faithfulness <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert report.redteam_bypasses == 0
    assert report.p99_ms >= 0.0
    assert report.cost_usd >= 0.0
    assert report.runs > 0


def test_every_stamp_is_present_and_derived_not_typed(measured):
    """The gate refuses an unstamped report, so an empty stamp would block every
    merge — and a HARD-CODED stamp is worse, because it never blocks anything and
    never tells the truth either."""
    versions = measured[1].versions
    assert set(versions) == {"model", "prompt", "corpus", "dataset"}
    assert all(versions.values())
    assert versions["model"] == "offline-stitcher"  # the tier that actually ran
    assert versions["prompt"].startswith("grounded-")


def test_editing_the_prompt_changes_the_prompt_stamp(monkeypatch):
    """The failure this catches: someone rewrites the prompt, leaves the version
    label alone, and every number recorded before and after gets compared as if
    it came from the same system."""
    assistant = build_assistant(Settings())
    before = versions_for(assistant)["prompt"]

    from assistant import composers

    def rewritten(goal, contexts, state, memories=None):
        return "a different prompt entirely"

    monkeypatch.setattr(composers, "grounded_prompt", rewritten)
    assert versions_for(assistant)["prompt"] != before


def test_the_corpus_and_dataset_stamps_follow_their_inputs(monkeypatch):
    assistant = build_assistant(Settings())
    before = versions_for(assistant)

    from assistant import report

    monkeypatch.setattr(report, "CORPUS", [*report.CORPUS, "a new document"])
    after = versions_for(assistant)
    assert after["corpus"] != before["corpus"]
    assert after["dataset"] == before["dataset"]  # only what changed moves


def test_a_breach_shows_up_as_a_bypass_the_gate_can_block_on():
    """The page says BREACHED; the JSON has to say a NUMBER, because CI does not
    read prose. This is the seam where a hidden breach would become a green
    merge."""
    sabotaged = build_assistant(Settings())
    sabotaged.approvals.consume = lambda *a, **k: "sabotaged-grant"  # type: ignore[method-assign]
    _, report = measure(sabotaged)
    assert report.redteam_bypasses > 0


def test_a_breach_cannot_pass_silently():
    """The safety property of the generator itself: sabotage containment and the
    section must scream, because a portfolio that hides a breach is the exact
    failure the whole course argues against."""
    sabotaged = build_assistant(Settings())
    sabotaged.rag.add(["warmup"])
    # sabotage: hand back a grant for whatever is about to run, so nothing is ever
    # contained. A real store cannot do this — that is the point of the class.
    sabotaged.approvals.consume = lambda *a, **k: "sabotaged-grant"  # type: ignore[method-assign]
    body, contained = redteam_section(sabotaged)
    assert contained is False
    assert "**BREACHED**" in body


def test_the_offline_page_says_its_tokens_are_estimated(page, measured):
    """The offline composer reports nothing, so the numbers are a word count.

    Printed with a dollar sign beside them, an estimate and an invoice look
    identical — which is how "cost: $0.0000 for 300 tokens" gets quoted as a
    measurement. The page has to say which one it is, and the JSON has to carry
    it too, because the gate reads the JSON.
    """
    assert measured[1].tokens_source == "estimated"
    assert "**estimated** by word split" in page


def test_provider_counts_win_over_the_word_split():
    """What the provider will invoice beats what we guessed it would.

    Ollama returns `prompt_eval_count` and `eval_count` on every completion and
    the adapter forwards them into the meter. They are not close to a word split
    — a tokenizer splits punctuation and subwords — so preferring them is the
    difference between a cost gate and a plausible-looking number.
    """
    from assistant import usage

    usage.take_last()
    usage.report(tokens_in=1234, tokens_out=56)
    used = usage.measure("three words here", "one")
    assert (used.tokens_in, used.tokens_out) == (1234, 56)
    assert used.source == "counted"


def test_a_count_is_used_once_and_never_inherited():
    """The failure a stale count causes is silent: the next exchange bills the
    previous one's tokens and every number stays plausible."""
    from assistant import usage

    usage.take_last()
    usage.report(tokens_in=99, tokens_out=99)
    usage.measure("a", "b")
    second = usage.measure("some words in a prompt", "and an answer")
    assert second.source == "estimated"
    assert second.tokens_in != 99


def test_a_half_reported_response_is_not_half_counted():
    """One count without the other is not a measurement, so the adapter reports
    neither and the estimate stands in, labelled."""
    from assistant import usage
    from assistant.adapters import _report_usage

    usage.take_last()
    usage.take_reported()
    _report_usage({"response": "hi", "eval_count": 12})
    assert usage.take_reported() is None

    _report_usage({"response": "hi", "prompt_eval_count": 30, "eval_count": 12})
    reported = usage.take_reported()
    assert reported is not None
    assert (reported.tokens_in, reported.tokens_out) == (30, 12)


def test_the_source_travels_with_the_number_onto_the_span():
    """A cost attribute looks the same either way; the span says which it is."""
    from assistant import observe

    assistant = build_assistant(Settings())
    assistant.rag.add(["Refunds are processed within 5 business days."])
    assistant.ask("How long do refunds take?")
    compose = [s for s in assistant.rec.spans() if s.name == observe.COMPOSE_SPAN]
    assert compose, "no compose span was recorded"
    assert compose[-1].attributes[observe.TOKENS_SOURCE] == "estimated"

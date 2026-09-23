"""The refusal contract: a reply is an answer or a refusal, never both.

Measured on `qwen3.5:9b` over the deployed stack, the grounded prompt used to
hand the model the refusal sentence and say "reply exactly this". The model read
that as a required part of the output format and appended it to answers it had
just given, correctly and with a citation — 6 to 8 times in 10 on questions with
no ambiguity in them at all:

    Travel expenses are reimbursed within ten business days of filing [c1].
    I don't know — that isn't in what I can see.

One reply said so outright: "(Note: The second sentence is required by the
prompt's constraint format)". Four rewordings of the instruction were measured
and none fixed it; the strongest of them only moved the failure from
contradiction to outright refusal. So the fix is not a wording: the model refuses
with a sentinel word that no answer contains, and the code — not the model —
enforces that a reply is one thing or the other.
"""

from assistant.composers import (
    ABSTAIN,
    NO_ANSWER,
    grounded_prompt,
    stop_at_refusal,
    without_refusal,
)

ANSWER = "Travel expenses are reimbursed within ten business days of filing [c1]."


def chunked(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_the_prompt_never_hands_the_model_the_refusal_sentence_to_copy():
    """The root cause, locked. A prompt containing the sentence is a prompt a
    model can paste the sentence out of."""
    prompt = grounded_prompt("when do i get paid", ["staff are paid monthly"], [])
    assert ABSTAIN not in prompt
    assert NO_ANSWER in prompt


def test_an_answer_with_the_sentinel_stapled_to_it_is_just_the_answer():
    assert without_refusal(f"{ANSWER} {NO_ANSWER}") == ANSWER


def test_the_sentinel_takes_its_trailing_commentary_with_it():
    """The model does not always stop at the sentinel — it explains itself after
    it. Everything from the sentinel on is format artefact, not answer."""
    reply = f"{ANSWER} {NO_ANSWER} (Note: the prompt requires this format.)"
    assert without_refusal(reply) == ANSWER


def test_the_sentinel_alone_is_the_refusal():
    assert without_refusal(NO_ANSWER) == ABSTAIN
    assert without_refusal(f"  {NO_ANSWER}\n") == ABSTAIN


def test_an_ordinary_answer_passes_through_untouched():
    assert without_refusal(ANSWER) == ANSWER


def test_an_empty_completion_is_a_refusal_not_an_empty_answer():
    assert without_refusal("   ") == ABSTAIN


def test_a_streamed_answer_never_leaks_the_sentinel_however_it_is_chunked():
    """The tail of a stream can be a prefix of the sentinel, so the filter has to
    hold back what it cannot yet rule out. Every chunk size is a different split
    of the sentinel across frames, and the client must see the same answer for
    all of them."""
    reply = f"{ANSWER} {NO_ANSWER}"
    for size in range(1, len(reply) + 1):
        streamed = "".join(stop_at_refusal(iter(chunked(reply, size))))
        assert streamed.strip() == ANSWER, f"leaked at chunk size {size}"
        assert NO_ANSWER not in streamed


def test_a_streamed_refusal_arrives_as_the_refusal():
    for size in range(1, len(NO_ANSWER) + 1):
        streamed = "".join(stop_at_refusal(iter(chunked(NO_ANSWER, size))))
        assert streamed == ABSTAIN, f"chunk size {size}"


def test_a_stream_with_no_sentinel_is_forwarded_whole():
    """Nothing may be held back permanently: a filter that swallows the last few
    characters of every answer would pass the leak tests above and be worse than
    the bug."""
    for size in range(1, len(ANSWER) + 1):
        assert "".join(stop_at_refusal(iter(chunked(ANSWER, size)))) == ANSWER


def test_a_stream_that_produces_nothing_still_says_something():
    assert "".join(stop_at_refusal(iter([]))) == ABSTAIN


def test_text_that_merely_starts_like_the_sentinel_is_not_held_forever():
    """`NO_ANSWERS ARE FINAL` shares a prefix with the sentinel and is not it."""
    reply = "The policy states NO_ANSWERS ARE FINAL [c1]."
    assert without_refusal(reply) == reply
    assert "".join(stop_at_refusal(iter(chunked(reply, 3)))) == reply

"""The validator's job is narrow, and these tests pin the boundary.

It answers "did this parse into the shape?" — never "is this correct?". The last
test here is the important one: a reply can be perfectly valid and completely
wrong, and the bench must call that a success. Catching it is Phase 3's job.
"""

from __future__ import annotations

from bench.tasks import CASES, Invoice, prompts, strip_fence, validate_invoice


def test_a_well_formed_reply_validates():
    assert validate_invoice('{"vendor": "Acme", "date": "2026-01-01", "total": 12.5}')


def test_prose_does_not_validate():
    assert not validate_invoice(
        "The vendor is Acme and the total is about twelve fifty."
    )


def test_a_wrong_date_format_does_not_validate():
    assert not validate_invoice(
        '{"vendor": "Acme", "date": "01/01/2026", "total": 12.5}'
    )


def test_a_negative_total_does_not_validate():
    assert not validate_invoice('{"vendor": "Acme", "date": "2026-01-01", "total": -5}')


def test_a_missing_field_does_not_validate():
    assert not validate_invoice('{"vendor": "Acme", "date": "2026-01-01"}')


def test_a_fenced_reply_still_validates():
    fenced = '```json\n{"vendor": "Acme", "date": "2026-01-01", "total": 1.0}\n```'
    assert validate_invoice(fenced)
    assert strip_fence(fenced).startswith("{")


def test_every_case_renders_into_a_prompt():
    rendered = prompts()
    assert len(rendered) == len(CASES)
    assert all(case in prompt for case, prompt in zip(CASES, rendered, strict=True))


def test_schema_valid_is_not_the_same_as_correct():
    """The trap the bench cannot see, and must not pretend to.

    Both of these satisfy `Invoice`. One is an extraction, the other is a model
    giving up neatly. Only an eval with expected values can tell them apart.
    """
    plausible = (
        '{"vendor": "Northwind Traders", "date": "2026-03-04", "total": 1240.50}'
    )
    surrender = '{"vendor": "unknown", "date": "1970-01-01", "total": 0}'
    assert validate_invoice(plausible)
    assert validate_invoice(surrender)
    assert Invoice.model_validate_json(surrender).total == 0

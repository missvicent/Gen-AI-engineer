"""The dataset's own gate: offline, deterministic, runs on every PR.

These tests fail on a broken *fixture*, not a broken system — which is the point.
A suite whose golden set is wrong produces confident nonsense.
"""

from dataclasses import replace

from src.dataset import (
    MIN_UNANSWERABLE,
    SLICES,
    leaked_questions,
    load_corpus,
    load_golden,
    near_duplicates,
    slice_counts,
    validate,
)

GOLDEN = "evals/golden.jsonl"
CORPUS = "evals/corpus.jsonl"


def rows():
    return load_golden(GOLDEN)


def chunks():
    return load_corpus(CORPUS)


def test_the_shipped_golden_set_is_clean():
    assert validate(rows(), chunks()) == []


def test_every_slice_is_populated():
    counts = slice_counts(rows())
    assert set(counts) == set(SLICES)
    assert all(n > 0 for n in counts.values()), counts


def test_abstain_slice_is_big_enough_to_mean_something():
    assert sum(r.expects_abstention for r in rows()) >= MIN_UNANSWERABLE


def test_answerable_rows_name_their_supporting_docs():
    """Without this field there are no judge-free retrieval metrics."""
    assert all(r.supporting_doc_ids for r in rows() if not r.expects_abstention)


def test_supporting_docs_exist_in_the_corpus():
    ids = {line["id"] for line in _corpus_records()}
    for row in rows():
        assert set(row.supporting_doc_ids) <= ids, row.id


def test_near_duplicate_is_detected():
    original = rows()
    planted = original + [
        replace(
            original[0],
            id="g-dupe",
            question="Once we approve a refund, how quickly should a supplier expect their money?",
        )
    ]
    assert ("g-001", "g-dupe") in near_duplicates(planted)
    assert any("near-duplicate" in p for p in validate(planted, chunks()))


def test_question_copied_out_of_a_chunk_is_detected():
    """Leakage: the question is lifted from the text it is supposed to retrieve."""
    planted = rows() + [
        replace(
            rows()[0],
            id="g-leak",
            question="Approved refunds are processed within five business days.",
        )
    ]
    assert "g-leak" in leaked_questions(planted, chunks())


def test_empty_slice_is_reported():
    without_multi_hop = [r for r in rows() if r.slice != "multi_hop"]
    assert any("multi_hop" in p for p in validate(without_multi_hop, chunks()))


def test_abstention_flag_must_agree_with_the_slice():
    bad = [replace(rows()[0], expects_abstention=True)]
    assert any("must agree" in p for p in validate(bad, chunks()))


def test_provenance_is_required():
    bare = [replace(r, source="", labeled_by="", labeled_on="") for r in rows()[:1]]
    assert any("missing provenance" in p for p in validate(bare, chunks()))


def _corpus_records():
    import json
    from pathlib import Path

    return [json.loads(line) for line in Path(CORPUS).read_text().splitlines() if line.strip()]

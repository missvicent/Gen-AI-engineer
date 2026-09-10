"""Tier 1 gate: deterministic, offline, runs on every PR."""

from collections import Counter

from src.harness import build_dataset, evaluate_nonllm, load_golden

GOLDEN = "evals/golden.jsonl"

# A stand-in pipeline. Swap in your real rag_pipeline(question) -> (answer, contexts).
KB = {
    "What is hybrid search?": (
        "hybrid search fuses keyword and vector retrieval",
        ["hybrid search fuses keyword and vector retrieval for better recall"],
    ),
    "How long do refunds take?": (
        "refunds are processed within five business days",
        ["refunds are processed within five business days of approval"],
    ),
    "What does a reranker do?": (
        "a reranker reorders retrieved candidates by relevance",
        ["a reranker reorders retrieved candidates by relevance"],
    ),
    "When was invoice INV-88231 paid?": (
        "invoice INV-88231 was paid on 3 July 2026",
        ["invoice INV-88231 was settled on 3 July 2026 by wire transfer"],
    ),
    "What is error code ERR-4021?": (
        "ERR-4021 means the payment gateway timed out",
        ["ERR-4021 means the payment gateway timed out"],
    ),
    "What is the capital of Mars?": ("I don't know", []),
}


def _pipeline(q: str):
    return KB.get(q, ("I don't know", []))


def test_golden_set_loads_with_slices():
    """A FLOOR on the row count, not an equality.

    Six rows is what this repo ships — three semantic, two exact, one unanswerable.
    Enough to prove the harness loads, scores, slices and gates; nowhere near enough
    to gate a real system on. The assignment is to grow it to thirty over your own
    corpus, so `== 6` would mean doing the assignment turns this suite red, and the
    lesson a red suite teaches is "do not add questions". Ten per slice is roughly
    where a slice mean stops being anecdote; at one row, the unanswerable slice is
    either 0.0 or 1.0 and nothing in between.

    What IS pinned is the shape: all three slices present, the unanswerable one
    included, because that is the slice people quietly drop when their abstention
    path starts failing.
    """
    g = load_golden(GOLDEN)
    assert len(g) >= 6, "the shipped fixture is the floor, not the target"
    counts = Counter(r["slice"] for r in g)
    assert set(counts) == {"semantic", "exact", "unanswerable"}, (
        f"a slice went missing: {counts}"
    )


def test_nonllm_gate_passes_on_a_good_pipeline():
    rows = build_dataset(load_golden(GOLDEN), _pipeline)
    scores = evaluate_nonllm(rows)
    # tier-1 bars are lexical, so they sit lower than the real RAGAS bars
    assert scores["context_recall_nonllm"] >= 0.60
    assert scores["answer_similarity_nonllm"] >= 0.60


def test_gate_catches_a_regression():
    """A pipeline that retrieves nothing must fail the gate — that's the point."""
    rows = build_dataset(load_golden(GOLDEN), lambda q: ("I don't know", []))
    scores = evaluate_nonllm(rows)
    assert scores["context_recall_nonllm"] < 0.60


def test_exact_match_slice_is_scored_separately():
    """Slicing tells you WHERE you're weak — exact-match is the usual culprit."""
    golden = [r for r in load_golden(GOLDEN) if r["slice"] == "exact"]
    rows = build_dataset(golden, _pipeline)
    assert evaluate_nonllm(rows)["context_recall_nonllm"] > 0.5

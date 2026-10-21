"""Sweep a score floor against the golden set, for one arm.

Throwaway measurement for step 3: a good floor blocks every unanswerable row
and no answerable row. Cuts sit just above each unanswerable score, so the
output reads "to block n bad rows, you also block m good ones". Run from
02-rag/before:

    PYTHONPATH=src:. uv run python scripts/sweep.py dense
    PYTHONPATH=src:. uv run python scripts/sweep.py bm25
    PYTHONPATH=src:. uv run --group release python scripts/sweep.py rerank
"""

from __future__ import annotations

import sys

from assistant.rag import RagStore, tokenize
from scripts.recall import CORPUS, GOLDEN, load_jsonl, to_list

ARM = sys.argv[1] if len(sys.argv) > 1 else "dense"

store = RagStore(to_list(load_jsonl(CORPUS)))
rows = load_jsonl(GOLDEN)

if ARM == "rerank":
    # The Reranker callable drops its scores, so ask the model directly.
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")


def top(question: str) -> float:
    if ARM == "rerank":
        return max(model.rerank(question, store.docs))
    arm = store._bm25 if ARM == "bm25" else store._dense
    hits = arm(tokenize(question))
    return hits[0][1] if hits else 0.0


bad = sorted(top(r["question"]) for r in rows if r.get("expects_abstention"))
good = [top(r["question"]) for r in rows if not r.get("expects_abstention")]
inv = top("INV-88231")

print(f"arm: {ARM}")
print(f"unanswerable max {max(bad):.3f} | answerable min {min(good):.3f}")
print(f"INV-88231 top score {inv:.3f}\n")
for cut in (s + 1e-6 for s in bad):
    blocked_bad = sum(s < cut for s in bad)
    blocked_good = sum(s < cut for s in good)
    print(
        f"{cut:8.3f}  bad blocked {blocked_bad}/{len(bad)}  "
        f"good blocked {blocked_good}/{len(good)}  "
        f"INV-88231 {'blocked' if inv < cut else 'ok'}"
    )

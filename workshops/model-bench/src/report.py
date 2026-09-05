"""Two renderings of the same run: one for a human, one for CI.

Given to you whole — column alignment is not the lesson. Note that both functions
call `rank()`, so they stay broken until `core.rank` works.

The JSON is not a nicety. A bench whose output only exists in a terminal cannot
be diffed between two weeks, which means it cannot catch the day a vendor quietly
swaps your model for a cheaper one.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .core import BenchRun, Row, rank

HEADERS = ("candidate", "model", "ok", "tok in", "tok out", "cost $", "p50 ms", "$/ok")


def _cells(row: Row) -> tuple[str, ...]:
    per_ok = "—" if row.ok == 0 else f"{row.cost_per_success:.6f}"
    return (
        row.candidate,
        row.model,
        f"{row.ok}/{row.cases}",
        str(row.tokens_in),
        str(row.tokens_out),
        f"{row.cost_usd:.6f}",
        f"{row.p50_ms:.0f}",
        per_ok,
    )


def table(run: BenchRun) -> str:
    """Ranked, aligned, and readable without a spreadsheet."""
    rows = [_cells(r) for r in rank(run)]
    widths = [
        max(len(header), *(len(r[i]) for r in rows)) if rows else len(header)
        for i, header in enumerate(HEADERS)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(HEADERS, widths, strict=True))
    rule = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(cell.ljust(w) for cell, w in zip(cells, widths, strict=True))
        for cells in rows
    ]
    footer = f"\ntask: {run.task} · total spend: ${run.total_cost:.6f}"
    return "\n".join([line, rule, *body]) + footer


def to_json(run: BenchRun) -> str:
    """The same numbers, diffable. Ranked order is preserved."""
    return json.dumps(
        {
            "task": run.task,
            "total_cost_usd": run.total_cost,
            "rows": [
                {
                    **asdict(r),
                    "success_rate": r.success_rate,
                    "cost_per_success": None if r.ok == 0 else r.cost_per_success,
                }
                for r in rank(run)
            ],
        },
        indent=2,
    )

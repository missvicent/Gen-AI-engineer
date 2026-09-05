"""`bench run` — the thing you actually demo.

python -m bench.cli --providers local,gpt-mini
python -m bench.cli --providers local --json > bench-$(date +%F).json
"""

from __future__ import annotations

import argparse
import sys

from .core import run_bench
from .providers import live_runner, resolve
from .report import table, to_json
from .tasks import CASES, prompts, validate_invoice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench", description="One task, every provider."
    )
    parser.add_argument(
        "--providers", default="local", help="comma-separated: local,gpt,gpt-mini"
    )
    parser.add_argument(
        "--cases", type=int, default=len(CASES), help="how many cases to run"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    args = parser.parse_args(argv)

    try:
        candidates = resolve(
            [n.strip() for n in args.providers.split(",") if n.strip()]
        )
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    run = run_bench(
        candidates=candidates,
        cases=prompts()[: args.cases],
        runner=live_runner,
        validate=validate_invoice,
        task="invoice-extraction",
    )
    print(to_json(run) if args.json else table(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

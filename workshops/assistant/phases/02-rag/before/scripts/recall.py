from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
EVALS = ROOT / "evals" / "golden-set" / "evals"
CORPUS = EVALS / "corpus.jsonl"
GOLDEN = EVALS / "golden.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    corpus = load_jsonl(CORPUS)
    golden = load_jsonl(GOLDEN)

    print(len(corpus))
    print(len(golden))


if __name__ == "__main__":
    main()

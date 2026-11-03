"""Label evals/my_rows.jsonl yourself, one row at a time.

    make label

Pass = every claim in the answer is supported by the chunks. The judge's score
stays hidden so it can't pull your label toward it. Saves after every row, and
running it again continues where you stopped.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

from judge_calibration.build_rows import MY_ROWS

ANSWERS = {"p": "pass", "f": "fail"}


def label_rows(path: Path, ask: Callable[[str], str] = input) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    todo = [row for row in rows if not row["human"]]

    for position, row in enumerate(todo, start=1):
        print(f"\n--- {row['id']} ({position} of {len(todo)} left to label)")
        print(f"Question: {row['question']}\n\nAnswer: {row['answer']}\n")
        for chunk in row["contexts"]:
            print(f"Chunk: {chunk}\n")

        choice = ""
        while choice not in (*ANSWERS, "q"):
            choice = ask("[p]ass, [f]ail or [q]uit: ").strip().lower()
        if choice == "q":
            break

        row.update(
            human=ANSWERS[choice],
            labeled_by="human",
            labeled_on=date.today().isoformat(),
        )
        # Replace in one step, so a crash mid-write can't damage your labels.
        temp = path.with_suffix(".tmp")
        text = "".join(json.dumps(r) + "\n" for r in rows)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)

    left = sum(not row["human"] for row in rows)
    print(f"\n{len(rows) - left} labeled, {left} left")


if __name__ == "__main__":
    label_rows(MY_ROWS)

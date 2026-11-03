"""Offline: fake model and judge, a temp file, and scripted answers."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from judge_calibration.build_rows import build_rows
from judge_calibration.calibration import load_labeled
from judge_calibration.label_rows import label_rows


def golden_row(id, expects_abstention=False):
    return SimpleNamespace(
        id=id,
        slice="semantic",
        question=f"question {id}",
        expects_abstention=expects_abstention,
    )


GOLDEN = [
    golden_row("a"),
    golden_row("abstain", expects_abstention=True),
    golden_row("no-chunks"),
    golden_row("nan"),
    golden_row("b"),
]


def build(path, calls=None, limit=None):
    def write_answer(question, contexts):
        if calls is not None:
            calls.append(question)
        return f"answer to {question}"

    build_rows(
        GOLDEN,
        lambda q: [] if "no-chunks" in q else ["a chunk"],
        write_answer,
        lambda q, a, c: math.nan if "nan" in q else 0.7,
        path,
        limit,
    )


def ids(path):
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def test_build_skips_rows_that_cant_be_judged(tmp_path):
    path = tmp_path / "rows.jsonl"
    build(path)
    assert ids(path) == ["a", "b"]


def test_rerun_only_adds_missing_rows_and_keeps_labels(tmp_path):
    path = tmp_path / "rows.jsonl"
    build(path, limit=1)
    assert ids(path) == ["a"]
    label_rows(path, ask=lambda _: "p")

    calls = []
    build(path, calls)
    assert ids(path) == ["a", "b"]
    assert "question a" not in calls
    assert json.loads(path.read_text().splitlines()[0])["human"] == "pass"


def test_labeling_hides_the_score_can_quit_and_resume(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    build(path)

    answers = iter(["x", "f", "q"])
    label_rows(path, ask=lambda _: next(answers))
    assert "0.7" not in capsys.readouterr().out
    assert [json.loads(line)["human"] for line in path.read_text().splitlines()] == [
        "fail",
        "",
    ]

    label_rows(path, ask=lambda _: "p")
    assert [r.human for r in load_labeled(path)] == ["fail", "pass"]

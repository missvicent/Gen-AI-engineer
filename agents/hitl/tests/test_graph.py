"""Your scratch agent as a graph (hitl_agent/graph.py) — written before graph.py.

    make test-integration

`build_graph(decide, tools, checkpointer)` is yours. These tests pin the card:
`delete_note` pauses at interrupt() before it runs, a human's yes/no resumes it
by thread_id, the pause survives a restart, and resuming replays from the
checkpoint, so the model is not asked again for the step that paused.

Offline: `decide` is a script, not a model. The notes are 4.2's in-memory notes,
reset before every test.
"""

from __future__ import annotations

import sqlite3

import agent_tools.tools as notes
import pytest
from react_from_scratch.agent import Decision

pytestmark = pytest.mark.integration

TOOLS = {
    fn.__name__: fn for fn in (notes.read_note, notes.draft_reply, notes.delete_note)
}
DELETE_2 = Decision(is_final=False, tool="delete_note", args={"note_id": "2"})
DONE = Decision(is_final=True, answer="done")


class Script:
    """A fake model: answers step i with steps[i], where i = len(history).
    Counts its calls, so a test can see whether resume asked it again."""

    def __init__(self, *steps: Decision):
        self.steps = steps
        self.calls = 0

    def __call__(self, goal: str, history: list[dict]) -> Decision:
        self.calls += 1
        return self.steps[len(history)]


@pytest.fixture(autouse=True)
def fresh_notes(monkeypatch):
    monkeypatch.setattr(notes, "_NOTES", {"1": "buy milk", "2": "call dentist"})
    monkeypatch.setattr(notes, "_APPROVALS", set())


def thread(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def test_read_only_tool_runs_without_a_pause():
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    read_1 = Decision(is_final=False, tool="read_note", args={"note_id": "1"})
    app = build_graph(Script(read_1, DONE), TOOLS, MemorySaver())

    out = app.invoke({"goal": "read note 1"}, thread("read"))
    assert "__interrupt__" not in out
    assert out["answer"] == "done"
    assert out["history"][0]["result"] == {"text": "buy milk"}


def test_delete_pauses_before_it_runs():
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    app = build_graph(Script(DELETE_2, DONE), TOOLS, MemorySaver())

    out = app.invoke({"goal": "delete note 2"}, thread("pause"))
    assert "__interrupt__" in out  # suspended, not finished
    asked = out["__interrupt__"][0].value
    assert asked["tool"] == "delete_note"
    assert asked["args"] == {"note_id": "2"}
    assert "2" in notes._NOTES  # nothing deleted while paused
    assert notes._APPROVALS == set()  # and no approval granted yet


def test_resume_yes_deletes_and_finishes():
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    app = build_graph(Script(DELETE_2, DONE), TOOLS, MemorySaver())
    app.invoke({"goal": "delete note 2"}, thread("yes"))

    out = app.invoke(Command(resume=True), thread("yes"))
    assert "2" not in notes._NOTES
    assert out["history"][-1]["result"] == {"deleted": "2"}
    assert out["answer"] == "done"


def test_resume_no_never_calls_delete():
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    called: list[dict] = []
    tools = {**TOOLS, "delete_note": lambda **args: called.append(args)}
    app = build_graph(Script(DELETE_2, DONE), tools, MemorySaver())
    app.invoke({"goal": "delete note 2"}, thread("no"))

    out = app.invoke(Command(resume=False), thread("no"))
    assert called == []  # the tool never ran
    assert "2" in notes._NOTES
    assert notes._APPROVALS == set()  # a "no" grants nothing
    assert "rejected" in str(out["history"][-1]["result"])  # the model is told
    assert out["answer"] == "done"


def test_the_pause_survives_a_restart(tmp_path):
    """Pause in one graph, finish in another that knows only the SQLite file."""
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    db = tmp_path / "hitl.sqlite"

    conn1 = sqlite3.connect(db, check_same_thread=False)
    app1 = build_graph(Script(DELETE_2, DONE), TOOLS, SqliteSaver(conn1))
    app1.invoke({"goal": "delete note 2"}, thread("durable"))
    conn1.close()  # the "kill"

    conn2 = sqlite3.connect(db, check_same_thread=False)
    app2 = build_graph(Script(DELETE_2, DONE), TOOLS, SqliteSaver(conn2))
    out = app2.invoke(Command(resume=True), thread("durable"))
    conn2.close()
    assert "2" not in notes._NOTES
    assert out["answer"] == "done"


def test_resume_does_not_ask_the_model_again():
    """Resume replays only the paused node. If the whole agent were one node,
    the model call that chose delete_note would run a second time."""
    from hitl_agent.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    decide = Script(DELETE_2, DONE)
    app = build_graph(decide, TOOLS, MemorySaver())

    app.invoke({"goal": "delete note 2"}, thread("replay"))
    assert decide.calls == 1  # chose delete_note, then paused

    app.invoke(Command(resume=True), thread("replay"))
    assert decide.calls == 2  # only the next step, not the paused one again

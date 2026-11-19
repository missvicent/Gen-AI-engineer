"""Tier 2: the same durable pause, in REAL LangGraph — interrupt() + a
checkpointer, resumed by thread_id, surviving a simulated process restart.

    make test-integration

The graph under test is `hitl.approval_graph`, in hitl_agent/, which is yours. This file
only supplies a checkpointer and a place to record sends; every assertion here is
about code you wrote. It used to build the graph itself, which made the real
implementation something you read rather than something you had to get right.

Runs offline: the graph's nodes are deterministic functions, no model involved.
"""

from __future__ import annotations

import sqlite3

import pytest
from hitl_agent.hitl import approval_graph

pytestmark = pytest.mark.integration


def test_interrupt_pauses_before_the_irreversible_step_and_resumes_by_thread_id():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    sent: list[str] = []
    app = approval_graph(MemorySaver(), sent.append)
    config = {"configurable": {"thread_id": "hitl-approve"}}

    result = app.invoke({"topic": "launch"}, config)
    assert "__interrupt__" in result  # the run is suspended, not finished
    assert sent == []  # containment: nothing fired while paused

    out = app.invoke(Command(resume=True), config)  # the human clicks approve
    assert out["approved"] is True
    assert sent == ["re: launch"]


def test_rejection_resumes_the_graph_without_firing_the_tool():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    sent: list[str] = []
    app = approval_graph(MemorySaver(), sent.append)
    config = {"configurable": {"thread_id": "hitl-reject"}}

    app.invoke({"topic": "launch"}, config)
    out = app.invoke(Command(resume=False), config)
    assert out["approved"] is False
    assert sent == []


def test_two_threads_pause_independently():
    """`thread_id` is the whole addressing scheme for a resume. Two runs paused
    at once must not resolve each other — approving one leaves the other exactly
    as suspended as it was."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    sent: list[str] = []
    app = approval_graph(MemorySaver(), sent.append)
    first = {"configurable": {"thread_id": "invoice-1"}}
    second = {"configurable": {"thread_id": "invoice-2"}}

    app.invoke({"topic": "invoice one"}, first)
    app.invoke({"topic": "invoice two"}, second)
    app.invoke(Command(resume=True), first)

    assert sent == ["re: invoice one"], "resuming one thread fired the other"
    assert app.invoke(Command(resume=False), second)["approved"] is False


def test_the_pause_survives_a_process_restart(tmp_path):
    """The whole reason HITL needs a real checkpointer: pause in one process,
    resume in another. Simulated here with two graph instances over one SQLite
    file — the second knows nothing the database doesn't."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    db_path = tmp_path / "hitl.db"
    config = {"configurable": {"thread_id": "hitl-durable"}}

    sent_before: list[str] = []
    conn1 = sqlite3.connect(db_path, check_same_thread=False)
    app1 = approval_graph(SqliteSaver(conn1), sent_before.append)
    app1.invoke({"topic": "quarterly report"}, config)
    assert sent_before == []
    conn1.close()  # the "crash"

    sent_after: list[str] = []
    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    app2 = approval_graph(SqliteSaver(conn2), sent_after.append)
    out = app2.invoke(Command(resume=True), config)
    assert out["approved"] is True
    assert sent_after == ["re: quarterly report"]
    conn2.close()

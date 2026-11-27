"""The agent on LangGraph (bakeoff/run_langgraph.py) — written before the runner.

Offline: `lc_model` (tests/conftest.py) is a scripted chat model that calls
read_note(note_id="2") once, then answers with what the tool returned.
"""

from __future__ import annotations

import sqlite3

import agent_tools.tools as notes


def test_answers_with_one_tool_call(lc_model):
    from bakeoff.contract import TASK, ToolCall, used_the_tool
    from bakeoff.run_langgraph import run_langgraph

    r = run_langgraph(TASK, notes.read_note, model=lc_model)

    assert r.framework == "langgraph"
    assert "call dentist" in r.answer
    assert r.tool_calls == [ToolCall(name="read_note", args={"note_id": "2"})]
    assert used_the_tool(r)


def test_the_run_survives_in_the_checkpointer(tmp_path, lc_model):
    """Durability: run over a SQLite file, close it, then read the run back with
    get_state on a SECOND app that knows only the file, not the one that ran."""
    from bakeoff.contract import TASK
    from bakeoff.run_langgraph import run_langgraph
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import START, MessagesState, StateGraph

    db = tmp_path / "bakeoff.sqlite"

    conn1 = sqlite3.connect(db, check_same_thread=False)
    run_langgraph(
        TASK, notes.read_note, model=lc_model, checkpointer=SqliteSaver(conn1)
    )
    conn1.close()  # the first app is gone

    saver2 = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
    # The runner picks its own thread id; the file tells us which one it used.
    threads = {c.config["configurable"]["thread_id"] for c in saver2.list(None)}
    assert len(threads) == 1, threads
    (thread_id,) = threads

    # A second app: any graph with a `messages` channel can read the run back.
    reader = StateGraph(MessagesState)
    reader.add_node("noop", lambda state: {})
    reader.add_edge(START, "noop")
    app2 = reader.compile(checkpointer=saver2)

    messages = app2.get_state({"configurable": {"thread_id": thread_id}}).values[
        "messages"
    ]
    assert any(
        isinstance(m, ToolMessage) and "call dentist" in str(m.content)
        for m in messages
    )
    assert "call dentist" in messages[-1].content

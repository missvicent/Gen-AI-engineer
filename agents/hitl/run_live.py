"""Kill-and-resume demo for your agent graph (hitl_agent/graph.py).

    make live [GOAL="..."]     new thread: runs to the pause, waits at approve? [y/n]
    make resume THREAD=<id>    new process, same SQLite file: asks again, finishes

Kill `make live` (Ctrl-C) at the prompt, then resume it by its thread id. The
model is 4.2's `decide` on the local model: ask before running, `ollama stop`
after. Notes live in memory, so a new process starts with notes "1" and "2".
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from hitl_agent.graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from react_from_scratch.agent import Decision

HERE = Path(__file__).parent
DB = HERE / ".hitl.sqlite"  # gitignored

# 4.2's run_live.py has the same module name as this file, so load it by path.
_spec = importlib.util.spec_from_file_location(
    "tools_run_live", HERE.parent / "tools" / "run_live.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("can't load agents/tools/run_live.py")
tools_live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tools_live)


def model_decide(goal: str, history: list[dict]) -> Decision:
    """4.2's decide reads history as (Decision, result) pairs; the graph saves
    plain dicts. Convert, call the model, and print the step."""
    pairs = [
        (Decision(is_final=False, tool=h["tool"], args=h["args"]), h["result"])
        for h in history
    ]
    if history:
        print(f"   -> {str(history[-1]['result'])[:200]}")
    d = tools_live.decide(goal, pairs)
    print(f"{len(history) + 1}. {d.tool}({d.args})" if not d.is_final else "final")
    return d


def run(app: Any, thread_id: str, first: Any) -> None:
    """Invoke, then answer every pause from the keyboard until the run ends."""
    config = {"configurable": {"thread_id": thread_id}}
    out = app.invoke(first, config)
    while "__interrupt__" in out:
        asked = out["__interrupt__"][0].value
        print(f"\nthread: {thread_id}")
        print(f"paused before {asked['tool']}({asked['args']})")
        print(f"kill now (Ctrl-C), then: make resume THREAD={thread_id}")
        yes = input("approve? [y/n] ").strip().lower() == "y"
        out = app.invoke(Command(resume=yes), config)
    print(f"\nanswer: {out.get('answer')}")


def main(decide: Any = model_decide) -> None:
    conn = sqlite3.connect(DB, check_same_thread=False)
    app = build_graph(decide, tools_live.TOOLS, SqliteSaver(conn))
    thread_id = os.environ.get("THREAD", "")
    try:
        if not thread_id:
            goal = os.environ.get("GOAL", "Delete note 2, I already called the dentist")
            thread_id = uuid.uuid4().hex[:8]
            print(f"thread: {thread_id}  goal: {goal}")
            run(app, thread_id, {"goal": goal})
            return
        # Resume: show what the saved checkpoint is waiting on, then ask again.
        state = app.get_state({"configurable": {"thread_id": thread_id}})
        if not state.interrupts:
            print(f"thread {thread_id}: nothing paused (finished or unknown)")
            return
        print(f"resuming thread {thread_id}  goal: {state.values.get('goal')}")
        asked = state.interrupts[0].value
        print(f"paused before {asked['tool']}({asked['args']})")
        yes = input("approve? [y/n] ").strip().lower() == "y"
        run(app, thread_id, Command(resume=yes))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

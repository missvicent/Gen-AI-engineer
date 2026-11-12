"""A ReAct agent loop written from scratch.

run_agent(goal, tools, decide, max_steps, deadline_s), each round:
  - check the wall-clock deadline first, so no model call starts after it
  - call decide(goal, history) -> a Decision (final answer, or a tool + args)
  - if final: return the answer
  - else: look the tool up by name and run it; any exception (unknown tool,
    bad input) becomes an "error: <Type>: <message>" result, not a crash
  - append (decision, result) to history and go round again
Stops with "max_steps_exceeded: ..." or "deadline_exceeded: ..." — never hangs.
The leash (steps, time, which tools exist) lives in this code, not the model.

`decide` is injected: scripted in tests, an LLM call in prod.
"""

from __future__ import annotations

import ast
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Decision:
    is_final: bool
    answer: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)


Tool = Callable[..., Any]
Decider = Callable[[str, list[tuple[Decision, Any]]], Decision]


def run_agent(
    goal: str,
    tools: dict[str, Tool],
    decide: Decider,
    max_steps: int = 8,
    deadline_s: float = 60.0,
) -> str:
    count = 0
    current_time = time.monotonic()
    history = []
    while count < max_steps:
        now = time.monotonic() - current_time
        if now > deadline_s:
            return "deadline_exceeded: we couldn't process the request in time"
        question = decide(goal, history)
        count += 1
        if question.is_final:
            return question.answer
        try:
            tool = tools[question.tool]
            result = tool(**question.args)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        history.append((question, result))
    return "max_steps_exceeded: we couldn't process your request"


_OPS: dict[type[ast.AST], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression (+ - * / % ** and parentheses).

    Model-supplied text is untrusted input, so this walks an AST allowlist
    instead of calling eval() — emptying __builtins__ does not sandbox eval.
    """

    def walk(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")

    return float(walk(ast.parse(expression, mode="eval").body))

"""TODO: write the agent loop from scratch.

run_agent(goal, tools, decide, max_steps, deadline_s):
  - loop up to max_steps AND a wall-clock deadline (the leash lives in CODE)
  - call decide(goal, state) -> a Decision (final answer, or a tool + args)
  - if final: return the answer
  - else: run the requested tool (unknown tool / exception -> observation, not crash)
  - append (decision, result) to state and continue
  - if the cap is hit: return a graceful failure, never hang

`decide` is injected (in tests it's scripted; in prod it calls your LLM client).
Reference: ../after/src/agent.py.
"""

from __future__ import annotations

import ast
import operator
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
    raise NotImplementedError  # TODO: build the loop with hard caps


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
